from abc import ABC, abstractmethod
import typing as tp
from collections import Counter
from functools import partial
from particles import resampling as rs
import numpy as np
import particles
from scipy.stats import multivariate_normal

from libs_new.utils import group_by_chunk, AdvancedOrderedDict, better_random_choice, max_idx, flattened_hilbert_sort, \
    auto_load_balance, zip_with_assert, composition_with_star, tqdm_enable, list_chain_star
from tqdm import tqdm

from libs_new.utils_math import CategoricalDistribution, coupling_rate_estimate

_Tp = tp.TypeVar('_Tp')  # particle type

class IntractableFK(ABC):
    """
    The idea of the smoother for intractable model is that, if two particles produce the same child during the forward filtering path via a coupling mechanism, every backward path that passes through that child can be continued via either of the two parents. This contrasts with the traditional one-parent-only model and thus prevents path degeneracy.
    """
    @abstractmethod
    def M0(self, N: int) -> tp.Sequence[_Tp]:...

    @abstractmethod
    def coupled_M(self, t, xp1: _Tp, xp2: _Tp) -> tp.Tuple[_Tp, _Tp, bool]:
        """
        returns true if coupling is successful
        """
        ...

    @abstractmethod
    def logG(self, t, x: tp.Sequence[_Tp]) -> tp.Sequence[float]:...

    @property
    @abstractmethod
    def T(self) -> int:
        """
        Convention: data are x0, x1, ..., xT
        """
        ...

    @classmethod
    def fromFKObject(cls, fk: particles.FeynmanKac, coupled_M: tp.Callable[['FromFKObject', int, _Tp, _Tp], tp.Tuple[_Tp, _Tp, bool]]):
        """
         Create a new intractable FeynmanKac object from a normal FeynmanKac object.
        :param fk: ordinary FeynmanKac object in the `particles` package. Must be an immutable object (important!)
        :param coupled_M: the coupled_M method of the new object
        Note: it is possible to mix multiple couplers by instantiating the utils_math.MixtureKernel class with different coupled_M functions
        """
        return FromFKObject(fk=fk, coupled_M=coupled_M)

class FromFKObject(IntractableFK):
    def __init__(self, fk: particles.FeynmanKac, coupled_M: tp.Callable[['FromFKObject', int, _Tp, _Tp], tp.Tuple[_Tp, _Tp, bool]]):
        self.normal_fk = fk
        self.mixin_coupled_M = coupled_M

    def M0(self, N: int) -> tp.Sequence[_Tp]:
        return self.normal_fk.M0(N)

    def logG(self, t, x: tp.Sequence[_Tp]) -> tp.Sequence[float]:
        return self.normal_fk.logG(t, x, x)

    @property
    def T(self):
        return self.normal_fk.T - 1

    def coupled_M(self, t, xp1: _Tp, xp2: _Tp) -> tp.Tuple[_Tp, _Tp, bool]:
        return self.mixin_coupled_M(self, t, xp1, xp2)

def no_coupled_M(ifk: FromFKObject, t: int, xp1: _Tp, xp2: _Tp) -> tp.Tuple[_Tp, _Tp, bool]:
    xp = np.array([xp1, xp2])
    x = ifk.normal_fk.M(t, xp)
    assert len(x) == 2
    return x[0], x[1], False

class _MoveResult(tp.NamedTuple):
    X: tp.Sequence[_Tp]
    backward_idx: tp.Sequence[tp.Sequence[int]]
    success_rate: float

class Reorderer(ABC):
    """
    Reorder particles after the resampling step. See usage in module IntractableSMC
    """
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def _permute(self, X: tp.Sequence[_Tp], idx: tp.Sequence[int]) -> tp.Sequence[int]:...

    def permute(self, X: tp.Sequence[_Tp], idx: tp.Sequence[int]) -> tp.Sequence[int]:
        res = self._permute(X=X, idx=idx)
        if self.debug:
            assert Counter(idx) == Counter(res)
        return res

class Permute(Reorderer):
    def _permute(self, X: tp.Sequence[_Tp], idx: tp.Sequence[int]) -> tp.Sequence[int]:
        # noinspection PyTypeChecker
        return idx[np.random.permutation(len(idx))]

_CountedType = tp.TypeVar('_CountedType')

def counter_walker(counter: tp.Union[tp.Dict[_CountedType, int], tp.Counter[_CountedType]]) -> tp.Sequence[_CountedType]:
    # tested 071121
    """
    Reorder the multiset defined by ``counter`` in a way that adjacent elements tend to be distinct, but close to each other (in the order defined by ``counter``).
    """
    ordered_dict = AdvancedOrderedDict(counter)
    current_key = ordered_dict.key_linkedlist.head.content
    res = []
    while True:
        res.append(current_key)
        ordered_dict[current_key] = ordered_dict[current_key] - 1
        current_key = _new_key(ordered_dict, current_key)
        if current_key is None:
            break
    return res

def _new_key(ordered_dict: AdvancedOrderedDict, current_key):
    """
    :param current_key: just exploited key
    :return: next exploited key
    """
    left_key = ordered_dict.previous_key(current_key)
    right_key = ordered_dict.next_key(current_key)
    possibilities = [left_key, current_key, right_key]
    weights = [(ordered_dict[k] if k is not None else -np.inf) for k in possibilities]
    weights[1] = 0
    assert (weights[0] != 0) and (weights[2] != 0)
    pos_idx = max_idx(weights)
    choice = better_random_choice([possibilities[i] for i in pos_idx])

    if ordered_dict[current_key] == 0:
        del ordered_dict[current_key]
        if choice == current_key:
            return None

    return choice

class Adjacent(Reorderer):
    # tested 071121
    def _permute(self, X: tp.Sequence[_Tp], idx: tp.Sequence[int]) -> tp.Sequence[int]:
        argsort = flattened_hilbert_sort(X[idx])
        sorted_idx = np.array(idx)[argsort]
        return counter_walker(Counter(sorted_idx))

class WrongDontUse(Reorderer):
    """
    A dummy and invalid reorderer. Use for testing only.
    """
    def _permute(self, X: tp.Sequence[_Tp], idx: tp.Sequence[int]) -> tp.Sequence[int]:
        # noinspection PyTypeChecker
        return idx[np.random.randint(low=0, high=len(idx), size=len(idx))]

class IntractableSMC:
    # tested 090921
    def __init__(self, fk: IntractableFK, N: int, reorderer: Reorderer, verbose: bool = False, ncores=None, start_method=None):
        """
        :param ncores: the number of cores used to run the SMC instance. If None, no parallelisation will be done.
        :param start_method: how parallel processes will be started. Only has effect if ncores is not None.
        """
        self.fk = fk
        self.N = N
        assert self.N % 2 == 0
        self.halfN = self.N // 2
        self.verbose = verbose
        self.reorderer = reorderer
        self.ncores = ncores
        self.start_method = start_method

        self.t = 0
        self.X: tp.Optional[tp.Sequence[_Tp]] = None
        self.wgts : tp.Optional[rs.Weights] = None
        self.backward_idx: tp.Optional[tp.Sequence[tp.Sequence[int]]] = None
        self.coupling_success_rate: float = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.t > self.fk.T:
            raise StopIteration
        if self.t == 0:
            # Degenerated resample and move
            self.X = self.fk.M0(self.N)
        else:
            # == Resample ==
            xp_idx = rs.systematic(self.wgts.W)
            xp_idx = self.reorderer.permute(X=self.X, idx=xp_idx)  # Randomisation needed, as systematic resampling puts similar particles next to each other

            # == Move ==
            self.X, self.backward_idx, self.coupling_success_rate = self.move(old_X=self.X, xp_idx=xp_idx, coupled_M=partial(self.fk.coupled_M, self.t), verbose=self.verbose, ncores=self.ncores, start_method=self.start_method)
            self.X, self.backward_idx = simultaneously_permute(self.X, self.backward_idx)
            self.X = np.array(self.X)
        # Reweight
        self.wgts = rs.Weights(lw=self.fk.logG(t=self.t, x=self.X))
        self.t += 1
        if self.verbose:
            self.print_info()
        return self

    def print_info(self):
        print('t={}, ESSr={}, couple_rate={}'.format(self.t-1, self.wgts.ESS/self.N, self.coupling_success_rate))

    @staticmethod
    def _move_old_version(old_X: tp.Sequence[_Tp], xp_idx: tp.Sequence[int], coupled_M: tp.Callable[[_Tp, _Tp], tp.Tuple[_Tp, _Tp, bool]], verbose=False) -> _MoveResult:
        """
        Deprecated. Use `move` instead.
        """
        newX = []
        backward_idx = []
        success_tracker = []

        gbc = group_by_chunk(xp_idx, 2)
        if verbose:
            gbc = tqdm(gbc)
        for i1, i2 in gbc:
            x1, x2, success = coupled_M(old_X[i1], old_X[i2])
            newX.extend([x1, x2])
            success_tracker.append(success)
            if success:
                backward_idx.extend([[i1, i2], [i1, i2]])
            else:
                backward_idx.extend([[i1], [i2]])

        return _MoveResult(X=newX, backward_idx=backward_idx, success_rate=sum(success_tracker)/len(success_tracker))

    @classmethod
    def move(cls, old_X: tp.Sequence[_Tp], xp_idx: tp.Sequence[int], coupled_M: tp.Callable[[_Tp, _Tp], tp.Tuple[_Tp, _Tp, bool]], verbose=False, ncores=None, start_method=None) -> _MoveResult:
        # tested
        idx_pairs = list(group_by_chunk(xp_idx, 2))
        x_pairs = [(old_X[i], old_X[j]) for i, j in idx_pairs]
        coupled_M = composition_with_star(coupled_M)
        if ncores is None:
            cMres = [coupled_M(xp) for xp in tqdm_enable(x_pairs, enable=verbose)]
        else:
            cMres = auto_load_balance(coupled_M, x_pairs, ncores, start_method, verbose)
        xres = [(e[0], e[1]) for e in cMres]
        success_stats = [e[2] for e in cMres]
        backward_idx = cls._backward_idx(idx_pairs=idx_pairs, success_stats=success_stats)
        return _MoveResult(X=list_chain_star(xres), backward_idx=backward_idx, success_rate=float(np.mean(success_stats)))

    @staticmethod
    def _backward_idx(idx_pairs: tp.Iterable[tp.Tuple[int, int]], success_stats: tp.Iterable[bool]) -> tp.List[tp.List[int]]:
        res = []
        for ip, s in zip_with_assert(idx_pairs, success_stats):
            assert len(ip) == 2
            if s:
                res.append(ip)
                res.append(ip)
            else:
                res.append([ip[0]])
                res.append([ip[1]])
        return res

def simultaneously_permute(*args):
    # tested, also tested usage in IntractableSMC (151121)
    assert len(set([len(a) for a in args])) == 1
    N = len(args[0])
    new_order = np.random.permutation(N)
    res = []
    for a in args:
        res.append([a[i] for i in new_order])
    return res

def estimate_coupling_probability(Xtm1: tp.Sequence[_Tp], Wtm1: CategoricalDistribution, Mt: tp.Callable[[_Tp], tp.Any], ncouples: int, ntrials: int, nMC: int, summary: tp.Literal['mean', 'median'] = 'median', Mt_vectorised=True) -> float:
    # tested
    """
    Estimate the coupling probability of the intractable smoother by approximating proposal distributions with Gaussian ones
    :param Xtm1: particles at time `t-1`
    :param Wtm1: Weights at time `t-1`
    :param Mt: Markov kernel
    :param ncouples: number of couples to take from Xtm1
    :param ntrials: number of times M_t is invoked for each particle in each couple
    :param nMC: number of Monte Carlo simulation to estimate total variation distance between two multivariate Gaussians
    :return: the estimated coupling probability
    """
    probas = []
    for _ in tqdm(range(ncouples)):
        i1, i2 = Wtm1.rvs(size=2)
        probas.append(_estimate_coupling_probability(x1=Xtm1[i1], x2=Xtm1[i2], Mt=Mt, ntrials=ntrials, nMC=nMC, Mt_vectorised=Mt_vectorised))
    return float(getattr(np, summary)(probas))

def _estimate_coupling_probability(x1, x2, Mt, ntrials, nMC, Mt_vectorised):
    if not Mt_vectorised:
        y1 = [Mt(x1) for _ in (range(ntrials))]
        y2 = [Mt(x2) for _ in (range(ntrials))]
    else:
        y1 = Mt(np.array([x1] * ntrials))
        y2 = Mt(np.array([x2] * ntrials))
    return _tv_via_gaussian_approx(y1=y1, y2=y2, nMC=nMC)

def _tv_via_gaussian_approx(y1, y2, nMC):
    dist1 = _gaussian_approx(y1)
    dist2 = _gaussian_approx(y2)
    return coupling_rate_estimate(d1=dist1, d2=dist2, N=nMC, cint=False)

def _gaussian_approx(sample):
    # tested
    mean = np.mean(sample, axis=0)
    cov = np.cov(sample, rowvar=False)
    return multivariate_normal(mean=mean, cov=cov)