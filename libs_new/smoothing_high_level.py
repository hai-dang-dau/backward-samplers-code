import typing as tp
from functools import partial
from libs_new.intractable_smoother import IntractableFK
from libs_new.utils import faster_np_r_tuple, composition, pack, zip_with_assert, dearray, convert_to_numpy_dim
from libs_new.utils_math import CategoricalDistribution
from operator import itemgetter
from particles import FeynmanKac
from abc import ABC, abstractmethod
from libs_new.mcmc_on_mesh import BackwardDistGaussianizer
import numpy as np
from libs_new.smoothing_generic_backwards import BackwardSampler, simple_mean as _simple_mean, BSResult, simple_scalar_prod as _simple_scalar_prod
from libs_new.stack_manager import iterate_over_tree, TreeIterator
from collections.abc import Generator
from particles.smoothing import ParticleHistory

_Ttm1 = tp.TypeVar('_Ttm1')  # Type of particle at time t - 1
_Tt = tp.TypeVar('_Tt')  # Type of particle at time t
_CostInfo = tp.TypeVar('_CostInfo')
_Tp = tp.TypeVar('_Tp')  # Generic type of particle

class AdditiveFunction:
    """
    Warning: additive function here can actually take vector values.
    """
    # tested
    def __init__(self, psi_0: tp.Callable[[_Tp], float], psi_t: tp.Callable[[int, _Ttm1, _Tt], float], output_numpy_dim: int = None):
        """
        :param output_numpy_dim: the dimension of the output (in the numpy terminology, an integer from 0 to infinity). only necessary if the cut_off function is to be used.
        """
        self._psi_0 = psi_0
        self._psi_t = psi_t
        self.output_numpy_dim = output_numpy_dim

    def psi(self, t: int) -> tp.Callable[..., float]:
        if t == 0:
            return self._psi_0
        else:
            return partial(self._psi_t, t)

    def phi(self, t:int) -> tp.Callable[[tp.Sequence[_Tp]], float]:
        return partial(self._phi, t)

    def _phi(self, t:int, x: tp.Sequence[_Tp]) -> float:
        if t < 0:
            raise ValueError
        if len(x) != t + 1:
            raise ValueError
        first_term = self._psi_0(x[0])
        other_terms = [self._psi_t(s, x[s-1], x[s]) for s in range(1, t + 1)]
        return first_term + sum(other_terms)

    @classmethod
    def combine(cls, lf: tp.List['AdditiveFunction']) -> 'AdditiveFunction':
        """
        Combine multiple scalar or vector-valued additive functions into a vector-valued additive function.
        """
        combined_psi_0 = partial(cls._combined_psi_0, list_psi_0=[f._psi_0 for f in lf])
        combined_psi_t = partial(cls._combined_psi_t, list_psi_t=[f._psi_t for f in lf])
        # noinspection PyTypeChecker
        return cls(combined_psi_0, combined_psi_t)

    @staticmethod
    def _combined_psi_0(x0, list_psi_0: tp.Sequence[tp.Callable[[_Tp], float]]):
        # return np.r_[tuple([psi_0(x0) for psi_0 in list_psi_0])]
        return faster_np_r_tuple([psi_0(x0) for psi_0 in list_psi_0])

    @staticmethod
    def _combined_psi_t(t, xtm1, xt, list_psi_t):
        # return np.r_[tuple([psi_t(t, xtm1, xt) for psi_t in list_psi_t])]
        return faster_np_r_tuple([psi_t(t, xtm1, xt) for psi_t in list_psi_t])

    def iterate(self, x):
        return AdtFctIterator(adf=self, x=x)

def cut_off(af: AdditiveFunction, t_cutoff: int) -> AdditiveFunction:
    # fully tested
    """
    Create a new additive function whose return value is a concatenation of that of the original one and the sum of its terms up to t_cutoff
    """
    if t_cutoff < 0:
        raise ValueError
    if af.output_numpy_dim is None:
        raise ValueError
    phi_star = AdditiveFunction(psi_0=af.psi(0), psi_t=_cut_psi_t(psi=af.psi, t_cutoff=t_cutoff, output_numpy_dim=af.output_numpy_dim))
    return AdditiveFunction.combine([af, phi_star])

FloatLike = tp.TypeVar('FloatLike')
PsiFunc = tp.Callable[[_Tp, _Tp], FloatLike]

class _cut_psi_t:
    def __init__(self, psi: tp.Callable[[int], PsiFunc], t_cutoff: int, output_numpy_dim: int):
        self.psi = psi
        self.t_cutoff = t_cutoff
        self.output_numpy_dim = output_numpy_dim

    def __call__(self, t: int, xtm1, xt) -> FloatLike:
        assert t >= 1
        if t <= self.t_cutoff:
            return self.psi(t)(xtm1, xt)
        else:
            regdim = max(self.output_numpy_dim, 1)
            res = [0.0] * regdim
            return convert_to_numpy_dim(res, self.output_numpy_dim)

def assert_function_equal(f1, f2, input_dim, input_length):
    inp = [dearray(np.random.rand(*input_dim)) for _ in range(input_length)]
    if np.random.rand() < 0.5:
        inp = np.array(inp)
    res1 = f1(inp)
    res2 = f2(inp)
    assert res1.shape == res2.shape
    assert np.allclose(res1, res2)
    print('{} ok'.format(res1))

class AdtFctIterator:
    # tested
    """
    Let phi be an additive function and x be a vector of length T.
    Then the iterator returns phi_0(x_0), phi_1(x_{0:1}), etc...
    """
    def __init__(self, adf: AdditiveFunction, x: tp.Iterable):
        self.adf = adf
        self.x_iterator = iter(x)

        self.xtm1 = None
        self.xtm2 = None
        self.t = 0
        self.current_var = None

    def __iter__(self):
        return self

    def __next__(self):
        # must return phi_t(x_t)
        xt = next(self.x_iterator)
        if self.t == 0:
            self.current_var = self.adf.psi(self.t)(xt)
        else:
            self.current_var = self.current_var + self.adf.psi(self.t)(self.xtm1, xt)

        self.xtm2 = self.xtm1
        self.xtm1 = xt
        self.t += 1

        return self.current_var

class SmoothableSSM(ABC):
    @abstractmethod
    def get_new_fk(self, *args, **kwargs) -> FeynmanKac:
        """
        Get a new Feynman-Kac model. Should not change attributes of `self`.
        """
        ...

    @abstractmethod
    def get_new_intractable_fk(self, *args, **kwargs) -> IntractableFK:
        ...

    @abstractmethod
    def logpt(self, t: int, xp, x) -> float:
        """
        Non-vectorized function for calculating the log transition probability logp_t(x_{t-1},x_t)
        """
        ...

    @abstractmethod
    def upper_bound_logpt(self, t:int, x) -> float:
        """
        Non-vectorized function for calculating an upper bound for logp_t(*, x_t). Currently only accepts bound that does not depend on x_t.
        """
        ...

    @property
    @abstractmethod
    def T(self) -> int:
        ...

    def _typical_additive_function(self, transform: tp.Callable[[_Tp], float]) -> AdditiveFunction:
        """
        In plain words, we wish to calculate the expectation of the vector-valued random variable
        [tr(X0), tr(X_{T/2}), tr(X_{T-1}), tr(X0) + ... + tr(X_T)]
        """
        # tested
        set_t = sorted({max(self.T-1, 0), self.T//2, 0})
        # noinspection PyTypeChecker
        return AdditiveFunction.combine([AdditiveFunction(psi_0=partial(self._psi_0, t=t, transform=transform), psi_t=partial(self._psi, t=t, transform=transform)) for t in set_t] +
                                        [AdditiveFunction(psi_0=transform, psi_t=composition(transform, itemgetter(2), pack))])

    @abstractmethod
    def typical_additive_function(self) -> AdditiveFunction:
        """
        The expectations of which will be calculated by default
        """
        ...

    @staticmethod
    def _psi_0(x, t, transform):
        return transform(x) if t == 0 else 0

    # noinspection PyUnusedLocal
    @staticmethod
    def _psi(s, x, y, t, transform):
        return transform(y) if t == s else 0

    @abstractmethod
    def backward_gaussianizer(self, t: int) -> BackwardDistGaussianizer:
        """
        Gaussianize the distribution of q_{t-1}(x_{t-1} | x_t).
        Useful for creating discrete Gaussian proposals.
        """
        ...

    @abstractmethod
    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist: CategoricalDistribution, t: int) -> tp.Sequence[tp.Sequence[float]]:
        """
        Transform the points X_{t-1}^{1:N} before sorting them by the Hilbert curve. Useful for the Hilbert MCMC backward kernel.
        """
        return x_tm1

    @abstractmethod
    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        """
        Returns the exact value of Q_t[additive_function(t)] if available and raises TypeError otherwise
        """
        raise TypeError

    @abstractmethod
    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        """
        Returns the exact value of Q_T[additive_function(t)] if available and raises TypeError otherwise
        """
        raise TypeError

    @property
    @abstractmethod
    def additive_function_len(self) -> int:
        """
        Returns the expected length of the vector returned by an application of typical_additive_function
        """
        ...

    @abstractmethod
    def filtering_stability_diag_function(self, t: int, xt) -> np.ndarray:
        """
        The expectation of this function will be calculated at each filtering step in order to diagnostic whether the filtering is stable
        """
        ...

_Ty = tp.TypeVar('_Ty')

class TNArray(tp.Generic[_Ty]):
    """
    Class to manipulate the T x N arrays generated by smoothing procedures
    """
    def __init__(self, arr: tp.Sequence[tp.Sequence[_Ty]]):
        self.T = len(arr) - 1
        self.N = len(arr[0])
        for arr_t in arr:
            if len(arr_t) != self.N:
                raise ValueError('The given array is not a T x N array')
        self.arr = arr

    def get_trajectory(self, n: int) -> tp.Sequence[_Ty]:
        return [s[n] for s in self.arr]

    def get_marginal(self, t:int) -> tp.Sequence[_Ty]:
        return self.arr[t]

    def get_marginals(self, list_t: tp.Sequence[int]) -> tp.Sequence[tp.Sequence[_Ty]]:
        return [[self.arr[t][n] for t in list_t] for n in range(self.N)]

class SmoothingResult(tp.NamedTuple):
    samples_idx: TNArray[int]
    samples: TNArray[_Tp]
    costs: TNArray[_CostInfo]

def backward_smoothing(X: tp.Sequence[tp.Sequence[_Tp]], filtering_dists: tp.Sequence[CategoricalDistribution], A: tp.Sequence[tp.Optional[tp.Sequence[int]]], backward_samplers: tp.Sequence[tp.Optional[BackwardSampler]]) -> SmoothingResult:
    # tested
    # Rational behind this design: X, W and A are changing objects and should be manipulated by the general FFBS or PaRIS wrapper. In contrast, transition log density and so on is included in the problem description, and may not be necessary for all backward samplers.
    """
    :param X: X[t] is the *filtering* particles at time `t`
    :param filtering_dists: filtering_dists[t] represents the *filtering* weights of particles at time `t`
    :param A: A[t] is the ancestors of particles X[t]. A[0] should be None.
    :param backward_samplers: backward_samplers[t] is used to sample X_{t-1} | X_t. backward_samplers[0] should be None.
    :return: a SmoothingResult object. The `costs` attribute is such that `costs[t]` gives the costs for backward-sampling *starting from* particles `X[t]`. Thus `costs[0]` should be [None] * N.
    """
    check_n_tilde = [bs.N_tilde == 1 for bs in backward_samplers if bs is not None]
    if not all(check_n_tilde):
        print('Warning: N_tilde is not set to 1.')

    T = len(X) - 1
    N = len(X[T])

    samples: tp.Sequence[tp.Union[None, tp.Sequence[_Tp]]] = [None] * (T + 1)
    samples_idx: tp.Sequence[tp.Union[None, tp.Sequence[int]]] = [None] * (T + 1)
    costs: tp.Sequence[tp.Union[None, tp.Sequence[_CostInfo]]] = [None] * (T + 1)

    samples_idx[T] = filtering_dists[T].rvs(size=N)
    samples[T] = X[T][samples_idx[T]]

    for t in range(T, 0, -1):  # we know x_t and we build x_{t-1}
        assert samples_idx[t] is not None
        assert samples[t] is not None
        smoothed_At = A[t][samples_idx[t]]
        sampler_res = backward_samplers[t].__call__(filtering_dist_tm1=filtering_dists[t-1], X_tm1=X[t-1], smoothed_X_t=samples[t], smoothed_A=smoothed_At)
        samples[t-1] = np.array([ls[-1] for ls in sampler_res.samples])
        samples_idx[t-1] = np.array([li[-1] for li in sampler_res.samples_idx])
        costs[t] = sampler_res.costs
        # DEBUG
        # print(get_acc_rate_array(costs[t], 'really_moved'))
        # END DEBUG

    costs[0] = [None] * N

    return SmoothingResult(samples_idx=TNArray(samples_idx), samples=TNArray(samples), costs=TNArray(costs))

class PaRIS_Node(tp.Generic[_Tp]):
    def __init__(self, content: _Tp, children: tp.Sequence[_Tp]):
        self.content = content
        self.children = children

class PaRIS_Skeleton(tp.Generic[_Tp]):
    def __init__(self):
        self.current_roots: tp.Sequence[PaRIS_Node[_Tp]] = []
        self.t: int = -1
        self.current_W: tp.Optional[CategoricalDistribution] = None

    def add(self, X: tp.Sequence[_Tp], children_idx: tp.Sequence[tp.Sequence[int]], W: CategoricalDistribution):
        if self.t == -1:
            lens = [len(cl) for cl in children_idx]
            if any([e != 0 for e in lens]):
                raise ValueError
        self.current_roots = [
            PaRIS_Node(x, [self.current_roots[c] for c in clist])
            for x, clist in zip_with_assert(X, children_idx)
        ]
        self.current_W = W
        self.t += 1

    def _forward_trajectory(self, bw_path: tp.Sequence[int], root: int) -> tp.Sequence[_Tp]:
        """
        :param bw_path: indicates how to navigate the backward tree to get from the root at time `T` to leaves at time `0`. In the classical PaRIS algorithm with N_tilde = 2, the bw_path thus consists of elements belonging to the set {0,1}.
        """
        res = [self.current_roots[root]]
        for direction in bw_path:
            res.append(res[-1].children[direction])
        assert len(res) == self.t + 1
        res = [n.content for n in res]
        return list(reversed(res))

    def sample_one_trajectory(self) -> tp.Sequence[_Tp]:
        # todo: report to Nicolas the catastrophic performance of extract_one_trajectory
        root = self.current_roots[self.current_W.rvs()]
        res = [root]
        while len(res[-1].children) > 0:
            res.append(np.random.choice(res[-1].children))
        assert len(res) == self.t + 1
        res = [n.content for n in res]
        return list(reversed(res))

    def iter_tree(self, root_id: int) -> tp.Iterable[tp.Sequence[_Tp]]:
        mapper = partial(self._mapper_tree_iterator, root_id=root_id)
        filterer = self._filter_tree_iterator
        return map(mapper, filter(filterer,
                                  iterate_over_tree(root=self.current_roots[root_id], get_children=partial(self.get_attr_proxy, name='children'), method='BFS')
                                  ))

    @staticmethod
    def get_attr_proxy(o, name):
        return getattr(o, name)

    def _filter_tree_iterator(self, ti: TreeIterator):
        return ti.depth == self.t

    def _mapper_tree_iterator(self, ti: TreeIterator, root_id: int) -> tp.Sequence[_Tp]:
        return self._forward_trajectory(ti.path, root_id)

    def calculate_skeleton_expectation(self, f: tp.Callable[[tp.Sequence[_Tp]], tp.Sequence[float]]) -> tp.Sequence[float]:
        # tested
        means_of_each_root = []
        for root_id in range(len(self.current_roots)):
            f_values = [f(traj) for traj in self.iter_tree(root_id)]
            means_of_each_root.append(_simple_mean(f_values))
        return sum([w * m for w, m in zip_with_assert(self.current_W.W, means_of_each_root)])

class PaRIS_Input(tp.NamedTuple):
    X: tp.Sequence[_Tt]
    W: CategoricalDistribution
    A: tp.Optional[tp.Sequence[int]]
    psi: tp.Union[tp.Callable[[_Tt], float], tp.Callable[[_Ttm1, _Tt], float]]
    backward_sampler: BackwardSampler

class PaRIS_Output(tp.NamedTuple):
    online_expectation: float
    sampler_result: tp.Optional[BSResult]

class PaRIS(Generator):
    def __init__(self, history: bool):
        self.history = history

        self.current_X: tp.Optional[tp.Sequence[_Ttm1]] = None
        self.current_W: tp.Optional[CategoricalDistribution] = None
        self.current_aux_S: tp.Optional[tp.Sequence[float]] = None
        self.skeleton = PaRIS_Skeleton()
        self.t: int = 0

    def throw(self, typ, val=None, tb=None):
        super().throw(typ, val, tb)

    def send(self, val: tp.Optional[PaRIS_Input]) -> tp.Optional[PaRIS_Output]:
        # Update mutuable attributes, then return.
        if val is None:
            return None
        valX, valW, valpsi = val.X, val.W, val.psi
        if self.t == 0:
            prodd = self._process0(valW, valX, valpsi)
            return PaRIS_Output(prodd, None)
        backward_sampler = val.backward_sampler.__call__(filtering_dist_tm1=self.current_W, X_tm1=self.current_X, smoothed_X_t=val.X, smoothed_A=val.A)
        backward_sampler_samples_idx = backward_sampler.samples_idx
        prodd = self._process_t(backward_sampler_samples_idx, valW, valX, valpsi)
        return PaRIS_Output(prodd, backward_sampler)

    def _process_t(self, backward_sampler_samples_idx, valW, valX, valpsi):
        indexized_phi = self._indexized_phi(phi=valpsi, Xtm1=self.current_X, Xt=valX)
        self.current_aux_S = self._new_aux_stats(self.current_aux_S, backward_sampler_samples_idx, indexized_phi)
        if self.history:
            self.skeleton.add(valX, backward_sampler_samples_idx, valW)
        self.current_X = valX
        self.current_W = valW
        self.t += 1
        # noinspection PyTypeChecker
        prodd = _simple_scalar_prod(self.current_aux_S, self.current_W.W)
        return prodd

    def _process0(self, valW, valX, valpsi):
        self.current_aux_S = np.array([valpsi(x) for x in valX])
        if self.history:
            self.skeleton.add(valX, [[] for _ in range(len(valX))], valW)
        self.current_X = valX
        self.current_W = valW
        self.t += 1
        # noinspection PyTypeChecker
        prodd = _simple_scalar_prod(self.current_aux_S, self.current_W.W)
        return prodd

    @classmethod
    def _indexized_phi(cls, phi: tp.Callable[[_Ttm1, _Tt], float], Xtm1: tp.Sequence, Xt: tp.Sequence[_Tt]) -> tp.Callable[[int, int], float]:
        # noinspection PyTypeChecker
        return partial(cls._indexized_phi_proxy, Xtm1=Xtm1, Xt=Xt, phi=phi)

    @staticmethod
    def _indexized_phi_proxy(i, j, Xtm1, Xt, phi):
        return phi(Xtm1[i], Xt[j])

    @staticmethod
    def _new_aux_stats(old_aux_stats: tp.Sequence[float], samples_idx: tp.Sequence[tp.Sequence[int]], indexized_phi: tp.Callable[[int, int], float]) -> tp.Sequence[float]:
        res = []
        for i, bw_sample in enumerate(samples_idx):
            # noinspection PyTypeChecker
            s1 = _simple_mean(old_aux_stats[bw_sample])
            s2 = _simple_mean([indexized_phi(j, i) for j in bw_sample])
            res.append(s1 + s2)
        return np.array(res)

def test_aux_stats_consistence(paris: PaRIS, additive_func: tp.Callable[[tp.Sequence[_Tp]], float]):
    aux_stats = paris.current_aux_S
    N = len(aux_stats)
    ref = []
    for root_id in range(N):
        ref.append(_simple_mean([additive_func(traj) for traj in paris.skeleton.iter_tree(root_id)]))
    assert np.allclose(aux_stats, ref)

def test_paris_dummy_sampler(skel: PaRIS_Skeleton, pf_hist: ParticleHistory):
    for pos, skel_elem in enumerate(skel.current_roots):
        for a, x in zip(reversed(pf_hist.A[1:]), reversed(pf_hist.X[:-1])):
            pos = a[pos]
            assert len(set([id(c) for c in skel_elem.children])) == 1
            skel_elem = skel_elem.children[0]
            assert np.allclose(x[pos], skel_elem.content)
