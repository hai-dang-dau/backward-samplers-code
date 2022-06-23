import typing as tp
from functools import cached_property, partial
from itertools import chain
from libs_new.intractable_smoother import IntractableFK, FromFKObject, no_coupled_M
from libs_new.smoothing_high_level import AdditiveFunction, SmoothableSSM, BackwardDistGaussianizer
import numpy as np
import particles
from particles import resampling as rs
from libs_new.utils import zip_with_assert, sum_of_each_row, do_sth_to_each_row_of, cached_function
import operator

from libs_new.utils_math import CategoricalDistribution, CoupledCategoricalDist, SmallSupportCategoricalDistribution as SmallCat

"""
Hidden Markov model. Used mainly for testing particle algorithms. Motivations are:
(a) Currently there has been no way to test algorithms on models with inhomogenous transition kernels,
(b) nor to test the complete smoothing distribution.
(c) Gaussian test can be misleading. On discrete spaces, histograms are constructed directly without kernels and smaller N can be enough to give precise results. Moreover, testing with large N could to be faster because calculating transition density is easy.
"""
# noinspection PyMethodParameters,PyAbstractClass
class _iHMM(particles.FeynmanKac):
    # The naming convention of this class is a little bit bizarre for historical reasons.
    # Indeed, this class was originally defined inside the get_new_fk method of the HMM class.
    # Then it was moved out to ensure pickability with multiprocessing.
    def __init__(self2, smoothable_obj):
        self2.self = smoothable_obj
        particles.FeynmanKac.__init__(self2, self2.self.T + 1)

    def M0(self2, N):
        return rs.multinomial(self2.self.initial_dist, N)

    def M(self2, t, xp):
        res = []
        for _xp in xp:
            target_prob = self2.self.transition_matrices[t][_xp, :]
            res.append(rs.multinomial_once(target_prob))
        return np.array(res)

    def logG(self2, t, xp, x):
        return np.log(self2.self.potentials[t][x])

class HMM(SmoothableSSM):
    def __init__(self, initial_dist: tp.Sequence[float], transition_matrices: tp.Sequence[tp.Optional[np.ndarray]], potentials: tp.Sequence[tp.Sequence[float]]):
        """
        :param potentials: potentials are expressed directly and not on the `log` scale
        """
        assert len(potentials) == len(transition_matrices)
        assert transition_matrices[0] is None
        assert all([valid_transition_matrix(m) for m in transition_matrices[1:]])

        self.initial_dist = initial_dist
        self.transition_matrices = transition_matrices
        self.potentials = potentials
        self.additive_shift = -len(potentials[0])/2

    @property
    def T(self):
        return len(self.potentials) - 1

    def get_new_fk(self) -> particles.FeynmanKac:
        # tested 081121
        return _iHMM(smoothable_obj=self)

    @cached_property
    def full_smoothing_dist(self) -> np.ndarray:
        # tested
        res = self.initial_dist
        for M, G in zip_with_assert(self.transition_matrices, self.potentials):
            if M is not None:
                res = res[..., np.newaxis] * M
            res = res * G
            res = res/res.sum()
        return res

    def partial_smoothing_dist(self, t_seq:tp.Sequence[int]) -> tp.Sequence[float]:
        # tested
        axis = list(range(self.T + 1))
        for t in t_seq:
            axis.remove(t)
        return np.sum(self.full_smoothing_dist, axis=tuple(axis))

    @classmethod
    def random_hmm(cls, support_sizes: tp.Sequence[int], highest_diff: float, use_exp: bool):
        def randomizer(size):
            if use_exp:
                return np.random.standard_exponential(size=size)
            else:
                return np.random.uniform(low=1, high=highest_diff, size=size)

        starting_dist = randomizer(size=support_sizes[0])
        starting_dist = starting_dist/starting_dist.sum()
        trans_mat = []
        potentials = []
        for i in range(len(support_sizes)):
            if i == 0:
                trans_mat.append(None)
            else:
                m = randomizer(size=(support_sizes[i-1], support_sizes[i]))
                m = do_sth_to_each_row_of(m, sum_of_each_row(m), '/')
                trans_mat.append(m)
            g = randomizer(size=support_sizes[i])
            potentials.append(g)
        return cls(initial_dist=starting_dist, transition_matrices=trans_mat, potentials=potentials)

    def logpt(self, t:int, x_tm1: int, x_t: int):
        return np.log(self.transition_matrices[t][x_tm1, x_t])

    def upper_bound_logpt(self, t:int, x):
        return np.log(np.max(self.transition_matrices[t]))

    def get_random_additive_function(self) -> AdditiveFunction:
        res = [np.random.rand(*m.shape) for m in chain([self.initial_dist], self.transition_matrices[1:])]
        # noinspection PyTypeChecker
        return AdditiveFunction(res[0].__getitem__, partial(self._tuplize_and_getitem, res))

    @staticmethod
    def _tuplize_and_getitem(res, i, *args):
        return res[i].__getitem__(args)

    def typical_additive_function(self) -> AdditiveFunction:
        return AdditiveFunction.combine([self._typical_additive_function(partial(operator.add, self.additive_shift)), self.get_random_additive_function()])

    def backward_gaussianizer(self, t: int) -> 'BackwardDistGaussianizer':
        raise AssertionError

    def exact_smoothing_expectation(self, t: int, unlock=False) -> float:
        if unlock:
            W = self.partial_smoothing_dist([t])
            nt = len(self.potentials[t])
            assert W.shape == (nt,)
            return float(np.sum(W * np.arange(nt)))
        else:
            raise TypeError

    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]: raise NotImplementedError

    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]: raise NotImplementedError

    @property
    def additive_function_len(self) -> int:
        set_t = sorted({max(self.T-1, 0), self.T//2, 0})
        return len(set_t) + 2

    def get_new_intractable_fk(self, coupled: bool) -> IntractableFK:
        fk = self.get_new_fk()
        if coupled:
            return IntractableFK.fromFKObject(fk=fk, coupled_M=self.coupled_M_mixin)
        else:
            return IntractableFK.fromFKObject(fk=fk, coupled_M=no_coupled_M)

    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist: CategoricalDistribution, t: int) -> tp.Sequence[tp.Sequence[float]]: raise NotImplementedError

    @cached_function
    def coupled_dist(self, t:int, i: int, j: int):
        assert t >= 1
        d1 = SmallCat.fromWeights(W=self.transition_matrices[t][i], unsafe_exp_and_normalise=True)
        d2 = SmallCat.fromWeights(W=self.transition_matrices[t][j], unsafe_exp_and_normalise=True)
        return CoupledCategoricalDist(d1, d2)

    # noinspection PyUnusedLocal
    def coupled_M_mixin(self, self2: FromFKObject, t, i, j) -> tp.Tuple[int, int, bool]:
        t, i, j = int(t), int(i), int(j)
        return self.coupled_dist(t, i, j).rvs()

    def filtering_stability_diag_function(self, t: int, xt) -> np.ndarray:
        return xt

def valid_transition_matrix(M: np.ndarray) -> bool:
    for i in range(len(M)):
        if not np.allclose(np.sum(M[i]), 1):
            return False
    if not np.all(M >= 0):
        return False
    return True

class Flattened(tp.NamedTuple):
    empirical_samples: tp.Sequence[int]
    proba: tp.Sequence[float]

def flatten(empirical_samples: tp.Sequence[tp.Sequence[int]], proba: np.ndarray) -> Flattened:
    # tested
    """
    Assign to each (T+1)-tuple (x0, ..., x_T) an integer code value and return the so flattened empirical samples and exact probability distribution of a joint discrete distribution.

    :param empirical_samples: a list of samples from the joint discrete distribution (X_0, ..., X_T). Each sample is therefore a sequence of T integers
    :param proba: the exact probability matrix, indexed by where proba[i0, ..., iT]
    """
    emps = np.ravel_multi_index(np.array(empirical_samples).T, proba.shape)
    proba = proba.ravel()
    return Flattened(empirical_samples=emps, proba=proba)