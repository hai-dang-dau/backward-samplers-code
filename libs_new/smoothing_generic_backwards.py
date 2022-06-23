import typing as tp
from functools import partial
from libs_new.utils_math import CategoricalDistribution, multiple_discrete_rejection_sampler, SimulatedCosts
from abc import ABC, abstractmethod
from libs_new.utils import reusable_map
from tqdm import tqdm
import numpy as np

_Ttm1 = tp.TypeVar('_Ttm1')  # Type of particle at time t - 1
_Tt = tp.TypeVar('_Tt')  # Type of particle at time t
_CostInfo = tp.TypeVar('_CostInfo')
_Tp = tp.TypeVar('_Tp')  # Generic type of particle

class TargetLogDensityCalculator:
    def __init__(self, log_transition_density: tp.Callable[[_Ttm1, _Tt], float]):
        self.log_transition_density = log_transition_density

    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], x_t: _Tt) -> tp.Callable[[int], float]:
        # noinspection PyTypeChecker
        return partial(self._target_log_density, filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t)

    def _target_log_density(self, i:int, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], x_t: _Tt) -> float:
        return filtering_dist_tm1.logpdf(i) + self.log_transition_density(X_tm1[i], x_t)

class BSResult(tp.NamedTuple):
    samples_idx: tp.Sequence[tp.Sequence[int]]
    samples: tp.Iterable[tp.Sequence[_Ttm1]]
    costs: tp.Sequence[_CostInfo]

class _BSResult(tp.NamedTuple):
    samples_idx: tp.Sequence[tp.Sequence[int]]
    costs: tp.Sequence[_CostInfo]

class BackwardSampler(ABC):
    def __init__(self, N_tilde: int, verbose: bool):
        self.N_tilde = N_tilde
        self.verbose = verbose

    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt], smoothed_A: tp.Sequence[int]) -> BSResult:
        _res = self._call(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, smoothed_X_t=smoothed_X_t, smoothed_A=smoothed_A)
        # samples = [X_tm1[seq] for seq in _res.samples_idx]
        samples = reusable_map(X_tm1.__getitem__, _res.samples_idx)
        return BSResult(samples_idx=_res.samples_idx, samples=samples, costs=_res.costs)

    @abstractmethod
    def _call(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt], smoothed_A: tp.Sequence[int]) -> _BSResult:
        ...

class RejectionBackwardSampler(BackwardSampler):
    # fully tested
    def __init__(self, target_log_density_calc: TargetLogDensityCalculator, log_transition_bound: float, N_tilde: int, verbose: bool):
        super().__init__(N_tilde=N_tilde, verbose=verbose)
        self.target_log_density_calc = target_log_density_calc
        self.log_transition_bound = log_transition_bound

    def _call(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt], smoothed_A: tp.Sequence[int]) -> _BSResult:
        samples_idx = []
        costs = []
        for x_t in tqdm(smoothed_X_t, disable=not self.verbose):
            log_target_density = self.target_log_density_calc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t)
            sampler = multiple_discrete_rejection_sampler(proposal=filtering_dist_tm1, log_target_density=log_target_density, log_M=self.log_transition_bound, N_tilde=self.N_tilde)
            samples_idx.append(np.array(sampler.samples))
            costs.append(sampler.costs)
        return _BSResult(samples_idx=samples_idx, costs=costs)

class DummyBackwardSampler(BackwardSampler):
    """
    A backward sampler which just returns the ancestor. The "cost" of sampling the ancestor is the parent itself. This backward sampler is useful for sanity testing purpose.
    """
    def _call(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt], smoothed_A: tp.Sequence[int]) -> _BSResult:
        return _BSResult(samples_idx=[[a] * self.N_tilde for a in smoothed_A], costs=smoothed_X_t)

def get_rejection_cost_array(costs: tp.Sequence[SimulatedCosts], cost_type: tp.Literal['hybrid', 'pure']):
    return float(np.mean([getattr(c, cost_type) for c in costs]))

def simple_mean(x):
    """
    Get the mean of an array of vector
    """
    # Alternatively:
    # np.mean(np.atleast_1d(x), axis=0)
    return sum(x)/len(x)

def simple_scalar_prod(x, y):
    return sum([u * v for u, v in zip(x, y)])

def simple_med(x):
    return np.median(np.atleast_1d(x), axis=0)