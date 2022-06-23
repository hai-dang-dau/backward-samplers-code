from libs_new.smoothing_generic_backwards import BackwardSampler, _BSResult, TargetLogDensityCalculator
import typing as tp
from libs_new.utils import zip_with_assert, describe
from libs_new.utils_math import CategoricalDistribution, MCMCRunner, MCMCKernel, IndepMHKernel, MHInfo, \
    MHCompositeKernel
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

_Ttm1 = tp.TypeVar('_Ttm1')  # Type of particle at time t - 1
_Tt = tp.TypeVar('_Tt')  # Type of particle at time t
_CostInfo = tp.TypeVar('_CostInfo')
_Tp = tp.TypeVar('_Tp')  # Generic type of particle

class MCMCStartKit(tp.NamedTuple):
    kernel: tp.Union[MCMCKernel, MHCompositeKernel]
    starting_point: int

class MCMCKernelsCreator(ABC):
    @abstractmethod
    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        ...

class KAdapter(ABC):
    @abstractmethod
    def __call__(self, mcmcs: tp.Sequence[MCMCStartKit]) -> tp.Sequence[int]:
        ...

class ConstantK(KAdapter):
    def __call__(self, mcmcs: tp.Sequence[MCMCStartKit]) -> tp.Sequence[int]:
        return [self.k] * len(mcmcs)

    def __init__(self, k: int):
        self.k = k

class UntilAccept(KAdapter):
    # tested
    def __init__(self, mul: int):
        self.mul = mul

    def __call__(self, mcmcs: tp.Sequence[MCMCStartKit]) -> tp.Sequence[int]:
        res = []
        for (kernel, start) in mcmcs:
            runner = MCMCRunner(kernel=kernel, starting_point=start)
            i = 0
            for obj in runner:
                i += 1
                if obj.current_info.accepted > 0:
                    break
            res.append(i)
        return [j * self.mul for j in res]


class MCMCBackwardSampler(BackwardSampler):
    # tested
    def __init__(self, k_adapter: KAdapter, mcmc_creator: MCMCKernelsCreator, N_tilde: int, verbose: bool):
        self.k_adapter = k_adapter
        self.mcmc_creator = mcmc_creator
        super().__init__(N_tilde=N_tilde, verbose=verbose)

    def _call(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1], smoothed_X_t: tp.Sequence[_Tt], smoothed_A: tp.Sequence[int]) -> _BSResult:
        mcmc_starters = self.mcmc_creator(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, smoothed_X_t=smoothed_X_t)
        chosen_ks = self.k_adapter(mcmc_starters)
        mcmc_kernels = [m.kernel for m in mcmc_starters]

        samples_idx = []
        cost_info = []

        for ker, a, k in tqdm(zip_with_assert(mcmc_kernels, smoothed_A, chosen_ks), disable=not self.verbose):
            mcmc_runner = MCMCRunner(kernel=ker, starting_point=a)
            mcmc_result = [a]
            mcmc_info = [None]
            for _, r in zip(range(k * self.N_tilde), mcmc_runner):
                mcmc_result.append(r.current_state)
                mcmc_info.append(r.current_info)
            samples_idx.append(mcmc_result)
            cost_info.append(mcmc_info)

        return _BSResult(samples_idx=samples_idx, costs=cost_info)

class IndepCreator(MCMCKernelsCreator):
    """
    This is related to the recent paper on Backward Importance Sampling by Le Corff and Martin. Their estimator is biased but will become unbiased if the ancestor is included in the backward sampling step. It is related to the Particle Gibbs sampler with T = 0.
    """
    # tested
    def __call__(self, filtering_dist_tm1: CategoricalDistribution, X_tm1: tp.Sequence[_Ttm1],
                 smoothed_X_t: tp.Sequence[_Tt]) -> tp.Sequence[MCMCStartKit]:
        res = []
        for x_t in smoothed_X_t:
            target_dens = self.target_log_density_calc(filtering_dist_tm1=filtering_dist_tm1, X_tm1=X_tm1, x_t=x_t)
            kernel = IndepMHKernel(log_target=target_dens, proposal_dist=filtering_dist_tm1)
            starting_point = filtering_dist_tm1.rvs()
            res.append(MCMCStartKit(kernel=kernel, starting_point=starting_point))
        return res

    def __init__(self, target_log_density_calc: TargetLogDensityCalculator):
        self.target_log_density_calc = target_log_density_calc

def get_acc_rate(costs: tp.Sequence[tp.Optional[MHInfo]], acc_type: tp.Literal['accepted', 'really_moved'] = 'accepted'):
    assert costs[0] is None
    nacc = sum([getattr(c, acc_type) for c in costs[1:]])
    return nacc/(len(costs) - 1)

def get_acc_rate_array(costs: tp.Sequence[tp.Sequence[MHInfo]], acc_type: tp.Literal['accepted', 'really_moved']) -> float:
    accs = [get_acc_rate(c, acc_type) for c in costs]
    return float(np.mean(accs))

def get_acc_rate_iq_range(costs: tp.Sequence[tp.Sequence[MHInfo]], acc_type: tp.Literal['accepted', 'really_moved'], qlow=0.25) -> tp.Tuple[float, float]:
    # todo: performanceTracker add this thingy!
    accs = [get_acc_rate(c, acc_type) for c in costs]
    return float(np.quantile(accs, qlow)), float(np.quantile(accs, 1-qlow))

class AccRateAnalyzer(tp.NamedTuple):
    acc: pd.Series
    really_moved: pd.Series
    niters: pd.Series

def acc_rate_analyzer(costs: tp.Sequence[tp.Sequence[MHInfo]]) -> AccRateAnalyzer:
    acc = [get_acc_rate(c, 'accepted') for c in costs]
    really_moved = [get_acc_rate(c, 'really_moved') for c in costs]
    niters = [len(c) - 1 for c in costs]
    return AccRateAnalyzer(acc=describe(acc), really_moved=describe(really_moved), niters=describe(niters))
