import typing as tp
# noinspection PyProtectedMember
from libs_new.smoothing_generic_backwards import _CostInfo, get_rejection_cost_array
import numpy as np
from libs_new.smoothing_mcmc_backwards import get_acc_rate_array, get_acc_rate
from libs_new.utils import add_to_list
from libs_new.utils_math import SimulatedCosts

# Everything tested version 190821

class PerformanceExtractor:
    perf_indicators: tp.List[tp.Callable[[tp.Sequence[_CostInfo], 'PerformanceTracker'], tp.Any]] = []


class RejectionPerformanceExtractor(PerformanceExtractor):
    perf_indicators = []

    # noinspection PyUnusedLocal
    @staticmethod
    @add_to_list(perf_indicators)
    def mean_pure_cost(costs, perf_track: 'PerformanceTracker'):
        return get_rejection_cost_array(costs, 'pure')

    # noinspection PyUnusedLocal
    @staticmethod
    @add_to_list(perf_indicators)
    def mean_hybrid_cost(costs, perf_track):
        return get_rejection_cost_array(costs, 'hybrid')

    @staticmethod
    @add_to_list(perf_indicators)
    def pure_cost_highlight(costs: tp.Sequence[SimulatedCosts], perf_track: 'PerformanceTracker'):
        pure_costs = np.array([c.pure for c in costs])
        highlight = pure_costs[perf_track.highlight_idx]
        return [int(e) for e in highlight]

class MCMCPerformanceExtractor(PerformanceExtractor):
    perf_indicators = []

    # noinspection PyUnusedLocal
    @staticmethod
    @add_to_list(perf_indicators)
    def mean_acc_rate(costs: tp.Sequence[_CostInfo], perf_track):
        return get_acc_rate_array(costs, 'accepted')

    # noinspection PyUnusedLocal
    @staticmethod
    @add_to_list(perf_indicators)
    def mean_really_moved_rate(costs, perf_track):
        return get_acc_rate_array(costs, 'really_moved')

    @staticmethod
    @add_to_list(perf_indicators)
    def acc_rate_highlight(costs: tp.Sequence[_CostInfo], perf_track: 'PerformanceTracker'):
        acc_rates = np.array([get_acc_rate(c, 'accepted') for c in costs])
        highlight = acc_rates[perf_track.highlight_idx]
        return [float(e) for e in highlight]

    @staticmethod
    @add_to_list(perf_indicators)
    def really_moved_rate_highlight(costs, perf_track):
        really_moved_rates = np.array([get_acc_rate(c, 'really_moved') for c in costs])
        highlight = really_moved_rates[perf_track.highlight_idx]
        return [float(e) for e in highlight]

class DummyPerformanceExtractor(PerformanceExtractor):
    pass

class PerformanceTracker:
    # tested, using 190821 file
    algo_to_indicators: tp.Mapping[str, tp.Type[PerformanceExtractor]] = {
        'reject': RejectionPerformanceExtractor,
        'mcmc_indep': MCMCPerformanceExtractor,
        'mcmc_hilbert': MCMCPerformanceExtractor,
        'dummy': DummyPerformanceExtractor,
        'mcmc_irhilbert': MCMCPerformanceExtractor
    }

    def __init__(self, algo: str, n_highlight: int, N: int, highlight_every: int):
        self.algo = algo
        self.n_highlight = n_highlight
        self.N = N
        self.highlight_idx = np.random.randint(low=0, high=N, size=n_highlight)
        self.highlight_every = highlight_every

        tracker_class = self.algo_to_indicators[algo]
        self.tracker_funcs = tracker_class.perf_indicators
        self.tracker_dict = {f.__name__: [] for f in self.tracker_funcs}
        self.t = 0

    def add(self, costs: tp.Sequence[_CostInfo]):
        dummy_cost = (costs is None) or all([o is None for o in costs])
        if dummy_cost:
            assert self.t == 0
        t_highlight_compatible = ((self.t-1) % self.highlight_every == 0)
        for f in self.tracker_funcs:
            self.tracker_dict[f.__name__].append(f(costs, self) if (not dummy_cost) and t_highlight_compatible else None)
        self.t += 1

    def get(self) -> tp.Dict[str, list]:
        return self.tracker_dict