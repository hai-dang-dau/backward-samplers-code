import typing as tp
from time import process_time

import numpy as np
from abc import abstractmethod
from libs_new import intractable_smoother as intrctsm
import particles

from libs_new.smoothing_high_level import SmoothableSSM, SmoothingResult
from libs_new.smoothing_ui import BackwardKernelOption, DummyBKOption, RejectionBKOption, IndependentMCMCBKOption, \
    HilbertMCMCBKOption, easy_FFBS, EasyPaRIS, IrreversibleHilbertBKOption, EasyIntractableSmoother
from libs_new.utils import read_json, temporary_numpy_seed, memory_protection
from programs.performance_tracker import PerformanceTracker
from libs_new.smoothing_generic_backwards import simple_scalar_prod, simple_med
from programs.workflow import MultipleRuns

# Everything tested version 190821

def parse_backward_option(d: dict) -> tp.Optional[BackwardKernelOption]:
    """
    :param d: dictionary resulted from one line of the input csv file
    """
    if d['smooth_mode'] == 'offline' or d['algo'].startswith('mcmc'):
        assert d['N_tilde'] == 1
    if d['algo'] == 'dummy':
        return DummyBKOption(verbose=False)
    if d['algo'] == 'reject':
        return RejectionBKOption(N_tilde=d['N_tilde'], verbose=False)
    if d['algo'] == 'mcmc_indep':
        return IndependentMCMCBKOption(k=d['k'], verbose=False)
    if d['algo'] == 'mcmc_hilbert':
        return HilbertMCMCBKOption(k=d['k'], max_rankset_size=d['max_rss'], verbose=False)
    if d['algo'] == 'mcmc_irhilbert':
        return IrreversibleHilbertBKOption(k=d['k'], verbose=False)
    if d['smooth_mode'] == 'intractable':
        assert d['algo'] == 'intractable'
        return
    raise ValueError

class _IntermediateResult(tp.NamedTuple):
    expectations: list
    costs: tp.MutableMapping[str, list]
    ESS_ratios: tp.List[float]

class SmoothingExperiment(MultipleRuns):
    # tested, even for intractable smoothers
    def one_run(self, d: dict):
        model = self.model_parser(d)
        get_new_fk_args = self.get_new_fk_args_parser(d)
        bw_option = parse_backward_option(d)
        expectations, costs, ESS_ratios = self.model_runner(d=d, model=model, bw_option=bw_option, get_new_fk_args=get_new_fk_args)
        costs['expectations'] = expectations
        costs['ESS_ratios'] = ESS_ratios
        return costs

    @classmethod
    def model_parser(cls, d:dict) -> SmoothableSSM:
        assert cls.config_path().endswith('/')
        file_path = cls.config_path() + d['config_file'] + '.json'
        config_dict: dict = read_json(file_path)
        seed_model = config_dict.pop('seed_model')
        model_class = config_dict.pop('model_class')
        with temporary_numpy_seed(seed_model):
            return getattr(cls.model_module(), model_class)(**config_dict)

    @staticmethod
    @abstractmethod
    def config_path() -> str:
        ...

    @staticmethod
    @abstractmethod
    def model_module():
        ...

    @staticmethod
    @abstractmethod
    def get_new_fk_args_parser(d: dict) -> dict:
        return dict(fk_type=d['fk_type'])

    @staticmethod
    def model_runner(d:dict, model: SmoothableSSM, bw_option: BackwardKernelOption, get_new_fk_args: dict) -> _IntermediateResult:
        if d['smooth_mode'] == 'offline':
            return model_runner_offline(d, model, bw_option, get_new_fk_args)
        if d['smooth_mode'] == 'online':
            return model_runner_online(d, model, bw_option, get_new_fk_args)
        if d['smooth_mode'] == 'intractable':
            return model_runner_intractable(d, model, bw_option, get_new_fk_args)
        raise ValueError

def model_runner_offline(d: dict, model: SmoothableSSM, bw_option: BackwardKernelOption, get_new_fk_args: dict, verbose_pf=False, accumulator=simple_med) -> _IntermediateResult:
    # tested
    # todo: add efficient memory protection + implement filtering stability diagnostics
    offline_smoother, pf = easy_FFBS(ssm=model, N=d['N'], backward_option=bw_option, resampling=d['resampling'], get_new_fk_args=get_new_fk_args, ESSrmin=d['ESSrmin'], verbose_pf=verbose_pf, memlimit=d['maxmem'])
    offline_smoother: SmoothingResult
    pf: particles.SMC
    expectations = []
    ESS_ratios = []
    perf_tracker = PerformanceTracker(algo=d['algo'], n_highlight=d['n_highlight'], N=d['N'], highlight_every=d['highlight_every'])
    sample = [offline_smoother.samples.get_trajectory(n) for n in range(d['N'])]
    adf_iterators = [model.typical_additive_function().iterate(x) for x in sample]
    for t in range(model.T + 1):
        memory_protection(d['maxmem'])
        phi_t = [next(it) for it in adf_iterators]
        mean = accumulator(phi_t)
        mean = [float(m) for m in mean]
        expectations.append(mean)
        # assert len(mean) == model.additive_function_len
        assert correct_len(mean, model.additive_function_len)
        perf_tracker.add(offline_smoother.costs.get_marginal(t))

        essr = pf.hist.wgts[t].ESS / d['N']
        ESS_ratios.append(float(essr))
    return _IntermediateResult(expectations=expectations, costs=perf_tracker.get(), ESS_ratios=ESS_ratios)

def model_runner_online(d: dict, model: SmoothableSSM, bw_option: BackwardKernelOption, get_new_fk_args: dict, verbose_pf=False) -> _IntermediateResult:
    # tested
    online_smoother = EasyPaRIS(ssm=model, backward_option=bw_option, N=d['N'], resampling=d['resampling'], history=False, ESSrmin=d['ESSrmin'], get_fk_kwargs=get_new_fk_args, verbose_pf=verbose_pf, skeleton_converter=d['skeleton_converter'])
    expectations = []
    ESS_ratios = []
    perf_tracker = PerformanceTracker(algo=d['algo'], n_highlight=d['n_highlight'], N=d['N'], highlight_every=d['highlight_every'])
    filtering_stability_diag = []
    exec_times = []
    for t, paris_output in timed_iterate(enumerate(online_smoother), exec_times):
        memory_protection(d['maxmem'])
        # assert len(paris_output.online_expectation) == model.additive_function_len
        assert correct_len(paris_output.online_expectation, model.additive_function_len)
        expectations.append(to_pure_python(paris_output.online_expectation))
        perf_tracker.add(paris_output.sampler_result.costs if paris_output.sampler_result is not None else None)

        essr = online_smoother.pf.wgts.ESS / d['N']
        ESS_ratios.append(float(essr))

        update_filtering_stability_diag(filtering_stability_diag, model, online_smoother, t)
    costs = perf_tracker.get()
    costs['filtering_stability_diag'] = filtering_stability_diag
    costs['exec_times'] = exec_times
    return _IntermediateResult(expectations=expectations, costs=costs, ESS_ratios=ESS_ratios)


def update_filtering_stability_diag(filtering_stability_diag, model, online_smoother, t):
    fX = [model.filtering_stability_diag_function(t=t, xt=xt) for xt in online_smoother.pf.X]
    W = online_smoother.pf.wgts.W
    filtering_stability_diag.append(to_pure_python(simple_scalar_prod(W, fX)))


def to_pure_python(x: tp.Union[float, np.ndarray]) -> tp.Union[float, list]:
    try:
        return float(x)
    except TypeError:
        # noinspection PyTypeChecker
        return x.tolist()

def correct_len(a: tp.Union[float, tp.Sequence[float]], b: int) -> bool:
    try:
        float(a)
        return b == 0
    except TypeError:
        return len(a) == b


# noinspection PyUnusedLocal
def model_runner_intractable(d: dict, model: SmoothableSSM, bw_option, get_new_fk_args: dict, verbose_pf=False) -> _IntermediateResult:
    # tested
    easy = EasyIntractableSmoother(ssm=model, N=d['N'], get_intractable_fk_option=get_new_fk_args, history=False, verbose=verbose_pf, reorderer=getattr(intrctsm, d['ancestor_coupling_mode'])())
    expectations = []
    costs = dict(coupling_rate=[])
    ESSr = []
    filtering_stability_diag = []
    exec_times = []
    for t, step in timed_iterate(enumerate(easy), exec_times):
        memory_protection(d['maxmem'])
        expectations.append(to_pure_python(step.online_expectation))
        assert correct_len(step.online_expectation, model.additive_function_len)
        costs['coupling_rate'].append(easy.pf.coupling_success_rate)
        ESSr.append(easy.pf.wgts.ESS/d['N'])

        update_filtering_stability_diag(filtering_stability_diag, model, easy, t)
    costs['filtering_stability_diag'] = filtering_stability_diag
    costs['exec_times'] = exec_times
    try:
        optimiser = easy.fk.mixin_coupled_M.kernels[0].keywords['optimiser']
        costs.update(optimiser.call_stats)
    except (KeyError, AttributeError):
        pass
    return _IntermediateResult(expectations=expectations, costs=costs, ESS_ratios=ESSr)

class timed_iterate:
    def __init__(self, iterable: tp.Iterable, save_target: list):
        self.iterator = iter(iterable)
        self.save_target = save_target

    def __iter__(self):
        return self

    def __next__(self):
        t1 = process_time()
        res = next(self.iterator)
        t2 = process_time()
        self.save_target.append(t2 - t1)
        return res