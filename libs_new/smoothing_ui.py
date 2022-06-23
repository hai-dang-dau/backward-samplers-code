import time

import particles
import typing as tp

from libs_new.intractable_paris_skeleton import IntractablePaRIS, iPaRIS_Input
from libs_new.intractable_smoother import IntractableSMC, Reorderer
from libs_new.utils import run_pf_with_memory_protection
from libs_new.utils_math import CategoricalDistribution
from libs_new.smoothing_mcmc_backwards import ConstantK, MCMCBackwardSampler, TargetLogDensityCalculator, IndepCreator
from libs_new.smoothing_generic_backwards import DummyBackwardSampler, RejectionBackwardSampler, BackwardSampler
from libs_new.skeleton_twister import SimpleResampleConverter, GaussianConstrictConverter, IdentityConverter
from libs_new.smoothing_high_level import PaRIS, PaRIS_Input, SmoothableSSM, SmoothingResult, backward_smoothing, \
    PaRIS_Output
from functools import partial
from libs_new.mcmc_on_mesh import NeighborMCMCKreator, HilbertMoveCreator, IrreversibleHilbertKernel
from abc import ABC, abstractmethod
import numpy as np

class BackwardKernelOption(ABC):
    @abstractmethod
    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        """
        :param t: `t` is parametrised such that the involved forward dynamics is m_t(x_{t-1}, x_t).
        """
        ...

    def get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> tp.Optional[BackwardSampler]:
        if t > 0:
            return self._get_backward_kernel(ssm=ssm, t=t)

    def get_all_backward_kernels(self, ssm: SmoothableSSM) -> tp.List[tp.Optional[BackwardSampler]]:
        return [self.get_backward_kernel(ssm=ssm, t=t) for t in range(0, ssm.T + 1)]

    @staticmethod
    def tldc(ssm: SmoothableSSM, t: int) -> TargetLogDensityCalculator:
        return TargetLogDensityCalculator(log_transition_density=partial(ssm.logpt, t))

class DummyBKOption(BackwardKernelOption):
    # tested
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        return DummyBackwardSampler(N_tilde=1, verbose=self.verbose)

class RejectionBKOption(BackwardKernelOption):
    # tested
    def __init__(self, N_tilde: int, verbose: bool):
        self.verbose = verbose
        self.N_tilde = N_tilde

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        tldc = self.tldc(ssm, t)
        return RejectionBackwardSampler(target_log_density_calc=tldc, log_transition_bound=ssm.upper_bound_logpt(t, ...), N_tilde=self.N_tilde, verbose=self.verbose)

class IndependentMCMCBKOption(BackwardKernelOption):
    # tested
    def __init__(self, k: int, verbose: bool):
        self.k = k
        self.verbose = verbose

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        tldc = self.tldc(ssm, t)
        k_adapter = ConstantK(k=self.k)
        mcmc_kernels_creator = IndepCreator(target_log_density_calc=tldc)
        return MCMCBackwardSampler(k_adapter=k_adapter, mcmc_creator=mcmc_kernels_creator, N_tilde=1, verbose=self.verbose)

class NeighborMCMCBKOption(BackwardKernelOption):
    # tested
    def __init__(self, k:int, cell_maxsize:int, verbose: bool):
        self.k = k
        self.verbose = verbose
        self.cell_maxsize = cell_maxsize

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        k_adapter = ConstantK(k=self.k)
        tldc = self.tldc(ssm, t)
        mcmc_creator = NeighborMCMCKreator(tldc=tldc, nmesh_maxsize=self.cell_maxsize)
        return MCMCBackwardSampler(k_adapter=k_adapter, mcmc_creator=mcmc_creator, N_tilde=1, verbose=self.verbose)

class HilbertMCMCBKOption(BackwardKernelOption):
    # tested
    def __init__(self, k: int, max_rankset_size: int, verbose: bool, use_lw: bool = True, puresort: bool = False):
        self.k = k
        self.max_rankset_size = max_rankset_size
        self.verbose = verbose
        self.use_lw = use_lw
        self.puresort = puresort

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        k_adapter = ConstantK(k=self.k)
        tldc = self.tldc(ssm=ssm, t=t)
        presort_transformer = partial(ssm.presort_transform, t=t)
        # noinspection PyTypeChecker
        mcmc_creator = HilbertMoveCreator(tldc=tldc, max_rss=self.max_rankset_size, presort_transformer=presort_transformer, use_lw=self.use_lw, puresort=self.puresort)
        return MCMCBackwardSampler(k_adapter=k_adapter, mcmc_creator=mcmc_creator, N_tilde=1, verbose=self.verbose)

class IrreversibleHilbertBKOption(BackwardKernelOption):
    # tested
    def __init__(self, k: int, verbose: bool):
        self.k = k
        self.verbose = verbose

    def _get_backward_kernel(self, ssm: SmoothableSSM, t: int) -> BackwardSampler:
        k_adapter = ConstantK(k=self.k)
        tldc = self.tldc(ssm=ssm, t=t)
        presort_transformer = partial(ssm.presort_transform, t=t)
        # noinspection PyTypeChecker
        mcmc_creator = IrreversibleHilbertKernel(tldc=tldc, presort_transformer=presort_transformer)
        return MCMCBackwardSampler(k_adapter=k_adapter, mcmc_creator=mcmc_creator, N_tilde=1, verbose=self.verbose)

def _identity(x):
    return x

def temper_ancestor(A: np.ndarray) -> np.ndarray:
    if A is not None:
        N = len(A)
        assert A.shape == (N, )
        return np.random.randint(low=0, high=N, size=N)

class EasyPaRIS:
    # tested
    def __init__(self, ssm: SmoothableSSM, backward_option: BackwardKernelOption, N:int, resampling: str, history: bool, ESSrmin: float = 0.5, get_fk_kwargs: dict = None, verbose_pf: bool = False, skeleton_converter: tp.Literal['identity', 'gconstrict', 'resample'] = 'identity', temper=_identity):
        if get_fk_kwargs is None:
            get_fk_kwargs = {}
        if skeleton_converter == 'identity':
            skeleton_converter = IdentityConverter()
        elif skeleton_converter == 'gconstrict':
            skeleton_converter = GaussianConstrictConverter()
            assert isinstance(backward_option, RejectionBKOption)
        elif skeleton_converter == 'resample':
            skeleton_converter = SimpleResampleConverter()
            assert isinstance(backward_option, RejectionBKOption)
        else:
            raise ValueError
        self.skeleton_converter = skeleton_converter
        self.ssm = ssm
        self.backward_option = backward_option
        self.pf = particles.SMC(fk=ssm.get_new_fk(**get_fk_kwargs), N=N, resampling=resampling, ESSrmin=ESSrmin, verbose=verbose_pf)
        self.paris = PaRIS(history=history)
        self.add_func = ssm.typical_additive_function()
        self.temper = temper

        self.paris.send(None)

    def __iter__(self):
        return self

    def __next__(self) -> tp.Optional[PaRIS_Output]:
        next(self.pf)
        t = self.pf.t - 1
        bw_sampler = self.backward_option.get_backward_kernel(self.ssm, t)
        X, lw = self.skeleton_converter(X=self.pf.X, lw=self.pf.wgts.lw)
        A = self.temper(self.pf.A) if isinstance(self.skeleton_converter, IdentityConverter) else None
        # If the skeleton_convert is not IdentityConverter, then the particles to which self.pf.A refers
        # may not be present on the skeleton!
        return self.paris.send(PaRIS_Input(X=X, W=CategoricalDistribution(lw), A=A, psi=self.add_func.psi(t), backward_sampler=bw_sampler))

class EasyFFBSResult(tp.NamedTuple):
    smoothing_result: SmoothingResult
    pf: particles.SMC

def easy_FFBS(ssm: SmoothableSSM, N: int, backward_option: BackwardKernelOption, resampling: str = 'systematic', get_new_fk_args: dict = None, ESSrmin: float = 0.5, verbose_pf: bool = False, debug_tamper=_identity, memlimit: float = np.inf) -> EasyFFBSResult:
    # tested
    if get_new_fk_args is None:
        get_new_fk_args = {}
    fk = ssm.get_new_fk(**get_new_fk_args)
    pf = particles.SMC(fk=fk, N=N, resampling=resampling, ESSrmin=ESSrmin, store_history=True, verbose=verbose_pf)
    # pf.run()
    run_pf_with_memory_protection(pf, maxmem=memlimit)

    debug_tamper(pf)  # tamper the result of pf before running the smoother. Use for testing purpose.

    filtering_dists = [CategoricalDistribution(wgts.lw) for wgts in pf.hist.wgts]
    bw_samplers = backward_option.get_all_backward_kernels(ssm)

    return EasyFFBSResult(backward_smoothing(pf.hist.X, filtering_dists, pf.hist.A, bw_samplers), pf)

def tamper_pf_history(pf: particles.SMC):
    N = pf.N
    for i, arr in enumerate(pf.hist.A):
        if i == 0:
            assert arr is None
            continue
        assert arr.shape == (N, )
        pf.hist.A[i] = np.random.randint(low=0, high=N, size=N)

class EasyIntractableSmoother:
    # tested 090921
    def __init__(self, ssm: SmoothableSSM, N: int, get_intractable_fk_option: dict, history: bool, reorderer: Reorderer, verbose: bool = False, ncores=None, start_method=None):
        self.ssm = ssm
        self.N = N
        self.verbose = verbose
        self.fk = ssm.get_new_intractable_fk(**get_intractable_fk_option)
        self.pf = IntractableSMC(fk=self.fk, N=self.N, verbose=verbose, reorderer=reorderer, ncores=ncores, start_method=start_method)
        self.paris = IntractablePaRIS(history=history)
        self.add_func = self.ssm.typical_additive_function()
        self.t = 0

        self.paris.send(None)

    def __iter__(self):
        return self

    def __next__(self):
        t1 = time.process_time() if self.pf.ncores is None else time.time()
        assert int(self.pf.t) == self.t
        next(self.pf)
        val = iPaRIS_Input(X=self.pf.X, W=CategoricalDistribution(lw=self.pf.wgts.lw), psi=self.add_func.psi(t=self.t), backward_idx=self.pf.backward_idx)
        self.t += 1
        t2 = time.process_time() if self.pf.ncores is None else time.time()
        if self.verbose:
            print('Elapsed time: {} secs'.format(t2 - t1))
        return self.paris.send(val)