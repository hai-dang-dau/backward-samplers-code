from abc import ABC
from functools import partial
import particles
from libs_new.coupling_gaussians import MultivariateGaussianViaK, lindvall_rogers_hybrid_coupling
from libs_new.intractable_smoother import IntractableFK, FromFKObject
from libs_new.smoothing_high_level import SmoothableSSM, AdditiveFunction
from libs_new.utils import AutomaticList
from libs_new.utils_math import DistributionLike
import typing as tp
import numpy as np
# noinspection PyProtectedMember
from libs_new.unvectorised_ssm import UnvectorisedSSM, _Tp, UnvectorisedBootstrap

Vector = tp.Sequence[float]
Matrix = np.ndarray

class SDE(UnvectorisedSSM, ABC):
    # fully tested
    def __init__(self, f: tp.Callable[[float, Vector], Vector], sigma_transition: tp.Callable[[float, Vector], Matrix], N_dcr: int):
        """
        :param N_dcr: number of discretisations
        """
        self.f = f
        self.sigma_transition = sigma_transition
        self.N_dcr = N_dcr
        self.epsilon_time = 1/self.N_dcr

    def PX(self, t: int, xp: _Tp) -> DistributionLike:
        return _one_step(sde=self, t=t, xp=xp)

class _one_step:
    def __init__(self, sde: SDE, t: int, xp):
        self.sde = sde
        self.t = t
        self.xp = xp
        self.used = False
        self.d = len(xp)

    def rvs(self):
        if self.used:
            raise AssertionError
        self.used = True
        x = self.xp
        t = self.t - 1
        eps = self.sde.epsilon_time
        for _ in range(self.sde.N_dcr):
            x = x + eps * self.sde.f(t, x) + (eps ** 0.5) * self.sde.sigma_transition(t, x) @ np.random.normal(size=self.d)
            t = t + eps
        assert abs(t - self.t) < 1e-8
        return x

    def logpdf(self, x):
        raise AssertionError

class SpannedSDE(UnvectorisedSSM):
    # fully tested
    def __init__(self, sde: SDE):
        self.sde = sde

    def PX0(self) -> DistributionLike:
        return self.sde.PX0()

    def PX(self, t: int, xp: _Tp) -> DistributionLike:
        real_t = (t - 1)/self.sde.N_dcr
        eps = self.sde.epsilon_time
        return MultivariateGaussianViaK(mu=xp + eps * self.sde.f(real_t, xp), K=(eps**0.5) * self.sde.sigma_transition(real_t, xp))

    def PY(self, t: int, x: _Tp) -> DistributionLike:
        if t % self.sde.N_dcr == 0:
            return self.sde.PY(t//self.sde.N_dcr, x)
        else:
            return DiracFilledData

class FillerClass:
    def __str__(self):
        return 'Filler'

    def __repr__(self):
        return self.__str__()

Filler = FillerClass()

class _DiracFilledData:
    @staticmethod
    def rvs():
        return Filler

    @staticmethod
    def logpdf(x):
        if x is not Filler:
            raise AssertionError
        else:
            return 0.0

DiracFilledData = _DiracFilledData()

Dtype = tp.TypeVar('Dtype')

def fill_data(orig_data: tp.List[Dtype], N_dist: int) -> tp.List[tp.Union[Dtype, FillerClass]]:
    # tested, including edge cases
    res = AutomaticList(Filler)
    for i, datum in enumerate(orig_data):
        res[i * N_dist] = datum
    return res.underlying_list

def _slow_down_f(t, x, f, alpha):
    return alpha * f(alpha * t, x)

def _slow_down_sigma(t, x, sigma, alpha, debug=False):
    return alpha ** (0.5 if not debug else 1) * sigma(alpha * t, x)

type_f = tp.Callable[[float, Vector], Vector]
type_sigma = tp.Callable[[float, Vector], Matrix]

def slowed_f_sigma(f: type_f, sigma: type_sigma, alpha: float, debug=False) -> tp.Tuple[type_f, type_sigma]:
    # tested
    """
    Given an equation dXt = f(t,Xt)dt + sigma(t, Xt)dWt, returns new functions f_tilde and sigma_tilde such that dYt = f_tilde(t, Yt)dt + sigma_tilde(t,Yt)dBt, where Y_t = X_{alpha * t}.
    """
    # noinspection PyTypeChecker
    return partial(_slow_down_f, f=f, alpha=alpha), partial(_slow_down_sigma, sigma=sigma, alpha=alpha, debug=debug)

class SmoothableSDE(SmoothableSSM):
    # tested
    def __init__(self, sde: SDE, data: tp.Sequence[Vector] = None, T: int = None, two_d: bool = False):
        self.sde = sde
        if data is None:
            hidden_states, data = self.sde.simulate(T=T+1)
        else:
            hidden_states = None
        self.hidden_states, self.data = hidden_states, data
        self.is_two_d_model = two_d

    def get_new_fk(self) -> particles.FeynmanKac:
        return UnvectorisedBootstrap(ssm=self.sde, data=self.data)

    def coupled_M(self, self2: FromFKObject, t: int, x1: Vector, x2: Vector) -> tp.Tuple[Vector, Vector, bool]:
        # tested
        assert isinstance(self2, FromFKObject)
        final_t = t
        t = t - 1  # current t
        eps = self.sde.epsilon_time
        f = self.sde.f
        sigma = self.sde.sigma_transition
        is2d = self.is_two_d_model
        for _ in range(self.sde.N_dcr):
            x1, x2 = self.coupled_M_single_step(eps, f, is2d, sigma, t, x1, x2)
            t = t + eps
        assert abs(t - final_t) < 1e-10
        return x1, x2, x1 is x2

    @staticmethod
    def coupled_M_single_step(eps, f, is2d, sigma, t, x1, x2, func=lindvall_rogers_hybrid_coupling):
        # noinspection PyTypeChecker
        dist1 = MultivariateGaussianViaK(mu=x1 + eps * f(t, x1), K=sigma(t, x1) * np.sqrt(eps))
        # noinspection PyTypeChecker
        dist2 = MultivariateGaussianViaK(mu=x2 + eps * f(t, x2), K=sigma(t, x2) * np.sqrt(eps))
        if x1 is not x2:
            x1, x2 = func(dist1, dist2, two_d=is2d)
        else:
            x1 = dist1.rvs()
            x1, x2 = x1, x1
        return x1, x2

    def get_new_intractable_fk(self, *args, **kwargs) -> IntractableFK:
        normal_fk = self.get_new_fk()
        return FromFKObject(fk=normal_fk, coupled_M=self.coupled_M)

    @property
    def T(self):
        return len(self.data) - 1

    def logpt(self, t: int, xp, x) -> float: raise TypeError
    def upper_bound_logpt(self, t:int, x) -> float: raise TypeError
    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:raise TypeError
    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:raise TypeError
    def backward_gaussianizer(self, t: int): raise TypeError
    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist, t: int) -> tp.Sequence[tp.Sequence[float]]: raise TypeError

    def typical_additive_function(self) -> AdditiveFunction:
        raise TypeError

    @property
    def additive_function_len(self) -> int:
        raise TypeError

    def filtering_stability_diag_function(self, t: int, xt) -> np.ndarray:
        return xt