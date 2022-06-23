import particles
import typing as tp
import numpy as np
from libs_new.smoothing_high_level import SmoothableSSM, AdditiveFunction
from libs_new.utils import do_sth_to_each_column_of, sth_of_each_row
from libs_new.utils_math import CategoricalDistribution


# noinspection PyAbstractClass
class FKConstrainedBrownian(particles.FeynmanKac):
    # tested with version 050921
    def __init__(self, T: int, d: int, constraints: tp.Sequence[float]):
        """
        :param constraints: constants a0, a1, ..., a_{d-1} that defines the ellipse in which the Brownian motion is confined:
        sum x_i**2 / a_i**2 = 1
        """
        super().__init__(T+1)
        self.d = d
        self.constraints = np.array(constraints)
        assert len(constraints) == d

    def M0(self, N):
        return np.random.normal(size=(N, self.d))

    def M(self, t, xp):
        return np.random.normal(loc=xp)

    def logG(self, t, xp, x):
        N, d = x.shape
        dm = sth_of_each_row(do_sth_to_each_column_of(x**2, self.constraints ** 2, '/'), 'sum')
        assert dm.shape == (N, )
        return np.where(dm <= 1, 0, -np.inf)

class ConstrainedBrownian(SmoothableSSM):
    # tested version 050921
    def __init__(self, T: int, d: int, constraints: tp.Sequence[float]):
        self._T = T
        self.d = d
        self.constraints = constraints

    @property
    def T(self):
        return self._T

    def get_new_fk(self, *args, **kwargs) -> particles.FeynmanKac:
        return FKConstrainedBrownian(T=self.T, d=self.d, constraints=self.constraints)

    def logpt(self, t: int, xp, x) -> float:
        return -1/2 * sum((x-xp)**2)

    def upper_bound_logpt(self, t:int, x) -> float:
        return 0

    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        return np.zeros(self.d)

    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        return np.zeros(self.d)

    def typical_additive_function(self) -> AdditiveFunction:
        return AdditiveFunction(psi_0=self._simple_psi_0, psi_t=self._simple_psi_t)

    def _simple_psi_0(self, p):
        assert p.shape == (self.d, )
        return p

    # noinspection PyUnusedLocal
    def _simple_psi_t(self, t: int, xtm1, xt) -> float:
        assert xt.shape == (self.d, )
        return xt if t <= self.T/2 else np.zeros(self.d)

    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist: CategoricalDistribution, t: int) -> tp.Sequence[tp.Sequence[float]]:
        return super().presort_transform(x_tm1=x_tm1, filtering_dist=filtering_dist, t=t)

    def backward_gaussianizer(self, t: int):
        raise AssertionError

    @property
    def additive_function_len(self) -> int:
        return self.d