import typing as tp
from abc import ABC, abstractmethod
from functools import partial

from libs_new.coupling_markov_processes import Optimiser, OptimisedCoupling, wrap_and_check, KISS_coupled_neighbour_map
from libs_new.coupling_optimisers import optimiser_list
from libs_new.intractable_smoother import FromFKObject, IntractableFK
from libs_new.smoothing_high_level import SmoothableSSM
from libs_new.unvectorised_ssm import UnvectorisedSSM, UnvectorisedBootstrap
from libs_new.utils import zip_with_assert
from libs_new.utils_math import DistributionLike, MarkovJumpProcess, MixtureKernel

SysState = tp.Tuple[int, ...]

def Hamming_distance(x: SysState, y: SysState) -> int:
    L = len(x)
    if L == 2:
        return _Hamming_distance_2D(x, y)
    if L == 3:
        return _Hamming_distance_3D(x, y)
    if L == 4:
        return _Hamming_distance_4D(x, y)
    return sum([abs(ex - ey) for ex, ey in zip(x, y)])

def _Hamming_distance_2D(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def _Hamming_distance_4D(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) + abs(x[3] - y[3])

def _Hamming_distance_3D(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])

concavify = {k: v for k, v in zip_with_assert([-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                              [-174.0, -110.0, -61.0, -25.0, 0.0, 16.0, 25.0, 29.0, 30.0])}

def concavified_Hamming_distance_increase(state1: SysState, state2: SysState, state1_prime: SysState, state2_prime: SysState) -> int:
    # tested
    """
    The ``prime`` means that state1_prime is either a neighbour of state1 or state1 itself. This function determines the optimisation criterion in choosing the coupling.
    """
    return concavify[Hamming_distance(state1_prime, state2_prime) - Hamming_distance(state1, state2)]

class product_dist:
    def __init__(self, *idists):
        self.idists = idists

    def rvs(self):
        return [d.rvs() for d in self.idists]

    def logpdf(self, x):
        return sum([d.logpdf(xi) for d, xi in zip(self.idists, x)])

class LatticeSSM(UnvectorisedSSM, ABC):
    @property
    @abstractmethod
    def dynamic(self) -> tp.Callable[[SysState], tp.Mapping[SysState, float]]:
        ...

    def PX(self, t: int, xp: SysState) -> DistributionLike:
        return one_step(dynamic=self.dynamic, starting_point=tuple(xp))

class one_step:
    def __init__(self, dynamic, starting_point):
        self.dynamic = dynamic
        self.starting_point = starting_point
        self.call = 0

    def rvs(self):
        if self.call > 0:
            raise AssertionError
        else:
            res = MarkovJumpProcess(self.dynamic, self.starting_point).get_xt(1)
            self.call += 1
            return res

    def logpdf(self, x):
        raise NotImplementedError

class SmoothableLatticeSSM(SmoothableSSM, ABC):
    def __init__(self, debug: bool):
        self.debug = debug

    @property
    @abstractmethod
    def ssm(self) -> LatticeSSM:
        ...

    @property
    @abstractmethod
    def data(self):
        ...

    def get_new_fk(self, *args, **kwargs):
        return UnvectorisedBootstrap(ssm=self.ssm, data=self.data)

    def coupled_M_mixin(self, self2: FromFKObject, t: int, xp1: SysState, xp2: SysState, optimiser: Optimiser) -> tp.Tuple[SysState, SysState, bool]:
        assert t > 0
        assert isinstance(self2, FromFKObject)
        dynamic = OptimisedCoupling(single_neighbour_function=self.ssm.dynamic, loss_function=concavified_Hamming_distance_increase, optimiser=optimiser, loss_vector_signature=self.loss_vector_signature)
        coupled, x1, x2 = self._return_couple(dynamic, xp1, xp2)
        return x1, x2, coupled

    def _return_couple(self, dynamic, xp1, xp2):
        if self.debug:
            dynamic = wrap_and_check(dynamic, self.ssm.dynamic)
        xp1, xp2 = to_pure_python(xp1), to_pure_python(xp2)
        x1, x2 = MarkovJumpProcess(dynamic, (xp1, xp2)).get_xt(1)
        x1, x2 = to_pure_python(x1), to_pure_python(x2)
        coupled = bool(x1 == x2)
        return coupled, x1, x2

    def KISS_coupled_M_mixin(self, self2: FromFKObject, t: int, xp1: SysState, xp2: SysState) -> tp.Tuple[SysState, SysState, bool]:
        # tested 141121
        assert t > 0
        assert isinstance(self2, FromFKObject)
        dynamic = partial(KISS_coupled_neighbour_map, single_neighbour_function=self.ssm.dynamic)
        coupled, x1, x2 = self._return_couple(dynamic, xp1, xp2)
        return x1, x2, coupled

    def get_new_intractable_fk(self, optimiser_name: str, optimiser_args: dict = None, optimiser_proportion: float = 1.0) -> IntractableFK:
        # tested that finder_coupling_args have some effect
        fk = self.get_new_fk()
        optimiser = optimiser_list[optimiser_name](**optimiser_args)
        optimised_coupled_M_mixin = partial(self.coupled_M_mixin, optimiser=optimiser)
        KISS_coupled_M_mixin = self.KISS_coupled_M_mixin
        final_M_mixin = MixtureKernel(kernels=[optimised_coupled_M_mixin, KISS_coupled_M_mixin], weights=[optimiser_proportion, 1 - optimiser_proportion])
        # noinspection PyTypeChecker
        return IntractableFK.fromFKObject(fk=fk, coupled_M=final_M_mixin)

    @property
    def T(self):
        return len(self.data) - 1

    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]: raise TypeError

    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]: raise TypeError

    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist, t: int) -> tp.Sequence[tp.Sequence[float]]: raise TypeError

    def logpt(self, t: int, xp, x) -> float: raise TypeError

    def upper_bound_logpt(self, t:int, x) -> float: raise TypeError

    def backward_gaussianizer(self, t: int): raise TypeError

    def filtering_stability_diag_function(self, t: int, xt):
        return xt

    @abstractmethod
    def loss_vector_signature(self, state1, state2):
        ...

def to_pure_python(x):
    return tuple([int(k) for k in x])

def increment(x: SysState, coor: int, val: int) -> SysState:
    return *x[0:coor], x[coor] + val, *x[coor+1:]