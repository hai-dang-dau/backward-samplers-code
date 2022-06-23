from abc import ABC, abstractmethod
from libs_new.utils_math import DistributionLike
import typing as tp
import particles
import numpy as np
from tqdm import tqdm

_Tp = tp.TypeVar('_Tp')  # particle type

"""
Warning: in this file, T is handled according to the convention in the particles package.
"""

class UnvectorisedSSM(ABC):
    # tested 120921
    @abstractmethod
    def PX0(self) -> DistributionLike:...

    @abstractmethod
    def PX(self, t: int, xp: _Tp) -> DistributionLike:...

    @abstractmethod
    def PY(self, t: int, x: _Tp) -> DistributionLike:...

    def simulate(self, T, debug=False):
        x = [self.PX0().rvs()]
        for t in tqdm(range(1, T), disable=not debug):
            x.append(self.PX(t=t, xp=x[-1]).rvs())
        y = [self.PY(t=tt, x=xx).rvs() for tt, xx in enumerate(x)]
        return x, y

# noinspection PyAbstractClass
class UnvectorisedBootstrap(particles.FeynmanKac):
    # tested120921
    def __init__(self, ssm: UnvectorisedSSM, data: tp.Sequence):
        self.ssm = ssm
        self.data = data
        super().__init__(T=len(data))

    def M0(self, N):
        x0_dist = self.ssm.PX0()
        return np.array([x0_dist.rvs() for _ in range(N)])

    def M(self, t, xp):
        return np.array([self.ssm.PX(t=t, xp=_xp).rvs() for _xp in xp])

    def logG(self, t, xp, x):
        return np.array([self.ssm.PY(t=t, x=_x).logpdf(self.data[t]) for _x in x])