from libs_new.unvectorised_ssm import UnvectorisedSSM, UnvectorisedBootstrap
from libs_new.utils_math import DistributionLike
from programs.lotka_volterra.model_new import FoodChain
import particles
import typing as tp
import numpy as np
import matplotlib.pyplot as plt

SysState = tp.Tuple[int, ...]

class Dirac:
    def __init__(self, loc: int):
        self.loc = int(loc)

    def rvs(self):
        return self.loc

    def logpdf(self, x: int):
        return -np.inf if x != self.loc else 0.0

class ConditionalFoodChain(UnvectorisedSSM):
    def __init__(self, food_chain: FoodChain, condition: tp.Callable[[SysState], bool]):
        self.food_chain = food_chain
        self.condition = condition

    def PX0(self):
        return self.food_chain.PX0()

    def PX(self, t: int, xp):
        return self.food_chain.PX(t=t, xp=xp)

    def PY(self, t: int, x: SysState) -> DistributionLike:
        return Dirac(int(self.condition(x)))

def one_trajectory(cfc: ConditionalFoodChain, T: int, N: int, verbose: bool) -> tp.List[SysState]:
    fk = UnvectorisedBootstrap(ssm=cfc, data=[1] * (T+1))
    pf = particles.SMC(fk=fk, N=N, ESSrmin=1.0, store_history=True, verbose=verbose)
    pf.run()
    return pf.hist.extract_one_trajectory()

def y_from_x(x: tp.List[SysState], food_chain: FoodChain) -> tp.List[SysState]:
    return [food_chain.PY(t=t, x=xt).rvs() for t, xt in enumerate(x)]

class ConditionalData(tp.NamedTuple):
    x: tp.List[SysState]
    y: tp.List[tp.List[int]]
    fig: plt.Figure

def conditional_data_creator(food_chain: FoodChain, T: int, N: int, cond: tp.Callable[[SysState], bool], verbose: bool) -> ConditionalData:
    # tested 251121
    cfc = ConditionalFoodChain(food_chain=food_chain, condition=cond)
    x = one_trajectory(cfc=cfc, T=T, N=N, verbose=verbose)
    y = y_from_x(x=x, food_chain=food_chain)
    fig = food_chain.plot(x=x, y=y, show=False)
    return ConditionalData(x=x, y=y, fig=fig)