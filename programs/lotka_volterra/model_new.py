from functools import partial, cached_property
import matplotlib.pyplot as plt
from libs_new.smoothing_high_level import AdditiveFunction
# noinspection PyProtectedMember
from libs_new.unvectorised_ssm import _Tp
from libs_new.utils import sureint, pickable_make_plot_function, zip_with_assert
from libs_new.utils_math import DistributionLike
from programs.lotka_volterra.integer_lattice_problems import SysState, increment, LatticeSSM, product_dist, \
    SmoothableLatticeSSM
import typing as tp
import numpy as np
import pandas as pd
from scipy.stats import norm

def LKV_transition(x: SysState, tau: tp.Sequence[float], beta: tp.Sequence[float]):
    # tested
    # warning: whether elements of x are of type int or int64 might have tangible effect on performance.
    # First, treat small cases for the sake of speed
    L = len(x)
    if L == 2:
        return _LKV_transition_2D(x=x, tau=tau, beta=beta)
    if L == 3:
        return _LKV_transition_3D(x=x, tau=tau, beta=beta)
    if L == 4:
        return _LKV_transition_4D(x=x, tau=tau, beta=beta)
    return {
        increment(x, 0, 1): beta[0] * x[0],
        **{
            increment(x, i, -1): beta[i] * x[i]
            for i in range(1, len(x))
        },
        increment(x, 0, -1): tau[0] * x[0] * (x[0] - 1)/2,
        **{
            increment(increment(x, i-1, -1), i, +1): tau[i] * x[i-1] * x[i]
            for i in range(1, len(x))
        }
    }

def _LKV_transition_2D(x, tau, beta):
    # tested
    return {(x[0] + 1, x[1]): beta[0] * x[0], (x[0], x[1] - 1): beta[1] * x[1], (x[0] - 1, x[1]): tau[0] * x[0] * (x[0] - 1)/2, (x[0]-1, x[1] + 1): tau[1] * x[0] * x[1]}

def _LKV_transition_4D(x, tau, beta):
    # tested
    return {(x[0]+1, x[1], x[2], x[3]): beta[0]*x[0], (x[0], x[1]-1, x[2], x[3]): beta[1]*x[1], (x[0], x[1], x[2]-1, x[3]): beta[2]*x[2], (x[0], x[1], x[2], x[3]-1): beta[3]*x[3], (x[0]-1, x[1], x[2], x[3]): tau[0]*x[0]*(x[0]-1)/2, (x[0]-1, x[1]+1, x[2], x[3]): tau[1] * x[0] * x[1], (x[0], x[1]-1, x[2]+1, x[3]): tau[2]*x[1]*x[2], (x[0], x[1], x[2]-1, x[3]+1): tau[3]*x[2]*x[3]}

def _LKV_transition_3D(x, tau, beta):
    # tested
    return {(x[0] + 1, x[1], x[2]): beta[0] * x[0], (x[0], x[1] - 1, x[2]): beta[1] * x[1], (x[0], x[1], x[2] - 1): beta[2] * x[2], (x[0] - 1, x[1], x[2]): tau[0] * x[0] * (x[0]-1)/2, (x[0]-1, x[1] + 1, x[2]): tau[1] * x[0] * x[1], (x[0], x[1]-1, x[2]+1): tau[2] * x[1] * x[2]}

def stationary_point(tau: tp.Sequence[float], beta: tp.Sequence[float], continuous_model=False) -> tp.Sequence[float]:
    # tested. Also testable in use.
    # tested for continuous case, for this as well as derived functions.
    # noinspection PyTypeChecker
    A, v = get_A_v(beta, tau, continuous_model=continuous_model)
    return np.linalg.solve(A, v)


def get_A_v(beta, tau, continuous_model=False):
    """
    Returns A, v such that the deterministic differential equation can be written as
    x'(t) = x * (A @ x - v)
    """
    d = len(tau)
    A = np.zeros((d, d))
    A[0, 0] = -tau[0] / 2
    for i in range(1, d):
        A[i, i - 1] = tau[i]
        A[i - 1, i] = -tau[i]
    v = np.zeros(d)
    v[1:] = beta[1:]
    v[0] = -beta[0] - (tau[0] / 2 if not continuous_model else 0)
    return A, v

def analyse_parameters(tau: tp.Sequence[float], beta: tp.Sequence[float], Gamma: np.ndarray = None) -> pd.DataFrame:
    continuous_model = Gamma is not None
    stationary = stationary_point(tau=tau, beta=beta, continuous_model=continuous_model)
    adjusted_stationary0 = stationary[0] if continuous_model else stationary[0] - 1
    nat_death = [tau[0] * stationary[0] * adjusted_stationary0/2] + [beta[i] * stationary[i] for i in range(1, len(beta))]
    prey_death = [tau[i+1] * stationary[i] * stationary[i+1] for i in range(0, len(beta)-1)] + [0]
    born = [beta[0] * stationary[0]] + [tau[i] * stationary[i-1] * stationary[i] for i in range(1, len(beta))]
    res = dict(stationary=stationary, nat_death=nat_death, prey_death=prey_death, born=born)
    if continuous_model:
        res['nat_fluc'] = np.array(stationary) * np.sqrt(np.diag(Gamma @ Gamma.T))
    return pd.DataFrame(res)

class BetaTau(tp.NamedTuple):
    beta: tp.Sequence[float]
    tau: tp.Sequence[float]

def get_parameters(x: tp.Sequence[float], nat_death_rates: tp.Sequence[float], continuous_model=False) -> BetaTau:
    # tested
    """
    Find the beta and tau parameters that provide a given equilibrium population `x` and given natural death rates for spieces 0, 1, ..., d-2. Note that the natural death rate for spiece d-1 is 100%.
    """
    adjusted_x0 = x[0] if continuous_model else (x[0] - 1)
    d = len(x)
    A = []
    add_equation = partial(_add_equation, A=A, length=2 * d)
    # Detailed balance conditions
    add_equation([d, 0, 1], [1, -0.5 * adjusted_x0, -x[1]])
    for i in range(1, d):
        add_equation([i, i+1, d+i], [x[i-1], (-x[i+1] if i < d-1 else 0), -1])
    # Natural death rate constraints
    r = nat_death_rates
    rp = [1-e for e in r]
    add_equation([0, 1], [0.5 * adjusted_x0/r[0], -x[1]/rp[0]])
    for i in range(1, d-1):
        add_equation([d+i, i+1], [1/r[i], -x[i+1]/rp[i]])
    # Normalisation condition
    add_equation([-1], [1])
    v = np.linalg.solve(np.array(A), np.array([0] * (2 * d - 1) + [1]))
    return BetaTau(v[d:], v[0:d])

def _add_equation(pos, coef, A, length):
    row = np.zeros(length)
    row[pos] = coef
    A.append(row)

class StabilityInfo(tp.NamedTuple):
    rho0: float
    rho1: float
    alive0: bool
    alive1: bool

def two_d_stability(beta, tau, Gamma=None) -> StabilityInfo:
    """
    Calculate the stability of the system in dimension two case. Based on Hening & Nguyen (2017).
    The two diagnostic statistics rho0 and rho1 are defined as:
    rho0 = beta0 - G @ G.T [0,0] / 2
    rho1 = 2 * tau1 * rho0 - tau0 * (beta1 + G @ G.T [1,1]/2)
    If rho0 <= 0, both animals are extinct.
    If rho0 > 0, the prey is persistent and the predator's persistence depends on the strict positivity of rho1.
    """
    if Gamma is None:
        Gamma = np.zeros((2, 2))
        beta, tau = continuous_equivalent(beta, tau)
    GGT = Gamma @ Gamma.T
    rho0 = beta[0] - GGT[0,0]/2
    rho1 = 2 * tau[1] * rho0 - tau[0] * (beta[1] + GGT[1,1]/2)
    return StabilityInfo(rho0=rho0, rho1=rho1, alive0=rho0 > 0, alive1=(rho0 > 0) and (rho1 > 0))

def continuous_equivalent(beta, tau):
    # tested
    return [beta[0] + tau[0]/2, *beta[1:]], tau

class DiscreteIntUniform:
    # tested
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def rvs(self):
        return np.random.randint(low=self.lo, high=self.hi)

    def logpdf(self, x):
        sureint(x)
        return -np.log(self.hi - self.lo) if self.lo <= x < self.hi else -np.inf

class NormalDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def logpdf(self, x):
        return norm.logpdf(x, loc=self.loc, scale=self.scale)

    def rvs(self):
        return np.random.normal(loc=self.loc, scale=self.scale)

class FoodChain(LatticeSSM):
    # tested
    def __init__(self, tau: tp.Sequence[float], beta: tp.Sequence[float], x0_low: tp.Sequence[float], x0_high: tp.Sequence[float], obs_coordinates: tp.Sequence[int], obs_log_sd: tp.Sequence[float]):
        self.tau = tau
        self.beta = beta
        self._dynamic = partial(LKV_transition, tau=tau, beta=beta)
        self.x0_low = x0_low
        self.x0_high = x0_high
        self.obs_coordinates = obs_coordinates
        self.obs_log_sd = obs_log_sd

    @property
    def dynamic(self) -> tp.Callable[[SysState], tp.Mapping[SysState, float]]:
        return self._dynamic

    def PX0(self) -> DistributionLike:
        idists = [DiscreteIntUniform(lo, hi) for lo, hi in zip(self.x0_low, self.x0_high)]
        return product_dist(*idists)

    def PY(self, t: int, x: _Tp) -> DistributionLike:
        # tested
        idists = [NormalDistribution(loc=np.log(max(1, x[c])), scale=sd) for c, sd in zip(self.obs_coordinates, self.obs_log_sd)]
        return product_dist(*idists)

    @cached_property
    def stationary_point(self):
        return stationary_point(tau=self.tau, beta=self.beta)

    def plot_one_sample(self, T: int):
        x, y = self.simulate(T)
        self.plot(x, y)

    @pickable_make_plot_function
    def plot(self, x, y, ax):
        y = np.exp(y)
        plot_data(x=x, y=y, x_star=self.stationary_point, obs_coors=self.obs_coordinates, ax=ax, show=False)

@pickable_make_plot_function
def plot_data(x, y, x_star, obs_coors: tp.Sequence[int], plot_stationary=True, aes: tp.Sequence[dict] = None, labels: tp.Sequence[str] = None, ax=None):
    # tested
    """
    :param x: series of Lotka-Volterra hidden data
    :param y: series of observations
    :param x_star: stationary point
    """
    T = len(x)
    assert len(y) == T
    if aes is None:
        aes = [dict() for _ in x_star]
    if labels is None:
        labels = [str(i) for i, _ in enumerate(x_star)]
    for c, ae, lab in zip_with_assert(range(len(x_star)), aes, labels):
        line, = ax.plot(np.arange(0, T), [_[c] for _ in x], label=lab, **ae)
        line: plt.Line2D
        color = line.get_color()
        if plot_stationary:
            ax.axhline(x_star[c], color=color, linestyle='dashed')
        if c in obs_coors:
            idx = obs_coors.index(c)
            ax.plot(np.arange(0, T), [_[idx] for _ in y], color=color, linestyle='dotted', label='observed {}'.format(lab))
    ax.legend()

class SmoothableFoodChain(SmoothableLatticeSSM):
    def __init__(self, tau, beta, x0_low, x0_high, obs_coordinates, obs_log_sd, T: int = None, data: tp.Sequence[tp.Sequence[float]] = None, debug: bool = False, obs_log_sd_simulate=None):
        """
        :param obs_log_sd_simulate: simulate data using other parameters than those of the model
        """
        self._ssm = FoodChain(tau=tau, beta=beta, x0_low=x0_low, x0_high=x0_high, obs_coordinates=obs_coordinates, obs_log_sd=obs_log_sd)
        if obs_log_sd_simulate is not None:
            ssm_simulate = FoodChain(tau=tau, beta=beta, x0_low=x0_low, x0_high=x0_high, obs_coordinates=obs_coordinates, obs_log_sd=obs_log_sd_simulate)
        else:
            ssm_simulate = self._ssm
        if data is None:
            self.hidden_states, self._data = ssm_simulate.simulate(T+1, debug=debug)
            if 0 in self.hidden_states[-1]:
                print('Warning: Extinction occured or nearly occured.')
        else:
            self.hidden_states = []
            self._data = data
        super().__init__(debug=debug)

    @property
    def ssm(self) -> FoodChain:
        return self._ssm

    @property
    def data(self):
        return self._data

    def typical_additive_function(self) -> AdditiveFunction:
        return AdditiveFunction(psi_0=self.af_component0, psi_t=self.af_component1)

    @property
    def additive_function_len(self) -> int:
        return len(self.ssm.x0_low)

    def af_component(self, xt):
        if self.debug:
            assert isinstance(xt, np.ndarray)
            [sureint(e) for e in xt]
        return np.log((xt + 1)/self._ssm.stationary_point)

    def af_component0(self, x0):
        return self.af_component(x0)

    # noinspection PyUnusedLocal
    def af_component1(self, t, xtm1, xt):
        return self.af_component(xt)

    def loss_vector_signature(self, state1, state2):
        diff = [x - y for x, y in zip(state1, state2)]
        return np.maximum(np.minimum(diff, 2), -2)