import typing as tp
from functools import partial
import numpy as np
from scipy.linalg import sqrtm

from libs_new.coupling_gaussians import MultivariateGaussianViaK
from libs_new.smoothing_high_level import AdditiveFunction, cut_off
from libs_new.utils import do_sth_to_each_row_of, pickable_make_plot_function
from programs.lotka_volterra.model_new import stationary_point, plot_data, get_parameters, analyse_parameters, \
    two_d_stability
from programs.sde.model import SmoothableSDE, SDE

Matrix = np.ndarray
Vector = tp.Sequence[float]

def lkv_sde_b_and_sigma(tau: tp.Sequence[float], beta: tp.Sequence[float], Gamma: Matrix):
    tau, beta, Gamma = [np.array(r) for r in (tau, beta, Gamma)]
    return partial(_bf, tau=tau, beta=beta), partial(_sigmaf, Gamma=Gamma)

# noinspection PyUnusedLocal
def _bf(t, x, tau, beta):
    # tested
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    res_b = -beta * x
    res_b[0] *= -1
    tau_part = tau[1:] * x[0:-1] * x[1:]
    res_tau = np.zeros(shape=x.shape)
    for i in [0, 1]:
        temp = np.zeros(x.shape)
        temp[i: len(temp) - 1 + i] = tau_part
        res_tau += (-1) ** (i+1) * temp
    res = res_b + res_tau
    res[0] -= tau[0] * x[0] * (x[0])/2
    return res

# noinspection PyUnusedLocal
def _sigmaf(t, x, Gamma):
    # tested
    return do_sth_to_each_row_of(Gamma, x, '*')

def Gamma_creator(std1, std2, corr):
    # tested
    cov = corr * std1 * std2
    cm = np.array([[std1**2, cov], [cov, std2**2]])
    try:
        res = np.linalg.cholesky(cm)
    except np.linalg.LinAlgError:
        res = sqrtm(cm)
    assert np.allclose(res @ res.T, cm)
    return res

class TauBetaGamma(tp.NamedTuple):
    tau: Vector
    beta: Vector
    Gamma: tp.Optional[Matrix]

class OtherLKParams(tp.NamedTuple):
    init_mean: Vector
    init_K: Matrix
    log_obs_K: Matrix

def improved_get_parameters_2d(x_star: Vector, nat_death_rates: Vector, deterministic_slowdown: float, noise0_std: float = None, noise1_std: float = None, corr: float = None, verbose=True) -> TauBetaGamma:
    beta, tau = get_parameters(x=x_star, nat_death_rates=nat_death_rates, continuous_model=noise0_std)
    beta = beta * deterministic_slowdown
    tau = tau * deterministic_slowdown
    if noise0_std is not None:
        Gamma = Gamma_creator(std1=noise0_std, std2=noise1_std, corr=corr)
    else:
        Gamma = None
    if verbose:
        print(analyse_parameters(tau=tau, beta=beta, Gamma=Gamma))
        print(two_d_stability(beta=beta, tau=tau, Gamma=Gamma))
    return TauBetaGamma(tau=tau, beta=beta, Gamma=Gamma)

class LKVSDE(SDE):
    # tested
    def __init__(self, tau: Vector, beta: Vector, Gamma: Matrix, init_mean: Vector, init_K: Matrix, log_obs_K: Matrix, N_dcr: int):
        b, sigma = lkv_sde_b_and_sigma(tau=tau, beta=beta, Gamma=Gamma)
        # noinspection PyTypeChecker
        super().__init__(f=b, sigma_transition=sigma, N_dcr=N_dcr)
        self.dynamic_params = TauBetaGamma(tau=tau, beta=beta, Gamma=Gamma)
        self.other_params = OtherLKParams(init_mean=init_mean, init_K=init_K, log_obs_K=log_obs_K)
        self.stationary_point = stationary_point(tau=tau, beta=beta, continuous_model=True)

    def PX0(self):
        return MultivariateGaussianViaK(mu=self.other_params.init_mean, K=self.other_params.init_K)

    def PY(self, t, x):
        x = np.maximum(np.abs(x), 1)
        return MultivariateGaussianViaK(mu=np.log(x), K=self.other_params.log_obs_K)

    @pickable_make_plot_function
    def plot(self, x, y, plot_stationary=True, aes=None, labels=None, ax=None):
        plot_data(x=x, y=np.exp(y), x_star=self.stationary_point, obs_coors=[0, 1], aes=aes, labels=labels, ax=ax, show=False, plot_stationary=plot_stationary)

class LKVContSmoothableSDE(SmoothableSDE):
    def __init__(self, sde: LKVSDE, data: tp.Sequence[Vector] = None, T: int = None, t_cutoff: int = None):
        super().__init__(sde=sde, data=data, T=T, two_d=True)
        self.t_cutoff = t_cutoff

    def typical_additive_function(self) -> AdditiveFunction:
        # noinspection PyTypeChecker
        res = AdditiveFunction(psi_0=self.simple_psi, psi_t=self.simple_psi, output_numpy_dim=2)
        if self.t_cutoff is not None:
            res = cut_off(af=res, t_cutoff=self.t_cutoff)
        return res

    def simple_psi(self, *args):
        sde: LKVSDE = self.sde
        return np.array(args[-1]) - sde.stationary_point

    @property
    def additive_function_len(self) -> int:
        return 2 if self.t_cutoff is None else 4

    def filtering_stability_diag_function(self, t: int, xt) -> np.ndarray:
        return xt

def user_friendly_ssm_creator(x_star: Vector, nat_death_rates: Vector, deterministic_slowdown: float, dynamic_noise0_std: float, dynamic_noise1_std: float, dynamic_noise_corr: float, init_mean: Vector, init_noise0_std: float, init_noise1_std: float, init_noise_corr: float, log_obs_noise0_std: float, log_obs_noise1_std: float, log_obs_noise_corr: float, N_dcr: int, data, T: int, t_cutoff:int, verbose: bool):
    # eye-tested
    tau, beta, Gamma = improved_get_parameters_2d(x_star=x_star, nat_death_rates=nat_death_rates, deterministic_slowdown=deterministic_slowdown, noise0_std=dynamic_noise0_std, noise1_std=dynamic_noise1_std, corr=dynamic_noise_corr, verbose=verbose)
    K_init = Gamma_creator(std1=init_noise0_std, std2=init_noise1_std, corr=init_noise_corr)
    K_log_obs = Gamma_creator(std1=log_obs_noise0_std, std2=log_obs_noise1_std, corr=log_obs_noise_corr)
    sde = LKVSDE(tau=tau, beta=beta, Gamma=Gamma, init_mean=init_mean, init_K=K_init, log_obs_K=K_log_obs, N_dcr=N_dcr)
    if verbose:
        print(dict(init_mean=init_mean, init_noise0_std=init_noise0_std, init_noise1_std=init_noise1_std))
    return LKVContSmoothableSDE(sde=sde, data=data, T=T, t_cutoff=t_cutoff)