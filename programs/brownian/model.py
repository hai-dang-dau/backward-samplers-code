import numpy as np
from abc import ABC, abstractmethod
from particles import kalman
import typing as tp
from scipy.stats import ortho_group, t as student, norm

from libs_new.intractable_smoother import IntractableFK, FromFKObject
from libs_new.utils_math import MultivariateNormalDistribution, random_mean_and_cov, integrate_normal_ratio, \
    integrate_varianced_normal_ratio, CategoricalDistribution, CoupledNormalDist
import particles
from particles import state_space_models as ssm
from libs_new.smoothing_high_level import SmoothableSSM, AdditiveFunction, BackwardDistGaussianizer
from libs_new.mcmc_on_mesh import ForwardGaussianDynamicsBDG
from functools import partial, cached_property
from scipy.stats import multivariate_normal

DEBUG = []

np_array_or_float = tp.Union[np.ndarray, float]

class LinearGaussianSmoothing(SmoothableSSM, ABC):
    """
    Wrapper that groups a number of convenient objects and methods to perform smoothing on Linear Gaussian models.

    :param covX: covariance of the noise added to X_{t+1} conditionally on X_t
    :param covY: covariance of the noise added to Y_t conditionally on X_t
    :param data: data for the model. If `None`, will be generated automatically
    :param T: if `data` is `None`, the final time T for the model
    :param F: transforms X_t to the mean of X_{t+1}. Defaults to the identity matrix
    :param G: transforms X_t to the mean of Y_t. Defaults to the generalized identity matrix
    :param mu0: mean of the distribution of X_0. Defaults to 0
    :param cov0: covariance of the distribution of X_0. Defaults to covX
    :param transform_before_sorting: whether to transform points X_{t-1} before sorting them by the Hilbert curve
    :param t_cutoff: cutoff level for offline smoothing benchmark (which calculates E[X_0 + X_1 + ... + X_{t_cutoff}|Y_{0:T}]
    """
    def __init__(self, covX: np_array_or_float, covY: np_array_or_float, data: tp.Sequence[np.ndarray] = None, T: int = None, F: np_array_or_float = None, G: np_array_or_float = None, mu0: np.ndarray = None, cov0: np.ndarray = None, transform_before_sorting: bool = False, t_cutoff : int = 0):
        self.ssm = kalman.MVLinearGauss(F=F, G=G, covX=covX, covY=covY, mu0=mu0, cov0=cov0)
        self.covX = self.ssm.covX; self.covY = self.ssm.covY; self.cov0 = self.ssm.cov0
        self.F = self.ssm.F; self.G = self.ssm.G; self.mu0 = self.ssm.mu0
        if data is None:
            data = self.ssm.simulate(T + 1)[1]
        self.data = data
        self.kalman = kalman.Kalman(ssm=self.ssm, data=self.data)
        self.kalman.smoother()
        self.dx = self.covX.shape[0]
        self.transition_dist = MultivariateNormalDistribution(mean=np.zeros(self.dx), cov=self.covX)
        self.transform_before_sorting = transform_before_sorting
        self.t_cutoff = t_cutoff

    def get_new_fk(self, fk_type: tp.Literal['Bootstrap', 'GuidedPF', 'AuxiliaryPF'] = 'AuxiliaryPF') -> particles.FeynmanKac:
        """
        Get a *fresh new* bootstrap FeynmanKac model for the problem. This function can safely be used multiple time on the same object.
        """
        return getattr(ssm, fk_type)(ssm=self.ssm, data=self.data)

    def exact_filtering_dist(self, t: int) -> kalman.MeanAndCov:
        return self.kalman.filt[t]

    def exact_smoothing_dist(self, t: int) -> kalman.MeanAndCov:
        return self.kalman.smth[t]

    def _logpt(self, xp, x) -> float:
        """Non-vectorized function for the log transition density"""
        return self.transition_dist.logpdf(x - self.F @ xp)

    # noinspection PyUnusedLocal
    def _upper_bound_logpt(self, x) -> float:
        """Non-vectorized function for the log upper bound of logpt"""
        return self.transition_dist.logpdf(np.zeros(self.dx))

    def logpt(self, t: int, xp, x) -> float:
        return self._logpt(xp, x)

    def upper_bound_logpt(self, t:int, x) -> float:
        return self._upper_bound_logpt(x)

    def vlogpt(self, xp: np.ndarray, x) -> np.ndarray:
        # tested
        """
        vectorized log transition density
        """
        return self.transition_dist.logpdf(x - xp @ self.F.T)

    def simulate_ffbs_oracle_exec_time(self, t:int, size: int, paris: bool = False) -> np.ndarray:
        # tested
        """
        Simulate execution time for FFBS-reject algorithm at step `t` (i.e. to generate particles at time `t`).
        """
        t = t + 1
        # Now: we know x_t and we want to generate x_{t-1}
        # q_{t-1}(x_{t-1}) ~ N(mu, Sigma)
        filtering_dist_tm1 = self.exact_filtering_dist(t-1)
        assert len(filtering_dist_tm1.mean) == 1
        mu = filtering_dist_tm1.mean[0]
        Sigma = filtering_dist_tm1.cov
        q_tm1_xt_dist = multivariate_normal(mean=self.F @ mu, cov=self.F @ Sigma @ self.F.T + self.covX)
        if not paris:
            Xt = multivariate_normal.rvs(mean=self.exact_smoothing_dist(t).mean[0], cov=self.exact_smoothing_dist(t).cov, size=size)
        else:
            Xt = q_tm1_xt_dist.rvs(size=size)
        acc_rates = np.exp(q_tm1_xt_dist.logpdf(Xt) - self._upper_bound_logpt(...))
        return np.random.geometric(p=np.where(acc_rates > 0, acc_rates, 1e-12))

    def mean_and_var_ffbs_exec_time(self, t:int) -> tp.Tuple[float, float]:
        # tested
        """
        Returns the (oracle) mean FFBS-reject execution time to simulate x_t
        """
        _ = self.exact_filtering_dist(t)
        mu_t, Sigma_t = _.mean[0], _.cov
        mu_t_pred, Sigma_t_pred = self.F @ mu_t, self.F @ Sigma_t @ self.F.T + self.covX
        _ = self.exact_smoothing_dist(t + 1)
        mu_tp1, Sigma_tp1 = _.mean[0], _.cov
        C = np.exp(self._upper_bound_logpt(...))
        mean = C * integrate_normal_ratio(mu_tp1, Sigma_tp1, mu_t_pred, Sigma_t_pred)
        var = 2 * C**2 * integrate_varianced_normal_ratio(mu_tp1, Sigma_tp1, mu_t_pred, Sigma_t_pred) - mean - mean**2
        return mean, var

    def typical_additive_function(self) -> AdditiveFunction:
        # tested using version 020921
        # noinspection PyTypeChecker
        return AdditiveFunction(psi_0=self.simple_psi_0, psi_t=self.simple_psi_t)

    @abstractmethod
    def simple_psi_0(self, x0):
        return self.simple_psi_t(t=0, xtm1=None, xt=x0)

    @abstractmethod
    def simple_psi_t(self, t, xtm1, xt):
        dx = len(self.covX)
        xt_list = np.atleast_1d(xt).tolist()
        if t <= self.t_cutoff:
            xt_list2 = xt_list
        else:
            xt_list2 = [0] * dx
        # noinspection PyTypeChecker
        res = np.array(xt_list + xt_list2)
        assert res.shape == (2 * dx,)
        return res

    @property
    @abstractmethod
    def additive_function_len(self) -> int:
        return 2 * len(self.covX)

    def reduced_model(self, t) -> 'LinearGaussianSmoothing':
        return DefaultLinearGaussianSmoothing(covX=self.covX, covY=self.covY, data=self.data[0:t+1], F=self.F, G=self.G, mu0=self.mu0, cov0=self.cov0)

    @abstractmethod
    def _A(self, t, mu, Sigma) -> tp.Union[float, tp.Sequence[float]]:
        """
        E[psi_t(x) | x ~ N(mu, Sigma)]
        """
        dx = len(self.covX)
        mu = np.reshape(mu, (dx,)).tolist()
        mu2 = mu if t <= self.t_cutoff else [0] * dx
        # noinspection PyTypeChecker
        res = np.array(mu + mu2)
        assert res.shape == (2 * dx,)
        return res

    def exact_offline_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        res = 0
        for s in range(0, t+1):
            smoothing_s = self.exact_smoothing_dist(s)
            res = res + self._A(s, smoothing_s.mean, smoothing_s.cov)
        return res

    def exact_online_expectation(self, t: int) -> tp.Union[float, tp.Sequence[float]]:
        res = 0
        reduced_model_t = self.reduced_model(t)
        for s in range(0, t+1):
            smoothing_s = reduced_model_t.exact_smoothing_dist(s)
            res = res + self._A(s, smoothing_s.mean, smoothing_s.cov)
        return res

    @property
    def T(self):
        return len(self.data) - 1

    def backward_gaussianizer(self, t: int) -> BackwardDistGaussianizer:
        assert t > 0
        return ForwardGaussianDynamicsBDG(self.F, self.covX)

    def presort_transform(self, x_tm1: tp.Sequence[tp.Sequence[float]], filtering_dist: CategoricalDistribution, t: int) -> tp.Sequence[tp.Sequence[float]]:
        # tested, version 16:45 16/08/2021
        if (not self.transform_before_sorting) or len(x_tm1[0]) == 1:
            return super().presort_transform(x_tm1, filtering_dist, t)
        try:
            weighted_Lambda = np.linalg.inv(np.cov(x_tm1, rowvar=False, aweights=filtering_dist.W))
            unweighted_Lambda = np.linalg.inv(np.cov(x_tm1, rowvar=False))
            transformer_Lambda = weighted_Lambda - unweighted_Lambda + self._fcf
            K = np.linalg.cholesky(transformer_Lambda).T
            # assert np.allclose(K.T @ K, transformer_Lambda)
            # DEBUG
            # DEBUG.append(K)
            # END DEBUG
            return x_tm1 @ K.T
        except np.linalg.LinAlgError:
            print('LinAlgError encountered. Switching to simple transform...')
            return super().presort_transform(x_tm1, filtering_dist, t)

    @cached_property
    def _fcf(self):
        return self.F.T @ np.linalg.inv(self.covX) @ self.F

    def filtering_stability_diag_function(self, t: int, xt) -> np.ndarray:
        return xt

class DefaultLinearGaussianSmoothing(LinearGaussianSmoothing):
    def simple_psi_0(self, x0):
        return super().simple_psi_0(x0)

    def simple_psi_t(self, t, xtm1, xt):
        return super().simple_psi_t(t, xtm1, xt)

    def _A(self, t, mu, Sigma) -> tp.Union[float, tp.Sequence[float]]:
        return super()._A(t, mu, Sigma)

    @property
    def additive_function_len(self) -> int:
        return super().additive_function_len

    def get_new_intractable_fk(self, *args, **kwargs):
        raise TypeError('No intractable fk here')

class RotatedBrownian(DefaultLinearGaussianSmoothing):
    def __init__(self, dx: int, dy: int, T: int, **kwargs):
        rmean_x, rcov_x = random_mean_and_cov(d=dx, range_std=(0.5, 1))
        rmean_y, rcov_y = random_mean_and_cov(d=dy, range_std=(0.5, 1))
        F = ortho_group.rvs(dx)
        super().__init__(covX=rcov_x, covY=rcov_y, T=T, F=F, **kwargs)

class HeavyQueueBrownian(SmoothableSSM):
    # tested
    """
    (X_t) evolves as a Brownian motion, except that the distribution of X0 is heavy-queued (product of independent t distributions of degree of freedom `df`). (Y_t) are noisy projections of X_t on some dimensions. Only the bootstrap filter is implemented to match the desired behavior for T = 1. In general cases however, guided filters are safe to use (prove it!).
    """
    def __init__(self, dx:int, dy:int, T:int, df: float):
        self._kalman = LinearGaussianSmoothing(covX=np.identity(dx), covY=np.identity(dy), T=T)
        self._df = df

    def get_new_fk(self) -> particles.FeynmanKac:
        kalman_fk = self._kalman.get_new_fk('Bootstrap')
        assert 'M0' not in kalman_fk.__dict__
        kalman_fk.M0 = partial(self._M0, df=self._df, dx=self._kalman.dx)
        return kalman_fk

    def logpt(self, t: int, xp, x) -> float:
        return self._kalman.logpt(t, xp, x)

    def upper_bound_logpt(self, t:int, x) -> float:
        return self._kalman.upper_bound_logpt(t, x)

    # noinspection PyPep8Naming
    @staticmethod
    def _M0(N:int, df: float, dx:int):
        """
        Generate `N` random variates on dimension `dx`, each of which is the product of independent Student distribution of `df` degree of freedom.
        """
        if np.isinf(df):
            generator = norm
        else:
            generator = student(df=df)
        return generator.rvs(size=N*dx).reshape((N, dx))

class Guarniero(DefaultLinearGaussianSmoothing):
    def __init__(self, d: int, T: tp.Optional[int], alpha: float = 0.4, sigma_y: float = 1, data=None):
        F = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                F[i, j] = alpha ** (1 + abs(i - j))
        super().__init__(covX=np.eye(d), covY=sigma_y * np.eye(d), T=T, F=F, data=data)

class GaussianTrack(DefaultLinearGaussianSmoothing):
    def __init__(self, dx: int, dy: int, T: int, rho: float, **kwargs):
        super().__init__(
            covX=(1-rho**2) * np.identity(dx),
            covY=(1-rho**2) * np.identity(dy),
            T=T,
            F=rho * np.identity(dx),
            cov0=np.identity(dx),
            **kwargs
                         )

def BasicOneDim_init(self, T: int, rho=0.9, sigma_x: float = 1, sigma_y=0.2):
    return LinearGaussianSmoothing.__init__(self, covX=np.array([[sigma_x**2]]), covY=np.array([[sigma_y**2]]), T=T, F=np.array([[rho]]), G=np.array([[1]]), mu0=np.array([0]), cov0=np.array([[sigma_x**2/(1 - rho**2)]]))

class BasicOneDim(DefaultLinearGaussianSmoothing):
    # noinspection PyMissingConstructor
    def __init__(self, T: int, rho=0.9, sigma_x=1, sigma_y=0.2):
        BasicOneDim_init(self, T, rho, sigma_x, sigma_y)

class IntractableBasicOneDim(LinearGaussianSmoothing):
    # tested version 090921
    # noinspection PyMissingConstructor
    def __init__(self, T: int, rho: float, sigma_x: float, sigma_y: float):
        BasicOneDim_init(self=self, T=T, rho=rho, sigma_x=sigma_x, sigma_y=sigma_y)

    def simple_psi_0(self, x0):
        return float(x0)

    def simple_psi_t(self, t, xtm1, xt):
        return float(xt)

    @property
    def additive_function_len(self) -> int:
        return 0

    def _A(self, t, mu, Sigma):
        return float(mu)

    def get_new_intractable_fk(self, deflate_coupling_ratio: float = 1) -> IntractableFK:
        fk = self.get_new_fk('Bootstrap')
        # noinspection PyTypeChecker
        return FromFKObject(fk=fk, coupled_M=partial(self.coupled_M_mixer, deflate_coupling_ratio=deflate_coupling_ratio))

    # noinspection PyUnusedLocal
    def coupled_M_mixer(self, fromfk: FromFKObject, t: int, xp1, xp2, deflate_coupling_ratio: float = 1) -> tp.Tuple[tp.Any, tp.Any, bool]:
        # tested version 041021
        assert xp1.shape == xp2.shape == (1, )
        F = float(self.F); sigma = float(self.covX**0.5)
        coupled_dist = CoupledNormalDist(mu1=F * float(xp1), mu2=F * float(xp2), sigma=sigma)
        if np.random.rand() < deflate_coupling_ratio:
            x1, x2, success = coupled_dist.rvs()
        else:
            x1 = np.random.normal(loc=F * float(xp1), scale=sigma)
            x2 = np.random.normal(loc=F * float(xp2), scale=sigma)
            success = False
        return [x1], [x2], success

def _L_matrix(d: int, rho: float):
    x = np.identity(d)
    for i in range(1, d):
        x[i, i-1] = -1 + rho
    return x

def _S_matrix(d: int, rho: float):
    return np.linalg.inv(_L_matrix(d=d, rho=rho))

class HierarchicalLike(DefaultLinearGaussianSmoothing):
    """
    The first coordinates of X (X_{0:T}^0) is a stationary Markov chain wrt N(0,1) and mixing parameter rho, i.e. X_t^0 ~ N(rho * X_{t-1}^0, 1 - rho**2).
    For the i-th coordinates of X (X_{0:T}^{i-1}), the transition from time t-1 to time t is that of a stationary Markov chain wrt N(X_t^{i-2}, 1) and mixing parameter rho.
    """
    def __init__(self, d, rho, sigma, T, t_cutoff):
        S = _S_matrix(d=d, rho=rho)
        cX = (1-rho**2) * S @ S.T
        F = rho * S
        G = np.array([0.0] * (d-1) + [1]).reshape((1, d))
        super().__init__(covX=cX, covY=[[sigma**2]], T=T, F=F, mu0=np.zeros(d), cov0=np.identity(d), t_cutoff=t_cutoff, G=G)