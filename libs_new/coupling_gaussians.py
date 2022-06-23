from functools import cached_property, partial

import numpy as np
import typing as tp
from scipy.linalg import solve, lu_factor, lu_solve

from libs_new.utils_math import DistributionLike, is_singular


def compound_cholesky(Sigma1, Sigma2):
    """
    Returns a lower triangular matrix L such that L @ Sigma1 @ L.T = Sigma2
    """
    K1 = np.linalg.cholesky(Sigma1)
    K2 = np.linalg.cholesky(Sigma2)
    return np.linalg.solve(K1.T, K2.T).T

def xtAy(x, A, y):
    return np.dot(x, A @ y)

def geometric_coupling_transform(x: tp.Sequence[float], mu1: tp.Sequence[float], mu2: tp.Sequence[float], Sigma1, Sigma2):
    """
    Transform a vector x ~ N(mu1, Sigma1) into another vector y ~ N(mu2, Sigma2) such that the pair forms a reflection-like coupling of the two distributions.
    """
    h = x - mu1
    h = compound_cholesky(Sigma1, Sigma2) @ h
    t = mu2 - mu1
    S2_inv_t = solve(Sigma2, t, assume_a='pos')
    h_star = h - 2 * np.dot(h, S2_inv_t)/np.dot(t, S2_inv_t) * t
    return mu2 + h_star

def Lindvall_Rogers_coupling_transform(dB, mu1, dist2: 'MultivariateGaussianViaK'):
    """
    Transform a vector x ~ mu1 + K1 @ N(0,Id) into another vector y ~ mu2 + K2 @ N(0,Id) such that the pair forms a reflection-like coupling of the two distributions.
    Let x = mu1 + K1 @ dB - only dB is actually used to make the transformation.
    We use the method described in Coupling of multidimensional diffusions by reflection (Lindvall and Rogers)
    """
    d = dist2.d
    y = mu1 - dist2.mu
    u = dist2.solve_Kx_equals_y(y)
    try:
        u /= np.sqrt(np.sum(u**2))
    except (ZeroDivisionError, FloatingPointError):
        return dist2.mu + dist2.K @ dB
    u = u.reshape((d, 1))
    H = np.identity(d) - 2 * (u @ u.T)
    dB_prime = H @ dB
    return dist2.mu + dist2.K @ dB_prime

def geometric_coupling(mu1, mu2, Sigma1, Sigma2):
    # tested
    """
    Returns a reflection-like coupling of N(mu1, Sigma1) and N(mu2, Sigma2)
    """
    # x1 = np.random.multivariate_normal(mean=mu1, cov=Sigma1)
    x1 = fast_multivariate_normal_rvs(mu=mu1, Sigma=Sigma1)
    x2 = geometric_coupling_transform(x=x1, mu1=mu1, mu2=mu2, Sigma1=Sigma1, Sigma2=Sigma2)
    return x1, x2

def Lindvall_Rogers_coupling(dist1: 'MultivariateGaussianViaK', dist2: 'MultivariateGaussianViaK'):
    # tested
    """
    Returns a reflection-like coupling of mu1 + K1 @ N(0, Id) and mu2 + K2 @ N(0, Id)
    """
    dB = np.random.normal(size=dist1.d)
    x1 = dist1.mu + dist1.K @ dB
    x2 = Lindvall_Rogers_coupling_transform(dB=dB, mu1=dist1.mu, dist2=dist2)
    return x1, x2

def fast_multivariate_normal_rvs(mu, Sigma):
    # tested
    d = Sigma.shape[0]
    assert mu.shape == (d, )
    K = np.linalg.cholesky(Sigma)
    return mu + K @ np.random.normal(size=d)

_t = tp.TypeVar('_t')

def _in_common_mass(x: _t, u: float, this: DistributionLike[_t], that: DistributionLike[_t]):
    """
    Verify whether (x,u) lies in the common mass of the graphs of two distributions, where (x,u) encodes the uniform distribution under the graph of ``this`` (u must be between 0 and 1).
    """
    assert 0 <= u <= 1
    return np.log(u) + this.logpdf(x) <= that.logpdf(x)

def _float_rand():
    return float(np.random.rand())

def hybrid_coupler(dist1: DistributionLike[_t], dist2: DistributionLike[_t], coupler: tp.Callable[[], tp.Tuple[_t, _t]]) -> tp.Tuple[_t, _t]:
    # tested with discrete distributions
    # tested again using lindvall_rogers_hybrid_coupling
    """
    Combine an existing coupler with the maximal coupling one. This may be useful when the existing coupler has good properties when the two distributions are far away, but has zero meeting probability (e.g. the reflection coupler given by Lindvall and Rogers 1986).
    """
    x0 = dist1.rvs()
    x0_in_shared = _in_common_mass(x0, _float_rand(), dist1, dist2)
    x1, x2 = coupler()
    if not x0_in_shared:
        return x1, x2
    else:
        u = _float_rand()
        x1_in_shared = _in_common_mass(x1, u, dist1, dist2)
        x2_in_shared = _in_common_mass(x2, u, dist2, dist1)
        return (x0 if x1_in_shared else x1), (x0 if x2_in_shared else x2)

class MultivariateGaussianViaK:
    # tested
    """
    The (unvectorised) multivariate Gaussian distribution class, specified by a matrix K such that K @ K.T = Sigma
    """
    def __init__(self, mu: tp.Sequence[float], K: np.ndarray):
        K = np.array(K, dtype=float)
        mu = np.array(mu)
        if (len(K.shape) != 2) or (K.shape[0] != K.shape[1]) or (mu.shape != (K.shape[0],)):
            raise ValueError
        self.K = K
        self.mu = mu
        self.d = len(mu)
        # self.lu_piv = lu_factor(self.K)
        # self.log_abs_det_K = float(np.sum(np.log(np.abs(np.diag(self.lu_piv[0])))))

    @cached_property
    def lu_piv(self):
        return lu_factor(self.K)

    @cached_property
    def log_abs_det_K(self):
        return float(np.sum(np.log(np.abs(np.diag(self.lu_piv[0])))))

    def logpdf(self, x) -> float:
        s = self.solve_Kx_equals_y(x - self.mu)
        return -self.d/2 * np.log(2 * np.pi) - self.log_abs_det_K - 0.5 * np.sum(s**2)

    def rvs(self):
        return self.mu + self.K @ np.random.normal(size=self.d)

    def solve_Kx_equals_y(self, y):
        return lu_solve(lu_and_piv=self.lu_piv, b=y)

def lindvall_rogers_hybrid_coupling(dist1: MultivariateGaussianViaK, dist2: MultivariateGaussianViaK, two_d=False):
    """
    :param two_d: quick and dirty optimisation for two-d matrices
    """
    if (dist1.d != 2 or dist2.d != 2) and two_d:
        raise ValueError
    np.seterr(invalid='raise', divide='raise')
    if is_singular(dist1.K, two_d=two_d) or is_singular(dist2.K, two_d=two_d):
        return dist1.rvs(), dist2.rvs()
    # tested
    return hybrid_coupler(dist1=dist1, dist2=dist2, coupler=partial(Lindvall_Rogers_coupling,dist1=dist1, dist2=dist2))