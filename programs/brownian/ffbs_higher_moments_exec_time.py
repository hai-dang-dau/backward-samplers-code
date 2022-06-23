import numpy as np
import typing as tp
from programs.brownian.model import LinearGaussianSmoothing
from scipy.linalg import sqrtm

Matrix = np.array

def highest_moment_from_cov(cov_pred_t: Matrix, cov_smooth_t: Matrix, covX: Matrix) -> float:
    """
    Calculate the highest order k such that E[T^k] is possibly finite, where T is the execution time of the FFBS-reject algorithm to generate backward index at time * t - 1 *.
    """
    sqrt_covX = sqrtm(covX)
    assert np.allclose(sqrt_covX, sqrt_covX.T)
    prec_pred = np.linalg.inv(cov_pred_t)
    prec_smth = np.linalg.inv(cov_smooth_t)
    smallest_eig = np.linalg.eigvalsh(sqrt_covX @ (prec_smth - prec_pred) @ sqrt_covX)[0]
    assert smallest_eig >= -1e-6
    return 1 + smallest_eig

def highest_moments_ffbs_exec_time(model: LinearGaussianSmoothing) -> tp.List[tp.Optional[float]]:
    # relatively tested (by playing around with examples and so on).
    # todo: need to retest after reimplementing highest_moment_from_cov (31/03/2022)
    """
    Returns a list of highest orders k such that E[tau^k]'s are possibly finite, where tau's are execution times of FFBS reject algorithm at different times t. The first element of the returned list is None.
    """
    res: tp.List[tp.Optional[float]] = [None]
    km = model.kalman
    return res + [highest_moment_from_cov(cov_pred_t=km.pred[t].cov, cov_smooth_t=km.smth[t].cov, covX=model.covX) for t in range(1, model.T + 1)]