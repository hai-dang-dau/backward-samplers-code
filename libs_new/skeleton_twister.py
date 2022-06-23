"""
To help make `new_smoother.py` more compact.
This file therefore must *not* import anything from new_smoother.
"""
from abc import ABC, abstractmethod
import typing as tp
from particles import resampling as rs
import numpy as np
from scipy.stats import multivariate_normal

Matrix = tp.Sequence[tp.Sequence[float]]

class SkeletonConverter(ABC):
    """
    Create a skeleton version of a weighted sample that will lower the backward sampling time using rejection-based algorithms.
    """
    @abstractmethod
    def __call__(self, X: Matrix, lw: tp.Sequence[float]) -> tp.Tuple[Matrix, tp.Sequence[float]]:
        ...

class IdentityConverter(SkeletonConverter):
    def __call__(self, X: Matrix, lw: tp.Sequence[float]) -> tp.Tuple[Matrix, tp.Sequence[float]]:
        return X, lw

_Tp = tp.TypeVar('_Tp')

def skeleton_resample(x: tp.Sequence[_Tp], lw: tp.Sequence[float], resample_lw: tp.Sequence[float]) -> tp.Tuple[tp.Sequence[_Tp], tp.Sequence[float]]:
    # tested
    """
    Resample weighted particles according to external resampling weights
    :param x: array of particles
    :param lw: log-weights of particles
    :param resample_lw: resampling log-weights
    :return: new particles and new log-weights
    """
    resample_W = rs.exp_and_normalise(resample_lw)
    resampled_idx = rs.systematic(resample_W)
    new_lw = lw[resampled_idx] - resample_lw[resampled_idx]
    # noinspection PyTypeChecker
    return x[resampled_idx], new_lw

class GaussianConstrictConverter(SkeletonConverter):
    def __call__(self, X: Matrix, lw: tp.Sequence[float]) -> tp.Tuple[Matrix, tp.Sequence[float]]:
        # self-tested
        mu = np.mean(X, axis=0)
        Sigma = np.cov(X, rowvar=False)
        dist = multivariate_normal(mean=mu, cov=Sigma)
        resampling_lw = dist.logpdf(X) / 2
        return skeleton_resample(X, lw, resampling_lw)

class SimpleResampleConverter(SkeletonConverter):
    # tested
    def __call__(self, X: Matrix, lw: tp.Sequence[float]) -> tp.Tuple[Matrix, tp.Sequence[float]]:
        return skeleton_resample(x=X, lw=lw, resample_lw=lw)