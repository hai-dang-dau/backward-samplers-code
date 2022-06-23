"""
A canonical example to test SDE solvers. The equation is given by dXt = b(Xt)dt + sigma_sde(Xt) @ dWt. The true solution has a non-trivial correlation between x[0] and x[1].
"""
# Before interpreting any test result, please run a histogram comparison of two normal samples with similar size to the experiment. This gives an idea of what to expect in terms of histogram similarity.
# I am not sure that we need another example. Should we do however, please consider simple Ornstein-Uhlenbeck equation and a smooth transformation (that's the difficult, but necessary, part).

import numpy as np
import typing as tp
from scipy.stats import norm
from programs.sde.model import SDE

Vector = tp.Sequence[float]
Matrix = np.ndarray

def _aux(t, x):
    return np.cbrt(x[0] - x[1] - t**2)

aux = _aux

def _check(x: Vector):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert x.shape == (2,)
    return x

def b(t: float, x: Vector) -> Vector:
    x = _check(x)
    return np.array([3 * _aux(t, x), -2 * t])

def sigma_sde(t: float, x: Vector) -> Matrix:
    x = _check(x)
    return np.array([[3 * _aux(t, x)**2, 1], [0, 1]])

def sample_sol(t: float) -> Vector:
    wt = np.random.normal(scale=t**0.5, size=2)
    return np.array([wt[0]**3 + wt[1], wt[1] - t**2])

class DiracZeroDist:
    @staticmethod
    def rvs():
        return np.array([0, 0])

norm_obj = norm()

class ReadySDE(SDE):
    def PX0(self):
        return DiracZeroDist()

    def PY(self, t, x):
        return norm_obj