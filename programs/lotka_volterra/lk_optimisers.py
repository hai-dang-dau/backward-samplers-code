from scipy.optimize import linprog
import typing as tp
from libs_new.sinkhorn import KLFunc, RealSinkhorn
import numpy as np

from libs_new.utils import cached_function

print('THIS FILE SHOULD NO LONGER BE USED.')

_klfunc = KLFunc()

@cached_function
def klfunc_tuple(A):
    return _klfunc(np.array(A))

def convert_to_tuple(matrix):
    return tuple([tuple(row) for row in matrix])

def klfunc(A):
    return klfunc_tuple(convert_to_tuple(A.tolist()))

class OptimizeResult(tp.NamedTuple):
    x: tp.Sequence[float]
    status: int

# noinspection PyUnusedLocal
def scipy_linprog(c, A_eq, b_eq, x0, method: str = 'interior-point'):
    """
    Use scipy's optimiser. Not recommended, as numerical difficulties are almost always present and sometimes answers are returned without satisfying the constraints (and without warnings either).
    """
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, method=method)
    return res

# noinspection PyUnusedLocal
def dummy_linprog(c, A_eq, b_eq, x0):
    """
    Default optimiser. Doesn't do much, but can still work reasonably if observations are recorded at sufficiently frequent intervals.
    """
    return OptimizeResult(x=x0, status=0)

def sinkhorn_linprog(c, A_eq, b_eq, x0, eta: float = 10, niter: int = 10):
    """
    May work well, but required tuning of eta and niter
    """
    try:
        real_sinkhorn = RealSinkhorn(A=A_eq, b=b_eq, c=c, kl_func=klfunc)
        return OptimizeResult(x=real_sinkhorn.run(x=x0, eta=eta, niter=niter, verbose=False), status=0)
    except RuntimeError:
        return OptimizeResult(x=x0, status=1)


# noinspection PyUnusedLocal
def always_independent(c, A_eq, b_eq, x0):
    # noinspection PyTypeChecker
    return OptimizeResult(x=None, status=1)