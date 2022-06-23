from libs_new.coupling_markov_processes import OptimizeResult, Optimiser
from scipy.optimize import linprog
from libs_new.sinkhorn import RealSinkhorn, KLFunc
import numpy as np
import typing as tp
from libs_new.utils import cached_function

# Tested version 011021

optimiser_list = {}

def register(opt):
    optimiser_list[opt.__name__] = opt
    return opt

@register
class ScipyLinOpt(Optimiser):
    def __init__(self, method: str = 'revised simplex', cache_threshold: float = None, linprog_options: dict = None):
        self.method = method
        self.linprog_options = linprog_options
        super().__init__(cache_threshold=cache_threshold)

    def _call(self, A, b, c, x0) -> OptimizeResult:
        c = np.array([_ for _ in c])
        return linprog(c=c, A_eq=A.materialise, b_eq=b, method=self.method, options=self.linprog_options)

kl_func = KLFunc()

@register
class SinkhornOpt(Optimiser):
    def __init__(self, eta: float = 20, niter: int = 10, reproject_algo: tp.Literal['free_variable'] = 'free_variable', cache_threshold: float = None):
        self.eta = eta
        self.niter = niter
        self.reproject_algo = reproject_algo
        super().__init__(cache_threshold=cache_threshold)

    def kl_func(self, A: np.ndarray):
        # noinspection PyTypeChecker
        A_tuplised = tuple([tuple(row) for row in A.tolist()])
        return self.cached_kl_func(A_tuplised)

    @cached_function
    def cached_kl_func(self, A):
        return kl_func(np.array(A))

    def _call(self, A, b, c, x0) -> OptimizeResult:
        c = np.array([_ for _ in c])
        res = RealSinkhorn(A=A.materialise, b=b, c=c, kl_func=self.kl_func, reproject_algo=self.reproject_algo).run(x=x0, eta=self.eta, niter=self.niter, verbose=False)
        return OptimizeResult(x=res, status=0)

@register
class AlwaysIndepOpt(Optimiser):
    def __init__(self):
        super().__init__(cache_threshold=None)

    def _call(self, A, b, c, x0) -> OptimizeResult:
        return OptimizeResult(x=x0, status=0)