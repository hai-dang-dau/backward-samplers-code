from functools import partial, cached_property

import numpy as np
import typing as tp
from tqdm import tqdm

from libs_new.utils import fast_allclose, cached_function
from libs_new.utils_math import newton_method, bisection_method

exp = np.exp

FAILED_CONVERGENCE = [0]

class KLFunc:
    def __call__(self, A):
        L = np.linalg.lstsq(A @ A.T, A, rcond=None)[0]
        assert np.allclose(np.linalg.inv(A @ A.T) @ A, L)
        K = -A.T @ L
        K += np.identity(int(K.shape[0]))
        return K, L

class Sinkhorn:
    # tested version 140921, then several times afterwards. Retested 021021 after adding the free_variable projection method
    # It definitely works, but there is much room for improvements.
    """
    Solve the following linear optimisation problem
    min <c, x>
    s.t. A @ x = b and x >= 0
    using a Sinkhorn-like algorithm.
    The initial guess x0 must be a strictly feasible solution, i.e. x > 0.
    """
    def __init__(self, A: np.ndarray, b: tp.Sequence[float], c: tp.Sequence[float], kl_func: KLFunc = KLFunc(), reproject_algo: tp.Literal['orthogonal', 'free_variable'] = 'orthogonal'):
        """
        :param A: numpy matrix of shape (n_constraints, n_variables)
        :param kl_func: function to calculate the K(A) and L(A) matrices. See the class KLFunc for more details
        """
        self.A = A
        self.b = b
        self.c = c
        self.K, self.L = kl_func(A)
        self.reproject_algo = reproject_algo

    def lam_from_x(self, x: tp.Sequence[float], eta: float) -> tp.Sequence[float]:
        y = self.c - 1/eta * self.grad_H(x)
        return self.L @ y

    def x_from_lam(self, lam: tp.Sequence[float], eta: float) -> tp.Sequence[float]:
        y = eta * (self.c - self.A.T @ lam)
        return self.inverse_grad_H(y)

    def reproject(self, x: tp.Sequence[float], x_old: tp.Sequence[float]) -> tp.Sequence[float]:
        return getattr(self, 'reproject_' + self.reproject_algo)(x=x, x_old=x_old)

    def reproject_orthogonal(self, x: tp.Sequence[float], x_old: tp.Sequence[float]) -> tp.Sequence[float]:
        x_proj = self.K @ x + self.L.T @ self.b
        return self.positive_convex_combination(x_proj, x_old)

    # noinspection PyUnusedLocal
    def reproject_free_variable(self, x: tp.Sequence[float], x_old: tp.Sequence[float]) -> tp.Sequence[float]:
        # noinspection PyTypeChecker
        return free_variable_projection(x=x, A=self.A, b=self.b)

    @staticmethod
    def grad_H(x: tp.Sequence[float]):
        # tested
        return -np.log(x) - 1

    @staticmethod
    def inverse_grad_H(y: tp.Sequence[float]):
        # tested
        return np.exp(-1 - y)

    def run_once(self, x: tp.Sequence[float], eta: float) -> tp.Sequence[float]:
        lam = self.lam_from_x(x, eta)
        x_prime = self.x_from_lam(lam, eta)
        x_new = self.reproject(x_prime, x)
        return x_new

    def initialise(self, x_initial, eta):
        return x_initial

    def finalise(self, x_initial, x):
        return x

    def run(self, x: tp.Sequence[float], eta: float, niter: int, verbose: bool):
        x_initial = x
        x = self.initialise(x_initial=x_initial, eta=eta)
        for _ in tqdm(range(niter)) if verbose else range(niter):
            x = self.run_once(x, eta)
        return self.finalise(x_initial=x_initial, x=x)

    def run_with_history(self, x: tp.Sequence[float], eta: float, niter: int, verbose: bool):
        res = [x, self.initialise(x_initial=x, eta=eta)]
        for _ in tqdm(range(niter)) if verbose else range(niter):
            res.append(self.run_once(res[-1], eta))
        res.append(self.finalise(x_initial=res[0], x=res[-1]))
        return res

    @staticmethod
    def positive_convex_combination(x: tp.Sequence[float], x_old: tp.Sequence[float]) -> tp.Sequence[float]:
        # tested
        """
        :param x: any vector
        :param x_old: a vector with nonnegative elements
        :return: a vector which is a convex combination of `x` and `x_old` and has nonnegative elements
        """
        if np.all(x >= 0):
            return x
        else:
            nid = np.where(x < 0)
            # noinspection PyTypeChecker
            cond = x[nid] / ((x-x_old)[nid])
            alp = max(cond)
            alp = 0.999 * alp + 0.001 * 1
            FAILED_CONVERGENCE[0] += 1
            # print('Reprojected!!!')
            # print('Reprojected. Old {} new {} final{}'.format(x_old, x, alp * x_old + (1-alp) * x))
            return alp * x_old + (1 - alp) * x

class ModifiedInteriorPoint(Sinkhorn):
    @staticmethod
    def grad_H(x: tp.Sequence[float]):
        return 1/x

    @staticmethod
    def inverse_grad_H(y: tp.Sequence[float]):
        return 1/y

def minus(f, g):
    return partial(_minus, f=f, g=g)

def _minus(x, f, g):
    return f(x) - g(x)

class RealSinkhorn(Sinkhorn):
    """
    RealSinkhorn is only applicable if the matrix A has elements of either 0 or 1.
    """
    # tested using 140921 test.
    @cached_property
    def roweq1(self) -> tp.Sequence[tp.Sequence[bool]]:
        res = []
        for row in self.A:
            res.append(np.where(row == 1)[0])
        return res

    def initialise(self, x_initial, eta):
        return exp(-1 - eta * self.c).tolist()

    def finalise(self, x_initial, x):
        res = self.reproject(x=np.array(x), x_old=x_initial)
        # if not (np.allclose(self.A @ res, self.b) and np.all(res >= 0)):
        #     raise RuntimeError('Optimisation failed')
        # else:
        #     return res
        return res

    def run_once(self, x: tp.Sequence[float], eta: float) -> tp.Sequence[float]:
        for i, bi in enumerate(self.b):
            x = self.exp_project(x=x, row_no=i, bi=bi)
        return x

    @classmethod
    def exp_project_old(cls, x: tp.Sequence[float], row: tp.Sequence[float], bi: float) -> tp.Sequence[float]:
        # tested
        # if this is unstable in applications, then Newton and bisection method must be combined
        """
        Solve the equation <row, x * v> = bi where v is a vector of the form exp(row * k) for some real number k. The function returns x * v.

        Due to the exponential function, special techniques are needed for numerical stability. x must be a nonnegative vector.

        Good benchmark:
        >>> from libs_new.sinkhorn import RealSinkhorn
        >>> import numpy as np
        >>> RealSinkhorn.exp_project(x=np.array([4, 2, 1]), row=np.array([-5, 1.5, 3.8]), bi=8080)
        >>> RealSinkhorn.exp_project(x=np.array([0, 1, 2]), row=np.array([3, -1, 1]), bi=8080)
        """
        positive_idx = (row >= 0); negative_idx = ~positive_idx
        row_pos = row[positive_idx]; row_neg = row[negative_idx]
        x_pos = x[positive_idx]; x_neg = - x[negative_idx]  # yep, minus here and not above.
        bi_pos = 1 + abs(bi); bi_neg = 1 + abs(bi) + bi

        hlist = [cls._f, cls._fprime, cls._ftwoprime]
        tar_pos, dtar_pos, d2tar_pos = [partial(h, r=row_pos, x=x_pos, b=bi_pos) for h in hlist]
        tar_neg, dtar_neg, d2tar_neg = [partial(h, r=row_neg, x=x_neg, b=bi_neg) for h in hlist]

        # noinspection PyTypeChecker
        k = newton_method(f=minus(tar_pos, tar_neg), fprime=minus(dtar_pos, dtar_neg), ftwoprime=minus(d2tar_pos, d2tar_neg), niter=5, verbose=False)
        new_x = x * np.exp(row * k)
        # print('Ax = {}, b = {}'.format(np.dot(row, new_x), bi))
        return new_x

    @classmethod
    def exp_project_old2(cls, x: tp.Sequence[float], row: tp.Sequence[float], bi: float) -> tp.Sequence[float]:
        # tested
        # noinspection PyTypeChecker
        k = bisection_method(f=partial(cls._simple_f, r=row, x=x, b=bi), niter=10, verbose=False)
        new_x = x * np.exp(row * float(k))
        # print('Ax = {}, b = {}'.format(np.dot(row, new_x), bi))
        return new_x

    def exp_project(self, x: tp.Sequence[float], row_no: int, bi: float) -> tp.Sequence[float]:
        # tested
        # assert set(row) == {0, 1}
        new_x = x.copy()
        temp = self.roweq1[row_no]
        mul = bi/sum([x[t] for t in temp])
        for t in temp:
            new_x[t] = new_x[t] * mul
        # new_x[temp] *= bi/sum(x[temp])
        # assert np.allclose(np.dot(new_x, row), bi)
        return new_x

    @staticmethod
    def _simple_f(k, r, x, b):
        return np.sum(r * x * exp(r * k)) - b

    @staticmethod
    def _f(k, r, x, b):
        return np.log(np.sum(r * x * exp(r * k)) + b)

    @staticmethod
    def _fprime(k, r, x, b):
        num = np.sum(r**2 * x * exp(r * k))
        den = np.sum(r * x * exp(r * k)) + b
        return num/den

    @staticmethod
    def _ftwoprime(k, r, x, b):
        one = np.sum(r * x * exp(r * k))
        two = np.sum(r**2 * x * exp(r * k))
        three = np.sum(r**3 * x * exp(r * k))
        return (three * (one + b) - two**2) / (two + b)**2

class FreeVariableProjection:
    # tested
    def find_positive_solution(self, A, b):
        """
        Find a positive solution to the system A @ x = b.
        Only applicable to matrices A appearing in the coupling of Markov processes
        """
        constraints, free_variables, ncols = self.analyse_matrix(A)
        res = [0] * ncols
        for v, c in zip(free_variables, constraints):
            res[v] = b[c]
        res = np.array(res)
        if not fast_allclose(A @ res, b):
            raise ValueError('The matrix A is not compatible')
        else:
            return res

    def analyse_matrix(self, A):
        A = tuple([tuple(row) for row in A.tolist()])
        return self._analyse_matrix(A)

    @cached_function
    def _analyse_matrix(self, A):
        return self.__analyse_matrix__(np.array(A))

    @staticmethod
    def __analyse_matrix__(A):
        nrows, ncols = A.shape
        free_variables = []
        constraints = []
        for v in range(ncols):
            col_pos = np.where(A[:, v] == 1)[0]
            if len(col_pos) == 1:
                free_variables.append(v)
                constraints.append(col_pos[0])
        return constraints, free_variables, ncols

    def __call__(self, x, A, b):
        """
        For compatible matrices A, find a positive solution of A @ x = b that is close to the given x.
        """
        b_prime = A @ x
        lbd = min(b / (b_prime + 1e-12))
        z = self.find_positive_solution(A, b - lbd * b_prime)
        return lbd * x + z

free_variable_projection = FreeVariableProjection()