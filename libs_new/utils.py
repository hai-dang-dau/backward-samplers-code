import math
import time
import warnings
from types import FunctionType
import particles
import os
import typing as tp
import json
import psutil
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial, wraps
import random
import string
from datetime import datetime

from hashlib import sha256
from tqdm import tqdm
from numba import NumbaWarning
from particles.hilbert import invlogit, hilbert_array
from scipy.linalg import LinAlgWarning
from scipy.optimize import OptimizeWarning
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import multiprocessing as mp
from itertools import zip_longest, chain, repeat, tee
import operator
from collections.abc import Iterable

DEBUG = []

class hook:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        inp = dict(args=args, kwargs=kwargs)
        oup = self.f(*args, **kwargs)
        DEBUG.append(dict(f=self.f, inp=inp, oup=oup))
        return oup

def make_plot_function(f: tp.Callable):
    """
    Decorator to wrap a plotting function.

    :param f: a function that takes as a keyword argument `ax` and draws on it.
    :return: the wrapped function
    """
    @wraps(f)
    def _f(*args, ax: plt.Axes = None, show: bool = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        f(*args, **kwargs, ax=ax)
        if show:
            fig.show()
        return fig
    return _f

class pickable_make_plot_function:
    # tested
    """
    A more complicated version of `make_plot_function` to ensure pickability.
    """
    def __init__(self, f: tp.Callable):
        self.f = f

    def __call__(self, *args, ax: plt.Axes = None, show: bool = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        self.f(*args, **kwargs, ax=ax)
        if show:
            fig.show()
        return fig

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return partial(self, instance)

@make_plot_function
def plot_histogram(data: tp.Iterable, exact_density: tp.Callable[[float], float] = None, extend: float = 0.2, title: str = None, ax: plt.Axes = None):
    """
    Plot the histogram of some data versus its known exact density if available. Can be useful to debug simulations.
    """
    ax.hist(data, bins='auto', density=True)

    xmin, xmax = min(data), max(data)
    delta = xmax - xmin
    xmin, xmax = xmin - extend*delta, xmax + extend*delta

    if exact_density is not None:
        xplot = np.linspace(xmin, xmax, 100)
        ax.plot(xplot, [exact_density(p) for p in xplot])

    if title is not None:
        ax.set_title(title)

def plot_histogram_vs_gaussian(data: tp.Iterable, extend: float = 0.2, title: str = None, ax: plt.Axes = None, show: bool = True):
    # tested
    loc = np.median(data)
    scale = np.quantile(data, norm.cdf(1)) - loc
    dist = norm(loc=loc, scale=scale)
    return plot_histogram(data=data, exact_density=dist.pdf, extend=extend, title=title, ax=ax, show=show)

def multiply_each_row_with(x, y):
    # tested
    """
    Multiply the i-th row of a 2D numpy array `x` with the i-th element of `y`

    :param x: numpy array of shape (m,n)
    :param y: numpy array of shape (m,)
    """
    return (x.T * y).T

def multiply_each_column_with(x, y):
    # tested
    """
    Multiply the j-th column of a 2D numpy array `x` with the j-th element of `y`

    :param x: numpy array of shape (m,n)
    :param y: numpy array of shape (n,)
    """
    return x * y

operator_map = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}

def do_sth_to_each_row_of(x: np.ndarray, y: np.ndarray, op: tp.Literal['+', '-', '*', '/']) -> np.ndarray:
    # tested
    """
    Do sth to row i of x with y[i].
    """
    return operator_map[op](x.T, y).T

def do_sth_to_each_column_of(x: np.ndarray, y:np.ndarray, op: tp.Literal['+', '-', '*', '/']) -> np.ndarray:
    # tested
    """
    Do sth to column i of x with y[i]
    """
    return operator_map[op](x, y)


def sum_of_each_row(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=1)

def sum_of_each_column(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=0)

def sth_of_each_row(x: np.ndarray, operation: str) -> np.ndarray:
    """
    Do a numpy operation on each row of `x`, then return the corresponding array. For example, sth_of_each_row(x, 'sum') is equivalent to sum_of_each_row(x)
    """
    return getattr(np, operation)(x, axis=1)

def sth_of_each_column(x: np.ndarray, operation: str) -> np.ndarray:
    """
    Do a numpy operation on each column of `x`, then return the corresponding array. For example, sth_of_each_column(x, 'sum') is equivalent to sum_of_each_column(x)
    """
    return getattr(np, operation)(x, axis=0)

class temporary_numpy_seed:
    """
    Context handler to temporary set numpy seed to some value, then revert back to original setting once done.
    """
    # tested
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # noinspection PyTypeChecker
        np.random.set_state(self.old_state)

def describe(x):
    """
    Describe a numpy array (print out its quantiles and its mean value).
    """
    return pd.Series(x).describe()

@make_plot_function
def compare_densities(data1: tp.Iterable[float], data2: tp.Iterable[float], ax: plt.Axes = None):
    # tested
    """
    Draw a plot to compare the densities of two datasets.

    :param data1, data2: datasets
    :return: nothing. A plot is displayed
    """
    sns.kdeplot(x=data1, ax=ax, label='1')
    sns.kdeplot(x=data2, ax=ax, label='2')
    ax.figure.legend()

def add_dummy_first_argument(f: tp.Callable):
    """
    Create a new function that takes all arguments of a given function plus a dummy first argument.

    :param f: any callable
    :return: a new callable ``g`` such that `` g(t, x) = f(x) ``
    """
    return partial(_dummy_first_argument, f)

# noinspection PyUnusedLocal
def _dummy_first_argument(f: tp.Callable, t, *args, **kwargs):
    return f(*args, **kwargs)

def random_string():
    """
    Create a random string to be used as file name
    """
    random.seed()
    res = ''
    for _ in range(6):
        res += random.choice(string.ascii_letters)
    res += '_' + str(datetime.now())
    for illegal_char in [':', '\\', '/']:
        res = res.replace(illegal_char, '-')
    return res

@make_plot_function
def plot_histogram_discrete(x: tp.Iterable, exact_pmf: tp.Callable[[int], float] = None, ax: plt.Axes = None):
    # tested
    """
    Plot the histogram of a discrete sample and compare it with the exact probability mass function if available.

    :param x: the sample
    :param exact_pmf: exact probability mass function
    :return: a figure
    """
    min_plot = min(x) - 1
    max_plot = max(x) + 2
    empirical_pmf = discrete_histogram(x)[1]

    fig = ax.figure
    ax.scatter(range(min_plot, max_plot), empirical_pmf, label='empirical')
    if exact_pmf is not None:
        dx = range(min_plot, max_plot)
        dy = [exact_pmf(v) for v in dx]
        ax.scatter(dx, dy, label='theoretical')
    fig.legend()

def discrete_histogram(x: tp.Iterable[int]):
    """
    Create the discrete histogram of `x`, suitable for plotting.

    :param x: iterable of integers
    :return: two arrays u and v such that v[i] is the empirical pmf of u[i]
    """
    min_plot = min(x) - 1
    max_plot = max(x) + 2
    empirical_pmf = np.zeros(max_plot - min_plot)
    counts = Counter(x)
    for k, v in counts.items():
        empirical_pmf[k - min_plot] = v
    empirical_pmf /= empirical_pmf.sum()

    return range(min_plot, max_plot), empirical_pmf

@make_plot_function
def compare_pmf(x: tp.Iterable[int], y: tp.Iterable[int], ax: plt.Axes = None, legend=True, s=None):
    """
    Compare the probability mass function of two discrete samples.
    """
    fig = ax.figure
    ax.scatter(*discrete_histogram(x), label='1', s=s)
    ax.scatter(*discrete_histogram(y), label='2', s=s)
    if legend:
        fig.legend()

def random_element_in_a_sequence(x: tp.Sequence):
    return x[np.random.randint(len(x))]

@make_plot_function
def qqplot_to_gaussian(x: np.ndarray, quantile: tp.Union[tp.Literal['auto'], float],ax: plt.Axes = None):
    # tested
    """
    Q-Q plot a sample with respect to the Gaussian distribution

    :param x: sample
    :param quantile: if 'auto', use statsmodel's engine. Otherwise, use a uniform grid between the value corresponding to the `1-quantile`-quantile and the `quantile`-quantile.
    :param ax: plt.Axes to plot on. If None, a new Axes will be created
    :return: if show, returns nothing. Otherwise the figure is returned.
    """
    x = np.array(x)
    loc = np.median(x)
    scale = np.quantile(x, norm.cdf(1)) - loc
    if quantile == 'auto':
        # noinspection PyTypeChecker
        qqplot(x, loc=loc, scale=scale, ax=ax, line='45')
    else:
        dist = norm(loc=loc, scale=scale)
        x_low, x_high = dist.ppf([1-quantile, quantile])
        grid_x = np.linspace(x_low, x_high)
        grid_quantile = dist.cdf(grid_x)
        grid_y = np.quantile(x, grid_quantile)
        ax.plot(grid_x, grid_x)
        ax.scatter(grid_x, grid_y)
        ax.set_xlabel('Theoretical quantile')
        ax.set_ylabel('Sample quantile')

def naive_sample_without_replacement_via_rejection(N, k):
    # tested
    if k > N:
        raise ValueError
    res = OrderedDict()
    for _ in range(k):
        while True:
            proposal = np.random.randint(low=0, high=N)
            if proposal not in res:
                break
        res.update({proposal: 0})
    return np.array(list(res))

class ParallelWrapper:
    def __init__(self, f, print_progress: bool = False):
        self.f = f
        self.print_progress = print_progress

    def __call__(self, x, queue, order: int, seed: int):
        np.random.seed(seed)
        t0 = time.process_time()
        result = self.f(x)
        t1 = time.process_time()
        if self.print_progress and order == 0:
            print('The first process finished in {} secs'.format(t1-t0))
        queue.put((order, result))

def parallelly_evaluate(f, parargs:list, n_cores:int, start_method:str, print_progress: bool = False) -> list:
    # tested
    ctx = mp.get_context(start_method)
    queue = ctx.Queue()

    wrapped_f = ParallelWrapper(f, print_progress=print_progress)
    seeds = naive_sample_without_replacement_via_rejection(2**32, len(parargs))
    processes = [ctx.Process(target=wrapped_f, args=(arg, queue, i, seed)) for (i, arg), seed in zip(enumerate(parargs), seeds)]

    # for process in processes:
    #     process.start()
    # res = [queue.get() for _ in parargs]
    # for process in processes:
    #     process.join()
    # noinspection PyTypeChecker
    res = _parallelly_evaluate(processes, queue, n_cores)

    res = {i:r for i, r in res}
    return [res[k] for k in sorted(res)]

def _parallelly_evaluate(processes: tp.List[mp.Process], queue: mp.Queue, n_cores:int) -> list:
    res = []

    def _get_one_result_and_append_to_res():
        incoming_result = queue.get()
        underlying_process = processes[incoming_result[0]]
        underlying_process.join(timeout=10)
        assert underlying_process.exitcode == 0
        underlying_process.close()
        res.append(incoming_result)

    for core, process in zip_longest(range(n_cores), processes):
        if (core is not None) and (process is not None):
            process.start()
        elif core is None:
            # there is no more core to execute processes, so we must wait for one process to finish before executing a new one
            _get_one_result_and_append_to_res()
            process.start()
    while len(res) < len(processes):
        _get_one_result_and_append_to_res()

    return res

def memory_protection(maxsize_in_MB: float):
    if psutil.Process(os.getpid()).memory_info().rss/1024**2 > maxsize_in_MB:
        raise MemoryError

def has_len(o):
    try:
        len(o)
    except TypeError:
        return False
    else:
        return True

def unpack(nested):
    try:
        while True:
            nested = nested[0]
    except TypeError:
        return nested

def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(content, path: str):
    with open(path, 'w') as f:
        json.dump(content, f)

class _jsondumpee:
    def __init__(self):
        self.size = 0
        self.cache = ''

    def write(self, x):
        assert isinstance(x, str)
        self.cache += x
        if len(self.cache) > 1000:
            self.size += self.cache.__sizeof__()
            self.cache = ''

    @property
    def size_in_MB(self):
        return self.size / (1024**2)

def estimate_size_json(content):
    f = _jsondumpee()
    # noinspection PyTypeChecker
    json.dump(content, f)
    return f.size_in_MB

def run_pf_with_memory_protection(pf: particles.SMC, maxmem: float):
    start_time = time.perf_counter()
    for _ in pf:
        memory_protection(maxmem)
        pass
    end_time = time.perf_counter()
    pf.cpu_time = end_time - start_time

@make_plot_function
def plot_one_function(f: tp.Callable[[float], float], ax: plt.Axes = None, start=None, stop=None, num=None, *args, **kwargs):
    x_plot = np.linspace(start=start, stop=stop, num=num)
    y = [f(x) for x in x_plot]
    ax.plot(x_plot, y, *args, **kwargs)

class group_by_chunk:
    # tested
    def __init__(self, iterable: tp.Iterable, n:int, last_chunk: bool = False):
        self.iterator = iter(iterable)
        self.n = n
        self.last_chunk = last_chunk

    def __iter__(self):
        return self

    def __next__(self) -> list:
        res = []
        try:
            for _ in range(self.n):
                res.append(next(self.iterator))
        except StopIteration:
            if (not self.last_chunk) or (len(res) == 0):
                raise
            else:
                return res
        else:
            return res

def quad_form(A: np.ndarray, x: np.ndarray) -> float:
    # tested
    """
    Calculate x.T @ A @ x and returns a scalar, whether `x` is a 0D or 1D vector.
    """
    # noinspection PyTypeChecker
    return np.sum((A@x) * x)

class conf_itvl(tp.NamedTuple):
    high: float
    low: float
    mean: float

def plot_ETN(T_sample: tp.Sequence[float], Ns: tp.Iterable[int], show: bool = True) -> tp.Tuple[dict, plt.Figure]:
    # tested
    """
    Plot the graph of E[min(T,N)] w.r.t. N
    """
    res = {}
    sample_len = len(T_sample)
    for N in Ns:
        minned = np.minimum(T_sample, N)
        # noinspection PyTypeChecker
        mean: float = np.mean(minned)
        std = np.std(minned)
        low = mean - 1.96 * std/sample_len**0.5
        high = mean + 1.96 * std/sample_len**0.5
        res[N] = conf_itvl(high=high, low=low, mean=mean)
    res = dict(sorted(res.items()))

    # Now plot.
    fig, ax = plt.subplots()  # type: (plt.Figure, plt.Axes)
    ax.plot(list(res.keys()), [c.mean for c in res.values()], marker='o')
    ax.fill_between(list(res.keys()), [c.low for c in res.values()], [c.high for c in res.values()], alpha=0.2)
    ax.set_xscale('log')
    if show:
        fig.show()

    return res, fig

def inverse_permutation(p: np.ndarray) -> np.ndarray:
    # tested
    """
    :param p: an array of length N and taking values in [0,N-1]. Should be thought of as a map.
    """
    res = np.zeros(len(p), dtype=int)
    res[p] = np.arange(len(p))
    return res

def _iterable(e):
    try:
        iter(e)
    except TypeError:
        return False
    else:
        return True

def recursively_convert_to_tuple(x: tp.Iterable) -> tuple:
    return tuple([e if not _iterable(e) else recursively_convert_to_tuple(e) for e in x])

_Ty1 = tp.TypeVar('_Ty1')
_Ty2 = tp.TypeVar('_Ty2')
_Ty3 = tp.TypeVar('_Ty3')
_Ty4 = tp.TypeVar('_Ty4')

class ZipWithAssert:
    """Like zip, but raises AssertionError if iterables are not of the same length."""
    # tested
    def __init__(self, *iterables: tp.Iterable):
        self.iterators = [iter(iterable) for iterable in iterables]

    def __iter__(self):
        return self

    def __next__(self):
        res = []
        for iterator in self.iterators:
            try:
                res.append(next(iterator))
            except StopIteration:
                pass
        if len(res) == 0:
            raise StopIteration
        elif len(res) == len(self.iterators):
            return tuple(res)
        else:
            raise AssertionError

@tp.overload
def zip_with_assert(i1: tp.Iterable[_Ty1], i2: tp.Iterable[_Ty2]) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2]]:
    ...

@tp.overload
def zip_with_assert(i1: tp.Iterable[_Ty1], i2: tp.Iterable[_Ty2], i3: tp.Iterable[_Ty3]) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2, _Ty3]]:
    ...

@tp.overload
def zip_with_assert(i1: tp.Iterable[_Ty1], i2: tp.Iterable[_Ty2], i3: tp.Iterable[_Ty3], i4: tp.Iterable[_Ty4]) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2, _Ty3, _Ty4]]:
    ...

def zip_with_assert(*args):
    return ZipWithAssert(*args)

def _proba_arr_to_pmf(j: int, proba_array: tp.Sequence[float]):
    if 0 <= j < len(proba_array):
        return proba_array[j]
    else:
        return 0

def proba_array_to_pmf(proba_array: tp.Sequence[float]) -> tp.Callable[[int], float]:
    # noinspection PyTypeChecker
    return partial(_proba_arr_to_pmf, proba_array=proba_array)

class reusable_map(Iterable):
    def __init__(self, func, iterable):
        self._func = func
        self._itrb = iterable

    def __iter__(self):
        return map(self._func, self._itrb)

def faster_np_r_tuple(x: tp.List[tp.Union[float, tp.Iterable[float]]]):
    """
    Create a numpy array from x
    """
    return np.array(list(chain(*map(_make_iterable, x))))

def _make_iterable(k):
    try:
        iter(k)
    except TypeError:
        return [k]
    else:
        return k

class composition:
    """
    Composition of functions (in the mathematical order)
    """
    # tested
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        res = self.funcs[-1](*args, **kwargs)
        for f in reversed(self.funcs[:-1]):
            res = f(res)
        return res

VERY_SMALL = 1e-13

@tp.overload
def log_minus_exp(a: float, b:float) -> float: ...
@tp.overload
def log_minus_exp(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
def log_minus_exp_old_slow(a, b):
    # tested
    # assert np.all(a > b - VERY_SMALL)
    with np.errstate(divide='ignore'):
        return np.where(a - b > VERY_SMALL, a + np.log1p(-np.exp(b - a)), -np.inf)

def log_minus_exp(a, b):
    # tested
    assert len(a) == len(b)
    res = []
    for _a, _b in zip(a, b):
        if _a - _b > VERY_SMALL:
            res.append(_a + math.log1p(-math.exp(_b-_a)))
        else:
            res.append(-np.inf)
    return res

def pack(*args):
    return args

def shut_off_numba_warnings():
    warnings.simplefilter('ignore', category=NumbaWarning)

def shut_off_opt_warnings():
    warnings.simplefilter('ignore', category=OptimizeWarning)
    warnings.simplefilter('ignore', category=LinAlgWarning)

class _iter_data_frame_by_row:
    # tested
    """
    Iter over a pandas DataFrame where a dictionary is returned for each row in the iloc order.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self.i == len(self.df):
            raise StopIteration
        res = self.df.iloc[self.i, :].to_dict()
        self.i = self.i + 1
        return res

@tp.overload
def iter_data_frame_by_row(df: pd.DataFrame) -> tp.Iterable[dict]:
    ...

def iter_data_frame_by_row(df: pd.DataFrame):
    return _iter_data_frame_by_row(df)

@tp.overload
def simultaneously_reorder(seq1: tp.Sequence[_Ty1], seq2: tp.Sequence[_Ty2]) -> tp.Tuple[tp.Sequence[_Ty1], tp.Sequence[_Ty2]]: pass

@tp.overload
def simultaneously_reorder(seq1: tp.Sequence[_Ty1], seq2: tp.Sequence[_Ty2], seq3: tp.Sequence[_Ty3]) -> tp.Tuple[tp.Sequence[_Ty1], tp.Sequence[_Ty2], tp.Sequence[_Ty3]]: pass

def simultaneously_reorder(*args: tp.Sequence):
    # tested
    """
    Simulatenously reorder several sequences of equal lengths by sorting them by the first sequence, then by the second sequence, and so on.
    :return: Newly reordered sequences
    """
    zipped_sorted = sorted(zip(*args))
    unzipped = map(list, zip(*zipped_sorted))
    return tuple(unzipped)

_ET = tp.TypeVar('_ET')

def better_random_choice(x: tp.Sequence[_ET]) -> _ET:
    """
    Randomly choose an element of `x`
    Probably faster than np.random.choice
    """
    return x[np.random.randint(len(x))]

class cached_function:
    # tested, version 17/08/21
    def __init__(self, f: FunctionType):
        self.f = f
        self.cache = {}  # (args, kwargs) -> result

    def __call__(self, *args, **kwargs):
        hashable_kwargs = tuple(i for i in kwargs.items())
        all_args = (args, hashable_kwargs)
        if all_args in self.cache:
            return self.cache[all_args]
        res = self.f(*args, **kwargs)
        self.cache[all_args] = res
        return res

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # DEBUG.append(None)
        bound_method = self.__class__(partial(self.f, instance))
        instance.__dict__[self.f.__name__] = bound_method
        return bound_method

class add_to_list:
    def __init__(self, liste: list):
        self.liste = liste

    def __call__(self, obj):
        self.liste.append(obj)
        return obj

def fast_allclose(a, b):
    diff = np.abs(a - b)
    try:
        return all(diff < 1e-6)
    except TypeError:
        return diff < 1e-6

def sureint(x):
    """
    Check that x is truly an integer, that is, not an array of more than 1 element nor a float.
    """
    res = int(x)
    assert abs(x - res) < 1e-12
    return res

_Ty = tp.TypeVar('_Ty')
_Sy = tp.TypeVar('_Sy')

class create_generator(tp.Iterable[_Ty]):
    # tested
    """
    Create a pickable, reusable generator.
    N.B. : This is merely a typed version of reusable_map...
    """
    def __init__(self, f: tp.Callable[[_Sy], _Ty], args: tp.Iterable[_Sy]):
        self.f = f
        self.args = args
        self.iter = iter(args)

    def __iter__(self) -> tp.Iterator[_Ty]:
        return self.__class__(f=self.f, args=self.args)

    def __next__(self) -> _Ty:
        return self.f(next(self.iter))

def flattened_hilbert_sort(x):
    # todo: report this to Nicolas
    N = len(x)
    return np.array(_fhs(x)).reshape(N)

# noinspection DuplicatedCode
def _fhs(x):
    d = 1 if x.ndim == 1 else x.shape[1]
    if d == 1:
        return np.argsort(x, axis=0)
    xs = invlogit((x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-11))
    maxint = np.floor(2 ** (62 / d))
    xint = np.floor(xs * maxint).astype(np.int)
    return np.argsort(hilbert_array(xint))

_Telem = tp.TypeVar('_Telem')

class LinkedListElem(tp.Generic[_Telem]):
    def __init__(self, next_elem: tp.Optional['LinkedListElem[_Telem]'] = None, previous_elem: tp.Optional['LinkedListElem[_Telem]'] = None, wrapper: 'LinkedList' = None, content: _Telem = None):
        self.next_elem = next_elem
        self.previous_elem = previous_elem
        self.wrapper = wrapper
        self.content = content

    def remove_me(self):
        # tested 061121
        """
        Remove the element from the list
        """
        if self.previous_elem is not None:
            self.previous_elem.next_elem = self.next_elem
        else:
            assert self is self.wrapper.head
            self.wrapper.head = self.next_elem
        if self.next_elem is not None:
            self.next_elem.previous_elem = self.previous_elem
        else:
            assert self is self.wrapper.tail
            self.wrapper.tail = self.previous_elem

class LinkedList:
    """
    Implementation of a linked list. Although Python already supports linked list via the collections.deque class, it is not possible to start from an element, sometimes jump to the left, sometimes jump to the right, and sometimes delete the element before jumping.

    [...]
    He said "Give me music and not just funk"
    He jumps to the left, he jumps to the right
    He jumps up and down all through the night
    [...]
    """
    def __init__(self, input_list: tp.Sequence):
        # tested 061121
        ll_elems = [LinkedListElem(wrapper=self, content=c) for c in input_list]
        for i in range(0, len(input_list) - 1):
            ll_elems[i].next_elem = ll_elems[i+1]
        for i in range(1, len(input_list)):
            ll_elems[i].previous_elem = ll_elems[i-1]
        self.head = ll_elems[0] if len(ll_elems) > 0 else None
        self.tail = ll_elems[-1] if len(ll_elems) > 0 else None

    def __iter__(self):
        return LinkedListIterator(start=self.head, direction='next_elem')

    def __reversed__(self):
        return LinkedListIterator(start=self.tail, direction='previous_elem')

    def _get_chain_element(self, i: int) -> LinkedListElem:
        # tested
        if i < 0:
            raise ValueError
        j = 0
        state = self.head
        while j < i and (state is not None):
            j = j + 1
            state = state.next_elem
        if state is None:
            raise ValueError
        else:
            return state

    @property
    def iter_full(self):
        return LinkedListIterator(start=self.head, direction='next_elem', full_element=True)

class LinkedListIterator:
    # tested 061121
    def __init__(self, start: LinkedListElem, direction: tp.Literal['next_elem', 'previous_elem'], full_element=False):
        self.state = start
        self.direction = direction
        self.full_element = full_element

    def __iter__(self):
        return self

    def __next__(self):
        if self.state is None:
            raise StopIteration
        else:
            res = self.state.content if (not self.full_element) else self.state
            self.state = getattr(self.state, self.direction)
            return res

_KeyType = tp.TypeVar('_KeyType')
_ValueType = tp.TypeVar('_ValueType')

class AdvancedOrderedDict(tp.Generic[_KeyType, _ValueType]):
    # tested 071121
    """
    An ordered dict implementation that supports finding the next and the previous keys
    """
    def __init__(self, d: tp.Dict[_KeyType, _ValueType]):
        self.ref_dict = OrderedDict(d)
        self.key_linkedlist = LinkedList([k for k in d])
        self.k_to_node: tp.Dict[tp.Any, LinkedListElem] = OrderedDict({k: node for k, node in zip_with_assert(d, self.key_linkedlist.iter_full)})

    def next_key(self, key: _KeyType) -> _KeyType:
        nextk_node = self.k_to_node[key].next_elem
        return nextk_node.content if nextk_node is not None else None

    def previous_key(self, key: _KeyType) -> _KeyType:
        previousk_node = self.k_to_node[key].previous_elem
        return previousk_node.content if previousk_node is not None else None

    def __delitem__(self, key):
        del self.ref_dict[key]
        self.k_to_node[key].remove_me()
        del self.k_to_node[key]

    def check_consistency(self):
        assert list(self.ref_dict) == list(self.key_linkedlist) == list(self.k_to_node)
        assert [v.content for v in self.k_to_node.values()] == list(self.ref_dict)

    def __getitem__(self, item: _KeyType) -> _ValueType:
        return self.ref_dict[item]

    def __setitem__(self, key, value):
        if key not in self.ref_dict:
            raise NotImplementedError('Inserting new key is not supported yet.')
        self.ref_dict[key] = value

def max_idx(x):
    z = max(x)
    return [i for i in range(len(x)) if x[i] == z]

class AutomaticList:
    """
    A list which automatically adds elements to itself so that the operations on the prescribed index can be carried out. In other words, a defaultdict-like implementation for list. Be careful: negative indexing leads to surprising behaviours.
    """
    def __init__(self, filler=None):
        self.filler = filler
        self.underlying_list = []

    def make_available(self, k):
        while True:
            try:
                self.underlying_list[k]
            except IndexError:
                self.underlying_list.append(self.filler)
            else:
                break

    def __getitem__(self, item):
        self.make_available(item)
        return self.underlying_list[item]

    def __setitem__(self, key, value):
        self.make_available(key)
        self.underlying_list[key] = value

    def __repr__(self):
        return self.underlying_list.__repr__()

    def __str__(self):
        return self.underlying_list.__str__()

class _RunOption(tp.NamedTuple):
    args: list
    show: bool

class _vectorise:
    def __init__(self, f):
        self.f = f

    def __call__(self, runopt: _RunOption):
        return [self.f(arg) for arg in tqdm(runopt.args, disable=not runopt.show)]

def multiple_next(it, n):
    return [next(it) for _ in range(n)]

def _divide_tasks(tasks: list, ncores):
    # tested
    assert ncores > 0 and ncores == int(ncores)
    ntasks = len(tasks)
    k = int(np.ceil(ntasks/ncores))
    b = k * ncores - ntasks
    a = ncores - b
    assert (a >= 0) and (b >= 0)

    res = []
    it = iter(tasks)
    for _ in range(a):
        res.append(multiple_next(it, k))
    for _ in range(b):
        res.append(multiple_next(it, k-1))

    try:
        next(it)
    except StopIteration:
        pass
    else:
        raise AssertionError

    return res

def auto_load_balance(f, parargs: list, ncores: int, start_method: str, print_progress: bool):
    # tested
    if ncores is None:
        return [f(arg) for arg in tqdm(parargs, disable=not print_progress)]
    partitioned_args = _divide_tasks(tasks=parargs, ncores=ncores)
    assert len(partitioned_args) == ncores
    show_it = chain([print_progress], repeat(False))
    runopts = [_RunOption(args=args, show=show) for args, show in zip(partitioned_args, show_it)]
    res = parallelly_evaluate(f=_vectorise(f), parargs=runopts, n_cores=ncores, start_method=start_method, print_progress=False)
    return list(chain(*res))

def auto_colour(s: str, seed=None) -> tp.Tuple[float, float, float]:
    """
    Convert, in a reproducible fashion, a string into a RGB Tuple representing some colour. Helpful for automatic colouring in matplotlib.
    """
    b = s.encode()
    hasher = sha256()
    hasher.update(b)
    digest: tp.List[str] = list(str(int(hasher.hexdigest(), 16)))
    res = []
    with temporary_numpy_seed(seed):
        for i in range(3):
            three_digits = np.random.choice(digest, size=3, replace=True)
            three_digits = [int(d) * 10**(-j-1) for j, d in enumerate(three_digits)]
            res.append(sum(three_digits))
    assert all([0 <= e <= 1 for e in res])
    # noinspection PyTypeChecker
    return tuple(res)

class Multiple(tp.Generic[_Ty]):
    """
    Used for updating a function that originally received as argument only one element of a certain type, so that it can handle multiple elements of that type without loss of backward compatibility. Each time multiple elements are to be passed, they have to be explicitly wrapped around by Multiple. A little bit quick and (too) dirty, but it works.
    """
    def __init__(self, list_elems: tp.List[_Ty]):
        if not isinstance(list_elems, list):
            raise ValueError
        self.list_elems = list_elems

    def __iter__(self):
        return iter(self.list_elems)

    @classmethod
    def multiplise(cls, e: tp.Union[_Ty, 'Multiple[_Ty]']) -> 'Multiple[_Ty]':
        if isinstance(e, cls):
            return e
        else:
            return cls([e])

_RetTyp = tp.TypeVar('_RetTyp')

class composition_with_star(tp.Generic[_RetTyp]):
    def __init__(self, f: tp.Callable[..., _RetTyp]):
        self.f = f

    def __call__(self, it: tp.Iterable) -> _RetTyp:
        return self.f(*it)

def tqdm_enable(it, enable=True):
    """
    Handy replacement for tqdm(iterator, disable=not verbose) [significant slowdown in case verbose=False]
    """
    if enable:
        return tqdm(it)
    else:
        return it

def list_chain_star(o: tp.Iterable[tp.Iterable]):
    """
    Concatenate the iterables contained *in* o
    >>> list_chain_star([[1, 2, 4], [3, 7], [8]])
    [1, 2, 4, 3, 7, 8]
    """
    return list(chain(*o))

def dearray(x):
    try:
        return float(x)
    except TypeError:
        return x.tolist()

def numpy_dim(x) -> int:
    """
    Calculate the programming dimension of x. If x is a float-like object, returns 0. If x is a vector in R^d, return d.
    """
    return sum(np.array(x).shape)

def convert_to_numpy_dim(x, d):
    # tested
    if d >= 1:
        res = np.atleast_1d(x)
    else:
        if d != 0:
            raise ValueError
        res = float(np.array(x))
    if numpy_dim(res) != d:
        raise ValueError
    return res

class non_trivial_function:
    # tested
    """
    Create a non-linear, asymmetric function from R^m to R^n. If m=n, the resulting function is also a diffeomorphism with easy to calculate derivatives and inverse. Useful for testing various algorithms.
    """
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        if input_dim < 1 or output_dim < 1:
            raise ValueError
        self.input_dim = input_dim
        self.output_dim = output_dim
        with temporary_numpy_seed(seed):
            self.matrix = np.random.rand(output_dim, input_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.input_dim, ):
            raise ValueError
        return self.matrix @ (x + np.sin(x))

_Ts = tp.TypeVar('_Ts')

def list_map(f, i):
    return list(map(f, i))

@pickable_make_plot_function
def smoothing_plot(true_states: tp.Sequence[_Ts], generated_trajectories: tp.Collection[tp.Sequence[_Ts]], phi: tp.Callable[[_Ts], float], title: str = None, ax: plt.Axes = None):
    ax.plot(list_map(phi, true_states), color='blue')
    for traj in generated_trajectories:
        ax.plot(list_map(phi, traj), color='grey', alpha=0.5)
    ax.set_title(title)

class TrajectoryPiece:
    def __init__(self, xendpoints: tp.Tuple[int, int], yendpoints: tp.Tuple[float, float]):
        self.x_endpoints = (int(xendpoints[0]), int(xendpoints[1]))
        self.y_endpoints = (float(yendpoints[0]), float(yendpoints[1]))

    def __hash__(self):
        return self.x_endpoints.__hash__() + self.y_endpoints.__hash__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (other.x_endpoints == self.x_endpoints) and np.allclose(other.y_endpoints, self.y_endpoints)

    @classmethod
    def from_trajectory(cls, traj: tp.Sequence[float]) -> tp.Sequence['TrajectoryPiece']:
        xpairs = pairwise(range(len(traj)))
        ypairs = pairwise(traj)
        return [cls(xp, yp) for xp, yp in zip_with_assert(xpairs, ypairs)]

    def __repr__(self):
        return '{}[x={}, y={}]'.format(self.__class__.__name__, self.x_endpoints, self.y_endpoints)

def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    Copied from Python's library documentation.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

@pickable_make_plot_function
def improved_smoothing_plot(true_states: tp.Sequence[_Ts], generated_trajectories: tp.Collection[tp.Sequence[_Ts]], phi: tp.Callable[[_Ts], float], style_true: dict, style_generated: dict, ax: plt.Axes = None):
    ax.plot(list_map(phi, true_states), **style_true)
    generated_trajectories = [list_map(phi, traj) for traj in generated_trajectories]
    pieces = [TrajectoryPiece.from_trajectory(traj) for traj in generated_trajectories]
    # noinspection PyTypeChecker
    pieces = list_chain_star(pieces)
    for p in pieces:
        assert isinstance(p, TrajectoryPiece)
    pieces: tp.Set[TrajectoryPiece] = set(pieces)
    style_generated = style_generated.copy()
    for i, p in enumerate(pieces):
        if i == 1:
            try:
                style_generated.pop('label')
            except KeyError:
                pass
        ax.plot(p.x_endpoints, p.y_endpoints, **style_generated)