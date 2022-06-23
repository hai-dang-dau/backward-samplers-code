"""
Each data extractor transforms * either * ONE csv line * or * ONE output line (i.e. one list of 50 dicts)
"""
import typing as tp
from abc import abstractmethod, ABC
from itertools import chain

import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from libs_new.utils import Multiple, zip_with_assert, list_map
from programs.workflow import FilteredResults, iterate_over_filtered_results


class _QOE(tp.NamedTuple):
    t: tp.Sequence[int]
    low: tp.Sequence[float]
    high: tp.Sequence[float]

def quantiles_of_expectation(runs: tp.List[tp.Dict], i: tp.Optional[int], q=0.25, name: str = 'expectations') -> _QOE:
    """
    :param i: index of the considered expectation (reminder: the `expectations` key of each run in runs is a list of vectors). i can also be set to None if each online expectation is a single float
    """
    t_array = []
    lows = []
    highs = []
    Tmax = len(runs[0][name])
    assert all([len(run[name]) == Tmax for run in runs])
    for t in range(Tmax):
        t_array.append(t)
        expectations = [float(_get_elem(run[name][t], i)) for run in runs]
        lows.append(np.quantile(expectations, q))
        highs.append(np.quantile(expectations, 1-q))
    return _QOE(t=t_array, low=np.array(lows), high=np.array(highs))

def _get_elem(a, b):
    return a[b] if b is not None else a

class PairFilteredResult(tp.NamedTuple):
    input_dict: dict
    output: tp.List[dict]

class Line2DInfo(tp.NamedTuple):
    x: tp.Sequence[float]
    y: tp.Sequence[float]

def plot_lines_from_output(filtered_results: FilteredResults, ax: plt.Axes, line_parameters: tp.Callable[[PairFilteredResult], dict], line_values: tp.Callable[[PairFilteredResult], Line2DInfo]) -> None:
    """
    Plot the `filtered_results` on `ax` by representing each of its input-output pairs by a line. The line is defined in the traditional x-y manner and other `line_parameters` (e.g. color, linestyle, label, ...).
    """
    for input_dict, output in iterate_over_filtered_results(filtered_results):
        pair = PairFilteredResult(input_dict=input_dict, output=output)
        param = line_parameters(pair)
        xyinfo = line_values(pair)
        # ax.plot(xyinfo.x, xyinfo.y, **param)
        plot_wrapper(ax=ax, line=xyinfo, param=param)

def plot_wrapper(ax: plt.Axes, line: Line2DInfo, param: dict):
    thin_to: int = param.pop('thin_to', None)
    if thin_to is not None:
        line = reduce_sample_rate(line=line, num=thin_to, desired_log=param.pop('thin_to_log_base', None))
    plotter = param.pop('plotter', 'plot')
    getattr(ax, plotter)(line.x, line.y, **param)

def draw_mean_really_moved_rate(pair: PairFilteredResult) -> Line2DInfo:
    highlight_every = pair.input_dict['highlight_every']
    output = pair.output
    T = len(output[0]['mean_really_moved_rate']) - 1
    t_array = []
    vals = []
    for t in range(0, T + 1):
        mean_rates = [run['mean_really_moved_rate'][t] for run in output]
        if (t > 0) and ((t - 1) % highlight_every == 0):
            assert all([isinstance(r, float) for r in mean_rates])
            assert all([not np.isnan(r) for r in mean_rates])
            t_array.append(t)
            vals.append(float(np.mean(mean_rates)))
        else:
            assert all([r is None for r in mean_rates])
    return Line2DInfo(x=t_array, y=vals)

def draw_quantile_really_moved_rate(pair: PairFilteredResult, q=0.25) -> tp.Tuple[Line2DInfo, Line2DInfo]:
    highlight_every = pair.input_dict['highlight_every']
    n_highlight = pair.input_dict['n_highlight']
    output = pair.output
    T = len(output[0]['mean_really_moved_rate']) - 1
    t_array = []
    lows = []
    highs = []
    for t in range(0, T + 1):
        highlights = [run['really_moved_rate_highlight'][t] for run in output]
        if (t - 1) % highlight_every != 0:
            assert all([h is None for h in highlights])
        else:
            assert all([len(h) == n_highlight for h in highlights])
            combined = list(chain(*highlights))
            combined = [float(e) for e in combined]
            t_array.append(t)
            lows.append(np.quantile(combined, q))
            highs.append(np.quantile(combined, 1-q))
    return Line2DInfo(t_array, lows), Line2DInfo(t_array, highs)

def get_all_estimates(runs: tp.List[dict], t: int, i: tp.Optional[int], name='expectations') -> tp.Set[float]:
    return [float(_get_elem(run[name][t], i)) for run in runs]

def boxplot_from_output(filtered_results: FilteredResults, ax: plt.Axes, boxname: tp.Callable[[PairFilteredResult], tp.Union[str, Multiple[str]]], boxcontent: tp.Callable[[PairFilteredResult], tp.Union[tp.Set[float], Multiple[tp.Set[float]]]]):
    labels = []
    vectors = []
    for input_dict, output in iterate_over_filtered_results(filtered_results):
        pair = PairFilteredResult(input_dict, output)
        names = Multiple.multiplise(boxname(pair))
        contents = Multiple.multiplise(boxcontent(pair))
        for name, content in zip_with_assert(names, contents):
            labels.append(name)
            vectors.append(np.array(content))
    ax.boxplot(vectors, labels=labels)

def boxcontent_rejection_exec_time_per_NT(pair: PairFilteredResult, expected_T: int) -> Multiple[tp.Set[float]]:
    """
    Given a pair of filtered result which corresponds to a smoothing experiment where the algorithm is 'reject', returns two sets of diagnostic values calculated for every run. The diagnostics values are the mean execution time per N per T for the pure rejection and the hybrid rejection variant.
    :param expected_T: T in the book's convention (e.g. X_T exists)
    :return two lists, first one for pure rejection and second one for hybrid
    """
    res = [[], []]
    if pair.input_dict['algo'] != 'reject':
        raise ValueError
    for run in pair.output:
        for i, key in enumerate(['mean_pure_cost', 'mean_hybrid_cost']):
            costs_versus_t = run[key]
            assert costs_versus_t[0] is None
            costs_versus_t = [float(e) for e in costs_versus_t[1:]]
            if len(costs_versus_t) != expected_T:
                raise ValueError
            res[i].append(np.mean(costs_versus_t))
    return Multiple(res)

def mean_of_sth_over_all_runs_wrt_t(pair: PairFilteredResult, thing: str = 'ESS_ratios') -> Line2DInfo:
    # tested
    L = len(pair.output[0][thing])
    nrun = len(pair.output)
    arr = np.array([d[thing] for d in pair.output])
    assert arr.shape == (nrun, L)
    y = np.mean(arr, axis=0)
    assert y.shape == (L, )
    return Line2DInfo(x=np.arange(L), y=y)

def boxcontent_ESS_ratios(pair: PairFilteredResult, thing: str = 'ESS_ratios') -> tp.Sequence[float]:
    """
    Returns the T + 1 values of averaged ESS ratios over all runs. Similar to mean_of_sth_over_all_runs_wrt_t, except that only returns the raw list.
    """
    x, y = mean_of_sth_over_all_runs_wrt_t(pair=pair, thing=thing)
    return y

def reduce_sample_rate_linear(line: Line2DInfo, num: int) -> Line2DInfo:
    spacing = len(line.x) // num
    return Line2DInfo(line.x[::spacing], line.y[::spacing])

def log_progressive_index(n: int, b: int, num: int) -> tp.Sequence[int]:
    # tested and auto-tested
    """
    Return a subset of integer indices from 0 to n such that it increases (approximately) in the logscale of base b. It will contain approximately num elements.
    """
    res = np.logspace(start=0, stop=np.log(n - 0.5)/np.log(b), num=num, base=b) - 1
    res = sorted(set([int(e) for e in res]))
    # print(res)
    if not all([0 <= e < n for e in res]):
        raise ValueError
    return res

def reduce_sample_rate_log(line: Line2DInfo, num: int, desired_log: int) -> Line2DInfo:
    idx = log_progressive_index(n=len(line.x), b=desired_log, num=num)
    newx = np.array(line.x)[idx]
    newy = np.array(line.y)[idx]
    return Line2DInfo(x=newx, y=newy)

def reduce_sample_rate(line: Line2DInfo, num: int, desired_log: tp.Optional[int]) -> Line2DInfo:
    if desired_log is None:
        return reduce_sample_rate_linear(line=line, num=num)
    else:
        return reduce_sample_rate_log(line=line, num=num, desired_log=desired_log)

class ExactPlotter(ABC):
    @abstractmethod
    def __call__(self, t):
        ...

def _pydoc_moving_average(iterable, n=3):
    # copied from Python documentation
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # http://en.wikipedia.org/wiki/Moving_average
    it = iter(iterable)
    d = deque(itertools.islice(it, n - 1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n

def _xy_moving_average(line: Line2DInfo, n) -> Line2DInfo:
    # tested
    x = list_map(int, _pydoc_moving_average(line.x, n))
    y = list(_pydoc_moving_average(line.y, n))
    return Line2DInfo(x=x, y=y)