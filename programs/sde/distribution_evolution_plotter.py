import typing as tp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from libs_new.utils import zip_with_assert

SampleType = tp.TypeVar('SampleType')

def distribution_evolution_plotter(sampler: tp.Callable[[float], tp.Sequence[SampleType]], functions: tp.Sequence[tp.Callable[[float, SampleType], float]], t_low: float, t_high: float, no_steps: int) -> tp.Sequence[plt.Figure]:
    """
    Plot the evolution of a changing distribution with respect to time `t`. The distribution may be supported on other sets than the set of real numbers, but must be projected to the latter by means of ``functions``. Each of these function will produce a separate figure.
    """
    figs_and_axss = [plt.subplots() for _ in functions]
    figs: tp.List[plt.Figure] = [_[0] for _ in figs_and_axss]
    axss: tp.List[plt.Axes] = [_[1] for _ in figs_and_axss]
    for t, c in _t_and_colors(t_low=t_low, t_high=t_high, no_steps=no_steps):
        s = _transformed_sample(sampler=sampler, t=t, functions=functions)
        _dispatch_to_axes(sample=s, axss=axss, color=c)
    return figs

def _t_and_colors(t_low: float, t_high: float, no_steps: int) -> tp.Iterable[tp.Tuple[float, str]]:
    ts = np.linspace(start=t_low, stop=t_high, num=no_steps)
    greys = [k/no_steps for k in range(1, no_steps + 1)]
    greys = reversed(list(map(str, greys)))
    return zip_with_assert(ts, greys)

def _transformed_sample(sampler: tp.Callable[[float], tp.Sequence[SampleType]], t: float, functions: tp.Sequence[tp.Callable[..., float]]) -> tp.Sequence[tp.Sequence[float]]:
    sample = sampler(t)
    return [[f(t, e) for e in sample] for f in functions]

def _dispatch_to_axes(sample: tp.Sequence[tp.Sequence[float]], axss: tp.Sequence[plt.Axes], color: str) -> None:
    for s, a in zip_with_assert(sample, axss):
        s = list(map(float, s))
        sns.kdeplot(x=s, color=color, ax=a)