from functools import partial
from operator import itemgetter
from programs.brownian.main import GaussianSmoothingExperiment
# noinspection PyUnresolvedReferences
from libs_new.utils import make_plot_function, composition, read_json, auto_colour, Multiple
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from programs.data_extractor import plot_lines_from_output, quantiles_of_expectation, PairFilteredResult, Line2DInfo, \
    draw_mean_really_moved_rate, draw_quantile_really_moved_rate, get_all_estimates, boxplot_from_output, \
    mean_of_sth_over_all_runs_wrt_t, ExactPlotter, boxcontent_rejection_exec_time_per_NT, boxcontent_ESS_ratios
from programs.lotka_volterra.main import LotkaVolterraSmoothingExperiment
from programs.sde.lotka_volterra_sde import LKVContSmoothableSDE
from programs.sde.main import LotkaVolterraSDEExperiment
from programs.workflow import filter_results
from tqdm import tqdm
import typing as tp
import numpy as np

# Tested: what we need to change is highlighted

"""
Please verify all the following:
(1) Sanity test
(2) Box plot of estimation at different times
(3) MSE graph of error
(4) ESS and coupling rate
(5) Execution time for each t
(6) Optimiser statistics if ScipyLinOpt is used
(7) Filtering stability diagnostics
"""

plt.style.use('ggplot')

# Change here : 1/7
result_path = './programs/brownian/output'
filtered_results = filter_results(result_path, [dict(config_file='guarniero_online_sigma_small')])
colorset = {'dummynan': 'red', 'mcmc_hilbert1.0': 'green', 'mcmc_indep1.0': 'yellow', 'mcmc_indep10.0': 'blue', 'mcmc_hilbert5.0': 'black', 'mcmc_indep1': 'yellow', 'mcmc_indep10': 'blue', 'mcmc_hilbert1': 'green', 'mcmc_hilbert5': 'black', 'mcmc_irhilbert2': 'green', 'mcmc_irhilbert2.0': 'green', 'mcmc_indep2.0': 'yellow', 'mcmc_indep2': 'yellow', 'mcmc_irhilbert10.0': 'black', 'mcmc_irhilbert10': 'black', 'intractable1': 'green', 'intractable1.0': 'green', 'dummyNone': 'red', 'intractableNone': 'green', 'intractableAlwaysIndepOpt': 'yellow', 'intractableScipyLinOpt': 'green', 'mcmc_indepNone': 'yellow', 'intractablenan': 'green', 'intractable0.0': 'yellow', 'intractable0.25': 'blue', 'intractable0': 'yellow'}
expectation_idx = 0
T_plot_low = 0; T_plot_high = 10
T_exact = 10
jump = 1
filtering_stability_diag = False

# Sanity check plotter preparation, all tested

def translate(c):
    if c == 'd':
        return 'N'
    if c == 'm':
        return 'M'
    if c == 'r':
        return 'R'
    return c

def translate2(c):
    if c == 'Bootstrap':
        return c
    elif c == 'GuidedPF':
        return 'Guided'
    else:
        raise ValueError

def line_parameters_low(pair: PairFilteredResult) -> dict:
    label = translate2(pair.input_dict['fk_type'])  # + translate(pair.input_dict['algo'][0])  # Change here: 2/7
    # color = auto_colour(label, 489)
    if label[0] == 'B':
        color = '0.0'
    elif label[0] == 'G':
        color = '0.7'
    else:
        raise ValueError
    if label[1] == 'N':
        style = dict(linestyle='dashed')
    elif label[1] == 'R':
        style = dict(plotter='scatter', marker='x')
    else:
        # assert label[1] == 'M'
        style = dict()
    return dict(label=label, color=color, **style)

def line_values_low(pair: PairFilteredResult) -> Line2DInfo:
    qoe = quantiles_of_expectation(pair.output, expectation_idx)
    return Line2DInfo(x=qoe.t[T_plot_low:T_plot_high], y=qoe.low[T_plot_low:T_plot_high])

def line_parameters_high(pair: PairFilteredResult) -> dict:
    param_low = line_parameters_low(pair)
    return dict(color=param_low['color'])

def line_values_high(pair: PairFilteredResult) -> Line2DInfo:
    qoe = quantiles_of_expectation(pair.output, expectation_idx)
    return Line2DInfo(x=qoe.t[T_plot_low:T_plot_high], y=qoe.high[T_plot_low:T_plot_high])

class GaussianExactPlotter(ExactPlotter):
    def __init__(self):
        self.model = GaussianSmoothingExperiment.model_parser(filtered_results.inputs.iloc[0].to_dict())

    def __call__(self, t):
        return self.model.exact_offline_expectation(t)[expectation_idx]

class LotkaVolterraExactPlotter(ExactPlotter):
    # tested
    def __init__(self):
        self.hidden_states = np.array(read_json('./programs/lotka_volterra/config/' + filtered_results.inputs.iloc[0]['config_file'] + '_hidden.json'))
        self.model = LotkaVolterraSmoothingExperiment.model_parser(filtered_results.inputs.iloc[0].to_dict())
        self.additive_function = self.model.typical_additive_function()

    def __call__(self, t):
        arr = np.array(self.additive_function.phi(t)(self.hidden_states[0:t+1]))
        return float(arr[expectation_idx])

class LkSDEExactPlotter(ExactPlotter):
    def __init__(self):
        # noinspection PyTypeChecker
        self.model: LKVContSmoothableSDE = LotkaVolterraSDEExperiment.model_parser(filtered_results.inputs.iloc[0].to_dict())
        self.additive_function = self.model.typical_additive_function()
        self.hidden_states = self.model.hidden_states

    def __call__(self, t):
        arr = np.array(self.additive_function.phi(t)(self.hidden_states[0:t+1]))
        return float(arr[expectation_idx])

exact_plotter: tp.Optional[ExactPlotter] = GaussianExactPlotter()  # Change here: 3/7 - note whether the exact plotter is online or offline!

# Sanity check plotter
@make_plot_function
def sanity_check(ax: plt.Axes):
    plot_lines_from_output(filtered_results, ax, line_parameters_low, line_values_low)
    plot_lines_from_output(filtered_results, ax, line_parameters_high, line_values_high)
    ax.legend()

    if exact_plotter is not None:
        exact_t = [t for t in range(0, T_exact + 1, jump)]
        exact_values = [float(exact_plotter(t)) for t in tqdm(exact_t)]
        ax.plot(exact_t, exact_values, color='black', linestyle='dotted')

# Squared interquantile range preparation
def squared_iq_range_line_values(pair: PairFilteredResult) -> Line2DInfo:
    # tested
    qoe = quantiles_of_expectation(pair.output, expectation_idx, name='expectations' if not filtering_stability_diag else 'filtering_stability_diag')
    return Line2DInfo(x=qoe.t, y=(qoe.high-qoe.low)**2)

# Squared interquantile range plotter
@make_plot_function
def squared_iq_range(ax: plt.Axes, log=True):
    # tested
    plot_lines_from_output(filtered_results, ax, line_parameters_low, squared_iq_range_line_values)
    ax.legend()
    if log:
        ax.set_xscale('log', basex=5)
        ax.set_yscale('log', basey=5)
    ax.set_ylabel('Squared interquartile range')
    ax.set_xlabel('Time')

# Mean really_moved rate plotter
@make_plot_function
def mean_really_moved_rate(ax: plt.Axes):
    # tested
    plot_lines_from_output(filtered_results, ax, line_parameters_low, draw_mean_really_moved_rate)
    ax.legend()
    ax.set_ylabel('Mean really moved rate')

# Really moved rate quantile plotter
@make_plot_function
def rm_rate_quantile(ax: plt.Axes):
    # tested
    lowrate = composition(itemgetter(0), draw_quantile_really_moved_rate)
    highrate = composition(itemgetter(1), draw_quantile_really_moved_rate)
    plot_lines_from_output(filtered_results, ax, line_parameters_low, lowrate)
    plot_lines_from_output(filtered_results, ax, line_parameters_high, highrate)
    ax.legend()
    ax.set_ylabel('Quantiles of really moved rates')

# Box plotter preparation
def boxname(pair: PairFilteredResult) -> tp.Union[str, Multiple[str]]:
    return translate(pair.input_dict['fk_type'][0]) + translate(pair.input_dict['algo'][0])
    # return pair.input_dict['algo'][0]  # Change here: 4/7

def boxcontent(pair: PairFilteredResult, t: int) -> tp.Union[tp.Set[float], Multiple[tp.Set[float]]]:
    # return boxcontent_rejection_exec_time_per_NT(pair=pair, expected_T=5000)
    return get_all_estimates(pair.output, t, expectation_idx)  # Change here: 5/7

@make_plot_function
def boxplot(ax: plt.Axes, t: int):
    boxplot_from_output(filtered_results, ax, boxname, partial(boxcontent, t=t))
    ax.set_ylabel(r'No. evals per particle at $t={}$'.format(t))
    # Change here 6/7
    if exact_plotter is not None:
        ax.axhline(float(exact_plotter(t)), linestyle='dotted', color='black')

def boxname_reject(pair):
    init = pair.input_dict['fk_type'][0]
    return Multiple([init + 'P', init + 'H'])

def boxcontent_reject(pair):
    return boxcontent_rejection_exec_time_per_NT(pair=pair, expected_T=500)

@make_plot_function
def boxplot_reject(ax):
    boxplot_from_output(filtered_results, ax, boxname_reject, boxcontent_reject)
    ax.set_ylabel('No. evals per N per T')

# Change here 7/7 (there is no more 7)
@make_plot_function
def mean_of_sth_with_respect_t(ax: plt.Axes, sth: str = 'ESS_ratios'):
    plot_lines_from_output(filtered_results, ax, line_parameters_low, partial(mean_of_sth_over_all_runs_wrt_t, thing=sth))
    ax.legend()

if __name__ == '__main__':
    pass

    # all_opts_stats = [dict(skipped=d['optcalls_skipped'], cached=d['optcalls_cached'], performed=d['optcalls_performed']) for d in filtered_results.outputs[0]]
    # fig1 = sanity_check()
    #
    # fig2 = squared_iq_range(show=True)
    # ax2: plt.Axes = fig2.axes[0]
    # # # #  ax2.set_xlim(1000, 10_000)
    # fig2.axes[0].set_yscale('log')
    # fig2.axes[0].set_xscale('log')
    # # # # ax2.set_aspect('equal')
    # fig2.show()
    # try:
    #     fig: plt.Figure = mean_really_moved_rate(show=False)
    #     axes: plt.Axes = fig.axes[0]
    #     axes.set_yscale('log')
    #     fig.show()
    # except Exception as e:
    #     tb = e.__traceback__
    #     raise
    # rm_rate_quantile()