import os
from programs.number_of_cores_limiter import limit_number_cores; limit_number_cores(1, os)
import matplotlib.pyplot as plt
import seaborn as sns
# noinspection PyUnresolvedReferences
from libs_new.utils import auto_load_balance, zip_with_assert, temporary_numpy_seed
from programs.sde.illustrations.supporting_funcs import SDEMeeting, PureLindvallMeeting, HybridLindvallMeeting
import typing as tp
from scipy import stats
import numpy as np

np.seterr(divide='raise', invalid='raise')

def f(x):
    assert x is not None
    return np.array([0])
def sigma(x):
    assert x is not None
    return np.identity(1)
a = np.array([0])
b = np.array([1.5])
Tmax = 5
nruns = 20_000
ncores: tp.Optional[int] = 50
mode = 'min'

delta0 = 0.002
deltaS = [0.2, 0.02]
name0 = r'MLR coupler, $\delta={}$'.format(delta0)
nameS = [r'MLR coupler, $\delta={}$'.format(delta) for delta in deltaS]
style0 = dict(color='grey')
styleS = [dict(color='grey', linestyle='dotted'),
          dict(color='black', linestyle='dotted')
          ]
engine0: tp.Type[SDEMeeting] = HybridLindvallMeeting
engineS: tp.Type[SDEMeeting] = HybridLindvallMeeting

# The game starts here
assert PureLindvallMeeting
assert HybridLindvallMeeting

ndebug0 = int(Tmax/delta0)
ndebugS = [int(Tmax/delta) for delta in deltaS]

def repeat(_):
    # noinspection PyTypeChecker
    return engine0(f=f, sigma=sigma, a=a, b=b, delta=delta0, n_debug=ndebug0, mode=mode).meeting_idx
meeting_idx0 = auto_load_balance(repeat, [None] * nruns, ncores, 'fork', True)

meeting_idxS = []
for delta, ndebug in zip_with_assert(deltaS, ndebugS):
    def repeat(_):
        # noinspection PyTypeChecker
        return engineS(f=f, sigma=sigma, a=a, b=b, delta=delta, n_debug=ndebug, mode=mode).meeting_idx
    meeting_idx = auto_load_balance(repeat, [None] * nruns, ncores, 'fork', True)
    meeting_idxS.append(meeting_idx)

# Post-processing meeting proba
def prop_notnone(arr):
    return len([e for e in arr if e is not None]) / len(arr)
meeting_proba0 = prop_notnone(meeting_idx0)
meeting_probaS = [prop_notnone(e) for e in meeting_idxS]

# Post-processing graph
def extract_notnone(arr, scale):
    return [float(e * scale) for e in arr if e is not None]
cidx0 = extract_notnone(meeting_idx0, delta0)
cidxS = [extract_notnone(e, delta) for e, delta in zip_with_assert(meeting_idxS, deltaS)]

fig, ax = plt.subplots()
sns.kdeplot(x=cidx0, ax=ax, label=name0, **style0)
for ds, name, style in zip_with_assert(cidxS, nameS, styleS):
    sns.kdeplot(x=ds, ax=ax, label=name, **style)

def show():
    ax.legend()
    fig.show()

# Brownian graph decoration
true_dist = stats.levy(scale=abs(float(b) - float(a))**2/4)
Z = true_dist.cdf(Tmax)
def true_dist_simulate():
    while True:
        x = true_dist.rvs()
        if x < Tmax:
            return x
sns.kdeplot(x=[true_dist_simulate() for _ in range(nruns)], ax=ax, label='True distribution', color='black')

ax.legend()
ax.set_xlabel('meeting time')

fig.savefig('./programs/sde/illustrations/brownian_delta0_meeting.pdf')

if __name__ == '__main__':
    pass