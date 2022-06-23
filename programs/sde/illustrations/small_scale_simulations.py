from operator import itemgetter
from libs_new import intractable_smoother as ismoother
from libs_new.smoothing_ui import EasyIntractableSmoother, EasyPaRIS, DummyBKOption
from libs_new.utils import smoothing_plot, improved_smoothing_plot
from programs.sde.lotka_volterra_sde import user_friendly_ssm_creator, LKVSDE
import numpy as np

seed_model = 2022
seed_exec = 77483

nat_death_rates = [0.2]
deterministic_slowdown = 1/4
dynamic_noise0_std = 1/10
dynamic_noise1_std = 1/10
log_obs_noise0_std = 0.2
log_obs_noise1_std = 0.2
N_dcr = 100
richman = True
reorderer = ismoother.Adjacent()
Nparticles = 100
T = 50
t_cutoff = 5

x_star = [100, 100]
dynamic_noise_corr = 0.5
init_mean = x_star
init_noise0_std = 10
init_noise1_std = 10
init_noise_corr = 0.5
log_obs_noise_corr = 0.5
ncores = 3

np.random.seed(seed_model)
ssm = user_friendly_ssm_creator(x_star=x_star, nat_death_rates=nat_death_rates, deterministic_slowdown=deterministic_slowdown, dynamic_noise0_std=dynamic_noise0_std, dynamic_noise1_std=dynamic_noise1_std, dynamic_noise_corr=dynamic_noise_corr, init_mean=init_mean, init_noise0_std=init_noise0_std, init_noise1_std=init_noise1_std, init_noise_corr=init_noise_corr, log_obs_noise0_std=log_obs_noise0_std, log_obs_noise1_std=log_obs_noise1_std, log_obs_noise_corr=log_obs_noise_corr, N_dcr=N_dcr, data=None, T=T, t_cutoff=t_cutoff, verbose=True)
sde: LKVSDE = ssm.sde
fig = sde.plot(x=ssm.hidden_states, y=ssm.data, show=False, aes=[dict(color='black'), dict(color='grey')], labels=['prey', 'predator'], plot_stationary=False)
# fig.axes[0].set_title('States and observations')
fig.show()

np.random.seed(seed_exec)
if richman:
    easy = EasyIntractableSmoother(ssm=ssm, N=Nparticles, get_intractable_fk_option=dict(), history=True, reorderer=reorderer, verbose=True, ncores=ncores, start_method='fork' if ncores is not None else None)
else:
    easy = EasyPaRIS(ssm=ssm, backward_option=DummyBKOption(verbose=True), N=Nparticles, resampling='systematic', history=True, ESSrmin=1.0, get_fk_kwargs=dict(), verbose_pf=True)

for j in easy:
    pass
    # print(j.online_expectation)
generated_trajs = [easy.paris.skeleton.sample_one_trajectory() for _ in range(30)]
lab = ['prey', 'predator']
for i in [0]:
    # fig_old = smoothing_plot(true_states=ssm.hidden_states, generated_trajectories=generated_trajs, phi=itemgetter(i), title='Coordinate {} richman {}'.format(i, richman))
    fig_old = smoothing_plot(true_states=ssm.hidden_states, generated_trajectories=generated_trajs, phi=itemgetter(i),
                             title=None)
    fig_new = improved_smoothing_plot(true_states=ssm.hidden_states, generated_trajectories=generated_trajs, phi=itemgetter(i), style_true=dict(color='black', label='Unobserved {} population'.format(lab[i])), style_generated=dict(color='grey', alpha=0.5, label='Smoothing trajectories'))

if __name__ == '__main__':
    pass