from programs.brownian.main import GaussianSmoothingExperiment

d = {
    'path_random_prefix': 'mr.bean',

    'seed_execution': 302,
    'nruns_per_core': 1,
    'ncores': 1,
    'maxmem': 512,
    'json_maxsize': 0,

    'config_file': 'guarniero_starter',

    'fk_type': 'GuidedPF',
    'N': 500,
    'ESSrmin': 1.0,
    'resampling': 'systematic',
    'skeleton_converter': 'identity',

    'smooth_mode': 'online',
    'algo': 'mcmc_indep',
    'k': 1,
    'N_tilde': 1,

    'n_highlight': 1,
    'highlight_every': 1
}
if __name__ == '__main__':
    pass

try:
    obj = GaussianSmoothingExperiment(d)  # should raise an error, now see obj.failed_to_serialise
except OSError as ose:
    obj: GaussianSmoothingExperiment = ose.__traceback__.tb_next.tb_frame.f_locals['self']
    failed_to_serialise = obj.failed_to_serialise
    model = obj.model_parser(d)
    raise
else:
    raise AssertionError