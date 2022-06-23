from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
from programs.workflow_smoothing import SmoothingExperiment
from programs.lotka_volterra import model_new
from programs.workflow import start_workflow
from libs_new.utils import shut_off_numba_warnings
import numpy as np

np.seterr(divide='raise', invalid='raise')
shut_off_numba_warnings()

class LotkaVolterraSmoothingExperiment(SmoothingExperiment):
    # tested 251121
    @staticmethod
    def config_path() -> str:
        return './programs/lotka_volterra/config/'

    @staticmethod
    def model_module():
        return model_new

    @staticmethod
    def get_new_fk_args_parser(d: dict) -> dict:
        if d['algo'] == 'intractable':
            optimiser_args = {}
            if d['optimiser_name'] == 'SinkhornOpt':
                optimiser_args['niter'] = d['sinkhorn_niter']
                optimiser_args['eta'] = d['sinkhorn_eta']
                optimiser_args['cache_threshold'] = d['cache_threshold']
            elif d['optimiser_name'] == 'ScipyLinOpt':
                optimiser_args['cache_threshold'] = d['cache_threshold']
            elif d['optimiser_name'] == 'AlwaysIndepOpt':
                pass
            else:
                raise ValueError
            return dict(optimiser_name=d['optimiser_name'], optimiser_args=optimiser_args, optimiser_proportion=d['optimiser_proportion'])
        elif d['algo'] == 'dummy':
            return dict(any='any')
        else:
            raise ValueError

if __name__ == '__main__':
    start_workflow('./programs/lotka_volterra/output/scape goat.csv', LotkaVolterraSmoothingExperiment)