from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
import numpy as np
from libs_new.utils import shut_off_numba_warnings
from programs.workflow_smoothing import SmoothingExperiment
from programs.brownian import model
from programs.workflow import start_workflow

np.seterr(divide='raise', invalid='raise')
shut_off_numba_warnings()

class GaussianSmoothingExperiment(SmoothingExperiment):
    # tested
    @staticmethod
    def model_module():
        return model

    @staticmethod
    def config_path() -> str:
        return './programs/brownian/config/'

    @staticmethod
    def get_new_fk_args_parser(d: dict) -> dict:
        if d['smooth_mode'] in ['online', 'offline']:
            return SmoothingExperiment.get_new_fk_args_parser(d)
        elif d['smooth_mode'] == 'intractable':
            return dict(deflate_coupling_ratio=d['deflate_coupling_ratio'])
        else:
            raise ValueError

if __name__ == '__main__':
    start_workflow('./programs/brownian/output/run_test.csv', GaussianSmoothingExperiment, verbose=True)