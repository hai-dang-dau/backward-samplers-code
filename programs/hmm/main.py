from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
import numpy as np
from libs_new.utils import shut_off_numba_warnings
from programs.workflow_smoothing import SmoothingExperiment
from programs.workflow import start_workflow
from libs_new.hmm_new import HMM

np.seterr(divide='raise', invalid='raise')
shut_off_numba_warnings()

class HMMSmoothingExperiment(SmoothingExperiment):
    # tested 091121
    # implies a tested general smoothing runner machinery
    @staticmethod
    def config_path() -> str:
        return './programs/hmm/config/'

    @staticmethod
    def model_module():
        return HMM

    @staticmethod
    def get_new_fk_args_parser(d: dict) -> dict:
        if d['smooth_mode'] == 'intractable':
            return dict(coupled=True)
        elif d['smooth_mode'] == 'online':
            return dict()
        else:
            raise ValueError

if __name__ == '__main__':
    start_workflow('./programs/hmm/output/scape goat.csv', HMMSmoothingExperiment, verbose=True)