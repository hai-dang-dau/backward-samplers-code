from programs.number_of_cores_limiter import limit_number_cores
import os; limit_number_cores(1, os)
import numpy as np
from libs_new.utils import shut_off_numba_warnings
from programs.workflow_smoothing import SmoothingExperiment
from programs.constrained_brownian import model
from programs.workflow import start_workflow

np.seterr(divide='raise', invalid='raise')
shut_off_numba_warnings()

class ConstrainedBrownianSmoothingExperiment(SmoothingExperiment):
    # tested
    @staticmethod
    def model_module():
        return model

    @staticmethod
    def config_path() -> str:
        return './programs/constrained_brownian/config/'

if __name__ == '__main__':
    start_workflow('./programs/constrained_brownian/output/run_full.csv', ConstrainedBrownianSmoothingExperiment, verbose=True)