import os
from programs.number_of_cores_limiter import limit_number_cores; limit_number_cores(1, os)

from libs_new.utils import shut_off_numba_warnings
from programs.workflow_smoothing import SmoothingExperiment
import numpy as np
from programs.sde import lotka_volterra_sde
from programs.workflow import start_workflow

np.seterr(invalid='raise', divide='raise', over='raise')
shut_off_numba_warnings()

class LotkaVolterraSDEExperiment(SmoothingExperiment):
    @staticmethod
    def config_path() -> str:
        return './programs/sde/config/'

    @staticmethod
    def model_module():
        return lotka_volterra_sde

    @staticmethod
    def get_new_fk_args_parser(d: dict) -> dict:
        return dict()

if __name__ == '__main__':
    start_workflow('./programs/sde/output/scape goat.csv', LotkaVolterraSDEExperiment, True)