import numpy as np
from programs.brownian.model import LinearGaussianSmoothing

def corr_matrix(d: int, rho: float) -> np.ndarray:
    """
    Create the matrix whose all off-diag elements are `rho` and diag elements are 1.
    """
    return np.zeros((d, d)) + rho + np.identity(d) * (1 - rho)

def get_model_parameters(model: LinearGaussianSmoothing) -> dict:
    return dict(covX=model.covX, covY=model.covY, data=model.data, F=model.F, G=model.G, mu0=model.mu0, cov0=model.cov0)