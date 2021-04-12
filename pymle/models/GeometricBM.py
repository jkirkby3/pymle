from typing import Union
import numpy as np
from pymle.Model import Model1D


class GeometricBM(Model1D):
    """
    Model for Geometric Brownian motion
    Parameters: mu, sigma
    """

    def __init__(self):
        super().__init__(has_exact_density=True, default_sim_method="Exact")

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * x

    def exact_density(self, x0: float, xt: float, t: float) -> float:
        mu, sigma = self._params
        mu = np.log(x0) + (mu - 0.5 * sigma * sigma) * t
        sigma = sigma * np.sqrt(t)

        return np.exp(-(np.log(xt) - mu) ** 2 / (2 * sigma * sigma)) / (xt * sigma * np.sqrt(2 * np.pi))

    def exact_step(self,
                   t: float,
                   dt: float,
                   x: Union[float, np.ndarray],
                   dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Override to perform exact step (rather than Eueler) """
        sig_sq_dt = self._params[1] * np.sqrt(dt)
        drift = (self._params[0] - 0.5 * self._params[1] ** 2) * dt
        return x * np.exp(drift + sig_sq_dt * dZ)

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """ GBM is always positive """
        return True
