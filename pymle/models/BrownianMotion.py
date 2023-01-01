from typing import Union
import numpy as np
from scipy.stats import norm
from pymle.Model import Model1D


class BrownianMotion(Model1D):
    """
    Model for (drifted) Brownian Motion
    Parameters:  [mu, sigma]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = mu   (constant)
        sigma(X,t) = sigma   (constant, >0)
    """

    def __init__(self):
        super().__init__(has_exact_density=True, default_sim_method='Exact')

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (x > -10000)  # todo: reshape?

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * (x > -10000)

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        mu, sigma = self._params
        mean_ = x0 + mu * dt
        return norm.pdf(xt, loc=mean_, scale=sigma * np.sqrt(dt))

    def exact_step(self,
                   t: float,
                   dt: float,
                   x: Union[float, np.ndarray],
                   dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Simple Brownian motion can be simulated exactly """
        sig_sq_dt = self._params[1] * np.sqrt(dt)
        return x + self._params[0] * dt + sig_sq_dt * dZ

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
