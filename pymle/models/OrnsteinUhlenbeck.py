from typing import Union
import numpy as np
from scipy.stats import norm

from pymle.Model import Model1D


class OrnsteinUhlenbeck(Model1D):
    """
    Model for OU (ornstein-uhlenbeck):
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (mu - X)
        sigma(X,t) = sigma * X
    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * (x > -10000)

    def exact_density(self, x0: float, xt: float, t: float) -> float:
        kappa, theta, sigma = self._params
        mu = theta + (x0 - theta) * np.exp(-kappa * t)
        # mu = X0*np.exp(-kappa*t) + theta*(1 - np.exp(-kappa*t))
        var = (1 - np.exp(-2 * kappa * t)) * (sigma * sigma / (2 * kappa))
        return norm.pdf(xt, loc=mu, scale=np.sqrt(var))

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
