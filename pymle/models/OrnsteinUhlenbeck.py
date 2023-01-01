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

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        kappa, theta, sigma = self._params
        mu = theta + (x0 - theta) * np.exp(-kappa * dt)
        # mu = X0*np.exp(-kappa*t) + theta*(1 - np.exp(-kappa*t))
        var = (1 - np.exp(-2 * kappa * dt)) * (sigma * sigma / (2 * kappa))
        return norm.pdf(xt, loc=mu, scale=np.sqrt(var))

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        kappa, alpha, eta = self._params
        m = 1
        x = xt

        output = (-m / 2) * np.log(2 * np.pi * dt) - np.log(eta) - ((x - x0) ** 2 / (2 * eta ** 2)) / dt \
                 + ((-(x ** 2 / 2) + x0 ** 2 / 2 + x * alpha - x0 * alpha) * kappa) / eta ** 2 \
                 - ((1 / (6 * eta ** 2)) * (kappa * (-3 * eta ** 2 + (
                x ** 2 + x0 ** 2 + x * (x0 - 3 * alpha) - 3 * x0 * alpha + 3 * alpha ** 2) * kappa))) * dt \
                 - (1 / 2) * (kappa ** 2 / 6) * dt ** 2 \
                 + (1 / 6) * ((4 * x ** 2 + 7 * x * x0 + 4 * x0 ** 2 - 15 * x * alpha
                               - 15 * x0 * alpha + 15 * alpha ** 2) * kappa ** 4) / (
                         60 * eta ** 2) * dt ** 3
        return np.exp(output)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
