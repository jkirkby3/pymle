from typing import Union
import numpy as np
from pymle.Model import Model1D


class GeometricBM(Model1D):
    """
    Model for Geometric Brownian motion
    Parameters: [mu, sigma]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = mu*X   (constant)
        sigma(X,t) = sigma*X   (constant, >0)
    """

    def __init__(self):
        super().__init__(has_exact_density=True, default_sim_method="Exact")

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * x

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        mu, sigma = self._params
        mu = np.log(x0) + (mu - 0.5 * sigma * sigma) * dt
        sigma = sigma * np.sqrt(dt)

        return np.exp(-(np.log(xt) - mu) ** 2 / (2 * sigma * sigma)) / (xt * sigma * np.sqrt(2 * np.pi))

    def AitSahalia_density(self, x0: float, xt: float, t0: float,  dt: float) -> float:
        a = 0
        b, d = self._params
        log = np.log
        exp = np.exp
        pi = np.pi

        x = xt
        dell = dt

        y = log(x) / d
        y0 = log(x0) / d

        E = exp(1)

        sx = d * x

        cYm1 = (-(1 / 2)) * (y - y0) ** 2
        cY0 = (E ** ((-d) * y) - E ** ((-d) * y0)) * (-(a / d ** 2)) + (y - y0) * (b / d - d / 2)

        if (y != y0).all():
            cY1 = (a ** 2 / (4 * d ** 3)) * ((E ** (-2 * d * y) - E ** (-2 * d * y0)) / (y - y0)) + \
                  ((a * b) / d ** 3 - a / d) * ((E ** ((-d) * y) - E ** ((-d) * y0)) / (y - y0)) - \
                  (2 * b - d ** 2) ** 2 / (8 * d ** 2)
            cY2 = (-(a ** 2 / (2 * d ** 3))) * ((E ** (-2 * d * y) - E ** (-2 * d * y0)) / (y - y0) ** 3) + \
                  ((2 * a) / d - (2 * a * b) / d ** 3) * ((E ** ((-d) * y) - E ** ((-d) * y0)) / (y - y0) ** 3) + \
                  (-(a ** 2 / (2 * d ** 2))) * ((E ** (-2 * d * y) + E ** (-2 * d * y0)) / (y - y0) ** 2) + \
                  (a - (a * b) / d ** 2) * ((E ** ((-d) * y) + E ** ((-d) * y0)) / (y - y0) ** 2)
        else:
            cY1 = (-4 * a ** 2 - 8 * a * (b - d ** 2) * E ** (d * y) - (-2 * b + d ** 2) ** 2 * E ** (2 * d * y)) / (
                        E ** (2 * d * y) * (8 * d ** 2))
            cY2 = ((1 / 6) * a * (-2 * a + (-b + d ** 2) * E ** (d * y))) / E ** (2 * d * y)

        output = (-(1 / 2)) * log(2 * pi * dell) - log(sx) + cYm1 / dell + cY0 + cY1 * dell + cY2 * (dell ** 2 / 2)

        return np.exp(output)

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

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
