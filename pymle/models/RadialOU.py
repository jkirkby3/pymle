from typing import Union
import numpy as np

from pymle.Model import Model1D


class RadialOU(Model1D):
    """
    Model for Radial OU (ornstein-uhlenbeck):
    Parameters: [kappa, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa / X - X
        sigma(X,t) = sigma
    """

    def __init__(self):
        super().__init__(has_exact_density=False)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] / x - x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1]

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        kappa, sigma = self._params
        x = xt

        dell = dt

        am1 = kappa
        a0 = 0
        a1 = -1
        a2 = 0
        b0 = sigma
        b1 = 0
        b2 = 0
        b3 = 0

        sx = b0 + b1 * x + b2 * x ** b3
        cm1 = -((x - x0) ** 2 / (2 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2)) + (
                    (x - x0) ** 3 * (b1 + b2 * b3 * x0 ** (-1 + b3))) / (2 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 3) + \
              ((x - x0) ** 4 * (
                          -11 * (b1 + b2 * b3 * x0 ** (-1 + b3)) ** 2 + 4 * b2 * (-1 + b3) * b3 * x0 ** (-2 + b3) *
                          (b0 + b1 * x0 + b2 * x0 ** b3))) / (24 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 4)

        c0 = ((x - x0) * ((-(b1 + b2 * b3 * x0 ** (-1 + b3))) * (b0 + b1 * x0 + b2 * x0 ** b3) + 2 *
                          (a0 + am1 / x0 + x0 * (a1 + a2 * x0)))) / (2 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2) + (
                         (x - x0) ** 2 *
                         ((-b2) * (-1 + b3) * b3 * x0 ** (-2 + b3) * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2 - 4 *
                          (b1 + b2 * b3 * x0 ** (-1 + b3)) * (a0 + am1 / x0 + x0 * (a1 + a2 * x0)) +
                          (b0 + b1 * x0 + b2 * x0 ** b3) * (2 * (a1 - am1 / x0 ** 2 + 2 * a2 * x0) +
                                                            (b1 + b2 * b3 * x0 ** (-1 + b3)) ** 2))) / (
                         4 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 3)

        c1 = (-(1 / (8 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2))) * (
                    -8 * (b1 + b2 * b3 * x0 ** (-1 + b3)) * (b0 + b1 * x0 + b2 * x0 ** b3) *
                    (a0 + am1 / x0 + x0 * (a1 + a2 * x0)) + 4 * (a0 + am1 / x0 + x0 * (a1 + a2 * x0)) ** 2 +
                    (b0 + b1 * x0 + b2 * x0 ** b3) ** 2 * (
                                4 * (a1 - am1 / x0 ** 2 + 2 * a2 * x0) + (b1 + b2 * b3 * x0 ** (-1 + b3)) ** 2 -
                                2 * b2 * (-1 + b3) * b3 * x0 ** (-2 + b3) * (b0 + b1 * x0 + b2 * x0 ** b3)))

        output = -(1 / 2) * np.log(2 * np.pi * dell) - np.log(sx) + cm1 / dell + c0 + c1 * dell

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
