from typing import Union
import numpy as np

# warning is not logged here. Perfect for clean unit test output
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

from pymle.Model import Model1D


class CEV(Model1D):
    """
    Model for CEV
    Parameters:  [kappa, mu, sigma,gamma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (mu - X)
        sigma(X,t) = sigma * (X)^gamma         (sigma>0)

    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x ** self._params[3]

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:

        kappa, mu, sigma, gamma = self._params
        x = xt

        dell = dt

        am1 = 0
        a0 = kappa * mu
        a1 = -kappa
        a2 = 0
        b0 = 0
        b1 = 0
        b2 = sigma
        b3 = gamma

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
