from typing import Union
import numpy as np

from pymle.Model import Model1D


class Threehalf(Model1D):
    """
    Model for 3/2 Process
    Parameters: [kappa,mu,sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa*X*(mu-X)
        sigma(X,t) = sigmaX^3/2
    """

    def __init__(self):
        super().__init__(has_exact_density=False)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return x * self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x ** 1.5

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        x = xt

        dell = dt

        kappa, mu, sigma = self._params

        am1 = 0
        a0 = 0
        a1 = kappa * mu
        a2 = -kappa

        b0 = 0
        b1 = 0
        b2 = sigma ** 2
        b3 = 3

        sx = np.sqrt(b0 + b1 * x + b2 * x ** b3)

        cm1 = -(((x - x0) ** 4 * (15 * b1 ** 2 * x0 ** 2 - 2 * b1 * b2 * b3 * (-19 + 4 * b3) * x0 ** (1 + b3) +
                                  b2 * b3 * x0 ** b3 * (-8 * b0 * (-1 + b3) + b2 * (8 + 7 * b3) * x0 ** b3))) / (
                        96 * x0 ** 2 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 3)) + \
              ((x - x0) ** 3 * (6 * b1 + 6 * b2 * b3 * x0 ** (-1 + b3))) / (
                      24 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2) - (x - x0) ** 2 / (
                      2 * (b0 + b1 * x0 + b2 * x0 ** b3))

        c0 = ((x - x0) * (4 * am1 + 4 * a0 * x0 - b1 * x0 + 4 * a1 * x0 ** 2 + 4 * a2 * x0 ** 3 -
                          b2 * b3 * x0 ** b3)) / (4 * x0 * (b0 + b1 * x0 + b2 * x0 ** b3)) + (
                     1 / (8 * x0 ** 2 * (b0 + b1 * x0 + b2 * x0 ** b3) ** 2)) * \
             ((x - x0) ** 2 * (
                     -4 * am1 * b0 - 8 * am1 * b1 * x0 + 4 * a1 * b0 * x0 ** 2 - 4 * a0 * b1 * x0 ** 2 + b1 ** 2
                     * x0 ** 2 + 8 * a2 * b0 * x0 ** 3 + 4 * a2 * b1 * x0 ** 4 - 4 * am1 * b2 * x0 ** b3 -
                     4 * am1 * b2 * b3 * x0 ** b3 +
                     b0 * b2 * b3 * x0 ** b3 - b0 * b2 * b3 ** 2 * x0 ** b3 + b2 ** 2 * b3 * x0 ** (2 * b3) -
                     4 * a0 * b2 * b3 * x0 ** (1 + b3) + 3 * b1 * b2 * b3 * x0 ** (
                             1 + b3) - b1 * b2 * b3 ** 2 * x0 ** (1 + b3) + 4 * a1 * b2 * x0 ** (2 + b3) -
                     4 * a1 * b2 * b3 * x0 ** (2 + b3) + 8 * a2 * b2 * x0 ** (3 + b3) - 4 * a2 * b2 * b3 * x0 ** (
                             3 + b3)))
        c1 = (1 / 8) * (-4 * (a1 - am1 / x0 ** 2 + 2 * a2 * x0) -
                        (b1 + b2 * b3 * x0 ** (-1 + b3)) ** 2 / (4 * (b0 + b1 * x0 + b2 * x0 ** b3)) +
                        (4 * (b1 + b2 * b3 * x0 ** (-1 + b3)) * (a0 + am1 / x0 + x0 * (a1 + a2 * x0))) / (
                                b0 + b1 * x0 + b2 * x0 ** b3) - (4 * (a0 + am1 / x0 + x0 * (a1 + a2 * x0)) ** 2) / (
                                b0 + b1 * x0 + b2 * x0 ** b3) +
                        ((-b1 ** 2) * x0 ** 2 + 2 * b1 * b2 * (-2 + b3) * b3 * x0 ** (1 + b3) + b2 * b3 * x0 ** b3 * (
                                2 * b0 * (-1 + b3) +
                                b2 * (-2 + b3) * x0 ** b3)) / (2 * x0 ** 2 * (b0 + b1 * x0 + b2 * x0 ** b3)))

        output = -(1 / 2) * np.log(2 * np.pi * dell) - np.log(sx) + cm1 / dell + c0 + c1 * dell

        return np.exp(output)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
