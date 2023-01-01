from typing import Union
import numpy as np

from pymle.Model import Model1D


class AitSahalia96(Model1D):
    """
    Model for Ait-Sahalia 1996, Mean Reverting
    Parameters: [alpha_0, alpha_1, alpha_2, theta_3, beta_0, beta_1, beta_2, beta_3]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = alpha[-1]X**-1+alpha[0]+alpha[1]X+alpha[2]X**2
        sigma(X,t) = sqrt(beta[0]+beta[1]X+beta[2]X**beta[3]
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] / x + self._params[1] + self._params[2] * x + self._params[3] * x * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return np.sqrt(self._params[4] + self._params[5] * x + self._params[6] * x ** self._params[7])

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        x = xt

        am1, a0, a1, a2, b0, b1, b2, b3 = self._params

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
                     -4 * am1 * b0 - 8 * am1 * b1 * x0 + 4 * a1 * b0 * x0 ** 2 - 4 * a0 * b1 * x0 ** 2 + b1 ** 2 *
                     x0 ** 2 + 8 * a2 * b0 * x0 ** 3 + 4 * a2 * b1 * x0 ** 4 - 4 * am1 * b2 * x0 ** b3 - 4 * am1 * b2 * b3 * x0 ** b3 +
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

        output = -(1 / 2) * np.log(2 * np.pi * dt) - np.log(sx) + cm1 / dt + c0 + c1 * dt

        return np.exp(output)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
