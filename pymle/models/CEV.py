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

    def AitSahalia_density1(self, x0: float, xt: float, t: float) -> float:
        b, a, c, d = self._params

        dell = t
        z = xt
        z0 = x0
        pi = np.pi
        log = np.log

        if ((a > 0) and (b > 0) and (c > 0) and (1 / 2 < d) and (d != 1)):
            if z != z0:
                output = -((z ** (1 - d) / (c * (-1 + d)) - z0 ** (1 - d) / (c * (-1 + d))) ** 2 / (2 * dell)) + \
                         (1 / (24 * c ** 2 * (2 - 9 * d + 9 * d ** 2) * ((-z ** d) * z0 + z * z0 ** d))) * (
                                 dell * z ** (-1 + 2 * d) * z0 ** (-1 + 2 * d) *
                                 (3 * c ** 4 * (-2 + d) * d * (2 + 9 * (-1 + d) * d) * z ** (1 - d) +
                                  4 * b ** 2 * (2 + 9 * (-1 + d) * d) * z ** (1 - d) * z0 ** (
                                          4 - 4 * d) + 12 * b * c ** 2 * (-1 + 2 * d) * (
                                          2 + 9 * (-1 + d) * d) * z **
                                  (1 - d) * z0 ** (2 - 2 * d) - 3 * c ** 4 * (-2 + d) * d * (
                                          2 + 9 * (-1 + d) * d) * z0 ** (1 - d) -
                                  4 * b ** 2 * (2 + 9 * (-1 + d) * d) * z ** (4 - 4 * d) * z0 ** (
                                          1 - d) - 12 * b * c ** 2 * (-1 + 2 * d) *
                                  (2 + 9 * (-1 + d) * d) * z ** (2 - 2 * d) * z0 ** (
                                          1 - d) + 12 * a ** 2 * b ** 2 * (-1 + d) *
                                  (-2 + 3 * d) * z ** (1 - 4 * d) * z0 ** (1 - 4 * d) * (
                                          z ** (3 * d) * z0 - z * z0 ** (3 * d)) -
                                  24 * a * b * (-1 + d) * (-1 + 3 * d) * z ** (1 - 4 * d) * z0 ** (1 - 4 * d) * (
                                          (-b) * z ** 2 * z0 ** (3 * d)
                                          - c ** 2 * (-2 + 3 * d) * z ** (2 * d) * z0 ** (3 * d) + z ** (
                                                  3 * d) * (
                                                  b * z0 ** 2 + c ** 2 * (-2 + 3 * d) * z0 ** (2 * d))))) + \
                         ((c - c * d) ** 3 * dell ** 2 * z ** (-2 - 3 * d) * z0 ** (-2 - 3 * d) * (
                                 (2 - 9 * d + 9 * d ** 2) * (z ** d * z0 - z * z0 ** d) ** 3 * \
                                 (-4 * b ** 2 * z ** 2 * z0 ** 2 + 3 * c ** 4 * (-2 + d) * d * z ** (
                                         2 * d) * z0 ** (2 * d)) - 12 * a ** 2 * b ** 2 * (
                                         -2 + 3 * d) * z ** 2 * z0 ** 2 * \
                                 ((1 + d) * z ** (3 * d) * z0 + (1 - 3 * d) * z ** (1 + 2 * d) * z0 ** d - (
                                         1 + d) * z * z0 ** (3 * d) + (-1 + 3 * d) * z ** d * z0 ** (
                                          1 + 2 * d)) - 24 * a * b * \
                                 (-1 + 3 * d) * z * z0 * (
                                         (-c ** 2) * d * (-2 + 3 * d) * z ** (3 * d) * z0 ** (2 + 2 * d) - b * (
                                         -2 + 3 * d) * z ** (2 + d) * z0 ** (
                                                 2 + 2 * d) + b * d * z ** 3 * z0 ** \
                                         (1 + 3 * d) - c ** 2 * (4 - 8 * d + 3 * d ** 2) * z ** (
                                                 1 + 2 * d) * z0 ** (1 + 3 * d) + (-2 + 3 * d) * z ** (
                                                 2 + 2 * d) * z0 ** d * \
                                         (b * z0 ** 2 + c ** 2 * d * z0 ** (2 * d)) + z ** (1 + 3 * d) * z0 * (
                                                 (-b) * d * z0 ** 2 + c ** 2 * (
                                                 4 - 8 * d + 3 * d ** 2) * z0 ** \
                                                 (2 * d))))) / (
                                 48 * c ** 3 * (-1 + d) * (2 + 9 * (-1 + d) * d) * (
                                 z ** (1 - d) - z0 ** (1 - d)) ** 3) - (1 / 2) * \
                         log(2 * dell * pi) - log(c * z ** d) + (1 / (2 * c ** 2 * (1 - 3 * d + 2 * d ** 2))) * (
                                 (b * (-2 * a * (-1 + d) * z * z0 ** (2 * d) + (-1 + 2 * d) * z ** 2 * z0 ** (2 * d) \
                                       - z ** (2 * d) * z0 * (2 * a - 2 * a * d - z0 + 2 * d * z0)) - c ** 2 * d * (
                                          1 - 3 * d + 2 * d ** 2) * z ** (2 * d) * \
                                  z0 ** (2 * d) * log(z) + c ** 2 * d * (1 - 3 * d + 2 * d ** 2) * z ** (
                                          2 * d) * z0 ** (2 * d) * log(z0)) / (z ** (2 * d) * z0 ** (2 * d)))

            else:
                output = (1 / (48 * z0 ** 4)) * (
                        dell ** 2 * (-4 * a ** 2 * b ** 2 * d * (1 + d) * z0 ** 2 + 4 * a * b ** 2 * \
                                     d * (-1 + 2 * d) * z0 ** 3 - 4 * b ** 2 * (
                                             -1 + d) ** 2 * z0 ** 4 + 3 * c ** 4 * (-2 + d) * (
                                             -1 + d) ** 2 * d * z0 ** (4 * d) - 4 * a * b * c ** 2 * (
                                             -2 + d) * d * z0 ** (1 + 2 * d))) + \
                         (1 / (8 * c ** 2)) * ((dell * (
                        -4 * a ** 2 * b ** 2 * z0 ** 2 + 8 * a * b ** 2 * z0 ** 3 - 4 * b ** 2 * z0 ** 4 + c ** 4 * (
                        -2 + d) * d * \
                        z0 ** (4 * d) + 8 * a * b * c ** 2 * d * z0 ** (1 + 2 * d) - 4 * b * c ** 2 * (
                                -1 + 2 * d) * z0 ** (2 + 2 * d))) / z0 ** (2 * (1 + d))) - (1 / 2) * \
                         log(2 * dell * pi) - log(c * z0 ** d)

        elif ((a > 0) and (b > 0) and (c > 0) and (d == 1)):
            if z != z0:
                output = (-(1 / 2)) * log(2 * dell * pi) - log(c * z) - (log(z) / c - log(z0) / c) ** 2 / (2 * dell) - \
                         (2 * a * b * (1 / z - 1 / z0) + (2 * b + c ** 2) * log(z) - (2 * b + c ** 2) * log(z0)) / (
                                 2 * c ** 2) + \
                         (dell * (2 * a * b * (z - z0) * ((-a) * b * z0 + z * ((-a) * b + 4 * (b + c ** 2) * z0)) - \
                                  (2 * b + c ** 2) ** 2 * z ** 2 * z0 ** 2 * log(z) + (
                                          2 * b + c ** 2) ** 2 * z ** 2 * z0 ** 2 * log(z0))) / (
                                 8 * c ** 2 * z ** 2 * z0 ** 2 * (log(z) - log(z0))) \
                         + (1 / (4 * z ** 2 * z0 ** 2 * (log(z) - log(z0)) ** 3)) * (
                                 a * b * dell ** 2 * (2 * (b + c ** 2) * z ** 2 * z0 * \
                                                      (-2 + log(z) - log(z0)) - a * b * z ** 2 * (
                                                              -1 + log(z) - log(z0)) + 2 * (
                                                              b + c ** 2) * z * z0 ** 2 * (2 + \
                                                                                           log(z) - log(
                                             z0)) + a * b * z0 ** 2 * (-1 - log(z) + log(z0))))
            else:
                output = (a * b * dell ** 2 * (-2 * a * b + (b + c ** 2) * z0)) / (12 * z0 ** 2) - (dell * (
                        4 * a ** 2 * b ** 2 - 8 * a * b * (b + c ** 2) * z0 + (2 * b + c ** 2) ** 2 * z0 ** 2)) / (
                                 8 * c ** 2 * z0 ** 2) \
                         - (1 / 2) * log(2 * dell * pi) - log(c * z0)

        else:
            output = -np.inf  # if parameter restrictions are not verified, send f to +infinity

        return np.exp(output)

    def AitSahalia_density(self, x0: float, xt: float, t: float) -> float:

        kappa, mu, sigma, gamma = self._params
        x = xt

        dell = t

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
