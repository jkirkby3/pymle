from typing import Union
import numpy as np

# warning is not logged here. Perfect for clean unit test output
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

from scipy.special import ive

from pymle.Model import Model1D


class CIR(Model1D):
    """
    Model for CIR (cox-ingersol-ross)
    Parameters:  [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (mu - X)
        sigma(X,t) = sigma * sqrt(X)         (sigma>0)

    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * np.sqrt(x)

    def exact_density(self, x0: float, xt: float, t: float) -> float:
        kappa, mu, sigma = self._params
        theta1 = kappa * mu
        theta2 = kappa
        theta3 = sigma

        et = np.exp(-theta2 * t)
        c = 2 * theta2 / (theta3 ** 2 * (1 - et))
        u = c * x0 * et
        v = c * xt
        q = 2 * theta1 / theta3 ** 2 - 1

        z = 2 * np.sqrt(u * v)  # Note: we apply exponentail scaling trick
        p = c * np.exp(-(u + v) + np.abs(z)) * (v / u) ** (q / 2)

        z = 2 * np.sqrt(u * v)
        p *= ive(q, z)
        return p

    def AitSahalia_density1(self, x0: float, xt: float, t: float) -> float:
        kappa, alpha, sigma = self._params

        # m = 1
        dell = t
        x = xt

        output = (1 / (26542080 * np.sqrt(dell) * np.sqrt(2 * np.pi) * sigma ** 6 * x ** 2 * x0 ** 2)) * \
                 (np.exp(((-(2 + dell * kappa)) * x + 4 * np.sqrt(x) * np.sqrt(x0) + (-2 + dell * kappa) * x0) / (
                         dell * sigma ** 2)) *
                  (np.sqrt(x) / sigma) ** (-(1 / 2) + (2 * alpha * kappa) / sigma ** 2) * (np.sqrt(x0) / sigma) ** (
                          3 / 2 - (2 * alpha * kappa) / sigma ** 2) * (
                          26542080 * sigma ** 6 * x ** (3 / 2) * x0 ** (
                          3 / 2) - 276480 * dell * sigma ** 4 * x * x0 * (
                                  48 * alpha ** 2 * kappa ** 2 - 48 * alpha * kappa * sigma ** 2 +
                                  9 * sigma ** 4 + 16 * kappa ** 2 * x ** (3 / 2) * np.sqrt(
                              x0) + 16 * kappa ** 2 * x * x0 + 16 * kappa ** 2 * np.sqrt(x) * np.sqrt(x0) * (
                                          -6 * alpha + x0)) + 1440 * dell ** 2 * sigma ** 2 * np.sqrt(
                      x) * np.sqrt(x0) * \
                          (9 * (
                                  256 * alpha ** 4 * kappa ** 4 - 512 * alpha ** 3 * kappa ** 3 * sigma ** 2 + 224 * alpha ** 2 * kappa ** 2 * sigma ** 4 + 32 * alpha * kappa * sigma ** 6 - 15 * sigma ** 8) + 256 * kappa ** 4 * x ** 3 * x0 + 512 * kappa ** 4 * \
                           x ** (5 / 2) * x0 ** (3 / 2) + 96 * kappa ** 2 * (
                                   16 * alpha ** 2 * kappa ** 2 - 16 * alpha * kappa * sigma ** 2 + 3 * sigma ** 4) * np.sqrt(
                                      x) * np.sqrt(x0) * (-6 * alpha + x0) + 768 * kappa ** 4 * x ** 2 * x0 * \
                           (-4 * alpha + x0) + 32 * kappa ** 2 * x * x0 * (
                                   336 * alpha ** 2 * kappa ** 2 - 48 * alpha * kappa * sigma ** 2 - 3 * sigma ** 4 - 96 * alpha * kappa ** 2 * x0 + 8 * kappa ** 2 * x0 ** 2) + 32 * kappa ** 2 * \
                           x ** (3 / 2) * np.sqrt(x0) * (
                                   48 * alpha ** 2 * kappa ** 2 - 48 * alpha * kappa * sigma ** 2 + 9 * sigma ** 4 - 96 * alpha * kappa ** 2 * x0 + 16 * kappa ** 2 * x0 ** 2)) - \
                          dell ** 3 * (135 * (
                          4096 * alpha ** 6 * kappa ** 6 - 12288 * alpha ** 5 * kappa ** 5 * sigma ** 2 + 6400 * alpha ** 4 * kappa ** 4 * sigma ** 4 + 7680 * alpha ** 3 * kappa ** 3 * sigma ** 6 - 5456 * alpha ** 2 * kappa ** 2 * sigma ** 8 - 432 * alpha * kappa * sigma ** 10 + 315 * sigma ** 12) + \
                                       20480 * kappa ** 6 * x ** (9 / 2) * x0 ** (
                                               3 / 2) + 61440 * kappa ** 6 * x ** 4 * x0 ** 2 + 2160 * kappa ** 2 * (
                                               256 * alpha ** 4 * kappa ** 4 - 512 * alpha ** 3 * kappa ** 3 * sigma ** 2 + 224 * alpha ** 2 * kappa ** 2 * sigma ** 4 + 32 * alpha * kappa * sigma ** 6 - 15 * sigma ** 8) * \
                                       np.sqrt(x) * np.sqrt(x0) * (-6 * alpha + x0) + 122880 * kappa ** 6 * x ** (
                                               7 / 2) * x0 ** (3 / 2) * (-3 * alpha + x0) + 720 * kappa ** 2 * (
                                               16 * alpha ** 2 * kappa ** 2 - 16 * alpha * kappa * sigma ** 2 + 3 * sigma ** 4) * x * x0 * (
                                               624 * alpha ** 2 * kappa ** 2 - 48 * \
                                               alpha * kappa * sigma ** 2 + 9 * sigma ** 4 - 192 * alpha * kappa ** 2 * x0 + 16 * kappa ** 2 * x0 ** 2) + 1536 * kappa ** 4 * x ** (
                                               5 / 2) * x0 ** (3 / 2) * (
                                               1680 * alpha ** 2 * kappa ** 2 - 240 * alpha * kappa * sigma ** 2 - 63 * sigma ** 4 - 720 * alpha * kappa ** 2 * x0 + \
                                               80 * kappa ** 2 * x0 ** 2) + 1280 * kappa ** 4 * x ** 3 * x0 * (
                                               144 * alpha ** 2 * kappa ** 2 - 144 * alpha * kappa * sigma ** 2 + 27 * sigma ** 4 - 576 * alpha * kappa ** 2 * x0 + 112 * kappa ** 2 * x0 ** 2) + 768 * kappa ** 4 * x ** 2 * x0 * \
                                       (-180 * alpha * (
                                               16 * alpha ** 2 * kappa ** 2 - 16 * alpha * kappa * sigma ** 2 + 3 * sigma ** 4) + 9 * (
                                                400 * alpha ** 2 * kappa ** 2 - 80 * alpha * kappa * sigma ** 2 - 17 * sigma ** 4) * x0 - 960 * alpha * kappa ** 2 * x0 ** 2 + 80 * kappa ** 2 * x0 ** 3) + 16 * kappa ** 2 * \
                                       x ** (3 / 2) * np.sqrt(x0) * (135 * (
                                  256 * alpha ** 4 * kappa ** 4 - 512 * alpha ** 3 * kappa ** 3 * sigma ** 2 + 224 * alpha ** 2 * kappa ** 2 * sigma ** 4 + 32 * alpha * kappa * sigma ** 6 - 15 * sigma ** 8) - 8640 * alpha * kappa ** 2 * \
                                                                     (
                                                                             48 * alpha ** 2 * kappa ** 2 - 16 * alpha * kappa * sigma ** 2 - sigma ** 4) * x0 + 288 * kappa ** 2 * (
                                                                             560 * alpha ** 2 * kappa ** 2 - 80 * alpha * kappa * sigma ** 2 - 21 * sigma ** 4) * x0 ** 2 - 23040 * alpha * kappa ** 4 * x0 ** 3 + 1280 * kappa ** 4 * x0 ** 4))))
        return output

    def AitSahalia_density(self, x0: float, xt: float, t: float) -> float:
        x = xt

        dell = t

        kappa, mu, sigma = self._params

        am1 = 0
        a0 = kappa * mu
        a1 = -kappa
        a2 = 0

        b0 = 0
        b1 = 0
        b2 = sigma ** 2
        b3 = 1

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
                     * x0 ** 2 + 8 * a2 * b0 * x0 ** 3 + 4 * a2 * b1 * x0 ** 4 - 4 * am1 * b2 * x0 ** b3 - 4 * am1 * b2 * b3 * x0 ** b3 +
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

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """ CIR is always non-negative """
        return True

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
