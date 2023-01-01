from typing import Union
import numpy as np
from pymle.Model import Model1D


class LinearSDE1(Model1D):
    """
    Model for linear SDE 1
    Parameters: [a,b, c,d]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = (a+bX)
        sigma(X,t) = (c+dX)   (c,d CANNOT BE ZEROs)
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] + self._params[1] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        a, b, f, d = self._params

        x = xt
        dell = dt

        sx = f + d * x
        y = np.log(1 + (d * x) / f) / d
        y0 = np.log(1 + (d * x0) / f) / d

        E = np.exp(1)

        cYm1 = (-(1 / 2)) * (y - y0) ** 2

        cY0 = (E ** ((-d) * y) - E ** ((-d) * y0)) * ((b * f - a * d) / (d ** 2 * f)) + (y - y0) * (
                    (2 * b - d ** 2) / (2 * d))

        if (y != y0).all():
            cY1 = (1 / (2 * d)) * (b ** 2 / (2 * d ** 2) + a ** 2 / (2 * f ** 2) - (a * b) / (d * f)) * (
                        (E ** (-2 * d * y) - E ** (-2 * d * y0)) / (y - y0)) + \
                  ((a * b) / (d ** 2 * f) - b ** 2 / d ** 3 + b / d - a / f) * (
                              (E ** ((-d) * y) - E ** ((-d) * y0)) / (y - y0)) - \
                  (2 * b - d ** 2) ** 2 / (8 * d ** 2)

        else:
            cY1 = -((a * d - b * f) ** 2 / (E ** (2 * d * y) * (2 * f ** 2 * d ** 2))) + \
                  ((b - d ** 2) * ((-a) * d + b * f)) / (E ** (d * y) * (f * d ** 2)) - (2 * b - d ** 2) ** 2 / (
                              8 * d ** 2)

        output = (-(1 / 2)) * np.log(2 * np.pi * dell) - np.log(sx) + cYm1 / dell + cY0 + cY1 * dell

        return np.exp(output)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
