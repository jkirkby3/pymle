from typing import Union
import numpy as np
from pymle.Model import Model1D


class LinearSDE2(Model1D):
    """
    Model for linear SDE 2
    Parameters: [a,b, c]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = (a+bX)
        sigma(X,t) = (cX)   (c CANNOT BE ZERO)
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] + self._params[1] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        a, b, d = self._params
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

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
