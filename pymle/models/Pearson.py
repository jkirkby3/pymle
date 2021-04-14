from typing import Union
import numpy as np

from pymle.Model import Model1D


class Pearson(Model1D):
    """
    Model for Pearson process
    Parameters: [kappa, mu, a, b, c]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (mu - X)
        sigma(X,t) = sqrt(2*kappa*(a*X^2 + b*X + c))
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (self._params[1] - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        a = self._params[2]
        b = self._params[3]
        c = self._params[4]
        return np.sqrt(2 * self._params[0] * (a * x * x + b * x + c))

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
