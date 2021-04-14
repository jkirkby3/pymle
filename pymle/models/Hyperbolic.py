from typing import Union
import numpy as np

from pymle.Model import Model1D


class Hyperbolic(Model1D):
    """
    Model for Hyperbolic process
    Parameters: [kappa, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = -kappa*X / sqrt(1 + X^2)
        sigma(X,t) = sigma
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return -self._params[0] * x / (np.sqrt(1 + x * x))

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * (x > -10000)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
