from typing import Union
import numpy as np

from pymle.Model import Model1D


class Jacobi(Model1D):
    """
    Model for Jacobi process
    Parameters: [kappa]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (0.5 - X)
        sigma(X,t) = sqrt(kappa*X*(1-X))
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (0.5 - x)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return np.sqrt(self._params[0] * np.abs(x * (1 - x)))

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def _set_is_positive(self, params: np.ndarray) -> bool:
        return True  # Process is always positive
