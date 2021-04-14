from typing import Union
import numpy as np

from pymle.Model import Model1D


class CKLS(Model1D):
    """
    Model for CKLS
    Parameters: [theta_1, theta_2, theta_3, theta_4]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = (theta_1 + theta_2*X)
        sigma(X,t) = theta_3 * X^(theta_4)
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] + self._params[1] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x ** self._params[3]

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def _set_is_positive(self, params: np.ndarray) -> bool:
        return params[0] > 0 and params[1] > 0 and params[3] > 0.5
