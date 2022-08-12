from typing import Union
import numpy as np


from pymle.Model import Model1D


class FellerRoot(Model1D):
    """
    Model for Feller Square Root Process
    Parameters: [theta_1, theta_2, theta_3]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = X*(theta_1 - X*(theta_3^3 - theta_1*theta_2))
        sigma(X,t) = theta_3 * X^(3/2)
    """

    def __init__(self):
        super().__init__(has_exact_density=False)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        c = self._params[2]**3 - self._params[0]*self._params[1]
        return x * (self._params[0] - x*c)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * x**1.5

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
