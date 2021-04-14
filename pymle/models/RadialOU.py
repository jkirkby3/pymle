from typing import Union
import numpy as np


from pymle.Model import Model1D


class RadialOU(Model1D):
    """
    Model for Radial OU (ornstein-uhlenbeck):
    Parameters: [kappa, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa / X - X
        sigma(X,t) = sigma
    """

    def __init__(self):
        super().__init__(has_exact_density=False)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] / x - x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[2] * (x > -10000)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
