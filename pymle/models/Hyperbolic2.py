from typing import Union
import numpy as np

from pymle.Model import Model1D


class Hyperbolic2(Model1D):
    """
    Model for Hyperbolic process II
    Parameters: [beta,gamma,delta,mu,sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = sigma**2/2(beta-gamma*X/sqrt(delta**2+(X-mu)))
        sigma(X,t) = sigma
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[4]**2/2*(self._params[0]-self._params[1]*x/np.sqrt(self._params[2]**2+(x-self._params[3])**2))

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[4]

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return 0.
