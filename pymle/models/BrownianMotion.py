from typing import Union
import numpy as np
from scipy.stats import norm
from pymle.Model import Model1D


class BrownianMotion(Model1D):
    """
    Model for (drifted) Brownian Motion:

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = mu   (constant)
        sigma(X,t) = sigma   (constant, >0)
    """

    def __init__(self):
        super().__init__(has_exact_density=True)

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[0] * (x > -10000)  # todo: reshape?

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * (x > -10000)

    def exact_density(self, x0: float, xt: float, t: float) -> float:
        mu, sigma = self._params
        mean_ = x0 + mu * t
        return norm.pdf(xt, loc=mean_, scale=sigma * np.sqrt(t))