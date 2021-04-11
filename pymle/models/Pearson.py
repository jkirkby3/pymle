from typing import Union
import numpy as np

from pymle.Model import Model1D


class Pearson(Model1D):
    """
    Model for Pearson process
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return -  self._params[0] * (x - self._params[1])

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        a = self._params[2]
        b = self._params[3]
        c = self._params[4]
        return np.sqrt(2 * self._params[0] * (a * x * x + b * x + c))
