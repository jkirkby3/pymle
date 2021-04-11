from typing import Union
import numpy as np

from pymle.Model import Model1D


class Jacobi(Model1D):
    """
    Generator for Jacobi process
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return -self._params[0] * (x - 0.5)

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return np.sqrt(self._params[0] * np.abs(x * (1 - x)))
