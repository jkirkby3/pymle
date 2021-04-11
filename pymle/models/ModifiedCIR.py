from typing import Union
import numpy as np

from pymle.Model import Model1D


class ModifiedCIR(Model1D):
    """
    Model for Modified CIR process
    """

    def __init__(self):
        super().__init__()

    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return -self._params[0] * x

    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._params[1] * np.sqrt(1 + x * x)
