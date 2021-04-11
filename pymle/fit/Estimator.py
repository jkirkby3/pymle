from abc import ABC
import numpy as np
from typing import List, Tuple
from pymle.Model import Model1D


class Estimator(ABC):
    def __init__(self,
                 sample: np.ndarray,
                 dt: float,
                 model: Model1D,
                 param_bounds: List[Tuple]):
        """
        Abstract base class for Diffusion Estimator
        :param sample: np.ndarray, a univariate time series sample from the diffusion (ascending order of time)
        :param dt: float, time step (time between diffusion steps, assumed uniform sampling frequency)
        :param model: the diffusion model. This defines the parametric family/model,
            the parameters of which will be fitted during estimation
        :param param_bounds: List[Tuple], a list of tuples, each tuple provides (lower,upper) bounds on the parameters,
            in order of the parameters as they are defined in the generator
        """
        self._sample = sample
        self._param_bounds = param_bounds
        self._dt = dt
        self._model = model

    def estimate_params(self, params0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: (array, float), the estimated params and final likelihood
        """
        raise NotImplementedError
