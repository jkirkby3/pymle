from abc import ABC
import numpy as np
from typing import List, Tuple
from pymle.Model import Model1D


class EstimatedResult(object):
    def __init__(self,
                 params: np.ndarray,
                 log_like: float,
                 sample_size: int):
        """
        Container for the result of estimation
        :param params: array, the estimated (optimal) params
        :param log_like: float, the final log-likelihood value (at optimum)
        :param sample_size: int, the size of sample used in estimation (don't include S0)
        """
        self.params = params
        self.log_like = log_like
        self.sample_size = sample_size

    @property
    def likelihood(self) -> float:
        """ The likelihood with estimated params """
        return np.exp(self.log_like)

    @property
    def aic(self) -> float:
        """ The AIC (Aikake Information Criteria) with estimated params """
        return 2 * (len(self.params) - self.log_like)

    @property
    def bic(self) -> float:
        """ The BIC (Bayesian Information Criteria) with estimated params """
        return len(self.params) * np.log(self.sample_size) - 2 * self.log_like

    def __str__(self):
        """ String representation of the class (for pretty printing the results) """
        return f'\nparams      | {self.params} \n' \
               f'sample size | {self.sample_size} \n' \
               f'likelihood  | {self.log_like} \n' \
               f'AIC         | {self.aic}\n' \
               f'BIC         | {self.bic}'


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

    def estimate_params(self, params0: np.ndarray) -> EstimatedResult:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: result, the estimated params and final likelihood
        """
        raise NotImplementedError
