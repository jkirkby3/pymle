from abc import ABC
import numpy as np
from typing import List, Tuple, Optional, Union

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
                 dt: Union[float, np.ndarray],
                 model: Model1D,
                 param_bounds: List[Tuple],
                 t0: Union[float, np.ndarray] = 0):
        """
        Abstract base class for Diffusion Estimator
        :param sample: np.ndarray, a univariate time series sample from the diffusion (ascending order of time)
        :param dt: float, time step (time between diffusion steps)
            Either supply a constant dt for all time steps, or supply a set of dt's equal in length to the sample
        :param model: the diffusion model. This defines the parametric family/model,
            the parameters of which will be fitted during estimation
        :param param_bounds: List[Tuple], a list of tuples, each tuple provides (lower,upper) bounds on the parameters,
            in order of the parameters as they are defined in the generator
        :param t0: Union[float, np.ndarray], optional parameter, if you are working with a time-homogenous model,
            then this doesnt matter. Else, its the set of times at which to evaluate the drift and diffusion
             coefficients
        """
        self._sample = sample
        self._param_bounds = param_bounds
        self._dt = dt
        self._model = model
        self._t0 = t0

        if isinstance(dt, np.ndarray):
            if len(dt) != len(sample) - 1:
                raise ValueError("If you supply a sequence of dt, it must be the same size as the sample - 1")
            if len(dt.shape) != len(self._sample.shape):
                raise ValueError("The second dimension of the dt and sample vectors must agree, should be 1")

        if isinstance(t0, np.ndarray):
            if len(t0) != len(sample) - 1:
                raise ValueError("If you supply a sequence of t0, it must be the same size as the sample - 1")
            if len(t0.shape) != len(self._sample.shape):
                raise ValueError("The second dimension of the t0 and sample vectors must agree, should be 1")

    def estimate_params(self, params0: np.ndarray) -> EstimatedResult:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: result, the estimated params and final likelihood
        """
        raise NotImplementedError
