from abc import abstractmethod
import numpy as np
from typing import List, Tuple, Callable, Union

from pymle.Model import Model1D
from pymle.fit.Minimizer import Minimizer, ScipyMinimizer
from pymle.fit.Estimator import Estimator, EstimatedResult


class LikelihoodEstimator(Estimator):
    def __init__(self,
                 sample: np.ndarray,
                 param_bounds: List[Tuple],
                 dt: Union[float, np.ndarray],
                 model: Model1D,
                 minimizer: Minimizer = ScipyMinimizer(),
                 t0: Union[float, np.ndarray] = 0):
        """
        Abstract base class for Diffusion Estimator
        :param sample: np.ndarray, a univariate time series sample from the diffusion (ascending order of time)
        :param param_bounds: List[Tuple], a list of tuples, each tuple provides (lower,upper) bounds on the parameters,
            in order of the parameters as they are defined in the generator
        :param dt: float, time step (time between diffusion steps)
            Either supply a constant dt for all time steps, or supply a set of dt's equal in length to the sample
        :param model: the diffusion model. This defines the parametric family/model,
            the parameters of which will be fitted during estimation
        :param t0: Union[float, np.ndarray], optional parameter, if you are working with a time-homogenous model,
            then this doesnt matter. Else, its the set of times at which to evaluate the drift and diffusion
             coefficients
        """
        super().__init__(sample=sample, param_bounds=param_bounds, dt=dt, model=model, t0=t0)
        self._min_prob = 1e-30  # used to floor probabilities when evaluating the log
        self._minimizer = minimizer

    def estimate_params(self, params0: np.ndarray) -> EstimatedResult:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: (array, float), the estimated params and final likelihood
        """
        return self._estimate_params(params0=params0, likelihood=self.log_likelihood_negative)

    @abstractmethod
    def log_likelihood_negative(self, params: np.ndarray) -> float:
        """
        -Log(Likelihood) function, we take negative so we can minimize it (to get maximum liklihood)
        NOTE: each call to this method sets the params during the estimation process, and evaluates the likelihood

        :param params: array, the params at which to evaluate -Log(Likelihood)
        :return: float, the -Log(Likelihood) for these parameters
        """
        raise NotImplementedError

    # ==================
    # Private
    # ==================

    def _estimate_params(self, params0: np.ndarray, likelihood: Callable) -> EstimatedResult:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: array, the estimated params
        """
        print(f"Initial Params: {params0}")
        print(f"Initial Likelihood: {-likelihood(params0)}")

        res = self._minimizer.minimize(function=likelihood, bounds=self._param_bounds, guess=params0)
        params = res.params

        final_like = -res.value
        print(f"Final Params: {params}")
        print(f"Final Likelihood: {final_like}")
        return EstimatedResult(params=params, log_like=final_like, sample_size=len(self._sample) - 1)
