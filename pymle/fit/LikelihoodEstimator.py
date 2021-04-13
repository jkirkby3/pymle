from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Callable
from scipy.optimize import minimize
from pymle.Model import Model1D

from pymle.fit.Estimator import Estimator


class LikelihoodEstimator(Estimator):
    def __init__(self,
                 sample: np.ndarray,
                 param_bounds: List[Tuple],
                 dt: float,
                 model: Model1D):
        """
        Abstract base class for Diffusion Estimator
        :param sample: np.ndarray, a univariate time series sample from the diffusion (ascending order of time)
        :param param_bounds: List[Tuple], a list of tuples, each tuple provides (lower,upper) bounds on the parameters,
            in order of the parameters as they are defined in the generator
        :param dt: float, time step (time between diffusion steps, assumed uniform sampling frequency)
        :param model: the diffusion model. This defines the parametric family/model,
            the parameters of which will be fitted during estimation
        """
        super().__init__(sample=sample, param_bounds=param_bounds, dt=dt, model=model)
        self._min_prob = 1e-30  # used to floor probabilities when evaluating the log

    def estimate_params(self, params0: np.ndarray) -> Tuple[np.ndarray, float]:
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

    def _estimate_params(self, params0: np.ndarray, likelihood: Callable) -> Tuple[np.ndarray, float]:
        """
        Main estimation function
        :param params0: array, the initial guess params
        :return: array, the estimated params
        """
        # options={'xatol': 1e-7, 'disp': True}
        options = {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-04, 'verbose': 1}
        print(f"Initial Params: {params0}")
        print(f"Initial Likelihood: {-likelihood(params0)}")

        method = 'trust-constr'  # optimization method to use
        res = minimize(lambda x: likelihood(x),
                       params0,
                       tol=5e-02,
                       method=method,
                       bounds=self._param_bounds,
                       options=options)

        params = res.x
        final_like = -likelihood(params)
        print(f"Final Params: {params}")
        print(f"Final Likelihood: {final_like}")
        return params, final_like
