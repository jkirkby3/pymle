import numpy as np
from pymle.Model import Model1D
from typing import List, Tuple
from pymle.fit.LikelihoodEstimator import LikelihoodEstimator


class ExactMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 dt: float,
                 model: Model1D,
                 param_bounds: List[Tuple]):
        """
        Exact MLE estimator for diffusion. This class will work with any model which defines an exact density

        :param sample: np.ndarray, a univariate time series sample from the diffusion (ascending order of time)
        :param param_bounds: List[Tuple], a list of tuples, each tuple provides (lower,upper) bounds on the parameters,
            in order of the parameters as they are defined in the generator
        :param dt: float, time step (time between diffusion steps, assumed uniform sampling frequency)
        :param model: the diffusion model. This defines the parametric family/model,
            the parameters of which will be fitted during estimation
        """
        super().__init__(param_bounds=param_bounds, model=model,
                         dt=dt, sample=sample)

    def likelihood(self, params: np.ndarray) -> float:
        """
        Calculate exact value of -Log(Likelihood), ONLY in case where the exact transition density
        is known. If it is not known, or not implemented in the generator, raises and error
        :param params: array of params to get likelihood for
        :return: float, the exact -Log(Likelihood) for these parameters
        """
        return self._likelihood_from_density(params=params, density=self._model.exact_density)


class KesslerMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 dt: float,
                 model: Model1D,
                 param_bounds: List[Tuple]):
        super().__init__(param_bounds=param_bounds, model=model,
                         dt=dt, sample=sample)

    def likelihood(self, params: np.ndarray) -> float:
        """
        Calculate exact value of -Log(Likelihood)
        :param params: array of params to get likelihood for
        :return: float, the exact -Log(Likelihood) for these parameters
        """
        return self._likelihood_from_density(params=params, density=self._model.kessler_density)


class ShojiOzakiMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 dt: float,
                 model: Model1D,
                 param_bounds: List[Tuple]):
        super().__init__(param_bounds=param_bounds, model=model,
                         dt=dt, sample=sample)

    def likelihood(self, params: np.ndarray) -> float:
        """
        Calculate exact value of -Log(Likelihood)
        :param params: array of params to get likelihood for
        :return: float, the exact -Log(Likelihood) for these parameters
        """
        return self._likelihood_from_density(params=params, density=self._model.shoji_ozaki_density)


class EulerMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 dt: float,
                 model: Model1D,
                 param_bounds: List[Tuple]):
        super().__init__(param_bounds=param_bounds, model=model,
                         dt=dt, sample=sample)

    def likelihood(self, params: np.ndarray) -> float:
        """
        Calculate exact value of -Log(Likelihood)
        :param params: array of params to get likelihood for
        :return: float, the exact -Log(Likelihood) for these parameters
        """
        return self._likelihood_from_density(params=params, density=self._model.euler_density)