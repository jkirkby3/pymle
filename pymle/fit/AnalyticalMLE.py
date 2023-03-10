import numpy as np
from typing import List, Tuple, Union

from pymle.fit.LikelihoodEstimator import LikelihoodEstimator
from pymle.core.TransitionDensity import TransitionDensity
from pymle.fit.Minimizer import Minimizer, ScipyMinimizer


class AnalyticalMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 param_bounds: List[Tuple],
                 dt: Union[float, np.ndarray],
                 density: TransitionDensity,
                 minimizer: Minimizer = ScipyMinimizer(),
                 t0: Union[float, np.ndarray] = 0):
        """
        Maximimum likelihood estimator based on some analytical represenation for the transition density.
        e.g. ExactDensity, EulerDensity, ShojiOzakiDensity, etc.
        :param sample: array, a single path draw from some theoretical model
        :param param_bounds: list of tuples, one tuple (lower,upper) of bounds for each parmater
        :param dt: float, time step (time between diffusion steps)
            Either supply a constant dt for all time steps, or supply a set of dt's equal in length to the sample
        :param density: transition density of some kind, attached to a model
        :param minimizer: Minimizer, the minimizer that is used to maximize the likelihood function. If none is
            supplied, then ScipyMinimizer is used by default
        :param t0: Union[float, np.ndarray], optional parameter, if you are working with a time-homogenous model,
            then this doesnt matter. Else, its the set of times at which to evaluate the drift and diffusion
             coefficients
        """
        super().__init__(sample=sample, param_bounds=param_bounds, dt=dt, model=density.model,
                         minimizer=minimizer, t0=t0)
        self._density = density

    def log_likelihood_negative(self, params: np.ndarray) -> float:
        self._model.params = params
        return -np.sum(np.log(np.maximum(self._min_prob,
                                         self._density(x0=self._sample[:-1], 
                                                       xt=self._sample[1:],
                                                       t0=self._t0,
                                                       dt=self._dt))))

