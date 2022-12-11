from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
import numpy as np
from scipy.optimize import minimize


class OptResult(object):
    def __init__(self,
                 params: np.ndarray,
                 value: float,
                 success: bool,
                 message: str = ""):
        """
        Result of optimization over parameters.
        :param params: np.ndarray, the optimal parameters
        :param value: float, the optimal objective value
        :param success: bool, true if optimization succeeded, else false
        :param message: str, messaging indicating status of optimization
        """
        self.params = params
        self.value = value
        self.success = success
        self.message = message


class Minimizer(ABC):
    """
    Abstract base class for a minimizer. Calibration routines will take an instance
    of the abstract class, allowing you to swap in your favorite minimizer, or use
    the one provided by the framework
    """
    @abstractmethod
    def minimize(self,
                 function: Callable,
                 bounds: List[Tuple] = None,
                 guess: np.ndarray = None) -> OptResult:
        """
        Minimize the objectives to obtain optimal params. Main function to override
        :param function: Callable, the objective function to minimizer
        :param bounds: list of Tuple, each tuple is a (lower, upper) bound pair for a param
            Use np.inf with an appropriate sign to disable bounds on all or some variables.
        :param guess: np.ndarray, initial guess of parameters
        :return: OptResult, the result of optimization
        """
        raise NotImplementedError


class ScipyMinimizer(Minimizer):
    def __init__(self,
                 method: str = 'trust-constr',
                 tol: float = 5e-03,
                 options: Optional[dict] = None):
        """
        Specific minimizer which simply wraps the scipy.optimize.minimize method for convenience
        For more details see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        :param method: str, which method to pass to optimizer
        :param tol: float, the function tolerance
        :param options: dict, options to pass to scipy.optimize.minimize
        """
        self._method = method  # optimization method to use
        self._tol = tol
        self._options = options or {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-04, 'verbose': 1}

    def minimize(self,
                 function: Callable,
                 bounds: Optional[List[Tuple]] = None,
                 guess: Optional[np.ndarray] = None) -> OptResult:
        """
        Minimize the objectives to obtain optimal params. Main function to override
        :param function: Callable, the objective function to minimizer
        :param bounds: list of Tuple, each tuple is a (lower, upper) bound pair for a param
            Use np.inf with an appropriate sign to disable bounds on all or some variables.
        :param guess: np.ndarray, initial guess of parameters
        :return: OptResult, the result of optimization
        """
        res = minimize(function,
                       guess,
                       tol=self._tol,
                       method=self._method,
                       bounds=bounds,
                       options=self._options)

        return OptResult(params=res.x, value=res.fun, success=res.success, message=res.message)

