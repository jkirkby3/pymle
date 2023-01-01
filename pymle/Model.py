from typing import Callable, Optional, Union
from abc import ABC, abstractmethod
import numpy as np


class Model1D(ABC):
    def __init__(self,
                 has_exact_density: bool = False,
                 default_sim_method: str = "Milstein"):
        """
        Base 1D model for SDE, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        :param has_exact_density: bool, set to true if an exact density is implemented
        :param default_sim_method: str, the default method for simulating. This can be overriden by simulator,
            but this allows you to set a good default (or perhaps exact method) per model
        """
        self._has_exact_density = has_exact_density
        self._params: Optional[np.ndarray] = None
        self._positive = False  # updated when params are set, indicates positivity of process
        self._default_sim_method = default_sim_method

    @abstractmethod
    def drift(self,
              x: Union[float, np.ndarray],
              t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ The drift term of the model """
        raise NotImplementedError

    @abstractmethod
    def diffusion(self,
                  x: Union[float, np.ndarray],
                  t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ The diffusion term of the model """
        raise NotImplementedError

    # ==============================
    # Optimization/Fit interface
    # ==============================

    @property
    def params(self) -> np.ndarray:
        """ Access the params """
        return self._params

    @params.setter
    def params(self, vals: np.ndarray):
        """ Set parameters, used by fitter to move through param space """
        self._positive = self._set_is_positive(params=vals)  # Check if the params ensure positive density
        self._params = vals

    # ==============================
    # Exact Transition Density and Simulation Step, override when available
    # ==============================

    @property
    def has_exact_density(self) -> bool:
        """ Return true if model has an exact density implemented """
        return self._has_exact_density

    def exact_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        In the case where the exact transition density,
        P(Xt, t | X0) is known, override this method
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time step between x0 and xt
        :return: probability
        """
        raise NotImplementedError

    def AitSahalia_density(self, x0: float, xt: float, t0: float, dt: float) -> float:
        """
        In the case where the Ait-Sahalia density expansion is known for this particular model, return it,
            else raises exception
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time of observing Xt
        :return: probability via Ait-Sahalia expansion
        """
        raise NotImplementedError

    def exact_step(self,
                   t: float,
                   dt: float,
                   x: Union[float, np.ndarray],
                   dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Exact Simulation Step, Implement if known (e.g. Browian motion or GBM) """
        raise NotImplementedError

    # ==============================
    # Derivatives (Numerical By Default)
    # ==============================

    def drift_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ Calculate first spatial derivative of drift, dmu/dx """
        h = 1e-05
        return (self.drift(x + h, t) - self.drift(x - h, t)) / (2 * h)

    def drift_t(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ Calculate first time derivative of drift, dmu/dt """
        h = 1e-05
        return (self.drift(x, t + h) - self.drift(x, t)) / h

    def drift_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ Calculate second spatial derivative of drift, d^2mu/dx^2 """
        h = 1e-05
        return (self.drift(x + h, t) - 2 * self.drift(x, t) + self.drift(x - h, t)) / (h * h)

    def diffusion_x(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ Calculate first spatial derivative of diffusion term, dsigma/dx """
        h = 1e-05
        return (self.diffusion(x + h, t) - self.diffusion(x - h, t)) / (2 * h)

    def diffusion_xx(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ Calculate second spatial derivative of diffusion term, d^2sigma/dx^2 """
        h = 1e-05
        return (self.diffusion(x + h, t) - 2 * self.diffusion(x, t) + self.diffusion(x - h, t)) / (h * h)

    # ==============================
    # Other properties
    # ==============================

    @property
    def is_positive(self) -> bool:
        """ Check if the model has non-negative paths, given the currently set parameters """
        return self._positive

    @property
    def default_sim_method(self) -> str:
        """ Default method used for simulation"""
        return self._default_sim_method

    # ==============================
    # Private
    # ==============================

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """
        Override this method to specify if the parameters ensure a non-negative process. This is used to
        ensuring sample paths are positive. If this is not overriden, no protection is added to ensure positivity
        when simulating
        :param params: paremeters, the positivity of process can be parameter dependent
        :return: bool, True if the parameters lead to a positive process
        """
        return False
