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

    @property
    def default_sim_method(self) -> str:
        """ Default method used for simulation"""
        return self._default_sim_method

    @property
    def params(self) -> np.ndarray:
        """ Access the params """
        return self._params

    @params.setter
    def params(self, vals: np.ndarray):
        """ Set parameters, used by fitter to move through param space """
        self._positive = self._set_is_positive(params=vals)  # Check if the params ensure positive density
        self._params = vals

    @property
    def is_positive(self) -> bool:
        """ Check if the model has non-negative paths, given the currently set parameters """
        return self._positive

    def _set_is_positive(self, params: np.ndarray) -> bool:
        """
        Override this method to specify if the parameters ensure a non-negative process. This is used to
        ensuring sample paths are positive. If this is not overriden, no protection is added to ensure positivity
        when simulating
        :param params: paremeters, the positivity of process can be parameter dependent
        :return: bool, True if the parameters lead to a positive process
        """
        return False

    @abstractmethod
    def drift(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ The drift term of the model """
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """ The diffusion term of the model """
        raise NotImplementedError

    @property
    def has_exact_density(self) -> bool:
        return self._has_exact_density

    # ==============================
    # Transition Density (and approximations)
    # ==============================

    def exact_density(self, x0: float, xt: float, t: float) -> float:
        """
        In the case where the exact transition density, P(Xt, t | X0) is known, override this method
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t: float, the time of observing Xt
        :return: probability
        """
        raise NotImplementedError

    def euler_density(self, x0: float, xt: float, t: float) -> float:
        """
        The transition density obtained via Euler expansion
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t: float, the time of observing Xt
        :return: probability
        """
        sig2t = (self.diffusion(x0, t) ** 2) * 2 * t
        mut = x0 + self.drift(x0, t) * t
        return np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)

    def shoji_ozaki_density(self, x0: float, xt: float, t: float) -> float:
        """
        The transition density obtained via Shoji Ozaki expansion
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t: float, the time of observing Xt
        :return: probability
        """
        sig = self.diffusion(x0, t)
        mu = self.drift(x0, t)

        Mt = 0.5 * sig ** 2 * self.drift_xx(x0, t) + self.drift_t(x0, t)
        Lt = self.drift_x(x0, t)
        if Lt == 0:
            B = sig * np.sqrt(t)
            A = x0 + mu * t + Mt * t ** 2 / 2
        else:
            B = sig * np.sqrt((np.exp(2 * Lt * t) - 1) / (2 * Lt))

            elt = np.exp(Lt * t) - 1
            A = x0 + mu / Lt * elt + Mt / (Lt ** 2) * (elt - Lt * t)

        return np.exp(-0.5 * ((xt - A) / B) ** 2) / (np.sqrt(2 * np.pi) * B)

    def elerian_density(self, x0: float, xt: float, t: float) -> float:
        """
        The transition density obtained via Milstein Expansion (Elarian density).
        When d(sigma)/dx = 0, reduces to Euler
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t: float, the time of observing Xt
        :return: probability
        """
        sig_x = self.diffusion_x(x0, t)
        if sig_x == 0:
            return self.euler_density(x0=x0, xt=xt, t=t)

        sig = self.diffusion(x0, t)
        mu = self.drift(x0, t)

        A = sig * sig_x * t * 0.5
        B = -0.5 * sig / sig_x + x0 + mu * t - A
        z = (xt - B) / A
        C = 1. / (sig_x ** 2 * t)
        if z <= 0:
            return 0
        # scz = np.sqrt(C * z)
        # ch = (np.exp(scz) + np.exp(-scz)) / 2   #
        ch = np.cosh(np.sqrt(C * z))
        return np.power(z, -0.5) * ch * np.exp(-0.5 * (C + z)) / (np.abs(A) * np.sqrt(2 * np.pi))

    def kessler_density(self, x0: float, xt: float, t: float) -> float:
        """
        The transition density obtained via Kessler expansion
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t: float, the time of observing Xt
        :return: probability
        """
        sig = self.diffusion(x0, t)
        sig2 = sig ** 2
        sig_x = self.diffusion_x(x0, t)
        sig_xx = self.diffusion_xx(x0, t)
        mu = self.drift(x0, t)
        mu_x = self.drift_x(x0, t)

        d = t ** 2 / 2
        E = x0 + mu * t + (mu * mu_x + 0.5 * sig2 * sig_xx) * d

        term = 2 * sig * sig_x
        V = x0 ** 2 + (2 * mu * x0 + sig2) * t + (2 * mu * (mu_x * x0 + mu + sig * sig_x) +
                                                  sig2 * (sig_xx * x0 + 2 * sig_x + term + sig * sig_xx)) * d - E ** 2
        V = np.sqrt(V)
        return np.exp(-0.5 * ((xt - E) / V) ** 2) / (np.sqrt(2 * np.pi) * V)

    # ==============================
    # Simulation Steps used by Simulation Schemes
    # ==============================

    def exact_step(self,
                   t: float,
                   dt: float,
                   x: Union[float, np.ndarray],
                   dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Exact Simulation Step, Implement if known """
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
