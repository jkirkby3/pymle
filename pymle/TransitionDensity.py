from abc import ABC, abstractmethod
import numpy as np
from pymle.Model import Model1D
from typing import Union


class TransitionDensity(ABC):
    def __init__(self, model: Model1D):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

        :param model: the SDE model, referenced during calls to the transition density
        """
        self._model = model

    @property
    def model(self) -> Model1D:
        """ Access to the underlying model """
        return self._model

    @abstractmethod
    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError


class ExactDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the exact transition density for a model (when available)
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        return self._model.exact_density(x0=x0, xt=xt, t0=t0, dt=dt)


class EulerDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig2t = (self._model.diffusion(x0, t0) ** 2) * 2 * dt
        mut = x0 + self._model.drift(x0, t0) * dt
        return np.exp(-(xt - mut) ** 2 / sig2t) / np.sqrt(np.pi * sig2t)


class OzakiDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)
        mu_x = self._model.drift_x(x0, t0)
        temp = mu * (np.exp(mu_x * dt) - 1) / mu_x

        Mt = x0 + temp
        Kt = (2 / dt) * np.log(1 + temp / x0)
        Vt = sig * np.sqrt((np.exp(Kt * dt) - 1) / Kt)

        return np.exp(-0.5 * ((xt - Mt) / Vt) ** 2) / (np.sqrt(2 * np.pi) * Vt)


class ShojiOzakiDensity(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Shoji-Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Shoji-Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)

        Mt = 0.5 * sig ** 2 * self._model.drift_xx(x0, t0) + self._model.drift_t(x0, t0)
        Lt = self._model.drift_x(x0, t0)
        if (Lt == 0).any():  # TODO: need to fix this
            B = sig * np.sqrt(dt)
            A = x0 + mu * dt + Mt * dt ** 2 / 2
        else:
            B = sig * np.sqrt((np.exp(2 * Lt * dt) - 1) / (2 * Lt))

            elt = np.exp(Lt * dt) - 1
            A = x0 + mu / Lt * elt + Mt / (Lt ** 2) * (elt - Lt * dt)

        return np.exp(-0.5 * ((xt - A) / B) ** 2) / (np.sqrt(2 * np.pi) * B)


class ElerianDensity(EulerDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Elerian (Milstein) approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Milstein Expansion (Elarian density).
        When d(sigma)/dx = 0, reduces to Euler
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig_x = self._model.diffusion_x(x0, t0)
        if (isinstance(x0, np.ndarray) and (sig_x == 0).any) or (not isinstance(x0, np.ndarray) and sig_x == 0):
            return super().__call__(x0=x0, xt=xt, t0=t0, dt=dt)

        sig = self._model.diffusion(x0, t0)
        mu = self._model.drift(x0, t0)

        A = sig * sig_x * dt * 0.5
        B = -0.5 * sig / sig_x + x0 + mu * dt - A
        z = (xt - B) / A
        C = 1. / (sig_x ** 2 * dt)

        scz = np.sqrt(C * z)
        cpz = -0.5 * (C + z)
        ch = (np.exp(scz + cpz) + np.exp(-scz + cpz))
        return np.power(z, -0.5) * ch / (2 * np.abs(A) * np.sqrt(2 * np.pi))


class KesslerDensity(EulerDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Kessler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Kessler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0)
        sig2 = sig ** 2
        sig_x = self._model.diffusion_x(x0, t0)
        sig_xx = self._model.diffusion_xx(x0, t0)
        mu = self._model.drift(x0, t0)
        mu_x = self._model.drift_x(x0, t0)

        d = dt ** 2 / 2
        E = x0 + mu * dt + (mu * mu_x + 0.5 * sig2 * sig_xx) * d

        term = 2 * sig * sig_x
        V = x0 ** 2 + (2 * mu * x0 + sig2) * dt + (2 * mu * (mu_x * x0 + mu + sig * sig_x) +
                                                   sig2 * (sig_xx * x0 + 2 * sig_x + term + sig * sig_xx)) * d - E ** 2
        V = np.sqrt(np.abs(V))
        return np.exp(-0.5 * ((xt - E) / V) ** 2) / (np.sqrt(2 * np.pi) * V)


class AitSahalia(TransitionDensity):
    def __init__(self, model: Model1D):
        """
        Class which represents the Ait-Sahalia approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model=model)

    def __call__(self,
                 x0: Union[float, np.ndarray],
                 xt: Union[float, np.ndarray],
                 t0: Union[float, np.ndarray],
                 dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        return self._model.AitSahalia_density(x0=x0, xt=xt, t0=t0, dt=dt)
