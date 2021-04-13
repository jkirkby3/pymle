from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from pymle.Model import Model1D


class Stepper(ABC):
    def __init__(self, model: Model1D):
        """
        Base Simulation Stepper class, which is responsible for implemeting a single step of a time-discretization
        scheme, e.g. Euler. Given the current state, it knows how to evolve the state by one time step, and is
        called sequentially during a path simulation.
        :param model: the SDE model
        """
        self._model = model

    @property
    def model(self) -> Model1D:
        """ Access to the underlying model """
        return self._model

    @abstractmethod
    def next(self,
             t: float,
             dt: float,
             x: Union[float, np.ndarray],
             dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given the current state, and random variate(s), evolves state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state, after evolving by one step
        """
        raise NotImplementedError

    def __call__(self,
                 t: float,
                 dt: float,
                 x: Union[float, np.ndarray],
                 dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Same as a call to next() """
        return self.next(t=t, dt=dt, x=x, dZ=dZ)

    @staticmethod
    def new_stepper(scheme: str, model: Model1D):
        """
        Factory method to construct a simulation stepper according to scheme
        :param scheme: str, name of the simulation scheme, e.g.
            'Euler', 'Milstein', 'Milstein2', 'Exact'
        :param model: Model1D, the SDE model to which the stepper is bound
        :return: Stepper, bound to the model, for particular scheme
        """
        if scheme == "Euler":
            return EulerStepper(model=model)
        elif scheme == "Milstein":
            return MilsteinStepper(model=model)
        elif scheme == "Exact":
            return ExactStepper(model=model)
        elif scheme == "Milstein2":
            return Milstein2Stepper(model=model)

        raise NotImplementedError


class ExactStepper(Stepper):
    def __init__(self, model: Model1D):
        """
        Exacg Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(self,
             t: float,
             dt: float,
             x: Union[float, np.ndarray],
             dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given the current state, and random variate(s), evolves state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state, after evolving by one step
        """
        return self._model.exact_step(t=t, dt=dt, x=x, dZ=dZ)


class EulerStepper(Stepper):
    def __init__(self, model: Model1D):
        """
        Euler Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(self,
             t: float,
             dt: float,
             x: Union[float, np.ndarray],
             dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given the current state, and random variate(s), evolves state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state, after evolving by one step
        """
        xp = x + self._model.drift(x, t) * dt \
             + self._model.diffusion(x, t) * np.sqrt(dt) * dZ
        return np.maximum(0., xp) if self._model.is_positive else xp


class MilsteinStepper(Stepper):
    def __init__(self, model: Model1D):
        """
        Milstein Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(self,
             t: float,
             dt: float,
             x: Union[float, np.ndarray],
             dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given the current state, and random variate(s), evolves state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state, after evolving by one step
        """
        xp = x + self._model.drift(x, t) * dt \
             + self._model.diffusion(x, t) * np.sqrt(dt) * dZ \
             + 0.5 * self._model.diffusion(x, t) * self._model.diffusion_x(x, t) * (dZ ** 2 - 1) * dt
        return np.maximum(0., xp) if self._model.is_positive else xp


class Milstein2Stepper(Stepper):
    def __init__(self, model: Model1D):
        """
        Milstein's 2nd Scheme - Simulation Step
        :param model: the SDE model
        """
        super().__init__(model=model)

    def next(self, t: float,
             dt: float,
             x: Union[float, np.ndarray],
             dZ: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given the current state, and random variate(s), evolves state by one step over time increment dt

        Note, this is the same as __call__, but with an interface that some people are more accustomed to
        :param t: float, current time
        :param dt: float, time increment (between now and next state transition)
        :param x: float or np.ndarray, current state
        :param dZ: float or np.ndarray, normal random variates, N(0,1), to evolve current state
        :return: next state, after evolving by one step
        """
        sig = self._model.diffusion(x, t)
        sig2 = sig ** 2
        sig_x = self._model.diffusion_x(x, t)
        sig_xx = self._model.diffusion_xx(x, t)
        mu = self._model.drift(x, t)
        mu_x = self._model.drift_x(x, t)
        mu_xx = self._model.drift_xx(x, t)

        xp = x + (mu - 0.5 * sig * sig_x) * dt \
             + sig * dZ * np.sqrt(dt) \
             + 0.5 * sig * sig_x * dt * dZ ** 2 \
             + dt ** 1.5 * (0.5 * mu * sig_x + 0.5 * mu_x * sig + 0.25 * sig2 * sig_xx) * dZ \
             + dt ** 2 * (0.5 * mu * mu_x + 0.25 * mu_xx * sig2)

        return np.maximum(0., xp) if self._model.is_positive else xp
