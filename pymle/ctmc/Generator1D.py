from typing import Callable, Optional
from scipy.linalg import expm
import numpy as np
from pymle.Model import Model1D


class Generator1D(object):
    def __init__(self,
                 model: Model1D,
                 method: int = 2):
        """
        THis represents the Generator, Q matrix.
        :param model: the model, which defines the drift and diffusion
        :param method: int, the Scheme for the generator:
            1 = Mijatovic Pistorious, 2 = Lo-Skindilias
        """
        self._method = method  # 1 = Mijatovic Pistorious, 2 = Lo-Skindilias

        self._model = model

        self._states: Optional[np.ndarray] = None  # the state space of CTMC
        self._d: Optional[np.ndarray] = None  # Diff the states, used to set generator

    @property
    def model(self) -> Model1D:
        return self._model

    @property
    def states(self) -> np.ndarray:
        return self._states

    @states.setter
    def states(self, vals: np.ndarray):
        self._states = vals
        self._d = vals[1:] - vals[:-1]

    @property
    def num_states(self) -> int:
        return len(self._states)

    def update_params(self, params: np.ndarray):
        """
        Update the current set of parameters.
        :param params:
        :return: None, updates internal parameters
        """
        self._model.params = params

    def make_Q(self) -> np.ndarray:
        """
        Construct the generator, Q matrix
        :return: 2-d array (matrix)
        """

        if self._states is None:
            raise RuntimeError("You must set the states before calling this function")

        if self._method == 1:
            return self._make_Q_MijaPistorious()
        elif self._method == 2:
            return self._make_Q_LoSkindilias()
        raise NotImplementedError

    def make_P(self, dt: float) -> np.ndarray:
        """
        Make probability transition matrix, P(dt) = expm(Q*dt)
        :param dt: float, the stepsize
        :return: P, the transition matrix for CTMC at step size dt
        """
        return expm(self.make_Q() * dt)

    ##################
    # PRIVATE
    ##################

    def _init_Q(self):
        """
        Initializes Q matrix, applies boundaries
        :return: Q matrix, mu (drift) vector
        """
        m = len(self._states)
        Q = np.zeros(shape=(m, m))
        mus = self._model.drift(self._states, 0)

        # Boundaries
        Q[0, 1] = mus[0] / self._d[0]
        Q[0, 0] = -Q[0, 1]

        Q[m - 1, m - 2] = mus[m - 1] / self._d[m - 2]
        Q[m - 1, m - 1] = - Q[m - 1, m - 2]
        return Q, mus

    def _make_Q_MijaPistorious(self) -> np.ndarray:
        m = len(self._states)
        Q, mus = self._init_Q()

        sig2s = np.power(self._model.diffusion(self._states, 0), 2)

        # Interior
        for i in range(1, m - 1):
            HD = self._d[i - 1]
            HU = self._d[i]
            Q[i, i - 1] = (sig2s[i] - HU * mus[i]) / (HD * (HU + HD))
            Q[i, i + 1] = (sig2s[i] + HD * mus[i]) / (HU * (HU + HD))
            Q[i, i] = - (Q[i, i - 1] + Q[i, i + 1])

        return Q

    def _make_Q_LoSkindilias(self) -> np.ndarray:
        m = len(self._states)
        Q, mus = self._init_Q()

        mu_plus = np.maximum(0, mus)
        mu_minus = np.maximum(0, -mus)
        sig2s = np.power(self._model.diffusion(self._states, 0), 2)

        # Interior
        for i in range(1, m - 1):
            HD = self._d[i - 1]
            HU = self._d[i]
            temp = sig2s[i] - (HU * mu_plus[i] + HD * mu_minus[i])
            AA = np.maximum(0, temp) / (HU + HD)
            Q[i, i - 1] = (mu_minus[i] + AA) / HD
            Q[i, i + 1] = (mu_plus[i] + AA) / HU
            Q[i, i] = - (Q[i, i - 1] + Q[i, i + 1])

        return Q
