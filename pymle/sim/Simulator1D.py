import numpy as np
from pymle.Model import Model1D
from pymle.sim.Stepper import Stepper


class Simulator1D(object):
    def __init__(self,
                 S0: float,
                 M: int,
                 dt: float,
                 model: Model1D,
                 sub_step: int = 5,
                 seed: int = None,
                 method: str = "Default"):
        """
        Class for simulating paths of diffusion (SDE) process
        Override the sim_path method

        :param S0: float, initial value of process
        :param M: int, number of time steps (path will be size M+1, as it contains S0)
        :param dt: float, time step size
        :param model: obj, the model
        :param sub_step: int, (optional, default=1). If greater than 1, do multiple sub-steps on each dt interval to
            reduce bias.
        :param seed: int, the random seed (used for reproducibility of experiments)
        :param method: str, the simulation scheme to use, e.g.:
            "Euler", "Milstein", "Milstein2", "Exact"
            If set to "Default", uses the default simulation defined by the model (for example, "Exact" if it is known)
        """
        self._S0 = S0
        self._M = M
        self._dt = dt
        self._model = model
        self._method = model.default_sim_method if method == "Default" else method
        self._sub_step = sub_step

        self.set_seed(seed=seed)

    def set_seed(self, seed: int = None):
        np.random.seed(seed=seed)
        return self

    @property
    def model(self) -> Model1D:
        """ Access the underlying model """
        return self._model

    def sim_path(self, num_paths: int = 1) -> np.ndarray:
        """
        Simulate a new path(s) of size M + 1
        :param num_paths: int, number of independent paths to simulate. By default, only
            one path (column array) is returned. If num_paths > 1, each column is a path
        :return: array, path(s) of process
        """
        if self._sub_step > 1 and self._method != "Exact":
            return self._sim_substep(num_paths=num_paths)

        stepper = Stepper.new_stepper(scheme=self._method, model=self._model)
        path = self._init_path(path_shape=(self._M + 1, num_paths))
        norms = np.random.normal(loc=0., scale=1., size=(self._M, num_paths))
        for i in range(self._M):
            path[i + 1, :] = stepper(t=i * self._dt, dt=self._dt, x=path[i, :], dZ=norms[i, :])
        return path

    # ====================
    # PRIVATE
    # ====================

    def _init_path(self, path_shape: tuple):
        path = np.zeros(shape=path_shape)
        path[0, :] = self._S0
        return path

    def _sim_substep(self, num_paths: int) -> np.ndarray:
        """ simulate using the sub-stepping routine (reduced bias) """
        stepper = Stepper.new_stepper(scheme=self._method, model=self._model)
        path = self._init_path(path_shape=(self._M * self._sub_step + 1, num_paths))
        norms = np.random.normal(loc=0., scale=1., size=(self._M * self._sub_step, num_paths))
        dt_sub = self._dt / self._sub_step  # divides dt into subintervals of length dt_sub

        for i in range(self._M * self._sub_step):
            path[i + 1, :] = stepper.next(t=i * dt_sub, dt=dt_sub, x=path[i, :], dZ=norms[i, :])

        return path[::self._sub_step]
