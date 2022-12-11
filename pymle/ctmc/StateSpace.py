import numpy as np
from typing import Tuple


class StateSpace(object):
    def __init__(self, states: np.ndarray):
        """
        Class to manage the CTMC state-space
        :param states: np.ndarray, a discrete set of states
        """
        self._states = states

    @property
    def states(self) -> np.ndarray:
        return self._states

    def bin_path(self, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin continuous sample paths according to the nearest state in the discrete set of states.
            ie we map a continuous path to a potential CTMC trajectory
        :param path: np.ndarray, sample path of diffusion
        :return: Tuple[np.ndarray, np.ndarray], same size / order as input path
            The first array in the tuple gives the "binned" path, ie the path mapped into state space
            The second array in tuple gives the index of each element in the binned path of the corresponding state
        """
        binned = np.zeros_like(path)
        index = np.zeros(shape=path.shape, dtype=int)
        for i in range(len(path)):
            binned[i], index[i] = find_nearest(self._states, path[i])

        return binned, index

    @staticmethod
    def from_sample(sample: np.ndarray,
                    is_positive: bool,
                    N_states: int,
                    how: str = 'uniform',
                    bump: float = 0.1):
        """

        :param sample:
        :param is_positive:
        :param N_states:
        :param bump:  add a bit beyond the min/max sample, to reduce boundary effects
        :return:
        """
        min_S = np.min(sample)
        max_S = np.max(sample)

        if is_positive:
            x1 = (1 - bump) * min_S
            xm = (1 + bump) * max_S
        else:
            width = max_S - min_S
            buffer = bump * width
            x1 = min_S - buffer
            xm = max_S + buffer

        return StateSpace(states=make_state_space(x1=x1, xm=xm, m=N_states, how=how))


def find_nearest(array: np.ndarray, value: float) -> Tuple[float, int]:
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        idx = idx - 1

    return array[idx], idx


def make_state_space(x1: float,
                     xm: float,
                     m: int,
                     how='uniform') -> np.ndarray:
    """
    Make a state space given the boundaries and size
    :param x1: left boundary
    :param xm: right boundary
    :param m: size (num grid points)
    :param how: str, the method (e.g. uniform)
    :return: np.ndarray, state space
    """
    states = np.zeros(m)
    states[0] = x1
    states[-1] = xm

    if how == 'uniform':
        dx = (xm - x1) / (m - 1.)
        for i in range(1, m - 1):
            states[i] = states[i - 1] + dx
    else:
        raise NotImplemented

    return states
