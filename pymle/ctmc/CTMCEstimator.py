import numpy as np
from typing import List, Tuple
from numba import jit

from pymle.ctmc.Generator1D import Generator1D
from pymle.fit.LikelihoodEstimator import LikelihoodEstimator
from pymle.fit.Minimizer import Minimizer, ScipyMinimizer


class CTMCEstimator(LikelihoodEstimator):
    def __init__(self,
                 binned_sample: np.ndarray,
                 s_index: np.ndarray,
                 dt: float,
                 generator: Generator1D,
                 param_bounds: List[Tuple],
                 minimizer: Minimizer = ScipyMinimizer(tol=5e-02)):
        """
        This is the CTMC MLE estimator
        :param binned_sample: array, this is the BINNED sample (ie, random trajectory along CTMC state space)
        :param s_index: array, the index of each state in the sample (e.g. the index of S[k] = i for some i)
        :param dt: time step
        :param generator: the Generator (this creates the Q matrix according to some model)
        :param param_bounds: parameter bounds - list of tuples, each tuple is (min, max) for a particular parameter
        """
        super().__init__(param_bounds=param_bounds, model=generator.model, dt=dt, sample=binned_sample,
                         minimizer=minimizer)
        self._generator = generator
        self._dt = dt
        self._states = generator.states

        # Binned CTMC sample
        self._sample = binned_sample  # sample of states
        self._s_index = s_index  # index of states

        self._m = len(self._states)

        # Transition counts
        self._do_vectorized = False
        self._C, self._band = self._init_transition_counts()

    @property
    def transition_counts(self):
        return self._C

    def log_likelihood_negative(self, params: np.ndarray) -> float:
        """
        -Log(Likelihood) function, we take negative so we can minimize it (to get maximum liklihood)
        :param params: array, the params at which to evaluate -Log(Likelihood)
        :return: float, the -Log(Likelihood) for these parameters
        """
        self._generator.update_params(params=params)
        P = self._generator.make_P(dt=self._dt)

        if self._do_vectorized:
            L = np.sum(self._C * np.log(np.maximum(self._min_prob, P)))
        else:
            L = _sum_C(self._m, self._band, self._C, P, self._min_prob)

        dx = self._states[1] - self._states[0]
        L -= (len(self._sample) - 1) * np.log(dx)  # Same effect as log(P[i, j] / dx)  above
        return -L  # Note, we return -Likelihood, to turn maximization to minimization problem

    #################
    # PRIVATE
    #################
    def _init_transition_counts(self) -> Tuple[np.ndarray, int]:
        """ Initialize the C matrix, which couts number of transitions from i->j """
        C = np.zeros(shape=(self._m, self._m))

        for i in range(len(self._sample) - 1):
            C[self._s_index[i], self._s_index[i + 1]] += 1

        # Compute the bandwidth in advance, to reduce the computational cost at each iteration, since we
        # know ahead of time the largest |i-j| for which C[i, j] > 0
        band = 0
        if not self._do_vectorized:
            for i in range(self._m):
                for j in range(self._m):
                    if C[i, j] > 0:
                        band = np.maximum(band, np.abs(i - j))
        return C, band


@jit(nopython=True, parallel=False)
def _sum_C(m, band, C, P, min_prob):
    L = 0
    # Only search within the band of known non-zeros
    for i in range(m):
        for j in range(max(0, i - band), min(i + band, m)):
            if C[i, j] > 0:
                L += C[i, j] * np.log(np.maximum(min_prob, P[i, j]))
    return L