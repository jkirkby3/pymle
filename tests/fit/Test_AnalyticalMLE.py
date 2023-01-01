import unittest
from pymle.models import CEV
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import AitSahalia, KesslerDensity, EulerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np


class Test_AnalyticalMLE(unittest.TestCase):
    def test_basics(self):

        # ===========================
        # Set the true model (CIR) params, to simulate the process
        # ===========================
        S0 = 0.4  # initial value of process

        kappa = 3  # rate of mean reversion
        mu = 2  # long term level of process
        sigma = 0.2  # volatility
        gamma = 1 / 2

        # ===========================
        # Create the true model to fit to
        # ===========================
        model = CEV()
        model.params = np.array([kappa, mu, sigma, gamma])

        # ===========================
        # Simulate a sample path (we will fit to this path)
        # ===========================
        T = 5  # num years of the sample
        freq = 250  # observations per year
        dt = 1. / freq
        seed = 123  # random seed: set to None to get new results each time

        simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
        sample = simulator.sim_path()

        self.assertAlmostEqual(np.min(sample), 0.4, 12)
        self.assertAlmostEqual(np.max(sample), 2.3242346558474853, 12)

        # ===========================
        # Fit maximum Likelihood estimators
        # ===========================
        # Set the parameter bounds for fitting  (kappa, mu, sigma)
        param_bounds = [(0, 10), (0, 4), (0.01, 1), (0.01, 2)]

        # Choose some initial guess for params fit
        guess = np.array([1, 0.1, 0.4, 0.2])

        # Fit using Euler MLE
        Euler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                                  density=EulerDensity(model)).estimate_params(guess)

        self.assertTrue(np.max(np.abs(Euler_est.params - [2.73130997, 2.0519191,  0.21379523, 0.39469582])) < 1e-07)
        self.assertAlmostEqual(Euler_est.aic, -6569.146216572386, 12)
        self.assertAlmostEqual(Euler_est.bic, -6548.622621251201, 12)

        # Fit using Euler MLE again, but this time supply a set of dt
        Euler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt*np.ones(shape=(len(sample) - 1, 1)),
                                  density=EulerDensity(model),
                                  t0=0,
                                  ).estimate_params(guess)

        self.assertTrue(np.max(np.abs(Euler_est.params - [2.73130997, 2.0519191,  0.21379523, 0.39469582])) < 1e-07)
        self.assertAlmostEqual(Euler_est.aic, -6569.146216572386, 12)
        self.assertAlmostEqual(Euler_est.bic, -6548.622621251201, 12)

        # Fit using Kessler MLE
        kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                                    density=KesslerDensity(model)).estimate_params(guess)

        self.assertTrue(np.max(np.abs(kessler_est.params - [2.43869419, 2.08727586, 0.25651524, 0.11623684])) < 1e-03)
        self.assertAlmostEqual(kessler_est.aic, -6558.3458499931085, 12)
        self.assertAlmostEqual(kessler_est.bic, -6537.8222546719235, 12)

        # Fit using AitSahalia MLE
        AitSahalia_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                                       density=AitSahalia(model)).estimate_params(guess)

        self.assertTrue(np.max(np.abs(AitSahalia_est.params - [2.68823016, 2.05359918, 0.21773247, 0.37567399])) < 1e-07)
        self.assertAlmostEqual(AitSahalia_est.aic, -6566.950162813345, 12)
        self.assertAlmostEqual(AitSahalia_est.bic, -6546.42656749216, 12)

