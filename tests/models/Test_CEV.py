import unittest
import numpy as np

from pymle.models.CEV import CEV


class Test_CEV(unittest.TestCase):
    def test_basics(self):
        model = CEV()

        kappa = 3  # rate of mean reversion
        mu = 2  # long term level of process
        sigma = 0.2  # volatility
        gamma = 1 / 2

        model.params = np.array([kappa, mu, sigma, gamma])

        for t in (0, 1, 1.5, 2):
            for x in (0.1, 0.3, 100):
                self.assertEqual(model.drift(x=x, t=t), kappa * (mu - x))
                self.assertEqual(model.diffusion(x=x, t=t), sigma * x ** gamma)

        self.assertTrue(model.has_exact_density)