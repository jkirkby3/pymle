import unittest

from tests.models.suite import test_suite as model_suite
from tests.fit.suite import test_suite as fit_suite

##############################################


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(model_suite())
    suite.addTests(fit_suite())
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
