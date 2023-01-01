import unittest
from tests.fit.Test_AnalyticalMLE import Test_AnalyticalMLE


def test_suite():
    suite = unittest.TestSuite()
    for test in (Test_AnalyticalMLE,):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
