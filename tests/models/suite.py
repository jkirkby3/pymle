import unittest
from tests.models.Test_CEV import Test_CEV


def test_suite():
    suite = unittest.TestSuite()
    for test in (Test_CEV,):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
