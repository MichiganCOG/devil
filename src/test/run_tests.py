import unittest

from . import tests

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
