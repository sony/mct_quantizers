import unittest
import sys
from tests.keras_tests.compatibility_tests.base_compatibility_test import WeightsPOTQuantizerBuildAndSaveTest, \
    BaseQuantizerBuildAndSaveTest

if __name__ == '__main__':
    mct_quantizers_version = sys.argv[1]
    suiteList = []
    test_loader = unittest.TestLoader()

    BaseQuantizerBuildAndSaveTest.VERSION = mct_quantizers_version

    suiteList.append(test_loader.loadTestsFromTestCase(WeightsPOTQuantizerBuildAndSaveTest))

    keras_save_models_suite = unittest.TestSuite(suiteList)

    unittest.TextTestRunner(verbosity=0).run(keras_save_models_suite)