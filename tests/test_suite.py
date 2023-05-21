import importlib
import unittest

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec(
    "tensorflow_model_optimization") is not None
found_pytorch = importlib.util.find_spec("torch") is not None and importlib.util.find_spec(
    "torchvision") is not None

if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    test_loader = unittest.TestLoader()
    MCTQuantizersSuite = unittest.TestSuite(suiteList)

    # Add TF tests only if tensorflow is installed
    if found_tf:
        package_tests = test_loader.discover("tests.keras_tests", pattern="test_*.py")
        MCTQuantizersSuite.addTests(package_tests)

    # Add Pytorch tests only if Pytorch is installed
    if found_pytorch:
        package_tests = test_loader.discover("tests.pytorch_tests", pattern="test_*.py")
        MCTQuantizersSuite.addTests(package_tests)

    unittest.TextTestRunner(verbosity=0).run(MCTQuantizersSuite)