import importlib
import unittest

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec(
    "tensorflow_model_optimization") is not None
found_pytorch = importlib.util.find_spec("torch")

if found_tf:
    from tests.keras_tests.test_get_quantizers import TestKerasGetInferableQuantizer
    from tests.keras_tests.test_activation_quantizer_holder import TestKerasActivationQuantizationHolder
    from tests.keras_tests.test_keras_quantization_wrapper import TestKerasQuantizationWrapper
    from tests.keras_tests.quantizers_tests.test_activation_inferable_quantizers import \
        TestKerasActivationInferableQuantizers
    from tests.keras_tests.quantizers_tests.test_activation_lut_pot_inferable_quantizer import \
        TestKerasActivationLutPotQuantizer
    from tests.keras_tests.quantizers_tests.test_illegal_activation_lut_pot_inferable_quantizer import \
        TestKerasActivationIllegalLutPotQuantizer
    from tests.keras_tests.quantizers_tests.test_illegal_weights_inferable_quantizer import \
        TestKerasWeightsIllegalInferableQuantizers
    from tests.keras_tests.quantizers_tests.test_illegal_weights_lut_inferable_quantizer import \
        TestKerasWeightsIllegalSymmetricLutQuantizer, TestKerasWeightsIllegalPotLutQuantizer
    from tests.keras_tests.quantizers_tests.test_weights_inferable_quantizer import TestKerasWeightsInferableQuantizers
    from tests.keras_tests.quantizers_tests.test_weights_lut_inferable_quantizer import TestKerasWeightsLutQuantizers


if found_pytorch:
    from tests.pytorch_tests.test_get_quantizers import TestPytorchGetInferableQuantizer

if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    test_loader = unittest.TestLoader()
    MCTQuantizersSuite = unittest.TestSuite()

    # Add TF tests only if tensorflow is installed
    if found_tf:
        # suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasGetInferableQuantizer))
        package_tests = test_loader.discover("tests.keras_tests", pattern="test_*.py")
        MCTQuantizersSuite.addTests(package_tests)

    # Add Pytorch tests only if Pytorch is installed
    if found_pytorch:
        # suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestPytorchGetInferableQuantizer))
        package_tests = test_loader.discover("tests.pytorch_tests", pattern="test_*.py")
        MCTQuantizersSuite.addTests(package_tests)

    unittest.TextTestRunner(verbosity=0).run(MCTQuantizersSuite)