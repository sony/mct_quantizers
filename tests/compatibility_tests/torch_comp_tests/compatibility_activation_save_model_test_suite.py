# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest
import sys

from tests.compatibility_tests.torch_comp_tests.base_activation_compatibility_test import BaseActivationQuantizerBuildAndSaveTest
from tests.compatibility_tests.torch_comp_tests.compatibility_save_model_test import \
    ActivationPOTQuantizerBuildAndSaveTest, ActivationPOTLutQuantizerBuildAndSaveTest, \
    ActivationUniformQuantizerBuildAndSaveTest, ActivationSymmetricQuantizerBuildAndSaveTest

if __name__ == '__main__':
    mct_quantizers_version = sys.argv[1]

    suiteList = []
    test_loader = unittest.TestLoader()

    BaseActivationQuantizerBuildAndSaveTest.VERSION = mct_quantizers_version

    suiteList.append(test_loader.loadTestsFromTestCase(ActivationPOTQuantizerBuildAndSaveTest))
    suiteList.append(test_loader.loadTestsFromTestCase(ActivationSymmetricQuantizerBuildAndSaveTest))
    suiteList.append(test_loader.loadTestsFromTestCase(ActivationUniformQuantizerBuildAndSaveTest))
    # suiteList.append(test_loader.loadTestsFromTestCase(ActivationPOTLutQuantizerBuildAndSaveTest))

    torch_save_models_suite = unittest.TestSuite(suiteList)

    test_result = unittest.TextTestRunner(verbosity=0).run(torch_save_models_suite)

    # Exit with a non-zero code if tests failed
    if not test_result.wasSuccessful():
        print(f"Encountered an error during save model tests with version {mct_quantizers_version}.")
        sys.exit(1)
