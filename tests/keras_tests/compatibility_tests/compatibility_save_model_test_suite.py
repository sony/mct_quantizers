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

from tests.keras_tests.compatibility_tests.base_compatibility_test import BaseQuantizerBuildAndSaveTest
from tests.keras_tests.compatibility_tests.compatibility_save_model_test import WeightsPOTQuantizerBuildAndSaveTest

if __name__ == '__main__':
    mct_quantizers_version = sys.argv[1]
    suiteList = []
    test_loader = unittest.TestLoader()

    BaseQuantizerBuildAndSaveTest.VERSION = mct_quantizers_version

    suiteList.append(test_loader.loadTestsFromTestCase(WeightsPOTQuantizerBuildAndSaveTest))

    keras_save_models_suite = unittest.TestSuite(suiteList)

    test_result = unittest.TextTestRunner(verbosity=0).run(keras_save_models_suite)

    # Exit with a non-zero code if tests failed
    if not test_result.wasSuccessful():
        print(f"Encountered an error during save model tests with version {mct_quantizers_version}.")
        sys.exit(1)
