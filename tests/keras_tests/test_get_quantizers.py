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

from mct_quantizers.common.base_inferable_quantizer import QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_uniform_inferable_quantizer \
    import ActivationUniformInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_pot_inferable_quantizer import \
    ActivationPOTInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import \
    ActivationSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_pot_inferable_quantizer import \
    WeightsPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
    WeightsSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import \
    WeightsUniformInferableQuantizer


class TestKerasGetInferableQuantizer(unittest.TestCase):
    def _get_inferable_quantizer_test(self, quant_target, quant_method, quantizer_base_class,
                                      expected_quantizer_class=None):
        quantizer_class = get_inferable_quantizer_class(quant_target=quant_target,
                                                        quant_method=quant_method,
                                                        quantizer_base_class=quantizer_base_class)

        self.assertTrue(issubclass(quantizer_class, quantizer_base_class))
        self.assertEqual(quantizer_class, expected_quantizer_class)

    def test_get_weight_pot_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Weights,
                                           quant_method=QuantizationMethod.POWER_OF_TWO,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=WeightsPOTInferableQuantizer)

    def test_get_weight_symmetric_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Weights,
                                           quant_method=QuantizationMethod.SYMMETRIC,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=WeightsSymmetricInferableQuantizer)

    def test_get_weight_uniform_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Weights,
                                           quant_method=QuantizationMethod.UNIFORM,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=WeightsUniformInferableQuantizer)

    def test_get_activation_pot_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Activation,
                                           quant_method=QuantizationMethod.POWER_OF_TWO,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=ActivationPOTInferableQuantizer)

    def test_get_activation_symmetric_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Activation,
                                           quant_method=QuantizationMethod.SYMMETRIC,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=ActivationSymmetricInferableQuantizer)

    def test_get_activation_uniform_quantizer(self):
        self._get_inferable_quantizer_test(quant_target=QuantizationTarget.Activation,
                                           quant_method=QuantizationMethod.UNIFORM,
                                           quantizer_base_class=BaseKerasInferableQuantizer,
                                           expected_quantizer_class=ActivationUniformInferableQuantizer)
