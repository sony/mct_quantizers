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
import tensorflow as tf

from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer
from tests.keras_tests.compatibility_tests.base_compatibility_test import BaseQuantizerLoadAndCompareTest
from tests.keras_tests.test_keras_quantization_wrapper import WEIGHT, DEPTHWISE_WEIGHT


class WeightsPOTQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def test_conv_pot_quantizer(self):
        layer = tf.keras.layers.Conv2D
        self.load_and_compare_model(quantizer_type=WeightsPOTInferableQuantizer,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

    def test_depthwise_pot_quantizer(self):
        layer = tf.keras.layers.DepthwiseConv2D
        self.load_and_compare_model(quantizer_type=WeightsPOTInferableQuantizer,
                                    layer_type=layer,
                                    weight_name=DEPTHWISE_WEIGHT)

    def test_convtrans_pot_quantizer(self):
        layer = tf.keras.layers.Conv2DTranspose
        self.load_and_compare_model(quantizer_type=WeightsPOTInferableQuantizer,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

    def test_dense_pot_quantizer(self):
        layer = tf.keras.layers.Dense
        self.load_and_compare_model(quantizer_type=WeightsPOTInferableQuantizer,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

