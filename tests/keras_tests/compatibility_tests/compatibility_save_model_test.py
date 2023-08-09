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

from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer
from tests.keras_tests.compatibility_tests.base_compatibility_test import BaseQuantizerBuildAndSaveTest, QUANTIZER2NAME, \
    LAYER2NAME, QUANTIZER2LAYER2ARGS
from tests.keras_tests.test_keras_quantization_wrapper import WEIGHT, DEPTHWISE_WEIGHT


class WeightsPOTQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsPOTInferableQuantizer

    def test_weights_pot_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)


class WeightsSymmetricQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsSymmetricInferableQuantizer

    def test_weights_symmetric_quantizer(self):
        self.conv_test(quantizer=self.quantizer)
        self.depthwise_test(quantizer=self.quantizer)
        self.convtrans_test(quantizer=self.quantizer)
        self.dense_test(quantizer=self.quantizer)


class WeightsUniformQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)