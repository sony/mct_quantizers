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
    WeightsUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer
from tests.compatibility_tests.keras_comp_tests.base_compatibility_test import BaseQuantizerLoadAndCompareTest


class WeightsPOTQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsSymmetricQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsUniformQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsPOTLutQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsSymmetricLutQuantizerLoadAndCompareTest(BaseQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

