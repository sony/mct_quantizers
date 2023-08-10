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
from keras.activations import swish, sigmoid
from keras.layers import ReLU, LeakyReLU, Add

from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer, \
    ActivationPOTInferableQuantizer, ActivationLutPOTInferableQuantizer, ActivationUniformInferableQuantizer, \
    ActivationSymmetricInferableQuantizer
from tests.compatibility_tests.keras_comp_tests.base_activation_compatibility_test import \
    BaseActivationQuantizerLoadAndCompareTest
from tests.compatibility_tests.keras_comp_tests.base_weights_compatibility_test import BaseWeightsQuantizerLoadAndCompareTest


class WeightsPOTQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsSymmetricQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsUniformQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsPOTLutQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class WeightsSymmetricLutQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.depthwise_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)


class ActivationPOTQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, ReLU)
        self.activation_test(self.quantizer_type, LeakyReLU)
        self.activation_test(self.quantizer_type, Add)
        self.activation_test(self.quantizer_type, swish)
        self.activation_test(self.quantizer_type, sigmoid)


class ActivationSymmetricQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, ReLU)
        self.activation_test(self.quantizer_type, LeakyReLU)
        self.activation_test(self.quantizer_type, Add)
        self.activation_test(self.quantizer_type, swish)
        self.activation_test(self.quantizer_type, sigmoid)


class ActivationUniformQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, ReLU)
        self.activation_test(self.quantizer_type, LeakyReLU)
        self.activation_test(self.quantizer_type, Add)
        self.activation_test(self.quantizer_type, swish)
        self.activation_test(self.quantizer_type, sigmoid)


class ActivationPOTLutQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationLutPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, ReLU)
        self.activation_test(self.quantizer_type, LeakyReLU)
        self.activation_test(self.quantizer_type, Add)
        self.activation_test(self.quantizer_type, swish)
        self.activation_test(self.quantizer_type, sigmoid)