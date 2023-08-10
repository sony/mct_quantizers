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
from keras.activations import swish, sigmoid
from keras.layers import ReLU, LeakyReLU, Add

from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer, \
    ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, ActivationUniformInferableQuantizer, \
    ActivationLutPOTInferableQuantizer
from tests.compatibility_tests.keras_comp_tests.base_activation_compatibility_test import \
    BaseActivationQuantizerBuildAndSaveTest
from tests.compatibility_tests.keras_comp_tests.base_weights_compatibility_test import BaseWeightsQuantizerBuildAndSaveTest


class WeightsPOTQuantizerBuildAndSaveTest(BaseWeightsQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsPOTInferableQuantizer

    def test_weights_pot_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)


class WeightsSymmetricQuantizerBuildAndSaveTest(BaseWeightsQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsSymmetricInferableQuantizer

    def test_weights_symmetric_quantizer(self):
        self.conv_test(quantizer=self.quantizer)
        self.depthwise_test(quantizer=self.quantizer)
        self.convtrans_test(quantizer=self.quantizer)
        self.dense_test(quantizer=self.quantizer)


class WeightsUniformQuantizerBuildAndSaveTest(BaseWeightsQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)


class WeightsPOTLutQuantizerBuildAndSaveTest(BaseWeightsQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsLUTPOTInferableQuantizer

    def test_weights_pot_lut_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)


class WeightsSymmetricLutQuantizerBuildAndSaveTest(BaseWeightsQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsLUTSymmetricInferableQuantizer

    def test_weights_pot_lut_quantizer(self):
        self.conv_test(self.quantizer)
        self.depthwise_test(self.quantizer)
        self.convtrans_test(self.quantizer)
        self.dense_test(self.quantizer)


class ActivationPOTQuantizerBuildAndSaveTest(BaseActivationQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = ActivationPOTInferableQuantizer

    def test_activation_pot_quantizer(self):
        self.activation_test(self.quantizer, ReLU, is_op=False)
        self.activation_test(self.quantizer, LeakyReLU, is_op=False)
        self.activation_test(self.quantizer, Add, is_op=True)
        self.activation_test(self.quantizer, lambda: swish, is_op=False, layer_type=swish)
        self.activation_test(self.quantizer, lambda: sigmoid, is_op=False, layer_type=sigmoid)


class ActivationSymmetricQuantizerBuildAndSaveTest(BaseActivationQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = ActivationSymmetricInferableQuantizer

    def test_activation_pot_quantizer(self):
        self.activation_test(self.quantizer, ReLU, is_op=False)
        self.activation_test(self.quantizer, LeakyReLU, is_op=False)
        self.activation_test(self.quantizer, Add, is_op=True)
        self.activation_test(self.quantizer, lambda: swish, is_op=False, layer_type=swish)
        self.activation_test(self.quantizer, lambda: sigmoid, is_op=False, layer_type=sigmoid)


class ActivationUniformQuantizerBuildAndSaveTest(BaseActivationQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = ActivationUniformInferableQuantizer

    def test_activation_pot_quantizer(self):
        self.activation_test(self.quantizer, ReLU, is_op=False)
        self.activation_test(self.quantizer, LeakyReLU, is_op=False)
        self.activation_test(self.quantizer, Add, is_op=True)
        self.activation_test(self.quantizer, lambda: swish, is_op=False, layer_type=swish)
        self.activation_test(self.quantizer, lambda: sigmoid, is_op=False, layer_type=sigmoid)


class ActivationPOTLutQuantizerBuildAndSaveTest(BaseActivationQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = ActivationLutPOTInferableQuantizer

    def test_activation_pot_quantizer(self):
        self.activation_test(self.quantizer, ReLU, is_op=False)
        self.activation_test(self.quantizer, LeakyReLU, is_op=False)
        self.activation_test(self.quantizer, Add, is_op=True)
        self.activation_test(self.quantizer, lambda: swish, is_op=False, layer_type=swish)
        self.activation_test(self.quantizer, lambda: sigmoid, is_op=False, layer_type=sigmoid)