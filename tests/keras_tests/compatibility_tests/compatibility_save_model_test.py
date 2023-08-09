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

    def test_conv_pot_quantizer(self):
        layer = tf.keras.layers.Conv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_depthwise_pot_quantizer(self):
        layer = tf.keras.layers.DepthwiseConv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=DEPTHWISE_WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_convtrans_pot_quantizer(self):
        layer = tf.keras.layers.Conv2DTranspose
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_dense_pot_quantizer(self):
        layer = tf.keras.layers.Dense
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(units=3),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))


class WeightsSymmetricQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsSymmetricInferableQuantizer

    def test_conv_sym_quantizer(self):
        layer = tf.keras.layers.Conv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_depthwise_sym_quantizer(self):
        layer = tf.keras.layers.DepthwiseConv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=DEPTHWISE_WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_convtrans_sym_quantizer(self):
        layer = tf.keras.layers.Conv2DTranspose
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_dense_sym_quantizer(self):
        layer = tf.keras.layers.Dense
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(units=3),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))


class WeightsUniformQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def setUp(self):
        self.quantizer = WeightsUniformInferableQuantizer

    def test_conv_sym_quantizer(self):
        layer = tf.keras.layers.Conv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_depthwise_sym_quantizer(self):
        layer = tf.keras.layers.DepthwiseConv2D
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=DEPTHWISE_WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_convtrans_sym_quantizer(self):
        layer = tf.keras.layers.Conv2DTranspose
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def test_dense_sym_quantizer(self):
        layer = tf.keras.layers.Dense
        self.build_and_save_model(quantizer=self.quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[self.quantizer][layer],
                                  layer=layer(units=3),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[self.quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))