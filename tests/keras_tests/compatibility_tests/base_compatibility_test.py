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
import tempfile
import tensorflow as tf
import numpy as np

from mct_quantizers import KerasQuantizationWrapper
from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer
from keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense

from tests.keras_tests.test_keras_quantization_wrapper import WEIGHT

KERAS_WEIGHTS_CHANNEL_AXIS_MAPPING = {Conv2D: 3,
                                      DepthwiseConv2D: 2,
                                      Dense: 1,
                                      Conv2DTranspose: 2}


def _build_model_with_quantize_wrapper(quant_weights_layer, input_shape, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = quant_weights_layer(inputs)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=model_name)

# def conv_test_model(input_shapes, model_name):
#     inputs = tf.keras.layers.Input(shape=input_shapes)
#     x = tf.keras.layers.Conv2D(filters=3, kernel_size=4)(inputs)
#     x = tf.keras.layers.ReLU()(x)
#     return tf.keras.Model(inputs=inputs, outputs=x, name=model_name)


# def depthwise_conv_test_model(input_shapes, model_name):
#     inputs = tf.keras.layers.Input(shape=input_shapes)
#     x = tf.keras.layers.DepthwiseConv2D(kernel_size=4)(inputs)
#     x = tf.keras.layers.ReLU()(x)
#     return tf.keras.Model(inputs=inputs, outputs=x, name=model_name)
#

# def _wrap_model_with_weights_quantizer(layer_type, weights_quantizer_class, quantizer_params_generator):
#     wrap_layer = KerasQuantizationWrapper(layer)
#     quantizer_params = quantizer_params_generator(layer)
#     weights_quantizer = weights_quantizer_class(**quantizer_params)
#     weight_name = DEPTHWISE_WEIGHT if isinstance(layer, DepthwiseConv2D) else WEIGHT
#     wrap_layer.add_weights_quantizer(weight_name, weights_quantizer)
#     wrap_layer.build(self.input_shapes)
#
#
#     return wrap_layer


class BaseQuantizerBuildAndSaveTest(unittest.TestCase):
    VERSION = "Not Initialized"

    def build_and_save_model(self, quantizer, quantizer_params, layer, model_name, input_shape, weight_name):

        weights_quantizer = quantizer(**quantizer_params)

        quant_weights_layer = KerasQuantizationWrapper(layer)
        quant_weights_layer.add_weights_quantizer(weight_name, weights_quantizer)

        model = _build_model_with_quantize_wrapper(quant_weights_layer=quant_weights_layer,
                                                   input_shape=input_shape,
                                                   model_name=model_name)

        wrapped_layers = [_l for _l in model.layers if isinstance(_l, KerasQuantizationWrapper)]
        self.assertEqual(len(wrapped_layers), 1)
        self.assertIsInstance(wrapped_layers[0].layer, type(layer))

        file_path = f'{model_name}.h5'
        tf.keras.models.save_model(model, file_path)


class WeightsPOTQuantizerBuildAndSaveTest(BaseQuantizerBuildAndSaveTest):

    def _quantizer_params_generator(self, threshold, per_channel, input_rank, channel_axis):
        return {'num_bits': 4,
                'per_channel': per_channel,
                'threshold': threshold,
                'input_rank': input_rank,
                'channel_axis': channel_axis}

    def test_conv_pot_quantizer(self):

        self.build_and_save_model(quantizer=WeightsPOTInferableQuantizer,
                                  quantizer_params=self._quantizer_params_generator(threshold=[2.0, 0.5, 4.0],
                                                                                    per_channel=True,
                                                                                    input_rank=4,
                                                                                    channel_axis=3),
                                  layer=tf.keras.layers.Conv2D(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_conv_pot",
                                  weight_name=WEIGHT,
                                  input_shape=(1, 8, 8, 3))