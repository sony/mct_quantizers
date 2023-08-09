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
import os
import unittest
import tensorflow as tf
from keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense

from mct_quantizers import KerasQuantizationWrapper, keras_load_quantized_model
from mct_quantizers.common.constants import WEIGHTS_QUANTIZERS
from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer
from tests.keras_tests.test_keras_quantization_wrapper import WEIGHT, DEPTHWISE_WEIGHT

LAYER2NAME = {Conv2D: 'conv', DepthwiseConv2D: 'depthwise', Conv2DTranspose: 'convtrans', Dense: 'dense'}

QUANTIZER2NAME = {WeightsPOTInferableQuantizer: 'pot',
                  WeightsSymmetricInferableQuantizer: 'sym',
                  WeightsUniformInferableQuantizer: 'unf'}

QUANTIZER2LAYER2ARGS = {**dict.fromkeys([WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer],
                                        {Conv2D:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'input_rank': 4,
                                              'channel_axis': 3
                                              },
                                         DepthwiseConv2D:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'input_rank': 4,
                                              'channel_axis': 2
                                              },
                                         Conv2DTranspose:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'input_rank': 4,
                                              'channel_axis': 2
                                              },
                                         Dense:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'input_rank': 2,
                                              'channel_axis': 1
                                              },
                                         }),
                        WeightsUniformInferableQuantizer: {Conv2D:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'input_rank': 4,
                                                                'channel_axis': 3
                                                                },
                                                           DepthwiseConv2D:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'input_rank': 4,
                                                                'channel_axis': 2
                                                                },
                                                           Conv2DTranspose:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'input_rank': 4,
                                                                'channel_axis': 2
                                                                },
                                                           Dense:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'input_rank': 2,
                                                                'channel_axis': 1
                                                                },
                                                           }
                        }


def _build_model_with_quantize_wrapper(quant_weights_layer, input_shape, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = quant_weights_layer(inputs)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=model_name)


class BaseQuantizerBuildAndSaveTest(unittest.TestCase):
    VERSION = None

    def build_and_save_model(self, quantizer, quantizer_params, layer, model_name, input_shape, weight_name):
        assert BaseQuantizerBuildAndSaveTest.VERSION is not None

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

    def conv_test(self, quantizer):
        layer = tf.keras.layers.Conv2D
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def depthwise_test(self, quantizer):
        layer = tf.keras.layers.DepthwiseConv2D
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=DEPTHWISE_WEIGHT,
                                  input_shape=(8, 8, 3))

    def convtrans_test(self, quantizer):
        layer = tf.keras.layers.Conv2DTranspose
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(filters=3, kernel_size=4),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))

    def dense_test(self, quantizer):
        layer = tf.keras.layers.Dense
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(units=3),
                                  model_name=f"{BaseQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))


class BaseQuantizerLoadAndCompareTest(unittest.TestCase):
    SAVED_VERSION = None

    def load_and_compare_model(self, quantizer_type, layer_type, weight_name):
        assert BaseQuantizerLoadAndCompareTest.SAVED_VERSION is not None

        model_path = (f"{BaseQuantizerLoadAndCompareTest.SAVED_VERSION}_"
                      f"{LAYER2NAME[layer_type]}_"
                      f"{QUANTIZER2NAME[quantizer_type]}.h5")

        loaded_model = keras_load_quantized_model(model_path)
        os.remove(model_path)

        tested_layer = [_l for _l in loaded_model.layers if isinstance(_l, KerasQuantizationWrapper) and
                        isinstance(_l.layer, layer_type)]

        self.assertEqual(len(tested_layer), 1, "Expecting exactly 1 layer of the tested layer type.")
        tested_layer = tested_layer[0]

        self.assertEqual(tested_layer.get_config()[WEIGHTS_QUANTIZERS][weight_name]['config'],
                         QUANTIZER2LAYER2ARGS[quantizer_type][layer_type])

    def conv_test(self, quantizer_type):
        layer = tf.keras.layers.Conv2D
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

    def depthwise_test(self, quantizer_type):
        layer = tf.keras.layers.DepthwiseConv2D
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=DEPTHWISE_WEIGHT)

    def convtrans_test(self, quantizer_type):
        layer = tf.keras.layers.Conv2DTranspose
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

    def dense_test(self, quantizer_type):
        layer = tf.keras.layers.Dense
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)
