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
import numpy as np
from keras.activations import swish, sigmoid
from keras.layers import ReLU, LeakyReLU, Add

from mct_quantizers import KerasActivationQuantizationHolder, keras_load_quantized_model
from mct_quantizers.keras.quantizers import ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, \
                                             ActivationUniformInferableQuantizer, ActivationLutPOTInferableQuantizer

LAYER2NAME = {ReLU: 'relu',
              LeakyReLU: 'leaky_relu',
              Add: 'add',
              swish: 'swish',
              sigmoid: 'sigmoid'}

QUANTIZER2NAME = {ActivationPOTInferableQuantizer: 'pot',
                  ActivationSymmetricInferableQuantizer: 'sym',
                  ActivationUniformInferableQuantizer: 'unf',
                  ActivationLutPOTInferableQuantizer: 'pot_lut'}

QUANTIZER2ARGS = {**dict.fromkeys([ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer],
                             {'num_bits': 4,
                              'threshold': [0.5],
                              'signed': True
                              }),
                  ActivationUniformInferableQuantizer:
                      {'num_bits': 4,
                       'min_range': [-2.0],
                       'max_range': [3.0]
                       },
                  ActivationLutPOTInferableQuantizer:
                      {'num_bits': 4,
                       'threshold': [0.5],
                       'signed': True,
                       'lut_values': [22.0, -53.0, 62.0, 0.0, -66.0, -21.0, 44.0, -40.0],
                       'lut_values_bitwidth': 8,
                       'eps': 1e-8
                       }
                  }


def _build_model_with_quantization_holder(act_layer, quant_activation_holder, input_shape, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=4)(inputs)
    act_output = act_layer(x)
    quant_output = quant_activation_holder(act_output)
    return tf.keras.Model(inputs=inputs, outputs=[quant_output, act_output], name=model_name)


def _build_model_with_operator_quantization_holder(act_layer, quant_activation_holder, input_shape, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=4)(inputs)
    y = tf.keras.layers.Conv2D(filters=3, kernel_size=4)(inputs)
    act_output = act_layer([x, y])
    quant_output = quant_activation_holder(act_output)
    return tf.keras.Model(inputs=inputs, outputs=[quant_output, act_output], name=model_name)


class BaseActivationQuantizerBuildAndSaveTest(unittest.TestCase):
    VERSION = None

    def build_and_save_model(self, quantizer, quantizer_params, layer, model_name, input_shape, is_op=False):
        assert BaseActivationQuantizerBuildAndSaveTest.VERSION is not None

        act_quantizer = quantizer(**quantizer_params)

        quant_act_holder = KerasActivationQuantizationHolder(activation_holder_quantizer=act_quantizer)


        if is_op:
            model = _build_model_with_operator_quantization_holder(act_layer=layer,
                                                                   quant_activation_holder=quant_act_holder,
                                                                   input_shape=input_shape,
                                                                   model_name=model_name)
        else:
            model = _build_model_with_quantization_holder(act_layer=layer,
                                                          quant_activation_holder=quant_act_holder,
                                                          input_shape=input_shape,
                                                          model_name=model_name)

        quant_holder_layer = [_l for _l in model.layers if isinstance(_l, KerasActivationQuantizationHolder)]
        self.assertEqual(len(quant_holder_layer), 1)

        # Verifying activation quantization after holder
        output = model(np.random.randn(1, *input_shape))
        self.assertTrue(np.any(output[0] != output[1]), "Expecting activation layer output to be different "
                                                        "from the activation holder layer output, which should be "
                                                        "quantized.")



        file_path = f'{model_name}.h5'
        tf.keras.models.save_model(model, file_path)

    def activation_test(self, quantizer, layer, is_op=False, layer_type=None):
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2ARGS[quantizer],
                                  layer=layer(),
                                  model_name=f"{BaseActivationQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer_type if layer_type is not None else layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  input_shape=(8, 8, 3),
                                  is_op=is_op)


class BaseActivationQuantizerLoadAndCompareTest(unittest.TestCase):
    SAVED_VERSION = None

    def load_and_compare_model(self, quantizer_type, layer_type):
        assert BaseActivationQuantizerLoadAndCompareTest.SAVED_VERSION is not None

        model_path = (f"{BaseActivationQuantizerLoadAndCompareTest.SAVED_VERSION}_"
                      f"{LAYER2NAME[layer_type]}_"
                      f"{QUANTIZER2NAME[quantizer_type]}.h5")

        loaded_model = keras_load_quantized_model(model_path)
        os.remove(model_path)

        tested_layer = [_l for _l in loaded_model.layers if isinstance(_l, KerasActivationQuantizationHolder)]

        self.assertEqual(len(tested_layer), 1, "Expecting exactly 1 layer of activation holder type.")
        tested_layer = tested_layer[0]

        self.assertEqual(tested_layer.activation_holder_quantizer.get_config(),
                         QUANTIZER2ARGS[quantizer_type],
                         f"Parameters of loaded quantizer should have the same values as saved quantizer, "
                         f"but some values are not match.")

    def activation_test(self, quantizer_type, layer):
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer)

