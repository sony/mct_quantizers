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
import tempfile
import unittest
from packaging import version

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from mct_quantizers.keras.activation_quantization_holder import KerasActivationQuantizationHolder
from mct_quantizers.keras.load_model import keras_load_quantized_model
from mct_quantizers.keras.metadata import add_metadata, get_metadata
from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_pot_inferable_quantizer import \
    ActivationPOTInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import \
    ActivationSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_uniform_inferable_quantizer import \
    ActivationUniformInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import \
    WeightsLUTPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_pot_inferable_quantizer import \
    WeightsPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
    WeightsSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import \
    WeightsUniformInferableQuantizer


class TestKerasLoadModel(unittest.TestCase):

    def _one_layer_model_save_and_load(self, layer_with_quantizer):
        model = keras.Sequential([layer_with_quantizer])

        x = np.random.randn(1, 99, 99, 3)
        pred = model(x)

        _, tmp_h5_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, tmp_h5_file)
        loaded_model = keras_load_quantized_model(tmp_h5_file)
        os.remove(tmp_h5_file)

        loaded_pred = loaded_model(x)
        self.assertTrue(np.all(loaded_pred == pred))

    def test_save_and_load_activation_pot(self):
        num_bits = 3
        thresholds = [4.]
        signed = True
        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    threshold=thresholds,
                                                    signed=signed)
        layer_with_quantizer = KerasActivationQuantizationHolder(quantizer)
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_activation_symmetric(self):
        num_bits = 3
        thresholds = [4.]
        signed = True
        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)
        layer_with_quantizer = KerasActivationQuantizationHolder(quantizer)
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_activation_uniform(self):
        num_bits = 3
        min_range = [1.]
        max_range = [4.]
        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)
        layer_with_quantizer = KerasActivationQuantizationHolder(quantizer)
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_activation_lut_pot(self):
        lut_values = [-25, 25]
        thresholds = [4.]
        num_bits = 3
        signed = True
        lut_values_bitwidth = 8
        eps = 1e-8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       lut_values=lut_values,
                                                       signed=signed,
                                                       threshold=thresholds,
                                                       lut_values_bitwidth=
                                                       lut_values_bitwidth,
                                                       eps=eps)

        layer_with_quantizer = KerasActivationQuantizationHolder(quantizer)
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_weights_pot(self):
        thresholds = [4., 0.5, 2.]
        num_bits = 2
        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                 per_channel=True,
                                                 threshold=thresholds,
                                                 channel_axis=3,
                                                 input_rank=4)
        layer_with_quantizer = KerasQuantizationWrapper(Conv2D(3, 3),
                                                        {'kernel': quantizer})
        self._one_layer_model_save_and_load(layer_with_quantizer)

        quantizer = WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                 per_channel=False,
                                                 threshold=[0.5])
        layer_with_quantizer = KerasQuantizationWrapper(TFOpLambda(tf.add),
                                                        {1: quantizer},
                                                        {1: np.random.random(3).astype(np.float32)})
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_weights_symmetric(self):
        thresholds = [3., 6., 2.]
        num_bits = 2
        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                       per_channel=True,
                                                       threshold=thresholds,
                                                       channel_axis=3,
                                                       input_rank=4)
        layer_with_quantizer = KerasQuantizationWrapper(Conv2D(3, 3),
                                                        {'kernel': quantizer})
        self._one_layer_model_save_and_load(layer_with_quantizer)

        quantizer = WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                       per_channel=False,
                                                       threshold=[0.7])
        layer_with_quantizer = KerasQuantizationWrapper(TFOpLambda(tf.matmul),
                                                        {1: quantizer},
                                                        {1: np.random.random((5, 3)).astype(np.float32)},
                                                        op_call_args=[False],  # note: not needed but added for testing purposes
                                                        op_call_kwargs={'transpose_b': True})
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_weights_uniform(self):
        min_range = [3., 6., 2.]
        max_range = [13., 16., 12.]
        num_bits = 2
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                     per_channel=True,
                                                     min_range=min_range,
                                                     max_range=max_range,
                                                     channel_axis=3,
                                                     input_rank=4)
        layer_with_quantizer = KerasQuantizationWrapper(Conv2D(3, 3),
                                                        {'kernel': quantizer})
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_weights_lut_symmetric(self):
        lut_values = [-25, 25]
        per_channel = True
        input_rank = 4
        num_bits = 8
        threshold = [3., 8., 7.]
        channel_axis = 3
        lut_values_bitwidth = 8
        eps = 1e-8
        quantizer = WeightsLUTSymmetricInferableQuantizer(num_bits=num_bits,
                                                          lut_values=lut_values,
                                                          threshold=threshold,
                                                          per_channel=per_channel,
                                                          channel_axis=channel_axis,
                                                          input_rank=input_rank,
                                                          lut_values_bitwidth=lut_values_bitwidth,
                                                          eps=eps)
        layer_with_quantizer = KerasQuantizationWrapper(Conv2D(3,3),
                                                        {'kernel': quantizer})
        self._one_layer_model_save_and_load(layer_with_quantizer)


    def test_save_and_load_weights_lut_pot(self):
        lut_values = [-25, 25]
        per_channel = True
        input_rank = 4
        num_bits = 8
        threshold = [1., 8., 4.]
        channel_axis = 3
        lut_values_bitwidth = 8
        eps = 1e-8
        quantizer = WeightsLUTPOTInferableQuantizer(num_bits=num_bits,
                                                    lut_values=lut_values,
                                                    threshold=threshold,
                                                    per_channel=per_channel,
                                                    channel_axis=channel_axis,
                                                    input_rank=input_rank,
                                                    lut_values_bitwidth=lut_values_bitwidth,
                                                    eps=eps)
        layer_with_quantizer = KerasQuantizationWrapper(Conv2D(3, 3),
                                                        {'kernel': quantizer})
        self._one_layer_model_save_and_load(layer_with_quantizer)

    def test_save_and_load_metadata(self):
        _input = tf.keras.layers.Input((8,))
        model = tf.keras.Model(inputs=_input, outputs = _input + 5)
        model = add_metadata(model, {'test': 'test123'})

        _, tmp_h5_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, tmp_h5_file)
        loaded_model = keras_load_quantized_model(tmp_h5_file)
        os.remove(tmp_h5_file)

        self.assertTrue(get_metadata(loaded_model) == get_metadata(model))

        _, tmp_keras_file = tempfile.mkstemp('.keras')
        keras.models.save_model(model, tmp_keras_file)
        loaded_model = keras_load_quantized_model(tmp_keras_file)
        os.remove(tmp_keras_file)

        self.assertTrue(get_metadata(loaded_model) == get_metadata(model))
