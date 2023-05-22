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

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mct_quantizers.keras.activation_quantization_holder import ActivationQuantizationHolder
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_uniform_inferable_quantizer import ActivationUniformInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_pot_inferable_quantizer import ActivationPOTInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import ActivationSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import ActivationLutPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_pot_inferable_quantizer import WeightsPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import WeightsSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import WeightsUniformInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import WeightsLUTPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import WeightsLUTSymmetricInferableQuantizer
from mct_quantizers.keras.load_model import keras_load_quantized_model

class TestKerasLoadModel(unittest.TestCase):

    def _quantization_holder_save_and_load(self, layer_with_quantizer):
        model = keras.Sequential([layer_with_quantizer])
        x = tf.ones((3, 3))
        pred = model(x)

        _, tmp_h5_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, tmp_h5_file)
        loaded_model = keras_load_quantized_model(tmp_h5_file)
        os.remove(tmp_h5_file)
        loaded_pred = loaded_model(x)
        self.assertTrue(np.all(loaded_pred==pred))

    def test_save_and_load_activation_pot(self):
        num_bits = 3
        thresholds = [4.]
        signed = True
        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    threshold=thresholds,
                                                    signed=signed)
        layer_with_quantizer = ActivationQuantizationHolder(quantizer)
        self._quantization_holder_save_and_load(layer_with_quantizer)

    def test_save_and_load_activation_symmetric(self):
        num_bits = 3
        thresholds = [4.]
        signed = True
        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)
        layer_with_quantizer = ActivationQuantizationHolder(quantizer)
        self._quantization_holder_save_and_load(layer_with_quantizer)

    def test_save_and_load_activation_uniform(self):
        num_bits = 3
        min_range = [1.]
        max_range = [4.]
        quantizer = ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                        min_range=min_range,
                                                        max_range=max_range)
        layer_with_quantizer = ActivationQuantizationHolder(quantizer)
        self._quantization_holder_save_and_load(layer_with_quantizer)


