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

import numpy as np
import tensorflow as tf

from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer


class TestKerasActivationLutPotQuantizer(unittest.TestCase):

    def test_lut_pot_signed_quantizer(self):
        lut_values = np.asarray([-25, 25], dtype=np.float32)
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

        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['lut_values'] == lut_values))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['lut_values_bitwidth'] == lut_values_bitwidth)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (lut_values_bitwidth - 1))

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (lut_values_bitwidth - 1) - 1
        clip_min = -2 ** (lut_values_bitwidth - 1)

        quant_tensor_values = (lut_values / (2 ** (lut_values_bitwidth - int(signed)))) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** (num_bits - 1)),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)
        expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        lut_values_assignments = tf.argmin(tf.abs(tensor - expanded_lut_values), axis=-1)
        centers = tf.gather(lut_values.flatten(), lut_values_assignments)

        self.assertTrue(np.all(centers / (2 ** (lut_values_bitwidth - 1)) * thresholds == quantized_tensor),
                        "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_lut_pot_unsigned_quantizer(self):
        lut_values = np.asarray([25, 85], dtype=np.float32)
        thresholds = [2.]
        num_bits = 3
        signed = False
        lut_values_bitwidth = 8
        eps = 1e-8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       lut_values=lut_values,
                                                       signed=signed,
                                                       threshold=thresholds,
                                                       lut_values_bitwidth=
                                                       lut_values_bitwidth,
                                                       eps=eps)
        thresholds = np.asarray(thresholds)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['lut_values'] == lut_values))
        self.assertTrue(quantizer_config['threshold'] == thresholds)
        self.assertTrue(quantizer_config['signed'] == signed)
        self.assertTrue(quantizer_config['lut_values_bitwidth'] == lut_values_bitwidth)
        self.assertTrue(quantizer_config['eps'] == eps)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, tf.float32)
        quantized_tensor = quantizer(input_tensor)

        # Using a unsigned quantization, so we expect all values to be between 0
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** lut_values_bitwidth)

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** lut_values_bitwidth - 1
        clip_min = 0

        quant_tensor_values = (lut_values / (2 ** lut_values_bitwidth)) * thresholds

        self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        tensor = tf.clip_by_value((input_tensor / (thresholds + eps)) * (2 ** lut_values_bitwidth),
                                  clip_value_max=clip_max, clip_value_min=clip_min)
        tensor = tf.expand_dims(tensor, -1)

        expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        lut_values_assignments = tf.argmin(tf.abs(tensor - expanded_lut_values), axis=-1)
        centers = tf.gather(lut_values.flatten(), lut_values_assignments)

        self.assertTrue(np.all(centers / (2 ** lut_values_bitwidth) * thresholds == quantized_tensor),
                                  "Quantized tensor values weren't assigned correctly")

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(quantized_tensor >= 0),
                                  f'Expected all values to be non-negative but quantized tensor is {quantized_tensor}')
