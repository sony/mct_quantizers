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
# ==============================================================================g
import unittest

import numpy as np
import torch

from mct_quantizers.pytorch.quantizer_utils import get_working_device
from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_pot_inferable_quantizer import \
    WeightsPOTInferableQuantizer
from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
    WeightsSymmetricInferableQuantizer
from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import \
    WeightsUniformInferableQuantizer


class TestPytorchWeightsIllegalInferableQuantizers(unittest.TestCase):

    def test_missing_axis_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[1])
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_illegal_threshold_not_pot(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         # Not POT threshold
                                         threshold=[3])
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

    def test_illegal_threshold_length_per_tensor_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         # More than one threshold in per-tensor quantization
                                         threshold=[2, 3])
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2', str(e.exception))

    def test_zero_not_in_range_uniform_quantizer(self):
        num_bits = 3
        min_range = [-10.7, 2.3, -6.6, 0]
        max_range = [-4.1, 4.7, 20, 7]
        quantizer = WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                     per_channel=True,
                                                     min_range=min_range,
                                                     max_range=max_range,
                                                     channel_axis=2)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 50, 4, 50) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # We expect each channel values to be between min_range to max_range for each channel
        for i in range(len(min_range)):
            expected_min_channel, expected_max_channel = min_range[i], max_range[i]
            channel_slice_i = fake_quantized_tensor[:, :, i, :]
            self.assertTrue(len(channel_slice_i.unique()) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(channel_slice_i.unique())} unique values')
            self.assertTrue(0 in channel_slice_i.unique(),
                            f'zero should be in quantization range, but quantized values are in set: {channel_slice_i.unique()}')
