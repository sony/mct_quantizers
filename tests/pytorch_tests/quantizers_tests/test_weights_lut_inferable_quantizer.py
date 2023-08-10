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
import torch

from mct_quantizers.pytorch.quantizer_utils import get_working_device
from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer


class TestPytorchWeightsLutQuantizers(unittest.TestCase):

    def _weights_lut_quantizer_test(self, inferable_quantizer, num_bits, threshold, lut_values,
                                    per_channel, channel_axis, lut_values_bitwidth):
        quantizer = inferable_quantizer(num_bits=num_bits,
                                        per_channel=per_channel,
                                        lut_values=lut_values,
                                        threshold=threshold,
                                        channel_axis=channel_axis,
                                        lut_values_bitwidth=lut_values_bitwidth)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))
        max_threshold = np.max(np.abs(threshold))
        delta_threshold = 1 / (2 ** (lut_values_bitwidth - 1))

        fake_quantized_tensor = fake_quantized_tensor.detach().cpu().numpy()

        self.assertTrue(np.max(
            fake_quantized_tensor) <= (max_threshold - delta_threshold), f'Quantized values should not contain values '
                                                                         f'greater than maximal threshold ')
        self.assertTrue(np.min(
            fake_quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                      f'threshold ')

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(fake_quantized_tensor))} unique values')

        clip_max = 2 ** (lut_values_bitwidth - 1) - 1
        clip_min = -2 ** (lut_values_bitwidth - 1)

        if per_channel:
            for i in range(len(threshold)):
                channel_slice_i = fake_quantized_tensor[:, :, :, i]
                channel_quant_tensor_values = lut_values / (2 ** (lut_values_bitwidth - 1)) * threshold[i]
                self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                                          f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                          f'{len(np.unique(channel_slice_i))} unique values')
                self.assertTrue(np.all(np.unique(channel_slice_i) == np.sort(channel_quant_tensor_values)))

                # Check quantized tensor assigned correctly
                tensor = torch.clip((input_tensor / threshold[i]) * (2 ** (lut_values_bitwidth - 1)),
                                    min=clip_min, max=clip_max)
                tensor = tensor[:, :, :, i].unsqueeze(-1)
                expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
                lut_values_assignments = torch.argmin(torch.abs(tensor - expanded_lut_values), dim=-1)
                centers = lut_values.flatten()[lut_values_assignments]

                self.assertTrue(
                    np.all(centers / (2 ** (lut_values_bitwidth - 1)) * threshold[i] == channel_slice_i),
                    "Quantized tensor values weren't assigned correctly")

        else:
            quant_tensor_values = lut_values / (2 ** (lut_values_bitwidth - 1)) * threshold
            self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                      f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                      f'{len(np.unique(fake_quantized_tensor))} unique values')
            self.assertTrue(np.all(np.unique(fake_quantized_tensor)
                                             == np.sort(quant_tensor_values)))

            # Check quantized tensor assigned correctly
            tensor = torch.clip((input_tensor / threshold) * (2 ** (lut_values_bitwidth - 1)),
                                min=clip_min, max=clip_max)
            tensor = tensor.unsqueeze(-1)
            expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
            lut_values_assignments = torch.argmin(torch.abs(tensor - expanded_lut_values), dim=-1)
            centers = lut_values.flatten()[lut_values_assignments]

            self.assertTrue(
                np.all(centers / (2 ** (lut_values_bitwidth - 1)) * threshold == fake_quantized_tensor),
                "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(fake_quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_weights_symmetric_lut_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        lut_values = np.asarray([-25, 25])
        per_channel = True
        num_bits = 3

        # test per channel
        threshold = np.asarray([3., 8., 7.])
        channel_axis = 3
        lut_values_bitwidth = 8
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)

        # test per channel and channel axis is not last
        threshold = np.asarray([3., 8., 7.])
        channel_axis = 1
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)

        # test per tensor
        threshold = np.asarray([3.])
        channel_axis = None
        per_channel = False
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)

    def test_weights_pot_lut_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        lut_values = np.asarray([-25, 25])
        per_channel = True
        num_bits = 3
        lut_values_bitwidth = 7

        # test per channel
        threshold = np.asarray([2., 8., 16.])
        channel_axis = 3
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)

        # test per channel and channel axis is not last
        threshold = np.asarray([2., 8., 16.])
        channel_axis = 1
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)

        # test per tensor
        threshold = np.asarray([4.])
        channel_axis = None
        per_channel = False
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, lut_values=lut_values,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         lut_values_bitwidth=lut_values_bitwidth)
