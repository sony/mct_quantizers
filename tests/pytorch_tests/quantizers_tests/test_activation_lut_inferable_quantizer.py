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
from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer


class TestKerasActivationLutPotQuantizer(unittest.TestCase):

    def test_lut_pot_signed_quantizer(self):
        lut_values = np.asarray([-25, 25])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = True
        lut_values_bitwidth = 8

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       lut_values=lut_values,
                                                       signed=signed,
                                                       lut_values_bitwidth=
                                                       lut_values_bitwidth,
                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))
        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (lut_values_bitwidth - int(signed)))

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

        quant_tensor_values = lut_values / (2 ** (lut_values_bitwidth - int(signed))) * thresholds
        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')
        self.assertTrue(np.all(np.unique(fake_quantized_tensor)
                                         == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (lut_values_bitwidth - 1) - 1
        clip_min = -2 ** (lut_values_bitwidth - 1)

        tensor = torch.clip((input_tensor / thresholds) * (2 ** (lut_values_bitwidth - int(signed))),
                            min=clip_min, max=clip_max)
        tensor = tensor.unsqueeze(-1)
        expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        lut_values_assignments = torch.argmin(torch.abs(tensor - expanded_lut_values), dim=-1)
        centers = lut_values.flatten()[lut_values_assignments]

        self.assertTrue(np.all(centers / (2 ** (lut_values_bitwidth - int(signed))) * thresholds ==
                                         fake_quantized_tensor), "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(fake_quantized_tensor < 0),
                                  f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')

    def test_lut_pot_unsigned_quantizer(self):
        lut_values = np.asarray([25, 45])
        thresholds = np.asarray([4.])
        num_bits = 3
        signed = False
        lut_values_bitwidth = 7

        quantizer = ActivationLutPOTInferableQuantizer(num_bits=num_bits,
                                                       lut_values=lut_values,
                                                       signed=signed,
                                                       lut_values_bitwidth=
                                                       lut_values_bitwidth,
                                                       threshold=thresholds)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a unsigned quantization, so we expect all values to be between 0
        # and max(threshold)
        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (lut_values_bitwidth - int(signed)))

        fake_quantized_tensor = fake_quantized_tensor.detach().cpu().numpy()

        self.assertTrue(np.max(
            fake_quantized_tensor) < (max_threshold - delta_threshold), f'Quantized values should not contain values '
                                                                        f'greater than maximal threshold ')
        self.assertTrue(np.min(
            fake_quantized_tensor) >= 0, f'Quantized values should not contain values lower than 0 ')

        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')

        quant_tensor_values = lut_values / (2 ** (lut_values_bitwidth - int(signed))) * thresholds
        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits,
                                  f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                  f'{len(np.unique(fake_quantized_tensor))} unique values')

        self.assertTrue(np.all(np.unique(fake_quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        clip_max = 2 ** lut_values_bitwidth - 1
        clip_min = 0

        tensor = torch.clip((input_tensor / thresholds) * (2 ** (lut_values_bitwidth - int(signed))),
                            min=clip_min, max=clip_max)
        tensor = tensor.unsqueeze(-1)
        expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        lut_values_assignments = torch.argmin(torch.abs(tensor - expanded_lut_values), dim=-1)
        centers = lut_values.flatten()[lut_values_assignments]

        self.assertTrue(np.all(centers / (2 ** (lut_values_bitwidth - int(signed))) * thresholds ==
                                         fake_quantized_tensor), "Quantized tensor values weren't assigned correctly")

        # Assert all values are non-negative (unsigned quantization)
        self.assertTrue(np.all(fake_quantized_tensor >= 0),
                                  f'Expected all values to be non-negative but quantized tensor is {fake_quantized_tensor}')
