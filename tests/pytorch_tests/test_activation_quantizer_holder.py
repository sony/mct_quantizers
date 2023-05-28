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

import torch
import numpy as np
from torch.fx import symbolic_trace

from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder
from mct_quantizers.pytorch.quantizers import ActivationSymmetricInferableQuantizer, ActivationPOTInferableQuantizer


class TestPytorchActivationQuantizationHolderInference(unittest.TestCase):

    def test_activation_quantization_holder_inference(self):
        num_bits = 3
        thresholds = np.array([4])
        signed = True

        quantizer = ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                          threshold=thresholds,
                                                          signed=signed)
        model = PytorchActivationQuantizationHolder(quantizer)

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.from_numpy(np.random.rand(1, 3, 50, 50). astype(np.float32) * 100 - 50, )
        # Quantize tensor
        quantized_tensor = model(input_tensor)

        self.assertTrue(model.activation_holder_quantizer.num_bits == num_bits)
        self.assertTrue(model.activation_holder_quantizer.threshold == thresholds)
        self.assertTrue(model.activation_holder_quantizer.signed == signed)


        # The maximal threshold is 4 using a signed quantization, so we expect all values to be between -4 and 4
        self.assertTrue(quantized_tensor.max().item() < thresholds[0], f'Quantized values should not contain values greater than maximal threshold ')
        self.assertTrue(quantized_tensor.min().item() >= -thresholds[0], f'Quantized values should not contain values lower than minimal threshold ')

        self.assertTrue(len(torch.unique(quantized_tensor)) <= 2 ** num_bits, f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has {len(np.unique(quantized_tensor))} unique values')
        # Assert some values are negative (signed quantization)
        self.assertTrue(torch.any(quantized_tensor < 0).item(), f'Expected some values to be negative but quantized tensor is {quantized_tensor}')


    def test_activation_quantization_holder_save_and_load(self):
        num_bits = 3
        thresholds = np.array([4])
        signed = True

        quantizer = ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    threshold=thresholds,
                                                    signed=signed)
        model = PytorchActivationQuantizationHolder(quantizer)
        x = torch.ones(1,1)
        model(x)
        fx_model = symbolic_trace(model)

        _, tmp_h5_file = tempfile.mkstemp('.pth')
        torch.save(fx_model, tmp_h5_file)
        loaded_model = torch.load(tmp_h5_file)
        os.remove(tmp_h5_file)
        loaded_model(x)
