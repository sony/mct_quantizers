# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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

from mct_quantizers.pytorch.preserving_activation_quantization_holder import PytorchPreservingActivationQuantizationHolder
from mct_quantizers.pytorch.activation_quantization_holder import PytorchActivationQuantizationHolder
from mct_quantizers.pytorch.quantizers import ActivationSymmetricInferableQuantizer, ActivationPOTInferableQuantizer, ActivationUniformInferableQuantizer


class TestPytorchPreservingActivationQuantizationHolderInference(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            (ActivationPOTInferableQuantizer, {"num_bits": 4, "threshold": [8], "signed": True}, True),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 8, "threshold": [2], "signed": False}, True),
            (ActivationUniformInferableQuantizer, {"num_bits": 7, "min_range": [-4.0], "max_range": [4.0]}, True),
            (ActivationPOTInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, False),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, False),
            (ActivationUniformInferableQuantizer, {"num_bits": 5, "min_range": [-3.0], "max_range": [3.0]}, False),
        ]
        self.expect_cases = [
            (ActivationPOTInferableQuantizer, {"num_bits": 4, "threshold": [8], "signed": True}, True),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 8, "threshold": [2], "signed": False}, True),
            (ActivationUniformInferableQuantizer, {"num_bits": 7, "min_range": [-4.03149606299213], "max_range": [3.96850393700787], "scale": [0.062992125984252]}, True),
            (ActivationPOTInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, False),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, False),
            (ActivationUniformInferableQuantizer, {"num_bits": 5, "min_range": [-3.09677419354839], "max_range": [2.90322580645161], "scale": [0.193548387096774]}, False),
        ]

    def test_preserving_activation_quantization_holder_inference(self):
        for (quantizer_class, quantizer_args, quantization_bypass), \
            (exp_quantizer_class, exp_quantizer_args, exp_quantization_bypass) in zip(self.test_cases, self.expect_cases):
            quantizer = quantizer_class(**quantizer_args)
            model = PytorchPreservingActivationQuantizationHolder(quantizer, quantization_bypass)

            # Initialize a random input to quantize between -50 to 50.
            input_tensor = torch.from_numpy(np.random.rand(1, 3, 50, 50). astype(np.float32) * 100 - 50, )
            # Quantize tensor
            quantized_tensor = model(input_tensor)

            # Only used when quantization_bypass is False
            exp_model = PytorchActivationQuantizationHolder(quantizer)
            exp_quantized_tensor = exp_model(input_tensor)

            self.assertTrue(isinstance(model.activation_holder_quantizer, exp_quantizer_class))
            self.assertTrue(model.quantization_bypass == exp_quantization_bypass)

            if isinstance(model.activation_holder_quantizer, ActivationUniformInferableQuantizer):
                self.assertTrue(model.activation_holder_quantizer.num_bits == exp_quantizer_args["num_bits"])
                self.assertTrue(np.allclose(model.activation_holder_quantizer.min_range, exp_quantizer_args["min_range"][0]))
                self.assertTrue(np.allclose(model.activation_holder_quantizer.max_range, exp_quantizer_args["max_range"][0]))
                self.assertTrue(np.allclose(model.activation_holder_quantizer.scale, exp_quantizer_args["scale"][0]))
            else:
                self.assertTrue(model.activation_holder_quantizer.num_bits == exp_quantizer_args["num_bits"])
                self.assertTrue(model.activation_holder_quantizer.threshold_np == exp_quantizer_args["threshold"])
                self.assertTrue(model.activation_holder_quantizer.signed == exp_quantizer_args["signed"])

                if not quantization_bypass:
                    # The maximal threshold is 4 using a signed quantization, so we expect all values to be between -4 and 4
                    self.assertTrue(quantized_tensor.max().item() < exp_quantizer_args["threshold"][0], f'Quantized values should not contain values greater than maximal threshold ')
                    self.assertTrue(quantized_tensor.min().item() >= -exp_quantizer_args["threshold"][0], f'Quantized values should not contain values lower than minimal threshold ')

            if quantization_bypass:
                # Value is the same for input and output
                self.assertTrue(np.allclose(quantized_tensor, input_tensor), f'Expected values are the same tensor but output tensor is {quantized_tensor}')
            else:   # quantization_bypass is False
                # Output value is the same as PytorchActivationQuantizationHolder
                self.assertTrue(np.allclose(quantized_tensor, exp_quantized_tensor), f'Expected values are the same as PytorchActivationQuantizationHolder but output tensor is {quantized_tensor}')

                self.assertTrue(len(torch.unique(quantized_tensor)) <= 2 ** quantizer_args["num_bits"], f'Quantized tensor expected to have no more than {2 ** quantizer_args["num_bits"]} unique values but has {len(np.unique(quantized_tensor))} unique values')
                # Assert some values are negative (signed quantization)
                self.assertTrue(torch.any(quantized_tensor < 0).item(), f'Expected some values to be negative but quantized tensor is {quantized_tensor}')


class TestPytorchPreservingActivationQuantizationHolder(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            (ActivationPOTInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, True),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 3, "threshold": [4], "signed": True}, True),
            (ActivationUniformInferableQuantizer, {"num_bits": 5, "min_range": [-3.0], "max_range": [3.0]}, True),
            (ActivationPOTInferableQuantizer, {"num_bits": 4, "threshold": [8], "signed": True}, False),
            (ActivationSymmetricInferableQuantizer, {"num_bits": 2, "threshold": [4], "signed": True}, False),
            (ActivationUniformInferableQuantizer, {"num_bits": 7, "min_range": [-2.0], "max_range": [2.0]}, False),
        ]

    def test_preserving_activation_quantization_holder_save_and_load(self):
        for quantizer_class, quantizer_args, quantization_bypass in self.test_cases:
            with self.subTest(quantizer_class=quantizer_class):
                quantizer = quantizer_class(**quantizer_args)
                model = PytorchPreservingActivationQuantizationHolder(quantizer, quantization_bypass)

                # Initialize a random input to quantize between -50 to 50.
                x = torch.from_numpy(np.random.rand(1, 3, 50, 50). astype(np.float32) * 100 - 50, )
                exp_output_tensor = model(x)

                fx_model = symbolic_trace(model)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                    tmp_pth_file = tmp_file.name

                try:
                    torch.save(fx_model, tmp_pth_file)
                    loaded_model = torch.load(tmp_pth_file)
                    output_tensor = loaded_model(x)

                    # Output value is the same as the quanization holder before saving.
                    self.assertTrue(np.allclose(output_tensor, exp_output_tensor), f'Expected values are the same as the quanization holder before saving but output tensor is {output_tensor}')
                finally:
                    os.remove(tmp_pth_file)


if __name__ == "__main__":
    unittest.main()
