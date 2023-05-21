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

import torch
import torch.nn as nn
import numpy as np

from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper


class ZeroWeightsQuantizer:
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to 0
    """

    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> nn.Parameter:

        return inputs * 0

    def initialize_quantization(self, tensor_shape, name, layer):
        return {}


class ZeroActivationsQuantizer:
    """
    A dummy quantizer for test usage - "quantize" the layer's activation to 0
    """

    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool = True) -> nn.Parameter:

        return inputs * 0

    def initialize_quantization(self, tensor_shape, name, layer):
        return {}


class TestPytorchWeightsQuantizationWrapper(unittest.TestCase):

    def setUp(self):
        self.input_shapes = [(1, 3, 8, 8)]
        self.inputs = [np.random.randn(*in_shape) for in_shape in self.input_shapes]
        self.layer = nn.Conv2d(3, 20, 3)

    def test_weights_quantization_wrapper(self):
        wrapper = PytorchQuantizationWrapper(self.layer)
        wrapper.add_weights_quantizer('weight', ZeroWeightsQuantizer())
        wrapper._set_weights_vars()
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, PytorchQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, nn.Conv2d))
        self.assertTrue(name == 'weight')
        self.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(torch.Tensor(self.inputs[0])) # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper.layer, 'weight')).any()) # check the weight are now quantized
        self.assertTrue((y[0,:,0,0] == getattr(wrapper.layer, 'bias')).any()) # check the wrapper's outputs are equal to biases

    def test_activation_quantization_wrapper(self):
        wrapper = PytorchQuantizationWrapper(self.layer, activation_quantizers=[ZeroActivationsQuantizer()])

        (quantizer) = wrapper._activation_vars[0]
        self.assertTrue(isinstance(quantizer, ZeroActivationsQuantizer))
        y = wrapper(torch.Tensor(self.inputs[0]))  # apply the wrapper on some random inputs
        self.assertTrue((y == 0).any())  # check the wrapper's outputs are equal to biases
