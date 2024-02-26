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

from mct_quantizers.common.constants import POSITIONAL_WEIGHT, QUANTIZED_POSITIONAL_WEIGHT
from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper

WEIGHT = 'weight'


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
        self.input_shapes = (1, 3, 8, 8)
        self.inputs = torch.from_numpy(np.random.randn(*self.input_shapes).astype(dtype=np.float32))
        self.sub_const = torch.from_numpy(np.random.randn(*self.input_shapes[1:]).astype(dtype=np.float32))
        self.linear_const = torch.from_numpy(np.random.randn(self.input_shapes[-1], 7).astype(dtype=np.float32))
        self.cat_const1 = torch.from_numpy(np.random.randn(*self.input_shapes).astype(dtype=np.float32))
        self.cat_const2 = torch.from_numpy(np.random.randn(*self.input_shapes).astype(dtype=np.float32))
        self.layers = [nn.Conv2d(3, 20, 3),
                       torch.sub,
                       torch.cat,
                       ]

    def test_weights_quantization_wrapper(self):
        wrapper = PytorchQuantizationWrapper(self.layers[0], {'weight': ZeroWeightsQuantizer()})
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, PytorchQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, nn.Conv2d))
        self.assertTrue(name == 'weight')
        self.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(torch.Tensor(self.inputs))  # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper.layer, 'weight')).any())  # check the weight are now quantized
        self.assertTrue((y[0, :, 0, 0] == getattr(wrapper.layer, 'bias')).any())  # check the wrapper's outputs are equal to biases

    def test_positional_weights_quantization_wrapper(self):
        wrapper = PytorchQuantizationWrapper(self.layers[1], {0: ZeroWeightsQuantizer()},
                                             weight_values={0: self.sub_const})
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, PytorchQuantizationWrapper))
        self.assertTrue(wrapper.layer is torch.sub)
        self.assertTrue(name == 0)
        self.assertTrue((weight == getattr(wrapper, f'{POSITIONAL_WEIGHT}_{name}')).all())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(self.inputs)  # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper, f'{QUANTIZED_POSITIONAL_WEIGHT}_{name}')).all())  # check the weight are now quantized
        self.assertTrue((y == self.layers[1](torch.zeros_like(self.sub_const), self.inputs)).all())  # check the wrapper's outputs are equal to biases

        wrapper = PytorchQuantizationWrapper(self.layers[1], {0: ZeroWeightsQuantizer()},
                                             weight_values={0: self.sub_const})
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, PytorchQuantizationWrapper))
        self.assertTrue(wrapper.layer is torch.sub)
        self.assertTrue(name == 0)
        self.assertTrue((weight == getattr(wrapper, f'{POSITIONAL_WEIGHT}_{name}')).all())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(self.inputs)  # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper, f'{QUANTIZED_POSITIONAL_WEIGHT}_{name}')).all())  # check the weight are now quantized
        self.assertTrue((y == self.layers[1](torch.zeros_like(self.sub_const), self.inputs)).all())  # check the wrapper's outputs are equal to biases

        wrapper = PytorchQuantizationWrapper(self.layers[2], {0: ZeroWeightsQuantizer(), 2: ZeroWeightsQuantizer()},
                                             weight_values={0: self.cat_const1, 2: self.cat_const2},
                                             op_call_kwargs={'dim': 1}, is_inputs_as_list=True)
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, PytorchQuantizationWrapper))
        self.assertTrue(wrapper.layer is torch.cat)
        self.assertTrue([wv[0] for wv in wrapper._weights_vars] == [0, 2])
        self.assertTrue((weight == getattr(wrapper, f'{POSITIONAL_WEIGHT}_{name}')).all())
        self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        y = wrapper(self.inputs)  # apply the wrapper on some random inputs
        self.assertTrue((0 == getattr(wrapper, f'{QUANTIZED_POSITIONAL_WEIGHT}_0')).all())  # check the weight are now quantized
        self.assertTrue((0 == getattr(wrapper, f'{QUANTIZED_POSITIONAL_WEIGHT}_2')).all())  # check the weight are now quantized
        self.assertTrue((y == self.layers[2]([torch.zeros_like(self.cat_const1),
                                              self.inputs,
                                              torch.zeros_like(self.cat_const2)],
                                             **wrapper.op_call_kwargs)).all())  # check the wrapper's outputs are equal to biases
