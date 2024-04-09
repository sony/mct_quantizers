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

from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper

keras = tf.keras
layers = keras.layers

WEIGHT = 'kernel'
DEPTHWISE_WEIGHT = 'depthwise_kernel'
CLASS_NAME = 'class_name'


class IdentityWeightsQuantizer:
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to the original weights
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs

    def initialize_quantization(self, tensor_shape, name, layer):
        return {}


class TestKerasQuantizationWrapper(unittest.TestCase):

    def setUp(self):
        self.input_shapes = [(1, 8, 8, 3)]
        self.inputs = [np.random.randn(*in_shape) for in_shape in self.input_shapes]

        inputs = layers.Input(shape=self.input_shapes[0][1:])
        x = layers.Conv2D(6, 7, use_bias=False)(inputs)
        self.sub_const = np.random.random(x.shape[-1]).astype(np.float32)
        x = tf.subtract(self.sub_const, x)
        self.mul_const = np.random.random((1, *x.shape[1:].as_list())).astype(np.float32)
        x = tf.keras.layers.Multiply()([x, self.mul_const])
        self.matmul_cont = np.random.random((2, x.shape[-1])).astype(np.float32)
        x = tf.matmul(x, self.matmul_cont, transpose_b=True)
        self.model = keras.Model(inputs=inputs, outputs=x)
        self.output_shapes = [(1, *x.shape[1:].as_list())]

    def test_weights_quantization_wrapper(self):
        _, conv_layer, sub_layer, mul_layer, matmul_layer = self.model.layers

        wrapper = KerasQuantizationWrapper(conv_layer, {WEIGHT: IdentityWeightsQuantizer()})

        sub_wrapper = KerasQuantizationWrapper(sub_layer, {0: IdentityWeightsQuantizer()},
                                               {0: self.sub_const})
        mul_wrapper = KerasQuantizationWrapper(mul_layer, {1: IdentityWeightsQuantizer()},
                                               {1: self.mul_const},
                                               is_inputs_as_list=True)
        matmul_wrapper = KerasQuantizationWrapper(matmul_layer, {1: IdentityWeightsQuantizer()},
                                                  {1: self.matmul_cont},
                                                  op_call_args=[False],
                                                  op_call_kwargs={'transpose_b': True})

        # build
        wrapper.build(self.input_shapes)
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, KerasQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.assertTrue(name == WEIGHT)
        self.assertTrue((weight == getattr(wrapper.layer, WEIGHT)).numpy().all())
        self.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        sub_wrapper.build(self.output_shapes)
        (name, weight, quantizer) = sub_wrapper._weights_vars[0]
        self.assertTrue(isinstance(sub_wrapper, KerasQuantizationWrapper))
        self.assertTrue(sub_wrapper.layer.function is tf.subtract)
        self.assertTrue(name == 0)
        self.assertTrue((weight == getattr(sub_wrapper, f'positional_weight_{name}')).numpy().all())
        self.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        mul_wrapper.build(self.output_shapes)
        (name, weight, quantizer) = mul_wrapper._weights_vars[0]
        self.assertTrue(isinstance(mul_wrapper, KerasQuantizationWrapper))
        self.assertTrue(isinstance(mul_wrapper.layer, tf.keras.layers.Multiply))
        self.assertTrue(name == 1)
        self.assertTrue((weight == getattr(mul_wrapper, f'positional_weight_{name}')).numpy().all())
        self.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        matmul_wrapper.build(self.output_shapes)
        (name, weight, quantizer) = matmul_wrapper._weights_vars[0]
        self.assertTrue(isinstance(matmul_wrapper, KerasQuantizationWrapper))
        self.assertTrue(matmul_wrapper.layer.function is tf.matmul or
                        matmul_wrapper.layer.function.__name__ == tf.matmul.__name__)  # fix for TF 2.15
        self.assertTrue(name == 1)
        self.assertTrue((weight == getattr(matmul_wrapper, f'positional_weight_{name}')).numpy().all())
        self.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        # call
        call_inputs = self.inputs[0]
        model_output = self.model(call_inputs)
        x = wrapper.call(call_inputs.astype('float32'))
        x = sub_wrapper.call(x)
        x = mul_wrapper.call(x)
        wrappers_output = matmul_wrapper.call(x)
        self.assertTrue((wrappers_output == model_output).numpy().all())
