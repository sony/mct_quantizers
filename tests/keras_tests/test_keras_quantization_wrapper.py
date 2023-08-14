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
        self.model = keras.Model(inputs=inputs, outputs=x)

    def test_weights_quantization_wrapper(self):
        conv_layer = self.model.layers[1]

        wrapper = KerasQuantizationWrapper(conv_layer)
        wrapper.add_weights_quantizer(WEIGHT, IdentityWeightsQuantizer())

        # build
        wrapper.build(self.input_shapes)
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.assertTrue(isinstance(wrapper, KerasQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.assertTrue(name == WEIGHT)
        self.assertTrue((weight == getattr(wrapper.layer, WEIGHT)).numpy().all())
        self.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        # call
        call_inputs = self.inputs[0]
        outputs = wrapper.call(call_inputs.astype('float32'))
        self.assertTrue((outputs == conv_layer(call_inputs)).numpy().all())

