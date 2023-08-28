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
from typing import List

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import QuantizationTarget, mark_quantizer, QuantizerID
from mct_quantizers.common.constants import FOUND_TF
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.common.quant_utils import adjust_range_to_include_zero


if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer
    from mct_quantizers.keras.validation_functions import validate_uniform_min_max_ranges, \
        validate_adjusted_min_max_ranges

    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    identifier=QuantizerID.INFERABLE)
    class WeightsUniformInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing weights using a uniform quantizer
        """
        def __init__(self,
                     num_bits: int,
                     min_range: List[float],
                     max_range: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
            """

            super(WeightsUniformInferableQuantizer, self).__init__()

            # Validate inputs properties
            validate_uniform_min_max_ranges(min_range,
                                            max_range)

            # Convert min/max to numpy arrays
            min_range_np, max_range_np = np.asarray(min_range), np.asarray(max_range)
            _min_range_np, _max_range_np = adjust_range_to_include_zero(min_range_np, max_range_np, num_bits)
            validate_adjusted_min_max_ranges(min_range=min_range_np,
                                             max_range=max_range_np,
                                             adj_min=_min_range_np,
                                             adj_max=_max_range_np)

            self.num_bits = num_bits
            self.min_range = _min_range_np.tolist()
            self.max_range = _max_range_np.tolist()
            self.max_range_np = _max_range_np
            self.min_range_np = _min_range_np

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(self.min_range_np) >= 1, f'In per-channel quantization min ranges list should be of length >= 1 but is {len(self.min_range_np)}'
                assert len(self.max_range_np) >= 1, f'In per-channel quantization max ranges list should be of length >= 1 but is {len(self.max_range_np)}'
            else:
                assert len(self.min_range_np) == 1, f'In per-tensor quantization min/max should be of length 1 but is {len(self.min_range)}'
                assert len(self.min_range_np) == 1, f'In per-tensor quantization min_range should be of length 1 but is {len(self.min_range_np)}'
                assert len(self.max_range_np) == 1, f'In per-tensor quantization max_range should be of length 1 but is {len(self.max_range_np)}'
                self.min_range_np = self.min_range_np[0]
                self.max_range_np = self.max_range_np[0]

            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            if per_channel and channel_axis not in [-1, self.input_rank - 1]:
                # If per-channel quantization is being used and the channel axis is not the last axis,
                # create a permutation vector to move the channel axis to the last position
                self.perm_vec = list(np.arange(self.input_rank))
                self.perm_vec[channel_axis] = self.input_rank - 1
                self.perm_vec[self.input_rank - 1] = channel_axis
            else:
                # If per-channel quantization is not being used or the channel axis is already the last axis,
                # set the permutation vector to None
                self.perm_vec = None

        def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            assert inputs.dtype==tf.float32, f'Input tensor was expected to be a float tensor but is of type {inputs.dtype}'

            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs, perm=self.perm_vec)

                # Quantize the input tensor using per-channel quantization
                q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs,
                                                                                    min=self.min_range_np,
                                                                                    max=self.max_range_np,
                                                                                    num_bits=self.num_bits)
                if self.perm_vec:
                    # Transpose the quantized tensor back to its original shape
                    q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)

                # Return the quantized tensor
                return q_tensor
            else:
                # If per-channel quantization is not being used, quantize the input tensor using regular quantization
                return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                    min=self.min_range_np,
                                                                    max=self.max_range_np,
                                                                    num_bits=self.num_bits)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'min_range', 'max_range', 'per_channel', 'channel_axis'
            """
            return {'per_channel': self.per_channel,
                    'num_bits': self.num_bits,
                    'max_range': self.max_range,
                    'min_range': self.min_range,
                    'channel_axis': self.channel_axis,
                    'input_rank': self.input_rank}

        @classmethod
        def from_config(cls, config):
            """
            Return an object with config
            Args:
                config(dict): dictionary of object configuration
            Returns: An object created with config
            """
            return cls(config.get('num_bits'),
                       config.get('min_range'),
                       config.get('max_range'),
                       config.get('per_channel'),
                       config.get('channel_axis', None),
                       config.get('input_rank', None))


else:
    class WeightsUniformInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow is mandatory '
                            'when using WeightsUniformInferableQuantizer. '
                            'Could not find Tensorflow package.')
