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
import warnings
from typing import List

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import QuantizationTarget, mark_quantizer, QuantizerID
from mct_quantizers.common.constants import FOUND_TF, LUT_VALUES_BITWIDTH, EPS
from mct_quantizers.common.quant_info import QuantizationMethod


if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantizer_utils import lut_quantizer
    from mct_quantizers.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.LUT_SYM_QUANTIZER],
                    identifier=QuantizerID.INFERABLE)
    class WeightsLUTSymmetricInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing weights using a lut symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     lut_values: List[float],
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     input_rank: int = None,
                     lut_values_bitwidth: int = LUT_VALUES_BITWIDTH,
                     eps: float = EPS):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                lut_values: the values in the look-up table to assign the weights to
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
                input_rank: number of dimensions of input tensor the quantizer quantizes
                lut_values_bitwidth: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(WeightsLUTSymmetricInferableQuantizer, self).__init__()

            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, np.float64)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            self.threshold = threshold
            self._np_threshold = np.asarray(threshold)
            self.lut_values = lut_values
            self._np_lut_values = np.asarray(lut_values, dtype=np.float32)

            if per_channel:
                assert input_rank is not None, f'Input rank is missing in per channel quantization'
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert -input_rank <= channel_axis < input_rank, \
                    f'Channel axis out of range. Must be {-input_rank} <= channel_axis < {input_rank}'
                assert len(threshold) >= 1, f'In per-channel quantization threshold list should be of length >= 1 ' \
                                            f'but is {len(threshold)} '
            else:
                assert len(threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is' \
                                            f' {len(threshold)}'

            assert len(np.unique(self._np_lut_values)) <= 2 ** num_bits, \
                f'Expected num of lut values to be less or equal than {2 ** num_bits} ' \
                f'but got {len(self._np_lut_values)}'

            assert not np.any(self._np_lut_values - self._np_lut_values.astype(int)), f'Expected lut values to be integers'

            # Weight quantization is signed, hence the quantization range is
            # [-2**(lut_values_bitwidth - 1), 2**(lut_values_bitwidth - 1) - 1]
            assert np.all((-1 * (2 ** (lut_values_bitwidth - 1)) <= self._np_lut_values) &
                          (self._np_lut_values <= (2 ** (lut_values_bitwidth - 1) - 1))), \
                f'Expected lut values in the quantization range'

            # num_bits must be less than lut_values_bitwidth
            assert num_bits <= lut_values_bitwidth, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {lut_values_bitwidth}'
            if num_bits == lut_values_bitwidth:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            self.num_bits = num_bits
            self.lut_values_bitwidth = lut_values_bitwidth
            self.eps = eps
            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            if per_channel and channel_axis not in [-1, self.input_rank - 1]:
                # If per-channel quantization is being used and the channel axis is not the last axis,
                # create a permutation vector to move the channel axis to the last position
                self.perm_vec = list(np.arange(self.input_rank))
                channel_axis = self.perm_vec[self.channel_axis]
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
            assert inputs.dtype == tf.float32, f'Input tensor was expected to be a float tensor but is of type ' \
                                               f'{inputs.dtype}'

            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs,
                                          perm=self.perm_vec)

                # Quantize the input tensor using per-channel quantization
                q_tensor = lut_quantizer(inputs,
                                         lut_values=self._np_lut_values.astype(np.float32),
                                         signed=True,
                                         threshold=self._np_threshold,
                                         lut_values_bitwidth=self.lut_values_bitwidth,
                                         eps=self.eps)
                if self.perm_vec:
                    # Transpose the quantized tensor back to its original shape
                    q_tensor = tf.transpose(q_tensor,
                                            perm=self.perm_vec)

                # Return the quantized tensor
                return q_tensor
            else:
                return lut_quantizer(inputs,
                                     lut_values=self._np_lut_values,
                                     signed=True,
                                     threshold=self._np_threshold,
                                     lut_values_bitwidth=self.lut_values_bitwidth,
                                     eps=self.eps)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'per_channel', 'num_bits', 'lut_values', 'threshold',
                 'channel_axis', 'input_rank', 'lut_values_bitwidth', 'eps'
            """
            return {'per_channel': self.per_channel,
                    'num_bits': self.num_bits,
                    'lut_values': self.lut_values,
                    'threshold': self.threshold,
                    'channel_axis': self.channel_axis,
                    'input_rank': self.input_rank,
                    'lut_values_bitwidth': self.lut_values_bitwidth,
                    'eps': self.eps}

        @classmethod
        def from_config(cls, config):
            """
            Return an object with config
            Args:
                config(dict): dictionary of object configuration
            Returns: An object created with config
            """
            return cls(config.get('num_bits'),
                       config.get('lut_values'),
                       config.get('threshold'),
                       config.get('per_channel'),
                       config.get('channel_axis', None),
                       config.get('input_rank', None),
                       config.get('lut_values_bitwidth', LUT_VALUES_BITWIDTH),
                       config.get('eps', EPS))

        @property
        def signed(self) -> bool:
            """
            Property to indicates that symmetric weights quantization is always signed.

            Returns: True by definition.

            """
            return True

else:
    class WeightsLUTSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow is mandatory '
                            'when using WeightsLUTSymmetricInferableQuantizer. '
                            'Could not find Tensorflow package.')
