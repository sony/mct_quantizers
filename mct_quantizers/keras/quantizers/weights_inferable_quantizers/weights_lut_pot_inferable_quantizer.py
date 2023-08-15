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

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TF, LUT_VALUES_BITWIDTH, EPS
from mct_quantizers.common.quant_info import QuantizationMethod


if FOUND_TF:
    from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
        WeightsLUTSymmetricInferableQuantizer


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.LUT_POT_QUANTIZER],
                    identifier=QuantizerID.INFERABLE)
    class WeightsLUTPOTInferableQuantizer(WeightsLUTSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a lut power-of-two quantizer
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

            super(WeightsLUTPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                  lut_values=lut_values,
                                                                  threshold=threshold,
                                                                  per_channel=per_channel,
                                                                  channel_axis=channel_axis,
                                                                  input_rank=input_rank,
                                                                  lut_values_bitwidth=lut_values_bitwidth,
                                                                  eps=eps)

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in self._np_threshold.flatten()])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {self._np_threshold}'

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

else:
    class WeightsLUTPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow is mandatory '
                            'when using WeightsLUTPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
