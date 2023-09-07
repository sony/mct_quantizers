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

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizerID
from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.common.quant_info import QuantizationMethod

if FOUND_TORCH:
    from mct_quantizers.pytorch.quantizers.base_pytorch_inferable_quantizer import BasePyTorchInferableQuantizer

    @mark_quantizer(quantization_target=None,
                    quantization_method=[QuantizationMethod.LUT_SYM_QUANTIZER],
                    identifier=QuantizerID.INFERABLE)
    class BaseLUTSymmetricInferableQuantizer(BasePyTorchInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     lut_values: List[float],
                     threshold: List[float],
                     signed: bool,
                     lut_values_bitwidth: int,
                     eps: float):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                lut_values: the values in the look-up table to assign the weights to
                threshold: threshold for quantizing values
                signed: whether or not to use signed quantization
                lut_values_bitwidth: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(BaseLUTSymmetricInferableQuantizer, self).__init__()

            assert isinstance(threshold,
                              list), f'Threshold is expected to be a list, but is of type {type(threshold)}'
            assert isinstance(lut_values,
                              list), f'lut_values is expected to be a list, but is of type {type(lut_values)}'


            self._threshold_np = np.asarray(threshold)
            self._lut_values_np = np.asarray(lut_values)

            assert len(np.unique(self._lut_values_np)) <= 2 ** num_bits, \
                f'Expected num of lut values to be less or equal than {2 ** num_bits} ' \
                f'but got {len(self._lut_values_np)}'

            assert not np.any(self._lut_values_np - self._lut_values_np.astype(int)), f'Expected lut values to be integers'

            if signed:
                assert np.all((-1 * (2 ** (lut_values_bitwidth - int(signed))) <= self._lut_values_np) &
                              (self._lut_values_np <= (2 ** (lut_values_bitwidth - int(signed)) - 1))), \
                    f'Expected lut values in the quantization range'
            else:
                assert np.all(self._lut_values_np <= (2 ** lut_values_bitwidth)), f'Expected lut values in the ' \
                                                                            f'quantization range'

            # If unsigned activation quantization, all lut_values must be positive
            if not signed:
                assert np.all(self._lut_values_np >= 0), f'Expected unsigned lut values in unsigned activation ' \
                                                          f'quantization'

            # num_bits must be less than lut_values_bitwidth
            assert num_bits <= lut_values_bitwidth, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {lut_values_bitwidth}'
            if num_bits == lut_values_bitwidth:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            self.threshold = threshold
            self.lut_values = lut_values

            self.signed = signed
            self.num_bits = num_bits
            self.lut_values_bitwidth = lut_values_bitwidth
            self.eps = eps

else:
    class BaseLUTSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory when using BaseLUTSymmetricInferableQuantizer. Could not '
                            'find torch package.')
