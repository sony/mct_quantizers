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

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizerID
from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.pytorch.quantizer_utils import get_working_device, to_torch_tensor, fix_range_to_include_zero

if FOUND_TORCH:
    from mct_quantizers.pytorch.quantizers.base_pytorch_inferable_quantizer import BasePyTorchInferableQuantizer


    @mark_quantizer(quantization_target=None,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    identifier=QuantizerID.INFERABLE)
    class BaseUniformInferableQuantizer(BasePyTorchInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     min_range: List[float],
                     max_range: List[float]):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing
                max_range: max quantization range for quantizing
            """

            super(BaseUniformInferableQuantizer, self).__init__()

            assert isinstance(min_range, list), f'min_range is expected to be a list, but is of type {type(min_range)}'
            assert isinstance(max_range, list), f'max_range is expected to be a list, but is of type {type(max_range)}'

            for _min, _max in zip(min_range, max_range):
                assert _min<_max, f"Max range must be greater than min value but min is {_min} and max is {_max}"

            # Align mix/max numpy arrays so they are torch Tensors on the working device
            min_range = to_torch_tensor(np.asarray(min_range)).to(get_working_device())
            max_range = to_torch_tensor(np.asarray(max_range)).to(get_working_device())

            min_range, max_range = fix_range_to_include_zero(min_range,
                                                             max_range,
                                                             num_bits)
            self.min_range = min_range
            self.max_range = max_range

            self.num_bits = num_bits
            self.min_quantized_domain = 0
            self.max_quantized_domain = 2 ** num_bits - 1


else:
    class BaseUniformInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BaseUniformInferableQuantizer. '
                            'Could not find torch package.')
