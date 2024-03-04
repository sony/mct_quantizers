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

from enum import Enum


class QuantizationMethod(Enum):
    """
    Method for quantization function selection:

    POWER_OF_TWO - Symmetric, uniform, threshold is power of two quantization.

    LUT_POT_QUANTIZER - quantization using a lookup table and power of 2 threshold.

    SYMMETRIC - Symmetric, uniform, quantization.

    UNIFORM - uniform quantization,

    LUT_SYM_QUANTIZER - quantization using a lookup table and symmetric threshold.

    """
    POWER_OF_TWO = 0
    LUT_POT_QUANTIZER = 1
    SYMMETRIC = 2
    UNIFORM = 3
    LUT_SYM_QUANTIZER = 4
