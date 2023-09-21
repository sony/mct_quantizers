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
from typing import Tuple

import numpy as np


def adjust_range_to_include_zero(range_min: np.ndarray,
                                 range_max: np.ndarray,
                                 n_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjusting the quantization range to include representation of 0.0 in the quantization grid.
    For per_channel quantization range_min\range_max should be tensors in the specific shape that allows
    quantization along the channel_axis.

    Args:
        range_min: min bound of the quantization range (before adjustment).
        range_max: max bound of the quantization range (before adjustment).
        n_bits: Number of bits to quantize the tensor.

    Returns: adjusted quantization range
    """
    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * np.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj

    min_positive = range_min > 0
    max_negative = range_max < 0
    mid_range = np.logical_and(np.logical_not(min_positive), np.logical_not(max_negative))

    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max

    # Make sure min_range_adj < 0 and max_range_adj > 0 to avoid small numeric error
    min_range_adj = np.minimum(min_range_adj, 0)
    max_range_adj = np.maximum(max_range_adj, 0)

    return min_range_adj, max_range_adj


def lut_quantizer_np(tensor_data: np.ndarray,
                     lut_values: np.ndarray,
                     signed: bool,
                     threshold: np.ndarray,
                     lut_values_bitwidth: int,
                     eps: float,
                     per_channel: bool,
                     channel_axis: int=None,
                     input_rank: int=None
                     ) -> np.ndarray:
    """
    Quantize a tensor using a non-uniform quantization based on the pre-defined values.
    """
    if per_channel:
        threshold_target_shape = [1] * input_rank
        threshold_target_shape[channel_axis] = -1
        threshold = np.reshape(threshold, threshold_target_shape)

    tensor = int_quantization_with_threshold(tensor_data,
                                             n_bits=lut_values_bitwidth,
                                             signed=signed,
                                             threshold=threshold,
                                             eps=eps)
    tensor = np.expand_dims(tensor, axis=-1)

    expanded_lut_values = lut_values.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
    lut_values_assignments = np.argmin(np.abs(tensor - expanded_lut_values), axis=-1)
    centers = lut_values.flatten()[lut_values_assignments]

    quant_tensor = (centers / (2 ** (lut_values_bitwidth - int(signed)))) * threshold

    return quant_tensor


def int_quantization_with_threshold(data: np.ndarray,
                                    n_bits: int,
                                    signed: bool,
                                    threshold: np.ndarray,
                                    eps: float) -> np.ndarray:
    """
    Divides data by threshold and quantize it to integers in the quantization range.
    """

    if signed:
        clip_max = 2 ** (n_bits - 1) - 1
        clip_min = -2 ** (n_bits - 1)
    else:
        clip_max = 2 ** n_bits - 1
        clip_min = 0

    return np.clip((data / (threshold + eps)) * (2 ** (n_bits - int(signed))),
                   clip_min, clip_max)
