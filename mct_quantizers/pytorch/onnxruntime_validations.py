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

import numpy as np


def validate_weight_params(input_tensor: np.ndarray,
                           min_range: np.ndarray,
                           max_range: np.ndarray,
                           per_channel: int,
                           channel_axis: int
                           ):
    """
    Validate onnxruntime weight quantization function params.
    """

    assert isinstance(input_tensor,
                      np.ndarray), f"Input tensor expected to be a numpy array but is {type(input_tensor)}"

    assert isinstance(min_range,
                      np.ndarray), f"min_range expected to be a numpy array but is {type(min_range)}"
    assert isinstance(max_range,
                      np.ndarray), f"max_range expected to be a numpy array but is {type(max_range)}"

    assert min_range.ndim == 1, f"min_range ndim expected to be 1 but is {min_range.ndim}"
    assert max_range.ndim == 1, f"max_range ndim expected to be 1 but is {max_range.ndim}"

    assert min_range.shape[0] == max_range.shape[
        0], f'Expected min and max ranges to have same shapes but min_range shape: {min_range.shape} and max_range ' \
            f'shape: {max_range.shape}'

    assert np.all(
        min_range < max_range), f"max_range should be greater than min_range but max: {max_range}, min: {min_range}"

    if per_channel:
        assert channel_axis is not None, f'In per channel quantization channel_axis must be provided'
        assert input_tensor.shape[channel_axis] == min_range.shape[
            0], f"Mismatch between channels to quantize {input_tensor.shape[channel_axis]} to number of quantization " \
                f"ranges: " \
                f"{min_range.shape[0]}"
    else:
        assert min_range.shape[0] == 1, f'In per tensor quantization only one quantization range should be provided'


def validate_activation_params(input_tensor: np.ndarray,
                               min_range: float,
                               max_range: float):
    """
    Validate onnxruntime activation quantization function params.
    """

    assert isinstance(input_tensor,
                      np.ndarray), f"Input tensor expected to be a numpy array but is {type(input_tensor)}"

    assert isinstance(min_range,
                      float), f"min_range expected to be float but is {type(min_range)}"
    assert isinstance(max_range,
                      float), f"max_range expected to be float but is {type(max_range)}"
    assert min_range < max_range, f"max_range should be greate than min_range but max: {max_range}, min: {min_range}"
