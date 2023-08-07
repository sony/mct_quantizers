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
from typing import Any

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.common.quant_info import QuantizationMethod

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_symmetric_inferable_quantizer import BaseSymmetricInferableQuantizer
    from mct_quantizers.pytorch.quantizer_utils import to_torch_tensor, get_working_device
    from onnxruntime_extensions import onnx_op, PyCustomOpDef


    @onnx_op(op_type="WeightsSymmetricQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_int64,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_bool,
                     PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_float])
    def weight_sym_ort(x, nbits, t, pc, axis):
        return quantize_sym_weights_numpy(x, nbits, t, pc, axis)


    def quantize_sym_weights_torch(input_tensor, num_bits, threshold, per_channel, channel_axis):
        if isinstance(threshold, np.ndarray):
            threshold = torch.tensor(threshold, dtype=torch.float32).to(get_working_device())

        input_tensor = input_tensor
        scale = threshold / (2 ** (num_bits - 1))
        _min, _max = -threshold, threshold - scale

        if per_channel:
            ones = [1] * input_tensor.ndim
            ones[channel_axis] = -1
            new_shape = tuple(ones)
            # Make sure min_values and max_values have the same shape as x along the first axis
            _min = torch.reshape(_min, new_shape)
            _max = torch.reshape(_max, new_shape)
            scale = torch.reshape(scale, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = torch.where(input_tensor < _min, _min, input_tensor)
        quantized = torch.round(torch.where(input_tensor > _max, _max, clipped_x) / scale) * scale

        return quantized


    def quantize_sym_weights_numpy(input_tensor, num_bits, threshold, per_channel, channel_axis):
        scale = threshold / (2 ** (num_bits - 1))
        _min, _max = -threshold, threshold - scale
        if per_channel:
            ones = np.ones(input_tensor.ndim)
            ones[channel_axis] = -1
            new_shape = tuple([int(x) for x in ones])
            # Make sure min_values and max_values have the same shape as x along the first axis
            _min = np.reshape(_min, new_shape)
            _max = np.reshape(_max, new_shape)
            scale = np.reshape(scale, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = np.where(input_tensor < _min, _min, input_tensor)
        quantized = np.round(np.where(input_tensor > _max, _max, clipped_x) / scale) * scale
        return quantized


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    identifier=QuantizerID.INFERABLE)
    class WeightsSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a symmetric quantizer.
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     per_channel: bool,
                     channel_axis: int = None,
                     use_custom_impl: bool = False
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
            """

            super(WeightsSymmetricInferableQuantizer, self).__init__(threshold=threshold,
                                                                     num_bits=num_bits,
                                                                     signed=True,
                                                                     use_custom_impl=use_custom_impl)

            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(
                    threshold) >= 1, f'In per-channel quantization threshold should be of length >= 1 but is ' \
                                     f'{len(threshold)}'
            else:
                assert len(
                    threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is ' \
                                     f'{len(threshold)}'

            self.per_channel = per_channel
            self.channel_axis = channel_axis

            self.scales = to_torch_tensor(self.scales).to(get_working_device())
            self.zero_points = torch.zeros(len(threshold), dtype=torch.int32).to(get_working_device())
            # self.threshold_np = threshold

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """

            if self.use_custom_impl and torch.jit.is_tracing():
                return WeightsSymmetricF.apply(inputs,
                                               self.num_bits,
                                               self.threshold_np,
                                               self.per_channel,
                                               self.channel_axis)

            inputs.requires_grad = False
            if self.per_channel:
                return torch.fake_quantize_per_channel_affine(inputs,
                                                              self.scales,
                                                              self.zero_points,
                                                              axis=self.channel_axis,
                                                              quant_min=self.min_quantized_domain,
                                                              quant_max=self.max_quantized_domain)
            return torch.fake_quantize_per_tensor_affine(inputs,
                                                         self.scales,
                                                         self.zero_points,
                                                         quant_min=self.min_quantized_domain,
                                                         quant_max=self.max_quantized_domain)


    class WeightsSymmetricF(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, num_bits, threshold, per_channel, channel_axis):
            return quantize_sym_weights_torch(input_tensor, num_bits, threshold, per_channel, channel_axis)

        @staticmethod
        def symbolic(g, input_tensor, num_bits, threshold, per_channel, channel_axis):
            return g.op("ai.onnx.contrib::WeightsSymmetricQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(num_bits, dtype=torch.int64)),
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(per_channel, dtype=torch.bool)),
                        g.op('Constant', value_t=torch.tensor(channel_axis, dtype=torch.int64))).setType(
                input_tensor.type())

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            raise NotImplementedError()


else:
    class WeightsSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')
