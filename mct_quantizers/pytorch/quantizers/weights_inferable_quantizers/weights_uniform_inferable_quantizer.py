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
from mct_quantizers.common.constants import FOUND_TORCH, FOUND_ONNXRUNTIME_EXTENSIONS, ONNX_CUSTOM_OP_DOMAIN
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.common.quant_utils import adjust_range_to_include_zero
from mct_quantizers.logger import Logger
from mct_quantizers.pytorch.onnxruntime_validations import validate_weight_params

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_uniform_inferable_quantizer import BaseUniformInferableQuantizer
    from mct_quantizers.pytorch.quantizer_utils import fix_range_to_include_zero, get_working_device, to_torch_tensor
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.base_weight_quantizer_autograd_function import \
        BaseWeightQuantizerAutogradFunction


    def quantize_uniform_weights_torch(input_tensor: torch.Tensor,
                                       num_bits: int,
                                       min_range: np.ndarray,
                                       max_range: np.ndarray,
                                       per_channel: bool,
                                       channel_axis: int=None):
        """
           Quantizes the input tensor uniformly using torch.

           Args:
               input_tensor (torch.Tensor): The input tensor to be quantized.
               num_bits (int): Number of bits to represent the quantized value.
                min_range (np.ndarray): min quantization range for quantizing weights
                max_range (np.ndarray): max quantization range for quantizing weights
                per_channel (bool): Quantize input tensor per-channel or per-tensor.
               channel_axis (int): Axis to quantize the tensor in case of per-channel quantization.

           Returns:
               Uniformly quantized tensor.
        """
        if isinstance(min_range, np.ndarray):
            min_range = torch.tensor(min_range, dtype=torch.float32).to(get_working_device())
        if isinstance(max_range, np.ndarray):
            max_range = torch.tensor(max_range, dtype=torch.float32).to(get_working_device())

        # adjusts the quantization rage so the quantization grid include zero.
        a, b = fix_range_to_include_zero(min_range, max_range, num_bits)

        # Compute the step size of quantized values.
        delta = (b - a) / (2 ** num_bits - 1)

        if per_channel:
            ones = [1] * input_tensor.ndim
            ones[channel_axis] = -1
            new_shape = tuple(ones)
            # Make sure min_values and max_values have the same shape as x along the first axis
            a = torch.reshape(a, new_shape)
            b = torch.reshape(b, new_shape)
            delta = torch.reshape(delta, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = torch.where(input_tensor < a, a, input_tensor)
        quantized = torch.round(torch.where(input_tensor > b, b, clipped_x) / delta) * delta
        return quantized


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    identifier=QuantizerID.INFERABLE)
    class WeightsUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing weights using unsigned uniform quantizer.
        """

        def __init__(self,
                     num_bits: int,
                     min_range: List[float],
                     max_range: List[float],
                     per_channel: bool,
                     channel_axis: int = None
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
            """
            super(WeightsUniformInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                   min_range=min_range,
                                                                   max_range=max_range)
            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert len(min_range) >= 1, f'In per-channel quantization min_range should be of length >= 1 but is {len(min_range)}'
                assert len(max_range) >= 1, f'In per-channel quantization max_range should be of length >= 1 but is {len(max_range)}'
            else:
                assert len(min_range) == 1, f'In per-tensor quantization min_range should be of length 1 but is {len(min_range)}'
                assert len(max_range) == 1, f'In per-tensor quantization max_range should be of length 1 but is {len(max_range)}'

            self.per_channel = per_channel
            self.channel_axis = channel_axis

            self.adjusted_min_range_np = self.min_range.cpu().numpy()
            self.adjusted_max_range_np = self.max_range.cpu().numpy()

            # Compute the step size of quantized values.
            self.scales = (self.max_range - self.min_range) / (2 ** num_bits - 1)
            self.zero_points = -(self.min_range / self.scales).int()  # zp has to be positive, and a <=0, so we multiply by -1

            self.scales = self.scales.to(get_working_device())
            self.zero_points = self.zero_points.to(get_working_device())

        def __call__(self,
                     inputs: torch.Tensor) -> torch.Tensor:
            """
            Weight fake quantizer
            Args:
                inputs: weights to quantize.

            Returns:
                quantized weights
            """
            if self.enable_reuse and not self.quantizer_first_run:
                return self.resue_outputs

            if self._use_custom_impl and torch.jit.is_tracing():
                outputs = WeightsUniformF.apply(inputs,
                                                 self.num_bits,
                                                 self.adjusted_min_range_np,
                                                 self.adjusted_max_range_np,
                                                 self.per_channel,
                                                 self.channel_axis)


            elif self.per_channel:
                inputs.requires_grad = False
                outputs = torch.fake_quantize_per_channel_affine(inputs,
                                                                  self.scales.flatten(),
                                                                  self.zero_points.flatten(),
                                                                  axis=self.channel_axis,
                                                                  quant_min=self.min_quantized_domain,
                                                                  quant_max=self.max_quantized_domain)
            else:
                inputs.requires_grad = False
                outputs = torch.fake_quantize_per_tensor_affine(inputs,
                                                                 self.scales,
                                                                 self.zero_points,
                                                                 quant_min=self.min_quantized_domain,
                                                                 quant_max=self.max_quantized_domain)

            if self.enable_reuse and self.quantizer_first_run:
                self.resue_outputs = outputs
                self.quantizer_first_run = False

            return outputs

    class WeightsUniformF(BaseWeightQuantizerAutogradFunction):
        """
        Custom autograd function for uniform weights quantizer.
        It provides a way to define a custom forward and symbolic operation
        and currently does not implement a backward operation.
        """

        @staticmethod
        def forward(ctx, input_tensor, num_bits, min_range, max_range, per_channel, channel_axis):
            """
             Forward computation function. This method performs the forward computation using
             the given quantize_sym_weights_torch function.

             Args:
                 ctx: An object that can be used to stash information for backward function.
                 input_tensor: The input tensor to be quantized.
                 num_bits: The number of bits to represent the quantized tensor.
                 min_range: min quantization range for quantizing weights
                 max_range: max quantization range for quantizing weights
                 per_channel: whether to use per-channel quantization
                 channel_axis: Axis of input to apply per-channel quantization on.

             Returns:
                 The quantized tensor.
             """
            return quantize_uniform_weights_torch(input_tensor, num_bits, min_range, max_range, per_channel,
                                                  channel_axis)

        @staticmethod
        def symbolic(g, input_tensor, num_bits, min_range, max_range, per_channel, channel_axis):
            """
            Symbolic method that defines the custom operation for ONNX export.

            Args:
                g: A graph object that represents the ONNX computation graph.
                input_tensor: The input tensor to be quantized.
                num_bits: The number of bits to represent the quantized value.
                min_range: min quantization range for quantizing weights
                max_range: max quantization range for quantizing weights
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.

            Returns:
                The node in the ONNX graph representing the output of this operation.
            """
            # When None is passed as channel_axis, the op has no attribute of channel_axis,
            # which creates conflict with the onnxruntime function. For this reason, if we quantize
            # per-tensor and channel_axis is None, we set it to 0.
            if not per_channel and channel_axis is None:
                channel_axis = 0

            return g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsUniformQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(min_range, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(max_range, dtype=torch.float32)),
                        num_bits_i=num_bits,
                        per_channel_i=int(per_channel),
                        channel_axis_i=channel_axis,
                        signed_i=WeightsUniformF.is_signed(),
                        **WeightsUniformF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())


else:
    class WeightsUniformInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing torch is mandatory '
                         'when using WeightsUniformInferableQuantizer. '
                         'Could not find torch package.')

if FOUND_ONNXRUNTIME_EXTENSIONS:
    from onnxruntime_extensions import onnx_op, PyCustomOpDef

    def quantize_uniform_weights_numpy(input_tensor: np.ndarray,
                                       num_bits: int,
                                       min_range: np.ndarray,
                                       max_range: np.ndarray,
                                       per_channel: bool,
                                       channel_axis: int=None):
        """
           Quantizes the input tensor uniformly using numpy.

           Args:
               input_tensor (np.ndarray): The input tensor to be quantized.
               num_bits (int): Number of bits to represent the quantized value.
                min_range (np.ndarray): min quantization range for quantizing weights
                max_range (np.ndarray): max quantization range for quantizing weights
                per_channel (bool): Quantize input tensor per-channel or per-tensor.
               channel_axis (int): Axis to quantize the tensor in case of per-channel quantization.

           Returns:
               Uniformly quantized tensor.
        """

        validate_weight_params(input_tensor=input_tensor,
                               per_channel=per_channel,
                               min_range=min_range,
                               max_range=max_range,
                               channel_axis=channel_axis)

        # adjusts the quantization rage so the quantization grid include zero.
        a, b = adjust_range_to_include_zero(min_range, max_range, num_bits)

        # Compute the step size of quantized values.
        delta = (b - a) / (2 ** num_bits - 1)
        if per_channel:
            ones = np.ones(input_tensor.ndim)
            ones[channel_axis] = -1
            new_shape = tuple([int(x) for x in ones])
            # Make sure min_values and max_values have the same shape as x along the first axis
            a = np.reshape(a, new_shape)
            b = np.reshape(b, new_shape)
            delta = np.reshape(delta, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = np.where(input_tensor < a, a, input_tensor)
        quantized = np.round(np.where(input_tensor > b, b, clipped_x) / delta) * delta
        return quantized

    # Add onnx op function to use during onnxruntime WeightsUniformQuantizer op inference
    # Using this decorator the op WeightsUniformQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsUniformQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float
                     ],
             outputs=[PyCustomOpDef.dt_float],
             attrs={
                 "num_bits": PyCustomOpDef.dt_int64,
                 "per_channel": PyCustomOpDef.dt_int64,
                 "channel_axis": PyCustomOpDef.dt_int64,
             }
             )
    def weight_uniform_ort(x, min_range, max_range, **kwargs):
        return quantize_uniform_weights_numpy(x,
                                              kwargs["num_bits"],
                                              min_range,
                                              max_range,
                                              kwargs["per_channel"],
                                              kwargs["channel_axis"]
                                              )
