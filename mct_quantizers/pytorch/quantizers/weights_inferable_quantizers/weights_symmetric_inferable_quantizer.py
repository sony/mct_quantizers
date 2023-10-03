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
from mct_quantizers.pytorch.onnxruntime_validations import validate_weight_params

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_symmetric_inferable_quantizer import BaseSymmetricInferableQuantizer
    from mct_quantizers.pytorch.quantizer_utils import to_torch_tensor, get_working_device
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.base_weight_quantizer_autograd_function import \
        BaseWeightQuantizerAutogradFunction


    def quantize_sym_weights_torch(input_tensor: torch.Tensor,
                                   num_bits: int,
                                   threshold: float,
                                   per_channel: bool,
                                   channel_axis: int):
        """
           Quantizes the input tensor symmetrically using torch.

           Args:
               input_tensor (torch.Tensor): The input tensor to be quantized.
               num_bits (int): Number of bits to represent the quantized value.
               threshold (float): The quantization threshold.
               per_channel (bool): Quantize input tensor per-channel or per-tensor.
               channel_axis (int): Axis to quantize the tensor in case of per-channel quantization.

           Returns:
               Symmetrically quantized tensor.
        """

        if isinstance(threshold, np.ndarray):
            threshold = torch.tensor(threshold, dtype=torch.float32).to(get_working_device())

        input_tensor = input_tensor
        scale = threshold / (2 ** (num_bits - 1))
        _min, _max = -threshold, threshold - scale

        if per_channel:
            ones = [1] * input_tensor.ndim
            ones[channel_axis] = -1
            new_shape = tuple(ones)
            _min = torch.reshape(_min, new_shape)
            _max = torch.reshape(_max, new_shape)
            scale = torch.reshape(scale, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = torch.where(input_tensor < _min, _min, input_tensor)
        quantized = torch.round(torch.where(input_tensor > _max, _max, clipped_x) / scale) * scale

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
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None
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
                                                                     signed=True)

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

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """

            if self._use_custom_impl and torch.jit.is_tracing():
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


    class WeightsSymmetricF(BaseWeightQuantizerAutogradFunction):
        """
        Custom autograd function for symmetric weights quantizer.
        It provides a way to define a custom forward and symbolic operation
        and currently does not implement a backward operation.
        """

        @staticmethod
        def forward(ctx, input_tensor, num_bits, threshold, per_channel, channel_axis):
            """
             Forward computation function. This method performs the forward computation using
             the given quantize_sym_weights_torch function.

             Args:
                 ctx: An object that can be used to stash information for backward function.
                 input_tensor: The input tensor to be quantized.
                 num_bits: The number of bits to represent the quantized tensor.
                 threshold: The quantization threshold.
                 per_channel: whether to use per-channel quantization
                 channel_axis: Axis of input to apply per-channel quantization on.

             Returns:
                 The quantized tensor.
             """
            return quantize_sym_weights_torch(input_tensor, num_bits, threshold, per_channel, channel_axis)

        @staticmethod
        def symbolic(g, input_tensor, num_bits, threshold, per_channel, channel_axis):
            """
            Symbolic method that defines the custom operation for ONNX export.

            Args:
                g: A graph object that represents the ONNX computation graph.
                input_tensor: The input tensor to be quantized.
                num_bits: The number of bits to represent the quantized value.
                threshold: The quantization threshold.
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

            return g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsSymmetricQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        num_bits_i=num_bits,
                        per_channel_i=int(per_channel),
                        channel_axis_i=channel_axis,
                        signed_i=int(WeightsSymmetricF.is_signed()),
                        **WeightsSymmetricF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())



else:
    class WeightsSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsSymmetricInferableQuantizer. '
                            'Could not find torch package.')

if FOUND_ONNXRUNTIME_EXTENSIONS:
    from onnxruntime_extensions import onnx_op, PyCustomOpDef
    def quantize_sym_weights_numpy(input_tensor: np.ndarray,
                                   num_bits: int,
                                   threshold: np.ndarray,
                                   per_channel: int,
                                   channel_axis: int=None):
        """
           Quantizes the input tensor symmetrically using numpy.

           Args:
               input_tensor (np.ndarray): The input tensor to be quantized.
               num_bits (int): Number of bits to represent the quantized value.
               threshold (float): The quantization threshold.
               per_channel (bool): Quantize input tensor per-channel or per-tensor.
               channel_axis (int): Axis to quantize the tensor in case of per-channel quantization.

           Returns:
               Symmetrically quantized tensor.
        """
        validate_weight_params(input_tensor=input_tensor,
                               per_channel=per_channel,
                               min_range=-threshold,
                               max_range=threshold,
                               channel_axis=channel_axis)

        scale = threshold / (2 ** (num_bits - 1))
        _min, _max = -threshold, threshold - scale
        if per_channel:
            ones = np.ones(input_tensor.ndim)
            ones[channel_axis] = -1
            new_shape = tuple([int(x) for x in ones])
            _min = np.reshape(_min, new_shape)
            _max = np.reshape(_max, new_shape)
            scale = np.reshape(scale, new_shape)

        # Use torch.where to clip the values in x
        clipped_x = np.where(input_tensor < _min, _min, input_tensor)
        quantized = np.round(np.where(input_tensor > _max, _max, clipped_x) / scale) * scale
        return quantized


    # Add onnx op function to use during onnxruntime WeightsSymmetricQuantizer op inference
    # Using this decorator the op WeightsSymmetricQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsSymmetricQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float],
             outputs=[PyCustomOpDef.dt_float],
             attrs={
                 "num_bits": PyCustomOpDef.dt_int64,
                 "per_channel": PyCustomOpDef.dt_int64,
                 "channel_axis": PyCustomOpDef.dt_int64,
             })
    def weight_sym_ort(input_tensor: np.ndarray,
                       threshold: np.ndarray,
                       **kwargs):

        return quantize_sym_weights_numpy(input_tensor,
                                          kwargs["num_bits"],
                                          threshold,
                                          kwargs["per_channel"],
                                          kwargs["channel_axis"]
                                          )
