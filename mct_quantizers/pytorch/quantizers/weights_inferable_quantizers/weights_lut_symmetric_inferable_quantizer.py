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
from mct_quantizers.common.constants import FOUND_TORCH, LUT_VALUES_BITWIDTH, EPS, ONNX_CUSTOM_OP_DOMAIN, \
    FOUND_ONNXRUNTIME_EXTENSIONS
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.common.quant_utils import lut_quantizer_np

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizer_utils import to_torch_tensor, get_working_device, lut_quantizer
    from mct_quantizers.pytorch.quantizers.base_lut_symmetric_inferable_quantizer import \
        BaseLUTSymmetricInferableQuantizer
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.base_weight_quantizer_autograd_function import \
        BaseWeightQuantizerAutogradFunction


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.LUT_SYM_QUANTIZER],
                    identifier=QuantizerID.INFERABLE)
    class WeightsLUTSymmetricInferableQuantizer(BaseLUTSymmetricInferableQuantizer):
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
                channel_axis: Axis of input to apply per-channel quantization on
                lut_values_bitwidth: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """

            super(WeightsLUTSymmetricInferableQuantizer, self).__init__(threshold=threshold,
                                                                        num_bits=num_bits,
                                                                        lut_values=lut_values,
                                                                        signed=True,
                                                                        lut_values_bitwidth=lut_values_bitwidth,
                                                                        eps=eps)

            self.per_channel = per_channel
            self.channel_axis = channel_axis
            self.input_rank = input_rank

            if per_channel:
                assert channel_axis is not None, f'Channel axis is missing in per channel quantization'
                assert input_rank is not None, f'input_rank is missing in per channel quantization'
                assert len(
                    threshold) >= 1, f'In per-channel quantization threshold should be of length >= 1 but is ' \
                                     f'{len(threshold)}'
            else:
                assert len(
                    threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is ' \
                                     f'{len(threshold)}'

            self._threshold_torch = to_torch_tensor(self._threshold_np).to(get_working_device())
            self._lut_values_torch = to_torch_tensor(self._lut_values_np).to(get_working_device())

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            if self._use_custom_impl and torch.jit.is_tracing():
                return WeightsLUTSymmetricF.apply(inputs,
                                                  self.num_bits,
                                                  self._lut_values_np,
                                                  self._threshold_np,
                                                  self.lut_values_bitwidth,
                                                  self.eps,
                                                  self.per_channel,
                                                  self.channel_axis,
                                                  self.input_rank
                                                  )

            inputs.requires_grad = False
            return lut_quantizer(inputs,
                                 lut_values=self._lut_values_torch,
                                 signed=True,
                                 threshold=self._threshold_torch,
                                 lut_values_bitwidth=self.lut_values_bitwidth,
                                 eps=self.eps,
                                 per_channel=self.per_channel,
                                 channel_axis=self.channel_axis,
                                 input_rank=self.input_rank
                                 )


    class WeightsLUTSymmetricF(BaseWeightQuantizerAutogradFunction):
        """
        Custom autograd function for symmetric weights quantizer.
        It provides a way to define a custom forward and symbolic operation
        and currently does not implement a backward operation.
        """

        @staticmethod
        def forward(ctx, input_tensor, num_bits, lut_values, threshold, lut_values_bitwidth, eps,
                    per_channel,
                    channel_axis,
                    input_rank
                    ):
            # Apply pass np arrays to support the symbolic op, so it needs to be converted to
            # torch since here we quantize a torch tensor
            threshold_torch = to_torch_tensor(threshold).to(get_working_device())
            lut_values_torch = to_torch_tensor(lut_values).to(get_working_device())

            return lut_quantizer(input_tensor,
                                 lut_values=lut_values_torch,
                                 signed=True,
                                 threshold=threshold_torch,
                                 lut_values_bitwidth=lut_values_bitwidth,
                                 eps=eps,
                                 per_channel=per_channel,
                                 channel_axis=channel_axis,
                                 input_rank=input_rank)

        @staticmethod
        def symbolic(g,
                     input_tensor,
                     num_bits,
                     lut_values,
                     threshold,
                     lut_values_bitwidth,
                     eps,
                     per_channel,
                     channel_axis,
                     input_rank):
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

            return g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsLUTSymmetricQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(lut_values, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        num_bits_i=num_bits,
                        per_channel_i=int(per_channel),
                        channel_axis_i=channel_axis,
                        input_rank_i=input_rank,
                        lut_values_bitwidth_i=lut_values_bitwidth,
                        eps_f=eps,
                        signed_i=int(WeightsLUTSymmetricF.is_signed()),
                        **WeightsLUTSymmetricF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())


else:
    class WeightsLUTSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsLUTSymmetricInferableQuantizer. '
                            'Could not find torch package.')

if FOUND_ONNXRUNTIME_EXTENSIONS:
    from onnxruntime_extensions import onnx_op, PyCustomOpDef
    def quantize_lut_sym_weights_numpy(input_tensor: np.ndarray,
                                       lut_values,
                                       threshold,
                                       lut_values_bitwidth,
                                       eps,
                                       per_channel,
                                       channel_axis=None,
                                       input_rank=None
                                       ):
        quantized_tensor = lut_quantizer_np(tensor_data=input_tensor,
                                            lut_values=lut_values,
                                            signed=True,
                                            threshold=threshold,
                                            lut_values_bitwidth=lut_values_bitwidth,
                                            eps=eps,
                                            per_channel=per_channel,
                                            channel_axis=channel_axis,
                                            input_rank=input_rank)
        return quantized_tensor


    # Add onnx op function to use during onnxruntime WeightsLUTSymmetricQuantizer op inference
    # Using this decorator the op WeightsLUTSymmetricQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsLUTSymmetricQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float],
             outputs=[PyCustomOpDef.dt_float],
             attrs={
                 "lut_values_bitwidth": PyCustomOpDef.dt_int64,
                 "eps": PyCustomOpDef.dt_float,
                 "per_channel": PyCustomOpDef.dt_int64,
                 "channel_axis": PyCustomOpDef.dt_int64,
                 "input_rank": PyCustomOpDef.dt_int64
             })
    def weight_lut_sym_ort(input_tensor: np.ndarray,
                           lut_values: np.ndarray,
                           threshold: np.ndarray,
                           **kwargs):
        return quantize_lut_sym_weights_numpy(input_tensor,
                                              lut_values,
                                              threshold,
                                              kwargs["lut_values_bitwidth"],
                                              kwargs["eps"],
                                              kwargs["per_channel"],
                                              kwargs["channel_axis"],
                                              kwargs["input_rank"])
