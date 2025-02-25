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


if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
        WeightsSymmetricInferableQuantizer, quantize_sym_weights_torch
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.base_weight_quantizer_autograd_function import \
        BaseWeightQuantizerAutogradFunction


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO],
                    identifier=QuantizerID.INFERABLE)
    class WeightsPOTInferableQuantizer(WeightsSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a power-of-two quantizer.
        """

        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     per_channel: bool,
                     channel_axis: int = None,
                     ):

            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
            """
            # target of Weights quantization
            super(WeightsPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               threshold=threshold,
                                                               per_channel=per_channel,
                                                               channel_axis=channel_axis)
            self.num_bits = num_bits
            self.threshold = threshold
            self.per_channel = per_channel
            self.channel_axis = channel_axis

            is_threshold_pot = np.all(
                np.round(np.log2(self.threshold_np.flatten())) == np.log2(self.threshold_np.flatten()))
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            if self.reuse and not self.quantizer_first_run:
                return self.resue_outputs

            if self._use_custom_impl and torch.jit.is_tracing():
                outputs = WeightsPOTF.apply(inputs,
                                             self.num_bits,
                                             self.threshold_np,
                                             self.per_channel,
                                             self.channel_axis)
            else:
                outputs = super(WeightsPOTInferableQuantizer, self).__call__(inputs)

            if self.reuse and self.quantizer_first_run:
                self.resue_outputs = outputs
                self.quantizer_first_run = False

            return outputs

    class WeightsPOTF(BaseWeightQuantizerAutogradFunction):
        """
        Custom autograd function for POT weights quantizer.
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

            return g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsPOTQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        num_bits_i=num_bits,
                        per_channel_i=int(per_channel),
                        channel_axis_i=channel_axis,
                        signed_i=int(WeightsPOTF.is_signed()),
                        **WeightsPOTF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())


else:
    class WeightsPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find torch package.')

if FOUND_ONNXRUNTIME_EXTENSIONS:
    from onnxruntime_extensions import onnx_op, PyCustomOpDef
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
        quantize_sym_weights_numpy


    # Add onnx op function to use during onnxruntime WeightsPOTQuantizer op inference
    # Using this decorator the op WeightsPOTQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::WeightsPOTQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float],
             outputs=[PyCustomOpDef.dt_float],
             attrs={
                 "num_bits": PyCustomOpDef.dt_int64,
                 "per_channel": PyCustomOpDef.dt_int64,
                 "channel_axis": PyCustomOpDef.dt_int64,
             }
             )
    def weight_pot_ort(input_tensor: np.ndarray, threshold: np.ndarray, **kwargs):
        return quantize_sym_weights_numpy(input_tensor,
                                          kwargs["num_bits"],
                                          threshold,
                                          kwargs["per_channel"],
                                          kwargs["channel_axis"])
