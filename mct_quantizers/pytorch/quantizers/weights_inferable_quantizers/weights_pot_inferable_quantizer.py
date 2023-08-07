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
from onnxruntime_extensions import onnx_op, PyCustomOpDef

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.common.quant_info import QuantizationMethod

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
        WeightsSymmetricInferableQuantizer, quantize_sym_weights_numpy, quantize_sym_weights_torch


    @onnx_op(op_type="WeightsPOTQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_int64,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_bool,
                     PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_float])
    def weight_pot_ort(x, nbits, t, pc, axis):
        return quantize_sym_weights_numpy(x, nbits, t, pc, axis)


    @mark_quantizer(quantization_target=QuantizationTarget.Weights,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO],
                    identifier=QuantizerID.INFERABLE)
    class WeightsPOTInferableQuantizer(WeightsSymmetricInferableQuantizer):
        """
        Class for quantizing weights using a power-of-two quantizer.
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
                threshold: threshold for quantizing activations
                per_channel: whether to use per-channel quantization
                channel_axis: Axis of input to apply per-channel quantization on.
            """
            # target of Weights quantization
            super(WeightsPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               threshold=threshold,
                                                               per_channel=per_channel,
                                                               channel_axis=channel_axis,
                                                               use_custom_impl=use_custom_impl)
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

            if self.use_custom_impl and torch.jit.is_tracing():
                return WeightsPOTF.apply(inputs,
                                         self.num_bits,
                                         self.threshold_np,
                                         self.per_channel,
                                         self.channel_axis)

            return super(WeightsPOTInferableQuantizer, self).__call__(inputs)


    class WeightsPOTF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, num_bits, threshold, per_channel, channel_axis):
            return quantize_sym_weights_torch(input_tensor, num_bits, threshold, per_channel, channel_axis)

        @staticmethod
        def symbolic(g, input_tensor, num_bits, threshold, per_channel, channel_axis):
            return g.op("ai.onnx.contrib::WeightsPOTQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(num_bits, dtype=torch.int64)),
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(per_channel, dtype=torch.bool)),
                        g.op('Constant', value_t=torch.tensor(channel_axis, dtype=torch.int64))).setType(
                input_tensor.type())

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            raise NotImplementedError()

else:
    class WeightsPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find torch package.')
