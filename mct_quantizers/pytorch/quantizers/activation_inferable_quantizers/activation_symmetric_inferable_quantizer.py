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
from typing import Any, List

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.common.quant_info import QuantizationMethod

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_symmetric_inferable_quantizer import BaseSymmetricInferableQuantizer

    from onnxruntime_extensions import onnx_op, PyCustomOpDef


    @onnx_op(op_type="ActivationSymmetricQuantizer",
             inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_bool, PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_float])
    def activation_sym_ort(input_tensor: np.ndarray,
                           threshold: float,
                           signed: bool,
                           num_bits: int):
        return quantize_sym_activations_numpy(input_tensor, threshold, signed, num_bits)


    def quantize_sym_activations_numpy(input_tensor: np.ndarray,
                                       threshold: float,
                                       signed: bool,
                                       num_bits: int):
        if signed:
            scale = threshold / (2 ** (num_bits - 1))
            min, max = -threshold, threshold - scale
        else:
            scale = threshold / (2 ** num_bits)
            min, max = 0, threshold - scale

        quantized = np.round(np.clip(input_tensor, min, max) / scale) * scale
        return quantized


    def quantize_sym_activations_torch(input_tensor: torch.Tensor,
                                       threshold: float,
                                       signed: bool,
                                       num_bits: int):
        if signed:
            scale = threshold / (2 ** (num_bits - 1))
            min, max = -threshold, threshold - scale
        else:
            scale = threshold / (2 ** num_bits)
            min, max = 0, threshold - scale

        quantized = torch.round(torch.clip(input_tensor, min, max) / scale) * scale
        return quantized


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    identifier=QuantizerID.INFERABLE)
    class ActivationSymmetricInferableQuantizer(BaseSymmetricInferableQuantizer):
        """
        Class for quantizing activations using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     signed: bool,
                     use_custom_impl=False):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether to use signed quantization or not
            """

            super(ActivationSymmetricInferableQuantizer, self).__init__(
                num_bits=num_bits,
                threshold=threshold,
                signed=signed)

            self.use_custom_impl = use_custom_impl

            assert self.threshold_np.shape[0] == 1
            self.threshold_np = self.threshold_np[0]

            # Activation supports only per-tensor quantization
            assert len(
                self.scales) == 1, f'For activation, quantization per channel is not supported and threshold should ' \
                                   f'be of length 1 but is {len(threshold)}'
            self.scales = self.scales[0]

            self.zero_points = 0

        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            if self.use_custom_impl and torch.jit.is_tracing():
                return ActivationSymF.apply(inputs,
                                            self.threshold_np,
                                            self.signed,
                                            self.num_bits)
            else:
                with torch.no_grad():
                    return torch.fake_quantize_per_tensor_affine(inputs,
                                                                 scale=self.scales,
                                                                 zero_point=self.zero_points,
                                                                 quant_min=self.min_quantized_domain,
                                                                 quant_max=self.max_quantized_domain)


    class ActivationSymF(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, threshold, signed, num_bits):
            return quantize_sym_activations_torch(input_tensor, threshold, signed, num_bits)

        @staticmethod
        def symbolic(g, input_tensor, threshold, signed, num_bits):
            return g.op("ai.onnx.contrib::ActivationSymmetricQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(signed, dtype=torch.bool)),
                        g.op('Constant', value_t=torch.tensor(num_bits, dtype=torch.int64))).setType(
                input_tensor.type())

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            raise NotImplementedError()

else:
    class ActivationSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find torch package.')
