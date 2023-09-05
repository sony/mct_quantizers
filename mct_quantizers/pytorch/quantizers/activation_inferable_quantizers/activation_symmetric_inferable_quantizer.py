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
from mct_quantizers.pytorch.onnxruntime_validations import validate_activation_params

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_symmetric_inferable_quantizer import BaseSymmetricInferableQuantizer
    from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.base_activation_quantizer_autograd_function import BaseActivationQuantizerAutogradFunction

    def quantize_sym_activations_torch(input_tensor: torch.Tensor,
                                       threshold: float,
                                       signed: bool,
                                       num_bits: int):
        """
        Quantizes the input tensor symmetrically using PyTorch.

        Args:
            input_tensor (torch.Tensor): The input tensor to be quantized.
            threshold (float): The quantization threshold.
            signed (bool): A flag indicating whether the quantization is signed or unsigned.
            num_bits (int): Number of bits to represent the quantized value.

        Returns:
            torch.Tensor: Symmetrically quantized tensor.
        """
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
                     signed: bool):
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

            assert len(threshold)==1, f'For activation, only per-tensor quantization is supported. Thus, threshold should be of length 1 but is {len(threshold)}'

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
            if self._use_custom_impl and torch.jit.is_tracing():
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


    class ActivationSymF(BaseActivationQuantizerAutogradFunction):
        """
        Custom autograd function for Symmetric activations quantizer.
        It provides a way to define a custom forward and symbolic operation
        and currently does not implement a backward operation.
        """

        @staticmethod
        def forward(ctx, input_tensor, threshold, signed, num_bits):
            """
             Forward computation function. This method performs the forward computation using
             the given quantize_sym_activations_torch function.

             Args:
                 ctx: An object that can be used to stash information for backward function.
                 input_tensor: The input tensor to be quantized.
                 threshold: The quantization threshold.
                 signed: A flag that indicates if the quantization is signed or unsigned.
                 num_bits: The number of bits to represent the quantized tensor.

             Returns:
                 The quantized tensor.
             """
            return quantize_sym_activations_torch(input_tensor, threshold, signed, num_bits)

        @staticmethod
        def symbolic(g, input_tensor, threshold, signed, num_bits):
            """
            Symbolic method that defines the custom operation for ONNX export.

            Args:
                g: A graph object that represents the ONNX computation graph.
                input_tensor: The input tensor to be quantized.
                threshold: The quantization threshold.
                signed: A flag that indicates if the quantization is signed or unsigned.
                num_bits: The number of bits to represent the quantized value.

            Returns:
                The node in the ONNX graph representing the output of this operation.
            """
            return g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::ActivationSymmetricQuantizer", input_tensor,
                        threshold_f=threshold,
                        signed_i=int(signed),
                        num_bits_i=num_bits,
                        **ActivationSymF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())


else:
    class ActivationSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find torch package.')


if FOUND_ONNXRUNTIME_EXTENSIONS:
    from onnxruntime_extensions import onnx_op, PyCustomOpDef

    # Add onnx op function to use during onnxruntime ActivationSymmetricQuantizer op inference
    # Using this decorator the op ActivationSymmetricQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::ActivationSymmetricQuantizer",
             inputs=[PyCustomOpDef.dt_float],
             outputs=[PyCustomOpDef.dt_float],
             attrs={"threshold": PyCustomOpDef.dt_float,
                    "signed": PyCustomOpDef.dt_int64,
                    "num_bits": PyCustomOpDef.dt_int64
                    }
             )
    def activation_sym_ort(input_tensor,
                           **kwargs):
        return quantize_sym_activations_numpy(input_tensor,
                                              kwargs["threshold"],
                                              kwargs["signed"],
                                              kwargs["num_bits"])


    def quantize_sym_activations_numpy(input_tensor: np.ndarray,
                                       threshold: float,
                                       signed: bool,
                                       num_bits: int):
        """
           Quantizes the input tensor symmetrically using numpy.

           Args:
               input_tensor (np.ndarray): The input tensor to be quantized.
               threshold (float): The quantization threshold.
               signed (bool): A flag indicating whether the quantization is signed or unsigned.
               num_bits (int): Number of bits to represent the quantized value.

           Returns:
               np.ndarray: Symmetrically quantized tensor.
        """

        validate_activation_params(input_tensor=input_tensor,
                               min_range=-threshold if signed else 0.,
                               max_range=threshold)

        if signed:
            scale = threshold / (2 ** (num_bits - 1))
            min, max = -threshold, threshold - scale
        else:
            scale = threshold / (2 ** num_bits)
            min, max = 0, threshold - scale

        quantized = np.round(np.clip(input_tensor, min, max) / scale) * scale
        return quantized
