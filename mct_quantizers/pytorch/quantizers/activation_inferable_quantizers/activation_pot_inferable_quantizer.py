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
    from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import ActivationSymmetricInferableQuantizer, quantize_sym_activations_torch
    from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.base_activation_quantizer_autograd_function import BaseActivationQuantizerAutogradFunction

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO],
                    identifier=QuantizerID.INFERABLE)
    class ActivationPOTInferableQuantizer(ActivationSymmetricInferableQuantizer):
        """
        Class for quantizing activations using power-of-two quantizer
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
            # target of Activation quantization
            super(ActivationPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                  signed=signed,
                                                                  threshold=threshold)

            is_threshold_pot = np.all(
                np.round(np.log2(self.threshold_np.flatten())) == np.log2(self.threshold_np.flatten()))
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

        def __call__(self, inputs):
            """
            Quantize an input tensor. If custom implementation is enabled, quantizer's autograd
            function class is applied.

            Args:
                inputs: Activation tensor to quantize.

            Returns:
                Quantized tensor.
            """
            if self._use_custom_impl and torch.jit.is_tracing():
                return ActivationPOTF.apply(inputs,
                                            self.threshold_np,
                                            self.signed,
                                            self.num_bits)
            return super(ActivationPOTInferableQuantizer, self).__call__(inputs)


    class ActivationPOTF(BaseActivationQuantizerAutogradFunction):
        """
        Custom autograd function for POT activations quantizer.
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
            added_op = g.op(f"{ONNX_CUSTOM_OP_DOMAIN}::ActivationPOTQuantizer",
                        input_tensor,
                        threshold_f=threshold,
                        signed_i=int(signed),
                        num_bits_i=num_bits,
                            **ActivationPOTF._get_metadata_attributes()
                        ).setType(
                input_tensor.type())
            return added_op


else:
    class ActivationPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationPOTInferableQuantizer. '
                            'Could not find torch package.')

if FOUND_ONNXRUNTIME_EXTENSIONS:
    from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import quantize_sym_activations_numpy
    from onnxruntime_extensions import onnx_op, PyCustomOpDef

    # Add onnx op function to use during onnxruntime ActivationPOTQuantizer op inference.
    # Using this decorator the op ActivationPOTQuantizer is defined using its inputs, outputs and attributes.
    @onnx_op(op_type=f"{ONNX_CUSTOM_OP_DOMAIN}::ActivationPOTQuantizer",
             inputs=[PyCustomOpDef.dt_float],
             outputs=[PyCustomOpDef.dt_float],
             attrs={"threshold": PyCustomOpDef.dt_float,
                    "signed": PyCustomOpDef.dt_int64,
                    "num_bits": PyCustomOpDef.dt_int64
                    })
    def activation_pot_ort(input_tensor,
                           **kwargs):
        return quantize_sym_activations_numpy(input_tensor,
                                              kwargs["threshold"],
                                              kwargs["signed"],
                                              kwargs["num_bits"]
                                              )
