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
from mct_quantizers.common.quant_utils import adjust_range_to_include_zero
from mct_quantizers.pytorch.quantizer_utils import fix_range_to_include_zero

if FOUND_TORCH:
    import torch
    from mct_quantizers.pytorch.quantizers.base_uniform_inferable_quantizer import BaseUniformInferableQuantizer

    from onnxruntime_extensions import onnx_op, PyCustomOpDef

    # Add onnx op function to use during onnxruntime ActivationUniformQuantizer op inference
    @onnx_op(op_type="ActivationUniformQuantizer",
             inputs=[PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_float])
    def activation_uniform_ort(input_tensor: np.ndarray,
                               min_range: float,
                               max_range: float,
                               num_bits: int):
        return quantize_uniform_activations_numpy(input_tensor,
                                                  min_range,
                                                  max_range,
                                                  num_bits)


    def quantize_uniform_activations_torch(tensor_data: torch.Tensor,
                                           range_min: float,
                                           range_max: float,
                                           n_bits: int) -> np.ndarray:
        """
       Quantizes the input tensor uniformly using PyTorch.

       Args:
           tensor_data (torch.Tensor): The input tensor to be quantized.
           range_min (float): The quantization min range.
           range_max (float): The quantization max range.
           n_bits (int): Number of bits to represent the quantized value.

       Returns:
           torch.Tensor: Uniformly quantized tensor.
       """

        range_min = torch.tensor([range_min])
        range_max = torch.tensor([range_max])

        # adjusts the quantization rage so the quantization grid include zero.
        a, b = fix_range_to_include_zero(range_min, range_max, n_bits)

        # Compute the step size of quantized values.
        delta = (b - a) / (2 ** n_bits - 1)

        # Clip data in range
        clipped_tensor = torch.clip(tensor_data, min=a, max=b)

        # Quantize the data between min/max of quantization range.
        q = delta * torch.round((clipped_tensor - a) / delta) + a
        return q


    def quantize_uniform_activations_numpy(tensor_data: np.ndarray,
                                           range_min: float,
                                           range_max: float,
                                           n_bits: int) -> np.ndarray:
        """
        Quantize a tensor according to given range (min, max) and number of bits.

        Args:
            tensor_data: Tensor values to quantize.
            range_min: minimum bound of the range for quantization (or array of min values per channel).
            range_max: maximum bound of the range for quantization (or array of max values per channel).
            n_bits: Number of bits to quantize the tensor.

        Returns:
            Quantized data.
        """

        # adjusts the quantization rage so the quantization grid include zero.
        a, b = adjust_range_to_include_zero(range_min, range_max, n_bits)

        # Compute the step size of quantized values.
        delta = (b - a) / (2 ** n_bits - 1)

        # Clip data in range
        clipped_tensor = np.clip(tensor_data, a_min=a, a_max=b)

        # Quantize the data between min/max of quantization range.
        q = delta * np.round((clipped_tensor - a) / delta) + a
        return q


    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    identifier=QuantizerID.INFERABLE)
    class ActivationUniformInferableQuantizer(BaseUniformInferableQuantizer):
        """
        Class for quantizing activations using an uniform quantizer
        """

        def __init__(self,
                     num_bits: int,
                     min_range: np.ndarray,
                     max_range: np.ndarray,
                     use_custom_impl: bool = False
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min range for quantizing activations
                max_range: max range for quantizing activations
            """
            super(ActivationUniformInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                      min_range=min_range,
                                                                      max_range=max_range,
                                                                      use_custom_impl=use_custom_impl)

            assert isinstance(min_range,list), f'min_range is expected to be a list, but is of type {type(min_range)}'
            assert isinstance(max_range,list), f'max_range is expected to be a list, but is of type {type(max_range)}'

            assert len(
                min_range) == 1, f'For activation, quantization per channel is not supported and min_range should be ' \
                                 f'of length 1 but is {len(min_range)}'
            assert len(
                max_range) == 1, f'For activation, quantization per channel is not supported and max_range should be ' \
                                 f'of length 1 but is {len(max_range)}'

            # Activation is per-tensor thus we expect only a single min/max values
            min_range = min_range[0]
            max_range = max_range[0]

            # fixing quantization range to include 0
            a = 0 if min_range > 0 else min_range
            b = 0 if max_range < 0 else max_range

            self.min_range = a
            self.max_range = b

            self.scale = float((b - a) / ((2 ** num_bits) - 1))
            self.zero_point = int(-np.round(a / self.scale))  # zp has to be positive, and a <=0, so we multiply by -1

        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            if self.use_custom_impl and torch.jit.is_tracing():
                return ActivationUniformF.apply(inputs, self.min_range, self.max_range, self.num_bits)
            else:
                with torch.no_grad():
                    return torch.fake_quantize_per_tensor_affine(inputs,
                                                                 scale=self.scale,
                                                                 zero_point=self.zero_point,
                                                                 quant_min=self.min_quantized_domain,
                                                                 quant_max=self.max_quantized_domain)


    class ActivationUniformF(torch.autograd.Function):
        """
        Custom autograd function for uniform activations quantizer.
        It provides a way to define a custom forward and symbolic operation
        and currently does not implement a backward operation.
        """

        @staticmethod
        def forward(ctx, input_tensor, min_range, max_range, num_bits):
            """
             Forward computation function. This method performs the forward computation using
             the given quantize_uniform_activations_torch function.

             Args:
                 ctx: An object that can be used to stash information for backward function.
                 input_tensor: The input tensor to be quantized.
                 min_range: The quantization min range.
                 max_range: The quantization max range.
                 num_bits: The number of bits to represent the quantized tensor.

             Returns:
                 The quantized tensor.
             """
            return quantize_uniform_activations_torch(input_tensor, min_range, max_range, num_bits)

        @staticmethod
        def symbolic(g, input_tensor, min_range, max_range, num_bits):
            """
            Symbolic method that defines the custom operation for ONNX export.

            Args:
                g: A graph object that represents the ONNX computation graph.
                input_tensor: The input tensor to be quantized.
                min_range: The quantization min range.
                max_range: The quantization max range.
                num_bits: The number of bits to represent the quantized value.

            Returns:
                The node in the ONNX graph representing the output of this operation.
            """
            return g.op("ai.onnx.contrib::ActivationUniformQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(min_range, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(max_range, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(num_bits, dtype=torch.int64))).setType(
                input_tensor.type())

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            """
            Backward computation function. Raises a NotImplementedError
            since backward is not needed for this op.

            Args:
                ctx (Any): A context object from the forward pass.
                grad_outputs (Any): Gradients w.r.t. the output tensor.
            """
            raise NotImplementedError()

else:
    class ActivationUniformInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationUniformInferableQuantizer. '
                            'Could not find torch package.')
