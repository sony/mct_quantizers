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
    from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_symmetric_inferable_quantizer import ActivationSymmetricInferableQuantizer, quantize_sym_activations_numpy, quantize_sym_activations_torch
    from onnxruntime_extensions import onnx_op, PyCustomOpDef

    @onnx_op(op_type="ActivationPOTQuantizer",
             inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float,
                     PyCustomOpDef.dt_bool, PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_float])
    def activation_pot_ort(x, t, s, nbits):
        return quantize_sym_activations_numpy(x, t, s, nbits)


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
                     signed: bool,
                     use_custom_impl: bool = False):
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
                                                                  threshold=threshold,
                                                                  use_custom_impl=use_custom_impl)

            is_threshold_pot = np.all(
                np.round(np.log2(self.threshold_np.flatten())) == np.log2(self.threshold_np.flatten()))
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

        def __call__(self, inputs):
            if self.use_custom_impl and torch.jit.is_tracing():
                return ActivationPOTF.apply(inputs,
                                            self.threshold_np,
                                            self.signed,
                                            self.num_bits)
            return super(ActivationPOTInferableQuantizer, self).__call__(inputs)


    class ActivationPOTF(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input_tensor, threshold, signed, num_bits):
            return quantize_sym_activations_torch(input_tensor, threshold, signed, num_bits)

        @staticmethod
        def symbolic(g, input_tensor, threshold, signed, num_bits):
            return g.op("ai.onnx.contrib::ActivationPOTQuantizer", input_tensor,
                        g.op('Constant', value_t=torch.tensor(threshold, dtype=torch.float32)),
                        g.op('Constant', value_t=torch.tensor(signed, dtype=torch.bool)),
                        g.op('Constant', value_t=torch.tensor(num_bits, dtype=torch.int64))).setType(
                input_tensor.type())

        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            raise NotImplementedError()

else:
    class ActivationPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using ActivationPOTInferableQuantizer. '
                            'Could not find torch package.')
