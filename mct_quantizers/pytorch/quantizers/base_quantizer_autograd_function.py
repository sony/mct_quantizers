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
from typing import Any, Dict

import torch
from mct_quantizers import __version__ as mctq_version
from mct_quantizers.common.constants import MCTQ_VERSION


class BaseQuantizerAutogradFunction(torch.autograd.Function):
    """
    Custom autograd function for quantizer.
    It provides a way to define a custom forward and symbolic operation
    and currently does not implement a backward operation.
    """

    @staticmethod
    def forward(ctx, input_tensor, **kwargs):
        """
         Forward computation function. This method performs the forward computation using
         the given quantize_sym_weights_torch function.
         """
        raise NotImplemented

    @staticmethod
    def symbolic(g, input_tensor, **kwargs):
        raise NotImplemented

    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backward computation function. Raises a NotImplementedError
        since backward is not needed for this op.

        Args:
            ctx (Any): A context object from the forward pass.
            grad_outputs (Any): Gradients w.r.t. the output tensor.
        """
        raise NotImplementedError()

    @staticmethod
    def _get_metadata_attributes() -> Dict[str,Any]:
        """

        Returns: Metadata dictionary for all quantizers onnx symbolic ops.

        """
        return {f"{MCTQ_VERSION}_s": mctq_version}