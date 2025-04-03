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
from abc import abstractmethod

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import FOUND_TORCH

if FOUND_TORCH:
    import torch


    class BasePyTorchInferableQuantizer(BaseInferableQuantizer):
        def __init__(self):
            """
            This class is a base quantizer for PyTorch quantizers for inference only.
            """
            super(BasePyTorchInferableQuantizer, self).__init__()
            # By default the custom forward implementation is disabled. If someone wants to enable it
            # enable_custom_impl should be invoked.
            self._use_custom_impl = False

            # Reuse output: run only first quantizer opertion, save the result
            # and return it for others quantizer operation
            self.reuse = False
            self.enable_reuse = False
            self.quantizer_first_run = True
            self.resue_outputs = None

        def enable_custom_impl(self):
            self._use_custom_impl = True

        def enable_reuse_quantizer(self):
            self.enable_reuse = True
            self.quantizer_first_run = True

        def disable_reuse_quantizer(self):
            self.enable_reuse = False

        @abstractmethod
        def __call__(self, inputs: torch.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            raise NotImplemented(f'{self.__class__.__name__} did not implement __call__')  # pragma: no cover
else:
    class BasePyTorchInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BasePyTorchInferableQuantizer. '
                            'Could not find torch package.')
