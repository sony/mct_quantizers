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

from mct_quantizers.common.constants import FOUND_TORCH
from mct_quantizers.logger import Logger

if FOUND_TORCH:
    import torch

    def pytorch_load_quantized_model(filepath: str, **kwargs):
        """
        This function wraps the Pytorch load model.

        Args:
            filepath: the model file path.
            kwargs: Key-word arguments to pass to Pytorch load function.

        Returns: A Pytorch Model

        """
        return torch.load(filepath, **kwargs)

else:
    def pytorch_load_quantized_model(filepath, **kwargs):
        """
        This function wraps the Pytorch load model.

        Args:
            filepath: the model file path.
            kwargs: Key-word arguments to pass to Pytorch load function.

        Returns: A Pytorch Model

        """
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_load_quantized_model. '
                        'Could not find torch package.')  # pragma: no cover
