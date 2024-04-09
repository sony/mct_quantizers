# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict

from mct_quantizers.common.constants import FOUND_TORCH, FRAMEWORK_VERSION, ONNX_VERSION, FOUND_ONNX

from mct_quantizers.logger import Logger
from mct_quantizers.common.metadata import verify_and_init_metadata

if FOUND_TORCH:
    import torch

    def add_metadata(model: torch.nn.Module, metadata: Dict):
        """
        Init the metadata dictionary and verify its compliance, then add it to the model under
        the 'metadata' attribute.

        Args:
            model (Module): Pytorch model to add the metadata to.
            metadata (Dict): The metadata dictionary.

        Returns:
            The model with attribute 'metadata' that holds the metadata dictionary.

        Example:
            Adding author name and model version to model's metadata

            >>> model_with_metadata = add_metadata(model, {'author': 'John Doe', 'model version': 3})
        """

        metadata = verify_and_init_metadata(metadata)
        if FRAMEWORK_VERSION not in metadata:
            metadata[FRAMEWORK_VERSION] = torch.__version__
        model.metadata = metadata
        return model

    def get_metadata(model: torch.nn.Module) -> Dict:
        """
        Get the metadata dictionary from model.

        Args:
            model (Module): Pytorch model to extract metadata from.

        Returns:
            The model's the metadata dictionary.

        Example:
                Get model's metadata.

                >>> metadata = get_metadata(model)
        """
        return getattr(model, 'metadata', {})

else:
    def add_metadata(model,
                     metadata):
            Logger.critical('Installing pytorch is mandatory '
                            'when using add_metadata. '
                            'Could not find Pytorch package.')  # pragma: no cover

    def get_metadata(model):
            Logger.critical('Installing pytorch is mandatory '
                            'when using get_metadata. '
                            'Could not find Pytorch package.')  # pragma: no cover


if FOUND_ONNX:
    import onnx

    def add_onnx_metadata(model: onnx.ModelProto, metadata: Dict):
        """
        Init the metadata dictionary and verify its compliance, then add it to the model metadata_props.

        Args:
            model (ModelProto): onnx model to add the metadata to.
            metadata (Dict): The metadata dictionary.

        Returns:
            The model with metadata dictionary.

        Example:
            Adding author name and model version to an onnx model's metadata

            >>> onnx_model_with_metadata = add_onnx_metadata(onnx_model, {'author': 'Jane Doe', 'model version': 3.5})
        """
        metadata = verify_and_init_metadata(metadata)
        if ONNX_VERSION not in metadata:
            metadata[ONNX_VERSION] = onnx.__version__

        for k, v in metadata.items():
            meta = model.metadata_props.add()
            meta.key, meta.value = k, v
        return model


    def get_onnx_metadata(model: onnx.ModelProto) -> Dict:
        """
        Get the metadata dictionary from model.

        Args:
            model (ModelProto): onnx model to add the metadata to.

        Returns:
            The model's the metadata dictionary.

        Example:
                Get model's metadata.

                >>> metadata = get_onnx_metadata(model)
        """
        return {prop.key: prop.value for prop in model.metadata_props}

else:
    def add_onnx_metadata(model,
                          metadata):
            Logger.critical('Installing onnx is mandatory '
                            'when using add_onnx_metadata. '
                            'Could not find onnx package.')  # pragma: no cover

    def get_onnx_metadata(model):
            Logger.critical('Installing onnx is mandatory '
                            'when using get_onnx_metadata. '
                            'Could not find onnx package.')  # pragma: no cover
