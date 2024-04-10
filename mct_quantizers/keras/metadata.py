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

from mct_quantizers.common.constants import FOUND_TF, FRAMEWORK_VERSION
from mct_quantizers.logger import Logger
from mct_quantizers.common.metadata import verify_and_init_metadata


if FOUND_TF:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable()
    class MetadataLayer(tf.keras.layers.Layer):
        """
        A layer for holding the metadata dictionary.
        """
        def __init__(self, metadata: Dict = None, **kwargs):
            self.metadata = metadata
            super(MetadataLayer, self).__init__(**kwargs)


    def add_metadata(model: tf.keras.Model, metadata: Dict):
        """
        Init the metadata dictionary and verify its compliance, then add it to the model under
        the 'metadata' attribute.
        Note that the metadata is also saved in a MetadataLayer layer, so it will be saved and loaded with keras easily.

        Args:
            model (Model): Keras model to add the metadata to.
            metadata (Dict): The metadata dictionary.

        Returns:
            The model with attribute 'metadata' that holds the metadata dictionary.

        Example:
                Adding author name and model version to model's metadata

                >>> model_with_metadata = add_metadata(model, {'author': 'John Doe', 'model version': 3})
        """
        metadata = verify_and_init_metadata(metadata)
        if FRAMEWORK_VERSION not in metadata:
            metadata[FRAMEWORK_VERSION] = tf.__version__

        model.metadata_layer = MetadataLayer(metadata=metadata)
        model.metadata = model.metadata_layer.metadata
        return model

    def get_metadata(model: tf.keras.Model) -> Dict:
        """
        Get the metadata dictionary from model.

        Args:
            model (Model): Keras model to extract metadata from.

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
            Logger.critical('Installing tensorflow is mandatory '
                            'when using add_metadata. '
                            'Could not find Tensorflow package.')  # pragma: no cover

    def get_metadata(model):
            Logger.critical('Installing tensorflow is mandatory '
                            'when using get_metadata. '
                            'Could not find Tensorflow package.')  # pragma: no cover
