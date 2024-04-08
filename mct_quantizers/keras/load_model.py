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

from mct_quantizers.common.constants import FOUND_TF
from mct_quantizers.common.get_all_subclasses import get_all_subclasses
from mct_quantizers.logger import Logger

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.python.saved_model.load_options import LoadOptions
    from mct_quantizers.keras.activation_quantization_holder import KerasActivationQuantizationHolder
    from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper
    from mct_quantizers.keras.metadata import MetadataLayer
    from mct_quantizers.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer
    keras = tf.keras

    def keras_load_quantized_model(filepath: str, custom_objects: Any = None, compile: bool = True,
                                   options: LoadOptions = None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in
                                       get_all_subclasses(BaseKerasInferableQuantizer)}
        all_inferable_names = list(qi_inferable_custom_objects.keys())
        if len(set(all_inferable_names)) < len(all_inferable_names):
            Logger.error(f"Found multiple quantizers with the same name that inherit from BaseKerasInferableQuantizer"
                         f"while trying to load a model.")

        qi_custom_objects = {**qi_inferable_custom_objects}

        # Add non-quantizers custom objects
        qi_custom_objects.update({KerasQuantizationWrapper.__name__: KerasQuantizationWrapper})
        qi_custom_objects.update({KerasActivationQuantizationHolder.__name__: KerasActivationQuantizationHolder})

        if custom_objects is not None:
            qi_custom_objects.update(custom_objects)
        # in keras format (v3) passing option is an error
        kwargs = {}
        if options is not None:
            kwargs['options'] = options

        # Load model
        loaded_model = tf.keras.models.load_model(filepath, custom_objects=qi_custom_objects, compile=compile, **kwargs)

        # Extract metadata if exists
        metadata_layers = [l for l in loaded_model.layers if isinstance(l, MetadataLayer)]
        if len(metadata_layers) > 0:
            if len(metadata_layers) > 1:
                Logger.warning('Found more than 1 MetadataLayer layers in model. Loading the metadata from the first layer only.')
            loaded_model.metadata_layer = metadata_layers[0]
            loaded_model.metadata = metadata_layers[0].metadata
        return loaded_model
else:
    def keras_load_quantized_model(filepath, custom_objects=None, compile=True, options=None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_load_quantized_model. '
                        'Could not find Tensorflow package.')  # pragma: no cover
