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
from typing import Dict, List, Any, Tuple

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import FOUND_TF, WEIGHTS_QUANTIZERS, STEPS, LAYER, TRAINING, MCTQ_VERSION
from mct_quantizers.logger import Logger
from mct_quantizers.common.get_all_subclasses import get_all_subclasses
from mct_quantizers import __version__ as mctq_version

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.python.util import tf_inspect
    from tensorflow.python.keras.utils.control_flow_util import smart_cond

    from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

    keras = tf.keras

    def _make_quantizer_fn(quantizer, x, training):
        """Use currying to return True/False specialized fns to the cond."""

        def quantizer_fn():
            return quantizer(x, training)

        return quantizer_fn


    def _weight_name(name: str) -> str:
        """Extracts the weight name from the full TensorFlow variable name.

        For example, returns 'kernel' for 'dense_2/kernel:0'.

        Args:
          name: TensorFlow variable name.

        Returns:
          Extracted weight name.
        """
        return name.split(':')[0].split('/')[-1]


    class KerasQuantizationWrapper(tf.keras.layers.Wrapper):
        def __init__(self,
                     layer,
                     weights_quantizers: Dict[str, BaseInferableQuantizer] = None,
                     **kwargs):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
            """
            super(KerasQuantizationWrapper, self).__init__(layer, **kwargs)
            self._track_trackable(layer, name='layer')
            self.weights_quantizers = weights_quantizers if weights_quantizers is not None else dict()

            self._mctq_version = mctq_version

        def add_weights_quantizer(self, param_name: str, quantizer: BaseInferableQuantizer):
            """
            This function adds a weights quantizer to existing wrapper

            Args:
                param_name: The name of the parameter to quantize
                quantizer: A quantizer.

            Returns: None

            """
            self.weights_quantizers.update({param_name: quantizer})

        @property
        def is_weights_quantization(self) -> bool:
            """
            This function check weights quantizer exists in wrapper.

            Returns: a boolean if weights quantizer exists

            """
            return self.num_weights_quantizers > 0

        @property
        def num_weights_quantizers(self) -> int:
            """
            Returns: number of weights quantizers
            """
            return len(self.weights_quantizers)

        def get_config(self):
            """
            Returns: Configuration of KerasQuantizationWrapper.

            """
            base_config = super(KerasQuantizationWrapper, self).get_config()
            config = {WEIGHTS_QUANTIZERS: {k: keras.utils.serialize_keras_object(v) for k, v in self.weights_quantizers.items()}}

            return_config = dict(list(base_config.items()) + list(config.items()))
            return_config[MCTQ_VERSION] = self._mctq_version

            return return_config

        @property
        def mctq_version(self):
            return self._mctq_version

        def _set_weights_vars(self, is_training: bool = True):
            """
            This function sets weights quantizers vars to the layer

            Args:
                is_training: Flag to indicate whether training or not

            Returns: None
            """
            self._weights_vars = []
            for name, quantizer in self.weights_quantizers.items():
                weight = getattr(self.layer, name)
                quantizer.initialize_quantization(weight.shape, _weight_name(weight.name) if is_training else None,
                                                  self)
                self._weights_vars.append((name, weight, quantizer))
                self._trainable_weights.append(weight) # Must when inherit from tf.keras.layers.Wrapper in tf2.10 and below

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  KerasQuantizationWrapper Configuration

            Returns: A KerasQuantizationWrapper

            """
            config = config.copy()
            qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in
                                           get_all_subclasses(BaseKerasInferableQuantizer)}
            weights_quantizers = {k: keras.utils.deserialize_keras_object(v,
                                                                          module_objects=globals(),
                                                                          custom_objects=qi_inferable_custom_objects) for k, v in config.pop(WEIGHTS_QUANTIZERS).items()}
            layer = tf.keras.layers.deserialize(config.pop(LAYER))

            v = config.pop(MCTQ_VERSION, None)

            obj = cls(layer=layer, weights_quantizers=weights_quantizers, **config)
            obj._mctq_version = mctq_version if v is None else v

            return obj

        def build(self, input_shape):
            """
            KerasQuantization Wrapper build function.
            Args:
                input_shape: the layer input shape

            Returns: None

            """
            super(KerasQuantizationWrapper, self).build(input_shape)

            self.optimizer_step = self.add_weight(
                STEPS,
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.dtypes.int32,
                trainable=False)

            self._set_weights_vars()

        def set_quantize_weights(self, quantized_weights: dict):
            """
            This function update layer weights after quantization.

            Args:
                quantized_weights: a dict of weight to update

            Returns: None

            """
            for weight_attr in self.weights_quantizers.keys():
                weight = quantized_weights.get(weight_attr)
                current_weight = getattr(self.layer, weight_attr)
                if current_weight.shape != weight.shape:
                    Logger.error(
                        f"Existing layer weight shape {current_weight.shape} is incompatible with provided weight "
                        f"shape {weight.shape}")  # pragma: no cover

                setattr(self.layer, weight_attr, weight)

        def call(self, inputs, training=None, **kwargs):
            """
            KerasQuantizationWrapper call functions
            Args:
                inputs: Input tensors to specified layer
                training: a boolean stating if layer is in training mode.
                **kwargs:

            Returns: tensors that simulate a quantized layer.

            """
            if training is None:
                training = tf.keras.backend.learning_phase()

            # Quantize all weights, and replace them in the underlying layer.
            quantized_weights = {}
            for name, unquantized_weight, quantizer in self._weights_vars:

                weights_quantizer_args_spec = tf_inspect.getfullargspec(quantizer.__call__).args
                if TRAINING in weights_quantizer_args_spec:
                    quantized_weight = smart_cond(
                        training,
                        _make_quantizer_fn(quantizer, unquantized_weight, True),
                        _make_quantizer_fn(quantizer, unquantized_weight, False))
                    quantized_weights.update({name: quantized_weight})
                else:
                    # Keras weights inferable quantizer
                    quantized_weight = quantizer(unquantized_weight)
                    quantized_weights.update({name: quantized_weight})

            self.set_quantize_weights(quantized_weights)

            args_spec = tf_inspect.getfullargspec(self.layer.call).args
            if TRAINING in args_spec:
                outputs = self.layer.call(inputs, training=training, **kwargs)
            else:
                outputs = self.layer.call(inputs, **kwargs)

            return outputs

        def convert_to_inferable_quantizers(self):
            """
            Convert layer's quantizers to inferable.

            Returns:
                None
            """
            # Weight quantizers
            inferable_weight_quantizers = {}
            if self.is_weights_quantization:
                for name, quantizer in self.weights_quantizers.items():
                    if hasattr(quantizer, 'convert2inferable') and callable(quantizer.convert2inferable):
                        inferable_weight_quantizers.update({name: quantizer.convert2inferable()})
                self.weights_quantizers = inferable_weight_quantizers

            # Create new layer with inferable quantizers
            inferable_quantizers_wrapper = self.from_config(self.get_config())
            inferable_quantizers_wrapper.layer.build(self.get_input_shape_at(0))
            layer_weights_list = []
            for weight_attr in self.weights_quantizers.keys():
                layer_weights_list.append(getattr(self.layer, weight_attr)) # quantized weights
            layer_weights_list.extend(self.layer.get_weights()) # non quantized weights
            inferable_quantizers_wrapper.layer.set_weights(layer_weights_list)

            # The wrapper inference is using the weights of the quantizers so it expectes to create them by running _set_weights_vars
            inferable_quantizers_wrapper._set_weights_vars(False)
            return inferable_quantizers_wrapper

        def get_weights_vars(self) -> List[Tuple[str, Any, BaseInferableQuantizer]]:
            """
            A getter of the layer's weights variables.

            Returns:
                List pf tuples of the wrapped layer's weights variables with weight name, values and assigned quantizer.

            """

            return self._weights_vars

        def get_quantized_weights(self) -> Dict[str, tf.Tensor]:
            """

            Returns: A dictionary of weights attributes to quantized weights.

            """
            quantized_weights = {}
            weights_var = self.get_weights_vars()
            for name, w, quantizer in weights_var:
                quantized_weights[name] = quantizer(w)
            return quantized_weights

else:
    class KerasQuantizationWrapper(object):
        def __init__(self,
                     layer,
                     weights_quantizers: Dict[str, BaseInferableQuantizer] = None):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
            """
            Logger.critical('Installing tensorflow is mandatory '
                            'when using KerasQuantizationWrapper. '
                            'Could not find Tensorflow package.')  # pragma: no cover
