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

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER, FOUND_TF, TRAINING, STEPS
from mct_quantizers.common.get_all_subclasses import get_all_subclasses
from mct_quantizers.logger import Logger

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


    class KerasActivationQuantizationHolder(keras.layers.Layer):
        """
        Keras layer to hold an activation quantizer and quantize during inference.
        """
        def __init__(self,
                     activation_holder_quantizer: BaseInferableQuantizer,
                     **kwargs):
            """

            Args:
                activation_holder_quantizer: Quantizer to use during inference.
                **kwargs: Key-word arguments for the base layer
            """

            super(KerasActivationQuantizationHolder, self).__init__(**kwargs)
            self.activation_holder_quantizer = activation_holder_quantizer

        def get_config(self):
            """
            config(dict): dictionary  of ActivationQuantizationHolder Configuration

            Returns: Configuration of ActivationQuantizationHolder.

            """
            base_config = super(KerasActivationQuantizationHolder, self).get_config()
            config = {
                ACTIVATION_HOLDER_QUANTIZER: keras.utils.serialize_keras_object(self.activation_holder_quantizer)}

            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  ActivationQuantizationHolder Configuration

            Returns: A ActivationQuantizationHolder object

            """
            qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in
                                           get_all_subclasses(BaseKerasInferableQuantizer)}
            config = config.copy()
            activation_holder_quantizer = keras.utils.deserialize_keras_object(config.pop(ACTIVATION_HOLDER_QUANTIZER),
                                                                               module_objects=globals(),
                                                                               custom_objects=qi_inferable_custom_objects)

            return cls(activation_holder_quantizer=activation_holder_quantizer,
                       **config)

        def build(self, input_shape):
            """
            ActivationQuantizationHolder build function.
            Args:
                input_shape: the layer input shape

            Returns: None

            """
            super(KerasActivationQuantizationHolder, self).build(input_shape)

            self.optimizer_step = self.add_weight(
                STEPS,
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.dtypes.int32,
                trainable=False)

            self.activation_holder_quantizer.initialize_quantization(None,
                                                                     self.name + '/out_',
                                                                     self)

        def call(self,
                 inputs: tf.Tensor,
                 training=None) -> tf.Tensor:
            """
            Quantizes the input tensor using the activation quantizer the ActivationQuantizationHolder holds.

            Args:
                inputs: Input tensors to quantize use the activation quantizer the object holds
                training: a boolean stating if layer is in training mode.

            Returns: Output of the activation quantizer (quantized input tensor).

            """
            if training is None:
                training = tf.keras.backend.learning_phase()

            activation_quantizer_args_spec = tf_inspect.getfullargspec(self.activation_holder_quantizer.__call__).args
            if TRAINING in activation_quantizer_args_spec:
                return smart_cond(
                    training,
                    _make_quantizer_fn(self.activation_holder_quantizer, inputs, True),
                    _make_quantizer_fn(self.activation_holder_quantizer, inputs, False))

            return self.activation_holder_quantizer(inputs)

        def convert_to_inferable_quantizers(self):
            """
            Convert layer's quantizer to inferable quantizer.

            Returns:
                None
            """
            if hasattr(self.activation_holder_quantizer, 'convert2inferable') and callable(
                    self.activation_holder_quantizer.convert2inferable):  # pragma: no cover
                self.activation_holder_quantizer = self.activation_holder_quantizer.convert2inferable()
                return self.from_config(self.get_config()) # return new layer with no weights. It assumes holder of inferable quantizers have no weights


else:
    class KerasActivationQuantizationHolder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow is mandatory '
                         'when using ActivationQuantizationHolder. '
                         'Could not find Tensorflow package.')
