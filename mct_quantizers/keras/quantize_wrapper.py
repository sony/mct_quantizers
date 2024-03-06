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
from typing import Dict, List, Any, Tuple, Union
from packaging import version
import numpy as np

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import FOUND_TF, WEIGHTS_QUANTIZERS, STEPS, WEIGHTS_VALUES, IS_INPUT_AS_LIST, \
    OP_CALL_ARGS, OP_CALL_KWARGS, LAYER, TRAINING, MCTQ_VERSION, POSITIONAL_WEIGHT, QUANTIZED_POSITIONAL_WEIGHT
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


    def _serialize_object(obj: Union[tf.Tensor, np.ndarray]) -> Dict:
        """
        A serialization function to replace keras.utils.serialize_keras_object for TF version < 2.13,
        which fails for objects of type tf.Tensor and np.ndarray, and mimic that function's operation
        in TF version >= 2.13.
        May be deleted when TF 2.12 is no longer supported.

        Args:
          obj: Object to be serialized.

        Returns:
            A dictionary with the object's config.
        """
        if isinstance(obj, tf.Tensor):
            return {'class_name': '__tensor__',
                    'config': {'value': obj.numpy().tolist(),
                               'dtype': obj.dtype.name}}
        elif isinstance(obj, np.ndarray):
            return {'class_name': '__numpy__',
                    'config': {'value': obj.tolist(),
                               'dtype': obj.dtype.name}}
        else:
            Logger.error(f'_serialize_object only accepts tf.Tensor or np.ndarray but got type {type(obj)}.')


    class KerasQuantizationWrapper(tf.keras.layers.Wrapper):
        def __init__(self,
                     layer: tf.keras.layers.Layer,
                     weights_quantizers: Dict[Union[int, str], BaseInferableQuantizer],
                     weight_values: Dict[int, Union[np.ndarray, tf.Tensor]] = None,
                     op_call_args: List = None,
                     op_call_kwargs: Dict[str, Any] = None,
                     is_inputs_as_list: bool = False,
                     **kwargs):
            """
            The KerasQuantizationWrapper takes a keras layer and quantization information and creates
            a quantized layer. The quantization information includes a quantizer per layer attribute for
            a keras layer that contains weight attributes (e.g. Conv2D, BatchNormalization, etc.). For
            layers that get constants (e.g. tf.add(KerasTensor, tf.constant), the quantization information
            also includes a weight values per attribute, the function call args & kwargs and a boolean for
            whether the layer\function accepts the inputs as a list (e.g. tf.concat or layers.Add). Note
            that for a layer with constants, the constants are referred to as "positional weights" whose
            attributes are integers representing the input index in the function\layer's inputs.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name or position to its quantizer.
                weight_values: A dictionary between a weight's position to its value.
                op_call_args: A list containing the layer's call arguments.
                op_call_kwargs: A dictionary containing the layer's call keyword arguments.
                is_inputs_as_list: A boolean indicating the layer accepts the input tensors as a list.

            Examples:

                Creating a quantized Conv2D (kernel only):

                >>> import mct_quantizers as mctq
                >>> import tensorflow as tf

                >>> attr_quant_dict = {'kernel': mctq.keras.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False)}
                >>> QuantizedConv2D = mctq.KerasQuantizationWrapper(tf.keras.layers.Conv2D(3,3), attr_quant_dict)

                creating a quantized function with a constant: tf.subtract(tf.constant, KerasTensor)

                >>> attr_quant_dict = {0: mctq.keras.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False)}
                >>> attr_values = {0: tf.constant([1, 2, 3], dtype=tf.float32)}
                >>> QuantizedSub = mctq.KerasQuantizationWrapper(TFOpLambda(tf.subtract), attr_quant_dict, attr_values)

                creating a quantized function with a constant and arguments: tf.matmul(KerasTensor, tf.constant, transpose_b=True)
                >>> attr_quant_dict = {1: mctq.keras.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False)}
                >>> attr_values = {1: tf.constant([[1,2,3], [4, 5, 6]], dtype=tf.float32)}
                >>> QuantizedMatmul = mctq.KerasQuantizationWrapper(TFOpLambda(tf.matmul), attr_quant_dict,
                >>>                                                 attr_values, op_call_kwargs={'transpose_b', True})

            """
            super(KerasQuantizationWrapper, self).__init__(layer, **kwargs)
            self._track_trackable(layer, name='layer')
            # making sure the attribute name is converted to the actual attribute field name in the layer.
            self.weights_quantizers = {_weight_name(k) if isinstance(k ,str) else k: v
                                       for k, v in weights_quantizers.items()}
            # Initialize positional weights:
            self.weight_values = weight_values if weight_values is not None else dict()
            for pos, weight_val in self.weight_values.items():
                if not isinstance(weight_val, (np.ndarray, tf.Tensor)):
                    Logger.error(f'Positional weight at position {pos} should be either an ndarray or a tf.Tensor,'
                                 f'but type is {type(weight_val)}')
            if version.parse(tf.__version__) < version.parse("2.13"):
                # Convert all values to tensors because keras.utils.serialize_keras_object fails for numpy array
                # before version 2.13. (TODO: remove this if-else when not supporting TF 2.12 and below)
                self.serialize_fn = _serialize_object
            else:
                self.serialize_fn = keras.utils.serialize_keras_object

            # Initialize functional layer arguments. For examples, see the class description.
            self.op_call_args = [] if op_call_args is None else op_call_args
            self.op_call_kwargs = {} if op_call_kwargs is None else op_call_kwargs
            self.is_inputs_as_list = is_inputs_as_list

            # Sanity checks
            # 1. If there are no weight_values: verify all weight_quantizers are strings
            # 2. If there are weight_values: verify all weight_quantizers and weight_values keys
            #    are integers, and that they match.
            # 3. A layer with both weights as attributes and positional weights is not supported.
            if len(self.weight_values) == 0:
                # expecting weights_quantizers keys to be all strings
                if not all([isinstance(w, str) for w in self.weights_quantizers]):
                    Logger.error('"weights_quantizers" keys should be all strings')
                self.is_str_attr = True
            else:
                # expecting both weights_quantizers and weight_values keys to be all integers.
                if not all([isinstance(w, int) for w in self.weight_values]):
                    Logger.error('All "weight_values" keys should be integers.')
                if not all([a == b for a, b in zip(weights_quantizers, weight_values)]):
                    Logger.error('Mismatch between "weights_quantizers" and "weight_values" keys')
                self.is_str_attr = False

            if not all([isinstance(w, (int, str)) for w in weights_quantizers]):
                Logger.error('All "weight_values" keys should be either strings ot integers')

            self._mctq_version = mctq_version

        @property
        def is_weights_quantization(self) -> bool:
            """
            This function check weights quantizer exists in wrapper.

            Returns: a boolean if weights quantizer exists.

            """
            return self.num_weights_quantizers > 0

        @property
        def num_weights_quantizers(self) -> int:
            """
            Returns: number of weights quantizers.
            """
            return len(self.weights_quantizers)

        def get_config(self):
            """
            Returns: Configuration of KerasQuantizationWrapper.

            """
            base_config = super(KerasQuantizationWrapper, self).get_config()
            config = {WEIGHTS_QUANTIZERS: {k: keras.utils.serialize_keras_object(v) for k, v in self.weights_quantizers.items()}}
            # Only create the wrapper attributes that handle positional weights if they exist, so the wrapper is forward
            # compatible with older MCTQ versions (at least until MCT will start quantizing positional weights)
            if len(self.weight_values) > 0:
                config[WEIGHTS_VALUES] = {k: self.serialize_fn(v) for k, v in self.weight_values.items()}
                config[OP_CALL_ARGS] = self.op_call_args
                config[OP_CALL_KWARGS] = self.op_call_kwargs
                config[IS_INPUT_AS_LIST] = self.is_inputs_as_list
            return_config = {**base_config, **config}
            return_config[MCTQ_VERSION] = self._mctq_version

            return return_config

        @property
        def mctq_version(self):
            return self._mctq_version

        def _set_weights_vars(self, is_training: bool = True):
            """
            This function sets weights quantizers vars to the layer.

            Args:
                is_training: Flag to indicate whether training or not.

            Returns: None
            """
            self._weights_vars = []
            for name, quantizer in self.weights_quantizers.items():
                if isinstance(name, str):
                    weight = getattr(self.layer, name)
                    _name = _weight_name(weight.name)
                    if is_training and not any([weight is w for w in self._trainable_weights]):
                        self._trainable_weights.append(weight)
                    elif not is_training and any([weight is w for w in self._non_trainable_weights]):
                        self._non_trainable_weights.append(weight)
                elif isinstance(name, int):
                    weight_value = self.weight_values[name]
                    _name = None
                    weight = self.add_weight(name=f'{POSITIONAL_WEIGHT}_{name}',
                                             shape=weight_value.shape,
                                             initializer=tf.keras.initializers.Constant(weight_value),
                                             trainable=False)
                    setattr(self, f'{POSITIONAL_WEIGHT}_{name}', weight)
                else:
                    Logger.error(f'A weight name ({name}) should be either "str" or "int", but has type {type(name)}')

                quantizer.initialize_quantization(weight.shape, _name if is_training else None,
                                                  self)

                # Add weight to wrapper weight lists (rather than the layer weight lists), because it will be deleted
                # from the layer's lists after the first call.
                self._weights_vars.append((name, weight, quantizer))

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  KerasQuantizationWrapper Configuration.

            Returns: A KerasQuantizationWrapper.

            """
            numpy_deserialization = lambda **_config: tf.constant(**_config).numpy()
            tensor_deserialization = lambda **_config: tf.constant(**_config)
            # When reading weight quantizers keys, which may be either a string with attribute name or an integer, the
            # key is always a string, so this function checks whether it was a string or integer before serialization.
            maybe_int = lambda x: int(x) if x.isdigit() else x

            config = config.copy()
            qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in
                                           get_all_subclasses(BaseKerasInferableQuantizer)}
            with keras.utils.custom_object_scope(qi_inferable_custom_objects):
                weights_quantizers = {maybe_int(k): keras.utils.deserialize_keras_object(v, module_objects=globals())
                                      for k, v in config.pop(WEIGHTS_QUANTIZERS).items()}

            # read weights_values in this scope so deserialize_keras_object knows how to interpret the serialization.
            with keras.utils.custom_object_scope({'__numpy__': numpy_deserialization,
                                                  '__tensor__': tensor_deserialization}):
                weights_values = {int(k): keras.utils.deserialize_keras_object(v)
                                  for k, v in config.pop(WEIGHTS_VALUES, {}).items()}

            layer = tf.keras.layers.deserialize(config.pop(LAYER))

            op_call_args = config.pop(OP_CALL_ARGS, [])
            op_call_kwargs = config.pop(OP_CALL_KWARGS, {})
            is_inputs_as_list = config.pop(IS_INPUT_AS_LIST, False)

            v = config.pop(MCTQ_VERSION, None)

            obj = cls(layer=layer, weights_quantizers=weights_quantizers, weight_values=weights_values,
                      op_call_args=op_call_args, op_call_kwargs=op_call_kwargs,
                      is_inputs_as_list=is_inputs_as_list, **config)
            obj._mctq_version = mctq_version if v is None else v

            return obj

        def build(self, input_shape):
            """
            KerasQuantization Wrapper build function.
            Args:
                input_shape: the layer input shape.

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
                quantized_weights: a dict of weight to update.

            Returns: None

            """
            for weight_attr in self.weights_quantizers:
                weight = quantized_weights.get(weight_attr)
                if isinstance(weight_attr, str):
                    # weight attribute is a string --> weight exists as attribute in layer so
                    # override is with quantized weight.
                    current_weight = getattr(self.layer, weight_attr)
                    setattr(self.layer, weight_attr, weight)
                elif isinstance(weight_attr, int):
                    # weight attribute is an integer --> weight doesn't exist as attribute in layer
                    # so create an attribute in wrapper for the quantized weight.
                    current_weight = getattr(self, f'{POSITIONAL_WEIGHT}_{weight_attr}')
                    setattr(self, f'{QUANTIZED_POSITIONAL_WEIGHT}_{weight_attr}', weight)
                else:
                    Logger.error(f'weight attribute should be either a string or an integer, but it is {type(weight_attr)}')  # pragma: no cover

                if current_weight.shape != weight.shape:
                    Logger.error(
                        f"Existing layer weight shape {current_weight.shape} is incompatible with provided weight "
                        f"shape {weight.shape}")  # pragma: no cover

        def call(self, inputs, training=None, **kwargs):
            """
            KerasQuantizationWrapper call functions.
            Args:
                inputs: Input tensors to specified layer.
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
                    # Keras weights inferable quantizer.
                    quantized_weight = quantizer(unquantized_weight)
                    quantized_weights.update({name: quantized_weight})

            self.set_quantize_weights(quantized_weights)

            if self.is_str_attr:
                args_spec = tf_inspect.getfullargspec(self.layer.call).args
                if TRAINING in args_spec:
                    kwargs.update({'training': training})
                outputs = self.layer.call(inputs, **kwargs)
            else:
                _inputs = inputs if isinstance(inputs, list) else [inputs]
                weight_positions = [w[0] for w in self._weights_vars]
                for pos in sorted(weight_positions):
                    _inputs.insert(pos, getattr(self, f'{QUANTIZED_POSITIONAL_WEIGHT}_{pos}'))

                if self.is_inputs_as_list:
                    outputs = self.layer.call(_inputs, *self.op_call_args, **self.op_call_kwargs)
                else:
                    outputs = self.layer.call(*(_inputs + self.op_call_args), **self.op_call_kwargs)

            return outputs

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
    class KerasQuantizationWrapper:
        def __init__(self,
                     layer,
                     weights_quantizers: Dict[str, BaseInferableQuantizer],
                     weight_values: Dict = None,
                     op_call_args: List = None,
                     op_call_kwargs: Dict[str, Any] = None,
                     is_inputs_as_list: bool = False):
            """
            Keras Quantization Wrapper takes a keras layer and quantizers and infer a quantized layer.

            Args:
                layer: A keras layer.
                weights_quantizers: A dictionary between a weight's name to its quantizer.
                weight_values: A dictionary between a weight's position to its value.
                op_call_args: A list containing the layer's call arguments.
                op_call_kwargs: A dictionary containing the layer's call keyword arguments.
                is_inputs_as_list: A boolean indicating the layer accepts the input tensors as a list.
            """
            Logger.critical('Installing tensorflow is mandatory '
                            'when using KerasQuantizationWrapper. '
                            'Could not find Tensorflow package.')  # pragma: no cover
