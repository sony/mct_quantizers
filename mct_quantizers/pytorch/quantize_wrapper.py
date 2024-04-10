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
from typing import List, Union, Any, Dict, Tuple, Callable

import inspect

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer
from mct_quantizers.common.constants import FOUND_TORCH, LAYER, TRAINING, POSITIONAL_WEIGHT, \
    QUANTIZED_POSITIONAL_WEIGHT
from mct_quantizers.logger import Logger

if FOUND_TORCH:
    import torch
    import torch.nn as nn


    class PytorchQuantizationWrapper(nn.Module):
        def __init__(self,
                     module: Union[nn.Module, Callable],
                     weights_quantizers: Dict[Union[int, str], BaseInferableQuantizer],
                     weight_values: Dict[int, torch.Tensor] = None,
                     op_call_args: List = None,
                     op_call_kwargs: Dict[str, Any] = None,
                     is_inputs_as_list: bool = False):
            """
            The PytorchQuantizationWrapper takes a Pytorch layer and quantization information and creates
            a quantized layer. The quantization information includes a quantizer per layer attribute for
            a Torch layer that contains weight attributes (e.g. Conv2d, BatchNorm2d, etc.). For
            layers that get constants (e.g. torch.add(Tensor, constant), the quantization information
            also includes a weight values per attribute, the function call args & kwargs and a boolean for
            whether the layer\function accepts the inputs as a list (e.g. torch.cat). Note that for a layer
            with constants, the constants are referred to as "positional weights" whose attributes are integers
            representing the input index in the function\layer's inputs.

            Args:
                module: A pytorch module or as function.
                weights_quantizers: A dictionary between a weight's name or position to its quantizer.
                weight_values: A dictionary between a weight's position to its value.
                op_call_args: A list containing the layer's call arguments.
                op_call_kwargs: A dictionary containing the layer's call keyword arguments.
                is_inputs_as_list: A boolean indicating the layer accepts the input tensors as a list.

            Examples:

                Creating a quantized Conv2d (weight only):

                >>> import mct_quantizers as mctq
                >>> import torch

                >>> attr_quant_dict = {'weight': mctq.pytorch.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False)}
                >>> QuantizedConv2D = mctq.PytorchQuantizationWrapper(torch.nn.Conv2d(3, 3, 3), attr_quant_dict)

                creating a quantized function with a constant: torch.sub(constant, Tensor)

                >>> attr_quant_dict = {0: mctq.pytorch.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False)}
                >>> attr_values = {0: torch.Tensor([1, 2, 3], dtype=torch.float32)}
                >>> QuantizedSub = mctq.PytorchQuantizationWrapper(torch.sub), attr_quant_dict, attr_values)

                creating a quantized function with constants and arguments: tf.cat([constant#1, Tensor, constant#2], dim=1)
                >>> attr_quant_dict = {0: mctq.pytorch.quantizers.WeightsPOTInferableQuantizer(4, [2.0], False),
                >>>                    2: mctq.pytorch.quantizers.WeightsPOTInferableQuantizer(4, [1.0], False)}
                >>> attr_values = {0: torch.Tensor([[1,2,3], [4, 5, 6]], dtype=torch.float32),
                >>>                2: torch.Tensor([[4,5,6], [4, 5, 6]], dtype=torch.float32)}
                >>> QuantizedConcat = mctq.PytorchQuantizationWrapper(torch.cat, attr_quant_dict, attr_values,
                >>>                                                   op_call_kwargs={'dim', 1})

            """
            super().__init__()
            if isinstance(module, nn.Module):
                self.add_module(LAYER, module)
            else:
                # Functional layers
                setattr(self, LAYER, module)

            self.weights_quantizers = weights_quantizers
            # Initialize positional weights:
            self.weight_values = weight_values if weight_values is not None else dict()
            for pos, weight_val in self.weight_values.items():
                if not isinstance(weight_val, torch.Tensor):
                    Logger.error(f'Positional weight at position {pos} should be a torch.Tensor, '
                                 f'but type is {type(weight_val)}.')

            # Initialize functional module arguments. For examples, see the class description.
            self.op_call_args = [] if op_call_args is None else op_call_args
            self.op_call_kwargs = {} if op_call_kwargs is None else op_call_kwargs
            self.is_inputs_as_list = is_inputs_as_list

            # Sanity checks:
            # 1. If there are no weight_values: verify all weight_quantizers are strings
            # 2. If there are weight_values: verify all weight_quantizers and weight_values keys
            #    are integers, and that they match.
            # 3. A module with both weights as attributes and positional weights is not supported.
            if len(self.weight_values) == 0:
                # expecting weights_quantizers keys to be all strings.
                if not all([isinstance(w, str) for w in self.weights_quantizers]):
                    Logger.error('"weights_quantizers" keys should be all strings')
                self.is_str_attr = True
            else:
                # expecting both weights_quantizers and weight_values keys to be all integers.
                if not all([isinstance(w, int) for w in self.weight_values]):
                    Logger.error('All "weight_values" keys should be integers')
                if not all([a == b for a, b in zip(weights_quantizers, weight_values)]):
                    Logger.error('Mismatch between "weights_quantizers" and "weight_values" keys')
                self.is_str_attr = False

            self._set_weights_vars(True)

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

        def convert_to_inferable_quantizers(self):
            """
            Convert the wrapper quantizers with inferable quantizers.

            """
            # Weight quantizers
            if self.is_weights_quantization:
                inferable_weight_quantizers = {}
                for name, quantizer in self.weights_quantizers.items():
                    if hasattr(quantizer, 'convert2inferable') and callable(quantizer.convert2inferable):
                        inferable_weight_quantizers.update({name: quantizer.convert2inferable()})
                self.weights_quantizers = inferable_weight_quantizers
                self._set_weights_vars(False)

        def _set_weights_vars(self, is_training: bool = True):
            """
            Initialize learnable weights as parameters in the wrapper, and their quantizers.

            Args:
                is_training: Whether working with InferableQuantizers or not. If so, do not register weight as parameter.

            """
            self._weights_vars = []

            # Init weights quantizers
            for name, quantizer in self.weights_quantizers.items():
                if self.is_str_attr:
                    if is_training:
                        weight = getattr(self.layer, name).detach()
                        delattr(self.layer, name)
                        setattr(self.layer, name, weight)
                        self.register_parameter(name, torch.nn.Parameter(weight, requires_grad=True))
                    else:
                        weight = getattr(self, name).detach()
                        delattr(self.layer, name)
                        setattr(self.layer, name, weight)
                    weight_var = getattr(self, name)
                else:
                    weight = self.weight_values[name]
                    self.register_parameter(f'{POSITIONAL_WEIGHT}_{name}',
                                            torch.nn.Parameter(weight, requires_grad=False))
                    setattr(self, f'{QUANTIZED_POSITIONAL_WEIGHT}_{name}', weight)
                    weight_var = getattr(self, f'{POSITIONAL_WEIGHT}_{name}')

                quantizer.initialize_quantization(weight.shape, name, self)
                self._weights_vars.append((name, weight_var, quantizer))

        def set_quantize_weights(self, quantized_weights: dict):
            """
            This function updates layer weights after quantization.

            Args:
                quantized_weights: a dict of weight to update.

            Returns: None

            """
            for weight_attr in self.weights_quantizers:
                weight = quantized_weights.get(weight_attr)
                if self.is_str_attr:
                    setattr(self.layer, weight_attr, weight)
                else:
                    setattr(self, f'{QUANTIZED_POSITIONAL_WEIGHT}_{weight_attr}', weight)

        def get_weights_vars(self) -> List[Tuple[str, Any, BaseInferableQuantizer]]:
            """
            A getter of the layer's weights variables.

            Returns:
                List pf tuples of the wrapped layer's weights variables with weight name, values and assigned quantizer.

            """

            return self._weights_vars

        def forward(self,
                    *args: List[Any],
                    **kwargs: Dict[str, Any]) -> Union[torch.Tensor, List[torch.Tensor]]:
            """
            PytorchQuantizationWrapper forward functions.
            Args:
                args: arguments to pass to internal layer.
                kwargs: key-word dictionary to pass to the internal layer.

            Returns: a tensor that simulates a quantized layer.

            """

            # ----------------------------------
            # Quantize all weights, and replace them in the underlying layer.
            # ----------------------------------
            if self.is_weights_quantization:

                quantized_weights = {}
                for name, unquantized_weight, quantizer in self._weights_vars:
                    s = inspect.signature(quantizer.__call__)
                    if TRAINING in s.parameters.keys():
                        quantized_weight = quantizer(unquantized_weight, self.training)
                    else:
                        quantized_weight = quantizer(unquantized_weight)

                    quantized_weights.update({name: quantized_weight})

                self.set_quantize_weights(quantized_weights)

            if not self.is_str_attr:
                # Positional weights need to be inserted in the wrapper input list according to their (key) position.
                args = list(args)
                weight_positions = [w[0] for w in self._weights_vars]
                for pos in sorted(weight_positions):
                    args.insert(pos, getattr(self, f'{QUANTIZED_POSITIONAL_WEIGHT}_{pos}'))

            _kwargs = {**self.op_call_kwargs, **kwargs}
            # ----------------------------------
            # Layer operation
            # ----------------------------------
            if self.is_inputs_as_list:
                outputs = self.layer(args, *self.op_call_args, **_kwargs)
            else:
                outputs = self.layer(*args, *self.op_call_args, **_kwargs)

            return outputs

        def get_quantized_weights(self) -> Dict[str, torch.Tensor]:
            """

            Returns: A dictionary of weights attributes to quantized weights.

            """
            quantized_weights = {}
            weights_var = self.get_weights_vars()
            for name, w, quantizer in weights_var:
                quantized_weights[name] = quantizer(w)
            return quantized_weights

else:
    class PytorchQuantizationWrapper:
        def __init__(self,
                     layer,
                     weight_quantizers: Dict[str, BaseInferableQuantizer],
                     weight_values: Dict = None,
                     op_call_args: List = None,
                     op_call_kwargs: Dict[str, Any] = None,
                     is_inputs_as_list: bool = False):
            """
            Pytorch Quantization Wrapper takes a pytorch module and quantizers and infer a quantized layer.

            Args:
                layer: A pytorch module.
                weight_quantizers: A dictionary between a weight's name to its quantizer.
                weight_values: A dictionary between a weight's position to its value.
                op_call_args: A list containing the layer's call arguments.
                op_call_kwargs: A dictionary containing the layer's call keyword arguments.
                is_inputs_as_list: A boolean indicating the layer accepts the input tensors as a list.
            """
            Logger.critical('Installing Pytorch is mandatory '
                            'when using PytorchQuantizationWrapper. '
                            'Could not find torch package.')  # pragma: no cover
