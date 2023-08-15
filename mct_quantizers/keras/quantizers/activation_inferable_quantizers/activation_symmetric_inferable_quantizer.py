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
from typing import List

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TF
from mct_quantizers.common.quant_info import QuantizationMethod


if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantizers.activation_inferable_quantizers.activation_uniform_inferable_quantizer \
        import ActivationUniformInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.SYMMETRIC],
                    identifier=QuantizerID.INFERABLE)
    class ActivationSymmetricInferableQuantizer(ActivationUniformInferableQuantizer):

        """
        Class for quantizing activations using a symmetric quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: List[float],
                     signed: bool):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """
            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            # In activation per-channel quantization is not supported thus we expect a single min/max value.
            assert len(threshold) == 1, f'In per-tensor quantization threshold should be of length 1 but is {len(threshold)}'
            assert all([isinstance(x, (float, np.float32, tf.float32)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            self.threshold = threshold
            self.signed = signed

            delta = self.threshold[0] / (2 ** (num_bits - int(self.signed)))
            # In activation quantization is per-tensor only - thus we pass the threshold as a list with a len of 1
            min_range = [-threshold[0]] if self.signed else [0.0]
            max_range = [self.threshold[0] - delta]

            super(ActivationSymmetricInferableQuantizer, self).__init__(num_bits=num_bits,
                                                                        min_range=min_range,
                                                                        max_range=max_range)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'signed', 'threshold'
            """
            return {'num_bits': self.num_bits,
                    'signed': self.signed,
                    'threshold': self.threshold}

        @classmethod
        def from_config(cls, config):
            """
            Return an object with config
            Args:
                config(dict): dictionary of object configuration
            Returns: An object created with config
            """
            return cls(config.get('num_bits'),
                       config.get('threshold'),
                       config.get('signed'))
else:
    class ActivationSymmetricInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow is mandatory '
                            'when using ActivationSymmetricInferableQuantizer. '
                            'Could not find Tensorflow package.')
