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
import warnings
from typing import List

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget, QuantizerID
from mct_quantizers.common.constants import FOUND_TF, LUT_VALUES_BITWIDTH, EPS
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.logger import Logger


if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantizer_utils import lut_quantizer
    from mct_quantizers.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.LUT_POT_QUANTIZER],
                    identifier=QuantizerID.INFERABLE)
    class ActivationLutPOTInferableQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing activations using lut power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     lut_values: List[int],
                     threshold: List[float],
                     signed: bool,
                     lut_values_bitwidth: int = LUT_VALUES_BITWIDTH,
                     eps: float = EPS):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                lut_values: the values in the look-up table to assign the weights to
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
                lut_values_bitwidth: Number of bits that determines the quantization range
                eps: Small value for numerical stability in division
            """
            # Call the superclass constructor with the given parameters, along with the target of Activation
            # quantization
            super(ActivationLutPOTInferableQuantizer, self).__init__()

            assert isinstance(threshold, list), f'Expected threshold to be of type list but is {type(threshold)}'
            assert all([isinstance(x, (float, np.float32, tf.float32)) for x in
                        threshold]), f'Expected threshold list to contain float or np.float values but found ' \
                                     f'{[type(x) for x in threshold]}'

            # In activation per-channel quantization is not supported thus we expect a single threshold value.
            assert len(threshold) == 1, f'In per-tensor quantization threshold should be of ' \
                                        f'length 1 but is {len(threshold)}'

            is_threshold_pot = np.all([int(np.log2(x)) == np.log2(x) for x in threshold])
            assert is_threshold_pot, f'Expected threshold to be power of 2 but is {threshold}'

            self.threshold = threshold

            # Convert lut_values to numpy array for all assertions, and convert it back to list before saving.
            # The reason for doing so is that during deserialization (when get_config is called) the returned value
            # is a list (even if it was a numpy array during serialization) thus the expected lut values type
            # must be a list. The conversion to numpy is to make assertions more clean.
            lut_values = np.asarray(lut_values)

            assert len(np.unique(lut_values)) <= 2 ** num_bits, \
                f'Expected num of lut values to be less or equal than {2 ** num_bits} ' \
                f'but got {len(lut_values)}'

            assert not np.any(lut_values - lut_values.astype(int)), f'Expected lut values to be integers'

            if signed:
                assert np.all((-1 * (2 ** (lut_values_bitwidth - int(signed))) <= lut_values) &
                              (lut_values <= (2 ** (lut_values_bitwidth - int(signed)) - 1))), \
                    f'Expected lut values in the quantization range'
            else:
                assert np.all(lut_values <= (2 ** lut_values_bitwidth)), \
                    f'Expected lut values in the quantization range'

            # num_bits must be less than lut_values_bitwidth
            assert num_bits <= lut_values_bitwidth, f'Look-Up-Table bit configuration has {num_bits} bits. It must be ' \
                                                  f'less then {lut_values_bitwidth}'
            if num_bits == lut_values_bitwidth:
                warnings.warn("Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be "
                              "inefficient in that case, consider using SymmetricInferableQuantizer instead")

            # If unsigned activation quantization, all lut_values must have the same sign
            if not signed:
                assert np.all(lut_values >= 0), f'Expected unsigned lut values in unsigned activation ' \
                                                     f'quantization '

            self.num_bits = num_bits
            # Save as a numpy array to avoid conversion during inference
            self._lut_values_as_np = lut_values
            # Save as a list for serialization purposes
            self.lut_values = lut_values.tolist()
            self.signed = signed
            self.lut_values_bitwidth = lut_values_bitwidth
            self.eps = eps

        def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """
            assert inputs.dtype == tf.float32, f'Input tensor was expected to be a float tensor but is of type ' \
                                               f'{inputs.dtype}'

            return lut_quantizer(inputs,
                                 lut_values=self._lut_values_as_np.astype(np.float32),
                                 signed=self.signed,
                                 # In activation per-channel quantization is not supported
                                 # thus we expect a single threshold value. Assertion is made in init.
                                 threshold=self.threshold[0],
                                 lut_values_bitwidth=self.lut_values_bitwidth,
                                 eps=self.eps)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'lut_values', 'threshold', 'signed',
                'lut_values_bitwidth', 'eps'
            """
            return {'num_bits': self.num_bits,
                    'lut_values': self.lut_values,
                    'threshold': self.threshold,
                    'signed': self.signed,
                    'lut_values_bitwidth': self.lut_values_bitwidth,
                    'eps': self.eps}

        @classmethod
        def from_config(cls, config):
            """
            Return an object with config
            Args:
                config(dict): dictionary of object configuration
            Returns: An object created with config
            """
            return cls(config.get('num_bits'),
                       config.get('lut_values'),
                       config.get('threshold'),
                       config.get('signed'),
                       config.get('lut_values_bitwidth', LUT_VALUES_BITWIDTH),
                       config.get('eps', EPS))

else:
    class ActivationLutPOTInferableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow is mandatory '
                         'when using ActivationLutPOTInferableQuantizer. '
                         'Could not find Tensorflow package.')
