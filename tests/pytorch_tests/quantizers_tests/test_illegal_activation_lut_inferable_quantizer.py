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
import unittest
import warnings

import numpy as np

from mct_quantizers.pytorch.quantizers.activation_inferable_quantizers.activation_lut_pot_inferable_quantizer import \
    ActivationLutPOTInferableQuantizer


class TestPytorchActivationIllegalLutPotQuantizer(unittest.TestCase):

    def test_illegal_pot_lut_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               lut_values=np.asarray([25., 85.]),
                                               # Not POT threshold
                                               threshold=np.asarray([3]),
                                               signed=True)
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               lut_values=np.asarray([25., 85.]),
                                               # Not float
                                               threshold=4,
                                               signed=True)

        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'int\'>',
                                   str(e.exception))

    def test_illegal_signed_lut_values_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               lut_values=np.asarray([-25., 85.]),
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.assertEqual('Expected unsigned lut values in unsigned activation quantization',
                                   str(e.exception))

    def test_illegal_num_of_lut_values_quantizer(self):
        lut_values = np.asarray([-25, 25, 3, 19, 55])
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=2,
                                               lut_values=lut_values,
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.assertEqual(f'Expected num of lut values to be less or equal than {2 ** 2} but got '
                                   f'{len(lut_values)}', str(e.exception))

    def test_illegal_lut_values_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               lut_values=np.asarray([-25.6, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=False)
        self.assertEqual('Expected lut values to be integers', str(e.exception))

    def test_illegal_lut_values_signed_range_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               lut_values_bitwidth=5)
        self.assertEqual('Expected lut values in the quantization range', str(e.exception))

    def test_illegal_lut_values_unsigned_range_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               lut_values=np.asarray([15, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=False,
                                               lut_values_bitwidth=4)
        self.assertEqual('Expected lut values in the quantization range', str(e.exception))

    def test_illegal_num_bit_bigger_than_lut_values_bitwidth_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=10,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               lut_values_bitwidth=8)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8',
                                   str(e.exception))

    def test_warning_num_bit_equal_lut_values_bitwidth_quantizer(self):
        with warnings.catch_warnings(record=True) as w:
            ActivationLutPOTInferableQuantizer(num_bits=8,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=np.asarray([2.]),
                                               signed=True,
                                               lut_values_bitwidth=8)
        self.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def test_illegal_threshold_type_quantizer(self):
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=[2.],
                                               signed=True,
                                               lut_values_bitwidth=8)
        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'list\'>',
                                   str(e.exception))

    def test_illegal_threshold_shape_quantizer(self):
        threshold = np.array([[4., 2.], [2., 16.]])
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=threshold,
                                               signed=True,
                                               lut_values_bitwidth=8)
        self.assertEqual(f'Threshold is expected to be flatten, but of shape {threshold.shape}',
                                   str(e.exception))

    def test_illegal_num_of_thresholds_quantizer(self):
        threshold = np.asarray([1., 4.])
        with self.assertRaises(Exception) as e:
            ActivationLutPOTInferableQuantizer(num_bits=3,
                                               lut_values=np.asarray([-25, 25]),
                                               threshold=threshold,
                                               signed=True,
                                               lut_values_bitwidth=8)
        self.assertEqual(
            'For activation, quantization per channel is not supported and threshold should be of length '
            '1 but is 2', str(e.exception))
