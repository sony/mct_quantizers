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

import numpy as np
import warnings

from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_lut_pot_inferable_quantizer import \
    WeightsLUTPOTInferableQuantizer
from mct_quantizers.pytorch.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer


class BasePytorchWeightsIllegalLutQuantizerTest(unittest.TestCase):

    def illegal_lut_values_inferable_quantizer_test(self, inferable_quantizer, threshold, lut_values,
                                                         per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('Expected lut values to be integers', str(e.exception))

    def illegal_num_of_lut_values_inferable_quantizer_test(self, inferable_quantizer, threshold, lut_values,
                                                                per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=2,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual(f'Expected num of lut values to be less or equal than {2 ** 2} but got '
                         f'{len(lut_values)}', str(e.exception))

    def illegal_lut_values_range_inferable_quantizer_test(self, inferable_quantizer, threshold, lut_values,
                                                               per_channel, channel_axis,
                                                               lut_values_bitwidth):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                lut_values_bitwidth=lut_values_bitwidth)
        self.assertEqual('Expected lut values in the quantization range', str(e.exception))

    def illegal_num_bit_bigger_than_lut_values_bitwidth_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                               lut_values,
                                                                               num_bits,
                                                                               per_channel, channel_axis,
                                                                               lut_values_bitwidth):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                lut_values_bitwidth=lut_values_bitwidth)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                         , str(e.exception))

    def warning_num_bit_equal_lut_values_bitwidth_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                         lut_values,
                                                                         num_bits,
                                                                         per_channel, channel_axis,
                                                                         lut_values_bitwidth):
        with warnings.catch_warnings(record=True) as w:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                lut_values_bitwidth=lut_values_bitwidth)
        self.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def illegal_num_of_thresholds_inferable_quantizer_test(self, inferable_quantizer, threshold, lut_values,
                                                           per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                         str(e.exception))

    def illegal_threshold_type_inferable_quantizer_test(self, inferable_quantizer, threshold, lut_values,
                                                        per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('Threshold is expected to be a list, but is of type <class \'numpy.ndarray\'>',
                         str(e.exception))


    def missing_channel_axis_inferable_quantizer(self, inferable_quantizer, threshold, lut_values,
                                                 per_channel):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                lut_values=lut_values,
                                threshold=threshold)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))


class TestPytorchWeightsIllegalSymmetricLutQuantizer(BasePytorchWeightsIllegalLutQuantizerTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inferable_quantizer = WeightsLUTSymmetricInferableQuantizer

    def test_illegal_lut_values_symmetric_lut_quantizer(self):
        self.illegal_lut_values_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=[2.],
                                                              lut_values=[-25.6, 25],
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_lut_values_symmetric_lut_quantizer(self):
        self.illegal_num_of_lut_values_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                     threshold=[2.],
                                                                     lut_values=[-25, 25, 3, 19, 55],
                                                                     per_channel=False,
                                                                     channel_axis=None)

    def test_illegal_lut_values_range_symmetric_lut_quantizer(self):
        self.illegal_lut_values_range_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                    threshold=[2.],
                                                                    lut_values=[-25, 25],
                                                                    per_channel=False,
                                                                    channel_axis=None,
                                                                    lut_values_bitwidth=5)

    def test_illegal_num_bit_bigger_than_lut_values_bitwidth_symmetric_lut_quantizer(self):
        self.illegal_num_bit_bigger_than_lut_values_bitwidth_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=[2.],
            lut_values=[-25, 25],
            per_channel=False,
            channel_axis=None,
            lut_values_bitwidth=8,
            num_bits=10)

    def test_warning_num_bit_equal_lut_values_bitwidth_symmetric_lut_quantizer(self):
        self.warning_num_bit_equal_lut_values_bitwidth_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=[2.],
            lut_values=[-25, 25],
            per_channel=False,
            channel_axis=None,
            lut_values_bitwidth=8,
            num_bits=8)

    def test_illegal_threshold_type_symmetric_lut_quantizer(self):
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                             threshold=np.asarray([3., 2.]),
                                                             lut_values=[-25, 25],
                                                             per_channel=False,
                                                             channel_axis=None)

    def test_illegal_num_of_thresholds_symmetric_lut_quantizer(self):
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                threshold=[2., 7.],
                                                                lut_values=[-25, 25],
                                                                per_channel=False,
                                                                channel_axis=None)

    def test_missing_channel_axis_symmetric_lut_quantizer(self):
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=self.inferable_quantizer,
                                                      threshold=[2.],
                                                      lut_values=[-25, 25],
                                                      per_channel=True)


class TestPytorchWeightsIllegalPotLutQuantizer(BasePytorchWeightsIllegalLutQuantizerTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inferable_quantizer = WeightsLUTPOTInferableQuantizer

    def test_threshold_not_pot_lut_quantizer(self):
        with self.assertRaises(Exception) as e:
            self.inferable_quantizer(num_bits=8,
                                     lut_values=[25., 85.],
                                     per_channel=False,
                                     # Not POT threshold
                                     threshold=[3])
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            self.inferable_quantizer(num_bits=8,
                                     lut_values=[25., 85.],
                                     per_channel=False,
                                     # More than one threshold in per-tensor quantization
                                     threshold=[2, 3])
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                         str(e.exception))

    def test_illegal_lut_values_pot_lut_quantizer(self):
        self.illegal_lut_values_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=[2.],
                                                              lut_values=[-25.6, 25],
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_lut_values_pot_lut_quantizer(self):
        self.illegal_num_of_lut_values_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                     threshold=[2.],
                                                                     lut_values=[-25, 25, 3, 19, 55],
                                                                     per_channel=False,
                                                                     channel_axis=None)

    def test_illegal_lut_values_range_pot_lut_quantizer(self):
        self.illegal_lut_values_range_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                    threshold=[2.],
                                                                    lut_values=[-25, 25],
                                                                    per_channel=False,
                                                                    channel_axis=None,
                                                                    lut_values_bitwidth=5)

    def test_illegal_num_bit_bigger_than_lut_values_bitwidth_pot_lut_quantizer(self):
        self.illegal_num_bit_bigger_than_lut_values_bitwidth_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=[2.],
            lut_values=[-25, 25],
            per_channel=False,
            channel_axis=None,
            lut_values_bitwidth=8,
            num_bits=10)

    def test_warning_num_bit_equal_lut_values_bitwidth_pot_lut_quantizer(self):
        self.warning_num_bit_equal_lut_values_bitwidth_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=[2.],
            lut_values=[-25, 25],
            per_channel=False,
            channel_axis=None,
            lut_values_bitwidth=8,
            num_bits=8)

    def tets_illegal_threshold_type_pot_lut_quantizer(self):
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                             threshold=[4., 2.],
                                                             lut_values=np.asarray([-25, 25]),
                                                             per_channel=False,
                                                             channel_axis=None)

    def test_illegal_num_of_thresholds_pot_lut_quantizer(self):
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                threshold=[2., 8.],
                                                                lut_values=[-25, 25],
                                                                per_channel=False,
                                                                channel_axis=None)

    def test_missing_channel_axis_pot_lut_quantizer(self):
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=self.inferable_quantizer,
                                                      threshold=[2.],
                                                      lut_values=[-25, 25],
                                                      per_channel=True)
