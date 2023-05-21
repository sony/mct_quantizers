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

    def illegal_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('Expected cluster centers to be integers', str(e.exception))

    def illegal_num_of_cluster_centers_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                                per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=2,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual(f'Expected num of cluster centers to be less or equal than {2 ** 2} but got '
                         f'{len(cluster_centers)}', str(e.exception))

    def illegal_cluster_centers_range_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                               per_channel, channel_axis,
                                                               multiplier_n_bits):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.assertEqual('Expected cluster centers in the quantization range', str(e.exception))

    def illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                               cluster_centers,
                                                                               num_bits,
                                                                               per_channel, channel_axis,
                                                                               multiplier_n_bits):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.assertEqual('Look-Up-Table bit configuration has 10 bits. It must be less then 8'
                         , str(e.exception))

    def warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(self, inferable_quantizer, threshold,
                                                                         cluster_centers,
                                                                         num_bits,
                                                                         per_channel, channel_axis,
                                                                         multiplier_n_bits):
        with warnings.catch_warnings(record=True) as w:
            inferable_quantizer(num_bits=num_bits,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis,
                                multiplier_n_bits=multiplier_n_bits)
        self.assertTrue(
            'Num of bits equal to multiplier n bits, Please be aware LUT quantizier may be inefficient '
            'in that case, consider using SymmetricInferableQuantizer instead'
            in [str(warning.message) for warning in w])

    def illegal_num_of_thresholds_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                           per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                         str(e.exception))

    def illegal_threshold_type_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                        per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual('Threshold is expected to be numpy array, but is of type <class \'list\'>',
                         str(e.exception))

    def illegal_threshold_shape_inferable_quantizer_test(self, inferable_quantizer, threshold, cluster_centers,
                                                         per_channel, channel_axis):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold,
                                channel_axis=channel_axis)
        self.assertEqual(f'Threshold is expected to be flatten, but of shape {threshold.shape}',
                         str(e.exception))

    def missing_channel_axis_inferable_quantizer(self, inferable_quantizer, threshold, cluster_centers,
                                                 per_channel):
        with self.assertRaises(Exception) as e:
            inferable_quantizer(num_bits=8,
                                per_channel=per_channel,
                                cluster_centers=cluster_centers,
                                threshold=threshold)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))


class TestPytorchWeightsIllegalSymmetricLutQuantizer(BasePytorchWeightsIllegalLutQuantizerTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inferable_quantizer = WeightsLUTSymmetricInferableQuantizer

    def test_illegal_cluster_centers_symmetric_lut_quantizer(self):
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=np.asarray([2.]),
                                                              cluster_centers=np.asarray([-25.6, 25]),
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_cluster_centers_symmetric_lut_quantizer(self):
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                     threshold=np.asarray([2.]),
                                                                     cluster_centers=np.asarray([-25, 25, 3, 19, 55]),
                                                                     per_channel=False,
                                                                     channel_axis=None)

    def test_illegal_cluster_centers_range_symmetric_lut_quantizer(self):
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                    threshold=np.asarray([2.]),
                                                                    cluster_centers=np.asarray([-25, 25]),
                                                                    per_channel=False,
                                                                    channel_axis=None,
                                                                    multiplier_n_bits=5)

    def test_illegal_num_bit_bigger_than_multiplier_n_bits_symmetric_lut_quantizer(self):
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=np.asarray([2.]),
            cluster_centers=np.asarray([-25, 25]),
            per_channel=False,
            channel_axis=None,
            multiplier_n_bits=8,
            num_bits=10)

    def test_warning_num_bit_equal_multiplier_n_bits_symmetric_lut_quantizer(self):
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=np.asarray([2.]),
            cluster_centers=np.asarray([-25, 25]),
            per_channel=False,
            channel_axis=None,
            multiplier_n_bits=8,
            num_bits=8)

    def test_illegal_threshold_type_symmetric_lut_quantizer(self):
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                             threshold=[3., 2.],
                                                             cluster_centers=np.asarray([-25, 25]),
                                                             per_channel=False,
                                                             channel_axis=None)

    def test_illegal_threshold_shape_symmetric_lut_quantizer(self):
        self.illegal_threshold_shape_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=np.array([[3., 2.], [2., 5.]]),
                                                              cluster_centers=np.asarray([-25, 25]),
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_thresholds_symmetric_lut_quantizer(self):
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                threshold=np.asarray([2., 7.]),
                                                                cluster_centers=np.asarray([-25, 25]),
                                                                per_channel=False,
                                                                channel_axis=None)

    def test_missing_channel_axis_symmetric_lut_quantizer(self):
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=self.inferable_quantizer,
                                                      threshold=np.asarray([2.]),
                                                      cluster_centers=np.asarray([-25, 25]),
                                                      per_channel=True)


class TestPytorchWeightsIllegalPotLutQuantizer(BasePytorchWeightsIllegalLutQuantizerTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inferable_quantizer = WeightsLUTPOTInferableQuantizer

    def test_threshold_not_pot_lut_quantizer(self):
        with self.assertRaises(Exception) as e:
            self.inferable_quantizer(num_bits=8,
                                     cluster_centers=np.asarray([25., 85.]),
                                     per_channel=False,
                                     # Not POT threshold
                                     threshold=np.asarray([3]))
        self.assertEqual('Expected threshold to be power of 2 but is [3]', str(e.exception))

        with self.assertRaises(Exception) as e:
            self.inferable_quantizer(num_bits=8,
                                     cluster_centers=np.asarray([25., 85.]),
                                     per_channel=False,
                                     # More than one threshold in per-tensor quantization
                                     threshold=np.asarray([2, 3]))
        self.assertEqual('In per-tensor quantization threshold should be of length 1 but is 2',
                         str(e.exception))

    def test_illegal_cluster_centers_pot_lut_quantizer(self):
        self.illegal_cluster_centers_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=np.asarray([2.]),
                                                              cluster_centers=np.asarray([-25.6, 25]),
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_cluster_centers_pot_lut_quantizer(self):
        self.illegal_num_of_cluster_centers_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                     threshold=np.asarray([2.]),
                                                                     cluster_centers=np.asarray([-25, 25, 3, 19, 55]),
                                                                     per_channel=False,
                                                                     channel_axis=None)

    def test_illegal_cluster_centers_range_pot_lut_quantizer(self):
        self.illegal_cluster_centers_range_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                    threshold=np.asarray([2.]),
                                                                    cluster_centers=np.asarray([-25, 25]),
                                                                    per_channel=False,
                                                                    channel_axis=None,
                                                                    multiplier_n_bits=5)

    def test_illegal_num_bit_bigger_than_multiplier_n_bits_pot_lut_quantizer(self):
        self.illegal_num_bit_bigger_than_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=np.asarray([2.]),
            cluster_centers=np.asarray([-25, 25]),
            per_channel=False,
            channel_axis=None,
            multiplier_n_bits=8,
            num_bits=10)

    def test_warning_num_bit_equal_multiplier_n_bits_pot_lut_quantizer(self):
        self.warning_num_bit_equal_multiplier_n_bits_inferable_quantizer_test(
            inferable_quantizer=self.inferable_quantizer,
            threshold=np.asarray([2.]),
            cluster_centers=np.asarray([-25, 25]),
            per_channel=False,
            channel_axis=None,
            multiplier_n_bits=8,
            num_bits=8)

    def tets_illegal_threshold_type_pot_lut_quantizer(self):
        self.illegal_threshold_type_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                             threshold=[4., 2.],
                                                             cluster_centers=np.asarray([-25, 25]),
                                                             per_channel=False,
                                                             channel_axis=None)

    def test_illegal_threshold_shape_pot_lut_quantizer(self):
        self.illegal_threshold_shape_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                              threshold=np.array([[4., 2.], [2., 8.]]),
                                                              cluster_centers=np.asarray([-25, 25]),
                                                              per_channel=False,
                                                              channel_axis=None)

    def test_illegal_num_of_thresholds_pot_lut_quantizer(self):
        self.illegal_num_of_thresholds_inferable_quantizer_test(inferable_quantizer=self.inferable_quantizer,
                                                                threshold=np.asarray([2., 8.]),
                                                                cluster_centers=np.asarray([-25, 25]),
                                                                per_channel=False,
                                                                channel_axis=None)

    def test_missing_channel_axis_pot_lut_quantizer(self):
        self.missing_channel_axis_inferable_quantizer(inferable_quantizer=self.inferable_quantizer,
                                                      threshold=np.asarray([2.]),
                                                      cluster_centers=np.asarray([-25, 25]),
                                                      per_channel=True)
