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

from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_pot_inferable_quantizer import \
    WeightsPOTInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_symmetric_inferable_quantizer import \
    WeightsSymmetricInferableQuantizer
from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_uniform_inferable_quantizer import \
    WeightsUniformInferableQuantizer


class TestKerasWeightsIllegalInferableQuantizers(unittest.TestCase):

    def test_illegal_num_of_thresholds_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=False,
                                               threshold=[3., 2.],
                                               channel_axis=None,
                                               input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=np.asarray([3., 2.]),
                                               channel_axis=None,
                                               input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[3., 2.],
                                               input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[3., 2.],
                                               channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def illegal_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         threshold=[3.],
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('Expected threshold to be power of 2 but is [3.]', str(e.exception))

    def test_illegal_num_of_thresholds_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=False,
                                         threshold=[3.0, 2.0],
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=np.asarray([3.0, 2.0]),
                                         channel_axis=None,
                                         input_rank=4)
        self.assertEqual('Expected threshold to be of type list but is <class \'numpy.ndarray\'>', str(e.exception))

    def test_missing_channel_axis_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=[3., 2.],
                                         input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_pot_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=[3., 2.],
                                         channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_illegal_num_of_minmax_uniform_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=False,
                                             min_range=[3., 2.],
                                             max_range=[4., 3.],
                                             channel_axis=None,
                                             input_rank=4)
        self.assertEqual('In per-tensor quantization min/max should be of length 1 but is 2', str(e.exception))

    def test_illegal_threshold_type_uniform_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3., 2.],
                                             max_range=4,
                                             channel_axis=None,
                                             input_rank=4)
        self.assertEqual('Expected max_range to be of type list but is <class \'int\'>', str(e.exception))

    def test_missing_channel_axis_uniform_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3.0, 2.0],
                                             max_range=[4.0, 3.0],
                                             input_rank=4)
        self.assertEqual('Channel axis is missing in per channel quantization', str(e.exception))

    def test_missing_input_rank_uniform_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3., 2.],
                                             max_range=[4., 3.],
                                             channel_axis=1)
        self.assertEqual('Input rank is missing in per channel quantization', str(e.exception))

    def test_out_of_range_channel_axis_POT_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsPOTInferableQuantizer(num_bits=8,
                                         per_channel=True,
                                         threshold=[3., 2.],
                                         channel_axis=-6,
                                         input_rank=4)
        self.assertEqual('Channel axis out of range. Must be -4 <= channel_axis < 4', str(e.exception))

    def test_out_of_range_channel_axis_symmetric_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsSymmetricInferableQuantizer(num_bits=8,
                                               per_channel=True,
                                               threshold=[3., 2.],
                                               channel_axis=-6,
                                               input_rank=4)
        self.assertEqual('Channel axis out of range. Must be -4 <= channel_axis < 4', str(e.exception))

    def test_out_of_range_channel_axis_uniform_quantizer(self):
        with self.assertRaises(Exception) as e:
            WeightsUniformInferableQuantizer(num_bits=8,
                                             per_channel=True,
                                             min_range=[3., 2.],
                                             max_range=[4., 3.],
                                             channel_axis=-6,
                                             input_rank=4)
        self.assertEqual('Channel axis out of range. Must be -4 <= channel_axis < 4', str(e.exception))
