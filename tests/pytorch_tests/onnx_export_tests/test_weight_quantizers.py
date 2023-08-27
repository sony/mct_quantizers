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
from mct_quantizers import __version__ as mctq_version

import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers import get_ort_session_options
from mct_quantizers import pytorch_quantizers
from mct_quantizers.common.constants import MCTQ_VERSION
from mct_quantizers.pytorch.quantizer_utils import get_working_device
from tests.pytorch_tests.onnx_export_tests.test_activation_quantizers import _export_model, _check_load_and_inference, _get_qparams_from_attributes_for_single_quantizer, _get_qparams_from_input_tensors_for_single_quantizer


class TestONNXExportWeightsQuantizers(unittest.TestCase):

    def setUp(self):
        self.device = get_working_device()

    def test_onnx_weight_symmetric(self):
        thresholds = [3., 6., 2., 3.]
        num_bits = 2
        per_channel = True
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                                          per_channel=per_channel,
                                                                          threshold=thresholds,
                                                                          channel_axis=channel_axis,
                                                                          )
        quantizer.enable_custom_impl()


        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 5),
                                                          {'weight': quantizer}).to(self.device)
        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_threshold = node_qparams[0]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']

        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'
        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'


    def test_onnx_weight_pot(self):
        thresholds = [0.5, 0.25, 2., 1.]
        num_bits = 2
        per_channel = True
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                                    per_channel=per_channel,
                                                                    threshold=thresholds,
                                                                    channel_axis=channel_axis,
                                                                    )
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 3),
                                                          {'weight': quantizer}).to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_threshold = node_qparams[0]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'

        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'


    def test_onnx_weight_uniform(self):
        min_range = [0.1, 0.1, 1., 0.]
        max_range = [0.5, 0.25, 2., 1.]
        num_bits = 2
        per_channel = True
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                                        min_range=min_range,
                                                                        max_range=max_range,
                                                                        per_channel=per_channel,
                                                                        channel_axis=channel_axis)
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 3),
                                                          {'weight': quantizer}).to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_min_range = node_qparams[0]
        onnx_max_range = node_qparams[1]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']

        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(np.zeros(shape=(4,))==onnx_min_range), f'Expected min_range in quantizer to be zeros after range adjustment but found {onnx_min_range}'
        assert np.all(max_range==onnx_max_range), f'Expected max_range in quantizer to be {max_range} but found {onnx_max_range}'
        assert onnx_channel_axis == channel_axis, f'Expected channel_axis in quantizer to be {channel_axis} but found {onnx_channel_axis}'

        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'

    def test_onnx_weight_symmetric_per_tensor(self):
        thresholds = [3.]
        num_bits = 2
        per_channel = False
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsSymmetricInferableQuantizer(num_bits=num_bits,
                                                                          per_channel=per_channel,
                                                                          threshold=thresholds,
                                                                          channel_axis=channel_axis
                                                                          )
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 5),
                                                          {'weight': quantizer}).to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_threshold = node_qparams[0]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'



        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'
        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'


    def test_onnx_weight_pot_per_tensor(self):
        thresholds = [0.5]
        num_bits = 2
        per_channel = False
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsPOTInferableQuantizer(num_bits=num_bits,
                                                                    per_channel=per_channel,
                                                                    threshold=thresholds,
                                                                    channel_axis=channel_axis
                                                                    )
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 3),
                                                          {'weight': quantizer}).to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_threshold = node_qparams[0]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']

        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'
        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'


    def test_onnx_weight_uniform_per_tensor(self):
        min_range = [0.1]
        max_range = [0.5]
        num_bits = 2
        per_channel = False
        channel_axis = 0

        quantizer = pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=num_bits,
                                                                        min_range=min_range,
                                                                        max_range=max_range,
                                                                        per_channel=per_channel,
                                                                        channel_axis=channel_axis)
        quantizer.enable_custom_impl()


        layer_with_quantizer = PytorchQuantizationWrapper(torch.nn.Conv2d(3, 4, 3),
                                                          {'weight': quantizer}).to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_min_range = node_qparams[0]
        onnx_max_range = node_qparams[1]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(np.zeros(shape=(1,))==onnx_min_range), f'Expected min_range in quantizer to be zeros after range adjustment but found {onnx_min_range}'
        assert np.all(max_range==onnx_max_range), f'Expected max_range in quantizer to be {max_range} but found {onnx_max_range}'
        assert onnx_channel_axis == channel_axis, f'Expected channel_axis in quantizer to be {channel_axis} but found {onnx_channel_axis}'
        onnx_signed = node_qparams['signed']
        assert onnx_signed == True, f'Expected signed in weight quantizer to be True but is {onnx_signed}'
        assert node_qparams[MCTQ_VERSION] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams[MCTQ_VERSION]}'


    def test_illegal_arguments(self):
        # Test valid min/max len
        per_channel = False
        thresholds = [3., 3.]
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  threshold=thresholds,
                                                                  channel_axis=0)
        self.assertEqual(f"In per-tensor quantization threshold should be of length 1 but is {len(thresholds)}", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  threshold=thresholds,
                                                                  channel_axis=0)
        self.assertEqual(f"In per-tensor quantization threshold should be of length 1 but is {len(thresholds)}", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  min_range=[-t for t in thresholds],
                                                                max_range=[1.],
                                                                  channel_axis=0)
        self.assertEqual(f"In per-tensor quantization min_range should be of length 1 but is {len(thresholds)}", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  min_range=[0.],
                                                                max_range=thresholds,
                                                                  channel_axis=0)
        self.assertEqual(f"In per-tensor quantization max_range should be of length 1 but is {len(thresholds)}", str(e.exception))


    # Test channel_axis exist in per-channel quantization
        per_channel = True
        thresholds = [3., 3.]
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsSymmetricInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  threshold=thresholds)
        self.assertEqual(f"Channel axis is missing in per channel quantization", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsPOTInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  threshold=thresholds)
        self.assertEqual(f"Channel axis is missing in per channel quantization", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  min_range=[-t for t in thresholds],
                                                                max_range=thresholds)
        self.assertEqual(f"Channel axis is missing in per channel quantization", str(e.exception))

        # Check min > max
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                  per_channel=per_channel,
                                                                  min_range=[-3.],
                                                                max_range=[-5.])
        self.assertEqual(f"Max range must be greater than min value but min is -3.0 and max is -5.0", str(e.exception))
        with self.assertRaises(Exception) as e:
            pytorch_quantizers.WeightsUniformInferableQuantizer(num_bits=8,
                                                                  per_channel=True,
                                                                  min_range=[-3., -2.],
                                                                max_range=[5., -3.])
        self.assertEqual(f"Max range must be greater than min value but min is -2.0 and max is -3.0", str(e.exception))
