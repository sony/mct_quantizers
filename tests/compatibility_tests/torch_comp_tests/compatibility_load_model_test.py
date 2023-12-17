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
import torch

from mct_quantizers.pytorch.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer, \
    ActivationPOTInferableQuantizer, ActivationUniformInferableQuantizer, \
    ActivationSymmetricInferableQuantizer
from tests.compatibility_tests.torch_comp_tests.base_activation_compatibility_test import \
    BaseActivationQuantizerLoadAndCompareTest
from tests.compatibility_tests.torch_comp_tests.base_weights_compatibility_test import BaseWeightsQuantizerLoadAndCompareTest
from tests.compatibility_tests.torch_comp_tests.onnx_utils import _get_qparams_from_input_tensors_for_single_quantizer, \
    _get_qparams_from_attributes_for_single_quantizer


class WeightsPOTQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):


    def setUp(self):
        self.quantizer_type = WeightsPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

    def _check_quantizer_init_from_onnx_model(self, onnx_file_path):
        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path,
                                                                                 'WeightsPOTQuantizer')
        onnx_threshold = node_qparams[0]
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path,
                                                                              'WeightsPOTQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        WeightsPOTInferableQuantizer(num_bits=onnx_nbits,
                                     threshold=onnx_threshold.tolist(),
                                     per_channel=onnx_per_channel,
                                     channel_axis=onnx_channel_axis)


class WeightsSymmetricQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

    def _check_quantizer_init_from_onnx_model(self, onnx_file_path):
        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path,
                                                                                 'WeightsSymmetricQuantizer')
        onnx_threshold = node_qparams[0]
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path,
                                                                              'WeightsSymmetricQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        WeightsSymmetricInferableQuantizer(num_bits=onnx_nbits,
                                           threshold=onnx_threshold.tolist(),
                                           per_channel=onnx_per_channel,
                                           channel_axis=onnx_channel_axis)


class WeightsUniformQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

    def _check_quantizer_init_from_onnx_model(self, onnx_file_path):
        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_min_range = node_qparams[0]
        onnx_max_range = node_qparams[1]

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        WeightsUniformInferableQuantizer(num_bits=onnx_nbits,
                                         min_range=onnx_min_range.tolist(),
                                         max_range=onnx_max_range.tolist(),
                                         per_channel=onnx_per_channel,
                                         channel_axis=onnx_channel_axis)


class WeightsPOTLutQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

    def _check_quantizer_init_from_onnx_model(self, onnx_file_path):
        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsLUTPOTQuantizer')
        lut_values_onnx = node_qparams[0]
        threshold_onnx = node_qparams[1]
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsLUTPOTQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        onnx_eps = node_qparams['eps']
        onnx_input_rank = node_qparams['input_rank']
        onnx_signed = node_qparams['signed']
        onnx_lut_values_bitwidth = node_qparams['lut_values_bitwidth']
        WeightsLUTPOTInferableQuantizer(num_bits=onnx_nbits,
                                        lut_values=lut_values_onnx.tolist(),
                                        threshold=threshold_onnx.tolist(),
                                        per_channel=onnx_per_channel,
                                        channel_axis=onnx_channel_axis,
                                        input_rank=onnx_input_rank,
                                        lut_values_bitwidth=onnx_lut_values_bitwidth,
                                        eps=onnx_eps)

class WeightsSymmetricLutQuantizerLoadAndCompareTest(BaseWeightsQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = WeightsLUTSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.conv_test(self.quantizer_type)
        self.convtrans_test(self.quantizer_type)
        self.dense_test(self.quantizer_type)

    def _check_quantizer_init_from_onnx_model(self, onnx_file_path):
        node_qparams = _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, 'WeightsLUTSymmetricQuantizer')
        lut_values_onnx = node_qparams[0]
        threshold_onnx = node_qparams[1]
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'WeightsLUTSymmetricQuantizer')
        onnx_nbits = node_qparams['num_bits']
        onnx_per_channel = node_qparams['per_channel']
        onnx_channel_axis = node_qparams['channel_axis']
        onnx_eps = node_qparams['eps']
        onnx_input_rank = node_qparams['input_rank']
        onnx_signed = node_qparams['signed']
        onnx_lut_values_bitwidth = node_qparams['lut_values_bitwidth']
        WeightsLUTSymmetricInferableQuantizer(num_bits=onnx_nbits,
                                              lut_values=lut_values_onnx.tolist(),
                                              threshold=threshold_onnx.tolist(),
                                              per_channel=onnx_per_channel,
                                              channel_axis=onnx_channel_axis,
                                              input_rank=onnx_input_rank,
                                              lut_values_bitwidth=onnx_lut_values_bitwidth,
                                              eps=onnx_eps)


class ActivationPOTQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationPOTInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, torch.nn.ReLU)
        self.activation_test(self.quantizer_type, torch.nn.LeakyReLU)
        self.activation_test(self.quantizer_type, torch.add)
        self.activation_test(self.quantizer_type, torch.nn.SiLU)
        self.activation_test(self.quantizer_type, torch.mul)

    def _check_quantizer_init_from_onnx_model(self, filepath):
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(filepath, 'ActivationPOTQuantizer')
        onnx_threshold = node_qparams['threshold']
        self.assertTrue(isinstance(onnx_threshold, float), f"Expected onnx_threshold to be float but is {type(onnx_threshold)}")

        onnx_signed = node_qparams['signed']
        onnx_nbits = node_qparams['num_bits']
        ActivationPOTInferableQuantizer(num_bits=onnx_nbits,
                                        threshold=[onnx_threshold],
                                        signed=onnx_signed)


class ActivationSymmetricQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationSymmetricInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, torch.nn.ReLU)
        self.activation_test(self.quantizer_type, torch.nn.LeakyReLU)
        self.activation_test(self.quantizer_type, torch.add)
        self.activation_test(self.quantizer_type, torch.nn.SiLU)
        self.activation_test(self.quantizer_type, torch.mul)

    def _check_quantizer_init_from_onnx_model(self, filepath):
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(filepath, 'ActivationSymmetricQuantizer')
        onnx_threshold = node_qparams['threshold']
        self.assertTrue(isinstance(onnx_threshold, float), f"Expected onnx_threshold to be float but is {type(onnx_threshold)}")

        onnx_signed = node_qparams['signed']
        onnx_nbits = node_qparams['num_bits']
        ActivationSymmetricInferableQuantizer(num_bits=onnx_nbits,
                                              threshold=[onnx_threshold],
                                              signed=onnx_signed)



class ActivationUniformQuantizerLoadAndCompareTest(BaseActivationQuantizerLoadAndCompareTest):

    def setUp(self):
        self.quantizer_type = ActivationUniformInferableQuantizer

    def test_weights_uniform_quantizer(self):
        self.activation_test(self.quantizer_type, torch.nn.ReLU)
        self.activation_test(self.quantizer_type, torch.nn.LeakyReLU)
        self.activation_test(self.quantizer_type, torch.add)
        self.activation_test(self.quantizer_type, torch.nn.SiLU)
        self.activation_test(self.quantizer_type, torch.mul)
    def _check_quantizer_init_from_onnx_model(self, filepath):
        node_qparams = _get_qparams_from_attributes_for_single_quantizer(filepath, 'ActivationUniformQuantizer')
        onnx_min_range = node_qparams['min_range']
        self.assertTrue(isinstance(onnx_min_range, float), f"Expected onnx_min_range to be float but is {type(onnx_min_range)}")
        onnx_max_range = node_qparams['max_range']
        self.assertTrue(isinstance(onnx_max_range, float), f"Expected onnx_max_range to be float but is {type(onnx_max_range)}")
        onnx_nbits = node_qparams['num_bits']
        ActivationUniformInferableQuantizer(num_bits=onnx_nbits,
                                            min_range=[onnx_min_range],
                                            max_range=[onnx_max_range])

