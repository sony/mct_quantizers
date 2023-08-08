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
from mct_quantizers.pytorch.quantizer_utils import get_working_device


class TestONNXExportWeightsQuantizers(unittest.TestCase):

    def setUp(self):
        self.device = get_working_device()

    def _export_model(self, model, onnx_file_path, input_shape=(1,3,8,8), opset_version=16):
        model_input_torch = torch.rand(input_shape).to(get_working_device())
        torch.onnx.export(model,
                          model_input_torch,
                          onnx_file_path,
                          opset_version=opset_version,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    def _check_load_and_inference(self, onnx_file_path, input_shape=(1,3,8,8)):
        sess = ort.InferenceSession(onnx_file_path, get_ort_session_options())
        model_input_np = np.random.rand(*input_shape).astype(np.float32)
        input_feed = {i.name: t for i, t in zip(sess.get_inputs(), [model_input_np])}
        sess.run([o.name for o in sess.get_outputs()], input_feed)

    def _get_qparams_for_single_quantizer(self, onnx_file_path, quantizer_type):
        # Test correct quantization params in onnx model
        onnx_model = onnx.load(onnx_file_path)
        # Create a dictionary with the name of the initializers as the key and their tensor values as the value
        constname_to_constvalue = {node.output[0]: numpy_helper.to_array(node.attribute[0].t) for node in
                                   onnx_model.graph.node if node.op_type == 'Constant'}
        q_nodes = [n for n in onnx_model.graph.node if n.op_type == quantizer_type]
        assert len(q_nodes) == 1
        node = q_nodes[0]
        node_qparams = [constname_to_constvalue[input_name] for input_name in node.input if
                        input_name in constname_to_constvalue]
        return node_qparams


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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_threshold = node_qparams[1]
        onnx_per_channel = node_qparams[2]
        onnx_channel_axis = node_qparams[3]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'


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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_threshold = node_qparams[1]
        onnx_per_channel = node_qparams[2]
        onnx_channel_axis = node_qparams[3]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'


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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_min_range = node_qparams[1]
        onnx_max_range = node_qparams[2]
        onnx_per_channel = node_qparams[3]
        onnx_channel_axis = node_qparams[4]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(np.zeros(shape=(4,))==onnx_min_range), f'Expected min_range in quantizer to be zeros after range adjustment but found {onnx_min_range}'
        assert np.all(max_range==onnx_max_range), f'Expected max_range in quantizer to be {max_range} but found {onnx_max_range}'
        assert onnx_channel_axis == channel_axis, f'Expected channel_axis in quantizer to be {channel_axis} but found {onnx_channel_axis}'







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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsSymmetricQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_threshold = node_qparams[1]
        onnx_per_channel = node_qparams[2]
        onnx_channel_axis = node_qparams[3]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'


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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsPOTQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_threshold = node_qparams[1]
        onnx_per_channel = node_qparams[2]
        onnx_channel_axis = node_qparams[3]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(thresholds==onnx_threshold), f'Expected threshold in quantizer to be {thresholds} but found {onnx_threshold}'
        assert onnx_channel_axis == channel_axis, f'Expected threshold in quantizer to be {channel_axis} but found ' \
                                             f'{onnx_channel_axis}'


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
        self._export_model(layer_with_quantizer,
                           onnx_file_path)

        self._check_load_and_inference(onnx_file_path)

        node_qparams = self._get_qparams_for_single_quantizer(onnx_file_path, 'WeightsUniformQuantizer')
        onnx_nbits = node_qparams[0]
        onnx_min_range = node_qparams[1]
        onnx_max_range = node_qparams[2]
        onnx_per_channel = node_qparams[3]
        onnx_channel_axis = node_qparams[4]


        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert onnx_per_channel == per_channel, f'Expected per_channel in quantizer to be {per_channel} but found {onnx_per_channel}'
        assert np.all(np.zeros(shape=(1,))==onnx_min_range), f'Expected min_range in quantizer to be zeros after range adjustment but found {onnx_min_range}'
        assert np.all(max_range==onnx_max_range), f'Expected max_range in quantizer to be {max_range} but found {onnx_max_range}'
        assert onnx_channel_axis == channel_axis, f'Expected channel_axis in quantizer to be {channel_axis} but found {onnx_channel_axis}'



