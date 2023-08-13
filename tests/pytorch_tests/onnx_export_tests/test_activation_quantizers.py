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

from mct_quantizers import PytorchActivationQuantizationHolder
from mct_quantizers import get_ort_session_options
from mct_quantizers import pytorch_quantizers
from mct_quantizers.pytorch.quantizer_utils import get_working_device


def _export_model(model, onnx_file_path, inputs_for_inference, opset_version=16):
    print(f'Exporting model to {onnx_file_path}')
    torch.onnx.export(model,
                      inputs_for_inference,
                      onnx_file_path,
                      opset_version=opset_version,
                      verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


def _check_load_and_inference(onnx_file_path, input_shape=(1, 3, 8, 8)):
    sess = ort.InferenceSession(onnx_file_path, get_ort_session_options())
    model_input_np = np.random.rand(*input_shape).astype(np.float32)
    input_feed = {i.name: t for i, t in zip(sess.get_inputs(), [model_input_np])}
    sess.run([o.name for o in sess.get_outputs()], input_feed)


def _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, quantizer_type):
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


def _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, quantizer_type):
    # Test correct quantization params in onnx model
    onnx_model = onnx.load(onnx_file_path)
    q_nodes = [n for n in onnx_model.graph.node if n.op_type == quantizer_type]
    assert len(q_nodes) == 1
    node = q_nodes[0]
    # Extract attributes as a key-value dictionary
    attributes_dict = {}
    for attribute in node.attribute:
        if attribute.HasField('f'):
            attributes_dict[attribute.name] = attribute.f
        elif attribute.HasField('i'):
            attributes_dict[attribute.name] = attribute.i
        elif attribute.HasField('s'):
            attributes_dict[attribute.name] = attribute.s.decode('utf-8')
        else:
            raise Exception(f'Encountered an unfamiliar attribute type in attribute {attribute}')

    return attributes_dict



class TestONNXExportActivationQuantizers(unittest.TestCase):

    def setUp(self):
        self.device = get_working_device()

    def test_onnx_activation_symmetric(self):
        num_bits = 3
        thresholds = [5.]
        signed = False
        quantizer = pytorch_quantizers.ActivationSymmetricInferableQuantizer(num_bits=num_bits,
                                                                             threshold=thresholds,
                                                                             signed=signed)
        quantizer.enable_custom_impl()
        layer_with_quantizer = PytorchActivationQuantizationHolder(quantizer)
        layer_with_quantizer.to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'ActivationSymmetricQuantizer')
        onnx_threshold = node_qparams['threshold']
        onnx_signed = node_qparams['signed']
        onnx_nbits = node_qparams['num_bits']

        assert onnx_threshold == thresholds[0], f'Expected threshold in quantizer to be {thresholds} but found ' \
                                             f'{onnx_threshold}'
        assert onnx_signed == signed, f'Expected signed in quantizer to be {signed} but found {onnx_signed}'
        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert node_qparams['mctq_version'] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams["mctq_version"]}'



    def test_onnx_activation_pot(self):
        num_bits = 3
        thresholds = [4.]
        signed = True
        quantizer = pytorch_quantizers.ActivationPOTInferableQuantizer(num_bits=num_bits,
                                                    threshold=thresholds,
                                                    signed=signed)
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchActivationQuantizationHolder(quantizer)
        layer_with_quantizer.to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))

        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'ActivationPOTQuantizer')
        onnx_threshold = node_qparams['threshold']
        onnx_signed = node_qparams['signed']
        onnx_nbits = node_qparams['num_bits']

        assert onnx_threshold == thresholds[0], f'Expected threshold in quantizer to be {thresholds} but found ' \
                                             f'{onnx_threshold}'
        assert onnx_signed == signed, f'Expected signed in quantizer to be {signed} but found {onnx_signed}'
        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert node_qparams['mctq_version'] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams["mctq_version"]}'

    def test_onnx_activation_uniform(self):
        num_bits = 3
        min_range = [4.]
        max_range = [5.]

        quantizer = pytorch_quantizers.ActivationUniformInferableQuantizer(num_bits=num_bits,
                                                                           min_range=min_range,
                                                                           max_range=max_range)
        quantizer.enable_custom_impl()

        layer_with_quantizer = PytorchActivationQuantizationHolder(quantizer)
        layer_with_quantizer.to(self.device)

        _, onnx_file_path = tempfile.mkstemp('.onnx')
        _export_model(layer_with_quantizer,
                      onnx_file_path,
                      torch.rand(1, 3, 8, 8).to(self.device))


        _check_load_and_inference(onnx_file_path)

        node_qparams = _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, 'ActivationUniformQuantizer')
        onnx_min_range = node_qparams['min_range']
        onnx_max_range = node_qparams['max_range']
        onnx_nbits = node_qparams['num_bits']


        assert onnx_min_range == 0, f'Expected threshold in min_range to be 0 (after range correction to include zero)' \
                                      f' but found {onnx_min_range}'
        assert onnx_max_range == max_range[0], f'Expected max_range in quantizer to be {max_range} but found {onnx_max_range}'
        assert onnx_nbits == num_bits, f'Expected num_bits in quantizer to be {num_bits} but found {onnx_nbits}'
        assert node_qparams['mctq_version'] == mctq_version, f'Expected version to be {mctq_version} but is {node_qparams["mctq_version"]}'



