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

import onnx
from onnx import numpy_helper

def _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path, quantizer_type):
    """
    Extracts quantization parameters from input tensors for a single quantizer node in an ONNX model.
    Useful for testing or verifying the correct quantization parameters in an ONNX model.

    Parameters:
    - onnx_file_path (str): Path to the ONNX model file.
    - quantizer_type (str): Type of the quantizer operator to search for in the ONNX model.

    Returns:
    - List: A list of numpy arrays representing the quantization parameters extracted from the input
      tensors of the specified quantizer node.
    """
    onnx_model = onnx.load(onnx_file_path)
    constname_to_constvalue = {node.output[0]: numpy_helper.to_array(node.attribute[0].t) for node in
                               onnx_model.graph.node if node.op_type == 'Constant'}
    q_nodes = [n for n in onnx_model.graph.node if n.op_type == quantizer_type]
    assert len(q_nodes) == 1
    node = q_nodes[0]
    node_qparams = [constname_to_constvalue[input_name] for input_name in node.input if
                    input_name in constname_to_constvalue]

    return node_qparams

def _get_qparams_from_attributes_for_single_quantizer(onnx_file_path, quantizer_type):
    """
    Retrieves quantization parameters directly from the attributes of a single quantizer node
    in an ONNX model.

    Parameters:
    - onnx_file_path (str): Path to the ONNX model file.
    - quantizer_type (str): Type of the quantizer operator to search for in the ONNX model.

    Returns:
    - Dict: A dictionary where keys are attribute names and values are the corresponding attribute
      values of the quantizer node.
    """
    onnx_model = onnx.load(onnx_file_path)
    q_nodes = [n for n in onnx_model.graph.node if n.op_type == quantizer_type]
    assert len(q_nodes) == 1
    node = q_nodes[0]
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
