import onnx
from onnx import numpy_helper

def _get_qparams_from_input_tensors_for_single_quantizer(onnx_file_path,
                                                         quantizer_type):
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

def _get_qparams_from_attributes_for_single_quantizer(onnx_file_path,
                                                      quantizer_type):
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