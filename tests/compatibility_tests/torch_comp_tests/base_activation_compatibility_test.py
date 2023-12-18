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
import os
import unittest
import torch
from mct_quantizers import get_ort_session_options
import onnxruntime as ort

from mct_quantizers import PytorchActivationQuantizationHolder
from mct_quantizers.pytorch.quantizers import ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, \
                                             ActivationUniformInferableQuantizer, ActivationLutPOTInferableQuantizer

LAYER2NAME = {torch.nn.ReLU: 'relu',
              torch.nn.LeakyReLU: 'leaky_relu',
              torch.add: 'add',
              torch.nn.SiLU: 'swish',
              torch.mul: 'mul'}

QUANTIZER2NAME = {ActivationPOTInferableQuantizer: 'pot',
                  ActivationSymmetricInferableQuantizer: 'sym',
                  ActivationUniformInferableQuantizer: 'unf',
                  ActivationLutPOTInferableQuantizer: 'pot_lut'}

QUANTIZER2ARGS = {**dict.fromkeys([ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer],
                             {'num_bits': 4,
                              'threshold': [0.5],
                              'signed': True
                              }),
                  ActivationUniformInferableQuantizer:
                      {'num_bits': 4,
                       'min_range': [-2.0],
                       'max_range': [3.0]
                       },
                  ActivationLutPOTInferableQuantizer:
                      {'num_bits': 4,
                       'threshold': [0.5],
                       'signed': True,
                       'lut_values': [22.0, -53.0, 62.0, 0.0, -66.0, -21.0, 44.0, -40.0],
                       'lut_values_bitwidth': 8,
                       'eps': 1e-8
                       }
                  }

def _build_model_with_quantization_holder(act_layer, quant_activation_holder, input_shape, model_name):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
            self.act_layer = act_layer
            self.quant_activation_holder = quant_activation_holder

        def forward(self, inp):
            z = self.conv(inp)
            y = self.act_layer(z)
            x = self.quant_activation_holder(y)
            return x, y

    return Model()


def _build_model_with_operator_quantization_holder(act_layer, quant_activation_holder, input_shape, model_name):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
            self.act_layer = act_layer
            self.quant_activation_holder = quant_activation_holder

        def forward(self, inp):
            z1 = self.conv1(inp)
            z2 = self.conv2(inp)
            y = self.act_layer(z1,z2)
            x = self.quant_activation_holder(y)
            return x, y

    return Model()

class BaseActivationQuantizerBuildAndSaveTest(unittest.TestCase):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VERSION = None

    def build_and_save_model(self, quantizer, quantizer_params, layer, model_name, input_shape, is_op=False):
        assert BaseActivationQuantizerBuildAndSaveTest.VERSION is not None
        act_quantizer = quantizer(**quantizer_params)
        act_quantizer.enable_custom_impl()

        quant_act_holder = PytorchActivationQuantizationHolder(activation_holder_quantizer=act_quantizer)


        if is_op:
            model = _build_model_with_operator_quantization_holder(act_layer=layer,
                                                                   quant_activation_holder=quant_act_holder,
                                                                   input_shape=input_shape,
                                                                   model_name=model_name)
        else:
            model = _build_model_with_quantization_holder(act_layer=layer,
                                                          quant_activation_holder=quant_act_holder,
                                                          input_shape=input_shape,
                                                          model_name=model_name)


        quant_holder_layer = [_l for _, _l in model.named_modules() if isinstance(_l, PytorchActivationQuantizationHolder)]
        self.assertEqual(len(quant_holder_layer), 1)

        rand_inp = torch.rand(1, *input_shape).to(BaseActivationQuantizerBuildAndSaveTest.device)
        model = model.to(BaseActivationQuantizerBuildAndSaveTest.device)

        # Verifying activation quantization after holder
        output = model(rand_inp)
        self.assertTrue(torch.any(output[0] != output[1]), "Expecting activation layer output to be different "
                                                           "from the activation holder layer output, which should be "
                                                           "quantized.")

        file_path = f'{model_name}.onnx'
        torch.onnx.export(model,
                          rand_inp,
                          file_path,
                          opset_version=16,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    def activation_test(self, quantizer, layer, is_op=False, layer_type=None):
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2ARGS[quantizer],
                                  layer=layer(),
                                  model_name=f"{BaseActivationQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer_type if layer_type is not None else layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  input_shape=(3, 8, 8),
                                  is_op=is_op)


class BaseActivationQuantizerLoadAndCompareTest(unittest.TestCase):
    SAVED_VERSION = None

    def load_and_compare_model(self, quantizer_type, layer_type):
        assert BaseActivationQuantizerLoadAndCompareTest.SAVED_VERSION is not None

        model_path = (f"{BaseActivationQuantizerLoadAndCompareTest.SAVED_VERSION}_"
                      f"{LAYER2NAME[layer_type]}_"
                      f"{QUANTIZER2NAME[quantizer_type]}.onnx")

        ort.InferenceSession(model_path,
                             get_ort_session_options(),
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self._check_quantizer_init_from_onnx_model(model_path)
        os.remove(model_path)

    def _check_quantizer_init_from_onnx_model(self, filepath):
        raise NotImplemented

    def activation_test(self, quantizer_type, layer):
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer)

