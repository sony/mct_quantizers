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

from torch.nn import Conv2d, Linear, ConvTranspose2d

from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer
from tests.pytorch_tests.test_pytorch_quantization_wrapper import WEIGHT
from mct_quantizers import get_ort_session_options
import onnxruntime as ort



LAYER2NAME = {Conv2d: 'conv',
              ConvTranspose2d: 'convtrans',
              Linear: 'dense'}

QUANTIZER2NAME = {WeightsPOTInferableQuantizer: 'pot',
                  WeightsSymmetricInferableQuantizer: 'sym',
                  WeightsUniformInferableQuantizer: 'unf',
                  WeightsLUTPOTInferableQuantizer: 'pot_lut',
                  WeightsLUTSymmetricInferableQuantizer: 'sym_lut'}

QUANTIZER2LAYER2ARGS = {**dict.fromkeys([WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer],
                                        {Conv2d:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'channel_axis': 0
                                              },
                                         ConvTranspose2d:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'channel_axis': 1
                                              },
                                         Linear:
                                             {'num_bits': 4,
                                              'threshold': [2.0, 0.5, 4.0],
                                              'per_channel': True,
                                              'channel_axis': 0
                                              },
                                         }),
                        WeightsUniformInferableQuantizer: {Conv2d:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'channel_axis': 0
                                                                },
                                                           ConvTranspose2d:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'channel_axis': 1
                                                                },
                                                           Linear:
                                                               {'num_bits': 4,
                                                                'min_range': [-1.0, 0.5, -0.5],
                                                                'max_range': [3.2, 1.4, 0.1],
                                                                'per_channel': True,
                                                                'channel_axis': 0
                                                                },
                                                           },
                        **dict.fromkeys([WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer],
                                         {Conv2d:
                                              {'num_bits': 4,
                                               'threshold': [2.0, 0.5, 4.0],
                                               'lut_values': [22.0, -53.0, 62.0, 0.0, -66.0, -21.0, 44.0, -40.0],
                                               'per_channel': True,
                                               'input_rank': 4,
                                               'channel_axis': 0,
                                               'lut_values_bitwidth': 8,
                                               'eps': 1e-08
                                               },
                                          ConvTranspose2d:
                                              {'num_bits': 4,
                                               'threshold': [2.0, 0.5, 4.0],
                                               'lut_values': [22.0, -53.0, 62.0, 0.0, -66.0, -21.0, 44.0, -40.0],
                                               'per_channel': True,
                                               'input_rank': 4,
                                               'channel_axis': 1,
                                               'lut_values_bitwidth': 8,
                                               'eps': 1e-08
                                               },
                                          Linear:
                                              {'num_bits': 4,
                                               'threshold': [2.0, 0.5, 4.0],
                                               'lut_values': [22.0, -53.0, 62.0, 0.0, -66.0, -21.0, 44.0, -40.0],
                                               'per_channel': True,
                                               'input_rank': 2,
                                               'channel_axis': 0,
                                               'lut_values_bitwidth': 8,
                                               'eps': 1e-08
                                               },
                                          })
                        }



def _build_model_with_quantize_wrapper(quant_weights_layer, input_shape, model_name):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.quant_weights_layer = quant_weights_layer
            self.relu = torch.nn.ReLU()

        def forward(self, inp):
            x = self.quant_weights_layer(inp)
            x = self.relu(x)
            return x

    return Model()


class BaseWeightsQuantizerBuildAndSaveTest(unittest.TestCase):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VERSION = None

    def build_and_save_model(self,
                             quantizer,
                             quantizer_params,
                             layer,
                             model_name,
                             input_shape,
                             weight_name):

        assert BaseWeightsQuantizerBuildAndSaveTest.VERSION is not None

        weights_quantizer = quantizer(**quantizer_params)
        weights_quantizer.enable_custom_impl()

        quant_weights_layer = PytorchQuantizationWrapper(layer, weights_quantizers={weight_name: weights_quantizer})

        model = _build_model_with_quantize_wrapper(quant_weights_layer=quant_weights_layer,
                                                   input_shape=input_shape,
                                                   model_name=model_name)

        wrapped_layers = [_l for _, _l in model.named_modules() if isinstance(_l, PytorchQuantizationWrapper)]
        self.assertEqual(len(wrapped_layers), 1)
        self.assertIsInstance(wrapped_layers[0].layer, type(layer))

        file_path = f'{model_name}.onnx'
        rand_inp = torch.rand(1, *input_shape).to(BaseWeightsQuantizerBuildAndSaveTest.device)
        model = model.to(BaseWeightsQuantizerBuildAndSaveTest.device)
        torch.onnx.export(model,
                          rand_inp,
                          file_path,
                          opset_version=16,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    def conv_test(self, quantizer):
        layer = Conv2d
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(in_channels=3, out_channels=3, kernel_size=4),
                                  model_name=f"{BaseWeightsQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(3, 8, 8))


    def convtrans_test(self, quantizer):
        layer = ConvTranspose2d
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(in_channels=3, out_channels=3, kernel_size=4),
                                  model_name=f"{BaseWeightsQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(3, 8, 8))

    def dense_test(self, quantizer):
        layer = Linear
        self.build_and_save_model(quantizer=quantizer,
                                  quantizer_params=QUANTIZER2LAYER2ARGS[quantizer][layer],
                                  layer=layer(in_features=3, out_features=3),
                                  model_name=f"{BaseWeightsQuantizerBuildAndSaveTest.VERSION}_"
                                             f"{LAYER2NAME[layer]}_"
                                             f"{QUANTIZER2NAME[quantizer]}",
                                  weight_name=WEIGHT,
                                  input_shape=(8, 8, 3))


class BaseWeightsQuantizerLoadAndCompareTest(unittest.TestCase):
    SAVED_VERSION = None

    def load_and_compare_model(self, quantizer_type, layer_type, weight_name):
        assert BaseWeightsQuantizerLoadAndCompareTest.SAVED_VERSION is not None

        model_path = (f"{BaseWeightsQuantizerLoadAndCompareTest.SAVED_VERSION}_"
                      f"{LAYER2NAME[layer_type]}_"
                      f"{QUANTIZER2NAME[quantizer_type]}.onnx")

        ort.InferenceSession(model_path,
                             get_ort_session_options(),
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self._check_quantizer_init_from_onnx_model(model_path)
        os.remove(model_path)

    def _check_quantizer_init_from_onnx_model(self, filepath):
        raise NotImplemented

    def conv_test(self, quantizer_type):
        layer = Conv2d
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)


    def convtrans_test(self, quantizer_type):
        layer = ConvTranspose2d
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)

    def dense_test(self, quantizer_type):
        layer = Linear
        self.load_and_compare_model(quantizer_type=quantizer_type,
                                    layer_type=layer,
                                    weight_name=WEIGHT)
