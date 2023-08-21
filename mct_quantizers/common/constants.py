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

import importlib.util


## Frameworks

TENSORFLOW = 'tensorflow'
TORCH = 'torch'
ONNX = 'onnx'
ONNXRUNTIME = 'onnxruntime'
ONNXRUNTIME_EXTENSIONS = 'onnxruntime_extensions'

FOUND_TF = importlib.util.find_spec(TENSORFLOW) is not None
FOUND_TORCH = importlib.util.find_spec(TORCH) is not None
FOUND_ONNX = importlib.util.find_spec(ONNX) is not None
FOUND_ONNXRUNTIME = importlib.util.find_spec(ONNXRUNTIME) is not None
FOUND_ONNXRUNTIME_EXTENSIONS = importlib.util.find_spec(ONNXRUNTIME_EXTENSIONS) is not None


## Quantization properties
IS_WEIGHTS = "is_weights"
IS_ACTIVATIONS = "is_activations"
WEIGHTS_QUANTIZERS = "weights_quantizer"
WEIGHTS_QUANTIZATION_METHOD = 'weights_quantization_method'
WEIGHTS_N_BITS = 'weights_n_bits'
WEIGHTS_QUANTIZATION_PARAMS = 'weights_quantization_params'
ENABLE_WEIGHTS_QUANTIZATION = 'enable_weights_quantization'
WEIGHTS_CHANNELS_AXIS = 'weights_channels_axis'
WEIGHTS_PER_CHANNEL_THRESHOLD = 'weights_per_channel_threshold'
MIN_THRESHOLD = 'min_threshold'
ACTIVATION_QUANTIZATION_METHOD = 'activation_quantization_method'
ACTIVATION_N_BITS = 'activation_n_bits'
ACTIVATION_QUANTIZATION_PARAMS = 'activation_quantization_params'
ENABLE_ACTIVATION_QUANTIZATION = 'enable_activation_quantization'


## Quantizers

QUANTIZATION_TARGET = 'quantization_target'
QUANTIZATION_METHOD = 'quantization_method'
QUANTIZER_ID = 'identifier'

# In KerasQuantizationWrapper and PytorchQuantizationWrapper multiple quantizers are kept
ACTIVATION_QUANTIZERS = "activation_quantizers"
# In ActivationQuantizationHolder only one quantizer is used thus a new attribute name is needed
ACTIVATION_HOLDER_QUANTIZER = "activation_holder_quantizer"

# Quantizer signature parameters:
NUM_BITS = 'num_bits'
SIGNED = 'signed'
THRESHOLD = 'threshold'
PER_CHANNEL = 'per_channel'
MIN_RANGE = 'min_range'
MAX_RANGE = 'max_range'
CHANNEL_AXIS = 'channel_axis'
INPUT_RANK = 'input_rank'
LUT_VALUES = 'lut_values'


## Constant values

LAYER = "layer"
STEPS = "optimizer_step"
TRAINING = "training"
EPS = 1e-8
LUT_VALUES_BITWIDTH = 8
MCTQ_VERSION = "mctq_version"

# ONNX ops domain
ONNX_CUSTOM_OP_DOMAIN = f"mct_quantizers"
