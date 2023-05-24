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

from mct_quantizers.common.base_inferable_quantizer import QuantizationTarget, BaseInferableQuantizer
from mct_quantizers.common.quant_info import QuantizationMethod
from mct_quantizers.keras.activation_quantization_holder import ActivationQuantizationHolder
from mct_quantizers.keras.load_model import keras_load_quantized_model
from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper
from mct_quantizers.pytorch.load_model import pytorch_load_quantized_model
from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper

