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

from mct_quantizers.common.constants import FOUND_ONNXRUNTIME, FOUND_ONNXRUNTIME_EXTENSIONS

if FOUND_ONNXRUNTIME and FOUND_ONNXRUNTIME_EXTENSIONS:
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    def get_ort_session_options() -> ort.SessionOptions:
        """
        Returns: Session options for loading onnxruntime inference session
         with custom implementation of onnx ops.
        """
        opt = ort.SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        return opt

else:
    def get_ort_session_options():
        raise Exception('Installing onnxruntime onnxruntime-extensions and is mandatory '
                        'when using get_ort_session_options. '
                        'Could not find a package.')

