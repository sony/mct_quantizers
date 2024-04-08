# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict
import sys

from mct_quantizers import __version__ as mctq_version
from mct_quantizers.logger import Logger
from mct_quantizers.common.constants import MCTQ_VERSION, PYTHON_VERSION


def verify_and_init_metadata(metadata: Dict = None):
    """
    Init the metadata dictionary and verify its compliance

    Args:
        metadata (Dict): metadata dictionary. Should contain only string keys and string\interger\float values.

    Returns:
        The metadata dictionary with added default values.

    """
    if not isinstance(metadata, dict):
        Logger.error(f'metadata should be a dictionary, but got type {type(metadata)}.')
    if not all([isinstance(k, str) for k in metadata.keys()]):
        Logger.error('metadata dictionary should only have string keys.')
    if not all([isinstance(v, (str, int, float)) for v in metadata.values()]):
        Logger.error('metadata dictionary should only have string, integer or float values.')

    if metadata is None:
        metadata = {}
    if PYTHON_VERSION not in metadata:
        metadata[PYTHON_VERSION] = sys.version
    if MCTQ_VERSION not in metadata:
        metadata[MCTQ_VERSION] = mctq_version

    return metadata
