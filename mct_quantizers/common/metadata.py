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
from typing import Dict, Any
import sys

from mct_quantizers import __version__ as mctq_version
from mct_quantizers.logger import Logger
from mct_quantizers.common.constants import MCTQ_VERSION, PYTHON_VERSION


def verify_and_init_metadata(metadata: Dict = None):
    """
    Init the metadata dictionary and verify its compliance

    Args:
        metadata (Dict): metadata dictionary. Should contain only string keys and string\interger\float values or
        list/dictionaries with these allowed data types.

    Returns:
        The metadata dictionary with added default values.

    """

    def _validate_metadata_value(value: Any) -> bool:
        """
        Validates that the value is either an int, float, str, list, or dict.
        For lists, all elements must be int, float, or str.
        For dicts, all keys must be str and all values must be int, float, str, list, or dict.

        Args:
            value (Any): The value to be validated.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        if isinstance(value, (int, float, str)):
            return True
        elif isinstance(value, list):
            return all(_validate_metadata_value(item) for item in value)
        elif isinstance(value, dict):
            return all(isinstance(k, str) and _validate_metadata_value(v) for k, v in value.items())
        return False

    if not isinstance(metadata, dict):
        Logger.error(f'metadata should be a dictionary, but got type {type(metadata)}.')
    if not all(isinstance(k, str) for k in metadata.keys()):
        Logger.error('metadata dictionary should only have string keys.')
    if not all(_validate_metadata_value(v) for v in metadata.values()):
        Logger.warning('metadata dictionary values should be strings, integers, floats, lists, '
                       'or dictionaries with appropriate inner values. Other types may cause issues '
                       'with saving/loading the metadata.')

    if metadata is None:
        metadata = {}
    if PYTHON_VERSION not in metadata:
        metadata[PYTHON_VERSION] = sys.version
    if MCTQ_VERSION not in metadata:
        metadata[MCTQ_VERSION] = mctq_version

    return metadata
