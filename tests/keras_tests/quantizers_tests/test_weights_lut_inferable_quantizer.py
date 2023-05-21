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
import unittest

import numpy as np
import tensorflow as tf

from mct_quantizers.keras.quantizers.weights_inferable_quantizers.weights_lut_symmetric_inferable_quantizer import \
    WeightsLUTSymmetricInferableQuantizer


class TestKerasWeightsLutQuantizers(unittest.TestCase):

    def _weights_lut_quantizer_test(self, inferable_quantizer, num_bits, threshold, cluster_centers,
                                    per_channel, channel_axis, input_rank, multiplier_n_bits, eps):
        quantizer = inferable_quantizer(num_bits=num_bits,
                                        per_channel=per_channel,
                                        cluster_centers=cluster_centers,
                                        threshold=threshold,
                                        channel_axis=channel_axis,
                                        input_rank=input_rank,
                                        multiplier_n_bits=multiplier_n_bits,
                                        eps=eps)

        # check config
        quantizer_config = quantizer.get_config()
        self.assertTrue(quantizer_config['num_bits'] == num_bits)
        self.assertTrue(np.all(quantizer_config['threshold'] == np.asarray(threshold)))
        self.assertTrue(np.all(quantizer_config['cluster_centers'] == cluster_centers))
        self.assertTrue(quantizer_config['per_channel'] == per_channel)
        self.assertTrue(quantizer_config['channel_axis'] == channel_axis)
        self.assertTrue(quantizer_config['input_rank'] == input_rank)
        self.assertTrue(quantizer_config['multiplier_n_bits'] == multiplier_n_bits)
        self.assertTrue(quantizer_config['eps'] == eps)

        # test permute
        perm_vec = list(np.arange(input_rank))
        if per_channel and channel_axis not in [-1, input_rank - 1]:
            perm_vec[channel_axis] = input_rank - 1
            perm_vec[input_rank - 1] = channel_axis

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = tf.constant(np.random.rand(1, 50, 50, 3) * 100 - 50, dtype=tf.float32)

        # change the input only when channel_axis is not the last axis
        input_tensor = tf.transpose(input_tensor, perm=perm_vec)

        # Quantize tensor
        quantized_tensor = quantizer(input_tensor)

        self.assertTrue(quantized_tensor.shape == input_tensor.shape, f'Quantized tensor should be in the same shape '
                                                                      f'as the input tensor')

        # return the output's channel axis to the last axis
        # change the input only when channel_axis is not the last axis
        quantized_tensor = tf.transpose(quantized_tensor, perm=perm_vec)

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))

        max_threshold = np.max(np.abs(threshold))
        delta_threshold = 1 / (2 ** (multiplier_n_bits - 1))

        self.assertTrue(np.max(
            quantized_tensor) <= max_threshold - delta_threshold, f'Quantized values should not contain values greater '
                                                                  f'than maximal threshold ')
        self.assertTrue(np.min(
            quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal '
                                                 f'threshold ')

        self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                        f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                        f'{len(np.unique(quantized_tensor))} unique values')

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (multiplier_n_bits - 1) - 1
        clip_min = -2 ** (multiplier_n_bits - 1)

        if per_channel:
            for i in range(len(threshold)):
                channel_slice_i = quantized_tensor[:, :, :, i]
                channel_quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold[i]
                self.assertTrue(len(np.unique(channel_slice_i)) <= 2 ** num_bits,
                                f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                                f'{len(np.unique(channel_slice_i))} unique values')
                self.assertTrue(np.all(np.unique(channel_slice_i) == np.sort(channel_quant_tensor_values)))

                # Check quantized tensor assigned correctly
                tensor = tf.clip_by_value((input_tensor / (threshold[i] + eps)) * (2 ** (num_bits - 1)),
                                          clip_value_max=clip_max, clip_value_min=clip_min)
                tensor = tf.expand_dims(tf.transpose(tensor, perm=perm_vec)[:, :, :, i], -1)
                expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
                cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
                centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

                self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[i] == channel_slice_i),
                                "Quantized tensor values weren't assigned correctly")
        else:
            quant_tensor_values = cluster_centers / (2 ** (multiplier_n_bits - 1)) * threshold
            self.assertTrue(len(np.unique(quantized_tensor)) <= 2 ** num_bits,
                            f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has '
                            f'{len(np.unique(quantized_tensor))} unique values')
            self.assertTrue(np.all(np.unique(quantized_tensor) == np.sort(quant_tensor_values)))

            # Check quantized tensor assigned correctly
            tensor = tf.clip_by_value((input_tensor / (threshold[0] + eps)) * (2 ** (num_bits - 1)),
                                      clip_value_max=clip_max, clip_value_min=clip_min)
            tensor = tf.expand_dims(tensor, -1)
            expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
            cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
            centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

            self.assertTrue(np.all(centers / (2 ** (multiplier_n_bits - 1)) * threshold[0] == quantized_tensor),
                            "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(quantized_tensor < 0),
                        f'Expected some values to be negative but quantized tensor is {quantized_tensor}')

    def test_weights_symmetric_lut_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer
        cluster_centers = np.asarray([-25, 25])
        per_channel = True
        input_rank = 4
        num_bits = 8

        # test per channel
        threshold = [3., 8., 7.]
        channel_axis = 3
        multiplier_n_bits = 8
        eps = 1e-8
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)

        # test per channel and channel axis is not last
        threshold = [3., 8., 7.]
        channel_axis = 1
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)

        # test per tensor
        threshold = [3.]
        channel_axis = None
        per_channel = False
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)

    def test_weights_pot_lut_quantizer(self):
        inferable_quantizer = WeightsLUTSymmetricInferableQuantizer

        cluster_centers = np.asarray([-25, 25])
        input_rank = 4
        num_bits = 8

        # test per channel
        per_channel = True
        threshold = [2., 8., 32.]
        channel_axis = 3
        multiplier_n_bits = 8
        eps = 1e-8
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)

        # test per channel and channel axis is not last
        threshold = [2., 8., 32.]
        channel_axis = 1
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)

        # test per tensor
        threshold = [4.]
        channel_axis = None
        per_channel = False
        self._weights_lut_quantizer_test(inferable_quantizer=inferable_quantizer, num_bits=num_bits,
                                         threshold=threshold, cluster_centers=cluster_centers,
                                         per_channel=per_channel, channel_axis=channel_axis,
                                         input_rank=input_rank, multiplier_n_bits=multiplier_n_bits,
                                         eps=eps)
