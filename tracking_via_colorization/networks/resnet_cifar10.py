from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .resnet import ResNet


class ResNetCifar10(ResNet):
    def __init__(self, is_training=True, data_format='channels_last', batch_norm_decay=0.997, batch_norm_epsilon=1e-5):
        super(ResNetCifar10, self).__init__(is_training, data_format, batch_norm_decay, batch_norm_epsilon)

        # Add one in case label starts with 1. No impact if label starts with 0.
        self.num_classes = 10 + 1

    def forward(self, x, input_data_format='channels_last'):
        # resnet_layer = self._residual_v2
        resnet_layer = self._bottleneck_residual_v2

        assert input_data_format in ('channels_first', 'channels_last')
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                x = tf.transpose(x, [0, 2, 3, 1])

        with tf.name_scope('stage0'):
            x = x / 128 - 1
            x = self._conv(x, kernel_size=3, filters=16, strides=1)
            x = self._batch_norm(x)
            x = self._relu(x)

        with tf.name_scope('stage1'):
            x = resnet_layer(x, kernel_size=3, in_filter=16, out_filter=16, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=16, out_filter=16, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=16, out_filter=16, stride=1)

        with tf.name_scope('stage2'):
            x = resnet_layer(x, kernel_size=3, in_filter=16, out_filter=32, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=32, out_filter=32, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=32, out_filter=32, stride=1)

        with tf.name_scope('stage3'):
            x = resnet_layer(x, kernel_size=3, in_filter=32, out_filter=64, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)

        with tf.name_scope('classifier'):
            x = self._global_avg_pool(x)
            x = self._fully_connected(x, self.num_classes)
        return x
