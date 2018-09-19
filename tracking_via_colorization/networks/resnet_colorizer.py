from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .resnet import ResNet


class ResNetColorizer(ResNet):
    def __init__(self, is_training=True, data_format='channels_last', batch_norm_decay=0.997, batch_norm_epsilon=1e-5):
        super(ResNetColorizer, self).__init__(is_training, data_format, batch_norm_decay, batch_norm_epsilon)

    def forward(self, images, labels, temperature=1.0, num_labels=16, batch=32, input_data_format='channels_last', **kwargs):
        # images [BATCH, HEIGHT(256), WIDTH(256), CHANNEL(1)]
        # labels [BATCH, HEIGHT(32), WIDTH(32), CHANNEL(1)]
        # features [BATCH, HEIGHT(32), WIDTH(32), CHANNEL(256)]
        features = self.feature(images, input_data_format)
        _, FEATURE_HEIGHT, FEATURE_WIDTH, FEATURE_CHANNELS = features.shape.as_list()

        real_batch = batch // 4
        splited_features = tf.split(features, num_or_size_splits=[3, 1] * real_batch, axis=0)
        splited_labels = tf.split(labels, num_or_size_splits=[3, 1] * real_batch, axis=0)

        # [BATCH // 4, REFERENCE(3), HEIGHT(32), WIDTH(32), CHANNEL(256)]

        with tf.name_scope('weight_matrix') as name_scope:
            reference_features = tf.stack([splited_features[i * 2] for i in range(real_batch)], axis=0)
            target_features = tf.stack([splited_features[i * 2 + 1] for i in range(real_batch)], axis=0)
            reference_labels = tf.stack([splited_labels[i * 2] for i in range(real_batch)], axis=0)
            target_labels = tf.stack([splited_labels[i * 2 + 1] for i in range(real_batch)], axis=0)

            # innerproduct [real_batch*3 * HEIGHT * WIDTH, real_batch*1 * HEIGHT * WIDTH, FEATURE_CHANNELS] -> [real_batch*3 * HEIGHT * WIDTH, real_batch*1 * HEIGHT * WIDTH]
            innerproduct = tf.reduce_sum(tf.reshape(reference_features, [-1, 1, FEATURE_CHANNELS]) * tf.reshape(target_features, [1, -1, FEATURE_CHANNELS]), axis=-1)
            weight_mat = tf.nn.softmax(innerproduct / temperature, 0)
            tf.logging.info('image after unit %s: %s', name_scope, weight_mat.get_shape())

        with tf.name_scope('prediction') as name_scope:
            # prediction [real_batch*3 * HEIGHT * WIDTH, real_batch*1 * HEIGHT * WIDTH, num_labels] -> [real_batch*1 * HEIGHT * WIDTH, num_labels]
            prediction = tf.reduce_sum(tf.expand_dims(weight_mat, -1) * tf.reshape(tf.one_hot(reference_labels, num_labels), [-1, 1, num_labels]), axis=0)
            prediction = tf.reshape(prediction, [real_batch, FEATURE_HEIGHT, FEATURE_WIDTH, num_labels])
            target_labels = tf.reshape(target_labels, [real_batch, FEATURE_HEIGHT, FEATURE_WIDTH, 1])
            tf.logging.info('image after unit %s: %s', name_scope, prediction.get_shape())

        return prediction, target_labels


    def feature(self, x, input_data_format='channels_last'):
        resnet_layer = self._residual_v2
        # resnet_layer = self._bottleneck_residual_v2

        height, width, channels = x.shape.as_list()[-3:]
        x = tf.reshape(x, [-1, height, width, channels])

        assert input_data_format in ('channels_first', 'channels_last')
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x, [0, 3, 1, 2])
            else:
                x = tf.transpose(x, [0, 2, 3, 1])

        with tf.name_scope('stage0'):
            x = x / 128 - 1
            x = self._conv(x, kernel_size=3, filters=64, strides=1)
            x = self._batch_norm(x)
            x = self._relu(x)

        with tf.name_scope('stage1'):
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=64, stride=1)

        with tf.name_scope('stage2'):
            x = resnet_layer(x, kernel_size=3, in_filter=64, out_filter=128, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=128, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=128, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=128, stride=1)

        with tf.name_scope('stage3'):
            x = resnet_layer(x, kernel_size=3, in_filter=128, out_filter=256, stride=2)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)

        with tf.name_scope('stage4'):
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)
            x = resnet_layer(x, kernel_size=3, in_filter=256, out_filter=256, stride=1)

        return x
