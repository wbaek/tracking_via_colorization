from __future__ import absolute_import

import os
import logging

import tensorflow as tf
#from tensorflow.python.client import device_lib

LOGGER = logging.getLogger(__name__)


current_index = 0
class Devices():
    @staticmethod
    def get_devices(gpu_ids=None, max_gpus=-1):
        if gpu_ids is None or (not gpu_ids and max_gpus > 0):
            gpu_ids = list(range(max_gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            devices = sess.list_devices()
        # devices = device_lib.list_local_devices(config)
        num_gpus = len([d for d in devices if 'GPU' in d.device_type])

        device_name = 'GPU' if num_gpus > 0 else 'CPU'
        device_counts = num_gpus if num_gpus > 0 else 1
        return {'name': device_name, 'count': device_counts}

    @staticmethod
    def get_device_spec(device, next_=True):
        global current_index
        if device in ('cpu', 'CPU'):
            device_spec = tf.DeviceSpec(device_type='CPU', device_index=0)
        else:
            device_spec = tf.DeviceSpec(device_type=device['name'], device_index=current_index)
            if next_:
                current_index = current_index + 1
                current_index = current_index % device['count']
        LOGGER.debug(device_spec.to_string())
        return device_spec
