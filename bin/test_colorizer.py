# pylint: disable=I1101
import os
import sys
import copy

import cv2
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tracking_via_colorization.utils.devices import Devices
from tracking_via_colorization.config import Config
from tracking_via_colorization.feeder.dataset.kinetics import Kinetics
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer


def dataflow(scale=1):
    ds = Kinetics('/data/public/rw/datasets/videos/kinetics', num_frames=1, skips=[0], shuffle=False)
    ds = df.MapDataComponent(ds, lambda images: cv2.resize(images[0], (256 * scale, 256 * scale)), index=1)
    ds = df.MapData(ds, lambda dp: [
        dp[0],
        dp[1],
        cv2.cvtColor(dp[1], cv2.COLOR_BGR2GRAY).reshape(256 * scale, 256 * scale, 1),
        cv2.cvtColor(np.float32(dp[1] / 255.0), cv2.COLOR_BGR2Lab),
    ])
    ds = df.MultiProcessPrefetchData(ds, nr_prefetch=32, nr_proc=1)
    return ds

def main(args):
    Config(args.config)
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s', args, Config.get_instance(), device_info)

    scale = args.scale
    ds = dataflow(scale)
    ds.reset_state()

    placeholders = {
        'features': tf.placeholder(tf.float32, (None, 2, 256 * scale, 256 * scale, 1), 'features'),
        'labels': tf.placeholder(tf.int64, (None, 2, 32 * scale, 32 * scale, 1), 'labels'),
    }
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer()
    hparams = tf.contrib.training.HParams(**hparams)

    estimator_spec= Colorizer.get('resnet', ResNetColorizer, num_reference=1)(
        features=placeholders['features'],
        labels=placeholders['labels'],
        mode=tf.estimator.ModeKeys.PREDICT,
        params=hparams
    )
    dummy_labels = np.zeros((1, 2, 32 * scale, 32 * scale, 1), dtype=np.int64)

    session = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(session, args.checkpoint)

    video_index = -1
    for frame, image, gray, lab in ds.get_data():
        if frame == 0:
            video_index += 1
            reference = copy.deepcopy([image, gray, lab])
            print('video index: %04d' % video_index)
        target = [image, gray, lab]

        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: np.expand_dims(np.stack([reference[1], target[1]], axis=0), axis=0),
            placeholders['labels']: dummy_labels,
        })

        # predictions['similarity'][ref_idx][tar_idx]
        indicies = np.argmax(predictions['similarity'], axis=-1).reshape((-1, ))
        mapping = np.zeros((32 * 32 * scale * scale, 2))
        for i, index in enumerate(indicies):
            mapping[i, :] = [index % (32 * scale), index // (32 * scale)]
        mapping = np.array(mapping, dtype=np.float32).reshape((32 * scale, 32 * scale, 2))

        height, width = mapping.shape[:2]
        
        predicted = cv2.remap(cv2.resize(reference[0], (width, height)), mapping, None, cv2.INTER_LINEAR)

        stacked = np.concatenate([cv2.resize(image, (width, height)), predicted], axis=1)
        cv2.imwrite('result_%04d_%04d.jpg' % (video_index, frame), stacked)
        #cv2.imwrite('similarity_%04d.jpg' % frame, predictions['similarity'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('-s', '--scale', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default=None)
    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
