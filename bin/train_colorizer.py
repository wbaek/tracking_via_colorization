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


def dataflow(centroids, num_process=16, shuffle=True):
    ds = Kinetics('/data/public/rw/datasets/videos/kinetics', num_frames=4, skips=[0, 4, 4, 8], shuffle=shuffle)
    ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=1)
    ds = df.MapData(ds, lambda dp: [dp[1][:3], copy.deepcopy(dp[1][:3]), dp[1][3:], copy.deepcopy(dp[1][3:])])

    # for images (ref, target)
    for idx in [0, 2]:
        ds = df.MapDataComponent(ds, lambda images: [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(256, 256, 1) for image in images], index=idx)

    # for labels (ref, target)
    for idx in [1, 3]:
        ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (32, 32)) for image in images], index=idx)
        ds = df.MapDataComponent(ds, lambda images: [cv2.cvtColor(np.float32(image / 255.0), cv2.COLOR_BGR2Lab)[:, :, 1:] for image in images], index=idx)
        ds = df.MapDataComponent(ds, lambda images: [np.array([np.argmin(np.linalg.norm(centroids-v, axis=1)) for v in image.reshape((-1, 2))]).reshape((32, 32, 1)) for image in images], index=idx)

    # stack for tensor
    ds = df.MapData(ds, lambda dp: [np.stack(dp[0] + dp[2], axis=0), np.stack(dp[1] + dp[3], axis=0)])

    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset
    ds = df.MultiProcessPrefetchData(ds, nr_prefetch=512, nr_proc=num_process)
    return ds

def get_input_fn(name, centroids, batch_size=32, num_process=16):
    _ = name
    ds = dataflow(centroids, num_process=num_process)
    ds.reset_state()

    def input_fn():
        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                ds.get_data,
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([4, 256, 256, 1]), tf.TensorShape([4, 32, 32, 1]))
            ).batch(batch_size)
        return dataset
    return input_fn

def main(args):
    Config(args.config)
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s', args, Config.get_instance(), device_info)

    with open(args.centroids, 'rb') as f:
        loaded_centroids = np.load(f)

    input_functions = {
        'train': get_input_fn('train', loaded_centroids, Config.get_instance()['mode']['train']['batch_size'], args.num_process),
        'eval': get_input_fn('test', loaded_centroids, Config.get_instance()['mode']['eval']['batch_size'], max(1, args.num_process // 4))
    }

    model_fn = Colorizer.get('resnet', ResNetColorizer, log_steps=1)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=30,
        save_summary_steps=10,
        session_config=None
    )
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer(
        learning_rate=args.lr
    )
    hparams = tf.contrib.training.HParams(**hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams
    )

    for epoch in range(50):
        estimator.train(input_fn=input_functions['train'], steps=1000)
        estimator.evaluate(input_fn=input_functions['eval'], steps=50)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--centroids', type=str, default='./datas/centroids/centroids_16k_cifar10_10000samples.npy')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-process', type=int, default=16)
    parser.add_argument('-c', '--config', type=str, default=None)
    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
