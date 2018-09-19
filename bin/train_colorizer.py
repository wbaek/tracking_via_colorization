import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tracking_via_colorization.config import Config
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer


def get_input_fn(name, centroids, batch_size=32):
    is_training = name == 'train'
    ds = df.dataset.Cifar10(name, shuffle=is_training)
    augmentors = [
        tp.imgaug.CenterPaste((40, 40)),
        tp.imgaug.RandomCrop((32, 32)),
        # tp.imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
    ]
    if is_training:
        ds = df.RepeatedData(ds, -1)
        ds = tp.AugmentImageComponent(ds, augmentors)

    ds = df.MapData(ds, lambda dp: [cv2.resize(dp[0], (256, 256)), cv2.resize(dp[0], (32, 32))])
    ds = df.MapData(ds, lambda dp: [cv2.cvtColor(dp[0], cv2.COLOR_RGB2GRAY).reshape((256, 256, 1)), cv2.cvtColor(dp[1], cv2.COLOR_RGB2Lab)[:, :, 1:]])
    ds = df.MapDataComponent(ds, lambda label: np.array([np.argmin(np.linalg.norm(centroids-v, axis=1)) for v in label.reshape((-1, 2))]).reshape((32, 32, 1)), index=1)

    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset
    ds.reset_state()

    def input_fn():
        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                ds.get_data,
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([256, 256, 1]), tf.TensorShape([32, 32, 1]))
            ).batch(batch_size)
        return dataset
    return input_fn

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--centroids', type=str, default='./datas/centroids/centroids_16k_cifar10_10000samples.npy')
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    Config(args.config)
    print(Config.get_instance())

    with open(args.centroids, 'rb') as f:
        centroids = np.load(f)

    input_functions = {
        'train': get_input_fn('train', centroids, Config.get_instance()['mode']['train']['batch_size']),
        'eval': get_input_fn('test', centroids, Config.get_instance()['mode']['eval']['batch_size'])
    }

    model_fn = Colorizer.get('resnet', ResNetColorizer, log_steps=1000)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=10,
        session_config=None
    )
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.MomentumOptimizer(
        learning_rate=Config.get_instance()['optimizer']['learning_rate'],
        momentum=Config.get_instance()['optimizer']['momentum']
    )
    hparams = tf.contrib.training.HParams(**hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=hparams
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    for epoch in range(50):
        estimator.train(input_fn=input_functions['train'], steps=(50000 // 32))
        estimator.evaluate(input_fn=input_functions['eval'], steps=100)

