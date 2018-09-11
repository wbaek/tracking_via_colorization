import os
import sys

import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import tracking_via_colorization as tc
from tracking_via_colorization.networks.base import Model
from tracking_via_colorization.networks.resnet_cifar10 import ResNetCifar10

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    ds = df.dataset.Cifar10('train', shuffle=True)
    pp_mean = ds.get_per_pixel_mean()
    augmentors = [
        tp.imgaug.CenterPaste((40, 40)),
        tp.imgaug.RandomCrop((32, 32)),
        tp.imgaug.Flip(horiz=True),
        tp.imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
    ]
    ds = tp.AugmentImageComponent(ds, augmentors)
    ds = df.MapData(ds, lambda x: tuple(x))  # for tensorflow.data.dataset
    ds.reset_state()

    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            ds.get_data,
            output_types=(tf.float32, tf.int64),
            output_shapes=(tf.TensorShape([32, 32, 3]), tf.TensorShape([]))
        )
        dataset = dataset.batch(32)
        return dataset
    model = Model.get('resnet', ResNetCifar10)

    config = tf.estimator.RunConfig(
        model_dir='./models/test',
        session_config=None
    )
    hparams = tf.contrib.training.HParams(**{
        'data_format': 'channels_last',
        'weight_decay': 2e-4,
        'batch_norm_decay': 0.997,
        'batch_norm_epsilon': 1e-5,
        'learning_rate': 0.1,
        'momentum': 0.9,
    })

    estimator = tf.estimator.Estimator(
        model_fn=model,
        config=config,
        params=hparams
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator.train(input_fn=input_fn, steps=200)

