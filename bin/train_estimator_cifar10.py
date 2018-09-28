import os
import sys

import tensorflow as tf
import tensorpack as tp
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tracking_via_colorization.config import Config
from tracking_via_colorization.networks.classifier import Classifier
from tracking_via_colorization.networks.resnet_cifar10 import ResNetCifar10

def get_input_fn(name, batch_size=32):
    is_training = name == 'train'
    ds = df.dataset.Cifar10(name, shuffle=is_training)
    augmentors = [
        tp.imgaug.CenterPaste((40, 40)),
        tp.imgaug.RandomCrop((32, 32)),
        tp.imgaug.Flip(horiz=True),
        #tp.imgaug.MapImage(lambda x: (x - pp_mean)/128.0),
    ]
    if is_training:
        ds = df.RepeatedData(ds, -1)
        ds = tp.AugmentImageComponent(ds, augmentors)
    ds = df.MapData(ds, tuple)  # for tensorflow.data.dataset
    ds.reset_state()

    def input_fn():
        with tf.name_scope('dataset'):
            dataset = tf.data.Dataset.from_generator(
                ds.get_data,
                output_types=(tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([32, 32, 3]), tf.TensorShape([]))
            ).batch(batch_size)
        return dataset
    return input_fn

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    Config(args.config)
    print(Config.get_instance())

    input_functions = {
        'train': get_input_fn('train', Config.get_instance()['mode']['train']['batch_size']),
        'eval': get_input_fn('test', Config.get_instance()['mode']['eval']['batch_size'])
    }

    model_fn = Classifier.get('resnet', ResNetCifar10)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=10,
        session_config=None
    )
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.MomentumOptimizer(
        learning_rate=0.001,
        momentum=0.9
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
