# pylint: disable=I1101
import os
import sys
import time
import copy

import cv2
import numpy as np
import tensorflow as tf
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from tracking_via_colorization.utils.devices import Devices
from tracking_via_colorization.config import Config
from tracking_via_colorization.feeder.dataset import Kinetics, Davis
from tracking_via_colorization.networks.colorizer import Colorizer
from tracking_via_colorization.networks.resnet_colorizer import ResNetColorizer

def get_resize(small_axis=256):
    def _resize(images):
        for image in images:
            height, width = image.shape[:2]
            aspect_ratio = 1.0 * width / height
            width = int(small_axis if aspect_ratio <= 1.0 else (small_axis * aspect_ratio))
            height = int(small_axis if aspect_ratio >= 1.0 else (small_axis / aspect_ratio))
            cv2.resize(image, (width, height))
        return images
    return _resize

def dataflow(name='davis', scale=1):
    if name == 'davis':
        ds = Davis('/data/public/rw/datasets/videos/davis/trainval', num_frames=1, shuffle=False)
    elif name == 'kinetics':
        ds = Kinetics('/data/public/rw/datasets/videos/kinetics', num_frames=1, skips=[0], shuffle=False)
    else:
        raise Exception('not support dataset %s' % name)

    if name != 'davis':
        ds = df.MapData(ds, lambda dp: [dp[0], dp[1], dp[1]])

    ds = df.MapData(ds, lambda dp: [
        dp[0], # index
        dp[1], # original
        dp[2], # mask
    ])
    size = (256 * scale, 256 * scale)

    ds = df.MapDataComponent(ds, get_resize(256 * scale), index=1)
    ds = df.MapDataComponent(ds, lambda images: cv2.resize(images[0], size), index=2)

    ds = df.MapData(ds, lambda dp: [
        dp[0], # index
        dp[1][0], # original
        cv2.cvtColor(cv2.resize(dp[1][0], size), cv2.COLOR_BGR2GRAY).reshape((size[0], size[1], 1)), # gray
        dp[2], # mask
    ])
    ds = df.MultiProcessPrefetchData(ds, nr_prefetch=32, nr_proc=1)
    return ds

def main(args):
    Config(args.config)
    device_info = Devices.get_devices(gpu_ids=args.gpus)
    tf.logging.info('\nargs: %s\nconfig: %s\ndevice info: %s', args, Config.get_instance(), device_info)

    scale = args.scale
    ds = dataflow(args.name, scale)
    ds.reset_state()

    num_inputs = args.num_reference + 1
    placeholders = {
        'features': tf.placeholder(tf.float32, (None, num_inputs, 256 * scale, 256 * scale, 1), 'features'),
        'labels': tf.placeholder(tf.int64, (None, num_inputs, 32 * scale, 32 * scale, 1), 'labels'),
    }
    hparams = Config.get_instance()['hparams']
    hparams['optimizer'] = tf.train.AdamOptimizer()
    hparams = tf.contrib.training.HParams(**hparams)

    estimator_spec = Colorizer.get('resnet', ResNetColorizer, num_reference=args.num_reference, predict_direction=args.direction)(
        features=placeholders['features'],
        labels=placeholders['labels'],
        mode=tf.estimator.ModeKeys.PREDICT,
        params=hparams
    )

    session = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(session, args.checkpoint)

    dummy_labels = np.zeros((1, num_inputs, 32 * scale, 32 * scale, 1), dtype=np.int64)

    video_index = -1
    start_time = time.time()
    num_images = 0
    for frame, image, gray, color in ds.get_data():
        curr = {'image': image, 'gray': gray, 'color': color}
        num_images += 1

        if frame == 0:
            tf.logging.info('avg elapsed time per image: %.3fsec', (time.time() - start_time) / num_images)
            start_time = time.time()
            num_images = 0
            video_index += 1
            dummy_features = [np.zeros((256 * scale, 256 * scale, 1), dtype=np.float32) for _ in range(num_inputs)]
            dummy_references = [np.zeros((256 * scale, 256 * scale, 3), dtype=np.uint8) for _ in range(args.num_reference)]
            prev = copy.deepcopy(curr)
            dummy_features = dummy_features[1:] + [prev['gray']]
            tf.logging.info('video index: %04d', video_index)

        if frame <= args.num_reference:
            dummy_features = dummy_features[1:] + [curr['gray']]
            dummy_references = dummy_references[1:] + [curr['color']]

        features = np.expand_dims(np.stack(dummy_features[1:] + [curr['gray']], axis=0), axis=0)
        predictions = session.run(estimator_spec.predictions, feed_dict={
            placeholders['features']: features,
            placeholders['labels']: dummy_labels,
        })

        matrix_size = 32 * 32 * scale * scale
        indicies = np.argmax(predictions['similarity'], axis=-1).reshape((-1,))
        mapping = np.zeros((matrix_size, 2))
        for i, index in enumerate(indicies):
            f = (index // (matrix_size)) % args.num_reference
            y = index // (32 * scale)
            x = index % (32 * scale)
            mapping[i, :] = [x, (args.num_reference - f - 1) * (32 * scale) + y]
        mapping = np.array(mapping, dtype=np.float32).reshape((32 * scale, 32 * scale, 2))

        height, width = mapping.shape[:2]
        reference_colors = np.concatenate(dummy_references, axis=0)

        predicted = cv2.remap(cv2.resize(reference_colors, (width, height * args.num_reference)), mapping, None, cv2.INTER_LINEAR)

        predicted = cv2.resize(predicted, (256 * scale, 256 * scale))
        # curr['color'] = np.copy(predicted)

        height, width = image.shape[:2]
        predicted = cv2.resize(predicted, (width, height))
        prev = copy.deepcopy(curr)

        if args.name == 'davis':
            _, mask = cv2.threshold(cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            predicted = cv2.add(cv2.bitwise_and(image, image, mask=mask_inv), predicted)
            predicted = cv2.addWeighted(image, 0.3, predicted, 0.7, 0)

        stacked = np.concatenate([image, predicted], axis=1)
        similarity = (np.copy(predictions['similarity']).reshape((32 * 32 * scale * scale * args.num_reference, -1)) * 255.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale))
        similarity = cv2.resize(cv2.dilate(similarity, kernel), (32 * scale * 2 * args.num_reference, 32 * scale * 2))

        output_dir = '%s/%04d' % (args.output, video_index)
        for name, result in [('image', stacked), ('similarity', similarity)]:
            folder = os.path.join(output_dir, name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite('%s/%04d.jpg' % (folder, frame), result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default=None)

    parser.add_argument('-s', '--scale', type=int, default=1)
    parser.add_argument('-n', '--num-reference', type=int, default=1)
    parser.add_argument('-d', '--direction', type=str, default='backward', help='[forward|backward] backward is default')

    parser.add_argument('--name', type=str, default='davis')
    parser.add_argument('-o', '--output', type=str, default='results')

    parsed_args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    tf.logging.set_verbosity(tf.logging.INFO)

    main(parsed_args)
