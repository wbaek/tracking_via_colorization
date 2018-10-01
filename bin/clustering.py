import os
import sys
import logging

import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorpack.dataflow as df

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='centroids.npy')
    parser.add_argument('-k', '--num-k', type=int, default=16)
    parser.add_argument('-n', '--num-samples', type=int, default=50000)

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)
    logging.info('args: %s', args)

    ds = df.dataset.Cifar10('train', shuffle=False)
    ds = df.MapDataComponent(ds, lambda image: cv2.resize(image, (32, 32)))
    ds = df.MapDataComponent(ds, lambda image: cv2.cvtColor(np.float32(image / 255.0), cv2.COLOR_RGB2Lab))
    ds = df.MapDataComponent(ds, lambda image: image[:, :, 1:])
    ds = df.MapDataComponent(ds, lambda image: image.reshape((-1, 2)))
    ds = df.RepeatedData(ds, -1)
    ds.reset_state()

    generator = ds.get_data()
    logging.info('initalized preprocessor')

    samples = []
    for _ in range(args.num_samples):
        samples.append(next(generator)[0])
    vectors = np.array(samples).reshape((-1, 2))
    logging.info('processed vector: %s', vectors.shape)

    kmeans = KMeans(args.num_k).fit(vectors)
    logging.info('fitted kmean clustering')

    centroids = np.array(kmeans.cluster_centers_)
    logging.info('centroids: %s', centroids)

    with open(args.output, 'wb') as f:
        centroids.dump(f)
    logging.info('dumped at %s', args.output)
