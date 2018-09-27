# -*- coding: utf-8 -*-
import os
import logging
import itertools

import ujson as json
import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class Kinetics():
    def __init__(self, base_path, shuffle=False):
        self.base_path = base_path
        self.metas = []

        metas = json.load(open(os.path.join(base_path, 'kinetics_train.json')))
        self.keys = sorted(metas.keys())
        for i, key in enumerate(self.keys):
            #if not os.path.exists(os.path.join(base_path, 'processed', key + '.mp4')):
            #    continue
            metas[key]['key'] = key
            self.metas.append(metas[key])
        self.index = list(range(len(self.keys)))
        if shuffle:
            self.index = np.random.permutation(self.index)

    @property
    def name(self):
        return 'kinetics'

    @property
    def names(self):
        return [self.metas[idx]['key'] for idx in self.index]

    def size(self, name=None):
        if name is None:
            return len(self.metas)
        raise NotImplemented

    def get_filename(self, name):
        if name not in self.keys:
            raise KeyError('not exists name at %s', name)
        LOGGER.debug('[Kinetics.get] %s', name)
        filename = os.path.join(self.base_path, 'processed', name + '.mp4')
        exists = os.path.exists(filename)
        return exists, filename

    def generator(self, num_frames=1, skips=[0]):
        for name in self.names:
            exists, filename = self.get_filename(name)
            if not exists:
                continue

            index = -1
            images = []
            capture = cv2.VideoCapture(filename)
            for i, skip in itertools.cycle(enumerate(skips)):
                for _ in range(skip):
                    capture.read()
                ret, image = capture.read()
                if not ret:
                    break
                images.append(image)
                if len(images) == num_frames:
                    index += 1
                    yield index, images
                    images = []

