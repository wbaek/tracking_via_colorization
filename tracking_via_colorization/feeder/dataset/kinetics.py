# -*- coding: utf-8 -*-
# pylint: disable=I1101
import os
import logging
import itertools

import cv2
import numpy as np
import ujson as json


LOGGER = logging.getLogger(__name__)


class Kinetics():
    def __init__(self, base_path, shuffle=False, num_frames=1, skips=(0,)):
        self.base_path = base_path
        self.shuffle = shuffle
        self.num_frames = num_frames
        self.skips = skips
        self.metas = []

        metas = json.load(open(os.path.join(base_path, 'kinetics_train.json')))
        self.keys = sorted(metas.keys())
        for _, key in enumerate(self.keys):
            metas[key]['key'] = key
            self.metas.append(metas[key])
        self.index = list(range(len(self.keys)))

    @property
    def name(self):
        return 'kinetics'

    @property
    def names(self):
        if self.shuffle:
            self.index = np.random.permutation(self.index)
        return [self.metas[idx]['key'] for idx in self.index]

    def reset_state(self):
        pass

    def __len__(self):
        return len(self.index)

    def size(self, name=None):
        if name is None:
            return len(self.metas)
        return self.__len__()

    def get_filename(self, name):
        if name not in self.keys:
            raise KeyError('not exists name at %s' % name)
        LOGGER.debug('[Kinetics.get] %s', name)
        filename = os.path.join(self.base_path, 'processed', name + '.mp4')
        exists = os.path.exists(filename)
        return exists, filename

    def get_data(self, num_frames=None, skips=None):
        return self.__iter__(num_frames, skips)

    def __iter__(self, num_frames=None, skips=None):
        num_frames = num_frames if num_frames is not None else self.num_frames
        skips = skips if skips is not None else self.skips

        for name in self.names:
            exists, filename = self.get_filename(name)
            if not exists:
                continue

            index = -1
            images = []
            capture = cv2.VideoCapture(filename)
            for _, skip in itertools.cycle(enumerate(skips)):
                for _ in range(skip):
                    capture.read()
                ret, image = capture.read()
                if not ret:
                    break
                images.append(image)
                if len(images) == num_frames:
                    index += 1
                    yield [index, images]
                    images = []
