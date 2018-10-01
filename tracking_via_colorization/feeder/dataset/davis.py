# -*- coding: utf-8 -*-
import os
import time
import logging

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class Davis():
    def __init__(self, base_path, name='train', resolution='Full-Resolution', shuffle=False, num_frames=1):
        self.base_path = base_path
        self.shuffle = shuffle
        self.num_frames = num_frames

        self.annotation_dir = os.path.join(base_path, 'Annotations', resolution)
        self.image_dir = os.path.join(base_path, 'JPEGImages', resolution)
        self._names = [line.strip() for line in open(os.path.join(base_path, 'ImageSets', '2017', name + '.txt')).readlines() if line.strip()]
        self.index = list(range(len(self._names)))

    @property
    def name(self):
        return 'davis'

    @property
    def names(self):
        if self.shuffle:
            self.index = np.random.permutation(self.index)
        return [self._names[idx] for idx in self.index]

    def reset_state(self):
        np.random.seed(int(time.time() * 1e7) % 2**32)

    def __len__(self):
        return len(self.index)

    def size(self):
        return self.__len__()

    def get_data(self, num_frames=None):
        return self.__iter__(num_frames)

    def __iter__(self, num_frames=None):
        num_frames = num_frames if num_frames is not None else self.num_frames

        for name in self.names:

            index = -1
            images = []
            annotations = []
            for image, annotation in zip(self._images(name), self._annotations(name)):
                if len(images) == num_frames:
                    images = []
                    annotations = []
                images.append(cv2.imread(image))
                annotations.append(cv2.imread(annotation))
                if len(images) == num_frames:
                    index += 1
                    yield [index, images, annotations]

    def _images(self, name):
        image_dir = os.path.join(self.image_dir, name)
        images = os.listdir(image_dir)
        images.sort()
        return [os.path.join(image_dir, image) for image in images]

    def _annotations(self, name):
        annotation_dir = os.path.join(self.annotation_dir, name)
        annotations = os.listdir(annotation_dir)
        annotations.sort()
        return [os.path.join(annotation_dir, image) for image in annotations]
