from __future__ import absolute_import

import copy as cp
import logging
import cv2


LOGGER = logging.getLogger(__name__)

class ImageProcess():
    @staticmethod
    def resize(small_axis=256, copy=False):
        def _resize(images):
            images = cp.deepcopy(images) if copy else images
            for idx, image in enumerate(images):
                height, width = image.shape[:2]
                aspect_ratio = 1.0 * width / height
                width = int(small_axis if aspect_ratio <= 1.0 else (small_axis * aspect_ratio))
                height = int(small_axis if aspect_ratio >= 1.0 else (small_axis / aspect_ratio))
                images[idx] = cv2.resize(image, (width, height))
            return images
        return _resize

    @staticmethod
    def crop(shape, copy=False):
        def _crop(images):
            images = cp.deepcopy(images) if copy else images
            target_height, target_width = shape[:2]
            for idx, image in enumerate(images):
                height, width = image.shape[:2]
                dx = max(0, (width - target_width) // 2)
                dy = max(0, (height - target_height) // 2)
                image = image.reshape((height, width, -1))[dy:dy+target_height, dx:dx+target_width, :]
                images[idx] = image
            return images
        return _crop
