# -*- cording: utf-8 -*-
r""" utils/io
"""
import os
import glob
import logging
import natsort
import cv2
import imageio

LOGGER = logging.getLogger(__name__)


class Reader():
    def __init__(self, path):
        self.path = path

    @staticmethod
    def create(path):
        if os.path.isfile(path):
            return VideoReader(path)
        if os.path.isdir(path):
            return ImageReader(path)
        raise NotImplementedError

    def open(self):
        pass

    def next(self):
        pass

class VideoReader(Reader):
    def open(self):
        self.capure = cv2.VideoCapture(self.path)
        return self

    def next(self):
        if self.capure.isOpened():
            _, image = self.capure.read()
        else:
            image = None
        return image

class ImageReader(Reader):
    def open(self):
        filenames = glob.glob(os.path.join(self.path, '*'))
        self.filenames = natsort.natsorted(filenames, alg=natsort.ns.IGNORECASE)
        self.index = 0
        return self

    def next(self):
        image = None
        while image is None and self.index < len(self.filenames):
            try:
                image = cv2.imread(self.filenames[self.index], cv2.IMREAD_COLOR)
            except Exception as e:
                LOGGER.warning('%s: %s (%s)', type(e), str(e), self.filenames[self.index])
            self.index += 1
        return image

class Writer():
    def __init__(self, filepath):
        self.filepath = filepath

    @staticmethod
    def create(filepath):
        if filepath[-3:] == 'mp4':
            return VideoWriter(filepath)
        if filepath[-3:] == 'gif':
            return GifWriter(filepath)
        LOGGER.error('not support writer file format must be a set xxx.mp4 or xxx.gif')
        raise NotImplementedError

    def write(self, images):
        pass

class ImageWriter(Writer):
    def __init__(self, filepath, extension='jpg'):
        super(ImageWriter, self).__init__(filepath)
        self.extension = extension

    def write(self, images):
        for i, image in enumerate(images):
            cv2.imwrite('%s/%04d.%s'%(self.filepath, i, self.extension), image)

class GifWriter(Writer):
    def write(self, images):
        with imageio.get_writer(self.filepath, mode='I') as writer:
            for image in images:
                writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class VideoWriter(Writer):
    def writer(self, images):
        output_format = 'X264'
        fourcc = cv2.VideoWriter_fourcc(*output_format)
        height, width, _ = images[0].shape
        video_writer = cv2.VideoWriter(output, fourcc, 30, (width, height))
        for image in images:
            video_writer.write(image)
        video_writer.release()
