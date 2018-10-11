# -*- coding: utf-8 -*-
r""" test Elapsed class """
import numpy as np
from tracking_via_colorization.utils.image_process import ImageProcess


def test_resize_small_axis():
    images = [np.zeros((480, 640, 3)), np.zeros((640, 480, 3))]
    outputs = ImageProcess.resize(small_axis=256, copy=True)(images)

    assert images[0].shape == (480, 640, 3)
    assert images[1].shape == (640, 480, 3)
    assert outputs[0].shape == (256, 341, 3)
    assert outputs[1].shape == (341, 256, 3)

def test_crop_center():
    images = [np.zeros((480, 640, 3)), np.zeros((640, 480, 3))]
    outputs = ImageProcess.crop((200, 300), copy=True)(images)

    assert images[0].shape == (480, 640, 3)
    assert images[1].shape == (640, 480, 3)
    assert outputs[0].shape == (200, 300, 3)
    assert outputs[1].shape == (200, 300, 3)

def test_resizw_with_crop_center():
    images = [np.zeros((480, 640, 3)), np.zeros((640, 480, 3))]
    outputs = ImageProcess.resize(small_axis=256, copy=True)(images)
    outputs = ImageProcess.crop((256, 256), copy=True)(outputs)

    assert images[0].shape == (480, 640, 3)
    assert images[1].shape == (640, 480, 3)
    assert outputs[0].shape == (256, 256, 3)
    assert outputs[1].shape == (256, 256, 3)
