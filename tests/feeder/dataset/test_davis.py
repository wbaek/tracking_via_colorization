import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest
import cv2
import numpy as np
from tensorpack import dataflow as df

from tracking_via_colorization.feeder.dataset import Davis


def test_davis():
    davis = Davis('/data/public/rw/datasets/videos/davis/trainval')
    assert davis.size() == 60
    assert davis.names[:5] == ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare']
    assert davis.names[-5:] == ['train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking']

def test_davis_shuffle():
    davis = Davis('/data/public/rw/datasets/videos/davis/trainval', shuffle=True)
    assert davis.size() == 60
    assert davis.names[:5] != ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare']
    assert davis.names[-5:] != ['train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking']

def test_davis_generator():
    davis = Davis('/data/public/rw/datasets/videos/davis/trainval')
    generator = davis.get_data()
    idx, images, annotations = next(generator)
    assert idx == 0
    assert len(images) == 1
    assert images[0].shape == (1080, 1920, 3)
    assert annotations[0].shape == (1080, 1920, 3)

    for _ in range(81 - 1):
        next(generator)

    idx, images, annotations = next(generator)
    assert idx == 81
    assert len(images) == 1
    assert images[0].shape == (1080, 1920, 3)
    assert annotations[0].shape == (1080, 1920, 3)

    idx, images, annotations = next(generator)
    assert idx == 0
    assert len(images) == 1
    assert images[0].shape == (1080, 1920, 3)
    assert annotations[0].shape == (1080, 1920, 3)

def test_davis_generator_with_num_frames():
    davis = Davis('/data/public/rw/datasets/videos/davis/trainval')
    generator = davis.get_data(num_frames=4)
    idx, images, annotations = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(1080, 1920, 3)] * 4
    assert [image.shape for image in annotations] == [(1080, 1920, 3)] * 4

    for _ in range(81 // 4 - 2):
        next(generator)

    idx, images, annotations = next(generator)
    assert idx == 19
    assert len(images) == 4
    assert [image.shape for image in images] == [(1080, 1920, 3)] * 4
    assert [image.shape for image in annotations] == [(1080, 1920, 3)] * 4

    idx, images, annotations = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(1080, 1920, 3)] * 4
    assert [image.shape for image in annotations] == [(1080, 1920, 3)] * 4

def test_davis_tensorpack_dataflow():
    ds = Davis('/data/public/rw/datasets/videos/davis/trainval', num_frames=4)

    ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=1)
    ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=2)
    ds = df.MapDataComponent(ds, lambda images: np.stack(images, axis=0), index=1)
    ds = df.MapDataComponent(ds, lambda images: np.stack(images, axis=0), index=2)
    ds = df.BatchData(ds, 6)

    ds.reset_state()
    generator = ds.get_data()
    for _ in range(10):
        _, images, annotations = next(generator)
        assert images.shape == (6, 4, 256, 256, 3)
        assert images.shape == (6, 4, 256, 256, 3)
