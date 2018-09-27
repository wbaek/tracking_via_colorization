import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest
import cv2
import numpy as np
from tensorpack import dataflow as df

from tracking_via_colorization.feeder.dataset import Kinetics


def test_kinetics():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics')
    assert kinetics.size() == 391782
    assert kinetics.names[:5] == ['---0dWlqevI', '---QUuC4vJs', '---aQ-tA5_A', '---j12rm3WI', '--0NTAs-fA0']
    assert kinetics.names[-5:] == ['zzyHRXPQP2I', 'zzyxMjfYpL0', 'zzz0-zDYts8', 'zzzZycxdZHk', 'zzz_3yWpTXo']

def test_kinetics_shuffle():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics', shuffle=True)
    assert kinetics.size() == 391782
    assert kinetics.names[:5] != ['---0dWlqevI', '---QUuC4vJs', '---aQ-tA5_A', '---j12rm3WI', '--0NTAs-fA0']
    assert kinetics.names[-5:] != ['zzyHRXPQP2I', 'zzyxMjfYpL0', 'zzz0-zDYts8', 'zzzZycxdZHk', 'zzz_3yWpTXo']

def test_kinetics_get():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics')
    assert kinetics.get_filename(kinetics.names[0]) == (True, '/data/public/rw/datasets/videos/kinetics/processed/---0dWlqevI.mp4')
    assert kinetics.get_filename(kinetics.names[-1]) == (False, '/data/public/rw/datasets/videos/kinetics/processed/zzz_3yWpTXo.mp4')
    with pytest.raises(KeyError) as exception_info:
        kinetics.get_filename('NOT_EXISTS_NAME')
    assert exception_info.value.args[0] == 'not exists name at NOT_EXISTS_NAME'

def test_kinetics_generator():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics')
    generator = kinetics.get_data()
    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 1
    assert images[0].shape == (720, 406, 3)

    for _ in range(299 - 1):
        next(generator)

    idx, images = next(generator)
    assert idx == 299
    assert len(images) == 1
    assert images[0].shape == (720, 406, 3)

    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 1
    assert images[0].shape == (240, 320, 3)

def test_kinetics_generator_with_num_frames():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics')
    generator = kinetics.get_data(num_frames=4)
    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(720, 406, 3)] * 4

    for _ in range(299 // 4 - 1):
        next(generator)

    idx, images = next(generator)
    assert idx == 74
    assert len(images) == 4
    assert [image.shape for image in images] == [(720, 406, 3)] * 4

    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(240, 320, 3)] * 4

def test_kinetics_generator_with_num_frames_skips():
    kinetics = Kinetics('/data/public/rw/datasets/videos/kinetics')
    generator = kinetics.get_data(num_frames=4, skips=[0, 4, 4, 8])
    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(720, 406, 3)] * 4

    for _ in range(299 // (1 + 5 + 5 + 9) - 1):
        next(generator)

    idx, images = next(generator)
    assert idx == 14
    assert len(images) == 4
    assert [image.shape for image in images] == [(720, 406, 3)] * 4

    idx, images = next(generator)
    assert idx == 0
    assert len(images) == 4
    assert [image.shape for image in images] == [(240, 320, 3)] * 4

def test_kinetics_tensorpack_dataflow():
    ds = Kinetics('/data/public/rw/datasets/videos/kinetics', num_frames=4, skips=[0, 4, 4, 8])

    ds = df.MapDataComponent(ds, lambda images: [cv2.resize(image, (256, 256)) for image in images], index=1)
    ds = df.MapDataComponent(ds, lambda images: np.stack(images, axis=0), index=1)
    ds = df.BatchData(ds, 6)

    ds.reset_state()
    generator = ds.get_data()
    for _ in range(10):
        _, images = next(generator)
        assert images.shape == (6, 4, 256, 256, 3)
