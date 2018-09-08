# -*- coding: utf-8 -*-
r""" test Elapsed class """
import time
import pytest
from tracking_via_colorization.utils import Elapsed


def test_elapsed():
    elapsed = Elapsed()

    time.sleep(1.0)
    elapsed.tic('one')

    time.sleep(0.5)
    elapsed.tic('two')

    time.sleep(0.1)
    elapsed.tic('three')

    report = elapsed.calc()

    assert report['one'] == pytest.approx(1.0, abs=1e-2)
    assert report['two'] == pytest.approx(0.5, abs=1e-2)
    assert report['three'] == pytest.approx(0.1, abs=1e-2)
    assert report['total'] == pytest.approx(1.6, abs=1e-2)

    assert str(elapsed) == 'total:{total:.3f} one:{one:.3f} two:{two:.3f} three:{three:.3f}'.format(**report)
