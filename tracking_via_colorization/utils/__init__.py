# -*- coding: utf-8 -*-
r"""deepconv_tracker/utils
"""
from __future__ import absolute_import

import warnings
warnings.filterwarnings("ignore")

from .elapsed import Elapsed
from .utils import grid_images
from .io import Reader, Writer
from .devices import Devices
from .multiple import build_learning_rate, average_gradients
