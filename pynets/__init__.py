# -*- coding: utf-8 -*-

"""Top-level package for PyNets."""

__author__ = """Derek Pisner"""
__email__ = 'dpisner@utexas.edu'

from .__about__ import (
    __version__,
    __copyright__,
    __credits__,
    __packagename__,
)

import warnings
from .stats import *
from .registration import *
from .dmri import *
from .fmri import *
from .core import *
from .plotting import *
warnings.filterwarnings("ignore")
