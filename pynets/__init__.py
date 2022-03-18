# -*- coding: utf-8 -*-

"""Top-level package for PyNets."""

__author__ = """Derek Pisner"""
__email__ = "dpisner@utexas.edu"

import warnings
import pickle5 as pickle
pickle.HIGHEST_PROTOCOL = 5

warnings.filterwarnings("ignore")

from .__about__ import (
    __version__,
    __copyright__,
    __credits__,
    __packagename__,
)

__all__ = [
    '__copyright__',
    '__credits__',
    '__packagename__',
    '__version__',
]

