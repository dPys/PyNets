# -*- coding: utf-8 -*-

"""Top-level package for PyNets."""

__author__ = """Derek Pisner"""
__email__ = 'dpisner@utexas.edu'

from .__about__ import (  # noqa
    __version__,
    __copyright__,
    __credits__,
    __packagename__,
)
import warnings
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore', r'cmp not installed')
warnings.filterwarnings('ignore', r'This has not been fully tested. Please report any failures.')
warnings.filterwarnings('ignore', r"can't resolve package from __spec__ or __package__")
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', ResourceWarning)
from .stats import *
from .registration import *
from .dmri import *
from .fmri import *
from .core import *
from .plotting import *
