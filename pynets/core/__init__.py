"""Implementation of the core workflow modules"""
__author__ = "Derek Pisner"
__email__ = "dpysalexander@gmail.com"

import warnings

import matplotlib
import pickle5 as pickle

pickle.HIGHEST_PROTOCOL = 5

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from . import (
    cloud,
    interfaces,
    nodemaker,
    thresholding,
    utils,
    workflows,
    database,
)
