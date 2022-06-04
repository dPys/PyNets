"""Implementation of the statistics modules"""
__author__ = "Derek Pisner"
__email__ = "dpysalexander@gmail.com"

import warnings

import matplotlib
import pickle5 as pickle

pickle.HIGHEST_PROTOCOL = 5

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from . import interfaces, utils
from .group import benchmarking, prediction
from .individual import multiplex, algorithms, spectral
