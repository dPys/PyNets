"""Implementation of the statistics modules"""
__author__ = "Derek Pisner"
__email__ = "dpysalexander@gmail.com"

import warnings

import matplotlib
import pickle5 as pickle

pickle.HIGHEST_PROTOCOL = 5

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from pynets.statistics import interfaces, utils
from pynets.statistics.group import benchmarking, prediction
from pynets.statistics.individual import algorithms, multiplex, spectral
