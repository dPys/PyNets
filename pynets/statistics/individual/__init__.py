# -*- coding: utf-8 -*-

__author__ = """Derek Pisner"""
__email__ = "dpysalexander@gmail.com"

import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from pynets.statistics.individual import algorithms, multiplex, spectral
