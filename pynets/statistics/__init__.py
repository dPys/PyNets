# -*- coding: utf-8 -*-

__author__ = """Derek Pisner"""
__email__ = "dpisner@utexas.edu"

import matplotlib
from . import interfaces, utils
from .group import benchmarking, prediction
from .individual import multiplex, algorithms, spectral
import warnings

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
