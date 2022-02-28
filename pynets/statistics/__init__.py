# -*- coding: utf-8 -*-

__author__ = """Derek Pisner"""
__email__ = "dpisner@utexas.edu"

import matplotlib
import warnings
from . import interfaces, utils
from .group import benchmarking, prediction
from .individual import multiplex, algorithms, spectral

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
