# -*- coding: utf-8 -*-

__author__ = """Derek Pisner"""
__email__ = "dpisner@utexas.edu"

import matplotlib
import pickle5 as pickle
pickle.HIGHEST_PROTOCOL = 5
from . import cloud, interfaces, nodemaker, thresholding, utils, workflows
import warnings

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
