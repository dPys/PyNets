#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
"""
import pytest
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams['figure.dpi'] = 100
import os
import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.plotting import adjacency
import tempfile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_plot_conn_mat_nonet_no_mask(connectivity_data):
    """
    Test plot_conn_mat_nonet_no_mask functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    signal = 'mean'
    conn_model = 'sps'
    atlas = 'random_parcellation'
    roi = None

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']

    start_time = time.time()
    adjacency.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
                                 ID, subnet, labels, roi, thr, node_size,
                                 smooth, hpass, signal)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()


def test_plot_conn_mat_nonet_mask(connectivity_data):
    """
    Test plot_conn_mat_nonet_mask functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    conn_model = 'sps'
    signal = 'mean'
    atlas = 'random_parcellation'
    roi = None

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']

    start_time = time.time()
    adjacency.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
                                 ID, subnet, labels, roi, thr, node_size,
                                 smooth, hpass, signal)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()
