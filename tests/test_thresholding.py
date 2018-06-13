#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@author: PSYC-dap3463

"""
import numpy as np
import time
from pathlib import Path
from pynets import thresholding

def test_thresh_and_fit1():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = 'Default'
    time_series_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_wb_net_ts.txt'
    ts_within_nodes = np.genfromtxt(time_series_file)
    conn_model = 'cov'
    thr = 0.95
    dens_thresh = False
    ID = '997'
    mask = None
    node_size = 'TEST'

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _] = thresholding.thresh_and_fit(dens_thresh, thr,
                                                                                       ts_within_nodes, conn_model,
                                                                                       network, ID, dir_path, mask,
                                                                                       node_size)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix_thr is not None
    assert est_path is not None
    assert edge_threshold is not None

def test_thresh_and_fit2():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = 'Default'
    time_series_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_wb_net_ts.txt'
    ts_within_nodes = np.genfromtxt(time_series_file)
    conn_model = 'cov'
    thr = 0.10
    dens_thresh = True
    ID = '997'
    mask = None
    node_size = 'TEST'

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _] = thresholding.thresh_and_fit(dens_thresh, thr,
                                                                                       ts_within_nodes, conn_model,
                                                                                       network, ID, dir_path, mask,
                                                                                       node_size)
    print("%s%s%s" % ('thresh_and_fit (Functional, density thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix_thr is not None
    assert est_path is not None
    assert edge_threshold is not None
