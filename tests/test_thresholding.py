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


def test_thresh_func():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/ryanhammonds/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    dens_thresh = False
    thr = 0.95
    smooth = 2
    conn_matrix=np.random.rand(3,3)
    conn_model = 'cov'
    network = 'Default'
    min_span_tree = False
    ID = '997'
    disp_filt = False 
    mask = None
    parc = False
    node_size = 'TEST'

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _, _] = thresholding.thresh_func(dens_thresh, thr, conn_matrix,
                                                                                       conn_model, network, ID, dir_path,
                                                                                       mask, node_size, min_span_tree, smooth, disp_filt, parc)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix_thr is not None
    assert edge_threshold is not None
    assert est_path is not None

def test_thresh_diff():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/ryanhammonds/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    dens_thresh = False
    thr = 0.95
    conn_matrix=np.random.rand(3,3)
    conn_model = 'cov'
    network = 'Default'
    min_span_tree = False
    ID = '997'
    mask = None
    node_size = 'parc'
    parc = True
    disp_filt = False

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _] = thresholding.thresh_diff(dens_thresh, thr, conn_model, network,
                                                                                          ID, dir_path, mask, node_size, conn_matrix,
                                                                                          parc, min_span_tree, disp_filt)
    print("%s%s%s" % ('thresh_and_fit (Functional, density thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix_thr is not None
    assert est_path is not None
    assert edge_threshold is not None
