#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets import thresholding


def test_thresh_func():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    dens_thresh = False
    thr = 0.95
    smooth = 2
    c_boot = 3
    conn_matrix=np.random.rand(3,3)
    conn_model = 'cov'
    network = 'Default'
    min_span_tree = False
    ID = '002'
    disp_filt = False
    roi = None
    parc = False
    node_size = 'TEST'
    hpass = 0.10
    prune = 1
    norm = 1
    binary = False
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    uatlas_select = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _] = thresholding.thresh_func(dens_thresh, thr, conn_matrix, conn_model,
    network, ID, dir_path, roi, node_size, min_span_tree, smooth, disp_filt,
    parc, prune, atlas_select, uatlas_select, label_names, coords, c_boot, norm, binary, hpass)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))

    assert conn_matrix_thr is not None
    assert edge_threshold is not None
    assert est_path is not None


# def test_thresh_diff():
#     # Set example inputs
#     base_dir = str(Path(__file__).parent/"examples")
#
#     dir_path = base_dir + '/002/fmri'
#     dens_thresh = False
#     thr = 0.95
#     conn_matrix=np.random.rand(3,3)
#     conn_model = 'cov'
#     network = 'Default'
#     min_span_tree = False
#     ID = '997'
#     roi = None
#     node_size = 'parc'
#     parc = True
#     disp_filt = False
#     atlas_select = 'whole_brain_cluster_labels_PCA200'
#     uatlas_select = None
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
#     labels_file = open(labels_file_path, 'rb')
#     label_names = pickle.load(labels_file)
#     coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     coord_file = open(coord_file_path, 'rb')
#     coords = pickle.load(coord_file)
#
#     start_time = time.time()
#     [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _, _, _, _, _] = thresholding.thresh_diff(dens_thresh, thr, conn_model, network, ID, dir_path,
#     roi, node_size, conn_matrix, parc, min_span_tree, disp_filt, atlas_select,
#     uatlas_select, label_names, coords)
#     print("%s%s%s" %
#     ('thresh_and_fit (Functional, density thresholding) --> finished: ',
#     str(np.round(time.time() - start_time, 1)), 's'))
#
#     assert conn_matrix_thr is not None
#     assert est_path is not None
#     assert edge_threshold is not None
