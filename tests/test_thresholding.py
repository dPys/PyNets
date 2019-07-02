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


def test_binarize():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    s = thresholding.binarize(thresholding.threshold_proportional(x, .41))
    assert np.sum(s) == 2.0


def test_normalize():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    s = thresholding.normalize(thresholding.threshold_proportional(x, .79))
    assert np.max(s) <= 1
    assert np.min(s) >= 0


def test_threshold_absolute():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    s = thresholding.threshold_absolute(x, 0.1)
    assert np.round(np.sum(s), 1) <= np.sum(x)


def test_invert():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    s = thresholding.invert(thresholding.threshold_proportional(x, .9))
    assert np.round(np.sum(s), 1) >= np.sum(x)


def test_autofix():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    x[1][1] = np.inf
    x[2][1] = np.nan
    s = thresholding.autofix(x)
    assert (np.nan not in s) and (np.inf not in s)


def test_density_thresholding():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.genfromtxt(base_dir + '/002/fmri/whole_brain_cluster_labels_PCA200/002_est_sps_raw_mat.txt')
    l = thresholding.est_density((thresholding.density_thresholding(x, 0.01)))
    h = thresholding.est_density((thresholding.density_thresholding(x, 0.04)))
    assert np.equal(l, 0.009748743718592965)
    assert np.equal(h, 0.037487437185929645)


def test_est_density():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.genfromtxt(base_dir + '/002/fmri/whole_brain_cluster_labels_PCA200/002_est_sps_raw_mat.txt')
    d = thresholding.est_density(x)
    assert np.round(d, 1) == 0.1


def test_thr2prob():
    base_dir = str(Path(__file__).parent/"examples")
    x = np.load(base_dir + '/002/fmri/002_Default_est_cov_raw_mat.npy')
    s = thresholding.normalize(x)
    s[0][0] = 0.0000001
    t = thresholding.thr2prob(s)
    assert float(len(t[np.logical_and(t < 0.001, t>0)])) == float(0.0)


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
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _] = thresholding.thresh_func(dens_thresh, thr, conn_matrix, conn_model,
    network, ID, dir_path, roi, node_size, min_span_tree, smooth, disp_filt,
    parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass)
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
#     atlas = 'whole_brain_cluster_labels_PCA200'
#     uatlas = None
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
#     labels_file = open(labels_file_path, 'rb')
#     labels = pickle.load(labels_file)
#     coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     coord_file = open(coord_file_path, 'rb')
#     coords = pickle.load(coord_file)
#
#     start_time = time.time()
#     [conn_matrix_thr, edge_threshold, est_path, _, _, _, _, _, _, _, _, _] = thresholding.thresh_diff(dens_thresh, thr, conn_model, network, ID, dir_path,
#     roi, node_size, conn_matrix, parc, min_span_tree, disp_filt, atlas,
#     uatlas, labels, coords)
#     print("%s%s%s" %
#     ('thresh_and_fit (Functional, density thresholding) --> finished: ',
#     str(np.round(time.time() - start_time, 1)), 's'))
#
#     assert conn_matrix_thr is not None
#     assert est_path is not None
#     assert edge_threshold is not None
