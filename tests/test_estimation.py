#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import nibabel as nib
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import estimation as fmri_estimation
from pynets.dmri import estimation as dmri_estimation
from pathlib import Path


def test_get_conn_matrix_cov():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    time_series_file = dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_rsn_net_ts.npy'
    time_series = np.load(time_series_file)
    conn_model = 'cov'
    node_size = 2
    smooth = 2
    c_boot = 0
    dens_thresh = False
    network = 'Default'
    ID = '002'
    roi = None
    min_span_tree = False
    disp_filt = False
    hpass = None
    parc = None
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
    [conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network,
    ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas,
    labels, coords, c_boot, norm, binary, hpass] = fmri_estimation.get_conn_matrix(time_series, conn_model,
    dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
    disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass)
    print("%s%s%s" %
    ('get_conn_matrix --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix is not None
    assert conn_model is not None
    assert dir_path is not None
    assert node_size is not None
    assert smooth is not None
    assert c_boot is not None
    assert dens_thresh is not None
    assert network is not None
    assert ID is not None
    #assert roi is not None
    assert min_span_tree is not None
    assert disp_filt is not None
    #assert parc is not None
    assert prune is not None
    assert atlas is not None
    #assert uatlas is not None
    #assert labels is not None
    assert coords is not None


def test_extract_ts_rsn_parc():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/002_parcels_Default.nii.gz'
    func_file = dir_path + '/002.nii.gz'
    roi = None
    network = 'Default'
    ID = '002'
    smooth = 2
    c_boot = 0
    boot_size = 3
    conf = None
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    mask = None
    hpass = None
    start_time = time.time()
    net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
    [ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas,
    labels, coords, c_boot, hpass] = fmri_estimation.extract_ts_parc(net_parcels_map_nifti,
    conf, func_file, coords, roi, dir_path, ID, network, smooth, atlas,
    uatlas, labels, c_boot, boot_size, hpass)
    print("%s%s%s" % ('extract_ts_parc --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None
    #assert node_size is not None
    #node_size is none


def test_extract_ts_rsn_coords():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    roi = None
    network = 'Default'
    ID = '002'
    conf = None
    node_size = 2
    smooth = 2
    c_boot = 0
    boot_size = 3
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    hpass = None
    start_time = time.time()
    [ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas,
     labels, coords, c_boot, hpass] = fmri_estimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path,
                                                                        ID, roi, network, smooth, atlas, uatlas,
                                                                        labels, c_boot, boot_size, hpass)
    print("%s%s%s" % ('extract_ts_coords --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None
    assert node_size is not None
    assert smooth is not None
    assert dir_path is not None
    assert c_boot is not None
