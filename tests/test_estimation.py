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
from pynets import estimation
from pathlib import Path


def test_get_conn_matrix_cov():
    base_dir = str(Path(__file__).parent/"examples")
    #ase_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    time_series_file = dir_path + '/coords_power_2011/997_wb_net_ts.txt'
    time_series = np.genfromtxt(time_series_file)
    conn_model = 'cov'


    node_size = 2
    smooth = 2
    c_boot = 0
    dens_thresh = False
    network = 'Default'
    ID = '997'
    roi = None
    min_span_tree = False
    disp_filt = False
    parc = None
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
    [conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network,
    ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select,
    label_names, coords, c_boot, norm, binary] = estimation.get_conn_matrix(time_series, conn_model,
    dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
    disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, c_boot, norm, binary)
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
    assert atlas_select is not None
    #assert uatlas_select is not None
    #assert label_names is not None
    assert coords is not None

def test_extract_ts_rsn_parc():
    # Set example inputs

    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_parcels_Default.nii.gz'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    roi = None
    network = 'Default'
    ID = '997'
    smooth = 2
    c_boot = 0
    boot_size = 3
    conf = None
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    uatlas_select = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)
    mask = None

    start_time = time.time()
    net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
    [ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select,
    label_names, coords, c_boot] = estimation.extract_ts_parc(net_parcels_map_nifti,
    conf, func_file, coords, roi, dir_path, ID, network, smooth, atlas_select,
    uatlas_select, label_names, c_boot, boot_size, mask)
    print("%s%s%s" % ('extract_ts_parc --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None
    #assert node_size is not None
    #node_size is none

def test_extract_ts_rsn_coords():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'

    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    roi = None
    network = 'Default'
    ID = '997'
    conf = None
    node_size = 2
    smooth = 2
    c_boot = 0
    boot_size = 3
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    uatlas_select = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)
    mask = None
    start_time = time.time()
    [ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select,
     label_names, coords, c_boot] = estimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, roi,
                                                                 network, smooth, atlas_select, uatlas_select,
                                                                 label_names, c_boot, boot_size, mask)
    print("%s%s%s" % ('extract_ts_coords --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None
    assert node_size is not None
    assert smooth is not None
    assert dir_path is not None
    assert c_boot is not None

# def test_extract_ts_rsn_parc_fast():
#     # Set example inputs
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_parcels_Default.nii.gz'
#     func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
#     roi = None
#     network = 'Default'
#     ID = '997'
#     conf = None
#     wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     file_ = open(wb_coords_file, 'rb')
#     coords = pickle.load(file_)
#
#     start_time = time.time()
#     net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
#     ts_within_nodes = estimation.extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, roi, dir_path,
#                                                       ID, network, fast=True)
#     print("%s%s%s" % ('extract_ts_parc (fast) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
#     assert ts_within_nodes is not None

# def test_extract_ts_rsn_coords_fast():
#     # Set example inputs
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
#     roi = None
#     network = 'Default'
#     ID = '997'
#     conf = None
#     node_size = 2
#     wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     file_ = open(wb_coords_file, 'rb')
#     coords = pickle.load(file_)
#
#     start_time = time.time()
#     ts_within_nodes = estimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, roi, network,
#                                                         fast=True)
#     print("%s%s%s" % ('extract_ts_coords (fast) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
#     assert ts_within_nodes is not None
