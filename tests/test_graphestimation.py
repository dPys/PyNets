#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:26:34 2017

@author: PSYC-dap3463
"""
import numpy as np
import nibabel as nib
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets import graphestimation
from pathlib import Path

def test_get_conn_matrix_cov():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    time_series_file = dir_path + '/coords_power_2011/997_wb_net_ts.txt'
    conn_model = 'cov'
    time_series = np.genfromtxt(time_series_file)

    start_time = time.time()
    conn_matrix = graphestimation.get_conn_matrix(time_series, conn_model)
    print("%s%s%s" % ('get_conn_matrix --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert conn_matrix is not None

def test_extract_ts_rsn_parc():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_parcels_Default.nii.gz'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = None
    network = 'Default'
    ID = '997'
    conf = None
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)

    start_time = time.time()
    net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
    ts_within_nodes = graphestimation.extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path,
                                                      ID, network)
    print("%s%s%s" % ('extract_ts_parc --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None

def test_extract_ts_rsn_coords():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = None
    network = 'Default'
    ID = '997'
    conf = None
    node_size = 2
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file, 'rb')
    coords = pickle.load(file_)

    start_time = time.time()
    ts_within_nodes = graphestimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network)
    print("%s%s%s" % ('extract_ts_coords --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert ts_within_nodes is not None

# def test_extract_ts_rsn_parc_fast():
#     # Set example inputs
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_parcels_Default.nii.gz'
#     func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
#     mask = None
#     network = 'Default'
#     ID = '997'
#     conf = None
#     wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     file_ = open(wb_coords_file, 'rb')
#     coords = pickle.load(file_)
#
#     start_time = time.time()
#     net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
#     ts_within_nodes = graphestimation.extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path,
#                                                       ID, network, fast=True)
#     print("%s%s%s" % ('extract_ts_parc (fast) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
#     assert ts_within_nodes is not None

# def test_extract_ts_rsn_coords_fast():
#     # Set example inputs
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
#     mask = None
#     network = 'Default'
#     ID = '997'
#     conf = None
#     node_size = 2
#     wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     file_ = open(wb_coords_file, 'rb')
#     coords = pickle.load(file_)
#
#     start_time = time.time()
#     ts_within_nodes = graphestimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network,
#                                                         fast=True)
#     print("%s%s%s" % ('extract_ts_coords (fast) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
#     assert ts_within_nodes is not None
