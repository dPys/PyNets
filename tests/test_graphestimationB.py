#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:26:34 2017

@author: PSYC-dap3463
"""
import numpy as np
import nibabel as nib
from pynets import graphestimation
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

def test_extract_ts_rsn_parc():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    net_parcels_map_nifti_file = dir_path + '/whole_brain_cluster_labels_PCA200/997_parcels_Default.nii.gz'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = None
    network='Default'
    ID='997'
    conf = None
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file,'rb')
    coords = pickle.load(file_)
    net_parcels_map_nifti = nib.load(net_parcels_map_nifti_file)
    ts_within_nodes = graphestimation.extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network)
    assert ts_within_nodes is not None

def test_extract_ts_rsn_coords():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = None
    network='Default'
    ID='997'
    conf = None
    node_size = 2
    wb_coords_file = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    file_ = open(wb_coords_file,'rb')
    coords = pickle.load(file_)
    ts_within_nodes = graphestimation.extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network)
    assert ts_within_nodes is not None
