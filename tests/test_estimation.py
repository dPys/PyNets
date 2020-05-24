#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import estimation as fmri_estimation
from pynets.dmri import estimation as dmri_estimation
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("conn_model", ['corr', 'sps', 'cov', 'partcorr'])
def test_get_conn_matrix_cov(conn_model):
    """
    Test for get_conn_matrix functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/BIDS/sub-0025427/ses-1/func"
    time_series_file = f"{base_dir}/miscellaneous/002_rsn-Default_net_ts.npy"
    time_series = np.load(time_series_file)
    node_size = 2
    smooth = 2
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
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    [conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network,
    ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas,
    labels, coords, norm, binary, hpass] = fmri_estimation.get_conn_matrix(time_series, conn_model,
    dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
    disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, hpass)
    print("%s%s%s" %
    ('get_conn_matrix --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert conn_matrix is not None
    assert conn_model is not None
    assert dir_path is not None
    assert node_size is not None
    assert smooth is not None
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
    """
    Test for extract_ts_parc functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    net_parcels_map_nifti_file = f"{base_dir}/miscellaneous/002_parcels_Default.nii.gz"
    dir_path = f"{base_dir}/BIDS/sub-0025427/ses-1/func"
    func_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    roi = None
    network = 'Default'
    ID = '002'
    smooth = 2
    conf = None
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    mask = None
    node_size = None
    hpass = None
    start_time = time.time()

    te = fmri_estimation.TimeseriesExtraction(net_parcels_nii_path=net_parcels_map_nifti_file, node_size=node_size,
                                              conf=conf, func_file=func_file, coords=coords, roi=roi, dir_path=dir_path,
                                              ID=ID, network=network, smooth=smooth, atlas=atlas, uatlas=uatlas,
                                              labels=labels, hpass=hpass, mask=mask)

    te.prepare_inputs()

    te.extract_ts_parc()
        
    print("%s%s%s" % ('extract_ts_parc --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert te.ts_within_nodes is not None
    #assert node_size is not None
    #node_size is none


@pytest.mark.parametrize("node_size", [pytest.param(0, marks=pytest.mark.xfail), '2', '8'])
@pytest.mark.parametrize("smooth", ['0', '2'])
def test_extract_ts_rsn_coords(node_size, smooth):
    """
    Test for extract_ts_coords functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/BIDS/sub-0025427/ses-1/func"
    func_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    roi = None
    network = 'Default'
    ID = '002'
    conf = None
    node_size = 2
    smooth = 2
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    mask = None
    hpass = None
    start_time = time.time()
    te = fmri_estimation.TimeseriesExtraction(net_parcels_nii_path=None, node_size=node_size,
                                              conf=conf, func_file=func_file, coords=coords, roi=roi, dir_path=dir_path,
                                              ID=ID, network=network, smooth=smooth, atlas=atlas, uatlas=uatlas,
                                              labels=labels, hpass=hpass,
                                              mask=mask)

    te.prepare_inputs()

    te.extract_ts_coords()

    print("%s%s%s" % ('extract_ts_coords --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert te.ts_within_nodes is not None
    assert te.node_size is not None
    assert te.smooth is not None
    assert te.dir_path is not None
