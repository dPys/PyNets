#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import time
import nibabel as nib
import os
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


# fMRI
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

    te.save_and_cleanup()

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

    te.save_and_cleanup()

    print("%s%s%s" % ('extract_ts_coords --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert te.ts_within_nodes is not None
    assert te.node_size is not None
    assert te.smooth is not None
    assert te.dir_path is not None


def test_timeseries_bootstrap():
    from nilearn.masking import apply_mask
    from pynets.registration import reg_utils

    blocklength = 1
    base_dir = str(Path(__file__).parent/"examples")
    func_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    roi_mask_img_RAS = reg_utils.reorient_img(roi, f"{base_dir}/outputs")

    func_img = nib.load(func_file)
    ts_data = apply_mask(func_img, roi_mask_img_RAS)
    block_size = int(int(np.sqrt(ts_data.shape[0])) * blocklength)

    boot_series = fmri_estimation.timeseries_bootstrap(ts_data, block_size)[0]
    assert boot_series.shape == ts_data.shape


def test_fill_confound_nans():
    import pandas as pd

    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/BIDS/sub-0025427/ses-1/func"
    conf = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_desc-confounds_regressors.tsv"
    conf_corr = fmri_estimation.fill_confound_nans(pd.read_csv(conf, sep='\t'), dir_path)
    assert not pd.read_csv(conf_corr, sep='\t').isnull().values.any()


# dMRI
def test_tens_mod_fa_est():
    from dipy.core.gradients import gradient_table
    from dipy.io import save_pickle

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab_file = f"{base_dir}/gtab.pkl"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    save_pickle(gtab_file, gtab)

    [fa_path, _, _, _] = dmri_estimation.tens_mod_fa_est(gtab_file, dwi_file, B0_mask)

    assert os.path.isfile(fa_path)


def test_create_anisopowermap():
    from dipy.core.gradients import gradient_table
    from dipy.io import save_pickle

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab_file = f"{base_dir}/gtab.pkl"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    save_pickle(gtab_file, gtab)

    [anisopwr_path, _, _, _] = dmri_estimation.create_anisopowermap(gtab_file, dwi_file, B0_mask)

    assert os.path.isfile(anisopwr_path)


def test_tens_mod_est():
    from dipy.core.gradients import gradient_table

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    data = nib.load(dwi_file).get_fdata()

    [mod_odf, model] = dmri_estimation.tens_mod_est(gtab, data, B0_mask)

    assert mod_odf is not None
    assert model is not None


def test_csa_mod_est():
    from dipy.core.gradients import gradient_table

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    data = nib.load(dwi_file).get_fdata()

    [csa_mod, model] = dmri_estimation.csa_mod_est(gtab, data, B0_mask)

    assert csa_mod is not None
    assert model is not None


def test_csd_mod_est():
    from dipy.core.gradients import gradient_table

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    data = nib.load(dwi_file).get_fdata()

    [csd_mod, model] = dmri_estimation.csd_mod_est(gtab, data, B0_mask)

    assert csd_mod is not None
    assert model is not None


def test_sfm_mod_est():
    from dipy.core.gradients import gradient_table

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    data = nib.load(dwi_file).get_fdata()

    [sf_odf, model] = dmri_estimation.sfm_mod_est(gtab, data, B0_mask)

    assert sf_odf is not None
    assert model is not None


@pytest.mark.parametrize("fa_wei", [True, False])
def test_streams2graph(fa_wei):
    from dipy.core.gradients import gradient_table
    from dipy.io import save_pickle

    base_dir = str(Path(__file__).parent/"examples")
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    conn_model = 'csd'
    min_length = 10
    error_margin = 6
    directget = 'prob'
    track_type = 'particle'
    target_samples = 1000
    overlap_thr = 1
    min_span_tree = True
    prune = 3
    norm = 6
    binary = False
    dir_path = f"{base_dir}/BIDS/sub-0025427/ses-1/func"
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    network = 'Default'
    ID = '003'
    parc = True
    disp_filt = False
    node_size = None
    dens_thresh = False
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    # Not actually normalized to mni-space in this test.
    atlas_mni = f"{dir_path}/whole_brain_cluster_labels_PCA200_dwi_track.nii.gz"
    streams = f"{base_dir}/miscellaneous/streamlines_est-csd_nodetype-parc_samples-10000streams_tt-local_dg-prob_ml-0.trk"
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab_file = f"{base_dir}/gtab.pkl"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    save_pickle(gtab_file, gtab)
    # Not actually normalized to mni-space in this test.
    warped_fa = dmri_estimation.tens_mod_fa_est(gtab_file, dwi_file, B0_mask)[0]

    conn_matrix = dmri_estimation.streams2graph(atlas_mni, streams, overlap_thr, dir_path, track_type, target_samples,
                                                conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree,
                                                disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary,
                                                directget, warped_fa, error_margin, min_length, fa_wei)[2]
    assert conn_matrix is not None
