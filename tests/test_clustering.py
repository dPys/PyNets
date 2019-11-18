#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import os
import numpy as np
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import clustools


def test_indx_1dto3d():
    """
    Test for indx_1dto3d functionality
    """
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_indx_3dto1d():
    """
    Test for indx_3dto1d functionality
    """
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_make_local_connectivity_tcorr():
    """
    Test for make_local_connectivity_tcorr functionality
    """
    print("testing make_local_connectivity_tcorr")
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    mask_file = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    image_file = dir_path + '/002.nii.gz'
    W = clustools.make_local_connectivity_tcorr(image_file, mask_file, thresh=0.50)

    assert W is not None


def test_make_local_connectivity_scorr():
    """
    Test for make_local_connectivity_scorr functionality
    """
    print("testing make_local_connectivity_scorr")
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    mask_file = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    image_file = dir_path + '/002.nii.gz'
    W = clustools.make_local_connectivity_scorr(image_file, mask_file, thresh=0.50)

    assert W is not None


def test_nil_parcellate():
    """
    Test for nil_parcellate
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    clust_mask = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    func_file = dir_path + '/002.nii.gz'
    local_corr = 'tcorr'
    conf = None
    nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']
    k = 50
    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    for clust_type in nilearn_clust_list:
        uatlas = "%s%s%s%s%s%s%s%s" % (dir_path, '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz')
        region_labels = clustools.nil_parcellate(func_file, clust_mask, k, clust_type, uatlas, dir_path, conf,
                                                 local_corr, detrending=True, standardize=True)
        assert region_labels is not None


def test_individual_clustering():
    """
    Test for individual_clustering
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    clust_mask = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    func_file = dir_path + '/002.nii.gz'
    ID = '002'
    k = 50
    vox_size = '2mm'
    conf = None
    clust_type = 'ward'
    [uatlas, atlas, clustering, _, _, _] = clustools.individual_clustering(func_file, conf, clust_mask, ID, k,
                                                                           clust_type, vox_size)

    assert uatlas is not None
    assert atlas is not None
    assert clustering is True
