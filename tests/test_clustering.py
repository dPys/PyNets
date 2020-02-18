#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import indexed_gzip
import nibabel as nib
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
    func_img = nib.load(image_file)
    mask_img = nib.load(mask_file)
    W = clustools.make_local_connectivity_tcorr(func_img, mask_img, thresh=0.50)

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
    func_img = nib.load(image_file)
    mask_img = nib.load(mask_file)
    W = clustools.make_local_connectivity_scorr(func_img, mask_img, thresh=0.50)

    assert W is not None


@pytest.mark.parametrize("local_corr", ['scorr', 'tcorr', 'allcorr'])
@pytest.mark.parametrize("clust_type", ['kmeans', 'ward', 'rena', pytest.param('single', marks=pytest.mark.xfail),
                                        pytest.param('average', marks=pytest.mark.xfail),
                                        pytest.param('complete', marks=pytest.mark.xfail)])
@pytest.mark.parametrize("k", [pytest.param(0, marks=pytest.mark.xfail), 100])
def test_ni_parcellate(local_corr, clust_type, k):
    """
    Test for ni_parcellate
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    clust_mask = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    func_file = dir_path + '/002.nii.gz'
    conf = None
    mask = None
    nip = clustools.NiParcellate(func_file=func_file, clust_mask=clust_mask, k=k, clust_type=clust_type,
                                 local_corr=local_corr, conf=conf, mask=mask)
    atlas = nip.create_clean_mask()
    nip.create_local_clustering(overwrite=True, r_thresh=0.5)
    uatlas = nip.parcellate()
    assert atlas is not None
    assert uatlas is not None
