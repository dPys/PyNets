#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import nibabel as nib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import clustools
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


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
    mask_file = f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-MNI152NLin6Asym_desc-" \
        f"smoothAROMAnonaggr_bold_short.nii.gz"
    func_img = nib.load(func_file)
    mask_img = nib.load(mask_file)
    W = clustools.make_local_connectivity_tcorr(func_img, mask_img, thresh=0.50)

    assert W is not None

    out_img = clustools.parcellate_ncut(W, 200, mask_img)

    assert isinstance(out_img, nib.Nifti1Image)


def test_make_local_connectivity_scorr():
    """
    Test for make_local_connectivity_scorr functionality
    """
    print("testing make_local_connectivity_scorr")
    base_dir = str(Path(__file__).parent/"examples")
    mask_file = f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-MNI152NLin6Asym_desc-" \
        f"smoothAROMAnonaggr_bold_short.nii.gz"
    func_img = nib.load(func_file)
    mask_img = nib.load(mask_file)
    W = clustools.make_local_connectivity_scorr(func_img, mask_img,
                                                thresh=0.50)

    assert W is not None

    out_img = clustools.parcellate_ncut(W, 200, mask_img)

    assert out_img is not None


@pytest.mark.parametrize("clust_type", ['kmeans', 'rena', 'average', 'complete', 'ward', 'ncut',
                                        pytest.param('single', marks=pytest.mark.xfail)])
# 1 connected component
def test_ni_parcellate(clust_type):
    """
    Test for ni_parcellate
    """
    import tempfile

    k = 20
    base_dir = str(Path(__file__).parent/"examples")
    out_dir = f"{base_dir}/outputs/sub-25659/ses-1/func"
    tmpdir = tempfile.TemporaryDirectory()
    if clust_type != 'ncut':
        local_corr = 'allcorr'
    else:
        local_corr = 'tcorr'
    clust_mask = f"{base_dir}/miscellaneous/rMFG_node6mm.nii.gz"
    mask = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_mask.nii.gz"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-MNI152NLin6Asym_desc-" \
        f"smoothAROMAnonaggr_bold_short.nii.gz"
    func_img = nib.load(func_file)
    nip = clustools.NiParcellate(func_file=func_file, clust_mask=clust_mask, k=k, clust_type=clust_type,
                                 local_corr=local_corr, outdir=out_dir, conf=None, mask=mask)
    atlas = nip.create_clean_mask()
    nip.create_local_clustering(overwrite=True, r_thresh=0.4)
    out_path = f"{str(tmpdir.name)}/parc_tmp.nii.gz"
    parcellation = clustools.parcellate(func_img, local_corr,
                              clust_type, nip._local_conn_mat_path,
                              nip.num_conn_comps,
                              nip._clust_mask_corr_img,
                              nip._standardize,
                              nip._detrending, nip.k, nip._local_conn,
                              nip.conf, tmpdir.name,
                              nip._conn_comps)

    nib.save(parcellation, out_path)
    assert out_path is not None
    assert atlas is not None


@pytest.mark.parametrize("clust_type", ['ward', 'ncut', 'kmeans', 'rena',
                                        pytest.param('average', marks=pytest.mark.xfail),
                                        pytest.param('complete', marks=pytest.mark.xfail)])
# >1 connected components
def test_ni_parcellate_mult_conn_comps(clust_type):
    """
    Test for ni_parcellate with multiple connected components
    """
    import tempfile

    base_dir = str(Path(__file__).parent/"examples")
    out_dir = f"{base_dir}/outputs/sub-25659/ses-1/func"
    tmpdir = tempfile.TemporaryDirectory()

    k = 150
    if clust_type != 'ncut':
        local_corr = 'allcorr'
    else:
        local_corr = 'tcorr'
    clust_mask = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    mask = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_" \
           f"mask.nii.gz"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-" \
                f"rest_space-MNI152NLin6Asym_desc-" \
        f"smoothAROMAnonaggr_bold_short.nii.gz"
    conf = f"{base_dir}/BIDS/sub-25659/ses-1/func/" \
           f"sub-25659_ses-1_task-rest_desc-confounds_regressors.tsv"
    func_img = nib.load(func_file)
    nip = clustools.NiParcellate(func_file=func_file, clust_mask=clust_mask,
                                 k=k, clust_type=clust_type,
                                 local_corr=local_corr, outdir=out_dir,
                                 conf=conf, mask=mask)

    atlas = nip.create_clean_mask()

    if not nip.uatlas:
        nip.uatlas = f"{tmpdir.name}/clust-{clust_type}_k{str(k)}.nii.gz"
    nip._clust_mask_corr_img = nib.load(clust_mask)
    nip.create_local_clustering(overwrite=True, r_thresh=0.4)
    out_path = f"{str(tmpdir.name)}/parc_tmp.nii.gz"

    parcellation = clustools.parcellate(func_img, local_corr,
                              clust_type, nip._local_conn_mat_path,
                              nip.num_conn_comps,
                              nip._clust_mask_corr_img,
                              nip._standardize,
                              nip._detrending, nip.k, nip._local_conn,
                              nip.conf, tmpdir.name,
                              nip._conn_comps)

    nib.save(parcellation, out_path)
    assert atlas is not None
    assert os.path.isfile(out_path)
