#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017
"""
import pytest
import pkg_resources
import numpy as np
import nibabel as nib
import sys
import tempfile
if sys.platform.startswith('win') is False:
    import indexed_gzip
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import clustering
from pynets.fmri.interfaces import NiParcellate
import os
import logging
from nipype.utils.filemanip import fname_presuffix
from nilearn.image import resample_to_img

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_indx_1dto3d():
    """
    Test for indx_1dto3d functionality
    """
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustering.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_indx_3dto1d():
    """
    Test for indx_3dto1d functionality
    """
    sz = np.random.rand(10, 10, 10)
    idx = 45

    x, y, z = clustering.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None


def test_make_local_connectivity_tcorr(fmri_estimation_data):
    """
    Test for make_local_connectivity_tcorr functionality
    """
    print("testing make_local_connectivity_tcorr")
    mask_file = fmri_estimation_data['mask_file2']
    func_file = fmri_estimation_data['func_file2']
    func_img = nib.load(func_file)
    mask_img = nib.load(mask_file)
    W = clustering.make_local_connectivity_tcorr(func_img, mask_img,
                                                 thresh=0.5)

    assert W is not None

    out_img = clustering.parcellate_ncut(W, 20, mask_img)

    assert isinstance(out_img, nib.Nifti1Image)


def test_make_local_connectivity_scorr(fmri_estimation_data):
    """
    Test for make_local_connectivity_scorr functionality
    """
    print("testing make_local_connectivity_scorr")
    mask_file = fmri_estimation_data['mask_file2']
    func_file = fmri_estimation_data['func_file2']
    func_img = nib.load(func_file)
    mask_img = nib.load(mask_file)
    W = clustering.make_local_connectivity_scorr(func_img, mask_img,
                                                 thresh=0.50)

    assert W is not None

    out_img = clustering.parcellate_ncut(W, 200, mask_img)

    assert out_img is not None


@pytest.mark.parametrize("clust_type", ['rena', 'average',
                                        'complete', 'ward',
                                        pytest.param('single',
                                                     marks=pytest.mark.xfail)])
# 1 connected component
def test_ni_parcellate(fmri_estimation_data, random_mni_roi_data,
                       clust_type):
    """
    Test for ni_parcellate
    """

    k = 10
    tmp = tempfile.TemporaryDirectory()

    tmpdir = str(tmp.name)
    os.makedirs(tmpdir, exist_ok=True)

    if clust_type != 'ncut':
        local_corr = 'allcorr'
    else:
        local_corr = 'tcorr'
    mask_file = fmri_estimation_data['mask_file']
    func_file = fmri_estimation_data['func_file']
    roi_file = random_mni_roi_data['roi_file']

    clust_mask_file = fname_presuffix(mask_file,
                                      suffix="clust_mask_file",
                                      use_ext=True)
    resample_to_img(
        nib.load(roi_file), nib.load(mask_file), interpolation="nearest"
    ).to_filename(clust_mask_file)
    func_img = nib.load(func_file)
    nip = NiParcellate(func_file=func_file, clust_mask=clust_mask_file,
                       k=k, clust_type=clust_type, local_corr=local_corr,
                       outdir=tmpdir)
    atlas = nip.create_clean_mask()
    nip.create_local_clustering(overwrite=True, r_thresh=0.5)
    out_path = f"{tmpdir}/parc_tmp.nii.gz"
    parcellation = clustering.parcellate(func_img, local_corr,
                                         clust_type, nip._local_conn_mat_path,
                                         nip.num_conn_comps,
                                         nip._clust_mask_corr_img,
                                         nip._standardize,
                                         nip._detrending, nip.k,
                                         nip._local_conn,
                                         nip.conf, tmpdir,
                                         nip._conn_comps)

    nib.save(parcellation, out_path)
    assert out_path is not None
    assert atlas is not None
    tmp.cleanup()


@pytest.mark.parametrize("clust_type", ['ward', 'ncut', 'rena',
                                        pytest.param('average',
                                                     marks=pytest.mark.xfail),
                                        pytest.param('complete',
                                                     marks=pytest.mark.xfail)])
# >1 connected components
def test_ni_parcellate_mult_conn_comps(fmri_estimation_data, clust_type):
    """
    Test for ni_parcellate with multiple connected components
    """
    tmp = tempfile.TemporaryDirectory()
    tmpdir = str(tmp.name)
    os.makedirs(tmpdir, exist_ok=True)

    k = 100
    if clust_type != 'ncut':
        local_corr = 'allcorr'
    else:
        local_corr = 'tcorr'
    clust_mask = pkg_resources.resource_filename(
        "pynets", "templates/rois/triple_net_ICA_overlap_3_sig_bin.nii.gz"
    )
    mask_file = fmri_estimation_data['mask_file']
    func_file = fmri_estimation_data['func_file']
    func_img = nib.load(func_file)

    clust_mask_file = fname_presuffix(mask_file,
                                suffix="clust_mask_file", use_ext=True)
    resample_to_img(
        nib.load(clust_mask), nib.load(mask_file), interpolation="nearest"
    ).to_filename(clust_mask_file)

    nip = NiParcellate(func_file=func_file, clust_mask=clust_mask_file,
                                  k=k, clust_type=clust_type,
                                  local_corr=local_corr, outdir=tmpdir,
                                  mask=mask_file)

    atlas = nip.create_clean_mask()

    if not nip.parcellation:
        nip.parcellation = f"{tmpdir}/clust-{clust_type}_k{str(k)}.nii.gz"
    nip._clust_mask_corr_img = nib.load(clust_mask_file)
    nip.create_local_clustering(overwrite=True, r_thresh=0.4)
    out_path = f"{tmpdir}/parc_tmp.nii.gz"

    parcellation = clustering.parcellate(func_img, local_corr,
                                         clust_type, nip._local_conn_mat_path,
                                         nip.num_conn_comps,
                                         nip._clust_mask_corr_img,
                                         nip._standardize,
                                         nip._detrending, nip.k,
                                         nip._local_conn,
                                         nip.conf, tmpdir,
                                         nip._conn_comps)

    nib.save(parcellation, out_path)
    assert atlas is not None
    assert os.path.isfile(out_path)
    tmp.cleanup()
