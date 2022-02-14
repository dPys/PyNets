#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
# import time
import nibabel as nib
import os
import pkg_resources
import tempfile
import logging
# from inspect import getargspec
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.dmri.estimation import (create_anisopowermap, tens_mod_fa_est,
                                    tens_mod_est, csa_mod_est, csd_mod_est,
                                    streams2graph, sfm_mod_est)
from nipype.utils.filemanip import fname_presuffix
from nilearn.image import resample_to_img

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_create_anisopowermap(dmri_estimation_data):
    """ Test creating an anisotropic power map."""

    gtab_file = dmri_estimation_data['gtab_file']
    dwi_file = dmri_estimation_data['dwi_file']
    B0_mask_file = dmri_estimation_data['B0_mask']

    assert os.path.isfile(create_anisopowermap(gtab_file, dwi_file,
                                               B0_mask_file)[0])


def test_tens_mod_fa_est(dmri_estimation_data):
    """Test tensor FA image estimation."""

    gtab_file = dmri_estimation_data['gtab_file']
    dwi_file = dmri_estimation_data['dwi_file']
    B0_mask = dmri_estimation_data['B0_mask']

    assert os.path.isfile(tens_mod_fa_est(gtab_file, dwi_file, B0_mask)[0])


def test_tens_mod_est(dmri_estimation_data):
    """Test tensor ODF model estimation."""
    gtab = dmri_estimation_data['gtab']
    dwi_file = dmri_estimation_data['dwi_file']
    dwi_img = nib.load(dwi_file)
    B0_mask_file = dmri_estimation_data['B0_mask']

    dwi_data = dwi_img.get_fdata()

    [mod_odf, model] = \
        tens_mod_est(gtab, dwi_data, B0_mask_file)

    assert mod_odf is not None
    assert model is not None


def test_csa_mod_est(dmri_estimation_data):
    """Test CSA model estmation."""

    gtab = dmri_estimation_data['gtab']
    dwi_file = dmri_estimation_data['dwi_file']
    dwi_img = nib.load(dwi_file)
    B0_mask_file = dmri_estimation_data['B0_mask']

    dwi_data = dwi_img.get_fdata()

    [csa_mod, model] = \
        csa_mod_est(gtab, dwi_data, B0_mask_file, sh_order=0)

    assert csa_mod is not None
    assert model is not None


def test_csd_mod_est(dmri_estimation_data):
    """Test CSD model estimation."""

    gtab = dmri_estimation_data['gtab']
    dwi_file_small = dmri_estimation_data['dwi_file_small']
    dwi_data_small = nib.load(dwi_file_small).get_fdata()
    B0_mask_file_small = dmri_estimation_data['B0_mask_small']

    [csd_mod, model] = csd_mod_est(gtab, dwi_data_small,
                                   B0_mask_file_small, sh_order=0)

    assert csd_mod is not None
    assert model is not None


def test_sfm_mod_est(dmri_estimation_data):
    """Test SFM model estimation."""

    gtab = dmri_estimation_data['gtab']
    dwi_file = dmri_estimation_data['dwi_file_small']
    dwi_data_small = nib.load(dwi_file).get_fdata()
    B0_mask_file = dmri_estimation_data['B0_mask_small']

    [sf_odf, model] = sfm_mod_est(gtab, dwi_data_small, B0_mask_file)

    assert sf_odf is not None
    assert model is not None


@pytest.mark.parametrize("conn_model", ['csa', 'csd', 'ten'])
def test_reconstruction(conn_model):
    """
    Test for reconstruction functionality
    """
    from pynets.dmri.estimation import reconstruction
    from dipy.core.gradients import gradient_table
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{dir_path}/sub-003_dwi.bvec"
    gtab = gradient_table(bvals, bvecs)
    dwi_file = f"{dir_path}/sub-003_dwi.nii.gz"
    wm_in_dwi = f"{dir_path}/wm_mask_dmri.nii.gz"

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    model, mod = reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)
    assert model is not None
    assert mod is not None


#@pytest.mark.parametrize("dsn", [False])
#@pytest.mark.parametrize("fa_wei", [True, False])
def test_streams2graph(dmri_estimation_data,
                       tractography_estimation_data,
                       random_mni_roi_data, dsn=False):
    from pynets.registration.register import direct_streamline_norm
    from dipy.core.gradients import gradient_table
    from dipy.io import save_pickle
    import random
    import pkg_resources

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    dwi_file = dmri_estimation_data['dwi_file']
    conn_model = 'csd'
    min_length = 10
    error_margin = 5
    traversal = 'prob'
    track_type = 'local'
    min_span_tree = False
    prune = 3
    norm = 6
    binary = False
    roi = random_mni_roi_data['roi_file']
    subnet = 'Default'
    ID = '003'
    parc = True
    disp_filt = False
    node_radius = None
    dens_thresh = False
    atlas = 'whole_brain_cluster_labels_PCA200'
    parcellation = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    streams = tractography_estimation_data['trk']
    B0_mask = dmri_estimation_data['B0_mask']
    bvals = dmri_estimation_data['fbvals']
    bvecs = dmri_estimation_data['fbvecs']
    gtab_file = dmri_estimation_data['gtab_file']
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    save_pickle(gtab_file, gtab)
    fa_path = tens_mod_fa_est(gtab_file, dwi_file, B0_mask)[0]
    ap_path = create_anisopowermap(gtab_file, dwi_file, B0_mask)[0]
    t1w_brain = dmri_estimation_data['t1w_file']
    t1w_gm = dmri_estimation_data['f_pve_gm']
    atlas_in_dwi = fname_presuffix(ap_path,
                                suffix="atlas_in_dwi", use_ext=True)
    resample_to_img(
        nib.load(parcellation), nib.load(fa_path), interpolation="nearest"
    ).to_filename(atlas_in_dwi)

    coords = [(random.random()*2.0, random.random()*2.0, random.random()*2.0)
              for _ in range(len(np.unique(nib.load(atlas_in_dwi).get_fdata()
                                           ))-1)]
    labels = np.arange(len(coords) + 1)[np.arange(len(coords
                                                      ) + 1) != 0].tolist()

    if dsn is True:
        os.makedirs(f"{dir_path}/dmri_reg/DSN", exist_ok=True)
        (streams,
            dir_path,
            track_type,
            conn_model,
            subnet,
            node_radius,
            dens_thresh,
            ID,
            roi,
            min_span_tree,
            disp_filt,
            parc,
            prune,
            atlas,
            parcellation,
            labels,
            coords,
            norm,
            binary,
            atlas_for_streams,
            traversal,
            fa_path,
            min_length
        ) = direct_streamline_norm(
            streams,
            fa_path,
            ap_path,
            dir_path,
            track_type,
            conn_model,
            subnet,
            node_radius,
            dens_thresh,
            ID,
            roi,
            min_span_tree,
            disp_filt,
            parc,
            prune,
            atlas,
            atlas_in_dwi,
            parcellation,
            labels,
            coords,
            norm,
            binary,
            t1w_gm,
            dir_path,
            [0.1, 0.2],
            [40, 30],
            traversal,
            min_length,
            t1w_brain,
            run_dsn=True
        )

    conn_matrix = streams2graph(
                    atlas_in_dwi,
                    streams,
                    os.path.dirname(streams),
                    track_type,
                    conn_model,
                    subnet,
                    node_radius,
                    dens_thresh,
                    ID,
                    roi,
                    min_span_tree,
                    disp_filt,
                    parc,
                    prune,
                    atlas,
                    atlas_in_dwi,
                    labels,
                    coords,
                    norm,
                    binary,
                    traversal,
                    fa_path,
                    min_length,
                    error_margin)[2]

    assert conn_matrix is not None
