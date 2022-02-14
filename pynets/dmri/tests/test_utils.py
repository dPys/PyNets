#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.dmri import utils as dmriutils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_extract_b0(dmri_estimation_data):

    dwi_path = dmri_estimation_data['dwi_file']
    fbvals = dmri_estimation_data['fbvals']

    b0_ixs = np.where(np.loadtxt(fbvals) <= 50)[0].tolist()[:2]
    out_path = dmriutils.extract_b0(dwi_path, b0_ixs)
    assert os.path.isfile(out_path)


def test_normalize_grads(dmri_estimation_data):
    """
    Test make_gtab_and_bmask functionality
    """
    fbvals = dmri_estimation_data['fbvals']
    fbvecs = dmri_estimation_data['fbvecs']
    bvals = np.loadtxt(fbvals)
    bvecs = np.loadtxt(fbvecs).T
    b0_threshold = 50
    bvecs_normed, bvals_normed = dmriutils.normalize_gradients(
        bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1, b_scale=True)
    assert bvecs_normed is not None
    assert bvals_normed is not None


def test_evaluate_streamline_plausibility(dmri_estimation_data,
                                          tractography_estimation_data):
    """
    Test evaluate_streamline_plausibility functionality
    """
    import nibabel as nib
    from pynets.dmri.utils import evaluate_streamline_plausibility
    from dipy.io.stateful_tractogram import Space, Origin
    from dipy.io.streamline import load_tractogram
    from dipy.io import load_pickle

    gtab_file = dmri_estimation_data['gtab_file']
    dwi_path = dmri_estimation_data['dwi_file']
    mask_file = tractography_estimation_data['mask']
    streams = tractography_estimation_data['trk']

    gtab = load_pickle(gtab_file)
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    tractogram = load_tractogram(
        streams,
        mask_img,
        to_origin=Origin.NIFTI,
        to_space=Space.VOXMM,
        bbox_valid_check=True,
    )
    streamlines = tractogram.streamlines
    cleaned = evaluate_streamline_plausibility(dwi_data, gtab, mask_data,
                                               streamlines)

    assert len(cleaned) > 0
    assert len(cleaned) <= len(streamlines)
