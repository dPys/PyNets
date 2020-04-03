#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.dmri import dmri_utils as dmriutils


def test_make_gtab_and_bmask():
    """
    Test make_gtab_and_bmask functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dwi_path = base_dir + '/002/dmri'
    fbval = dwi_path + '/bval.bval'
    fbvec = dwi_path + '/bvec.bvec'
    dwi_file = dwi_path + '/iso_eddy_corrected_data_denoised.nii.gz'
    network = 'Default'
    node_size = 6
    atlases = ['Power', 'Shirer', 'Shen', 'Smith']

    for atlas in atlases:
        [gtab_file, B0_bet, B0_mask, dwi_file] = dmriutils.make_gtab_and_bmask(
            fbval, fbvec, dwi_file, network, node_size, atlas, dwi_path)

    assert gtab_file is not None
    assert B0_bet is not None
    assert B0_mask is not None
    assert dwi_file is not None


def test_make_mean_b0():
    base_dir = str(Path(__file__).parent/"examples")
    dwi_path = base_dir + '/002/dmri'
    dwi_file = dwi_path + '/iso_eddy_corrected_data_denoised.nii.gz'
    mean_file_out = dmriutils.make_mean_b0(dwi_file)
    assert mean_file_out is not None


def test_normalize_grads():
    """
    Test make_gtab_and_bmask functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dwi_path = base_dir + '/002/dmri'
    fbval = dwi_path + '/bval.bval'
    fbvec = dwi_path + '/bvec.bvec'
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec)
    b0_threshold = 50
    bvecs_normed, bvals_normed = dmriutils.normalize_gradients(bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1,
                                                               b_scale=True)
    assert bvecs_normed is not None
    assert bvals_normed is not None
