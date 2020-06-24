#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.dmri import dmri_utils as dmriutils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


# def test_extract_b0():
#
#     base_dir = str(Path(__file__).parent/"examples")
#     dwi_path = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz"
#     fbval = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_bval.bval"
#     b0_ixs = np.where(np.loadtxt(fbval) <= 50)[0].tolist()[:2]
#     out_path = dmriutils.extract_b0(dwi_path, b0_ixs)
#     assert os.path.isfile(out_path)


def test_normalize_grads():
    """
    Test make_gtab_and_bmask functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    fbval = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_bval.bval"
    fbvec = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_bvec.bvec"
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec)
    b0_threshold = 50
    bvecs_normed, bvals_normed = dmriutils.normalize_gradients(bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1,
                                                               b_scale=True)
    assert bvecs_normed is not None
    assert bvals_normed is not None
