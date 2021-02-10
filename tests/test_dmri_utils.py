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
from pynets.dmri import utils as dmriutils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_extract_b0():

    base_dir = str(Path(__file__).parent/"examples")
    dwi_path = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz"
    fbval = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_bval.bval"
    b0_ixs = np.where(np.loadtxt(fbval) <= 50)[0].tolist()[:2]
    out_path = dmriutils.extract_b0(dwi_path, b0_ixs)
    assert os.path.isfile(out_path)


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
    bvecs_normed, bvals_normed = dmriutils.normalize_gradients(
        bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1, b_scale=True)
    assert bvecs_normed is not None
    assert bvals_normed is not None


def test_evaluate_streamline_plausibility():
    """
    Test evaluate_streamline_plausibility functionality
    """
    import nibabel as nib
    from pynets.dmri.utils import evaluate_streamline_plausibility
    from dipy.io.stateful_tractogram import Space, Origin
    from dipy.io.streamline import load_tractogram
    from dipy.io import load_pickle

    base_dir = str(Path(__file__).parent / "examples")
    gtab_file = f"{base_dir}/miscellaneous/tractography/gtab.pkl"
    dwi_path = f"{base_dir}/miscellaneous/tractography/sub-OAS31172_" \
               f"ses-d0407_dwi_reor-RAS_res-2mm.nii.gz"
    B0_mask = f"{base_dir}/miscellaneous/tractography/mean_B0_bet_mask.nii.gz"
    streams = f"{base_dir}/miscellaneous/tractography/streamlines_csa_" \
              f"20000_parc_curv-[40_30]_step-[0.1_0.2_0.3_0.4_0.5]_" \
              f"directget-prob_minlength-20.trk"

    gtab = load_pickle(gtab_file)
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    B0_mask_img = nib.load(B0_mask)
    B0_mask_data = B0_mask_img.get_fdata()
    tractogram = load_tractogram(
        streams,
        B0_mask_img,
        to_origin=Origin.NIFTI,
        to_space=Space.VOXMM,
        bbox_valid_check=False,
    )
    streamlines = tractogram.streamlines
    cleaned = evaluate_streamline_plausibility(dwi_data, gtab, B0_mask_data,
                                               streamlines)

    assert len(cleaned) > 0
    assert len(cleaned) <= len(streamlines)
