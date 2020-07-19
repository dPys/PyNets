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


# def test_evaluate_streamline_plausibility():
#     """
#     Test evaluate_streamline_plausibility functionality
#     """
#     import nibabel as nib
#     from pynets.registration import reg_utils
#     from pynets.dmri.dmri_utils import evaluate_streamline_plausibility
#     from dipy.core.gradients import gradient_table
#     from dipy.io.stateful_tractogram import Space, Origin
#     from dipy.io.streamline import load_tractogram
#
#     base_dir = str(Path(__file__).parent / "examples")
#     test_dir = f"{base_dir}/003/test_out/test_check_orient_and_dims"
#     fbval = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_bval.bval"
#     fbvec = f"{base_dir}/outputs/pynets/sub-25659/ses-1/dwi/final_" \
#             f"preprocessed_dwi_bvecs_reor.bvec"
#     dwi_path = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi" \
#                f".nii.gz"
#     streams = f"{base_dir}/outputs/pynets/sub-25659/ses-1/dwi/" \
#               f"DesikanKlein2012/tractography/streamlines_SalVentAttn_csa_" \
#               f"20000_parc_curv-[40_30]_step-[0.1_0.2_0.3_0.4_0.5]_" \
#               f"directget-clos_minlength-0.trk"
#     dwi_res = reg_utils.check_orient_and_dims(dwi_path, test_dir, '2mm')
#     dwi_img = nib.load(dwi_res)
#     tractogram = load_tractogram(
#         streams,
#         dwi_img,
#         to_origin=Origin.TRACKVIS,
#         to_space=Space.VOXMM,
#         bbox_valid_check=True,
#     )
#     streamlines = tractogram.streamlines
#     gtab = gradient_table(fbval, fbvec)
#     dwi_data = dwi_img.get_fdata()
#     cleaned = evaluate_streamline_plausibility(dwi_data, gtab, streamlines)
#
#     assert len(cleaned) > 0
