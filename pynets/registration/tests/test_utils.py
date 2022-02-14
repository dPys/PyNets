#!/usr/bin/env python
"""
Created on Monday July 29 16:19:14 2019
"""
import pytest
import numpy as np
from pynets.registration import utils
import pkg_resources
import os
import nibabel as nib
import indexed_gzip
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import warnings
warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_align():
    """
    Test align functionality
    """
    import pkg_resources

    # Linear registration
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    inp = f"{anat_dir}/sub-003_T1w_brain.nii.gz"
    ref = pkg_resources.resource_filename("pynets",
                                          f"templates/MNI152_T1_brain_"
                                          f"2mm.nii.gz")
    out = f"{anat_dir}/highres2standard.nii.gz"
    xfm_out = f"{anat_dir}/highres2standard.mat"

    utils.align(inp, ref, xfm=xfm_out, out=out, dof=12, searchrad=True,
                bins=256, interp=None, cost="mutualinfo",
                sch=None, wmseg=None, init=None)

    highres2standard_linear = nib.load(out)
    assert highres2standard_linear is not None


def test_applyxfm():
    """
    Test applyxfm functionality
    """
    import pkg_resources

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"

    ## First test: Apply xfm from test_align to orig anat img.
    inp = f"{anat_dir}/sub-003_T1w_brain.nii.gz"
    ref = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz")
    xfm = f"{anat_dir}/highres2standard.mat"
    aligned = f"{anat_dir}/highres2standard_2.nii.gz"
    utils.applyxfm(ref, inp, xfm, aligned, interp='trilinear', dof=6)
    # Check test_applyfxm = test_align outputs
    out_applyxfm = nib.load(aligned)
    out_applyxfm_data = out_applyxfm.get_data()
    out_align_file = f"{anat_dir}/highres2standard.nii.gz"
    out_align = nib.load(out_align_file)
    out_align_data = out_align.get_data()
    check_eq_arrays = np.array_equal(out_applyxfm_data, out_align_data)
    assert check_eq_arrays is True

    ## Second test: Apply xfm to standard space roi (invert xfm first) >> native space roi.
    # ref is native space anat image
    ref = f"{anat_dir}/sub-003_T1w.nii.gz"
    # input is standard space precuneus mask
    inp = f"{anat_dir}/precuneous_thr_bin.nii.gz"
    # xfm is standard2native from convert_xfm -omat standard2highres.mat highres2standard.mat
    xfm = f"{anat_dir}/standard2highres.mat"
    # precuenus mask in antive space
    aligned = f"{anat_dir}/precuneous2highres.nii.gz"

    utils.applyxfm(ref, inp, xfm, aligned, interp='trilinear', dof=6)
    test_out = nib.load(aligned)
    assert test_out is not None


def test_align_nonlinear():
    """
    Test align_nonlinear functionality
    """
    import pkg_resources

    # Nonlinear normalization
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    inp = f"{anat_dir}/sub-003_T1w_brain.nii.gz"
    ref = pkg_resources.resource_filename("pynets",
                                          f"templates/"
                                          f"MNI152_T1_brain_2mm.nii.gz")
    out = f"{anat_dir}/highres2standard_nonlinear.nii.gz"
    warp = f"{anat_dir}/highres2standard_warp"
    # affine mat created from test_align above.
    xfm = f"{anat_dir}/highres2standard.mat"

    utils.align_nonlinear(inp, ref, xfm, out, warp, ref_mask=None,
                          in_mask=None, config=None)

    highres2standard_nonlin = nib.load(out)
    assert highres2standard_nonlin is not None


def test_combine_xfms():
    """
    Test combine_xfms functionality
    """
    # Combine func2anat and anat2std to create func2std mat
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    xfm1 = f"{anat_dir}/example_func2highres.mat"
    xfm2 = f"{anat_dir}/highres2standard.mat"
    xfmout = f"{anat_dir}/example_func2standard.mat"

    utils.combine_xfms(xfm1, xfm2, xfmout)
    test_out = np.genfromtxt(xfmout, delimiter='  ')
    assert test_out is not None


def test_invwarp():
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    ref = f"{anat_dir}/sub-003_T1w.nii.gz"
    warp = f"{anat_dir}/highres2standard_warp"
    out = f"{anat_dir}/highres2standard_warp_inv.nii.gz"
    utils.inverse_warp(ref, out, warp)
    out_warp = nib.load(out)
    assert out_warp is not None


def test_apply_warp():
    import pkg_resources
    # Warp original anat to standard space using warp img (had to
    # invwarp first) and linear mats
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    ref = pkg_resources.resource_filename("pynets",
                                          f"templates/MNI152_T1_brain_"
                                          f"2mm.nii.gz")
    inp = f"{anat_dir}/sub-003_T1w.nii.gz"
    out = f"{anat_dir}/highres2standard_test_apply_warp.nii.gz"
    warp = f"{anat_dir}/highres2standard_warp.nii.gz"
    xfm = f"{anat_dir}/highres2standard.mat"

    utils.apply_warp(ref, inp, out, warp, xfm=xfm, mask=None, interp=None,
                     sup=False)
    # highres2standard_apply_warp = f"{anat_dir}/highres2standard_test_apply_warp.nii.gz"
    # highres2standard_apply_warp = nib.load(highres2standard_apply_warp)
    # highres2standard_apply_warp = highres2standard_apply_warp.get_data()
    #
    # highres2standard_align_nonlinear = nib.load(f"{anat_dir}/highres2standard_nonlinear.nii.gz")
    # highres2standard_align_nonlinear = highres2standard_align_nonlinear.get_data()
    # check_eq_arrays = np.allclose(highres2standard_apply_warp.astype('float32'),
    #                                  highres2standard_align_nonlinear.astype('float32'))
    # assert check_eq_arrays is True
    assert os.path.isfile(out)


def test_segment_t1w():
    """
    Test segment_t1w functionality
    """
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    anat_dir = f"{base_dir}/003/anat"
    t1w = f"{anat_dir}/sub-003_T1w_brain.nii.gz"
    basename = f"test_segment_t1w"
    out = utils.segment_t1w(t1w, basename, max_iter=10)
    print(out)
    assert out is not None


def test_match_target_vox_res():
    """
    Test match_target_vox_res functionality
    """
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    test_out = f"{base_dir}/003/test_out/test_match_target_vox_res"

    # Orig anat input has isotropic (1x1x1mm) dimensions.
    anat_img_file = f"{test_out}/sub-003_T1w_pre_res_res-2mm.nii.gz"
    anat_vox_size = '2mm'
    anat_out_dir = test_out
    anat_img_file = utils.match_target_vox_res(anat_img_file,
                                               anat_vox_size,
                                               anat_out_dir,
                                               remove_orig=False)
    anat_new_img = nib.load(anat_img_file)
    anat_dims = anat_new_img.header.get_zooms()
    anat_success = True
    for anat_dim in np.round(anat_dims[:3], 2):
        if anat_dim != 2:
            anat_success = False

    # Orig dMRI image has anisotropic (1.75x1.75x3mm) dimensions.
    dwi_img_file = f"{test_out}/sub-003_dwi_pre_res_res-1mm.nii.gz"
    dwi_vox_size = '1mm'
    dwi_out_dir = test_out
    dwi_img_file = utils.match_target_vox_res(dwi_img_file, dwi_vox_size,
                                              dwi_out_dir,
                                              remove_orig=False)
    dwi_new_img = nib.load(dwi_img_file)
    dwi_dims = dwi_new_img.header.get_zooms()
    dwi_success = True
    for dwi_dim in np.round(dwi_dims[:3], 2):
        if dwi_dim != 1:
            dwi_success = False

    assert anat_img_file is not None
    assert anat_success is True
    assert dwi_img_file is not None
    assert dwi_success is True


def test_reorient_dwi():
    """
    Test reorient_dwi functionality
    """
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    test_dir = f"{base_dir}/003/test_out/test_reorient_dwi"

    # iso_eddy_corrected_data_denoised_LAS.nii.gz was the original image in
    # radiological orientation. fslswapdim and fslorient manually used to
    # create RAS image. This test attempts to convert RAS image back to LAS.
    # Confirms by checking output array is equal to origal LAS image array.

    dwi_prep_rad = f"{test_dir}/iso_eddy_corrected_data_denoised_LAS.nii.gz"
    dwi_prep_neu = f"{test_dir}/iso_eddy_corrected_data_denoised_RAS.nii.gz"
    bvecs_orig = f"{test_dir}/bvec.bvec"
    out_dir = f"{test_dir}/output"

    dwi_prep_out, bvecs_out = utils.reorient_dwi(dwi_prep_neu, bvecs_orig,
                                                 out_dir)

    orig_rad = nib.load(dwi_prep_rad)
    orig_rad_data = orig_rad.get_data()

    reorient_rad = nib.load(dwi_prep_out)
    reorient_rad_data = reorient_rad.get_data()

    reorient_check = np.array_equal(orig_rad_data, reorient_rad_data)
    bvec_check = np.array_equal(bvecs_orig, bvecs_out)

    assert bvec_check is False
    assert reorient_check is True


def test_reorient_img():
    """
    Test reorient_img functionality
    """
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    test_dir = f"{base_dir}/003/test_out/test_reorient_img"

    # X axis increasing right to left (Radiological)
    img_in_radio = f"{test_dir}/sub-003_T1w_LAS.nii.gz"
    out_radio_dir = f"{test_dir}/output_LAS"

    # X axis increase from left to right (Neurological)
    img_in_neuro = f"{test_dir}/sub-003_T1w_RAS.nii.gz"
    out_neuro_dir = f"{test_dir}/output_RAS"

    # Outputs should be in neurological orientation.
    LAStoRAS_img_out = utils.reorient_img(img_in_radio, out_radio_dir)
    RAStoRAS_img_out = utils.reorient_img(img_in_neuro, out_neuro_dir)

    # Original RAS data
    orig_RAS_img = nib.load(img_in_neuro)

    # Output from LAS input
    LAStoRAS_img = nib.load(LAStoRAS_img_out)

    # Output from RAS input
    RAStoRAS_img = nib.load(RAStoRAS_img_out)

    # Assert that arrays are equal
    check_LAS_input = np.allclose(LAStoRAS_img.affine.astype('int'),
                                  orig_RAS_img.affine.astype('int'))
    check_RAS_input = np.allclose(RAStoRAS_img.affine.astype('int'),
                                  orig_RAS_img.affine.astype('int'))
    check_both_outputs = np.allclose(LAStoRAS_img.affine.astype('int'),
                                     RAStoRAS_img.affine.astype('int'))

    assert check_LAS_input is True
    assert check_RAS_input is True
    assert check_both_outputs is True


def test_orient_reslice():
    """
    Test orient_reslice functionality
    """
    # This test has a bak folder in its test_dir.
    # To replicate test rm data in test_dir and cp from bak
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    test_dir = f"{base_dir}/003/test_out/test_check_orient_and_dims"

    # Antomical: 1x1x1mm
    anat_LAS = f"{test_dir}/anat_LAS/sub-003_T1w_LAS.nii.gz"
    anat_RAS = f"{test_dir}/anat_RAS/sub-003_T1w_RAS.nii.gz"
    # Diffusion: 2x2x2mm
    dmri_LAS = f"{test_dir}/dmri_LAS/iso_eddy_corrected_data_denoised_LAS." \
               f"nii.gz"
    dmri_RAS = f"{test_dir}/dmri_RAS/iso_eddy_corrected_data_denoised_RAS." \
               f"nii.gz"
    bvecs_LAS = f"{test_dir}/dmri_LAS/bvec.orig.bvec"
    bvecs_RAS = f"{test_dir}/dmri_RAS/bvec.trans.bvec"

    anat_LAStoRAS = utils.orient_reslice(anat_LAS, test_dir, '2mm',
                                                bvecs=None)
    anat_RAStoRAS = utils.orient_reslice(anat_RAS, test_dir, '2mm',
                                                bvecs=None)
    dmri_LAStoRAS, bvecs_LAStoRAS = utils.orient_reslice(
        dmri_LAS, test_dir, '1mm', bvecs=bvecs_LAS)
    dmri_RAStoRAS, bvecs_RAStoRAS = utils.orient_reslice(
        dmri_RAS, test_dir, '1mm', bvecs=bvecs_RAS)

    anat_LAStoRAS = nib.load(anat_LAStoRAS)

    anat_RAStoRAS = nib.load(anat_RAStoRAS)

    dmri_LAStoRAS = nib.load(dmri_LAStoRAS)

    dmri_RAStoRAS = nib.load(dmri_RAStoRAS)

    # Assert that output arrays are identical.
    anat_check = np.allclose(anat_LAStoRAS.affine.astype('int'),
                             anat_RAStoRAS.affine.astype('int'))
    dmri_check = np.allclose(dmri_LAStoRAS.affine.astype('int'),
                             dmri_RAStoRAS.affine.astype('int'))

    # Assert that voxel dimensions in ouputs are correct.
    anat_LAStoRAS_dims = anat_LAStoRAS.header.get_zooms()
    anat_RAStoRAS_dims = anat_RAStoRAS.header.get_zooms()
    dmri_LAStoRAS_dims = dmri_LAStoRAS.header.get_zooms()
    dmri_RAStoRAS_dims = dmri_RAStoRAS.header.get_zooms()

    anat_LAStoRAS_success = True
    anat_RAStoRAS_success = True
    dmri_LAStoRAS_success = True
    dmri_RAStoRAS_success = True

    for anat_LAStoRAS_dim in anat_LAStoRAS_dims[:3]:
        if anat_LAStoRAS_dim != 2:
            anat_LAStoRAS_success = False

    for anat_RAStoRAS_dim in anat_RAStoRAS_dims[:3]:
        if anat_RAStoRAS_dim != 2:
            anat_RAStoRAS_success = False

    for dmri_LAStoRAS_dim in dmri_LAStoRAS_dims[:3]:
        if dmri_LAStoRAS_dim != 1:
            dmri_LAStoRAS_success = False

    print(dmri_RAStoRAS_dims)
    for dmri_RAStoRAS_dim in dmri_RAStoRAS_dims[:3]:
        if dmri_RAStoRAS_dim != 1:
            dmri_RAStoRAS_success = False

    # Checks arrays
    assert anat_check is True
    assert dmri_check is True
    # Checks voxel dimensions
    assert anat_LAStoRAS_success is True
    assert anat_RAStoRAS_success is True
    assert dmri_LAStoRAS_success is True
    assert dmri_RAStoRAS_success is True


def test_make_median_b0():
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    dwi_file = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_" \
               f"dwi.nii.gz"
    mean_file_out = utils.median(dwi_file)

    assert os.path.isfile(mean_file_out)


@pytest.mark.parametrize("mask_type", ['Normal', 'Empty'])
def test_gen_mask(mask_type):
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    t1w_head = f"{base_dir}/003/anat/sub-003_T1w.nii.gz"
    t1w_brain = f"{base_dir}/003/anat/t1w_brain.nii.gz"
    if mask_type == 'Normal':
        mask = f"{base_dir}/003/anat/t1w_brain_mask.nii.gz"
    elif mask_type == 'Empty':
        mask = None
    utils.gen_mask(t1w_head, t1w_brain, mask)


def test_deep_skull_strip():
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    t1w_head = f"{base_dir}/003/anat/sub-003_T1w.nii.gz"
    img = nib.load(t1w_head)
    t1w_brain_mask = f"{base_dir}/003/anat/t1w_brain_mask.nii.gz"
    t1w_data = img.get_fdata()
    t1w_brain_mask = utils.deep_skull_strip(t1w_data, t1w_brain_mask, img)

    assert os.path.isfile(t1w_brain_mask)


def test_rescale_affine_to_center():
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

    input_affine = nib.load(f"{base_dir}/003/anat/sub-003_T1w.nii.gz").affine

    out_aff = utils.rescale_affine_to_center(input_affine)
    assert np.allclose(out_aff, input_affine)

    out_aff = utils.rescale_affine_to_center(input_affine,
                                             voxel_dims=[2, 2, 2])
    assert np.allclose(out_aff, input_affine* np.array([2, 2, 2, 1]))


def test_wm_syn():
    from dipy.align import imwarp, imaffine

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    t1w_brain = f"{base_dir}/003/anat/t1w_brain.nii.gz"
    ap_path = f"{base_dir}/003/anat/aniso_power_tmp.nii.gz"

    [mapping, affine_map, warped_fa] = utils.wm_syn(t1w_brain, ap_path,
                                                    f"{base_dir}/003/dmri")
    assert os.path.isfile(warped_fa) is True
    assert isinstance(mapping, imwarp.DiffeomorphicMap)
    assert isinstance(affine_map, imaffine.AffineMap) and \
           affine_map.affine.shape == (4, 4)
