#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner
"""
import os
import numpy as np
import sys
import nibabel as nib
from nipype.utils.filemanip import fname_presuffix
import warnings

warnings.filterwarnings("ignore")
try:
    FSLDIR = os.environ["FSLDIR"]
except KeyError as e:
    print(e, "FSLDIR environment variable not set!")
    sys.exit(1)


def gen_mask(t1w_head, t1w_brain, mask):
    import time
    import os.path as op
    from pynets.registration import utils as regutils
    from nilearn.image import math_img

    t1w_brain_mask = f"{op.dirname(t1w_head)}/t1w_brain_mask.nii.gz"

    img = nib.load(t1w_head)

    if mask is not None and op.isfile(mask):
        from nilearn.image import resample_to_img

        print(f"Using {mask}...")
        mask_img = nib.load(mask)
        nib.save(
            resample_to_img(
                mask_img,
                img),
            t1w_brain_mask)
    else:
        # Perform skull-stripping if mask not provided.
        img = nib.load(t1w_head)
        t1w_data = img.get_fdata(dtype=np.float32)
        try:
            t1w_brain_mask = deep_skull_strip(t1w_data, t1w_brain_mask, img)
        except RuntimeError as e:
            print(e, 'Deepbrain extraction failed...')

        del t1w_data

    # Threshold T1w brain to binary in anat space
    t_img = nib.load(t1w_brain_mask)
    img = math_img("img > 0.0", img=t_img)
    img.to_filename(t1w_brain_mask)
    t_img.uncache()
    time.sleep(0.5)

    t1w_brain = regutils.apply_mask_to_image(t1w_head, t1w_brain_mask,
                                             t1w_brain)
    time.sleep(0.5)

    assert op.isfile(t1w_brain)
    assert op.isfile(t1w_brain_mask)

    return t1w_brain, t1w_brain_mask


def deep_skull_strip(t1w_data, t1w_brain_mask, img):
    import tensorflow as tf
    if tf.__version__ > "2.0.0":
        import tensorflow.compat.v1 as tf
    from deepbrain import Extractor
    import logging

    print('Attempting deepbrain skull-stripping...')
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    ext = Extractor()
    prob = ext.run(t1w_data)
    mask = prob > 0.5
    nib.save(
        nib.Nifti1Image(mask, affine=img.affine,
                        header=img.header),
        t1w_brain_mask,
    )
    img.uncache()
    return t1w_brain_mask


def atlas2t1w2dwi_align(
    uatlas,
    uatlas_parcels,
    atlas,
    t1w_brain,
    t1w_brain_mask,
    mni2t1w_warp,
    t1_aligned_mni,
    ap_path,
    mni2t1_xfm,
    t1w2dwi_xfm,
    wm_gm_int_in_dwi,
    aligned_atlas_t1mni,
    aligned_atlas_skull,
    dwi_aligned_atlas,
    dwi_aligned_atlas_wmgm_int,
    B0_mask,
    mni2dwi_xfm,
    simple,
):
    """
    A function to perform atlas alignment atlas --> T1 --> dwi.
    Tries nonlinear registration first, and if that fails, does a linear
    registration instead. For this to succeed, must first have called
    t1w2dwi_align.
    """
    import time
    from nilearn.image import resample_to_img
    from pynets.core.utils import checkConsecutive
    from pynets.registration import utils as regutils
    from nilearn.image import math_img
    from nilearn.masking import intersect_masks

    template_img = nib.load(t1_aligned_mni)
    if uatlas_parcels:
        atlas_img_orig = nib.load(uatlas_parcels)
    else:
        atlas_img_orig = nib.load(uatlas)

    old_count = len(np.unique(np.asarray(atlas_img_orig.dataobj)))

    uatlas_res_template = resample_to_img(
        atlas_img_orig, template_img, interpolation="nearest"
    )

    uatlas_res_template = nib.Nifti1Image(
        np.asarray(uatlas_res_template.dataobj).astype('uint16'),
        affine=uatlas_res_template.affine,
        header=uatlas_res_template.header,
    )
    nib.save(uatlas_res_template, aligned_atlas_t1mni)

    if simple is False:
        try:
            regutils.apply_warp(
                t1w_brain,
                aligned_atlas_t1mni,
                aligned_atlas_skull,
                warp=mni2t1w_warp,
                interp="nn",
                sup=True,
                mask=t1w_brain_mask,
            )
            time.sleep(0.5)

            # Apply linear transformation from template to dwi space
            regutils.applyxfm(ap_path, aligned_atlas_skull, t1w2dwi_xfm,
                              dwi_aligned_atlas, interp="nearestneighbour")
            time.sleep(0.5)
        except BaseException:
            print(
                "Warning: Atlas is not in correct dimensions, or input is low"
                " quality,\nusing linear template registration.")

            regutils.applyxfm(t1w_brain, aligned_atlas_t1mni, mni2t1_xfm,
                              aligned_atlas_skull, interp="nearestneighbour")
            time.sleep(0.5)
            combine_xfms(mni2t1_xfm, t1w2dwi_xfm, mni2dwi_xfm)
            time.sleep(0.5)
            regutils.applyxfm(ap_path, aligned_atlas_t1mni, mni2dwi_xfm,
                              dwi_aligned_atlas, interp="nearestneighbour")
            time.sleep(0.5)
    else:
        regutils.applyxfm(t1w_brain, aligned_atlas_t1mni, mni2t1_xfm,
                          aligned_atlas_skull, interp="nearestneighbour")
        time.sleep(0.5)
        combine_xfms(mni2t1_xfm, t1w2dwi_xfm, mni2dwi_xfm)
        time.sleep(0.5)
        regutils.applyxfm(ap_path, aligned_atlas_t1mni, mni2dwi_xfm,
                          dwi_aligned_atlas, interp="nearestneighbour")
        time.sleep(0.5)

    atlas_img = nib.load(dwi_aligned_atlas)
    wm_gm_img = nib.load(wm_gm_int_in_dwi)
    wm_gm_mask_img = math_img("img > 0", img=wm_gm_img)
    atlas_mask_img = math_img("img > 0", img=atlas_img)

    atlas_img_corr = nib.Nifti1Image(
        np.asarray(atlas_img.dataobj).astype('uint16'),
        affine=atlas_img.affine,
        header=atlas_img.header,
    )

    # Get the union of masks
    dwi_aligned_atlas_wmgm_int_img = intersect_masks(
        [wm_gm_mask_img, atlas_mask_img], threshold=0,
        connected=False
    )

    nib.save(atlas_img_corr, dwi_aligned_atlas)
    nib.save(dwi_aligned_atlas_wmgm_int_img, dwi_aligned_atlas_wmgm_int)

    dwi_aligned_atlas = regutils.apply_mask_to_image(dwi_aligned_atlas,
                                                     B0_mask,
                                                     dwi_aligned_atlas)

    time.sleep(0.5)

    dwi_aligned_atlas_wmgm_int = regutils.apply_mask_to_image(
        dwi_aligned_atlas_wmgm_int, B0_mask, dwi_aligned_atlas_wmgm_int)

    time.sleep(0.5)
    final_dat = atlas_img_corr.get_fdata()
    unique_a = sorted(set(np.array(final_dat.flatten().tolist())))

    if not checkConsecutive(unique_a):
        print("Warning! Non-consecutive integers found in parcellation...")

    new_count = len(unique_a)
    diff = np.abs(np.int(float(new_count) - float(old_count)))
    print(f"Previous label count: {old_count}")
    print(f"New label count: {new_count}")
    print(f"Labels dropped: {diff}")

    atlas_img.uncache()
    atlas_img_corr.uncache()
    atlas_img_orig.uncache()
    atlas_mask_img.uncache()
    wm_gm_img.uncache()
    wm_gm_mask_img.uncache()

    return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_skull


def roi2dwi_align(
    roi,
    t1w_brain,
    roi_in_t1w,
    roi_in_dwi,
    ap_path,
    mni2t1w_warp,
    t1wtissue2dwi_xfm,
    mni2t1_xfm,
    template,
    simple,
):
    """
    A function to perform alignment of a waymask from
    MNI space --> T1w --> dwi.
    """
    import time
    from pynets.registration import utils as regutils
    from nilearn.image import resample_to_img

    roi_img = nib.load(roi)
    template_img = nib.load(template)

    roi_img_res = resample_to_img(
        roi_img, template_img, interpolation="nearest"
    )
    roi_res = f"{roi.split('.nii')[0]}_res.nii.gz"
    nib.save(roi_img_res, roi_res)

    # Apply warp or transformer resulting from the inverse MNI->T1w created
    # earlier
    if simple is False:
        regutils.apply_warp(t1w_brain, roi, roi_in_t1w, warp=mni2t1w_warp)
    else:
        regutils.applyxfm(t1w_brain, roi, mni2t1_xfm, roi_in_t1w)

    time.sleep(0.5)
    # Apply transform from t1w to native dwi space
    regutils.applyxfm(ap_path, roi_in_t1w, t1wtissue2dwi_xfm, roi_in_dwi)

    return roi_in_dwi


def waymask2dwi_align(
    waymask,
    t1w_brain,
    ap_path,
    mni2t1w_warp,
    mni2t1_xfm,
    t1wtissue2dwi_xfm,
    waymask_in_t1w,
    waymask_in_dwi,
    B0_mask_tmp_path,
    template,
    simple,
):
    """
    A function to perform alignment of a waymask from
    MNI space --> T1w --> dwi.
    """
    import time
    from pynets.registration import utils as regutils
    from nilearn.image import resample_to_img

    # Apply warp or transformer resulting from the inverse MNI->T1w created
    # earlier
    waymask_img = nib.load(waymask)
    template_img = nib.load(template)

    waymask_img_res = resample_to_img(
        waymask_img, template_img,
    )
    waymask_res = f"{waymask.split('.nii')[0]}_res.nii.gz"
    nib.save(waymask_img_res, waymask_res)

    if simple is False:
        regutils.apply_warp(
            t1w_brain,
            waymask_res,
            waymask_in_t1w,
            warp=mni2t1w_warp)
    else:
        regutils.applyxfm(t1w_brain, waymask_res, mni2t1_xfm, waymask_in_t1w)

    time.sleep(0.5)
    # Apply transform from t1w to native dwi space
    regutils.applyxfm(
        ap_path,
        waymask_in_t1w,
        t1wtissue2dwi_xfm,
        waymask_in_dwi)

    time.sleep(0.5)

    waymask_in_dwi = regutils.apply_mask_to_image(waymask_in_dwi,
                                                  B0_mask_tmp_path,
                                                  waymask_in_dwi)

    return waymask_in_dwi


def roi2t1w_align(
        roi,
        t1w_brain,
        mni2t1_xfm,
        mni2t1w_warp,
        roi_in_t1w,
        template,
        simple):
    """
    A function to perform alignment of a roi from MNI space --> T1w.
    """
    import time
    from pynets.registration import utils as regutils
    from nilearn.image import resample_to_img

    roi_img = nib.load(roi)
    template_img = nib.load(template)

    roi_img_res = resample_to_img(
        roi_img, template_img, interpolation="nearest"
    )
    roi_res = f"{roi.split('.nii')[0]}_res.nii.gz"
    nib.save(roi_img_res, roi_res)

    # Apply warp or transformer resulting from the inverse MNI->T1w created
    # earlier
    if simple is False:
        regutils.apply_warp(t1w_brain, roi_res, roi_in_t1w, warp=mni2t1w_warp)
    else:
        regutils.applyxfm(t1w_brain, roi_res, mni2t1_xfm, roi_in_t1w)

    time.sleep(0.5)

    return roi_in_t1w


def RegisterParcellation2MNIFunc_align(
    uatlas,
    template,
    template_mask,
    t1w_brain,
    t1w2mni_xfm,
    aligned_atlas_t1w,
    aligned_atlas_mni,
    t1w2mni_warp,
    simple
):
    """
    A function to perform atlas alignment from T1w atlas --> MNI.
    """
    import time
    from pynets.registration import utils as regutils
    from nilearn.image import resample_to_img

    atlas_img = nib.load(uatlas)
    t1w_brain_img = nib.load(t1w_brain)

    uatlas_res_template = resample_to_img(
        atlas_img, t1w_brain_img, interpolation="nearest"
    )

    uatlas_res_template = nib.Nifti1Image(
        np.asarray(uatlas_res_template.dataobj).astype('uint16'),
        affine=uatlas_res_template.affine,
        header=uatlas_res_template.header,
    )
    nib.save(uatlas_res_template, aligned_atlas_t1w)

    if simple is False:
        try:
            regutils.apply_warp(
                template,
                aligned_atlas_t1w,
                aligned_atlas_mni,
                warp=t1w2mni_warp,
                interp="nn",
                sup=True,
            )
            time.sleep(0.5)
        except BaseException:
            print(
                "Warning: Atlas is not in correct dimensions, or input is "
                "low quality,\nusing linear template registration.")

            regutils.align(
                aligned_atlas_t1w,
                template,
                init=t1w2mni_xfm,
                out=aligned_atlas_mni,
                dof=6,
                searchrad=True,
                interp="nearestneighbour",
                cost="mutualinfo",
            )
            time.sleep(0.5)
    else:
        regutils.align(
            aligned_atlas_t1w,
            template,
            init=t1w2mni_xfm,
            out=aligned_atlas_mni,
            dof=6,
            searchrad=True,
            interp="nearestneighbour",
            cost="mutualinfo",
        )
        time.sleep(0.5)
    return aligned_atlas_mni


def atlas2t1w_align(
    uatlas,
    uatlas_parcels,
    atlas,
    t1w_brain,
    t1w_brain_mask,
    t1_aligned_mni,
    mni2t1w_warp,
    mni2t1_xfm,
    gm_mask,
    aligned_atlas_t1mni,
    aligned_atlas_skull,
    aligned_atlas_gm,
    simple,
    gm_fail_tol=5
):
    """
    A function to perform atlas alignment from atlas --> T1w.
    """
    import time
    from pynets.registration import utils as regutils
    from nilearn.image import resample_to_img
    # from pynets.core.utils import checkConsecutive

    template_img = nib.load(t1_aligned_mni)
    if uatlas_parcels:
        atlas_img_orig = nib.load(uatlas_parcels)
    else:
        atlas_img_orig = nib.load(uatlas)

    # old_count = len(np.unique(np.asarray(atlas_img_orig.dataobj)))

    uatlas_res_template = resample_to_img(
        atlas_img_orig, template_img, interpolation="nearest"
    )

    uatlas_res_template = nib.Nifti1Image(
        np.asarray(uatlas_res_template.dataobj).astype('uint16'),
        affine=uatlas_res_template.affine,
        header=uatlas_res_template.header,
    )
    nib.save(uatlas_res_template, aligned_atlas_t1mni)

    if simple is False:
        try:
            regutils.apply_warp(
                t1w_brain,
                aligned_atlas_t1mni,
                aligned_atlas_skull,
                warp=mni2t1w_warp,
                interp="nn",
                sup=True,
                mask=t1w_brain_mask,
            )
            time.sleep(0.5)
        except BaseException:
            print(
                "Warning: Atlas is not in correct dimensions, or input is low "
                "quality,\nusing linear template registration.")

            regutils.applyxfm(t1w_brain, aligned_atlas_t1mni, mni2t1_xfm,
                              aligned_atlas_skull, interp="nearestneighbour")
            time.sleep(0.5)
    else:
        regutils.applyxfm(t1w_brain, aligned_atlas_t1mni, mni2t1_xfm,
                          aligned_atlas_skull, interp="nearestneighbour")
        time.sleep(0.5)

    # aligned_atlas_gm = regutils.apply_mask_to_image(aligned_atlas_skull,
    #                                                 gm_mask,
    #                                                 aligned_atlas_gm)
    aligned_atlas_gm = regutils.apply_mask_to_image(aligned_atlas_skull,
                                                    t1w_brain_mask,
                                                    aligned_atlas_gm)

    time.sleep(0.5)
    atlas_img = nib.load(aligned_atlas_gm)

    atlas_img_corr = nib.Nifti1Image(
        np.asarray(atlas_img.dataobj).astype('uint16'),
        affine=atlas_img.affine,
        header=atlas_img.header,
    )
    nib.save(atlas_img_corr, aligned_atlas_gm)
    # final_dat = atlas_img_corr.get_fdata()
    # unique_a = sorted(set(np.array(final_dat.flatten().tolist())))
    #
    # if not checkConsecutive(unique_a):
    #     print("\nWarning! non-consecutive integers found in parcellation...")
    # new_count = len(unique_a)
    # diff = np.abs(np.int(float(new_count) - float(old_count)))
    # print(f"Previous label count: {old_count}")
    # print(f"New label count: {new_count}")
    # print(f"Labels dropped: {diff}")
    # if diff > gm_fail_tol:
    #     print(f"Grey-Matter mask too restrictive >{str(gm_fail_tol)} for this "
    #           f"parcellation. Falling back to the T1w mask...")
    #     aligned_atlas_gm = regutils.apply_mask_to_image(aligned_atlas_skull,
    #                                                     t1w_brain_mask,
    #                                                     aligned_atlas_gm)
    #     time.sleep(5)
    template_img.uncache()
    atlas_img_orig.uncache()
    atlas_img.uncache()
    atlas_img_corr.uncache()

    return aligned_atlas_gm, aligned_atlas_skull


def segment_t1w(t1w, basename, opts=""):
    """
    A function to use FSL's FAST to segment an anatomical
    image into GM, WM, and CSF prob maps.

    Parameters
    ----------
    t1w : str
        File path to an anatomical T1-weighted image.
    basename : str
        A basename to use for output files.
    opts : str
        Additional options that can optionally be passed to fast.
        Desirable options might be -P, which will use prior probability maps
        if the input T1w MRI is in standard space.

    Returns
    -------
    out : str
        File path to the probability map Nifti1Image consisting of GM, WM,
        and CSF in the 4th dimension.
    """
    print("Segmenting Anatomical Image into WM, GM, and CSF...")
    # run FAST, with options -t for the image type and -n to
    # segment into CSF (pve_0), GM (pve_1), WM (pve_2)
    cmd = f"fast -t 1 {opts} -n 3 -o {basename} {t1w}"
    os.system(cmd)
    out = {}  # the outputs
    out["wm_prob"] = f"{basename}_{'pve_2.nii.gz'}"
    out["gm_prob"] = f"{basename}_{'pve_1.nii.gz'}"
    out["csf_prob"] = f"{basename}_{'pve_0.nii.gz'}"
    return out


def align(
    inp,
    ref,
    xfm=None,
    out=None,
    dof=12,
    searchrad=True,
    bins=256,
    interp=None,
    cost="mutualinfo",
    sch=None,
    wmseg=None,
    init=None,
):
    """
    Aligns two images using linear registration (FSL's FLIRT).

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for
            alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration
            alignment.
        dof : int
            Number of degrees of freedom to use in the alignment.
        searchrad : bool
            Indicating whether to use the predefined searchradius parameter
            (180 degree sweep in x, y, and z). Default is True.
        bins : int
            Number of histogram bins. Default is 256.
        interp : str
            Interpolation method to use. Default is mutualinfo.
        sch : str
            Optional file path to a FLIRT schedule file.
        wmseg : str
            Optional file path to white-matter segmentation Nifti1Image for
            boundary-based registration (BBR).
        init : str
            File path to a transformation matrix in .xfm format to use as an
            initial guess for the alignment.

    """
    cmd = f"flirt -in {inp} -ref {ref}"
    if xfm is not None:
        cmd += f" -omat {xfm}"
    if out is not None:
        cmd += f" -out {out}"
    if dof is not None:
        cmd += f" -dof {dof}"
    if bins is not None:
        cmd += f" -bins {bins}"
    if interp is not None:
        cmd += f" -interp {interp}"
    if cost is not None:
        cmd += f" -cost {cost}"
    if searchrad is not None:
        cmd += " -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    if sch is not None:
        cmd += f" -schedule {sch}"
    if wmseg is not None:
        cmd += f" -wmseg {wmseg}"
    if init is not None:
        cmd += f" -init {init}"
    print(cmd)
    os.system(cmd)
    return


def align_nonlinear(
        inp,
        ref,
        xfm,
        out,
        warp,
        ref_mask=None,
        in_mask=None,
        config=None):
    """
    Aligns two images using nonlinear registration and stores the transform
    between them.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for
            alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration
            alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp
            following alignment.
        ref_mask : str
            Optional file path to a mask in reference image space.
        in_mask : str
            Optional file path to a mask in input image space.
        config : str
            Optional file path to config file specifying command line
            arguments.

    """
    cmd = f"fnirt --in={inp} --ref={ref} --aff={xfm} --iout={out} " \
          f"--cout={warp} --warpres=8,8,8"
    if ref_mask is not None:
        cmd += f" --refmask={ref_mask}"
    if in_mask is not None:
        cmd += f" --inmask={in_mask}"
    if config is not None:
        cmd += f" --config={config}"
    print(cmd)
    os.system(cmd)
    return


def applyxfm(ref, inp, xfm, aligned, interp="trilinear", dof=6):
    """
    Aligns two images with a given transform.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for
            alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        aligned : str
            File path to input Nifti1Image output following registration
            alignment.
        interp : str
            Interpolation method to use. Default is trilinear.
        dof : int
            Number of degrees of freedom to use in the alignment.

    """
    cmd = f"flirt -in {inp} -ref {ref} -out {aligned} -init {xfm} -interp" \
          f" {interp} -dof {dof} -applyxfm"
    print(cmd)
    os.system(cmd)
    return


def apply_warp(
        ref,
        inp,
        out,
        warp=None,
        xfm=None,
        mask=None,
        interp=None,
        sup=False):
    """
    Applies a warp to a Nifti1Image which transforms the image to the
    reference space used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for
            alignment.
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        out : str
            File path to input Nifti1Image output following registration
            alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp
            following alignment.
        xfm : str
            File path for the transformation matrix input in .xfm.
        mask : str
            Optional file path to a mask in reference image space.
        interp : str
            Interpolation method to use.
        sup : bool
            Intermediary supersampling of output. Default is False.

    """
    cmd = f"applywarp --ref={ref} --in={inp} --out={out}"
    if xfm is not None:
        cmd += f" --premat={xfm}"
    if warp is not None:
        cmd += f" --warp={warp}"
    if mask is not None:
        cmd += f" --mask={mask}"
    if interp is not None:
        cmd += f" --interp={interp}"
    if sup is True:
        cmd += " --super --superlevel=a"
    print(cmd)
    os.system(cmd)
    return


def inverse_warp(ref, out, warp):
    """
    Generates the inverse of a warp from a reference image space to the input
    image used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for
            alignment.
        out : str
            File path to input Nifti1Image output following registration
            alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp
            following alignment.

    """
    cmd = f"invwarp --warp={warp} --out={out} --ref={ref}"
    print(cmd)
    os.system(cmd)
    return


def combine_xfms(xfm1, xfm2, xfmout):
    """
    A function to combine two transformations, and output the resulting
    transformation.

    Parameters
    ----------
        xfm1 : str
            File path to the first transformation.
        xfm2 : str
            File path to the second transformation.
        xfmout : str
            File path to the output transformation.

    """
    cmd = f"convert_xfm -omat {xfmout} -concat {xfm1} {xfm2}"
    print(cmd)
    os.system(cmd)
    return


def invert_xfm(in_mat, out_mat):
    import os
    cmd = f"convert_xfm -omat {out_mat} -inverse {in_mat}"
    print(cmd)
    os.system(cmd)
    return out_mat


def apply_mask_to_image(input, mask, output):
    import os

    cmd = f"fslmaths {input} -mas {mask} -thrp 0.0001 {output}"
    print(cmd)
    os.system(cmd)

    return output


def get_wm_contour(wm_map, mask, wm_edge):
    import os
    cmd = f"fslmaths {wm_map} -edge -bin -mas {mask} {wm_edge}"
    print(cmd)
    os.system(cmd)
    return wm_edge


def vdc(n, vox_size):
    vdc, denom = 0, 1
    while n:
        denom *= vox_size
        n, remainder = divmod(n, vox_size)
        vdc += remainder / denom
    return vdc


def warp_streamlines(
    adjusted_affine,
    ref_grid_aff,
    mapping,
    warped_fa_img,
    streams_in_curr_grid,
    brain_mask,
):
    from dipy.tracking import utils
    from dipy.tracking.streamline import (
        values_from_volume,
        transform_streamlines,
        Streamlines,
    )

    # Deform streamlines, isocenter, and remove streamlines outside brain
    streams_in_brain = [
        sum(d, s)
        for d, s in zip(
            values_from_volume(
                mapping.get_forward_field(), streams_in_curr_grid, ref_grid_aff
            ),
            streams_in_curr_grid,
        )
    ]
    streams_final_filt = Streamlines(
        utils.target_line_based(
            transform_streamlines(
                transform_streamlines(
                    streams_in_brain,
                    np.linalg.inv(adjusted_affine)),
                np.linalg.inv(
                    warped_fa_img.affine),
            ),
            np.eye(4),
            brain_mask,
            include=True,
        ))

    return streams_final_filt


def rescale_affine_to_center(input_affine, voxel_dims=[1, 1, 1],
                             target_center_coords=None):
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using
        the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine.copy()
    # Decompose the image affine to allow scaling
    u, s, v = np.linalg.svd(target_affine[:3, :3], full_matrices=False)

    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims

    # Reconstruct the affine
    target_affine[:3, :3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3, 3] = target_center_coords
    return target_affine


def wm_syn(t1w_brain, ap_path, working_dir, fa_path=None,
           template_fa_path=None):
    """
    A function to perform SyN registration

    Parameters
    ----------
        t1w_brain  : str
            File path to the skull-stripped T1w brain Nifti1Image.
        ap_path : str
            File path to the AP moving image.
        working_dir : str
            Path to the working directory to perform SyN and save outputs.
        fa_path : str
            File path to the FA moving image.
        template_fa_path  : str
            File path to the T1w-connformed template FA reference image.
    """
    import uuid
    from time import strftime
    from dipy.align.imaffine import (
        MutualInformationMetric,
        AffineRegistration,
        transform_origins,
    )
    from dipy.align.transforms import (
        TranslationTransform3D,
        RigidTransform3D,
        AffineTransform3D,
    )
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric

    # from dipy.viz import regtools
    # from nilearn.image import resample_to_img

    ap_img = nib.load(ap_path)
    t1w_brain_img = nib.load(t1w_brain)
    static = np.asarray(t1w_brain_img.dataobj, dtype=np.float32)
    static_affine = t1w_brain_img.affine
    moving = np.asarray(ap_img.dataobj, dtype=np.float32)
    moving_affine = ap_img.affine

    affine_map = transform_origins(
        static, static_affine, moving, moving_affine)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affine_reg = AffineRegistration(
        metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors
    )
    transform = TranslationTransform3D()

    params0 = None
    translation = affine_reg.optimize(
        static, moving, transform, params0, static_affine, moving_affine
    )
    transform = RigidTransform3D()

    rigid_map = affine_reg.optimize(
        static,
        moving,
        transform,
        params0,
        static_affine,
        moving_affine,
        starting_affine=translation.affine,
    )
    transform = AffineTransform3D()

    # We bump up the iterations to get a more exact fit:
    affine_reg.level_iters = [1000, 1000, 100]
    affine_opt = affine_reg.optimize(
        static,
        moving,
        transform,
        params0,
        static_affine,
        moving_affine,
        starting_affine=rigid_map.affine,
    )

    # We now perform the non-rigid deformation using the Symmetric
    # Diffeomorphic Registration(SyN) Algorithm:
    metric = CCMetric(3)
    level_iters = [10, 10, 5]

    # Refine fit
    if template_fa_path is not None:
        from nilearn.image import resample_to_img
        fa_img = nib.load(fa_path)
        template_img = nib.load(template_fa_path)
        template_img_res = resample_to_img(template_img, t1w_brain_img)
        static = np.asarray(template_img_res.dataobj, dtype=np.float32)
        static_affine = template_img_res.affine
        moving = np.asarray(fa_img.dataobj, dtype=np.float32)
        moving_affine = fa_img.affine
    else:
        static = np.asarray(t1w_brain_img.dataobj, dtype=np.float32)
        static_affine = t1w_brain_img.affine
        moving = np.asarray(ap_img.dataobj, dtype=np.float32)
        moving_affine = ap_img.affine

    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(
        static, moving, static_affine, moving_affine, affine_opt.affine
    )
    warped_moving = mapping.transform(moving)

    # Save warped FA image
    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    warped_fa = f"{working_dir}/warped_fa_{run_uuid}.nii.gz"
    nib.save(
        nib.Nifti1Image(
            warped_moving,
            affine=static_affine),
        warped_fa)

    # # We show the registration result with:
    # regtools.overlay_slices(static, warped_moving, None, 0,
    # "Static", "Moving",
    #                         "%s%s%s%s" % (working_dir,
    #                         "/transformed_sagittal_", run_uuid, ".png"))
    # regtools.overlay_slices(static, warped_moving, None,
    # 1, "Static", "Moving",
    #                         "%s%s%s%s" % (working_dir,
    #                         "/transformed_coronal_", run_uuid, ".png"))
    # regtools.overlay_slices(static, warped_moving,
    # None, 2, "Static", "Moving",
    #                         "%s%s%s%s" % (working_dir,
    #                         "/transformed_axial_", run_uuid, ".png"))

    return mapping, affine_map, warped_fa


def median(in_file):
    """Average a 4D dataset across the last dimension using median."""
    out_file = fname_presuffix(in_file, suffix="_mean.nii.gz", use_ext=False)

    img = nib.load(in_file)
    if img.dataobj.ndim == 3:
        return in_file
    if img.shape[-1] == 1:
        nib.squeeze_image(img).to_filename(out_file)
        return out_file

    median_data = np.median(img.get_fdata(dtype="float32"), axis=-1)

    hdr = img.header.copy()
    hdr.set_xyzt_units("mm")
    hdr.set_data_dtype(np.float32)
    nib.Nifti1Image(median_data, img.affine, hdr).to_filename(out_file)
    return out_file


def check_orient_and_dims(
        infile,
        outdir,
        vox_size,
        bvecs=None,
        overwrite=True):
    """
    An API to reorient any image to RAS+ and resample any image to a given
    voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a Nifti1Image.
    outdir : str
        Path to base derivatives directory.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.
    overwrite : bool
        Boolean indicating whether to overwrite existing outputs. Default is
        True.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.

    """
    from pynets.registration.utils import (
        reorient_dwi,
        reorient_img,
        match_target_vox_res,
    )
    import time

    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            [infile, bvecs] = reorient_dwi(
                infile, bvecs, outdir, overwrite=overwrite)
            time.sleep(0.5)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.5)
            print(outfile)
        else:
            outfile = infile
    elif (vols > 1) and (bvecs is None):
        # func case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir, overwrite=overwrite)
            time.sleep(0.5)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.5)
            print(outfile)
        else:
            outfile = infile
    else:
        # t1w case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir, overwrite=overwrite)
            time.sleep(0.5)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.5)
            print(outfile)
        else:
            outfile = infile

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs


def normalize_xform(img):
    """ Set identical, valid qform and sform matrices in an image
    Selects the best available affine (sform > qform > shape-based), and
    coerces it to be qform-compatible (no shears).
    The resulting image represents this same affine as both qform and sform,
    and is marked as NIFTI_XFORM_ALIGNED_ANAT, indicating that it is valid,
    not aligned to template, and not necessarily preserving the original
    coordinates.
    If header would be unchanged, returns input image.
    """
    # Let nibabel convert from affine to quaternions, and recover xform
    tmp_header = img.header.copy()
    tmp_header.set_qform(img.affine)
    xform = tmp_header.get_qform()
    xform_code = 2

    # Check desired codes
    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    if all(
        (
            qform is not None and np.allclose(qform, xform),
            sform is not None and np.allclose(sform, xform),
            int(qform_code) == xform_code,
            int(sform_code) == xform_code,
        )
    ):
        return img

    new_img = img.__class__(
        np.asarray(
            img.dataobj),
        xform,
        img.header)

    # Unconditionally set sform/qform
    new_img.set_sform(xform, xform_code)
    new_img.set_qform(xform, xform_code)

    return new_img


def reorient_dwi(dwi_prep, bvecs, out_dir, overwrite=True):
    """
    A function to reorient any dwi image and associated bvecs to RAS+.

    Parameters
    ----------
    dwi_prep : str
        File path to a dwi Nifti1Image.
    bvecs : str
        File path to corresponding bvecs file.
    out_dir : str
        Path to output directory.

    Returns
    -------
    out_fname : str
        File path to the reoriented dwi Nifti1Image.
    out_bvec_fname : str
        File path to corresponding reoriented bvecs file.

    """
    import os
    from pynets.registration.utils import normalize_xform

    fname = dwi_prep
    bvec_fname = bvecs

    out_bvec_fname = (
        f"{out_dir}/{dwi_prep.split('/')[-1].split('.nii')[0]}_bvecs_reor.bvec"
    )

    input_img = nib.load(fname)
    input_axcodes = nib.aff2axcodes(input_img.affine)
    reoriented = nib.as_closest_canonical(input_img)
    normalized = normalize_xform(reoriented)
    # Is the input image oriented how we want?
    new_axcodes = ("R", "A", "S")
    if normalized is not input_img:
        out_fname = (
            f"{out_dir}/{dwi_prep.split('/')[-1].split('.nii')[0]}_"
            f"reor-RAS.nii.gz"
        )
        if (
            overwrite is False
            and os.path.isfile(out_fname)
            and os.path.isfile(out_bvec_fname)
        ):
            pass
        else:
            print(f"Reorienting {dwi_prep} to RAS+...")

            # Flip the bvecs
            transform_orientation = nib.orientations.ornt_transform(
                nib.orientations.axcodes2ornt(input_axcodes),
                nib.orientations.axcodes2ornt(new_axcodes),
            )
            bvec_array = np.genfromtxt(bvec_fname)
            if bvec_array.shape[0] != 3:
                bvec_array = bvec_array.T
            if not bvec_array.shape[0] == transform_orientation.shape[0]:
                raise ValueError("Unrecognized bvec format")

            output_array = np.zeros_like(bvec_array)
            for this_axnum, (axnum, flip) in enumerate(transform_orientation):
                output_array[this_axnum] = bvec_array[int(axnum)] * float(flip)
            np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")
    else:
        out_fname = (
            f"{out_dir}/{dwi_prep.split('/')[-1].split('.nii')[0]}_"
            f"noreor-RAS.nii.gz"
        )
        out_bvec_fname = bvec_fname

    if (
        overwrite is False
        and os.path.isfile(out_fname)
        and os.path.isfile(out_bvec_fname)
    ):
        pass
    else:
        normalized.to_filename(out_fname)
        normalized.uncache()
        input_img.uncache()
        del normalized, input_img
    return out_fname, out_bvec_fname


def reorient_img(img, out_dir, overwrite=True):
    """
    A function to reorient any non-dwi image to RAS+.

    Parameters
    ----------
    img : str
        File path to a Nifti1Image.
    out_dir : str
        Path to output directory.

    Returns
    -------
    out_name : str
        File path to reoriented Nifti1Image.

    """
    from pynets.registration.utils import normalize_xform

    # Load image, orient as RAS
    orig_img = nib.load(img)
    normalized = normalize_xform(nib.as_closest_canonical(orig_img))

    # Image may be reoriented
    if normalized is not orig_img:
        print(f"{'Reorienting '}{img}{' to RAS+...'}")
        out_name = (
            f"{out_dir}/{img.split('/')[-1].split('.nii')[0]}_"
            f"reor-RAS.nii.gz"
        )
    else:
        out_name = (
            f"{out_dir}/{img.split('/')[-1].split('.nii')[0]}_"
            f"noreor-RAS.nii.gz"
        )

    if overwrite is False and os.path.isfile(out_name):
        pass
    else:
        normalized.to_filename(out_name)
        orig_img.uncache()
        normalized.uncache()
        del orig_img

    return out_name


def match_target_vox_res(img_file, vox_size, out_dir, overwrite=True,
                         remove_orig=True):
    """
    A function to resample an image to a given isotropic voxel resolution.

    Parameters
    ----------
    img_file : str
        File path to a Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    out_dir : str
        Path to output directory.

    Returns
    -------
    img_file : str
        File path to resampled Nifti1Image.

    """
    import os
    from dipy.align.reslice import reslice

    # Check dimensions
    orig_img = img_file
    img = nib.load(img_file, mmap=False)

    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    if vox_size == "1mm":
        new_zooms = (1.0, 1.0, 1.0)
    elif vox_size == "2mm":
        new_zooms = (2.0, 2.0, 2.0)

    if (abs(zooms[0]), abs(zooms[1]), abs(zooms[2])) != new_zooms:
        img_file_res = (
            f"{out_dir}/{os.path.basename(img_file).split('.nii')[0]}_"
            f"res-{vox_size}.nii.gz"
        )
        if overwrite is False and os.path.isfile(img_file_res):
            img_file = img_file_res
            pass
        else:
            import gc
            data = img.get_fdata(dtype=np.float32)
            print(f"Reslicing image {img_file} to {vox_size}...")
            data2, affine2 = reslice(
                data, img.affine, zooms, new_zooms
            )
            nib.save(
                nib.Nifti1Image(
                    data2,
                    affine=affine2),
                img_file_res)
            img_file = img_file_res
            del data2, data
            gc.collect()
    else:
        img_file_nores = (
            f"{out_dir}/{os.path.basename(img_file).split('.nii')[0]}_"
            f"nores-{vox_size}"
            f".nii.gz")
        if overwrite is False and os.path.isfile(img_file_nores):
            img_file = img_file_nores
            pass
        else:
            nib.save(img, img_file_nores)
            img_file = img_file_nores

    if os.path.isfile(orig_img) and remove_orig is True:
        os.remove(orig_img)

    return img_file
