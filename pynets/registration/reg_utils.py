#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import numpy as np
import indexed_gzip
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

try:
    FSLDIR = os.environ['FSLDIR']
except KeyError:
    print('FSLDIR environment variable not set!')


def segment_t1w(t1w, basename, opts=''):
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
        Desirable options might be -P, which will use prior probability maps if the input T1w MRI is in standard space.

    Returns
    -------
    out : str
        File path to the probability map Nifti1Image consisting of GM, WM, and CSF in the 4th dimension.
    """
    print("Segmenting Anatomical Image into WM, GM, and CSF...")
    # run FAST, with options -t for the image type and -n to
    # segment into CSF (pve_0), GM (pve_1), WM (pve_2)
    cmd = "fast -t 1 {} -n 3 -o {} {}".format(opts, basename, t1w)
    os.system(cmd)
    out = {}  # the outputs
    out['wm_prob'] = "{}_{}".format(basename, "pve_2.nii.gz")
    out['gm_prob'] = "{}_{}".format(basename, "pve_1.nii.gz")
    out['csf_prob'] = "{}_{}".format(basename, "pve_0.nii.gz")
    return out


def align(inp, ref, xfm=None, out=None, dof=12, searchrad=True, bins=256, interp=None, cost="mutualinfo", sch=None,
          wmseg=None, init=None):
    """
    Aligns two images using linear registration (FSL's FLIRT).

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration alignment.
        dof : int
            Number of degrees of freedom to use in the alignment.
        searchrad : bool
            Indicating whether to use the predefined searchradius parameter (180 degree sweep in x, y, and z).
            Default is True.
        bins : int
            Number of histogram bins. Default is 256.
        interp : str
            Interpolation method to use. Default is mutualinfo.
        sch : str
            Optional file path to a FLIRT schedule file.
        wmseg : str
            Optional file path to white-matter segmentation Nifti1Image for boundary-based registration (BBR).
        init : str
            File path to a transformation matrix in .xfm format to use as an initial guess for the alignment.
    """
    cmd = "flirt -in {} -ref {}".format(inp, ref)
    if xfm is not None:
        cmd += " -omat {}".format(xfm)
    if out is not None:
        cmd += " -out {}".format(out)
    if dof is not None:
        cmd += " -dof {}".format(dof)
    if bins is not None:
        cmd += " -bins {}".format(bins)
    if interp is not None:
        cmd += " -interp {}".format(interp)
    if cost is not None:
        cmd += " -cost {}".format(cost)
    if searchrad is not None:
        cmd += " -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    if sch is not None:
        cmd += " -schedule {}".format(sch)
    if wmseg is not None:
        cmd += " -wmseg {}".format(wmseg)
    if init is not None:
        cmd += " -init {}".format(init)
    print(cmd)
    os.system(cmd)
    return


def align_nonlinear(inp, ref, xfm, out, warp, ref_mask=None, in_mask=None, config=None):
    """
    Aligns two images using nonlinear registration and stores the transform between them.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
        ref_mask : str
            Optional file path to a mask in reference image space.
        in_mask : str
            Optional file path to a mask in input image space.
        config : str
            Optional file path to config file specifying command line arguments.
    """
    cmd = "fnirt --in={} --ref={} --aff={} --iout={} --cout={} --warpres=8,8,8"
    cmd = cmd.format(inp, ref, xfm, out, warp, config)
    if ref_mask is not None:
        cmd += " --refmask={}".format(ref_mask)
    if in_mask is not None:
        cmd += " --inmask={}".format(in_mask)
    if config is not None:
        cmd += " --config={}".format(config)
    print(cmd)
    os.system(cmd)
    return


def applyxfm(ref, inp, xfm, aligned, interp='trilinear', dof=6):
    """
    Aligns two images with a given transform.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        aligned : str
            File path to input Nifti1Image output following registration alignment.
        interp : str
            Interpolation method to use. Default is trilinear.
        dof : int
            Number of degrees of freedom to use in the alignment.
    """
    cmd = "flirt -in {} -ref {} -out {} -init {} -interp {} -dof {} -applyxfm".format(inp, ref, aligned, xfm, interp,
                                                                                      dof)
    print(cmd)
    os.system(cmd)
    return


def apply_warp(ref, inp, out, warp, xfm=None, mask=None, interp=None, sup=False):
    """
    Applies a warp to a Nifti1Image which transforms the image to the reference space used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        mask : str
            Optional file path to a mask in reference image space.
        interp : str
            Interpolation method to use.
        sup : bool
            Intermediary supersampling of output. Default is False.
    """
    cmd = "applywarp --ref={} --in={} --out={} --warp={}".format(ref, inp, out, warp)
    if xfm is not None:
        cmd += " --premat={}".format(xfm)
    if mask is not None:
        cmd += " --mask={}".format(mask)
    if interp is not None:
        cmd += " --interp={}".format(interp)
    if sup is True:
        cmd += " --super --superlevel=a"
    print(cmd)
    os.system(cmd)
    return


def inverse_warp(ref, out, warp):
    """
    Generates the inverse of a warp from a reference image space to the input image space.
    space used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
    """
    cmd = "invwarp --warp=" + warp + " --out=" + out + " --ref=" + ref
    print(cmd)
    os.system(cmd)
    return


def combine_xfms(xfm1, xfm2, xfmout):
    """
    A function to combine two transformations, and output the resulting transformation.

    Parameters
    ----------
        xfm1 : str
            File path to the first transformation.
        xfm2 : str
            File path to the second transformation.
        xfmout : str
            File path to the output transformation.
    """
    cmd = "convert_xfm -omat {} -concat {} {}".format(xfmout, xfm1, xfm2)
    print(cmd)
    os.system(cmd)
    return


def wm_syn(template_path, fa_path, working_dir):
    """
    A function to perform ANTS SyN registration

    Parameters
    ----------
        template_path  : str
            File path to the template reference image.
        fa_path : str
            File path to the FA moving image.
        working_dir : str
            Path to the working directory to perform SyN and save outputs.
    """
    import uuid
    from time import strftime
    from dipy.align.imaffine import MutualInformationMetric, AffineRegistration, transform_origins
    from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    from dipy.viz import regtools

    fa_img = nib.load(fa_path)
    template_img = nib.load(template_path)

    static = np.asarray(template_img.dataobj)
    static_affine = template_img.affine
    moving = np.asarray(fa_img.dataobj).astype(np.float32)
    moving_affine = fa_img.affine

    affine_map = transform_origins(static, static_affine, moving, moving_affine)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affine_reg = AffineRegistration(metric=metric, level_iters=level_iters,
                                    sigmas=sigmas, factors=factors)
    transform = TranslationTransform3D()

    params0 = None
    translation = affine_reg.optimize(static, moving, transform, params0,
                                      static_affine, moving_affine)
    transform = RigidTransform3D()

    rigid_map = affine_reg.optimize(static, moving, transform, params0,
                                    static_affine, moving_affine,
                                    starting_affine=translation.affine)
    transform = AffineTransform3D()

    # We bump up the iterations to get a more exact fit:
    affine_reg.level_iters = [1000, 1000, 100]
    affine_opt = affine_reg.optimize(static, moving, transform, params0,
                                     static_affine, moving_affine,
                                     starting_affine=rigid_map.affine)

    # We now perform the non-rigid deformation using the Symmetric Diffeomorphic Registration(SyN) Algorithm:
    metric = CCMetric(3)
    level_iters = [10, 10, 5]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                           affine_opt.affine)
    warped_moving = mapping.transform(moving)

    # Save warped FA image
    run_uuid = '%s_%s' % (strftime('%Y%m%d_%H%M%S'), uuid.uuid4())
    warped_fa = '{}/warped_fa_{}.nii.gz'.format(working_dir, run_uuid)
    nib.save(nib.Nifti1Image(warped_moving, affine=static_affine), warped_fa)

    # We show the registration result with:
    regtools.overlay_slices(static, warped_moving, None, 0, "Static", "Moving",
                            "%s%s%s%s" % (working_dir, "/transformed_sagittal_", run_uuid, ".png"))
    regtools.overlay_slices(static, warped_moving, None, 1, "Static", "Moving",
                            "%s%s%s%s" % (working_dir, "/transformed_coronal_", run_uuid, ".png"))
    regtools.overlay_slices(static, warped_moving, None, 2, "Static", "Moving",
                            "%s%s%s%s" % (working_dir, "/transformed_axial_", run_uuid, ".png"))

    return mapping, affine_map, warped_fa


def check_orient_and_dims(infile, vox_size, bvecs=None, overwrite=True):
    """
    An API to reorient any image to RAS+ and resample any image to a given voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a dwi Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.
    overwrite : bool
        Boolean indicating whether to overwrite existing outputs. Default is True.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    elif (vols > 1) and (bvecs is None):
        # func case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    else:
        # t1w case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)

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
    if all((qform is not None and np.allclose(qform, xform),
            sform is not None and np.allclose(sform, xform),
            int(qform_code) == xform_code, int(sform_code) == xform_code)):
        return img

    new_img = img.__class__(np.asarray(img.dataobj), xform, img.header)
    # Unconditionally set sform/qform
    new_img.set_sform(xform, xform_code)
    new_img.set_qform(xform, xform_code)

    return new_img


def reorient_dwi(dwi_prep, bvecs, out_dir):
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
    from pynets.registration.reg_utils import normalize_xform
    fname = dwi_prep
    bvec_fname = bvecs
    out_bvec_fname = "%s%s" % (out_dir, '/bvecs_reor.bvec')

    input_img = nib.load(fname)
    input_axcodes = nib.aff2axcodes(input_img.affine)
    reoriented = nib.as_closest_canonical(input_img)
    normalized = normalize_xform(reoriented)
    # Is the input image oriented how we want?
    new_axcodes = ('R', 'A', 'S')
    if normalized is not input_img:
        out_fname = "%s%s%s%s%s" % (out_dir, '/', dwi_prep.split('/')[-1].split('.nii')[0], '_reor-RAS.nii',
                                    dwi_prep.split('/')[-1].split('.nii')[1])
        print("%s%s%s" % ('Reorienting ', dwi_prep, ' to RAS+...'))

        # Flip the bvecs
        transform_orientation = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(input_axcodes),
                                                                nib.orientations.axcodes2ornt(new_axcodes))
        bvec_array = np.loadtxt(bvec_fname)
        if bvec_array.shape[0] != 3:
            bvec_array = bvec_array.T
        if not bvec_array.shape[0] == transform_orientation.shape[0]:
            raise ValueError("Unrecognized bvec format")
        output_array = np.zeros_like(bvec_array)
        for this_axnum, (axnum, flip) in enumerate(transform_orientation):
            output_array[this_axnum] = bvec_array[int(axnum)] * float(flip)
        np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")
    else:
        out_fname = "%s%s%s%s%s" % (out_dir, '/', dwi_prep.split('/')[-1].split('.nii')[0], '_noreor-RAS.nii',
                                    dwi_prep.split('/')[-1].split('.nii')[1])
        out_bvec_fname = bvec_fname

    normalized.to_filename(out_fname)
    normalized.uncache()
    input_img.uncache()
    del normalized, input_img
    return out_fname, out_bvec_fname


def reorient_img(img, out_dir):
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
    from pynets.registration.reg_utils import normalize_xform

    # Load image, orient as RAS
    orig_img = nib.load(img)
    normalized = normalize_xform(nib.as_closest_canonical(orig_img))

    # Image may be reoriented
    if normalized is not orig_img:
        print("%s%s%s" % ('Reorienting ', img, ' to RAS+...'))
        out_name = "%s%s%s%s%s" % (out_dir, '/', img.split('/')[-1].split('.nii')[0], '_reor-RAS.nii',
                                   img.split('/')[-1].split('.nii')[1])
    else:
        out_name = "%s%s%s%s%s" % (out_dir, '/', img.split('/')[-1].split('.nii')[0], '_noreor-RAS.nii',
                                   img.split('/')[-1].split('.nii')[1])

    normalized.to_filename(out_name)
    orig_img.uncache()
    normalized.uncache()
    del orig_img

    return out_name


def match_target_vox_res(img_file, vox_size, out_dir):
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
    from dipy.align.reslice import reslice

    # Check dimensions
    img = nib.load(img_file)
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    if vox_size == '1mm':
        new_zooms = (1., 1., 1.)
    elif vox_size == '2mm':
        new_zooms = (2., 2., 2.)

    if (abs(zooms[0]), abs(zooms[1]), abs(zooms[2])) != new_zooms:
        print('Reslicing image ' + img_file + ' to ' + vox_size + '...')
        img_file_res = "%s%s%s%s%s%s%s" % (out_dir, '/', os.path.basename(img_file).split('.nii')[0], '_res-',
                                           vox_size, '.nii', os.path.basename(img_file).split('.nii')[1])
        data2, affine2 = reslice(np.asarray(img.dataobj), img.affine, zooms, new_zooms)
        nib.save(nib.Nifti1Image(data2, affine=affine2), img_file_res)
        img_file = img_file_res
        del data2
    else:
        img_file_nores = "%s%s%s%s%s%s%s" % (out_dir, '/', os.path.basename(img_file).split('.nii')[0], '_nores-',
                                             vox_size, '.nii', os.path.basename(img_file).split('.nii')[1])
        nib.save(img, img_file_nores)
        img_file = img_file_nores

    img.uncache()
    del img
    return img_file
