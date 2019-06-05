# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
try:
    FSLDIR = os.environ['FSLDIR']
except KeyError:
    print('FSLDIR environment variable not set!')


def segment_t1w(t1w, basename, opts=''):
    """
    A function to use FSL's FAST to segment an anatomical
    image into GM, WM, and CSF prob maps.

    **Positional Arguments:**

        t1w:
            - an anatomical T1w image.
        basename:
            - the basename for outputs. Often it will be
              most convenient for this to be the dataset,
              followed by the subject, followed by the step of
              processing. Note that this anticipates a path as well;
              ie, /path/to/dataset_sub_nuis, with no extension.
        opts:
            - additional options that can optionally be passed to
              fast. Desirable options might be -P, which will use
              prior probability maps if the input T1w MRI is in
              standard space.
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
          wmseg=None, init=None, finesearch=None):
    """
    Aligns two images and stores the transform between them
    **Positional Arguments:**
            inp:
                - Input impage to be aligned as a nifti image file
            ref:
                - Image being aligned to as a nifti image file
            xfm:
                - Returned transform between two images
            out:
                - determines whether the image will be automatically
                aligned.
            dof:
                - the number of degrees of freedom of the alignment.
            searchrad:
                - a bool indicating whether to use the predefined
                searchradius parameter (180 degree sweep in x, y, and z).
            interp:
                - the interpolation method to use.
            sch:
                - the optional FLIRT schedule file.
            wmseg:
                - an optional white-matter segmentation for bbr.
            init:
                - an initial guess of an alignment.
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


def align_nonlinear(inp, ref, xfm, out, warp, ref_mask=None, in_mask=None, config=None):
    """
    Aligns two images using nonlinear methods and stores the
    transform between them.

    **Positional Arguments:**

        inp:
            - the input image.
        ref:
            - the reference image.
        affxfm:
            - the affine transform to use.
        warp:
            - the path to store the nonlinear warp.
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


def applyxfm(ref, inp, xfm, aligned, interp='trilinear', dof=6):
    """
    Aligns two images with a given transform

    **Positional Arguments:**

            inp:
                - Input impage to be aligned as a nifti image file
            ref:
                - Image being aligned to as a nifti image file
            xfm:
                - Transform between two images
            aligned:
                - Aligned output image as a nifti image file
    """
    cmd = "flirt -in {} -ref {} -out {} -init {} -interp {} -dof {} -applyxfm"
    cmd = cmd.format(inp, ref, aligned, xfm, interp, dof)
    print(cmd)
    os.system(cmd)


def apply_warp(ref, inp, out, warp, xfm=None, mask=None, interp=None, sup=False):
    """
    Applies a warp from the functional to reference space
    in a single step, using information about the structural->ref
    mapping as well as the functional to structural mapping.

    **Positional Arguments:**

        inp:
            - the input image to be aligned as a nifti image file.
        out:
            - the output aligned image.
        ref:
            - the image being aligned to.
        warp:
            - the warp from the structural to reference space.
        premat:
            - the affine transformation from functional to
            structural space.
    """
    cmd = "applywarp --ref=" + ref + " --in=" + inp + " --out=" + out +\
          " --warp=" + warp
    if xfm is not None:
        cmd += " --premat=" + xfm
    if mask is not None:
        cmd += " --mask=" + mask
    if interp is not None:
        cmd += " --interp=" + interp
    if sup is True:
        cmd += " --super --superlevel=a"
    print(cmd)
    os.system(cmd)


def inverse_warp(ref, out, warp):
    """
    Applies a warp from the functional to reference space
    in a single step, using information about the structural->ref
    mapping as well as the functional to structural mapping.

    **Positional Arguments:**

        inp:
            - the input image to be aligned as a nifti image file.
        out:
            - the output aligned image.
        ref:
            - the image being aligned to.
        warp:
            - the warp from the structural to reference space.
        premat:
            - the affine transformation from functional to
            structural space.
    """
    cmd = "invwarp --warp=" + warp + " --out=" + out + " --ref=" + ref
    print(cmd)
    os.system(cmd)


def combine_xfms(xfm1, xfm2, xfmout):
    """
    A function to combine two transformations, and output the
    resulting transformation.

    **Positional Arguments**
        xfm1:
            - the path to the first transformation
        xfm2:
            - the path to the second transformation
        xfmout:
            - the path to the output transformation
    """
    cmd = "convert_xfm -omat {} -concat {} {}".format(xfmout, xfm1, xfm2)
    print(cmd)
    os.system(cmd)


def transform_to_affine(streams, header, affine):
    """

    :param streams:
    :param header:
    :param affine:
    :return:
    """
    from dipy.tracking.utils import move_streamlines
    rotation, scale = np.linalg.qr(affine)
    streams = move_streamlines(streams, rotation)
    scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1. / header['voxel_sizes']))
    scale[0:3, 3] = abs(scale[0:3, 3])
    streams = move_streamlines(streams, scale)
    return streams
