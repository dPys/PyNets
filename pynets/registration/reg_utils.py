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
np.warnings.filterwarnings('ignore')
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
    cmd = "flirt -in {} -ref {} -out {} -init {} -interp {} -dof {} -applyxfm"
    cmd = cmd.format(inp, ref, aligned, xfm, interp, dof)
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


def transform_to_affine(streams, header, affine):
    """
    A function to transform .trk file of tractography streamlines to a given affine.

    Parameters
    ----------
    streams : str
        File path to input streamline array sequence in .trk format.
    header : Dict
        Nibabel trackvis header object to use for transformed streamlines file.
    affine : array
        4 x 4 2D numpy array representing the target affine for streamline transformation.

    Returns
    -------
    streams : str
        File path to transformed streamline array sequence in .trk format.
    """
    from dipy.tracking.utils import move_streamlines
    rotation, scale = np.linalg.qr(affine)
    streams = move_streamlines(streams, rotation)
    scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1. / header['voxel_sizes']))
    scale[0:3, 3] = abs(scale[0:3, 3])
    streams = move_streamlines(streams, scale)
    return streams
