#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import warnings
import os
import indexed_gzip
import nibabel as nib
import numpy as np
from nipype.utils.filemanip import fname_presuffix
warnings.filterwarnings("ignore")


def normalize_gradients(bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1, b_scale=True):
    """
    Normalize b-vectors and b-values.

    The resulting b-vectors will be of unit length for the non-zero b-values.
    The resultinb b-values will be normalized by the square of the
    corresponding vector amplitude.

    Parameters
    ----------
    bvecs : m x n 2d array
        Raw b-vectors array.
    bvals : 1d array
        Raw b-values float array.
    b0_threshold : float
        Gradient threshold below which volumes and vectors are considered B0's.

    Returns
    -------
    bvecs : m x n 2d array
        Unit-normed b-vectors array.
    bvals : 1d int array
        Vector amplitude square normed b-values array.
    """
    from dipy.core.gradients import round_bvals
    bvals = np.array(bvals, dtype='float32')
    bvecs = np.array(bvecs, dtype='float32')

    b0s = bvals < b0_threshold
    b0_vecs = np.linalg.norm(bvecs, axis=1) < bvec_norm_epsilon

    # Check for bval-bvec discrepancy.
    if not np.all(b0s == b0_vecs):
        raise ValueError(
            'Inconsistent bvals and bvecs (%d, %d low-b, respectively).' %
            (b0s.sum(), b0_vecs.sum()))

    # Rescale b-vals if requested
    if b_scale:
        bvals[~b0s] *= np.linalg.norm(bvecs[~b0s], axis=1) ** 2

    # Ensure b0s have (0, 0, 0) vectors
    bvecs[b0s, :3] = np.zeros(3)

    # Round bvals
    bvals = round_bvals(bvals)

    # Rescale b-vecs, skipping b0's, on the appropriate axis to unit-norm length.
    bvecs[~b0s] /= np.linalg.norm(bvecs[~b0s], axis=1)[..., np.newaxis]
    return bvecs, bvals.astype('uint16')


def median(in_file):
    """Average a 4D dataset across the last dimension using median."""
    out_file = fname_presuffix(in_file, suffix="_mean_b0.nii.gz", use_ext=True)

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


def generate_sl(streamlines):
    """
    Helper function that takes a sequence and returns a generator
    Parameters
    ----------
    streamlines : sequence
        Usually, this would be a list of 2D arrays, representing streamlines
    Returns
    -------
    generator
    """
    for sl in streamlines:
        yield sl


def extract_b0(in_file, b0_ixs, out_path=None):
    """
    Extract the *b0* volumes from a DWI dataset.

    Parameters
    ----------
    in_file : str
        DWI NIfTI file.
    b0_ixs : list
        List of B0 indices in `in_file`.
    out_path : str
        Optionally specify an output path.

    Returns
    -------
    out_path : str
       4D NIfTI file consisting of B0's.
    """
    if out_path is None:
        out_path = fname_presuffix(
            in_file, suffix='_b0', use_ext=True)

    img = nib.load(in_file)
    data = img.get_fdata()

    b0 = data[..., b0_ixs]

    hdr = img.header.copy()
    hdr.set_data_shape(b0.shape)
    hdr.set_xyzt_units('mm')
    nib.Nifti1Image(b0.astype(hdr.get_data_dtype()), img.affine, hdr).to_filename(out_path)
    return out_path
