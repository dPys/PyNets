#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import os
import indexed_gzip
import nibabel as nib
import numpy as np
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

    Examples
    --------
    >>> bvecs = np.vstack((np.zeros(3), 2.0 * np.eye(3), -0.8 * np.eye(3), np.ones(3)))
    >>> bvals = np.array([1000] * bvecs.shape[0])
    >>> normalize_gradients(bvecs, bvals, 50)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    >>> bvals[0] = 0.0
    >>> norm_vecs, norm_vals = normalize_gradients(bvecs, bvals)
    >>> np.all(norm_vecs[0] == 0)
    True

    >>> norm_vecs[1, ...].tolist()
    [1.0, 0.0, 0.0]

    >>> norm_vals[0]
    0
    >>> norm_vals[1]
    4000
    >>> norm_vals[-2]
    600
    >>> norm_vals[-1]
    3000

    >>> norm_vecs, norm_vals = normalize_gradients(bvecs, bvals, b_scale=False)
    >>> norm_vals[0]
    0
    >>> np.all(norm_vals[1:] == 1000)
    True

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


def make_mean_b0(in_file):
    import time
    b0_img = nib.load(in_file)
    b0_img_data = b0_img.get_fdata()
    b0_img.uncache()
    mean_b0 = np.mean(b0_img_data, axis=3, dtype=b0_img_data.dtype)
    mean_file_out = in_file.split(".nii")[0] + "_mean_b0.nii.gz"
    nib.save(nib.Nifti1Image(mean_b0, affine=b0_img.affine, header=b0_img.header), mean_file_out)
    while os.path.isfile(mean_file_out) is False:
        time.sleep(1)
    del b0_img_data
    return mean_file_out


def make_gtab_and_bmask(fbval, fbvec, dwi_file, network, node_size, atlas, b0_thr=50):
    """
    Create gradient table from bval/bvec, and a mean B0 brain mask.

    Parameters
    ----------
    fbval : str
        File name of the b-values file.
    fbvec : str
        File name of the b-vectors file.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.

    Returns
    -------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    nodif_b0_bet : str
        File path to mean brain-extracted B0 image.
    B0_mask : str
        File path to mean B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import os
    from dipy.io import save_pickle
    import os.path as op
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from pynets.dmri.dmri_utils import make_mean_b0, normalize_gradients

    outdir = op.dirname(dwi_file)

    namer_dir = outdir + '/dmri_tmp'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    B0_bet = "%s%s" % (namer_dir, "/mean_B0_bet.nii.gz")
    B0_mask = "%s%s" % (namer_dir, "/mean_B0_bet_mask.nii.gz")
    fbvec_norm = "%s%s" % (namer_dir, "/bvec_normed.bvec")
    fbval_norm = "%s%s" % (namer_dir, "/bval_normed.bvec")
    gtab_file = "%s%s" % (namer_dir, "/gtab.pkl")
    all_b0s_file = "%s%s" % (namer_dir, "/all_b0s.nii.gz")

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    bvecs_norm, bvals_norm = normalize_gradients(bvecs, bvals, b0_threshold=b0_thr)

    # Save corrected
    np.savetxt(fbval_norm, bvals_norm)
    np.savetxt(fbvec_norm, bvecs_norm)

    # Creating the gradient table
    gtab = gradient_table(bvals_norm, bvecs_norm)

    # Correct b0 threshold
    gtab.b0_threshold = b0_thr

    # Get b0 indices
    b0s = np.where(gtab.bvals <= gtab.b0_threshold)[0]
    print("%s%s" % ('b0\'s found at: ', b0s))

    # Correct bvals to set 0's for B0 based on thresh
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    # Extract and Combine all b0s collected, make mean b0
    print("Extracting b0's...")
    b0_vols = []
    dwi_img = nib.load(dwi_file)
    all_b0s_aff = dwi_img.affine.copy()
    dwi_data = np.asarray(dwi_img.dataobj)
    dwi_img.uncache()
    for b0 in b0s:
        print(b0)
        b0_vols.append(dwi_data[:, :, :, b0])

    all_b0s_aff[3][3] = len(b0_vols)
    nib.save(nib.Nifti1Image(np.stack(b0_vols, axis=3), affine=all_b0s_aff), all_b0s_file)
    mean_b0_file = make_mean_b0(all_b0s_file)

    # Create mean b0 brain mask
    cmd = 'bet ' + mean_b0_file + ' ' + B0_bet + ' -m -f 0.2'
    os.system(cmd)

    del dwi_data

    return gtab_file, B0_bet, B0_mask, dwi_file
