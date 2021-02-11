#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
import warnings
import os
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
import numpy as np
from nipype.utils.filemanip import fname_presuffix

warnings.filterwarnings("ignore")


def normalize_gradients(
    bvecs, bvals, b0_threshold, bvec_norm_epsilon=0.1, b_scale=True
):
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

    bvals = np.array(bvals, dtype="float32")
    bvecs = np.array(bvecs, dtype="float32")

    b0s = bvals < b0_threshold
    b0_vecs = np.linalg.norm(bvecs, axis=1) < bvec_norm_epsilon

    # Check for bval-bvec discrepancy.
    if not np.all(b0s == b0_vecs):
        if bvals.shape[0] == bvecs.shape[0]:
            print(UserWarning(
                "Inconsistent B0 locations in bvals and bvecs "
                "(%d, %d low-b, respectively)..."
                % (b0s.sum(), b0_vecs.sum())
            ))
            # Ensure b0s have (0, 0, 0) vectors
            bvecs[b0s, :3] = np.zeros(3)
        else:
            raise ValueError(
                f"Inconsistent number of bvals ({bvals.shape[0]}) and bvecs "
                f"({bvecs.shape[0]}).")

    # Rescale b-vals if requested
    if b_scale:
        bvals[~b0s] *= np.linalg.norm(bvecs[~b0s], axis=1) ** 2

    # Ensure b0s have (0, 0, 0) vectors
    bvecs[b0s, :3] = np.zeros(3)

    # Round bvals
    bvals = round_bvals(bvals)

    # Rescale b-vecs, skipping b0's, on the appropriate axis to unit-norm
    # length.
    bvecs[~b0s] /= np.linalg.norm(bvecs[~b0s], axis=1)[..., np.newaxis]
    return bvecs, bvals.astype("uint16")


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
        out_path = fname_presuffix(in_file, suffix="_b0", use_ext=True)

    img = nib.load(in_file)

    b0 = np.asarray(img.dataobj, dtype=np.float32)[..., b0_ixs]

    hdr = img.header.copy()
    hdr.set_data_shape(b0.shape)
    hdr.set_xyzt_units("mm")
    nib.Nifti1Image(
        b0.astype(
            hdr.get_data_dtype()),
        img.affine,
        hdr).to_filename(out_path)
    return out_path


def evaluate_streamline_plausibility(dwi_data, gtab, mask_data, streamlines,
                                     affine=np.eye(4),
                                     sphere='repulsion724'):
    """
    Linear Fascicle Evaluation (LiFE) takes any connectome and uses a
    forward modelling approach to predict diffusion measurements in the
    same brain.

    Parameters
    ----------
    dwi_data : array
        4D array of dwi data.
    gtab : Obj
        DiPy object storing diffusion gradient information.
    mask_data : array
       3D Brain mask.
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.

    Returns
    -------
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.

    References
    ----------
    .. [1] Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell B.A. (2014).
     Validation and statistical inference in living connectomes.
     Nature Methods 11: 1058-1063. doi:10.1038/nmeth.3098
    """
    import dipy.tracking.life as life
    import dipy.core.optimize as opt
    from dipy.tracking._utils import _mapping_to_voxel
    # from dipy.data import get_sphere
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines

    original_count = len(streamlines)

    streamlines_long = nib.streamlines. \
        array_sequence.ArraySequence(
            [
                s
                for s in streamlines
                if len(s) >= float(10)
            ]
        )
    print('Removing streamlines with negative voxel indices...')
    # Remove any streamlines with negative voxel indices
    lin_T, offset = _mapping_to_voxel(np.eye(4))
    streamlines_positive = []
    for sl in streamlines_long:
        inds = np.dot(sl, lin_T)
        inds += offset
        if not inds.min().round(decimals=6) < 0:
            streamlines_positive.append(sl)
    del streamlines_long

    # Filter resulting streamlines by those that stay entirely
    # inside the ROI of interest
    mask_data = np.array(mask_data, dtype=bool, copy=False)
    streamlines_in_brain = Streamlines(utils.target(
        streamlines_positive, np.eye(4),
        mask_data, include=True
    ))
    streamlines_in_brain = [i for i in streamlines_in_brain]
    del streamlines_positive
    print('Fitting fiber model...')

    # ! Remember this 4d masking function !
    data_in_mask = np.nan_to_num(np.broadcast_to(mask_data[..., None],
                                                 dwi_data.shape
                                                 ).astype('bool') * dwi_data)
    # ! Remember this 4d masking function !

    fiber_model = life.FiberModel(gtab)
    fiber_fit = fiber_model.fit(data_in_mask, streamlines_in_brain,
                                affine=affine,
                                sphere=False)
    # sphere = get_sphere(sphere)
    # fiber_fit = fiber_model.fit(data_in_mask, streamlines_in_brain,
    #                             affine=affine,
    #                             sphere=sphere)
    streamlines = list(np.array(streamlines_in_brain)[
        np.where(fiber_fit.beta > 0)[0]])
    pruned_count = len(streamlines)
    if pruned_count == 0:
        print(UserWarning('\nWarning LiFE skipped due to implausible values '
                          'detected in model betas. This does not '
                          'necessarily invalidate the '
                          'tractography. Rather it could indicate that '
                          'you\'ve sampled too few streamlines, or that the '
                          'sampling scheme is simply incompatible with the '
                          'LiFE model. Is your acquisition hemispheric? '
                          'Also check the gradient table for errors. \n'))
        return streamlines_in_brain
    else:
        del streamlines_in_brain

    model_predict = fiber_fit.predict()
    model_error = model_predict - fiber_fit.data
    model_rmse = np.sqrt(np.mean(model_error[:, 10:] ** 2, -1))
    beta_baseline = np.zeros(fiber_fit.beta.shape[0])
    pred_weighted = np.reshape(opt.spdot(fiber_fit.life_matrix,
                                         beta_baseline),
                               (fiber_fit.vox_coords.shape[0],
                                np.sum(~gtab.b0s_mask)))
    mean_pred = np.empty((fiber_fit.vox_coords.shape[0],
                          gtab.bvals.shape[0]))
    S0 = fiber_fit.b0_signal
    mean_pred[..., gtab.b0s_mask] = S0[:, None]
    mean_pred[..., ~gtab.b0s_mask] = (pred_weighted +
                                      fiber_fit.mean_signal
                                      [:, None]) * S0[:, None]
    mean_error = mean_pred - fiber_fit.data
    mean_rmse = np.sqrt(np.mean(mean_error ** 2, -1))
    print(f"Original # Streamlines: {original_count}")
    print(f"Final # Streamlines: {pruned_count}")
    print(f"Streamlines removed: {pruned_count - original_count}")
    print(f"Mean RMSE: {np.mean(mean_rmse)}")
    print(f"Mean Model RMSE: {np.mean(model_rmse)}")
    print(f"Mean Reduction RMSE: {np.mean(mean_rmse - model_rmse)}")
    return streamlines
