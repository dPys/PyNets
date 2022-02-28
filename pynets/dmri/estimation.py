#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import os
import matplotlib
import warnings
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def tens_mod_fa_est(gtab_file, dwi_file, B0_mask):
    """
    Estimate a tensor FA image to use for registrations.

    Parameters
    ----------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted Nifti1Image.
    fa_md_path : str
        File path to FA/MD mask Nifti1Image.
    """
    import os
    from dipy.io import load_pickle
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

    gtab = load_pickle(gtab_file)

    data = nib.load(dwi_file, mmap=False).get_fdata(dtype=np.float32)

    print("Reconstructing tensors...")
    nodif_B0_img = nib.load(B0_mask, mmap=False)
    nodif_B0_mask_data = nodif_B0_img.get_fdata().astype("bool")
    model = TensorModel(gtab)
    mod = model.fit(data, nodif_B0_mask_data)
    FA = fractional_anisotropy(mod.evals)
    # MD = mean_diffusivity(mod.evals)
    # FA_MD = np.logical_or(
    #     FA >= 0.2, (np.logical_and(
    #         FA >= 0.08, MD >= 0.0011)))
    # FA_MD[np.isnan(FA_MD)] = 0
    FA = np.nan_to_num(np.asarray(FA.astype('float32')))

    fa_path = f"{os.path.dirname(B0_mask)}{'/tensor_fa.nii.gz'}"
    nib.save(
        nib.Nifti1Image(
            FA,
            nodif_B0_img.affine),
        fa_path)

    # md_path = f"{os.path.dirname(B0_mask)}{'/tensor_md.nii.gz'}"
    # nib.save(
    #     nib.Nifti1Image(
    #         MD.astype(
    #             np.float32),
    #         nodif_B0_img.affine),
    #     md_path)

    nodif_B0_img.uncache()
    del FA

    return fa_path, B0_mask, gtab_file, dwi_file


def create_anisopowermap(gtab_file, dwi_file, B0_mask):
    """
    Estimate an anisotropic power map image to use for registrations.

    Parameters
    ----------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    anisopwr_path : str
        File path to the anisotropic power Nifti1Image.
    B0_mask : str
        File path to B0 brain mask Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted Nifti1Image.

    References
    ----------
    .. [1] Chen, D. Q., Dell’Acqua, F., Rokem, A., Garyfallidis, E., Hayes, D.,
      Zhong, J., & Hodaie, M. (2018). Diffusion Weighted Image Co-registration:
      Investigation of Best Practices. PLoS ONE.

    """
    import os
    from dipy.io import load_pickle
    from dipy.reconst.shm import anisotropic_power
    from dipy.core.sphere import HemiSphere, Sphere
    from dipy.reconst.shm import sf_to_sh

    gtab = load_pickle(gtab_file)

    dwi_vertices = gtab.bvecs[np.where(gtab.b0s_mask == False)]

    gtab_hemisphere = HemiSphere(
        xyz=gtab.bvecs[np.where(gtab.b0s_mask == False)])

    try:
        assert len(gtab_hemisphere.vertices) == len(dwi_vertices)
    except BaseException:
        gtab_hemisphere = Sphere(
            xyz=gtab.bvecs[np.where(gtab.b0s_mask == False)])

    img = nib.load(dwi_file)
    aff = img.affine

    anisopwr_path = f"{os.path.dirname(B0_mask)}{'/aniso_power.nii.gz'}"

    if os.path.isfile(anisopwr_path):
        pass
    else:
        print("Reconstructing anisotropic power map...")
        nodif_B0_img = nib.load(B0_mask)
        dwi_data = img.get_fdata(dtype=np.float32)
        for b0 in sorted(list(np.where(gtab.b0s_mask)[0]), reverse=True):
            dwi_data = np.delete(dwi_data, b0, 3)

        anisomap = anisotropic_power(
            sf_to_sh(
                dwi_data,
                gtab_hemisphere,
                sh_order=2))
        anisomap[np.isnan(anisomap)] = 0
        masked_data = anisomap * \
            np.asarray(nodif_B0_img.dataobj).astype("bool")
        img = nib.Nifti1Image(masked_data.astype(np.float32), aff)
        img.to_filename(anisopwr_path)
        nodif_B0_img.uncache()
        del anisomap
        img.uncache()

    return anisopwr_path, B0_mask, gtab_file, dwi_file


def tens_mod_est(gtab, data, B0_mask):
    """
    Estimate a tensor ODF model from dwi data.

    Parameters
    ----------
    gtab : Obj
        DiPy object storing diffusion gradient information
    data : array
        4D numpy array of diffusion image data.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    mod_odf : ndarray
        Coefficients of the tensor reconstruction.
    model : obj
        Fitted tensor model.

    References
    ----------
    .. [1] Basser PJ, Mattielo J, LeBihan (1994). MR diffusion tensor
      spectroscopy and imaging.
    .. [2] Pajevic S, Pierpaoli (1999). Color schemes to represent the
      orientation of anisotropic tissues from diffusion tensor data:
      application to white matter fiber tract mapping in the human brain.

    """
    from dipy.reconst.dti import TensorModel
    from dipy.data import get_sphere

    print("Reconstructing tensors...")
    model = TensorModel(gtab)
    mod = model.fit(data,
                    np.nan_to_num(
                        np.asarray(
                            nib.load(B0_mask).dataobj)
                    ).astype("bool"))
    mod_odf = mod.odf(get_sphere("repulsion724"))
    return mod_odf, model


def csa_mod_est(gtab, data, B0_mask, sh_order=8):
    """
    Estimate a Constant Solid Angle (CSA) model from dwi data.

    Parameters
    ----------
    gtab : Obj
        DiPy object storing diffusion gradient information
    data : array
        4D numpy array of diffusion image data.
    B0_mask : str
        File path to B0 brain mask.
    sh_order : int
        The order of the SH model. Default is 8.

    Returns
    -------
    csa_mod : ndarray
        Coefficients of the csa reconstruction.
    model : obj
        Fitted csa model.

    References
    ----------
    .. [1] Aganj, I., et al. 2009. ODF Reconstruction in Q-Ball Imaging
      with Solid Angle Consideration.

    """
    from dipy.reconst.shm import CsaOdfModel

    print("Reconstructing using CSA...")
    model = CsaOdfModel(gtab, sh_order=sh_order)
    csa_mod = model.fit(data, np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")).shm_coeff
    # Clip any negative values
    csa_mod = np.clip(csa_mod, 0, np.max(csa_mod, -1)[..., None])
    return csa_mod.astype("float32"), model


def csd_mod_est(gtab, data, B0_mask, sh_order=8):
    """
    Estimate a Constrained Spherical Deconvolution (CSD) model from dwi data.

    Parameters
    ----------
    gtab : Obj
        DiPy object storing diffusion gradient information.
    data : array
        4D numpy array of diffusion image data.
    B0_mask : str
        File path to B0 brain mask.
    sh_order : int
        The order of the SH model. Default is 8.

    Returns
    -------
    csd_mod : ndarray
        Coefficients of the csd reconstruction.
    model : obj
        Fitted csd model.

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
      the fibre orientation distribution in diffusion MRI:
      Non-negativity constrained super-resolved spherical
      deconvolution
    .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
      Probabilistic Tractography Based on Complex Fibre Orientation
      Distributions
    .. [3] Côté, M-A., et al. Medical Image Analysis 2013. Tractometer:
      Towards validation of tractography pipelines
    .. [4] Tournier, J.D, et al. Imaging Systems and Technology
      2012. MRtrix: Diffusion Tractography in Crossing Fiber Regions

    """
    from dipy.reconst.csdeconv import (
        ConstrainedSphericalDeconvModel,
        recursive_response,
    )

    print("Reconstructing using CSD...")
    B0_mask_data = np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")
    response = recursive_response(
        gtab,
        data,
        mask=B0_mask_data,
        sh_order=sh_order,
        peak_thr=0.01,
        init_fa=0.08,
        init_trace=0.0021,
        iter=8,
        convergence=0.001,
        parallel=False
    )
    # print(f"CSD Reponse: {response}")
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    csd_mod = model.fit(data, B0_mask_data).shm_coeff
    csd_mod = np.clip(csd_mod, 0, np.max(csd_mod, -1)[..., None])
    del response, B0_mask_data
    return csd_mod.astype("float32"), model


def mcsd_mod_est(gtab, data, B0_mask, wm_in_dwi, gm_in_dwi, vent_csf_in_dwi,
                 sh_order=8, roi_radii=10):
    """
    Estimate a Constrained Spherical Deconvolution (CSD) model from dwi data.

    Parameters
    ----------
    gtab : Obj
        DiPy object storing diffusion gradient information.
    data : array
        4D numpy array of diffusion image data.
    B0_mask : str
        File path to B0 brain mask.
    sh_order : int
        The order of the SH model. Default is 8.

    Returns
    -------
    csd_mod : ndarray
        Coefficients of the csd reconstruction.
    model : obj
        Fitted csd model.

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
      the fibre orientation distribution in diffusion MRI:
      Non-negativity constrained super-resolved spherical
      deconvolution
    .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
      Probabilistic Tractography Based on Complex Fibre Orientation
      Distributions
    .. [3] Côté, M-A., et al. Medical Image Analysis 2013. Tractometer:
      Towards validation of tractography pipelines
    .. [4] Tournier, J.D, et al. Imaging Systems and Technology
      2012. MRtrix: Diffusion Tractography in Crossing Fiber Regions

    """
    import dipy.reconst.dti as dti
    from nilearn.image import math_img
    from dipy.core.gradients import unique_bvals_tolerance
    from dipy.reconst.mcsd import (mask_for_response_msmt,
                                   response_from_mask_msmt,
                                   multi_shell_fiber_response,
                                   MultiShellDeconvModel)

    print("Reconstructing using MCSD...")

    B0_mask_data = np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")

    # Load tissue maps and prepare tissue classifier
    gm_mask_img = math_img("img > 0.10", img=gm_in_dwi)
    gm_data = np.asarray(gm_mask_img.dataobj, dtype=np.float32)

    wm_mask_img = math_img("img > 0.15", img=wm_in_dwi)
    wm_data = np.asarray(wm_mask_img.dataobj, dtype=np.float32)

    vent_csf_in_dwi_img = math_img("img > 0.50", img=vent_csf_in_dwi)
    vent_csf_in_dwi_data = np.asarray(vent_csf_in_dwi_img.dataobj,
                                      dtype=np.float32)

    # Fit a simple DTI model
    tenfit = dti.TensorModel(gtab).fit(data)

    # Obtain the FA and MD metrics
    FA = tenfit.fa
    MD = tenfit.md

    indices_csf = np.where(((FA < 0.2) & (vent_csf_in_dwi_data > 0.50)))
    indices_gm = np.where(((FA < 0.2) & (gm_data > 0.10)))
    indices_wm = np.where(((FA >= 0.2) & (wm_data > 0.15)))

    selected_csf = np.zeros(FA.shape, dtype='bool')
    selected_gm = np.zeros(FA.shape, dtype='bool')
    selected_wm = np.zeros(FA.shape, dtype='bool')

    selected_csf[indices_csf] = True
    selected_gm[indices_gm] = True
    selected_wm[indices_wm] = True

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(
        gtab, data, roi_radii=roi_radii,
        wm_fa_thr=np.nanmean(FA[selected_wm]),
        gm_fa_thr=np.nanmean(FA[selected_gm]),
        csf_fa_thr=np.nanmean(FA[selected_csf]),
        gm_md_thr=np.nanmean(MD[selected_gm]),
        csf_md_thr=np.nanmean(MD[selected_csf]))

    mask_wm *= wm_data.astype('int64')
    mask_gm *= gm_data.astype('int64')
    mask_csf *= vent_csf_in_dwi_data.astype('int64')

    # nvoxels_wm = np.sum(mask_wm)
    # nvoxels_gm = np.sum(mask_gm)
    # nvoxels_csf = np.sum(mask_csf)

    response_wm, response_gm, response_csf = response_from_mask_msmt(
        gtab, data, mask_wm, mask_gm, mask_csf)

    response_mcsd = multi_shell_fiber_response(sh_order=8,
                                               bvals=unique_bvals_tolerance(
                                                   gtab.bvals),
                                               wm_rf=response_wm,
                                               gm_rf=response_gm,
                                               csf_rf=response_csf)

    model = MultiShellDeconvModel(gtab, response_mcsd, sh_order=sh_order)
    mcsd_mod = model.fit(data, B0_mask_data).shm_coeff

    mcsd_mod = np.clip(mcsd_mod, 0, np.max(mcsd_mod, -1)[..., None])
    del response_mcsd, B0_mask_data
    return mcsd_mod.astype("float32"), model


def sfm_mod_est(gtab, data, B0_mask):
    """
    Estimate a Sparse Fascicle Model (SFM) from dwi data.

    Parameters
    ----------
    gtab : Obj
        DiPy object storing diffusion gradient information.
    data : array
        4D numpy array of diffusion image data.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    sf_mod : ndarray
        Coefficients of the sfm reconstruction.
    model : obj
        Fitted sf model.

    References
    ----------
    .. [1] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
      N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
      (2015). Evaluating the accuracy of diffusion MRI models in white
      matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272
    .. [2] Ariel Rokem, Kimberly L. Chan, Jason D. Yeatman, Franco
      Pestilli,  Brian A. Wandell (2014). Evaluating the accuracy of diffusion
      models at multiple b-values with cross-validation. ISMRM 2014.

    """
    from dipy.data import get_sphere
    import dipy.reconst.sfm as sfm

    sphere = get_sphere("repulsion724")
    print("Reconstructing using SFM...")
    model = sfm.SparseFascicleModel(
        gtab, sphere=sphere, l1_ratio=0.5, alpha=0.001)
    sf_mod = model.fit(data, mask=np.nan_to_num(np.asarray(nib.load(
        B0_mask).dataobj)).astype("bool"))
    sf_odf = sf_mod.odf(sphere)
    sf_odf = np.clip(sf_odf, 0, np.max(sf_odf, -1)[..., None])
    return sf_odf.astype("float32"), model


def reconstruction(conn_model, gtab, dwi_data, B0_mask):
    """
    Estimate a tensor model from dwi data.

    Parameters
    ----------
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd',
        'sfm').
    gtab : Obj
        DiPy object storing diffusion gradient information.
    dwi_data : array
        4D array of dwi data.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    mod_fit : ndarray
        Fitted connectivity reconstruction model.
    mod : obj
        Connectivity reconstruction model.

    References
    ----------
    .. [1] Soares, J. M., Marques, P., Alves, V., & Sousa, N. (2013).
      A hitchhiker’s guide to diffusion tensor imaging.
      Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2013.00031
    """
    from pynets.dmri.estimation import (
        csa_mod_est,
        csd_mod_est,
        sfm_mod_est,
        tens_mod_est,
        mcsd_mod_est
    )

    if conn_model == "csa" or conn_model == "CSA":
        [mod_fit, mod] = csa_mod_est(gtab, dwi_data, B0_mask)
    # elif conn_model == "mcsd" or conn_model == "MCSD":
    #     [mod_fit, mod] = mcsd_mod_est(gtab, dwi_data, B0_mask, wm_in_dwi,
    #                                   gm_in_dwi, vent_csf_in_dwi)
    elif conn_model == "csd" or conn_model == "CSD":
        [mod_fit, mod] = csd_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == "sfm" or conn_model == "SFM":
        [mod_fit, mod] = sfm_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == "ten" or conn_model == "tensor" or \
            conn_model == "TEN":
        [mod_fit, mod] = tens_mod_est(gtab, dwi_data, B0_mask)
    else:
        raise ValueError(
            "Error: No valid reconstruction model specified. See the "
            "`-mod` flag."
        )

    del dwi_data

    return mod_fit, mod


def streams2graph(
    atlas_for_streams,
    streams,
    dir_path,
    track_type,
    conn_model,
    subnet,
    node_radius,
    dens_thresh,
    ID,
    roi,
    min_span_tree,
    disp_filt,
    parc,
    prune,
    atlas,
    parcellation,
    labels,
    coords,
    norm,
    binary,
    traversal,
    warped_fa,
    min_length,
    error_margin
):
    """
    Use tracked streamlines as a basis for estimating a structural connectome.

    Parameters
    ----------
    atlas_for_streams : str
        File path to atlas parcellation Nifti1Image in T1w-conformed space.
    streams : str
        File path to streamline array sequence in .trk format.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    traversal : str
        The statistical approach to tracking. Options are:
        det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    warped_fa : str
        File path to MNI-space warped FA Nifti1Image.
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    error_margin : int
        Euclidean margin of error for classifying a streamline as a connection
         to an ROI. Default is 2 voxels.

    Returns
    -------
    atlas_for_streams : str
        File path to atlas parcellation Nifti1Image in T1w-conformed space.
    streams : str
        File path to streamline array sequence in .trk format.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    dir_path : str
        Path to directory containing subject derivative data for given run.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    traversal : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    error_margin : int
        Euclidean margin of error for classifying a streamline as a connection
         to an ROI. Default is 2 voxels.

    References
    ----------
    .. [1] Sporns, O., Tononi, G., & Kötter, R. (2005). The human connectome:
      A structural description of the human brain. PLoS Computational Biology.
      https://doi.org/10.1371/journal.pcbi.0010042
    .. [2] Sotiropoulos, S. N., & Zalesky, A. (2019). Building connectomes
      using diffusion MRI: why, how and but. NMR in Biomedicine.
      https://doi.org/10.1002/nbm.3752
    .. [3] Chung, M. K., Hanson, J. L., Adluru, N., Alexander, A. L., Davidson,
      R. J., & Pollak, S. D. (2017). Integrative Structural Brain subnet
      Analysis in Diffusion Tensor Imaging. Brain Connectivity.
      https://doi.org/10.1089/brain.2016.0481
    """
    import gc
    import time
    from dipy.tracking.streamline import Streamlines, values_from_volume
    from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
    import networkx as nx
    from itertools import combinations
    from collections import defaultdict
    from pynets.core import utils, nodemaker
    from pynets.dmri.utils import generate_sl
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space, Origin
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    fa_wei = hardcoded_params[
        "StructuralNetworkWeighting"]["fa_weighting"][0]
    fiber_density = hardcoded_params[
        "StructuralNetworkWeighting"]["fiber_density"][0]
    overlap_thr = hardcoded_params[
        "StructuralNetworkWeighting"]["overlap_thr"][0]
    roi_neighborhood_tol = \
        hardcoded_params['tracking']["roi_neighborhood_tol"][0]

    start = time.time()

    if float(roi_neighborhood_tol) <= float(error_margin):
        raise ValueError('roi_neighborhood_tol preset cannot be less than '
                         'the value of the structural connectome error'
                         '_margin parameter.')
    else:
        print(f"Using fiber-roi intersection tolerance: {error_margin}...")

    # Load FA
    fa_img = nib.load(warped_fa)

    # Load parcellation
    roi_img = nib.load(atlas_for_streams)
    atlas_data = np.around(np.asarray(roi_img.dataobj))
    roi_zooms = roi_img.header.get_zooms()
    roi_shape = roi_img.shape

    # Read Streamlines
    if streams is not None:
        streamlines = [
            i.astype(np.float32)
            for i in Streamlines(
                load_tractogram(
                    streams,
                    fa_img,
                    to_origin=Origin.NIFTI,
                    to_space=Space.VOXMM
                ).streamlines
            )
        ]

        # Remove streamlines with negative voxel indices
        lin_T, offset = _mapping_to_voxel(np.eye(4))
        streams_filtered = []
        neg_vox = False
        for sl in streamlines:
            inds = np.dot(sl, lin_T)
            inds += offset
            if not inds.min().round(decimals=6) < 0:
                streams_filtered.append(sl)
            else:
                neg_vox = True

        if neg_vox is True:
            print(UserWarning("Negative voxel indices detected! "
                              "Check FOV"))

        streamlines = streams_filtered
        del streams_filtered
        # from fury import actor, window, colormap
        # renderer = window.Renderer()
        # template_actor = actor.contour_from_roi(roi_img.get_fdata(),
        #                                         color=(50, 50, 50),
        #                                         opacity=1)
        # renderer.add(template_actor)
        # lines_actor = actor.line(streamlines,
        #                                colormap.line_colors(streamlines))
        # renderer.add(lines_actor)
        # window.show(renderer)
        #
        # roi_img.uncache()

        if fa_wei is True:
            fa_weights = values_from_volume(
                np.asarray(fa_img.dataobj, dtype=np.float32),
                streamlines, np.eye(4)
            )
            global_fa_weights = list(utils.flatten(fa_weights))
            min_global_fa_wei = min([i for i in global_fa_weights if i > 0])
            max_global_fa_wei = max(global_fa_weights)
            fa_weights_norm = []
            # Here we normalize by global FA
            for val_list in fa_weights:
                fa_weights_norm.append(
                    np.nanmean(
                        (val_list - min_global_fa_wei)
                        / (max_global_fa_wei - min_global_fa_wei)
                    )
                )

        # Make streamlines into generators to keep memory at a minimum
        total_streamlines = len(streamlines)
        sl = [generate_sl(i) for i in streamlines]
        del streamlines
        gc.collect()

        # Instantiate empty networkX graph object & dictionary and create
        # voxel-affine mapping
        lin_T, offset = _mapping_to_voxel(np.eye(4))
        mx = len(np.unique(atlas_data.astype("uint16"))) - 1
        g = nx.Graph(ecount=0, vcount=mx)
        edge_dict = defaultdict(int)
        node_dict = dict(
            zip(np.unique(atlas_data.astype("uint16"))[1:], np.arange(mx) + 1))

        # Add empty vertices with label volume attributes
        for node in range(1, mx + 1):
            g.add_node(node, roi_volume=np.sum(
                atlas_data.astype("uint16") == node)
            )

        # Build graph
        pc = 0
        bad_idxs = []
        fiberlengths = {}
        fa_weights_dict = {}
        print(f"Quantifying fiber-ROI intersection for {atlas}:")
        for ix, s in enumerate(sl):
            # Percent counter
            pcN = int(round(100*float(ix / total_streamlines)))
            if pcN % 10 == 0 and ix > 0 and pcN > pc:
                pc = pcN
                print(f"{pcN}%")

            # Map the streamlines coordinates to voxel coordinates and get
            # labels for label_volume
            s = Streamlines(s)
            if s.data.shape[0] == 0:
                continue
            vox_coords = _to_voxel_coordinates(s, lin_T, offset)

            [i, j, k] = np.vstack(np.array([
                nodemaker.get_sphere(coord, error_margin, roi_zooms, roi_shape)
                for coord in vox_coords
            ])).T

            # get labels for label_volume
            lab_arr = atlas_data[i, j, k]
            # print(lab_arr)
            endlabels = []
            for jx, lab in enumerate(np.unique(lab_arr).astype("uint32")):
                if (lab > 0) and (np.sum(lab_arr == lab) >= overlap_thr):
                    try:
                        endlabels.append(node_dict[lab])
                    except BaseException:
                        bad_idxs.append(jx)
                        print(
                            f"Label {lab} missing from parcellation. Check "
                            f"registration and ensure valid input "
                            f"parcellation file.")

            for edge in combinations(endlabels, 2):
                # Get fiber lengths along edge
                if fiber_density is True:
                    if not (edge[0], edge[1]) in fiberlengths.keys():
                        fiberlengths[(edge[0], edge[1])] = [len(vox_coords)]
                    else:
                        fiberlengths[(edge[0],
                                      edge[1])].append(len(vox_coords))

                # Get FA values along edge
                if fa_wei is True:
                    if not (edge[0], edge[1]) in fa_weights_dict.keys():
                        fa_weights_dict[(edge[0],
                                         edge[1])] = [fa_weights_norm[ix]]
                    else:
                        fa_weights_dict[(edge[0],
                                         edge[1])].append(fa_weights_norm[ix])

                edge_dict[tuple(sorted(tuple([int(node) for node in
                                              edge])))] += 1

            g.add_weighted_edges_from([(k[0],
                                        k[1], count) for
                                       k, count in edge_dict.items()])

            del lab_arr, endlabels
            gc.collect()

        del sl
        gc.collect()

        # Add fiber density attributes for each edge
        # Adapted from the nnormalized fiber-density estimation routines of
        # Sebastian Tourbier.
        if fiber_density is True:
            print("Redefining edges on the basis of fiber density...")
            # Summarize total fibers and total label volumes
            total_fibers = 0
            total_volume = 0
            u_start = -1
            for u, v, d in g.edges(data=True):
                total_fibers += len(d)
                if u != u_start:
                    total_volume += g.nodes[int(u)]['roi_volume']
                u_start = u

            ix = 0
            for u, v, d in g.edges(data=True):
                if d['weight'] > 0:
                    fiber_density = (float(((float(d['weight']) /
                                             float(total_fibers)) /
                                            float(np.nanmean(fiberlengths[
                                                                 (u, v)]))) *
                                           ((2.0 * float(total_volume)) /
                                            (g.nodes[int(u)]['roi_volume'] +
                                               g.nodes[int(v)]['roi_volume']))
                                           )) * 1000
                else:
                    fiber_density = 0
                g.edges[u, v].update({"fiber_density": fiber_density})
                ix += 1

        if fa_wei is True:
            print("Re-weighting edges by mean FA along each edge's associated "
                  "bundles...")
            # Add FA attributes for each edge
            ix = 0
            for u, v, d in g.edges(data=True):
                if d['weight'] > 0:
                    edge_average_fa = np.nanmean(fa_weights_dict[(u, v)])
                else:
                    edge_average_fa = np.nan
                g.edges[u, v].update({"fa_weight": edge_average_fa})
                ix += 1

        # Summarize weights
        if fa_wei is True and fiber_density is True:
            for u, v, d in g.edges(data=True):
                g.edges[u, v].update({"final_weight":
                                      (d['fa_weight'])*d['fiber_density']})
        elif fiber_density is True and fa_wei is False:
            for u, v, d in g.edges(data=True):
                g.edges[u, v].update({"final_weight": d['fiber_density']})
        elif fa_wei is True and fiber_density is False:
            for u, v, d in g.edges(data=True):
                g.edges[u, v].update({"final_weight":
                                      d['fa_weight']*d['weight']})
        else:
            for u, v, d in g.edges(data=True):
                g.edges[u, v].update({"final_weight": d['weight']})

        # Convert weighted graph to numpy matrix
        conn_matrix_raw = nx.to_numpy_array(g, weight='final_weight')

        # Enforce symmetry
        conn_matrix = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

        print("Structural graph completed:\n", str(time.time() - start))

        if len(bad_idxs) > 0:
            bad_idxs = sorted(list(set(bad_idxs)), reverse=True)
            for j in bad_idxs:
                del labels[j], coords[j]
    else:
        print(UserWarning('No valid streamlines detected. '
                          'Proceeding with an empty graph...'))
        mx = len(np.unique(atlas_data.astype("uint16"))) - 1
        conn_matrix = np.zeros((mx, mx))

    assert len(coords) == len(labels) == conn_matrix.shape[0]

    if subnet is not None:
        atlas_name = f"{atlas}_{subnet}_stage-rawgraph"
    else:
        atlas_name = f"{atlas}_stage-rawgraph"

    utils.save_coords_and_labels_to_json(coords, labels, dir_path,
                                         atlas_name, indices=None)

    coords = np.array(coords)
    labels = np.array(labels)

    if parc is True:
        node_radius = "parc"

    # Save unthresholded
    utils.save_mat(
        conn_matrix,
        utils.create_raw_path_diff(
            ID,
            subnet,
            conn_model,
            roi,
            dir_path,
            node_radius,
            track_type,
            parc,
            traversal,
            min_length,
            error_margin
        ),
    )

    return (
        atlas_for_streams,
        streams,
        conn_matrix,
        track_type,
        dir_path,
        conn_model,
        subnet,
        node_radius,
        dens_thresh,
        ID,
        roi,
        min_span_tree,
        disp_filt,
        parc,
        prune,
        atlas,
        parcellation,
        labels,
        coords,
        norm,
        binary,
        traversal,
        min_length,
        error_margin
    )
