#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import indexed_gzip
import nibabel as nib

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

    data = nib.load(dwi_file).get_fdata()

    print("Generating tensor FA image to use for registrations...")
    nodif_B0_img = nib.load(B0_mask)
    nodif_B0_mask_data = np.nan_to_num(np.asarray(
        nodif_B0_img.dataobj)).astype("bool")
    model = TensorModel(gtab)
    mod = model.fit(data, nodif_B0_mask_data)
    FA = fractional_anisotropy(mod.evals)
    MD = mean_diffusivity(mod.evals)
    FA_MD = np.logical_or(
        FA >= 0.2, (np.logical_and(
            FA >= 0.08, MD >= 0.0011)))
    FA[np.isnan(FA)] = 0
    FA_MD[np.isnan(FA_MD)] = 0

    fa_path = f"{os.path.dirname(B0_mask)}{'/tensor_fa.nii.gz'}"
    nib.save(
        nib.Nifti1Image(
            FA.astype(
                np.float32),
            nodif_B0_img.affine),
        fa_path)

    fa_md_path = f"{os.path.dirname(B0_mask)}{'/tensor_fa_md.nii.gz'}"
    nib.save(
        nib.Nifti1Image(
            FA_MD.astype(
                np.float32),
            nodif_B0_img.affine),
        fa_md_path)

    nodif_B0_img.uncache()
    del FA, FA_MD

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
    from dipy.core.sphere import HemiSphere
    from dipy.reconst.shm import sf_to_sh

    gtab = load_pickle(gtab_file)
    gtab_hemisphere = HemiSphere(
        xyz=gtab.bvecs[np.where(gtab.b0s_mask == False)])

    img = nib.load(dwi_file)
    aff = img.affine

    anisopwr_path = f"{os.path.dirname(B0_mask)}{'/aniso_power.nii.gz'}"

    if os.path.isfile(anisopwr_path):
        pass
    else:
        print("Generating anisotropic power map to use for registrations...")
        nodif_B0_img = nib.load(B0_mask)

        dwi_data = np.asarray(img.dataobj, dtype=np.float32)
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

    sphere = get_sphere("repulsion724")
    B0_mask_data = np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")
    print("Generating tensor model...")
    model = TensorModel(gtab)
    mod = model.fit(data, B0_mask_data)
    mod_odf = mod.odf(sphere)
    del B0_mask_data
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

    print("Fitting CSA model...")
    model = CsaOdfModel(gtab, sh_order=sh_order)
    B0_mask_data = np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")
    csa_mod = model.fit(data, B0_mask_data).shm_coeff
    del B0_mask_data
    return csa_mod, model


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

    print("Fitting CSD model...")
    B0_mask_data = np.nan_to_num(np.asarray(
        nib.load(B0_mask).dataobj)).astype("bool")
    print("Reconstructing...")
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
        parallel=False,
    )
    print("CSD Reponse: " + str(response))
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    csd_mod = model.fit(data, B0_mask_data).shm_coeff
    del response, B0_mask_data
    return csd_mod, model


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
    print("Fitting SF model...")
    B0_mask_data = np.nan_to_num(np.asarray(nib.load(
        B0_mask).dataobj)).astype("bool")
    print("Reconstructing...")
    model = sfm.SparseFascicleModel(
        gtab, sphere=sphere, l1_ratio=0.5, alpha=0.001)
    sf_mod = model.fit(data, mask=B0_mask_data)
    sf_odf = sf_mod.odf(sphere)

    del B0_mask_data
    return sf_odf, model


def streams2graph(
    atlas_mni,
    streams,
    overlap_thr,
    dir_path,
    track_type,
    target_samples,
    conn_model,
    network,
    node_size,
    dens_thresh,
    ID,
    roi,
    min_span_tree,
    disp_filt,
    parc,
    prune,
    atlas,
    uatlas,
    labels,
    coords,
    norm,
    binary,
    directget,
    warped_fa,
    error_margin,
    min_length,
    fa_wei=True,
):
    """
    Use tracked streamlines as a basis for estimating a structural connectome.

    Parameters
    ----------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    overlap_thr : int
        Number of voxels for which a given streamline must intersect with an ROI
        for an edge to be counted.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
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
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
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
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    warped_fa : str
        File path to MNI-space warped FA Nifti1Image.
    error_margin : int
        Euclidean margin of error for classifying a streamline as a connection to an ROI. Default is 2 voxels.
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    fa_wei :  bool
        Scale streamline count edges by fractional anistropy (FA). Default is False.

    Returns
    -------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
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
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
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
    directget : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    References
    ----------
    .. [1] Sporns, O., Tononi, G., & Kötter, R. (2005). The human connectome:
      A structural description of the human brain. PLoS Computational Biology.
      https://doi.org/10.1371/journal.pcbi.0010042
    .. [2] Sotiropoulos, S. N., & Zalesky, A. (2019). Building connectomes
      using diffusion MRI: why, how and but. NMR in Biomedicine.
      https://doi.org/10.1002/nbm.3752
    .. [3] Chung, M. K., Hanson, J. L., Adluru, N., Alexander, A. L., Davidson,
      R. J., & Pollak, S. D. (2017). Integrative Structural Brain Network
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
    from pynets.dmri.dmri_utils import generate_sl
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space, Origin

    start = time.time()

    # Load parcellation
    roi_img = nib.load(atlas_mni)
    atlas_data = np.around(np.asarray(roi_img.dataobj))
    roi_zooms = roi_img.header.get_zooms()
    roi_shape = roi_img.shape

    # Read Streamlines
    streamlines = [
        i.astype(np.float32)
        for i in Streamlines(
            load_tractogram(
                streams,
                roi_img,
                to_space=Space.RASMM,
                to_origin=Origin.NIFTI,
                bbox_valid_check=True,
            ).streamlines
        )
    ]
    roi_img.uncache()

    if fa_wei is True:
        fa_weights = values_from_volume(
            np.asarray(nib.load(warped_fa).dataobj, dtype=np.float32),
            streamlines, np.eye(4)
        )
        global_fa_weights = list(utils.flatten(fa_weights))
        min_global_fa_wei = min(i for i in global_fa_weights if i > 0)
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
    sl = [generate_sl(i) for i in streamlines]
    del streamlines

    # Instantiate empty networkX graph object & dictionary and create
    # voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4))
    mx = len(np.unique(atlas_data.astype("uint16"))) - 1
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)
    node_dict = dict(
        zip(np.unique(atlas_data.astype("uint16"))[1:], np.arange(mx) + 1))

    # Add empty vertices
    for node in range(1, mx + 1):
        g.add_node(node)

    # Build graph
    ix = 0
    bad_idxs = []
    for s in sl:
        # Map the streamlines coordinates to voxel coordinates and get labels
        # for label_volume
        vox_coords = _to_voxel_coordinates(Streamlines(s), lin_T, offset)
        lab_coords = [
            nodemaker.get_sphere(coord, error_margin, roi_zooms, roi_shape)
            for coord in vox_coords
        ]
        [i, j, k] = np.vstack(np.array(lab_coords)).T

        # get labels for label_volume
        lab_arr = atlas_data[i, j, k]
        endlabels = []
        for ix, lab in enumerate(np.unique(lab_arr).astype("uint32")):
            if (lab > 0) and (np.sum(lab_arr == lab) >= overlap_thr):
                try:
                    endlabels.append(node_dict[lab])
                except BaseException:
                    bad_idxs.append(ix)
                    print(
                        f"Label {lab} missing from parcellation. Check registration and ensure valid input "
                        f"parcellation file.")

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]

        if fa_wei is True:
            # Add edgelist to g, weighted by average fa of the streamline
            g.add_weighted_edges_from(edge_list, weight=fa_weights_norm[ix])
        else:
            g.add_weighted_edges_from(edge_list)
        ix = ix + 1

        del lab_coords, lab_arr, endlabels, edges, edge_list

    gc.collect()

    if fa_wei is True:
        # Add average fa weights to streamline counts
        for u, v in list(g.edges):
            h = g.get_edge_data(u, v)
            edge_att_dict = {}
            for e, w in h.items():
                if w not in edge_att_dict.keys():
                    edge_att_dict[w] = []
                else:
                    edge_att_dict[w].append(e)
            for key in edge_att_dict.keys():
                edge_att_dict[key] = np.nanmean(edge_att_dict[key])
            vals = []
            for e2, w2 in edge_att_dict.items():
                vals.append(float(e2) * float(w2))
            g.edges[u, v].update({"weight": np.nanmean(vals)})

    # Convert to numpy matrix
    conn_matrix_raw = nx.to_numpy_array(g)

    # Impose symmetry
    conn_matrix = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

    print("Graph Building Complete:\n", str(time.time() - start))

    if len(bad_idxs) > 0:
        bad_idxs = sorted(list(set(bad_idxs)), reverse=True)
        for j in bad_idxs:
            del labels[j], coords[j]

    coords = np.array(coords)
    labels = np.array(labels)

    assert len(coords) == len(labels) == conn_matrix.shape[0]

    return (
        atlas_mni,
        streams,
        conn_matrix,
        track_type,
        target_samples,
        dir_path,
        conn_model,
        network,
        node_size,
        dens_thresh,
        ID,
        roi,
        min_span_tree,
        disp_filt,
        parc,
        prune,
        atlas,
        uatlas,
        labels,
        coords,
        norm,
        binary,
        directget,
        min_length,
    )
