#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import indexed_gzip
import nibabel as nib
warnings.filterwarnings("ignore")


def tens_mod_fa_est(gtab_file, dwi_file, B0_mask):
    '''
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
    '''
    import os
    from dipy.io import load_pickle
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dti import fractional_anisotropy

    data = nib.load(dwi_file).get_fdata()
    gtab = load_pickle(gtab_file)

    print('Generating simple tensor FA image to use for registrations...')
    nodif_B0_img = nib.load(B0_mask)
    model = TensorModel(gtab)
    mod = model.fit(data, nodif_B0_img.get_fdata().astype('bool'))
    FA = fractional_anisotropy(mod.evals)
    FA[np.isnan(FA)] = 0
    fa_path = "%s%s" % (os.path.dirname(B0_mask), '/tensor_fa.nii.gz')
    nib.save(nib.Nifti1Image(FA.astype(np.float32), nodif_B0_img.affine), fa_path)
    nodif_B0_img.uncache()
    del FA
    return fa_path, B0_mask, gtab_file, dwi_file


def csa_mod_est(gtab, data, B0_mask):
    '''
    Estimate a Constant Solid Angle (CSA) model from dwi data.

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
    csa_mod : obj
        Spherical harmonics coefficients of the CSA-estimated reconstruction model.
    '''
    from dipy.reconst.shm import CsaOdfModel
    print('Fitting CSA model...')
    model = CsaOdfModel(gtab, sh_order=6)
    csa_mod = model.fit(data, nib.load(B0_mask).get_fdata().astype('bool')).shm_coeff
    del model
    return csa_mod


def csd_mod_est(gtab, data, B0_mask):
    '''
    Estimate a Constrained Spherical Deconvolution (CSD) model from dwi data.

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
    csd_mod : obj
        Spherical harmonics coefficients of the CSD-estimated reconstruction model.
    '''
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
    print('Fitting CSD model...')
    B0_mask_data = nib.load(B0_mask).get_fdata().astype('bool')
    print('Reconstructing...')
    response = recursive_response(gtab, data, mask=B0_mask_data, sh_order=8, peak_thr=0.01, init_fa=0.08,
                                  init_trace=0.0021, iter=8, convergence=0.001, parallel=False)
    print('CSD Reponse: ' + str(response))
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_mod = model.fit(data, B0_mask_data).shm_coeff
    del model, response, B0_mask_data
    return csd_mod


def streams2graph(atlas_mni, streams, overlap_thr, dir_path, track_type, target_samples, conn_model, network, node_size,
                  dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels,
                  coords, norm, binary, directget, warped_fa, error_margin, fa_wei=True):
    '''
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
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    '''
    from dipy.tracking.streamline import Streamlines, values_from_volume
    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    import networkx as nx
    from itertools import combinations
    from collections import defaultdict
    from pynets.core import utils, nodemaker
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space
    import time

    # Load parcellation
    roi_img = nib.load(atlas_mni)
    atlas_data = np.around(roi_img.get_fdata())
    roi_zooms = roi_img.header.get_zooms()
    roi_shape = roi_img.shape

    # Read Streamlines
    streamlines = Streamlines(load_tractogram(streams, roi_img, to_space=Space.RASMM, shifted_origin=True,
                                              bbox_valid_check=False).streamlines)
    roi_img.uncache()

    fa_weights = values_from_volume(nib.load(warped_fa).get_fdata(), streamlines, np.eye(4))
    global_fa_weights = list(utils.flatten(fa_weights))
    min_global_fa_wei = min(global_fa_weights)
    max_global_fa_wei = max(global_fa_weights)
    fa_weights_norm = []
    for val_list in fa_weights:
        fa_weights_norm.append((val_list - min_global_fa_wei) / (max_global_fa_wei - min_global_fa_wei))

    # Instantiate empty networkX graph object & dictionary and create voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4))
    mx = len(np.unique(atlas_data.astype('uint8'))) - 1
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)
    node_dict = dict(zip(np.unique(atlas_data.astype('uint8')) + 1, np.arange(mx) + 1))

    # Add empty vertices
    for node in range(1, mx + 1):
        g.add_node(node)

    # Build graph
    start_time = time.time()

    ix = 0
    for s in streamlines:
        # Map the streamlines coordinates to voxel coordinates and get labels for label_volume
        i, j, k = np.vstack(np.array([nodemaker.get_sphere(coord, error_margin, roi_zooms,
                                                           roi_shape) for coord in
                                      _to_voxel_coordinates(s, lin_T, offset)])).T

        # get labels for label_volume
        lab_arr = atlas_data[i, j, k]
        endlabels = []
        for lab in np.unique(lab_arr).astype('uint8'):
            if (lab > 0) and (np.sum(lab_arr == lab) >= overlap_thr):
                try:
                    endlabels.append(node_dict[lab])
                except UserWarning:
                    print("%s%s%s" % ('Label ', lab, ' missing from parcellation. Check registration and ensure valid '
                                                     'input parcellation file.'))

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]

        if fa_wei is True:
            # Add edgelist to g, weighted by average fa of the streamline
            g.add_weighted_edges_from(edge_list, weight=np.nanmean(fa_weights_norm[ix]))
        else:
            g.add_weighted_edges_from(edge_list)
        ix = ix + 1

    print("%s%s%s" % ('Graph construction runtime: ', np.round(time.time() - start_time, 1), 's'))
    del streamlines

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
            g.edges[u, v].update({'weight': np.nanmean(vals)})

    # Convert to numpy matrix
    conn_matrix_raw = nx.to_numpy_matrix(g)

    # Enforce symmetry
    conn_matrix = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

    return atlas_mni, streams, conn_matrix, track_type, target_samples, dir_path, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, directget
