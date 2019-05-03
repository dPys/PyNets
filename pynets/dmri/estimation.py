# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib


def tens_mod_fa_est(gtab_file, dwi, nodif_B0_mask):
    import os
    from dipy.io import load_pickle
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dti import fractional_anisotropy

    data = nib.load(dwi).get_fdata()
    gtab = load_pickle(gtab_file)

    print('Fitting tensor model...')
    nodif_B0_img = nib.load(nodif_B0_mask)
    nodif_B0_mask_data = nodif_B0_img.get_fdata().astype('bool')
    nodif_B0_affine = nodif_B0_img.affine
    model = TensorModel(gtab)
    mod = model.fit(data, nodif_B0_mask_data)
    print('Computing anisotropy measures (FA, MD, RGB)')
    FA = fractional_anisotropy(mod.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA.astype(np.float32), nodif_B0_affine)
    fa_path = "%s%s" % (os.path.dirname(nodif_B0_mask), '/tensor_fa.nii.gz')
    nib.save(fa_img, fa_path)
    return fa_path


def tens_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.dti import TensorModel
    from dipy.data import get_sphere
    print('Fitting tensor model...')
    sphere = get_sphere('repulsion724')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_fdata().astype('bool')
    model = TensorModel(gtab)
    mod = model.fit(data, wm_in_dwi_mask)
    tensor_odf = mod.odf(sphere)
    return tensor_odf


def csa_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.shm import CsaOdfModel
    print('Fitting CSA model...')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_fdata().astype('bool')
    model = CsaOdfModel(gtab, sh_order=6)
    mod = model.fit(data, wm_in_dwi_mask)
    return mod.shm_coeff


def csd_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
    print('Fitting CSD model...')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_fdata().astype('bool')
    try:
        print('Attempting to use spherical harmonic...')
        model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
    except:
        print('Falling back to recursive response...')
        response = recursive_response(gtab, data, mask=wm_in_dwi_mask, sh_order=8,
                                      peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8, convergence=0.001,
                                      parallel=False)
        print('CSD Reponse: ' + str(response))
        model = ConstrainedSphericalDeconvModel(gtab, response)
    mod = model.fit(data, wm_in_dwi_mask)
    return mod.shm_coeff


def streams2graph(atlas_mni, streams, overlap_thr, dir_path, track_type, target_samples, conn_model, network, node_size,
                  dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names,
                  coords, norm, binary, curv_thr_list, step_list, voxel_size='2mm'):
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    import networkx as nx
    from itertools import combinations
    from collections import defaultdict
    import time

    # Read Streamlines
    streamlines_mni = nib.streamlines.load(streams)
    streamlines = Streamlines(streamlines_mni.streamlines)

    # Load parcellation
    atlas_data = nib.load(atlas_mni).get_fdata()

    # Instantiate empty networkX graph object & dictionary
    # Create voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4), voxel_size)
    mx = len(np.unique(atlas_data.astype(np.int64)))
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)
    node_dict = dict(zip(np.unique(atlas_data), np.arange(mx)))

    # Add empty vertices
    for node in range(mx):
        g.add_node(node)

    # Build graph
    start_time = time.time()
    for s in streamlines:
        # Map the streamlines coordinates to voxel coordinates
        points = _to_voxel_coordinates(s, lin_T, offset)

        # get labels for label_volume
        i, j, k = points.T
        lab_arr = atlas_data[i, j, k]
        endlabels = []
        for lab in np.unique(lab_arr):
            if lab > 0:
                if np.sum(lab_arr == lab) >= overlap_thr:
                    endlabels.append(node_dict[lab])

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]
        g.add_weighted_edges_from(edge_list)
    print("%s%s%s" % ('Graph construction runtime: ', str(np.round(time.time() - start_time, 1)), 's'))

    # Convert to numpy matrix
    conn_matrix_raw = nx.to_numpy_matrix(g)

    # Enforce symmetry
    conn_matrix_symm = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

    # Remove background label
    conn_matrix = conn_matrix_symm[1:, 1:]

    return conn_matrix, track_type, target_samples, dir_path, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, norm, binary
