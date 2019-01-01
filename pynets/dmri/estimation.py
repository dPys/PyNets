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
from dipy.tracking.streamline import Streamlines
import networkx as nx
from itertools import combinations
from collections import defaultdict
import time
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)


def streams2graph(atlas_mni, streams, overlap_thr, dir_path, voxel_size='2mm'):

    # Read Streamlines
    streamlines_mni = nib.streamlines.load(streams).streamlines
    streamlines = Streamlines(streamlines_mni)

    # Load parcellation
    atlas_data = nib.load(atlas_mni).get_data()

    # Instantiate empty networkX graph object & dictionary
    # Create voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4), voxel_size)
    mx = atlas_data.max()
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)

    # Add empty vertices
    for node in range(mx):
        g.add_node(node)

    # Build graph
    start_time = time.time()
    stream_viz = []
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
                    endlabels.append(lab)
                    stream_viz.append(s)

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]
        g.add_weighted_edges_from(edge_list)
    print("%s%s%s" % ('Graph construction runtime: ', str(np.round(time.time() - start_time, 1)), 's'))

    # Stack and save remaining streamlines
    stream_viz_list = np.vstack(stream_viz)
    nib.streamlines.save(Streamlines(stream_viz_list), "%s%s%s%s" % (dir_path, '/streamlines_graph_', overlap_thr, '_overlap.trk'))

    # Convert to numpy matrix
    conn_matrix = nx.to_numpy_matrix(g)

    # Enforce symmetry
    conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.T)

    # Remove background label
    conn_matrix_symm = conn_matrix_symm[1:, 1:]

    # Save matrix
    est_path_raw = "%s%s%s%s" % (dir_path, '/conn_matrix_', overlap_thr, '_overlap.npy')
    np.save(est_path_raw, conn_matrix_symm)
    return conn_matrix_symm
