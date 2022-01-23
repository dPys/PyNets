#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
"""
import os
import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
import pytest
import pkg_resources
from pynets.plotting import brain, adjacency
import tempfile
import logging
import networkx as nx

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_plot_conn_mat_nonet_no_mask(connectivity_data):
    """
    Test plot_conn_mat_nonet_no_mask functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    signal = 'mean'
    conn_model = 'sps'
    atlas = 'random_parcellation'
    roi = None

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']

    start_time = time.time()
    adjacency.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
                                 ID, subnet, labels, roi, thr, node_size,
                                 smooth, hpass, signal)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()

def test_plot_conn_mat_nonet_mask(connectivity_data):
    """
    Test plot_conn_mat_nonet_mask functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    conn_model = 'sps'
    signal = 'mean'
    atlas = 'random_parcellation'
    roi = None

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']

    start_time = time.time()
    adjacency.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
                                 ID, subnet, labels, roi, thr, node_size,
                                 smooth, hpass, signal)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()


def test_plot_all_nonet_no_mask(random_mni_roi_data, connectivity_data):
    """
    Test plot_all_nonet_no_mask functionality
    """

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    conn_model = 'sps'
    parlistfile = None
    atlas = 'random_parcellation'
    roi = random_mni_roi_data['roi_file']
    prune = 1
    norm = 1
    hpass = 0.1
    binary = False
    signal = 'mean'
    edge_threshold = '99%'

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']
    coords = connectivity_data['coords']

    start_time = time.time()
    brain.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, subnet,
                        labels, roi, coords, thr, node_size, edge_threshold,
                        smooth, prune, parlistfile, norm, binary, hpass,
                        signal)
    print("%s%s%s" % ('plot_all --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_all_nonet_with_mask(connectivity_data):
    """
    Test plot_all_nonet_with_mask functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = None
    ID = '002'
    thr = 0.95
    node_size = 3
    smooth = 2
    prune = 3
    norm = 3
    hpass = 0.1
    binary = False
    conn_model = 'corr'
    atlas = 'random_parcellation'
    parlistfile = None
    roi = None
    signal = 'mean'
    edge_threshold = '99%'

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']
    coords = connectivity_data['coords']

    # Force an isolate in the matrix
    conn_matrix[:, 0] = 0
    conn_matrix[0, :] = 0

    # Adds coverage
    coords = np.array(coords)
    labels = np.array(labels)

    start_time = time.time()
    brain.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, subnet,
                        labels, roi, coords, thr, node_size, edge_threshold,
                        smooth, prune, parlistfile, norm, binary, hpass,
                        signal, edge_color_override=True)
    print("%s%s%s" % ('plot_all (Masking version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()


@pytest.mark.parametrize("subnet", ["Default", None])
def test_plot_timeseries(connectivity_data, subnet):
    """
    Test plot_timeseries functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    ID = '002'
    atlas = 'random_parcellation'
    time_series = connectivity_data['time_series']
    labels = connectivity_data['labels']

    start_time = time.time()
    brain.plot_timeseries(time_series, subnet, ID, dir_path, atlas, labels)
    print("%s%s%s" % ('plot_timeseries --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()

@pytest.mark.parametrize("plot_overlaps", [True, False])
def test_plot_network_clusters(connectivity_data, plot_overlaps):
    """ Test plotting subnet clusters"""

    from pynets.statistics.individual.algorithms import \
        community_resolution_selection

    temp_file = tempfile.NamedTemporaryFile(mode='w+', prefix='figure',
                                            suffix='.png')
    fname = str(temp_file.name)

    conn_matrix = connectivity_data['conn_matrix']

    G = nx.from_numpy_matrix(np.abs(conn_matrix))
    _, communities, _, _ = community_resolution_selection(G)
    plot_labels = True

    brain.plot_network_clusters(G, communities, fname,
                                plot_overlaps=plot_overlaps,
                                plot_labels=plot_labels)

    assert os.path.isfile(fname)

    temp_file.close()


@pytest.mark.parametrize("prune, node_cmap",
    [
        pytest.param(3, 0),
        pytest.param(False, "Set2")
    ]
)
def test_create_gb_palette(connectivity_data, prune, node_cmap):
    """Test palette creation."""
    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']
    coords = connectivity_data['coords']

    # Force an isolate in the matrix
    conn_matrix[:, 0] = 0
    conn_matrix[0, :] = 0

    color_theme = 'binary'

    palette = brain.create_gb_palette(conn_matrix, color_theme, coords, labels,
                                      prune=prune, node_cmap=node_cmap)
    for param in palette:
        assert param is not None


def test_plot_conn_mat_rois_gt_100(connectivity_data):
    """
    Test plot_conn_mat_rois_gt_100 functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    conn_matrix = connectivity_data['conn_matrix']
    labels = connectivity_data['labels']

    start_time = time.time()
    adjacency.plot_conn_mat(conn_matrix, labels, dir_path, cmap='Blues')
    print("%s%s%s" % ('plot_timeseries --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    tmp.cleanup()

@pytest.mark.parametrize("roi", [True, False])
def test_plot_all_struct(connectivity_data, gen_mat_data, roi):
    """Test structural plotting."""

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    labels = connectivity_data['labels']
    coords = connectivity_data['coords']
    conn_matrix = gen_mat_data(m=len(labels), n=len(labels),
                               asfile=False, mat_type='sb')['mat_list'][0]

    # Adds coverage
    coords = np.array(coords)
    labels = np.array(labels)

    conn_model = 'corr'
    atlas = 'random_parcellation'
    ID = '002'
    subnet = None
    if roi:
        roi = pkg_resources.resource_filename(
        "pynets", "templates/rois/pDMN_3_bin.nii.gz")
    else:
        roi = None
    thr = 0.95
    node_size = None
    edge_threshold = '99%'
    prune = 3
    parcellation = None
    norm = 3
    binary = False
    traversal = 'prob'
    track_type = 'particle'
    min_length = 10
    error_margin = 2

    brain.plot_all_struct(conn_matrix, conn_model, atlas, dir_path, ID,
                          subnet, labels, roi, coords, thr, node_size,
                          edge_threshold, prune, parcellation, norm, binary,
                          track_type, traversal, min_length, error_margin)
    tmp.cleanup()

# def test_plot_all_struct_func(connectivity_data):
#     """Test structural and functional plotting."""
#     import multinetx as mx
#     base_dir = str(Path(__file__).parent/"examples")
#     temp_dir = tempfile.TemporaryDirectory()
#     dir_path = str(temp_dir.name)
#
#     labels = connectivity_data['labels'][:10]
#     coords = connectivity_data['coords'][:10]
#     metadata = {'coords': coords, 'labels': labels}
#
#     name = 'output_fname'
#
#     func_file = (f"{base_dir}/miscellaneous/graphs/rawgraph_sub-002_modality-func_rsn-Default_roi-002_parcels_resampled2roimask_pDMN_3_bin_model-sps_template-MNI152_T1_nodetype-spheres-6mm_smooth-2fwhm_hpass-FalseHz_extract-mean.npy")
#     dwi_file = (f"{base_dir}/miscellaneous/graphs/rawgraph_sub-002_modality-dwi_rsn-Default_roi-002_parcels_resampled2roimask_pDMN_3_bin_model-sps_template-MNI152_T1_nodetype-spheres-6mm_samples-2streams_tracktype-local_directget-prob_minlength-200.npy")
#     modality_paths = (func_file, dwi_file)
#
#     G_func = nx.from_numpy_matrix(np.load(func_file))
#     G_dwi = nx.from_numpy_matrix(np.load(dwi_file))
#
#     mG_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.gpickle')
#     mG = mx.MultilayerGraph(list_of_layers=[G_func, G_dwi])
#
#     # This is a hack to get i/o working. There is an nx inhertiance issue that prevents reading.
#     intra_layer_edges = mG.intra_layer_edges
#     del mG.intra_layer_edges
#
#     nx.write_gpickle(mG, mG_path.name)
#
#     plot_gen.plot_all_struct_func(mG_path.name, dir_path, name, modality_paths, metadata)
#
#     temp_dir.cleanup()
#     mG_path.close()


# @pytest.mark.parametrize("nan", [True, False])
# def test_plot_graph_measure_hists(nan):
#     """Test plotting histograms from metric dataframe."""
#     import pandas as pd
#
#     base_dir = str(Path(__file__).parent/"examples")
#
#     tmp = tempfile.TemporaryDirectory()
#     dir_path = str(tmp.name)
#     os.makedirs(dir_path, exist_ok=True)
#
#     df_csv = f"{base_dir}/miscellaneous/metrics_sub-OAS31172_ses-d0407_" \
#              f"topology_auc_clean.csv"
#
#     # Hack the dataframe
#     if nan is True:
#         df = pd.read_csv(df_csv)
#         df[df.columns[4]] = np.nan
#         df.to_csv(f"{dir_path}/TEST.csv", index=False)
#         fig = brain.plot_graph_measure_hists(f"{dir_path}/TEST.csv")
#     else:
#         fig = brain.plot_graph_measure_hists(df_csv)
#     assert fig is not None
#     tmp.cleanup()

# def test_plot_conn_mat_rois_lt_100():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_est-sps_thrtype-PROP_thr-0.94.txt")
#     labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     out_path_fig = '/tmp/'
#
#     start_time = time.time()
#     plotting.plot_conn_mat(conn_matrix, labels, out_path_fig, cmap='Blues')
#     print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


# def test_plot_conn_mat_struct():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_est-sps_thrtype-PROP_thr-0.94.txt")
#     conn_model = 'sps'
#     atlas = 'random_parcellation'
#     ID = '997'
#     subnet = None
#     labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     roi = None
#     thr = 0.95
#     node_size = 2
#     smooth = 2
#
#     plotting.plot_conn_mat_struct(conn_matrix, conn_model, atlas, dir_path, ID, subnet, labels, roi, thr,
#                                   node_size, smooth)

# def test_structural_plotting():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_est-sps_thrtype-PROP_thr-0.94.txt")
#     labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     atlas = 'random_parcellation'
#     ID = '002'
#     bedpostx_dir = base_dir + 'bedpostx_s002.bedpostX'
#     subnet = None
#     parc = True
#     roi = None
#     coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
#     coord_file = open(coord_file_path, 'rb')
#     coords = pickle.load(coord_file)
#     conn_model = 'sps'
#     thr = 0.95
#     node_size = 2
#     smooth = 2
#
#     plotting.structural_plotting(conn_matrix, labels, atlas, ID, bedpostx_dir, subnet, parc, roi, coords,
#                                  dir_path, conn_model, thr, node_size, smooth)


# def test_plot_graph_measure_hists():
#     df_concat = np.random.rand(4,4)
#     measures = [1, 2, 3]
#     net_pick_file = '/Users/ryanhammonds/tmp/tmp/tmp'
#
#     plotting.plot_graph_measure_hists(df_concat, measures, net_pick_file)
