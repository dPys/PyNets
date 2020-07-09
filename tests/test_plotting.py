#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

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
from pynets.plotting import plot_gen, plot_graphs
import tempfile
import logging
import pkg_resources
import networkx as nx

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_plot_conn_mat_nonet_no_mask(plotting_data):
    """
    Test plot_conn_mat_nonet_no_mask functionality
    """

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    extract_strategy = 'mean'
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = None

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']

    start_time = time.time()
    plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
    ID, network, labels, roi, thr, node_size, smooth, hpass, extract_strategy)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


def test_plot_conn_mat_nonet_mask(plotting_data):
    """
    Test plot_conn_mat_nonet_mask functionality
    """

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    hpass = 0.1
    conn_model = 'sps'
    extract_strategy = 'mean'
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = None

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']

    start_time = time.time()
    plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
    ID, network, labels, roi, thr, node_size, smooth, hpass, extract_strategy)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


def test_plot_all_nonet_no_mask(plotting_data):
    """
    Test plot_all_nonet_no_mask functionality
    """

    base_dir = str(Path(__file__).parent/"examples")
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    conn_model = 'sps'
    parlistfile = None
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    prune = 1
    norm = 1
    hpass = 0.1
    binary = False
    extract_strategy = 'mean'
    edge_threshold = '99%'

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']
    coords = plotting_data['coords']

    start_time = time.time()
    plot_gen.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi,
                           coords, thr, node_size, edge_threshold, smooth, prune,
                           parlistfile, norm, binary, hpass, extract_strategy)
    print("%s%s%s" % ('plot_all --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


def test_plot_all_nonet_with_mask(plotting_data):
    """
    Test plot_all_nonet_with_mask functionality
    """

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)
    network = None
    ID = '002'
    thr = 0.95
    node_size = 'None'
    smooth = 2
    prune = False
    norm = 1
    hpass = 0.1
    binary = False
    conn_model = 'sps'
    atlas = b'whole_brain_cluster_labels_PCA200'
    parlistfile = None
    roi = None
    extract_strategy = 'mean'
    edge_threshold = '99%'

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']
    coords = plotting_data['coords']

    # Force an isolate in the matrix
    conn_matrix[:, 0] = 0
    conn_matrix[0, :] = 0

    # Adds coverage
    coords = np.array(coords)
    labels = np.array(labels)

    start_time = time.time()
    plot_gen.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, coords, thr,
                           node_size, edge_threshold, smooth, prune, parlistfile, norm, binary, hpass, extract_strategy,
                           edge_color_override=True)
    print("%s%s%s" % ('plot_all (Masking version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


@pytest.mark.parametrize("comm, network",
    [
        pytest.param('nodes', 'Default'),
        pytest.param('links', None),
        pytest.param(None, None)
    ]
)
def test_plot_connectogram(comm, network, plotting_data):
    """
    Test plot_connectogram functionality
    """

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    ID = '002'
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    prune = 2

    conn_matrix = plotting_data['conn_matrix']
    labels = np.arange(conn_matrix.shape[0] + 1)[np.arange(conn_matrix.shape[0] + 1) != 0].tolist()

    # Force an isolate in the matrix
    conn_matrix[:, 0] = 0
    conn_matrix[0, :] = 0

    start_time = time.time()
    plot_gen.plot_connectogram(conn_matrix, conn_model, atlas, dir_path, ID, network, labels,
                               comm=comm, prune=prune)
    print("%s%s%s" % ('plot_connectogram --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    if comm:
        assert os.path.isfile(f"{dir_path}/index.html")

    temp_dir.cleanup()


@pytest.mark.parametrize("network", ["Default", None])
def test_plot_timeseries(plotting_data, network):
    """
    Test plot_timeseries functionality
    """

    base_dir = str(Path(__file__).parent/"examples")
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    ID = '002'
    atlas = 'whole_brain_cluster_labels_PCA200'
    time_series = np.load(f"{base_dir}/miscellaneous/002_rsn-Default_net_ts.npy")

    labels = plotting_data['labels']

    start_time = time.time()
    plot_gen.plot_timeseries(time_series, network, ID, dir_path, atlas, labels)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


@pytest.mark.parametrize("plot_overlaps", [True, False])
def test_plot_network_clusters(plotting_data, plot_overlaps):
    """ Test plotting network clusters"""

    from pynets.stats.netstats import community_resolution_selection

    temp_file = tempfile.NamedTemporaryFile(mode='w+', prefix='figure', suffix='.png')
    fname = str(temp_file.name)

    conn_matrix = plotting_data['conn_matrix']

    G = nx.from_numpy_matrix(np.abs(conn_matrix))
    _, communities, _, _ = community_resolution_selection(G)
    plot_labels = True

    plot_gen.plot_network_clusters(G, communities, fname, plot_overlaps=plot_overlaps,
                                   plot_labels=plot_labels)

    assert os.path.isfile(fname)

    temp_file.close()


@pytest.mark.parametrize("prune, node_cmap",
    [
        pytest.param(True, None),
        pytest.param(False, "Set2")
    ]
)
def test_create_gb_palette(plotting_data, prune, node_cmap):
    """Test palette creation."""
    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']
    coords = plotting_data['coords']

    # Force an isolate in the matrix
    conn_matrix[:, 0] = 0
    conn_matrix[0, :] = 0

    color_theme = 'binary'

    palette = plot_gen.create_gb_palette(conn_matrix, color_theme, coords, labels,
                                         prune=prune, node_cmap=node_cmap)
    for param in palette:
        assert param is not None


def test_plot_conn_mat_rois_gt_100(plotting_data):
    """
    Test plot_conn_mat_rois_gt_100 functionality
    """

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']

    start_time = time.time()
    plot_graphs.plot_conn_mat(conn_matrix, labels, dir_path, cmap='Blues')
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    temp_dir.cleanup()


@pytest.mark.parametrize("roi", [True, False])
def test_plot_all_struct(plotting_data, roi):
    """Test structural plotting."""
    base_dir = str(Path(__file__).parent/"examples")
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    conn_matrix = plotting_data['conn_matrix']
    labels = plotting_data['labels']
    coords = plotting_data['coords']

    # Adds coverage
    coords = np.array(coords)
    labels = np.array(labels)

    conn_model = 'corr'
    atlas = b'whole_brain_cluster_labels_PCA200'
    ID = '002'
    network = None
    if roi:
        roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    else:
        roi = None
    thr = 0.95
    node_size = None
    edge_threshold = '99%'
    prune = True
    uatlas = None
    target_samples = 500
    norm = 6
    binary = False
    directget = 'prob'
    track_type = 'particle'
    min_length = 10

    plot_gen.plot_all_struct(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi,
                             coords, thr, node_size, edge_threshold, prune, uatlas, target_samples,
                             norm, binary, track_type, directget, min_length)

    temp_dir.cleanup()


def test_plot_all_struct_func(plotting_data):
    """Test structural and functional plotting."""
    import multinetx as mx
    base_dir = str(Path(__file__).parent/"examples")
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = str(temp_dir.name)

    labels = plotting_data['labels'][:10]
    coords = plotting_data['coords'][:10]
    metadata = {'coords': coords, 'labels': labels}

    name = 'output_fname'

    func_file = (f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_roi-002_parcels_re"
                 f"sampled2roimask_pDMN_3_bin_raw_est-sps_nodetype-spheres-6mm_smooth-2fwhm_hpass-"
                 f"FalseHz_raw.npy")
    dwi_file = (f"{base_dir}/miscellaneous/graphs/002_modality-dwi_rsn-Default_roi-002_parcels_res"
                f"ampled2roimask_pDMN_3_bin_raw_est-sps_parc_samples-2streams_tt-local_dg-prob_ml-"
                f"200_raw.npy")
    modality_paths = (func_file, dwi_file)

    G_func = nx.from_numpy_matrix(np.load(func_file))
    G_dwi = nx.from_numpy_matrix(np.load(dwi_file))

    mG_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.gpickle')
    mG = mx.MultilayerGraph(list_of_layers=[G_func, G_dwi])

    # This is a hack to get i/o working. There is an nx inhertiance issue that prevents reading.
    intra_layer_edges = mG.intra_layer_edges
    del mG.intra_layer_edges

    nx.write_gpickle(mG, mG_path.name)

    plot_gen.plot_all_struct_func(mG_path.name, dir_path, name, modality_paths, metadata)

    temp_dir.cleanup()
    mG_path.close()


@pytest.mark.parametrize("nan", [True, False])
def test_plot_graph_measure_hists(nan):
    """Test plotting histograms from metric dataframe."""
    import pandas as pd

    base_dir = str(Path(__file__).parent/"examples")
    temp_dir = tempfile.TemporaryDirectory()
    dir_name = str(temp_dir.name)

    df_csv = (f"{base_dir}/miscellaneous/sub-OAS31172_ses-d0407_topology_auc_clean.csv")

    # Hack the dataframe
    if nan is True:
        df = pd.read_csv(df_csv)
        df[df.columns[4]] = np.nan
        df.to_csv(f"{dir_name}/TEST.csv", index=False)
        fig = plot_gen.plot_graph_measure_hists(f"{dir_name}/TEST.csv")
    else:
        fig = plot_gen.plot_graph_measure_hists(df_csv)
    assert fig is not None
    temp_dir.cleanup()


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
#     atlas = 'whole_brain_cluster_labels_PCA200'
#     ID = '997'
#     network = None
#     labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     roi = None
#     thr = 0.95
#     node_size = 2
#     smooth = 2
#
#     plotting.plot_conn_mat_struct(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, thr,
#                                   node_size, smooth)

# def test_structural_plotting():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_est-sps_thrtype-PROP_thr-0.94.txt")
#     labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     atlas = 'whole_brain_cluster_labels_PCA200'
#     ID = '002'
#     bedpostx_dir = base_dir + 'bedpostx_s002.bedpostX'
#     network = None
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
#     plotting.structural_plotting(conn_matrix, labels, atlas, ID, bedpostx_dir, network, parc, roi, coords,
#                                  dir_path, conn_model, thr, node_size, smooth)


# def test_plot_graph_measure_hists():
#     df_concat = np.random.rand(4,4)
#     measures = [1, 2, 3]
#     net_pick_file = '/Users/ryanhammonds/tmp/tmp/tmp'
#
#     plotting.plot_graph_measure_hists(df_concat, measures, net_pick_file)
