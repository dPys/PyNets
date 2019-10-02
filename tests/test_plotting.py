#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.plotting import plot_gen, plot_graphs


def test_plot_conn_mat_nonet_no_mask():
    """
    Test plot_conn_mat_nonet_no_mask functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    c_boot = 2
    hpass = 0.1
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
    ID, network, labels, roi, thr, node_size, smooth, c_boot, hpass)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_nonet_mask():
    """
    Test plot_conn_mat_nonet_mask functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    c_boot = 3
    hpass = 0.1
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path,
    ID, network, labels, roi, thr, node_size, smooth, c_boot, hpass)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_all_nonet_no_mask():
    """
    Test plot_all_nonet_no_mask functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    c_boot = 3
    conn_model = 'sps'
    parlistfile = None
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = None
    prune = 1
    norm = 1
    hpass = 0.1
    binary = False
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.94.txt')
    edge_threshold = '99%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    #coords already a list
    plot_gen.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID,
    network, labels, roi, coords, edge_threshold, thr, node_size, smooth,
    prune, parlistfile, c_boot, norm, binary, hpass)
    print("%s%s%s" % ('plot_all --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_all_nonet_with_mask():
    """
    Test plot_all_nonet_with_mask functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    thr = 0.95
    node_size = 2
    smooth = 2
    c_boot = 3
    prune = 1
    norm = 1
    hpass = 0.1
    binary = False
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    parlistfile = None
    roi = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.94.txt')
    edge_threshold = '99%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    #coords already a list
    plot_gen.plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID,
    network, labels, roi, coords, edge_threshold, thr, node_size, smooth,
    prune, parlistfile, c_boot, norm, binary, hpass)
    print("%s%s%s" % ('plot_all (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_connectogram():
    """
    Test plot_connectogram functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    conn_model = 'sps'
    atlas = 'whole_brain_cluster_labels_PCA200'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plot_gen.plot_connectogram(conn_matrix, conn_model, atlas, dir_path,
    ID, network, labels)
    print("%s%s%s" % ('plot_connectogram --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_timeseries():
    """
    Test plot_timeseries functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = None
    ID = '002'
    atlas = 'whole_brain_cluster_labels_PCA200'
    time_series = np.load(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_rsn_net_ts.npy')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plot_gen.plot_timeseries(time_series, network, ID, dir_path, atlas, labels)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_rois_gt_100():
    """
    Test plot_conn_mat_rois_gt_100 functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.95.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)
    out_path_fig = '/tmp/fig'

    start_time = time.time()
    plot_graphs.plot_conn_mat(conn_matrix, labels, out_path_fig)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


# def test_plot_conn_mat_rois_lt_100():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.95.txt')
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     out_path_fig = '/tmp/'
#
#     start_time = time.time()
#     plotting.plot_conn_mat(conn_matrix, labels, out_path_fig)
#     print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


# def test_plot_conn_mat_struct():
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.95.txt')
#     conn_model = 'sps'
#     atlas = 'whole_brain_cluster_labels_PCA200'
#     ID = '997'
#     network = None
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
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
#     conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/002_Default_est_sps_0.95.txt')
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
#     labels_file = open(labels_file_path,'rb')
#     labels = pickle.load(labels_file)
#     atlas = 'whole_brain_cluster_labels_PCA200'
#     ID = '002'
#     bedpostx_dir = base_dir + 'bedpostx_s002.bedpostX'
#     network = None
#     parc = True
#     roi = None
#     coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
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
