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
from pynets import plotting


def test_plot_conn_mat_nonet_no_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    smooth = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_conn_mat_func(conn_matrix, conn_model, atlas_select, dir_path,
    ID, network, label_names, mask, thr, node_size, smooth)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_nonet_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    smooth = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_conn_mat_func(conn_matrix, conn_model, atlas_select, dir_path,
    ID, network, label_names, mask, thr, node_size, smooth)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_all_nonet_no_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    smooth = 2
    conn_model = 'sps'
    parlistfile = None
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    prune = 1
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    edge_threshold = '99%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    #coords already a list
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID,
    network, label_names, mask, coords, edge_threshold, thr, node_size, smooth,
    prune, parlistfile)
    print("%s%s%s" % ('plot_all --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_all_nonet_with_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    smooth = 2
    prune = 1
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    parlistfile = None
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    edge_threshold = '99%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    start_time = time.time()
    #coords already a list
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID,
    network, label_names, mask, coords, edge_threshold, thr, node_size, smooth,
    prune, parlistfile)
    print("%s%s%s" % ('plot_all (Masking version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_connectogram():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path,
    ID, network, label_names)
    print("%s%s%s" % ('plot_connectogram --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_timeseries():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    time_series = np.load(dir_path + '/997_Default_wb_net_ts.npy')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_timeseries(time_series, network, ID, dir_path, atlas_select, labels)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_rois_gt_100():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.95.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    out_path_fig = '/tmp/fig'

    start_time = time.time()
    plotting.plot_conn_mat(conn_matrix, label_names, out_path_fig)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_rois_lt_100():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.95.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    out_path_fig = '/tmp/'

    start_time = time.time()
    plotting.plot_conn_mat(conn_matrix, label_names, out_path_fig)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))


def test_plot_conn_mat_struct():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.95.txt')
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    ID = '997'
    network = None
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    mask = None
    thr = 0.95
    node_size = 2
    smooth = 2

    plotting.plot_conn_mat_struct(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, thr,
                                  node_size, smooth)

# def test_structural_plotting():
#     base_dir = str(Path(__file__).parent/"examples")
#     #base_dir = '/Users/ryanhammonds/Applications/PyNets/tests/examples'
#     dir_path = base_dir + '/997'
#     conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.95.txt')
#     labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
#     labels_file = open(labels_file_path,'rb')
#     label_names = pickle.load(labels_file)
#     atlas_select = 'whole_brain_cluster_labels_PCA200'
#     ID = '002'
#     bedpostx_dir = base_dir + 'bedpostx_s002.bedpostX'
#     network = None
#     parc = True
#     mask = None
#     coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
#     coord_file = open(coord_file_path, 'rb')
#     coords = pickle.load(coord_file)
#     conn_model = 'sps'
#     thr = 0.95
#     node_size = 2
#     smooth = 2
#
#     plotting.structural_plotting(conn_matrix, label_names, atlas_select, ID, bedpostx_dir, network, parc, mask, coords,
#                                  dir_path, conn_model, thr, node_size, smooth)


# def test_plot_graph_measure_hists():
#     df_concat = np.random.rand(4,4)
#     measures = [1, 2, 3]
#     net_pick_file = '/Users/ryanhammonds/tmp/tmp/tmp'
#
#     plotting.plot_graph_measure_hists(df_concat, measures, net_pick_file)
