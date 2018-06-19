#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@author: PSYC-dap3463

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
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    label_names = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_conn_mat_func(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, thr, node_size)
    print("%s%s%s" % ('plot_conn_mat_func --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

def test_plot_conn_mat_nonet_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_conn_mat_func(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, thr, node_size)
    print("%s%s%s" % ('plot_conn_mat_func (Masking version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

def test_plot_all_nonet_no_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
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
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold, thr, node_size)
    print("%s%s%s" % ('plot_all --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

def test_plot_all_nonet_with_mask():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    thr = 0.95
    node_size = 2
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
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
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold, thr, node_size)
    print("%s%s%s" % ('plot_all (Masking version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

def test_plot_connectogram():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
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
    plotting.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)
    print("%s%s%s" % ('plot_connectogram --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

def test_plot_timeseries():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    network = None
    ID = '997'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    time_series = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_Default_net_ts.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)

    start_time = time.time()
    plotting.plot_timeseries(time_series, network, ID, dir_path, atlas_select, labels)
    print("%s%s%s" % ('plot_timeseries --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
