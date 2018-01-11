#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@author: PSYC-dap3463

"""
from pathlib import Path
from pynets import plotting
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

def test_plot_conn_mat_nonet_no_mask():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    
    plotting.plot_conn_mat(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask)

def test_plot_conn_mat_nonet_mask():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    
    plotting.plot_conn_mat(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask)

    
def test_plot_all_nonet_no_mask():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_est_sps_0.94.txt')
    edge_threshold='95%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_coords_wb.pkl'
    coord_file = open(coord_file_path,'rb')
    coords = pickle.load(coord_file)
    
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold)

def test_plot_all_nonet_with_mask():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    mask = None
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_est_sps_0.94.txt')
    edge_threshold='95%'
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_coords_wb.pkl'
    coord_file = open(coord_file_path,'rb')
    coords = pickle.load(coord_file)
    
    plotting.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold)

        
def test_plot_connectogram():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    conn_model = 'sps'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    conn_matrix = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_est_sps_0.94.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    
    plotting.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)

def test_plot_timeseries():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    network=None
    ID = '997'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    time_series = np.genfromtxt(dir_path + '/whole_brain_cluster_labels_PCA200/997_wb_net_ts.txt')
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/WB_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)
    
    plotting.plot_timeseries(time_series, network, ID, dir_path, atlas_select, labels)

