#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets import utils


def test_export_to_pandas():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    csv_loc = dir_path + '/whole_brain_cluster_labels_PCA200/997_net_metrics_sps_0.9_pDMN_3_bin.csv'
    network = None
    roi = None
    ID = '997'

    outfile = utils.export_to_pandas(csv_loc, ID, network, roi)
    assert outfile is not None


def test_save_RSN_coords_and_labels_to_pickle():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    label_names = pickle.load(labels_file)
    network = None

    utils.save_RSN_coords_and_labels_to_pickle(coords, label_names, dir_path, network)


def test_save_nifti_parcels_map():
    import nibabel as nib
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    ID='997'
    dir_path = base_dir + '/997'
    roi = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    utils.save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti)


def test_save_ts_to_file():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    roi = None
    c_boot = 3
    network = None
    ID = '997'
    dir_path = base_dir + '/997'
    ts_within_nodes = '/tmp/'

    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
