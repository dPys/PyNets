#!/usr/bin/env python
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
from pynets import utils


def test_export_to_pandas():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    csv_loc = dir_path + '/whole_brain_cluster_labels_PCA200/997_net_metrics_sps_0.9_pDMN_3_bin.csv'
    network = None
    mask = None
    ID = '997'

    outfile = utils.export_to_pandas(csv_loc, ID, network, mask)
    assert outfile is not None


def test_individual_tcorr_clustering():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    clust_mask = dir_path + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    ID='997'
    k = 3
    clust_type = 'kmeans'

    [uatlas_select, atlas_select, clustering, _, _, _] = utils.individual_tcorr_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5)
    assert uatlas_select is not None
    assert atlas_select is not None
    assert clustering is True


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
    mask = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    utils.save_nifti_parcels_map(ID, dir_path, mask, network, net_parcels_map_nifti)


def test_save_ts_to_file():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    mask = None
    network = None
    ID = '997'
    dir_path = base_dir + '/997'
    ts_within_nodes = '/tmp/'

    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)
