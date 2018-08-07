#!/usr/bin/env python

import numpy as np
import time
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets import utils


def test_nilearn_atlas_helper():
    atlases = ['atlas_aal', 'atlas_allen_2011', 'atlas_talairach_tissue', 'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe', 'atlas_talairach_hemisphere', 'atlas_harvard_oxford', 'atlas_basc_multiscale_2015', 'atlas_craddock_2012', 'atlas_destrieux_2009', 'coords_dosenbach_2010', 'coords_power_2011']
    label_names_list = []
    networks_list_list = []
    parlistfile_list = []
    for atlas_select in atlases:
        [label_names, networks_list, parlistfile] = utils.nilearn_atlas_helper(atlas_select)
        print(label_names)
        print(networks_list)
        print(parlistfile)
        label_names_list.append(label_names)
        networks_list_list.append(networks_list)
        parlistfile_list.append(parlistfile)

    labs_length = len(label_names_list)
    nets_length = len(networks_list_list)
    par_length = len(parlistfile_list)
    atlas_length = len(atlases)
    assert labs_length is atlas_length
    assert nets_length is atlas_length
    assert par_length is atlas_length


def test_export_to_pandas():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    csv_loc = dir_path + '/whole_brain_cluster_labels_PCA200/997_net_metrics_sps_0.9_pDMN_3_bin.csv'
    network = None
    mask = None
    ID = '997'

    outfile = utils.export_to_pandas(csv_loc, ID, network, mask)


def test_individual_tcorr_clustering():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    clust_mask = dir_path + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    ID='997'
    k = 3

    [parlistfile, atlas_select, dir_path] = utils.individual_tcorr_clustering(func_file, clust_mask, ID, k, thresh=0.5)
    assert parlistfile is not None
    assert atlas_select is not None
    assert dir_path is not None


def test_save_RSN_coords_and_labels_to_pickle():
    base_dir = str(Path(__file__).parent/"examples")
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
    mask = None
    network = None
    ID = '997'
    dir_path = base_dir + '/997'
    ts_within_nodes = '/tmp/'

    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)
