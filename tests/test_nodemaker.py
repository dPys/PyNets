#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@author: PSYC-dap3463

"""
import numpy as np
import time
from pathlib import Path
from pynets import nodemaker


def test_nodemaker_tools_parlistfile_RSN():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    network = 'Default'
    parc = True

    start_time = time.time()
    [coords, _, _, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    start_time = time.time()
    [net_coords, net_parcel_list, net_label_names, network] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list)
    print("%s%s%s" % ('create_parcel_atlas --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_label_names, dir_path)
    print("%s%s%s" % ('gen_network_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None
    assert net_parcel_list is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert network is not None


def test_nodemaker_tools_nilearn_coords_RSN():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    network = 'Default'
    atlas_select = 'coords_dosenbach_2010'
    parc = False
    parcel_list = None

    start_time = time.time()
    [coords, _, _, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords, _, net_label_names, network] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None
    assert network is not None


def test_nodemaker_tools_masking_parlistfile_RSN():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    network = 'Default'
    ID = '997'
    perc_overlap = 0.10
    parc = True

    start_time = time.time()
    [coords, _, _, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    start_time = time.time()
    [net_coords, net_parcel_list, net_label_names, network] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords_masked, net_label_names_masked, net_parcel_list_masked] = nodemaker.parcel_masker(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID, perc_overlap)
    print("%s%s%s" % ('parcel_masker --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list_masked)
    print("%s%s%s" % ('create_parcel_atlas --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_label_names_masked, dir_path)
    print("%s%s%s" % ('gen_network_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None
    assert net_parcel_list is not None
    assert net_coords_masked is not None
    assert net_label_names_masked is not None
    assert net_parcel_list_masked is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert network is not None


def test_nodemaker_tools_masking_coords_RSN():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    atlas_select = 'coords_dosenbach_2010'
    network='Default'
    parc = False
    parcel_list = None
    error = 2

    start_time = time.time()
    [coords, _, _, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Masking RSN version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords, _, net_label_names, network] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    print("%s%s%s" % ('get_node_membership (Masking RSN version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords_masked, net_label_names_masked] = nodemaker.coord_masker(mask, net_coords, net_label_names, error)
    print("%s%s%s" % ('coord_masker (Masking RSN version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_coords_masked is not None
    assert net_label_names is not None
    assert net_label_names_masked is not None
    assert network is not None


def test_nodemaker_tools_parlistfile_WB():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'

    start_time = time.time()
    [WB_coords, _, _, WB_parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels (User-atlas whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    WB_label_names = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()

    start_time = time.time()
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list)
    print("%s%s%s" % ('create_parcel_atlas (User-atlas whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_label_names is not None
    assert WB_parcel_list is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None


def test_nodemaker_tools_nilearn_coords_WB():
    # Set example inputs
    atlas_select = 'coords_dosenbach_2010'

    start_time = time.time()
    [WB_coords, _, _, WB_label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_label_names is not None


def test_nodemaker_tools_masking_parlistfile_WB():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    ID = '997'
    parc = True
    perc_overlap = 0.10

    start_time = time.time()
    [WB_coords, _, _, WB_parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels (Masking whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    WB_label_names = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()

    start_time = time.time()
    [_, _, WB_parcel_list_masked] = nodemaker.parcel_masker(mask, WB_coords, WB_parcel_list, WB_label_names, dir_path, ID, perc_overlap)
    print("%s%s%s" % ('parcel_masker (Masking whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list_masked)
    print("%s%s%s" % ('create_parcel_atlas (Masking whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [WB_net_parcels_map_nifti_unmasked, WB_coords_unmasked, _] = nodemaker.node_gen(WB_coords, WB_parcel_list, WB_label_names, dir_path, ID, parc)
    print("%s%s%s" % ('node_gen (Masking whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [WB_net_parcels_map_nifti_masked, WB_coords_masked, WB_label_names_masked] = nodemaker.node_gen_masking(mask, WB_coords, WB_parcel_list, WB_label_names, dir_path, ID, parc)
    print("%s%s%s" % ('node_gen_masking (Masking whole-brain version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_label_names is not None
    assert WB_parcel_list is not None
    assert WB_coords_masked is not None
    assert WB_label_names_masked is not None
    assert WB_parcel_list_masked is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert WB_net_parcels_map_nifti_unmasked is not None
    assert WB_coords_unmasked is not None
    assert WB_net_parcels_map_nifti_masked is not None
    assert WB_coords_masked is not None


def test_nodemaker_tools_masking_coords_WB():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/997'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    atlas_select = 'coords_dosenbach_2010'
    error = 2

    start_time = time.time()
    [WB_coords, _, _, WB_label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Masking whole-brain coords version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [WB_coords_masked, WB_label_names_masked] = nodemaker.coord_masker(mask, WB_coords, WB_label_names, error)
    print("%s%s%s" % ('coord_masker (Masking whole-brain coords version) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_coords is not None
    assert WB_coords_masked is not None
    assert WB_label_names is not None
    assert WB_label_names_masked is not None


def test_WB_fetch_nodes_and_labels1():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc = True

    start_time = time.time()
    [_, coords, atlas_name, _, parcel_list, par_max, parlistfile, dir_path] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)
    print("%s%s%s" % ('WB_fetch_nodes_and_labels (Parcel Nodes) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    assert dir_path is not None


def test_WB_fetch_nodes_and_labels2():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc = False

    start_time = time.time()
    [_, coords, atlas_name, _, parcel_list, par_max, parlistfile, dir_path] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)
    print("%s%s%s" % ('WB_fetch_nodes_and_labels (Spherical Nodes) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    assert dir_path is not None


def test_RSN_fetch_nodes_and_labels1():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc = True

    start_time = time.time()
    [RSN_label_names, RSN_coords, atlas_name, _, parcel_list, par_max, parlistfile, _] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)
    print("%s%s%s" % ('RSN_fetch_nodes_and_labels (Parcel Nodes) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None


def test_RSN_fetch_nodes_and_labels2():
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc = False

    start_time = time.time()
    [RSN_label_names, RSN_coords, atlas_name, _, parcel_list, par_max, parlistfile, _] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)
    print("%s%s%s" % ('RSN_fetch_nodes_and_labels (Spherical Nodes) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None
