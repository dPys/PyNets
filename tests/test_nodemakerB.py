#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@author: PSYC-dap3463

"""
from pathlib import Path
from pynets import nodemaker
import numpy as np

def test_WB_fetch_nodes_and_labels1():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc=True

    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    assert dir_path is not None

def test_WB_fetch_nodes_and_labels2():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc=False

    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    assert dir_path is not None

def test_RSN_fetch_nodes_and_labels1():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc=True

    [RSN_label_names, RSN_coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None

def test_RSN_fetch_nodes_and_labels2():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    ref_txt = None
    parc=False

    [RSN_label_names, RSN_coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None
