#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import os
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pynets.fmri import clustools


def test_make_local_connectivity_tcorr():
    """
    Test_get_neighbors_1d functionality
    """
    print("testing make_local_connectivity_tcorr")
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    mask_file = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    image_file = dir_path + '/002.nii.gz'
    k = 50
    outfile = dir_path + '/variant-tcorr' + str(k) + '_roi.npy'

    outfile = clustools.make_local_connectivity_tcorr(image_file, mask_file, outfile, thresh=0.75)

    assert outfile is not None


def test_make_local_connectivity_scorr():
    """
    Test for make_local_connectivity_scorr functionality
    """
    print("testing make_local_connectivity_scorr")
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    mask_file = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    image_file = dir_path + '/002.nii.gz'
    k = 50
    outfile = dir_path + '/variant-scorr' + str(k) + '_roi.npy'

    outfile = clustools.make_local_connectivity_scorr(image_file, mask_file, outfile, thresh=0.02)

    assert outfile is not None


def test_nil_parcellate():
    """
    Test for scorr connectivity
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    mask_file = base_dir + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    func_file = dir_path + '/002.nii.gz'
    nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']
    k = 50
    mask_name = os.path.basename(mask_file).split('.nii.gz')[0]
    for clust_type in nilearn_clust_list:
        uatlas = "%s%s%s%s%s%s%s%s" % (dir_path, '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz')
        region_labels = clustools.nil_parcellate(func_file, mask_file, k, clust_type, uatlas)

    assert region_labels is not None
