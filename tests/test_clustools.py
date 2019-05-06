#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.fmri import clustools

def test_individual_tcorr_clustering():
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/rxh180012/PyNets-development/tests/examples'
    dir_path = base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    clust_mask = dir_path + '/triple_net_ICA_overlap_3_sig_bin.nii.gz'
    ID='997'
    k = 3
    clust_type = 'kmeans'

    [uatlas_select, atlas_select,
     clustering, _, _, _] = clustools.individual_tcorr_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5)
    assert uatlas_select is not None
    assert atlas_select is not None
    assert clustering is True
