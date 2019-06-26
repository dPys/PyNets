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


def test_individual_clustering():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    clust_mask = base_dir + '/pDMN_3_bin.nii.gz'
    ID = '002'
    k = 3
    clust_type = 'kmeans'

    [uatlas_select, atlas_select,
     clustering, _, _, _] = clustools.individual_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5)
    assert uatlas_select is not None
    assert atlas_select is not None
    assert clustering is True
