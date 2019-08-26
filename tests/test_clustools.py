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
import numpy as np


def test_individual_clustering():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    clust_mask = base_dir + '/pDMN_3_bin.nii.gz'
    ID = '002'
    k = 3
    clust_type = 'kmeans'

    [uatlas, atlas,
     clustering, _, _, _] = clustools.individual_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5)
    assert uatlas is not None
    assert atlas is not None
    assert clustering is True
    
    
def test_ncut():
    W = np.random.rand(100,100)
    nbEigenValues = 100
    eigen_val, eigen_vec = clustools.ncut(W, nbEigenValues)
    
    assert eigen_val is not None
    assert eigen_vec is not None
    
def test_discretisation():
    W = np.random.rand(100,100)
    nbEigenValues = 100
    eigen_val, eigen_vec = clustools.ncut(W, nbEigenValues)

    eigenvec_discrete = clustools.discretisation(eigen_vec)
    assert eigenvec_discrete is not None
    

def test_indx_1dto3d():
    sz = np.random.rand(10,10,10)
    idx = 45
    
    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None
    
def test_indx_3dto1d():
    sz = np.random.rand(10,10,10)
    idx = 45
    
    x, y, z = clustools.indx_1dto3d(idx, sz)
    assert x is not None
    assert y is not None
    assert z is not None