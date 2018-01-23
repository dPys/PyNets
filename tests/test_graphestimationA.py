#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:26:34 2017

@author: PSYC-dap3463
"""
import numpy as np
import nibabel as nib
from pynets import graphestimation
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

def test_get_conn_matrix():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    time_series_file = dir_path + '/coords_power_2011/997_wb_net_ts.txt'
    time_series = np.genfromtxt(time_series_file)
    conn_model_list = ['sps', 'cov', 'corr', 'partcorr']
    for conn_model in conn_model_list:
        conn_matrix = graphestimation.get_conn_matrix(time_series, conn_model)
        assert conn_matrix is not None
