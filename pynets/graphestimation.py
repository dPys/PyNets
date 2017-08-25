import sys
import argparse
import os
import nilearn
import numpy as np
import networkx as nx
import pandas as pd
import nibabel as nib
import seaborn as sns
import numpy.linalg as npl
import matplotlib
import sklearn
import matplotlib
import warnings
import pynets
#warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib import colors
from nipype import Node, Workflow
from nilearn import input_data, masking, datasets
from nilearn import plotting as niplot
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nibabel.affines import apply_affine
from nipype.interfaces.base import isdefined, Undefined
from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits
try:
    import brainiak
    from brainiak.fcma.util import compute_correlation
except ImportError:
    pass

def get_conn_matrix(time_series, conn_model, NETWORK, ID, dir_path, thr):
    if conn_model == 'corr':
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
        est_path = dir_path + '/' + ID + '_est_corr' + '_' + str(thr) + '.txt'
    elif conn_model == 'corr_fast':
        try:
            conn_matrix = compute_correlation(time_series,time_series)
            est_path = dir_path + '/' + ID + '_est_corr_fast' + '_' + str(thr) + '.txt'
        except RuntimeError:
            print('Cannot run accelerated correlation computation due to a missing dependency. You need brainiak installed!')
    elif conn_model == 'partcorr':
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
        est_path = dir_path + '/' + ID + '_est_part_corr' + '_' + str(thr) + '.txt'
    elif conn_model == 'cov' or conn_model == 'sps':
        ##Fit estimator to matrix to get sparse matrix
        estimator = GraphLassoCV()
        try:
            print("Fitting Lasso estimator...")
            est = estimator.fit(time_series)
        except RuntimeError:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            #from sklearn.covariance import GraphLasso, empirical_covariance, shrunk_covariance
            #emp_cov = empirical_covariance(time_series)
            #for i in np.arange(0.8, 0.99, 0.01):
                #shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                #alphaRange = 10.0 ** np.arange(-8,0)
                #for alpha in alphaRange:
                    #try:
                        #estimator_shrunk = GraphLasso(alpha)
                        #est=estimator_shrunk.fit(shrunk_cov)
                        #print("Calculated graph-lasso covariance matrix for alpha=%s"%alpha)
                        #break
                    #except FloatingPointError:
                        #print("Failed at alpha=%s"%alpha)
            #if estimator_shrunk == None:
                #pass
            #else:
                #break
            print('Unstable Lasso estimation. Try again!')
            sys.exit()

        if NETWORK != None:
            est_path = dir_path + '/' + ID + '_' + NETWORK + '_est%s'%('_sps_inv' if conn_model=='sps' else 'cov') + '_' + str(thr) + '.txt'
        else:
            est_path = dir_path + '/' + ID + '_est%s'%('_sps_inv' if conn_model=='sps' else 'cov') + '_' + str(thr) + '.txt'
        if conn_model == 'sps':
            try:
                conn_matrix = -estimator.precision_
            except:
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov':
            try:
                conn_matrix = estimator.covariance_
            except:
                conn_matrix = estimator_shrunk.covariance_
    np.savetxt(est_path, conn_matrix, delimiter='\t')
    return(conn_matrix, est_path)
