# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""

import sys
import os
import numpy as np
#warnings.simplefilter("ignore")
from nilearn.connectome import ConnectivityMeasure
from nilearn import input_data
from sklearn.covariance import GraphLassoCV
try:
    from brainiak.fcma.util import compute_correlation
except ImportError:
    pass
from inverse_covariance import QuicGraphLasso, QuicGraphLassoCV, QuicGraphLassoEBIC, AdaptiveGraphLasso

def get_conn_matrix(time_series, conn_model):
    if conn_model == 'corr':
        # credit: nilearn
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'corr_fast':
        # credit: brainiak
        try:
            conn_matrix = compute_correlation(time_series,time_series)
        except RuntimeError:
            print('Cannot run accelerated correlation computation due to a missing dependency. You need brainiak installed!')
    elif conn_model == 'partcorr':
        # credit: nilearn
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'tangent':
        # credit: nilearn
        conn_measure = ConnectivityMeasure(kind='tangent')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'sps':
        ##Fit estimator to matrix to get sparse matrix
        estimator = GraphLassoCV()
        try:
            print("Fitting Lasso Estimator...")
            estimator.fit(time_series)
        except RuntimeError:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            from sklearn.covariance import GraphLasso, empirical_covariance, shrunk_covariance
            emp_cov = empirical_covariance(time_series)
            for i in np.arange(0.8, 0.99, 0.01):
                shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                alphaRange = 10.0 ** np.arange(-8,0)
                for alpha in alphaRange:
                    try:
                        estimator_shrunk = GraphLasso(alpha)
                        estimator_shrunk.fit(shrunk_cov)
                        print("Calculated graph-lasso covariance matrix for alpha=%s"%alpha)
                        break
                    except FloatingPointError:
                        print("Failed at alpha=%s"%alpha)
                if estimator_shrunk == None:
                    pass
                else:
                    break
                print('Unstable Lasso estimation. Try again!')
                sys.exit()

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
    elif conn_model == 'QuicGraphLasso':
        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        model.fit(time_series)
        conn_matrix = model.precision_
        
    elif conn_model == 'QuicGraphLassoCV':
        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphLassoCV(
            init_method='cov',
            verbose=1)
        model.fit(time_series)
        conn_matrix = model.precision_
    
    elif conn_model == 'QuicGraphLassoEBIC':
        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphLassoEBIC(
            init_method='cov',
            verbose=1)
        model.fit(time_series)
        conn_matrix = model.precision_
    
    elif conn_model == 'AdaptiveQuicGraphLasso':
        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveGraphLasso(
                estimator=QuicGraphLassoEBIC(
                    init_method='cov',
                ),
                method='binary',
            )
        model.fit(time_series)
        conn_matrix = model.estimator_.precision_

    return(conn_matrix)

def extract_ts_wb_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network):
    ##extract time series from whole brain parcellaions:
    parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0, memory='joblib.Memory', memory_level=10, standardize=True)
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    print('\n' + 'Time series has {0} samples'.format(ts_within_nodes.shape[0]) + ' and ' + str(len(coords)) + ' volumetric ROI\'s\n')
    ##Save time series as txt file
    if mask is None:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + network + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_wb_net_ts.txt'
    else:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_wb_net_ts.txt'
    np.savetxt(out_path_ts, ts_within_nodes)
    return(ts_within_nodes)
    
def extract_ts_wb_coords(node_size, conf, func_file, coords, dir_path, ID, mask, thr, network):
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True, memory='joblib.Memory', memory_level=10, standardize=True)
    ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)
    print('\n' + 'Time series has {0} samples'.format(ts_within_nodes.shape[0]) + ' and ' + str(len(coords)) + ' coordinate ROI\'s\n')
    ##Save time series as txt file
    if mask is None:
        out_path_ts=dir_path + '/' + ID + '_wb_net_ts.txt'
    else:
        out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_net_ts.txt'
    np.savetxt(out_path_ts, ts_within_nodes)
    return(ts_within_nodes)