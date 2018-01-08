# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import numpy as np

def get_conn_matrix(time_series, conn_model):
    import warnings
    warnings.simplefilter("ignore")
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphLassoCV
    try:
        from brainiak.fcma.util import compute_correlation
    except ImportError:
        pass
    
    if conn_model == 'corr':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'corr_fast':
        # credit: brainiak
        try:
            print('\nComputing accelerated fcma correlation matrix...\n')
            conn_matrix = compute_correlation(time_series,time_series)
        except RuntimeError:
            print('Cannot run accelerated correlation computation due to a missing dependency. You need brainiak installed!')
    elif conn_model == 'partcorr':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'tangent':
        # credit: nilearn
        print('\nComputing tangent matrix...\n')
        conn_measure = ConnectivityMeasure(kind='tangent')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'sps':
        ##Fit estimator to matrix to get sparse matrix
        estimator = GraphLassoCV()
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            try:
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
            except:
                raise ValueError('Unstable Lasso estimation! Shrinkage failed.')

        if conn_model == 'sps':
            try:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            except:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov':
            try:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            except:
                conn_matrix = estimator_shrunk.covariance_    
    elif conn_model == 'QuicGraphLasso':
        from inverse_covariance import QuicGraphLasso
        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
        
    elif conn_model == 'QuicGraphLassoCV':
        from inverse_covariance import QuicGraphLassoCV
        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    
    elif conn_model == 'QuicGraphLassoEBIC':
        from inverse_covariance import QuicGraphLassoEBIC
        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    
    elif conn_model == 'AdaptiveQuicGraphLasso':
        from inverse_covariance import AdaptiveGraphLasso, QuicGraphLassoEBIC
        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveGraphLasso(
                estimator=QuicGraphLassoEBIC(
                    init_method='cov',
                ),
                method='binary',
            )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_

    return(conn_matrix)

def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network):
    import os
    from nilearn import input_data
    ##extract time series from whole brain parcellaions:
    parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0, standardize=True)
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    print('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]) + ' and ' + str(len(coords)) + ' volumetric ROI\'s\n')
    ##Save time series as txt file
    if mask is None:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + network + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_wb_net_ts.txt'
    else:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + network + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_wb_net_ts.txt'
    np.savetxt(out_path_ts, ts_within_nodes)
    return(ts_within_nodes)
    
def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network):
    import os
    from nilearn import input_data
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True, standardize=True)
    ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)
    print('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]) + ' and ' + str(len(coords)) + ' coordinate ROI\'s\n')
    ##Save time series as txt file
    if mask is None:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + network + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_wb_net_ts.txt'
    else:
        if network is not None:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + network + '_rsn_net_ts.txt'
        else:
            out_path_ts=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_wb_net_ts.txt'
    np.savetxt(out_path_ts, ts_within_nodes)
    return(ts_within_nodes)