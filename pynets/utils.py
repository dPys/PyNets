#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017

@author: PSYC-dap3463
"""
import sys
import os
import nibabel as nib
import pandas as pd
import numpy as np
import time
import sklearn as sk
import scipy as sp
from sklearn import cluster
from nilearn import datasets, input_data
from nilearn.image import concat_imgs
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from sklearn.feature_extraction import image
from sklearn.cluster import FeatureAgglomeration

def nilearn_atlas_helper(atlas_select):
    try:
        parlistfile=getattr(datasets, 'fetch_%s' % atlas_select)().maps
        try:
            label_names=getattr(datasets, 'fetch_%s' % atlas_select)().labels
        except:
            label_names=None
        try:
            networks_list = getattr(datasets, 'fetch_%s' % atlas_select)().networks
        except:
            networks_list = None
    except:
        print('Extraction from nilearn datasets failed!')
        sys.exit()
    return(label_names, networks_list, parlistfile)

def convert_atlas_to_volumes(atlas_path, img_list):
    atlas_dir = os.path.dirname(atlas_path)
    ref_txt = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '.txt'
    fourd_file = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '_4d.nii.gz'
    if os.path.isfile(ref_txt):
        try:
            all4d = concat_imgs(img_list)
            nib.save(all4d, fourd_file)

            ##Save individual 3D volumes as individual files
            volumes_dir = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '_volumes'
            if not os.path.exists(volumes_dir):
                os.makedirs(volumes_dir)

            j = 0
            for img in img_list:
                volume_path = volumes_dir + '/parcel_' + str(j)
                nib.save(img, volume_path)
                j = j + 1

        except:
            print('Image concatenation failed for: ' + str(atlas_path))
            volumes_dir = None
    else:
        print('Atlas reference file not found!')
        volumes_dir = None
    return(volumes_dir)

##save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, network, mask, out_file=None):
    if mask != None:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network + '_' + str(os.path.basename(mask).split('.')[0])
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list'

    metric_list_names = pickle.load(open(met_list_picke_path, 'rb'))
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
    df = df.T
    column_headers={k: v for k, v in enumerate(metric_list_names)}
    df = df.rename(columns=column_headers)
    df['id'] = range(1, len(df) + 1)
    cols = df.columns.tolist()
    ix = cols.index('id')
    cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
    df = df[cols_ID]
    df['id'] = df['id'].astype('object')
    df.id = df.id.replace(1,ID)
    out_file = csv_loc.split('.csv')[0]
    df.to_pickle(out_file)
    return(out_file)

def do_dir_path(atlas_select, in_file):
    dir_path = os.path.dirname(os.path.realpath(in_file)) + '/' + atlas_select
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def create_est_path(ID, network, conn_model, thr, mask, dir_path):
    if mask != None:
        if network != None:
            est_path = dir_path + '/' + ID + '_' + network + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.txt'
        else:
            est_path = dir_path + '/' + ID + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.txt'       
    else:
        if network != None:
            est_path = dir_path + '/' + ID + '_' + network + '_est_' + str(conn_model) + '_' + str(thr) + '.txt'
        else:
            est_path = dir_path + '/' + ID + '_est_' + str(conn_model) + '_' + str(thr) + '.txt'
    return est_path

def create_unthr_path(ID, network, conn_model, mask, dir_path):
    if mask != None:
        if network != None:
            unthr_path = dir_path + '/' + ID + '_' + network + '_est_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_unthresh_mat.txt'
        else:
            unthr_path = dir_path + '/' + ID + '_est_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_unthresh_mat.txt'       
    else:
        if network != None:
            unthr_path = dir_path + '/' + ID + '_' + network + '_est_' + str(conn_model) + '_unthresholded_mat.txt'
        else:
            unthr_path = dir_path + '/' + ID + '_est_' + str(conn_model) + '_unthresh_mat.txt'
    return unthr_path

def create_csv_path(ID, network, conn_model, thr, mask, dir_path):
    if mask != None:
        if network != None:
            out_path = dir_path + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.csv'
    else:
        if network != None:
            out_path = dir_path + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr) + '.csv'
    return out_path

def timeseries_bootstrap(tseries, block_size):
    """
    Adapted from PyBASC: @Aki Nikolaidis
    Generates a bootstrap sample derived from the input time-series.  Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries

    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
       changes in brain connectivity for functional MRI. Statistica Sinica,
       special issue on Statistical Challenges and Advances in Brain Science,
       2008, 18: 1253-1268.
    """

    k = np.ceil(float(tseries.shape[0])/int(block_size))
    r_ind = np.floor(np.random.rand(1,int(k))*tseries.shape[0])
    blocks = np.dot(np.arange(0,int(block_size))[:,np.newaxis], np.ones([1,int(k)]))
    block_offsets = np.dot(np.ones([int(block_size),1]), r_ind)
    block_mask = (blocks + block_offsets).flatten('F')[:tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])
    return tseries[block_mask.astype('int'), :]

def adjacency_matrix(cluster_pred):
    """
    Adapted from PyBASC: @Aki Nikolaidis
    Calculate adjacency matrix for given cluster predictions

    Parameters
    ----------
    cluster_pred : array_like
        A matrix of shape (`N`, `1`) with `N` samples

    Returns
    -------
    A : array_like
        Adjacency matrix of shape (`N`,`N`)
    """
    x = cluster_pred.copy()
    if(len(x.shape) == 1):
        x = x[:, np.newaxis]
    # Force the cluster indexing to be positive integers
    if(x.min() <= 0):
        x += -x.min() + 1
    A = np.dot(x**-1., x.T) == 1
    return A

def cluster_timeseries(X, k, similarity_metric, affinity_threshold, neighbors = 10):
    """
    Adapted from PyBASC: @Aki Nikolaidis
    Cluster a given timeseries

    Parameters
    ----------
    X : array_like
        A matrix of shape (`N`, `M`) with `N` samples and `M` dimensions
    n_clusters : integer
        Number of clusters
    similarity_metric : {'k_neighbors', 'correlation', 'data'}
        Type of similarity measure for spectral clustering.  The pairwise similarity measure
        specifies the edges of the similarity graph. 'data' option assumes X as the similarity
        matrix and hence must be symmetric.  Default is kneighbors_graph [1]_ (forced to be
        symmetric)
    affinity_threshold : float
        Threshold of similarity metric when 'correlation' similarity metric is used.

    Returns
    -------
    y_pred : array_like
        Predicted cluster labels

    References
    ----------
    .. [1] http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.kneighbors_graph.html
    """
    X = np.array(X)
    X_dist = sp.spatial.distance.pdist(X, metric = similarity_metric)
    X_dist = sp.spatial.distance.squareform(X_dist)
    sim_matrix=1-X_dist
    sim_matrix[np.isnan((sim_matrix))]=0
    sim_matrix[sim_matrix<float(affinity_threshold)]=0
    sim_matrix[sim_matrix>1]=1
    spectral = cluster.SpectralClustering(k, eigen_solver='arpack', random_state = 5, affinity="precomputed", assign_labels='discretize')
    spectral.fit(sim_matrix)
    y_pred = spectral.labels_.astype(np.int)
    return y_pred

def ism_clustering(ID, mask, k, dir_path, func_file, n_bootstraps = 100, affinity_threshold = 0.80):
    print('Calculating individual stability matrix from: ' + func_file)
    data = nib.load(func_file).get_data().astype('float64')
    mask_img = nib.load(mask)
    mask_data = mask_img.get_data().astype('float64').astype('bool')
    Y = data[mask_data].T
    Y = sk.preprocessing.normalize(Y, norm='l2')
    samples = Y.shape[0]
    block_size = int(np.sqrt(samples))
    print('Block size: ', block_size)
    S = np.zeros((samples, samples))
    i = 1
    print('Bootstrapping:')
    for bootstrap_i in range(n_bootstraps):
        print(str(i))
        Y_b1 = timeseries_bootstrap(Y, block_size)
        S += adjacency_matrix(cluster_timeseries(Y_b1, k, similarity_metric = 'correlation', affinity_threshold = affinity_threshold)[:,np.newaxis])
        i = i + 1
    S /= n_bootstraps
    S=S.astype("uint8")
  
def ward_clustering(ID, clust_mask, k, func_file):
    k = int(k)
    mask_img = nib.load(clust_mask)
    nifti_masker = input_data.NiftiMasker(mask_img = mask_img, memory='joblib.Memory',
                                          memory_level=1,
                                          standardize=False)
    fmri_masked = nifti_masker.fit_transform(func_file)
    mask_data = nifti_masker.mask_img_.get_data().astype(bool)
    shape = mask_data.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                       n_z=shape[2], mask=mask_data)
    start = time.time()
    ward = FeatureAgglomeration(n_clusters=k, connectivity=connectivity,
                                linkage='ward', memory='joblib.Memory')
    ward.fit(fmri_masked)
    print('Ward agglomeration for subject ' + str(ID) + ' with '+ str(k) + ' clusters: %.2fs' % (time.time() - start))

    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)
    #first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation", display_mode='xz')
    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    atlas_select = mask_name + '_k' + str(k)
    dir_path = do_dir_path(atlas_select, func_file)
    parlistfile = dir_path + '/' + str(ID) + '_' + mask_name + '_parc_k_' + str(k) + '.nii.gz'
    labels_img.to_filename(parlistfile)
    print('Saving file to: ' + str(parlistfile) + '\n')
    return(parlistfile, atlas_select, dir_path)