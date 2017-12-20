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

def collect_pandas_df(allFiles, ID, network, out_file=None):
    import pandas as pd
    import os
    from itertools import chain
    missingness_thresh = 0.10
    frame = pd.DataFrame()
    list_ = []

    subject_path = os.path.dirname(allFiles[0])
    for file_ in allFiles:
        df = pd.read_pickle(file_)
        list_.append(df)

    list_ = list_[0:-1]
    try:
        ##Concatenate and find mean across dataframes
        list_of_dicts = [cur_df.T.to_dict().values() for cur_df in list_]
        df_concat = pd.concat(list_, axis=1)
        df_concat = pd.DataFrame(list(chain(*list_of_dicts)))
        df_concatted = df_concat.loc[:, df_concat.columns != 'id'].mean().to_frame().transpose()
        df_concatted['id'] = df_concat['id'].head(1)
        print('Concatenating for ' + str(ID))
        df_concatted.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_' + net_name + '_mean')
    except:
        print('NO OBJECTS TO CONCATENATE FOR ' + str(ID))
        continue
    
    allFiles = []
    for ID in all_subs:
        atlas_name =  ID + '_' + net_name + '_' + str(clusters) + 'clusters'
        if clusters is None:
            path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + str(ID) + '_' + name_of_network_pickle + '_mean'
        else:
            path_name = working_path + '/' + str(ID) + '/' + atlas_name + '/' + str(ID) + '_' + name_of_network_pickle + '_' + net_name + '_mean'
        if os.path.isfile(path_name):
            print(path_name)
            allFiles.append(path_name)
    
    allFiles.sort()
    
    frame = pd.DataFrame()
    list_ = []
    
    for file_ in allFiles:
        try:
            df = pd.read_pickle(file_)
            bad_cols = [c for c in df.columns if str(c).isdigit()]
            for j in bad_cols:
                df = df.drop(j, 1)
            df_no_id = df.loc[:, df.columns != 'id']
            new_names = [(i, network + '_' + i) for i in df_no_id.iloc[:, 1:].columns.values]
            df.rename(columns = dict(new_names), inplace=True)
            list_.append(df)
        except:
            print('File: ' + file_ + ' is corrupted!')
            continue
    
    list_of_dicts = [cur_df.T.to_dict().values() for cur_df in list_]
    frame = pd.DataFrame(list(chain(*list_of_dicts)))
    
    nas_list=[]
    for column in frame:
        thresh=float(np.sum(frame[column].isnull().values))/len(frame)
        if thresh > missingness_thresh:
            nas_list.append(str(frame[column].name).encode().decode('utf-8'))
            print('Removing ' + str(frame[column].name.encode().decode('utf-8')) + ' due to ' + str(round(100*(frame[column].isnull().sum())/len(frame),1)) + '% missing data...')
    
    ##Remove variables that have too many NAs
    for i in range(len(nas_list)):
        try:
            ##Delete those variables with high correlations
            frame.drop(nas_list[i], axis=0, inplace=True)
        except:
            pass
    
    ##Fix column order
    frame = frame[frame.columns[::-1]]
    
    ##Replace zeroes with nan
    try:
        frame[frame == 0] = np.nan
    except:
        pass
    
    out_path = working_path + '/' + name_of_network_pickle + '_' + net_name + '_' + str(clusters) + '_output.csv'
    frame.to_csv(out_path, index=False)    

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
    
def spt_constr_clustering(ID, clust_mask, k, func_file):
    from pynets import nodemaker
    from math import sqrt
    from nipype.interfaces.fsl import Cluster
    from nilearn.image import new_img_like
    from subprocess import (PIPE, Popen)
    from nilearn.regions import connected_label_regions
    def invoke(command):
        return Popen(command, stdout=PIPE, shell=True).stdout.read()
    
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
        
    ##Get spatially distinct clusters as separate volumes using Gaussian Random Field Theory (GRFT)
    cl = Cluster()
    clust_command = FSLDIR + '/bin/'
    cl.inputs.threshold = 0.5
    cl.inputs.out_index_file = 'clusters.nii.gz'
    cl.inputs.in_file = clust_mask
    clst_cmd_final = clust_command + cl.cmdline
    pre_file_path = clst_cmd_final.split('--in=')[0].split(' --oindex')[0]
    post_file_path = clst_cmd_final.split('--in=')[1].split(' --oindex')[1]
    in_file_path = clst_cmd_final.split('--in=')[1].split(' --oindex')[0]
    in_file_path_quotes = '"%s"' % (in_file_path)
    cmd = pre_file_path + '--in=' + in_file_path_quotes + ' --oindex' + post_file_path
    invoke(cmd)  
    cluster_file_out = cl.inputs.out_index_file
    region_labels = connected_label_regions(cluster_file_out, min_size=3)
    
    ##Further parcellate large clusters using hierarchical clustering
    bna_data = np.round(region_labels.get_data(),1)
    ##Get an array of unique parcels
    bna_data_for_coords_uniq = np.unique(bna_data)
    ##Number of parcels:
    par_max = len(bna_data_for_coords_uniq) - 1
    bna_data = bna_data.astype('int16')
    img_stack = []
    for idx in range(1, par_max+1):
        roi_img = bna_data == bna_data_for_coords_uniq[idx].astype('int16')
        roi_img = roi_img.astype('int16')
        img_stack.append(roi_img)
    img_stack = np.array(img_stack).astype('int16')
    img_list = []
    for idy in range(par_max):
        roi_img_nifti = new_img_like(region_labels, img_stack[idy])
        img_list.append(roi_img_nifti)
    
    labels_list = []
    for mask_img in img_list:
        mask_data = mask_img.get_data().astype(bool)       
        mask_feats = np.sum(mask_data == True)
        if mask_feats < 10:
            continue
        elif mask_feats < 50:
            labels_list.append(mask_img)
        else:
            shape = mask_data.shape
            nifti_masker = input_data.NiftiMasker(mask_img = mask_img, memory='joblib.Memory',
                                                  memory_level=1,
                                                  standardize=False)
            fmri_masked = nifti_masker.fit_transform(func_file)
            
            
            connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                               n_z=shape[2], mask=mask_data)
            
            K = int(np.round(sqrt(mask_feats),1))
            start = time.time()
            ward = FeatureAgglomeration(n_clusters= K, connectivity=connectivity,
                                        linkage='ward', memory='joblib.Memory')
            ward.fit(fmri_masked)
            print('Ward agglomeration for subject ' + str(ID) + ' with '+ str(k) + ' clusters: %.2fs' % (time.time() - start))
        
            labels = ward.labels_ + 1
            labels_img = nifti_masker.inverse_transform(labels)
            
            labels_list.append(labels_img)

    ##Reorder all resulting clusters into single list
    img_list = []
    for bna_img in labels_list:
        bna_data = np.round(bna_img.get_data(),1)
        ##Get an array of unique parcels
        bna_data_for_coords_uniq = np.unique(bna_data)
        ##Number of parcels:
        par_max = len(bna_data_for_coords_uniq) - 1
        bna_data = bna_data.astype('int16')
        img_stack = []
        for idx in range(1, par_max+1):
            roi_img = bna_data == bna_data_for_coords_uniq[idx].astype('int16')
            roi_img = roi_img.astype('int16')
            img_stack.append(roi_img)
        img_stack = np.array(img_stack).astype('int16')
        for idy in range(par_max):
            roi_img_nifti = new_img_like(bna_img, img_stack[idy])
            img_list.append(roi_img_nifti)
    
    ##Create modified atlas
    [labels_img, _] = nodemaker.create_parcel_atlas(img_list)
        
    #first_plot = plotting.plot_roi(labels_img, mask_img, title="Ward parcellation", display_mode='xz')
    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    n_clust = len(img_list)
    os.remove(cluster_file_out)
    atlas_select = mask_name + '_clust_' + str(n_clust)
    dir_path = do_dir_path(atlas_select, func_file)
    parlistfile = dir_path + '/' + str(ID) + '_' + mask_name + '_parc_clust_' + str(k) + '.nii.gz'
    labels_img.to_filename(parlistfile)
    print('Saving file to: ' + str(parlistfile) + '\n')
    return(parlistfile, atlas_select, dir_path)