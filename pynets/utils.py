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
from nilearn import datasets
from nilearn.image import concat_imgs
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
    
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

def individual_tcorr_clustering(func_file, clust_mask, ID, k, thresh = 0.5):
    import os
    from pynets import utils
    from pynets.clustools import make_image_from_bin_renum, binfile_parcellate, make_local_connectivity_tcorr
    
    print('\nCreating atlas at cluster level ' + str(k) + '...\n')
    working_dir = os.path.dirname(func_file)
    outfile = working_dir + '/rm_tcorr_conn_' + str(ID) + '.npy'
    outfile_parc = working_dir + '/rm_tcorr_indiv_cluster_' + str(ID)
   
    make_local_connectivity_tcorr( func_file, clust_mask, outfile, thresh )
        
    binfile_parcellate(outfile, outfile_parc, int(k))
    
    ##write out for group mean clustering
    binfile=working_dir + '/rm_tcorr_indiv_cluster_' + str(ID) + '_' + str(k) + '.npy'        
    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    atlas_select = mask_name + '_k' + str(k)
    dir_path = utils.do_dir_path(atlas_select, func_file)
    parlistfile = dir_path + '/' + str(ID) + '_' + mask_name + '_parc_k' + str(k) + '.nii.gz'
    make_image_from_bin_renum(parlistfile,binfile,clust_mask)   
    
    return(parlistfile, atlas_select, dir_path)

def assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask):
    ID_dir = str(os.path.dirname(input_file).split('.')[0])
    if mask != None:
        if network != None:
            out_path = ID_dir + '/' + str(atlas_select) + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0])
        else:
            out_path = ID_dir + '/' + str(atlas_select) + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            out_path = ID_dir + '/' + str(atlas_select) + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr)
        else:
            out_path = ID_dir + '/' + str(atlas_select) + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr)
    return out_path

def collect_pandas_df(input_file, atlas_select, clust_mask, k_min, k_max, k, k_step, min_thr, max_thr, step_thr, multi_thr, thr, mask, ID, network, k_clustering, conn_model, in_csv, user_atlas_list, out_file=None):
    import pandas as pd
    import numpy as np
    import os
    import re
    from itertools import chain
     
    if multi_thr==True:
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr), float(max_thr), float(step_thr)),decimals=2).tolist()]
    else:
        iter_thresh = None
        
    net_pickle_mt_list = []
    if k_clustering == 2:
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
        for k in k_list:
            atlas_select = mask_name + '_k' + str(k)
            if iter_thresh is not None:
                for thr in iter_thresh:
                    net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
            else: 
                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
    elif k_clustering == 1:
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
        atlas_select = mask_name + '_k' + str(k)
        if iter_thresh is not None:
            for thr in iter_thresh:
                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
        else:
            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
    elif user_atlas_list:
        for parlistfile in user_atlas_list:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            if iter_thresh is not None:
                for thr in iter_thresh:
                    net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
            else: 
                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
    elif k_clustering == 0 and atlas_select is not None:
        if iter_thresh is not None:
            for thr in iter_thresh:
                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))
        else: 
            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask))            
    
    if len(net_pickle_mt_list) > 1:
        print('\n\n\n' + str(network))
        print(str(net_pickle_mt_list) + '\n\n\n')
        subject_path = os.path.dirname(os.path.dirname(net_pickle_mt_list[0]))
        name_of_network_pickle = 'net_metrics_' + net_pickle_mt_list[0].split('_0.')[0].split('net_metrics_')[1]
        net_pickle_mt_list.sort()
    
        list_ = []
        for file_ in net_pickle_mt_list:
            df = pd.read_pickle(file_)
            try:
                node_cols = [s for s in list(df.columns) if re.search(r'[0-9]',s)]
                df = df.drop(node_cols, axis=1)
            except:
                print('Error: Node column removal failed for mean stats file...')
            list_.append(df)
            
        try:
            ##Concatenate and find mean across dataframes
            list_of_dicts = [cur_df.T.to_dict().values() for cur_df in list_]
            df_concat = pd.concat(list_, axis=1)
            df_concat = pd.DataFrame(list(chain(*list_of_dicts)))
            df_concatted = df_concat.loc[:, df_concat.columns != 'id'].mean().to_frame().transpose()
            df_concatted['id'] = df_concat['id'].head(1)
            df_concatted = df_concatted[df_concatted.columns[::-1]]
            print('\nConcatenating dataframes for ' + str(ID) + '...\n')
            if network:
                df_concatted.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_' + network + '_mean')
                df_concatted.to_csv(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_' + network + '_mean.csv', index = False) 
            else:
                df_concatted.to_pickle(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_mean')
                df_concatted.to_csv(subject_path + '/' + str(ID) + '_' + name_of_network_pickle + '_mean.csv', index = False)  
        except:
            print('\nWARNING: DATAFRAME CONCATENATE FAILED FOR ' + str(ID) + '!\n')
            pass
    else:
        print('\nNo Dataframe objects to concatenate for ' + str(ID) + '!\n')
        pass

def output_echo(est_path, thr):
    pass
    return(est_path, thr)