#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import sys
import os
import nibabel as nib
import numpy as np


def nilearn_atlas_helper(atlas_select):
    from nilearn import datasets
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
    return label_names, networks_list, parlistfile


##save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, network, mask, out_file=None):
    import pandas as pd
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    ##Check for existence of csv_loc
    if os.path.isfile(csv_loc) == False:
        raise FileNotFoundError('\n\n\nERROR: Missing netmetrics csv file output. Cannot export to pandas df!')

    if mask is not None:
        if network is not None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network + '_' + str(os.path.basename(mask).split('.')[0])
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network is not None:
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
    df.to_pickle(out_file, protocol=2)
    return out_file


def do_dir_path(atlas_select, in_file):
    dir_path = os.path.dirname(os.path.realpath(in_file)) + '/' + atlas_select
    if not os.path.exists(dir_path) and atlas_select is not None:
        os.makedirs(dir_path)
    elif atlas_select is None:
        raise ValueError("Error: cannot create directory for a null atlas!")
    return dir_path


def create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size):
    if mask is not None:
        if network is not None:
            est_path = dir_path + '/' + str(ID) + '_' + network + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size) + '.npy'
        else:
            est_path = dir_path + '/' + str(ID) + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size) + '.npy'
    else:
        if network is not None:
            est_path = dir_path + '/' + str(ID) + '_' + network + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.npy'
        else:
            est_path = dir_path + '/' + str(ID) + '_est_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.npy'
    return est_path


def create_unthr_path(ID, network, conn_model, mask, dir_path):
    if mask is not None:
        if network is not None:
            unthr_path = dir_path + '/' + str(ID) + '_' + network + '_est_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_unthresh_mat.npy'
        else:
            unthr_path = dir_path + '/' + str(ID) + '_est_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_unthresh_mat.npy'
    else:
        if network is not None:
            unthr_path = dir_path + '/' + str(ID) + '_' + network + '_est_' + str(conn_model) + '_unthresholded_mat.npy'
        else:
            unthr_path = dir_path + '/' + str(ID) + '_est_' + str(conn_model) + '_unthresh_mat.npy'
    return unthr_path


def create_csv_path(ID, network, conn_model, thr, mask, dir_path, node_size):
    if mask is not None:
        if network is not None:
            out_path = dir_path + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size) + '.csv'
        else:
            out_path = dir_path + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size) + '.csv'
    else:
        if network is not None:
            out_path = dir_path + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size) + '.csv'
        else:
            out_path = dir_path + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size) + '.csv'
    return out_path


def individual_tcorr_clustering(func_file, clust_mask, ID, k, thresh = 0.5):
    import os
    from pynets import utils
    from pynets.clustools import make_image_from_bin_renum, binfile_parcellate, make_local_connectivity_tcorr

    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    atlas_select = mask_name + '_k' + str(k)
    print('\nCreating atlas at cluster level ' + str(k) + ' for ' + str(atlas_select) + '...\n')
    working_dir = os.path.dirname(func_file) + '/' + atlas_select
    outfile = working_dir + '/rm_tcorr_conn_' + str(ID) + '.npy'
    outfile_parc = working_dir + '/rm_tcorr_indiv_cluster_' + str(ID)
    binfile=working_dir + '/rm_tcorr_indiv_cluster_' + str(ID) + '_' + str(k) + '.npy'
    dir_path = utils.do_dir_path(atlas_select, func_file)
    parlistfile = dir_path + '/' + mask_name + '_k' + str(k) + '.nii.gz'

    make_local_connectivity_tcorr( func_file, clust_mask, outfile, thresh )

    binfile_parcellate(outfile, outfile_parc, int(k))

    ##write out for group mean clustering
    make_image_from_bin_renum(parlistfile,binfile,clust_mask)

    return parlistfile, atlas_select, dir_path


def assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size):
    nilearn_parc_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    nilearn_coord_atlases=['harvard_oxford', 'msdl', 'coords_power_2011', 'smith_2009', 'basc_multiscale_2015', 'allen_2011', 'coords_dosenbach_2010']
    if conn_model == 'prob':
        ID_dir = str(os.path.dirname(input_file))
    else:
        ID_dir = str(os.path.dirname(input_file).split('.')[0])
    if mask is not None:
        if network is not None:
            if atlas_select in nilearn_parc_atlases or atlas_select in nilearn_coord_atlases:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size)
            else:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size)
        else:
            if atlas_select in nilearn_parc_atlases or atlas_select in nilearn_coord_atlases:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size)
            else:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(node_size)
    else:
        if network is not None:
            if atlas_select in nilearn_parc_atlases or atlas_select in nilearn_coord_atlases:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size)
            else:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size)
        else:
            if atlas_select in nilearn_parc_atlases or atlas_select in nilearn_coord_atlases:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size)
            else:
                out_path = ID_dir + '/' + str(atlas_select) + '/' + str(ID) + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(node_size)
    return out_path


def collect_pandas_df(input_file, atlas_select, clust_mask, k_min, k_max, k, k_step, min_thr, max_thr, step_thr, multi_thr, thr, mask, ID, network, k_clustering, conn_model, in_csv, user_atlas_list, clust_mask_list, multi_atlas, node_size, node_size_list, parc, out_file=None):
    import pandas as pd
    import numpy as np
    import os
    from itertools import chain
    from pynets import utils
    from pynets.utils import assemble_mt_path

    if parc is True:
        node_size = 'parc'

    if multi_thr is True:
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr), float(max_thr), float(step_thr)),decimals=2).tolist()]
    else:
        iter_thresh = None

    net_pickle_mt_list = []
    if k_clustering == 4:
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        for clust_mask in clust_mask_list:
            mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            for k in k_list:
                atlas_select = mask_name + '_k' + str(k)
                if node_size_list:
                    for node_size in node_size_list:
                        if iter_thresh is not None:
                            for thr in iter_thresh:
                                try:
                                    net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                                except RuntimeWarning:
                                    print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                                    pass
                        else:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for K=' + str(k))
                                pass
                else:
                    if iter_thresh is not None:
                        for thr in iter_thresh:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                                pass
                    else:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k))
                            pass
    elif k_clustering == 2:
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
        for k in k_list:
            atlas_select = mask_name + '_k' + str(k)
            if node_size_list:
                for node_size in node_size_list:
                    if iter_thresh is not None:
                        for thr in iter_thresh:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                                pass
                    else:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k))
                            pass
            else:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for K=' + str(k))
                        pass
    elif k_clustering == 1:
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
        atlas_select = mask_name + '_k' + str(k)
        if node_size_list:
            for node_size in node_size_list:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for K=' + str(k))
                        pass
        else:
            if iter_thresh is not None:
                for thr in iter_thresh:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                        pass
            else:
                try:
                    net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                except RuntimeWarning:
                    print('Missing results path for K=' + str(k))
                    pass
    elif k_clustering == 3:
        for clust_mask in clust_mask_list:
            mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            atlas_select = mask_name + '_k' + str(k)
            if node_size_list:
                for node_size in node_size_list:
                    if iter_thresh is not None:
                        for thr in iter_thresh:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                                pass
                    else:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k))
                            pass
            else:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for K=' + str(k) + ' and thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for K=' + str(k))
                        pass
    elif user_atlas_list and len(user_atlas_list) > 1:
        for parlistfile in user_atlas_list:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            if node_size_list:
                for node_size in node_size_list:
                    if iter_thresh is not None:
                        for thr in iter_thresh:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for atlas=' + str(atlas_select) + ' and thr=' + str(thr))
                                pass
                    else:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for atlas=' + str(atlas_select))
                            pass
            else:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for atlas=' + str(atlas_select) + ' and thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for atlas=' + str(atlas_select))
                        pass
    elif multi_atlas:
        for atlas_select in multi_atlas:
            if node_size_list:
                for node_size in node_size_list:
                    if iter_thresh is not None:
                        for thr in iter_thresh:
                            try:
                                net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                            except RuntimeWarning:
                                print('Missing results path for atlas=' + str(atlas_select) + ' and thr=' + str(thr))
                                pass
                    else:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for atlas=' + str(atlas_select))
                            pass
            else:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for atlas=' + str(atlas_select) + ' and thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for atlas=' + str(atlas_select))
                        pass
    elif k_clustering == 0 and atlas_select is not None:
        if node_size_list:
            for node_size in node_size_list:
                if iter_thresh is not None:
                    for thr in iter_thresh:
                        try:
                            net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                        except RuntimeWarning:
                            print('Missing results path for thr=' + str(thr))
                            pass
                else:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path')
                        pass
        else:
            if iter_thresh is not None:
                for thr in iter_thresh:
                    try:
                        net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                    except RuntimeWarning:
                        print('Missing results path for thr=' + str(thr))
                        pass
            else:
                try:
                    net_pickle_mt_list.append(assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size))
                except RuntimeWarning:
                    print('Missing results path')
                    pass

    ##Check for existence of net_pickle files, condensing final list to only those that were actually produced.
    [net_pickle_mt_list, _] = utils.check_est_path_existence(net_pickle_mt_list)

    if len(net_pickle_mt_list) > 1:
        print('\n\nList of result files to concatenate:\n' + str(net_pickle_mt_list) + '\n\n')
        subject_path = os.path.dirname(os.path.dirname(net_pickle_mt_list[0]))
        name_of_network_pickle = 'net_metrics_' + net_pickle_mt_list[0].split('_0.')[0].split('net_metrics_')[1]
        net_pickle_mt_list.sort()

        list_ = []
        for file_ in net_pickle_mt_list:
            df = pd.read_pickle(file_)
            try:
                node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]
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
        except RuntimeWarning:
            print('\nWARNING: DATAFRAME CONCATENATION FAILED FOR ' + str(ID) + '!\n')
            pass
    else:
        print('\nSingle dataframe for: ' + str(ID) + '\n')
        pass
    return

def build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size):
    import numpy as np
    from pynets import utils
    if multi_thr==True:
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
                                                float(max_thr), float(step_thr)),decimals=2).tolist()]
        for thr in iter_thresh:
            if node_size_list:
                for node_size in node_size_list:
                    est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
                    est_path_list.append(est_path_tmp)
            else:
                est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
                est_path_list.append(est_path_tmp)
                iter_thresh = [thr] * len(est_path_list)
    else:
        if node_size_list:
            for node_size in node_size_list:
                est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
                est_path_list.append(est_path_tmp)
                iter_thresh = [thr] * len(est_path_list)
        else:
            est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
            est_path_list.append(est_path_tmp)
            iter_thresh = [thr] * len(est_path_list)
    return iter_thresh, est_path_list


def build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size):
    from pynets import utils
    if multi_nets is not None:
        num_networks = len(multi_nets)
        for network in multi_nets:
            dir_path = utils.do_dir_path(atlas_select, input_file)
            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
    else:
        num_networks =  1
        dir_path = utils.do_dir_path(atlas_select, input_file)
        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
    return iter_thresh, est_path_list, num_networks, dir_path


def create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh):
    if node_size_list:
        for node_size in node_size_list:
            if multi_thr == True:
                for thr in iter_thresh:
                    network_list.append(network)
            else:
                network_list.append(network)
    else:
        if multi_thr == True:
            for thr in iter_thresh:
                network_list.append(network)
        else:
            network_list.append(network)
    return network_list


def check_est_path_existence(est_path_list):
    import os
    est_path_list_ex = []
    bad_ixs = []
    i = -1
    for est_path in est_path_list:
        i = i + 1
        if os.path.isfile(est_path) == True:
            est_path_list_ex.append(est_path)
        else:
            #print('\n\nWarning: Missing ' + est_path + '...\n\n')
            bad_ixs.append(i)
            continue
    return(est_path_list_ex, bad_ixs)


def save_RSN_coords_and_labels_to_pickle(coords, label_names, dir_path, network):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    ##Save coords to pickle
    coord_path = dir_path + '/' + network + '_func_coords_rsn.pkl'
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    ##Save labels to pickle
    labels_path = dir_path + '/' + network + '_func_labelnames_rsn.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f, protocol=2)
    return


def save_nifti_parcels_map(ID, dir_path, mask, network, net_parcels_map_nifti):
    if mask:
        if network:
            net_parcels_nii_path = dir_path + '/' + str(ID) + '_parcels_masked_' + network + '_' + str(os.path.basename(mask).split('.')[0]) + '.nii.gz'
        else:
            net_parcels_nii_path = dir_path + '/' + str(ID) + '_parcels_masked_' + str(os.path.basename(mask).split('.')[0]) + '.nii.gz'
    else:
        if network:
            net_parcels_nii_path = dir_path + '/' + str(ID) + '_parcels_' + network + '.nii.gz'
        else:
            net_parcels_nii_path = dir_path + '/' + str(ID) + '_parcels.nii.gz'

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return


def cuberoot(x):
    return np.sign(x) * np.abs(x)**(1 / 3)


def compile_iterfields(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list, est_path):
    import numpy as np
    import os
    from pynets import utils

    ##Build iterfields
    if multi_atlas is not None or user_atlas_list is not None or multi_thr is True or multi_nets is not None or k_clustering != 1 or k_clustering != 0 or node_size_list is not None:
        ##Create est_path_list iterfield based on iterables across atlases, RSN's, k-values, and thresholding ranges
        est_path_list = []
        if k_clustering == 2:
            print('\nPipeline iterated for ' + str(ID) + ' across multiple clustering resolutions...\n')
            mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            for k in k_list:
                atlas_select = mask_name + '_k' + str(k)
                [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
        elif k_clustering == 3:
            print('\nPipeline iterated for ' + str(ID) + ' across multiple masks at a single clustering resolution...\n')
            for clust_mask in clust_mask_list:
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                atlas_select = mask_name + '_k' + str(k)
                [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
        elif k_clustering == 4:
            print('\nPipeline iterated for ' + str(ID) + ' across multiple clustering resolutions and masks...\n')
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            for clust_mask in clust_mask_list:
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                for k in k_list:
                    atlas_select = mask_name + '_k' + str(k)
                    [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
        elif multi_atlas is not None:
            print('\nPipeline iterated for ' + str(ID) + ' across multiple atlases: ' + '\n'.join(str(n) for n in multi_atlas) + '...\n')
            for atlas_select in multi_atlas:
                [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
        elif user_atlas_list:
            print('\nPipeline iterated for ' + str(ID) + ' across multiple atlases: ' + '\n'.join(str(a) for a in user_atlas_list) + '...\n')
            for parlistfile in user_atlas_list:
                atlas_select = parlistfile.split('/')[-1].split('.')[0]
                [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
        else:
            if multi_thr is True:
                [iter_thresh, est_path_list, num_networks, dir_path] = utils.build_multi_net_paths(multi_nets, atlas_select, input_file, multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
            else:
                if multi_nets is not None:
                    for network in multi_nets:
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
                else:
                    if node_size_list:
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list, node_size_list, node_size)
                    else:
                        est_path_list = [est_path]
                        iter_thresh = [thr]

        if node_size_list:
            num_node_sizes = len(node_size_list)
        else:
            num_node_sizes = 1

        ##Check existence of each est_path in est_path_list, returning a modified list with only those paths that do exist.
        #[est_path_list, bad_ixs] = utils.check_est_path_existence(est_path_list)

        ##Create network_list based on iterables across atlases, RSN's, k-values, and thresholding ranges
        if multi_nets is not None:
            print('\nPipeline iterated for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
            network_list = []
            if multi_atlas is not None:
                for atlas in multi_atlas:
                    for network in multi_nets:
                        network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
            elif user_atlas_list:
                for atlas in user_atlas_list:
                    for network in multi_nets:
                        network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
            elif k_clustering == 2:
                k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                for k in k_list:
                    for network in multi_nets:
                        network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
            elif k_clustering == 4:
                k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        for network in multi_nets:
                            network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
            elif k_clustering == 3:
                for clust_mask in clust_mask_list:
                    for network in multi_nets:
                        network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
            else:
                for network in multi_nets:
                    network_list = utils.create_net_list(node_size_list, network, network_list, multi_thr, iter_thresh)
        elif network is not None and multi_nets is None:
            network_list = [network] * len(est_path_list)
        else:
            network_list = [None] * len(est_path_list)

        ##Remove network from network_list based on bad ixs found for est_path_list
        #if len(bad_ixs) > 0 and len(network_list) > 1:
            #for i in bad_ixs:
                 #network_list.pop(i)

        if multi_thr is True:
            thr_lst = []
            for path in est_path_list:
                thr_lst.append(str(thr))
            thr = thr_lst
        else:
            thr = iter_thresh

        if num_node_sizes > 1:
            node_size_lst = []
            for path in est_path_list:
                node_size_lst.append(node_size)
            node_size = node_size_lst
        else:
            node_size = [node_size] * len(est_path_list)

        est_path = est_path_list
        network = network_list
        ID = [str(ID)] * len(est_path_list)
        mask = [mask] * len(est_path_list)
        conn_model = [conn_model] * len(est_path_list)
        k_clustering = [k_clustering] * len(est_path_list)
        prune = [prune] * len(est_path_list)

    return est_path, thr, network, ID, mask, conn_model, k_clustering, prune, node_size


def save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes):
    import os
    ##Save time series as txt file
    if mask is None:
        if network is not None:
            out_path_ts="%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_wb_net_ts.npy')
        else:
            out_path_ts="%s%s%s%s" % (dir_path, '/', ID, '_wb_net_ts.npy')
    else:
        if network is not None:
            out_path_ts="%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_', network, '_rsn_net_ts.npy')
        else:
            out_path_ts="%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_wb_net_ts.npy')
    np.save(out_path_ts, ts_within_nodes)
    return
