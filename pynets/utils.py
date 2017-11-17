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
from pathlib import Path
from nilearn import datasets, input_data
from nilearn.image import concat_imgs
from pynets import nodemaker, thresholding, graphestimation, plotting
from nilearn import plotting as niplot
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
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_WB' + '_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_WB'

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
    df['id'].values[0] = ID
    out_file = csv_loc.split('.csv')[0]
    df.to_pickle(out_file)
    return(out_file)

def WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc):
    ##Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    if atlas_select in nilearn_atlases:
        [label_names, networks_list, parlistfile] = nilearn_atlas_helper(atlas_select)

    ##Get coordinates and/or parcels from atlas
    if parlistfile is None and parc == False:
        print('Fetching coordinates and labels from nilearn coordinate-based atlases')
        ##Fetch nilearn atlas coords
        [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    else:
        ##Fetch user-specified atlas coords
        [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
        networks_list = None
        ##Describe user atlas coords
        print('\n' + str(atlas_select) + ' comes with {0} '.format(par_max) + 'parcels' + '\n')

    ##Labels prep
    try:
        label_names
    except:
        if ref_txt is not None and os.path.exists(ref_txt):
            atlas_select = os.path.basename(ref_txt).split('.txt')[0]
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            label_names = dict_df['Region'].tolist()
        else:
            try:
                atlas_ref_txt = atlas_select + '.txt'
                ref_txt = Path(__file__)/'atlases'/atlas_ref_txt
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            except:
                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    if label_names is None:
        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    try:
        atlas_name
    except:
        atlas_name = atlas_select
    return(label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile)

def RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc):
    ##Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    if atlas_select in nilearn_atlases:
        [label_names, networks_list, parlistfile] = nilearn_atlas_helper(atlas_select)

    ##Get coordinates and/or parcels from atlas
    if parlistfile is None and parc == False:
        print('Fetching coordinates and labels from nilearn coordinate-based atlases')
        ##Fetch nilearn atlas coords
        [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    else:
        ##Fetch user-specified atlas coords
        [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
        networks_list = None

    ##Labels prep
    try:
        label_names
    except:
        if ref_txt is not None and os.path.exists(ref_txt):
            atlas_select = os.path.basename(ref_txt).split('.txt')[0]
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            label_names = dict_df['Region'].tolist()
        else:
            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    if label_names is None:
        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    try:
        atlas_name
    except:
        atlas_name = atlas_select
    return(label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile)

def node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parc):
    ##Mask Parcels
    if parc == True:
        [coords, label_names, parcel_list_masked] = nodemaker.parcel_masker(mask, coords, parcel_list, label_names, dir_path, ID)
        [net_parcels_map_nifti, parcel_list_adj] = nodemaker.create_parcel_atlas(parcel_list_masked)     
        net_parcels_nii_path = dir_path + '/' + ID + '_parcels_masked_' + str(os.path.basename(mask).split('.')[0]) + '.nii.gz'
        nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    ##Mask Coordinates
    elif parc == False:
        [coords, label_names] = nodemaker.coord_masker(mask, coords, label_names)
        ##Save coords to pickle
        coord_path = dir_path + '/WB_func_coords_' + str(os.path.basename(mask).split('.')[0]) + '.pkl'
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f)
        net_parcels_map_nifti = None
    ##Save labels to pickle
    labels_path = dir_path + '/WB_func_labelnames_' + str(os.path.basename(mask).split('.')[0]) + '.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f)
    return(net_parcels_map_nifti, coords, label_names)

def node_gen(coords, parcel_list, label_names, dir_path, ID, parc):     
    if parc == True:
        [net_parcels_map_nifti, parcel_list_adj] = nodemaker.create_parcel_atlas(parcel_list)
        net_parcels_nii_path = dir_path + '/' + ID + '_wb_parcels.nii.gz'
        nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    else:
        net_parcels_map_nifti = None
        print('No additional masking...')
    ##Save coords to pickle
    coord_path = dir_path + '/WB_func_coords_wb.pkl'
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f)
    ##Save labels to pickle
    labels_path = dir_path + '/WB_func_labelnames_wb.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f)
    return(net_parcels_map_nifti, coords, label_names)

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

def do_dir_path(atlas_select, in_file):
    dir_path = os.path.dirname(os.path.realpath(in_file)) + '/' + atlas_select
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def thresh_and_fit(adapt_thresh, dens_thresh, thr, ts_within_nodes, conn_model, network, ID, dir_path, mask):
    if adapt_thresh is not False:
        try:
            est_path2 = dir_path + '/' + ID + '_structural_est.txt'
            if os.path.isfile(est_path2) == True:
                [conn_matrix, est_path, edge_threshold, thr] = thresholding.adaptive_thresholding(ts_within_nodes, conn_model, network, ID, est_path2, dir_path)
            else:
                print('No structural mx found! Exiting...')
                sys.exit()
        except:
            print('No structural mx assigned! Exiting...')
            sys.exit()
    elif dens_thresh is None:
        edge_threshold = str(float(thr)*100) +'%'
        [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_nodes, conn_model, network, ID, dir_path, mask, thr)
        conn_matrix = thresholding.threshold_proportional(conn_matrix, float(thr), dir_path)
    elif dens_thresh is not None:
        [conn_matrix, est_path, edge_threshold, thr] = thresholding.density_thresholding(ts_within_nodes, conn_model, network, ID, dens_thresh, dir_path)
    ##Normalize connectivity matrix (weights between 0-1)
    conn_matrix = thresholding.normalize(conn_matrix)
    return(conn_matrix, edge_threshold, est_path)

def plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold):
    ##Plot connectogram
    if len(conn_matrix) > 20:
        try:
            plotting.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)
        except RuntimeError:
            print('\n\n\nError: Connectogram plotting failed!')
    else:
        print('Error: Cannot plot connectogram for graphs smaller than 20 x 20!')

    ##Plot adj. matrix based on determined inputs
    atlas_graph_title = plotting.plot_conn_mat(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask)

    ##Plot connectome
    if mask != None:
        out_path_fig=dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '_connectome_viz.png'
    else:
        out_path_fig=dir_path + '/' + ID + '_connectome_viz.png'
    niplot.plot_connectome(conn_matrix, coords, title=atlas_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)
    return