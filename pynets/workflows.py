# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import warnings
warnings.simplefilter("ignore")
from pynets import nodemaker
from pynets.diffconnectometry import prepare_masks, run_struct_mapping
from pynets import utils

def wb_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc, ref_txt, procmem, dir_path):

    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = utils.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parc)
    else:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen(coords, parcel_list, label_names, dir_path, ID, parc)

    ##Extract time-series from nodes
    if parc == True:
        ##extract time series from whole brain parcellaions:
        ts_within_nodes = utils.extract_ts_wb_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network)
    else:
        ##Extract within-spheres time-series from funct file
        ts_within_nodes = utils.extract_ts_wb_coords(node_size, conf, func_file, coords, dir_path, ID, mask, thr, network)

    ##Threshold and fit connectivity model
    [conn_matrix, edge_threshold, est_path] = utils.thresh_and_fit(adapt_thresh, dens_thresh, thr, ts_within_nodes, conn_model, network, ID, dir_path, mask)

    ##Plotting
    if plot_switch == True:
        utils.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, edge_threshold)
    return est_path, thr

def RSN_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc, ref_txt, procmem, dir_path):
           
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    ##Get coord membership dictionary
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)

    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen_masking(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID, parc)            
    else:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen(net_coords, net_parcel_list, net_label_names, dir_path, ID, parc)

    ##Extract time-series from nodes
    if parc == True:
        ##extract time series from whole brain parcellaions:
        net_ts = utils.extract_ts_wb_parc(net_parcels_map_nifti, conf, func_file, net_coords, mask, dir_path, ID, network)
    else:
        ##Extract within-spheres time-series from funct file
        net_ts = utils.extract_ts_wb_coords(node_size, conf, func_file, net_coords, dir_path, ID, mask, thr, network)

    ##Threshold and fit connectivity model
    [conn_matrix, edge_threshold, est_path] = utils.thresh_and_fit(adapt_thresh, dens_thresh, thr, net_ts, conn_model, network, ID, dir_path, mask)   

    ##Plotting
    if plot_switch == True:
        utils.plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, net_label_names, mask, net_coords, edge_threshold)
    return est_path, thr

def wb_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path):
    
    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = utils.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)
        
    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parcels)
    else:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen(coords, parcel_list, label_names, dir_path, ID, parcels)

    ##Prepare Volumes
    if parcels == True:
        print('\n' + 'Converting 3d atlas image file to 4d image of atlas volume masks...' + '\n')
        volumes_dir = utils.convert_atlas_to_volumes(parlistfile, parcel_list)
        coords=None
    else:
        volumes_dir=None

    ##Prepare seed, avoidance, and waypoint masks
    print('\n' + 'Running node preparation...' + '\n')
    [vent_CSF_diff_mask_path, WM_diff_mask_path] = prepare_masks(ID, bedpostx_dir, network, coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads)

    ##Run all stages of probabilistic structural connectometry
    print('\n' + 'Running probabilistic structural connectometry...' + '\n')
    est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads, vent_CSF_diff_mask_path, WM_diff_mask_path)

    return est_path2
            
def RSN_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path):

    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)    
     
    ##Get coord membership dictionary
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, nodif_brain_mask_path, coords, label_names, parcels, parcel_list)

    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen_masking(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)            
    else:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen(net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)
            
    ##Prepare Volumes
    if parcels == True:
        print('\n' + 'Converting 3d atlas image file to 4d image of atlas volume masks...' + '\n')
        volumes_dir = utils.convert_atlas_to_volumes(parlistfile, net_parcel_list)
        net_coords=None
    else:
        volumes_dir=None

    ##Prepare seed, avoidance, and waypoint masks
    print('\n' + 'Running node preparation...' + '\n')
    [vent_CSF_diff_mask_path, WM_diff_mask_path] = prepare_masks(ID, bedpostx_dir, network, net_coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads)

    ##Run all stages of probabilistic structural connectometry
    print('\n' + 'Running probabilistic structural connectometry...' + '\n')
    est_path2 = run_struct_mapping(ID, bedpostx_dir, network, net_coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads, vent_CSF_diff_mask_path, WM_diff_mask_path)

    return est_path2