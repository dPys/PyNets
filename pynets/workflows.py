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
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from numpy import genfromtxt
from pathlib import Path
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
from pynets import nodemaker, thresholding, plotting, graphestimation
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

def wb_connectome_with_us_atlas_coords(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc):
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']

    ##Input is nifti file
    func_file=input_file

    ##Test if atlas_select is a nilearn atlas
    if atlas_select in nilearn_atlases:
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
            print('PyNets is not ready for multi-scale atlases like BASC just yet!')
            sys.exit()

    ##Fetch user-specified atlas coords
    [coords, atlas_name, par_max] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    atlas_select=atlas_name

    try:
        label_names
    except:
        try:
            atlas_ref_txt = atlas_name + '.txt'
            ref_txt = Path(__file__)/'atlases'/atlas_ref_txt
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            indices = dict_df.Index.unique().tolist()
            label_names = dict_df['Region'].tolist()
        except:
            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    ##Get subject directory path
    dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_select
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    ##Describe user atlas coords
    print('\n' + atlas_name + ' comes with {0} '.format(par_max) + 'parcels' + '\n')
    print('\n'+ 'Stacked atlas coordinates in array of shape {0}.'.format(coords.shape) + '\n')

    ##Mask coordinates
    if mask is not None:
        [coords, label_names] = nodemaker.coord_masker(mask, coords, label_names)

    ##Save coords and label_names to pickles
    coord_path = dir_path + '/coords_wb_' + str(thr) + '.pkl'
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f)

    labels_path = dir_path + '/labelnames_wb_' + str(thr) + '.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f)

    ##extract time series from whole brain parcellaions:
    parcellation = nib.load(parlistfile)
    parcel_masker = input_data.NiftiLabelsMasker(labels_img=parcellation, background_label=0, memory='nilearn_cache', memory_level=10, standardize=True)
    ts_within_parcels = parcel_masker.fit_transform(func_file, confounds=conf)
    print('\n' + 'Time series has {0} samples'.format(ts_within_parcels.shape[0]) + '\n')

    if bedpostx_dir is not None and anat_loc is not None:
        from pynets.diffconnectometry import run_struct_mapping
        network= 'whole_brain'
        if anat_loc is not None:
            parcels = True
        else:
            parcels = False
            dict_df = None
            indices = None
        try:
            est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords, node_size, atlas_select, atlas_name, label_names, plot_switch, parcels, dict_df, indices, anat_loc)
        except RuntimeError:
            print('Whole-brain Structural Graph Estimation Failed!')

    ##Save time series as txt file
    out_path_ts=dir_path + '/' + ID + '_whole_brain_ts_within_parcels.txt'
    np.savetxt(out_path_ts, ts_within_parcels)

    ##Fit connectivity model
    if adapt_thresh is not False:
        try:
            est_path2
            if os.path.isfile(est_path2) == True:
                [conn_matrix, est_path, edge_threshold, thr] = thresholding.adaptive_thresholding(ts_within_parcels, conn_model, network, ID, est_path2, dir_path)
            else:
                print('No structural mx found! Exiting...')
                sys.exit()
        except:
            print('No structural mx assigned! Exiting...')
            sys.exit()

    elif dens_thresh is None:
        edge_threshold = str(float(thr)*100) +'%'
        [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_parcels, conn_model, network, ID, dir_path, thr)
        conn_matrix = thresholding.threshold_proportional(conn_matrix, float(thr), dir_path)
        conn_matrix = thresholding.normalize(conn_matrix)
    elif dens_thresh is not None:
        [conn_matrix, est_path, edge_threshold, thr] = thresholding.density_thresholding(ts_within_parcels, conn_model, network, ID, dens_thresh, dir_path)

    if plot_switch == True:
        ##Plot connectogram
        plotting.plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names)

        ##Plot adj. matrix based on determined inputs
        atlas_graph_title = plotting.plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names, mask)

        ##Plot connectome viz for all Yeo networks
        out_path_fig=dir_path + '/' + ID + '_connectome_viz.png'
        niplot.plot_connectome(conn_matrix, coords, title=atlas_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)
    return est_path, thr

def wb_connectome_with_nl_atlas_coords(input_file, ID, atlas_select, network, node_size, mask, thr, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc):
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']

    ##Input is nifti file
    func_file=input_file

    ##Fetch nilearn atlas coords
    [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)

    ##Get subject directory path
    dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_select
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if label_names is None:
        try:
            atlas_ref_txt = atlas_name + '.txt'
            ref_txt = Path(__file__)/'atlases'/atlas_ref_txt
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            indices = dict_df.Index.unique().tolist()
            label_names = dict_df['Region'].tolist()
        except:
            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    ##Mask coordinates
    if mask is not None:
        [coords, label_names] = nodemaker.coord_masker(mask, coords, label_names)

    ##Save coords and label_names to pickles
    coord_path = dir_path + '/coords_wb_' + str(thr) + '.pkl'
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f)

    labels_path = dir_path + '/labelnames_wb_' + str(thr) + '.pkl'
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f)

    ##Extract within-spheres time-series from funct file
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), memory='nilearn_cache', memory_level=10, verbose=2, standardize=True)
    ts_within_spheres = spheres_masker.fit_transform(func_file, confounds=conf)
    print('\n' + 'Time series has {0} samples'.format(ts_within_spheres.shape[0]) + '\n')

    if bedpostx_dir is not None and anat_loc is not None:
        from pynets.diffconnectometry import run_struct_mapping
        network= 'whole_brain'
        if anat_loc is not None:
            parcels = True
        else:
            parcels = False
            dict_df = None
            indices = None

        try:
            est_path2 = est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords, node_size, atlas_select, atlas_name, label_names, plot_switch, parcels, dict_df, indices, anat_loc)
        except RuntimeError:
            print('Whole-brain Structural Graph Estimation Failed!')

    ##Save time series as txt file
    out_path_ts=dir_path + '/' + ID + '_whole_brain_ts_within_spheres.txt'
    np.savetxt(out_path_ts, ts_within_spheres)

    ##Fit connectivity model
    if adapt_thresh is not False:
        try:
            est_path2
            if os.path.isfile(est_path2) == True:
                [conn_matrix, est_path, edge_threshold, thr] = thresholding.adaptive_thresholding(ts_within_spheres, conn_model, network, ID, est_path2, dir_path)
            else:
                print('No structural mx found! Exiting...')
                sys.exit()
        except:
            print('No structural mx assigned! Exiting...')
            sys.exit()

    elif dens_thresh is None:
        edge_threshold = str(float(thr)*100) +'%'
        [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_spheres, conn_model, network, ID, dir_path, thr)
        conn_matrix = thresholding.threshold_proportional(conn_matrix, float(thr), dir_path)
        conn_matrix = thresholding.normalize(conn_matrix)
    elif dens_thresh is not None:
        [conn_matrix, est_path, edge_threshold, thr] = thresholding.density_thresholding(ts_within_spheres, conn_model, network, ID, dens_thresh, dir_path)

    if plot_switch == True:
        ##Plot connectogram
        plotting.plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names)

        ##Plot adj. matrix based on determined inputs
        plotting.plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names, mask)

        ##Plot connectome viz for all Yeo networks
        out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_connectome_viz.png'
        niplot.plot_connectome(conn_matrix, coords, title=atlas_name, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)
    return est_path, thr

def network_connectome(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc):
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']

    ##Input is nifti file
    func_file=input_file

    ##Test if atlas_select is a nilearn atlas
    if atlas_select in nilearn_atlases:
        atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
        try:
            parlistfile=atlas.maps
            try:
                label_names=atlas.labels
            except:
                label_names=None
            try:
                networks_list = atlas.networks
            except:
                networks_list = None
        except RuntimeError:
            print('Error, atlas fetching failed.')
            sys.exit()

    if parlistfile == None and atlas_select not in nilearn_atlases:
        ##Fetch nilearn atlas coords
        [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)

        try:
            networks_list
        except:
            networks_list = None

        net_coords = nodemaker.get_membership_from_coords(network, func_file, coords, networks_list)

        try:
            label_names=label_names.tolist()
        except:
            try:
                atlas_ref_txt = atlas_name + '.txt'
                ref_txt = Path(__file__)/'atlases'/atlas_ref_txt
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                indices = dict_df.Index.unique().tolist()
                label_names = dict_df['Region'].tolist()
            except:
                label_names=list(np.arange(len(net_coords)) + 1)

        ##Get subject directory path
        dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_select
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ##If masking, remove those coords that fall outside of the mask
        if mask != None:
            [net_coords, label_names] = nodemaker.coord_masker(mask, net_coords, label_names)

        ##Save coords and label_names to pickles
        coord_path = dir_path + '/coords_' + network + '_' + str(thr) + '.pkl'
        with open(coord_path, 'wb') as f:
            pickle.dump(net_coords, f)

        labels_path = dir_path + '/labelnames_' + network + '_' + str(thr) + '.pkl'
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f)

        if bedpostx_dir is not None and anat_loc is not None:
            from pynets.diffconnectometry import run_struct_mapping
            if anat_loc is not None:
                parcels = True
            else:
                parcels = False
                dict_df = None
                indices = None

            try:
                est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords, node_size, atlas_select, atlas_name, label_names, plot_switch, parcels, dict_df, indices, anat_loc)
            except RuntimeError:
                print('Whole-brain Structural Graph Estimation Failed!')

    else:
        ##Fetch user-specified atlas coords
        [coords_all, atlas_name, par_max] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
        coords = list(tuple(x) for x in coords_all)

        ##Get subject directory path
        dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ##Get coord membership dictionary
        try:
            networks_list
        except:
            networks_list = None

        net_coords = nodemaker.get_membership_from_coords(network, func_file, coords, networks_list)

        try:
            label_names
        except:
            label_names=list(np.arange(len(net_coords)) + 1)

        if mask != None:
            [net_coords, label_names] = nodemaker.coord_masker(mask, net_coords, label_names)

        ##Save coords and label_names to pickles
        coord_path = dir_path + '/coords_' + network + '_' + str(thr) + '.pkl'
        with open(coord_path, 'wb') as f:
            pickle.dump(net_coords, f)

        labels_path = dir_path + '/labelnames_' + network + '_' + str(thr) + '.pkl'
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f)

        if bedpostx_dir is not None and anat_loc is not None:
            from pynets.diffconnectometry import run_struct_mapping
            if anat_loc is not None:
                parcels = True
            else:
                parcels = False
                dict_df = None
                indices = None
            try:
                est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords_MNI, node_size, atlas_select, atlas_name, label_names, plot_switch, parcels, dict_df, indices, anat_loc)
            except RuntimeError:
                print('Network Structural Graph Estimation Failed!')

        ##Generate network parcels image
        #net_parcels_img_path = gen_network_parcels(parlistfile, network, labels)
        #parcellation = nib.load(net_parcels_img_path)
        #parcel_masker = input_data.NiftiLabelsMasker(labels_img=parcellation, background_label=0, memory='nilearn_cache', memory_level=5, standardize=True)
        #ts_within_parcels = parcel_masker.fit_transform(func_file)
        #net_ts = ts_within_parcels

    ##Grow ROIs
    masker = input_data.NiftiSpheresMasker(seeds=net_coords, radius=float(node_size), allow_overlap=True, memory_level=10, memory='nilearn_cache', verbose=2, standardize=True)
    ts_within_spheres = masker.fit_transform(func_file, confounds=conf)
    net_ts = ts_within_spheres

    ##Save time series as txt file
    out_path_ts=dir_path + '/' + ID + '_' + network + '_net_ts.txt'
    np.savetxt(out_path_ts, net_ts)

    ##Fit connectivity model
    if adapt_thresh is not False:
        try:
            est_path2
            if os.path.isfile(est_path2) == True:
                [conn_matrix, est_path, edge_threshold, thr] = thresholding.adaptive_thresholding(ts_within_spheres, conn_model, network, ID, est_path2, dir_path)
            else:
                print('No structural mx found! Exiting...')
                sys.exit()
        except:
            print('No structural mx assigned! Exiting...')
            sys.exit()
    elif dens_thresh is None:
        edge_threshold = str(float(thr)*100) +'%'
        [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_spheres, conn_model, network, ID, dir_path, thr)
        conn_matrix = thresholding.threshold_proportional(conn_matrix, float(thr), dir_path)
        conn_matrix = thresholding.normalize(conn_matrix)
    elif dens_thresh is not None:
        [conn_matrix, est_path, edge_threshold, thr] = thresholding.density_thresholding(ts_within_spheres, conn_model, network, ID, dens_thresh, dir_path)

    if plot_switch == True:
        ##Plot connectogram
        plotting.plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names)

        ##Plot adj. matrix based on determined inputs
        plotting.plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names, mask)

        ##Plot network time-series
        plotting.plot_timeseries(net_ts, network, ID, dir_path, atlas_name, label_names)

        ##Plot connectome viz for specific Yeo networks
        title = "Connectivity Projected on the " + network
        out_path_fig=dir_path + '/' + ID + '_' + network + '_connectome_plot.png'
        niplot.plot_connectome(conn_matrix, net_coords, edge_threshold=edge_threshold, title=title, display_mode='lyrz', output_file=out_path_fig)
    return est_path, thr
