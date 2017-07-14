#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#    PyNets: A Python-Powered Workflow for Network Analysis of Resting-State fMRI (rsfMRI) and Diffusion MRI (dMRI)
#    Copyright (C) 2017
#    ORIGINAL AUTHOR: Derek A. Pisner (University of Texas at Austin)
#    DEVELOPERS: Andrew Reineberg, Aki nikolaidis, Charles Laidi
#
#    PyNets is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyNets is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the complete GNU Affero General Public
#    License with PyNets in a file called license.txt. If not, and/or you simply have
#    questions about licensing and copyright/patent restrictions with PyNets, please
#    contact the primary author, Derek Pisner, at dpisner@utexas.edu
import sys
import argparse
import os
##Import sklearn here due to DeprecationWarning
from sklearn.model_selection import train_test_split

if len(sys.argv) < 1:
    print("\nMissing command-line inputs! See help options with the -h flag")
    sys.exit()

####Parse arguments####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
        metavar='Path to input file',
        default=None,
        required=True,
        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR the path to an 4D time-series text/csv file OR the path of a pre-made graph that has been thresholded and standardized appropriately')
    parser.add_argument('-ID',
        metavar='Subject ID',
        default=None,
        required=True,
        help='A subject ID that is also the name of the directory containing the input file')
    parser.add_argument('-a',
        metavar='Atlas',
        default='coords_power_2011',
        help='Specify a single coordinate atlas parcellation of those availabe in nilearn. Default is Power-264 node atlas. Available atlases are:\n\nabide_pcp\nadhd\natlas_aal\natlas_basc_multiscale_2015\natlas_craddock_2012\natlas_destrieux_2009\natlas_harvard_oxford\natlas_msdl\natlas_smith_2009\natlas_yeo_2011\ncobre\ncoords_dosenbach_2010\ncoords_power_2011\nhaxby\nhaxby_simple\nicbm152_2009\nicbm152_brain_gm_mask\nmegatrawls_netmats\nmixed_gambles\nmiyawaki2008\nnyu_rest\noasis_vbm')
# parser.add_argument('-ma', '--multiatlas',
#     default='All')
    parser.add_argument('-ua',
        metavar='Path to parcellation file',
        default=None,
        help='Path to nifti-formatted parcellation image file')
    parser.add_argument('-n',
        metavar='RSN',
        default=None,
        help='Optionally specify an atlas-defined network name from the following list of RSNs:\n\nDMN\nFPTC\nDA\nSN\nVA')
    parser.add_argument('-thr',
        metavar='Graph threshold',
        default='0.95',
        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is 0.95')
    parser.add_argument('-ns',
        metavar='Node size',
        default='3',
        help='Optionally specify a coordinate-based node radius size. Default is 3 voxels')
    parser.add_argument('-m',
        metavar='Path to mask image',
        default=None,
        help='Optionally specify a thresholded inverse-binarized mask image such as a group ICA-derived network volume, to retain only those network nodes contained within that mask')
    parser.add_argument('-an',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting designations and network statistic extraction for all Yeo RSNs in the specified atlas')
    parser.add_argument('-model',
        metavar='Connectivity',
        default='corr',
        help='Optionally specify matrix estimation type: corr, cov, or sps for correlation, covariance, or sparse-inverse covariance, respectively')
    parser.add_argument('-at',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate adaptive thresholding')
    args = parser.parse_args()

###Set Arguments to global variables###
input_file=args.i
ID=args.ID
atlas_select=args.a
NETWORK=args.n
thr=args.thr
node_size=args.ns
mask=args.m
conn_model=args.model
all_nets=args.an
parlistfile=args.ua
adapt_thresh=args.at
#######################################

##Check required inputs for existence
if input_file is None:
    print("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format or a path to a time-series text/csv file, with the -i flag")
    sys.exit()
elif not os.path.isfile(input_file):
    print("Error: Input file does not exist.")
    sys.exit()
if ID is None:
    print("Error: You must include a subject ID in your command line call")
    sys.exit()
if adapt_thresh == True and conn_model == 'sps':
    print("Adaptive thresholding not available for sparse inverse covariance model")
    sys.exit()

##Print inputs verbosely
print("\n\n\n" + "------------------------------------------------------------------------")
print ("INPUT FILE: " + input_file)
print("\n")
print ("SUBJECT ID: " + str(ID))
print("\n")
if parlistfile != None:
    atlas_name = parlistfile.split('/')[-1].split('.')[0]
    print ("ATLAS: " + str(atlas_name))
else:
    print ("ATLAS: " + str(atlas_select))
print("\n")
if NETWORK != None:
    print ("NETWORK: " + str(NETWORK))
elif NETWORK == None:
    print("USING WHOLE-BRAIN CONNECTOME..." )
print("-------------------------------------------------------------------------" + "\n\n\n")

##Set directory path containing input file
dir_path = os.path.dirname(os.path.realpath(input_file))

##Set pynets directory
pynets_dir = os.path.dirname(os.path.abspath(__file__))

##Import core modules
import warnings
warnings.filterwarnings("ignore")
import gzip
import nilearn
import cPickle
import numpy as np
import networkx as nx
import pandas as pd
import nibabel as nib
import seaborn as sns
import numpy.linalg as npl
from numpy import genfromtxt
from matplotlib import colors
from nipype import Node, Workflow
from nilearn import input_data, plotting, masking, datasets
from matplotlib import pyplot as plt
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nibabel.affines import apply_affine
from nipype.interfaces.base import isdefined, Undefined
from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

##Set input list for all workflow nodes
import_list=["import sys", "import os", "from sklearn.model_selection import train_test_split", "import warnings", "import gzip", "import nilearn", "import cPickle", "import numpy as np", "import networkx as nx", "import pandas as pd", "import nibabel as nib", "import seaborn as sns", "import numpy.linalg as npl", "from numpy import genfromtxt", "from matplotlib import colors", "from nipype import Node, Workflow", "from nilearn import input_data, plotting, masking, datasets", "from matplotlib import pyplot as plt", "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu", "from nipype.interfaces import io as nio", "from nilearn.input_data import NiftiLabelsMasker", "from nilearn.connectome import ConnectivityMeasure", "from nibabel.affines import apply_affine", "from nipype.interfaces.base import isdefined, Undefined", "from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso", "from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits"]

##Import time-series/graph, fit matrix model, plot matrix, plot connectome
def mat_funcs(input_file, ID, atlas_select, NETWORK, pynets_dir, node_size, mask, thr, parlistfile, all_nets, conn_model, adapt_thresh):
    def fetch_nilearn_atlas_coords(atlas_select):
        atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
        atlas_name = atlas['description'].splitlines()[0]
        print('\n' + atlas_name + ' comes with {0}'.format(atlas.keys()) + '\n')
        coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
        print('Stacked atlas coordinates in array of shape {0}.'.format(coords.shape) + '\n')
        return(coords, atlas_name)

    def get_ref_net(bna_img, par_data, x, y, z):
        ##Network membership dictionary-->enhance/expand based on Yeo and others.
        ##A variety of membership determination schemes could be implemented...
        ref_dict = {0:'UNKNOWN', 1:'VIS', 2:'SENS', 3:'DA', 4:'VA', 5:'LIMBIC', 6:'FPTC', 7:'DMN'}
        ##apply_affine(aff, (x,y,z)) # vox to mni
        aff_inv=npl.inv(bna_img.affine)
        ##mni to vox
        vox_coord = apply_affine(aff_inv, (x, y, z))
        return ref_dict[int(par_data[int(vox_coord[0]),int(vox_coord[1]),int(vox_coord[2])])]

    def get_mem_dict(pynets_dir, func_file, coords):
        bna_img = nib.load(func_file)
        par_path = pynets_dir + '/RSN_refs/Yeo7.nii.gz'
        par_img = nib.load(par_path)
        par_data = par_img.get_data()
        membership = pd.Series(list(tuple(x) for x in coords), [get_ref_net(bna_img, par_data, coord[0],coord[1],coord[2]) for coord in coords])
        membership_plotting = pd.Series([get_ref_net(bna_img, par_data, coord[0],coord[1],coord[2]) for coord in coords])
        return(membership, membership_plotting)

    def coord_masker(mask, coords):
        mask_data, _ = masking._load_mask_img(mask)
        mask_coords = list(zip(*np.where(mask_data != 0)))
        for coord in coords:
            if coord in mask_coords:
                print('Removing coordinate: ' + str(coord) + ' since it falls outside of mask...')
                coords.remove(coord)
        return coords

    def coord_masker_with_tuples(mask, coords):
        mask_data, _ = masking._load_mask_img(mask)
        mask_coords = list(zip(*np.where(mask_data != 0)))
        for coord in coords:
            if tuple(coord) not in mask_coords:
                print('Removing coordinate: ' + str(tuple(coord)) + ' since it falls outside of mask...')
                ix = np.where(coords == coord)[0][0]
                coords = np.delete(coords, ix, axis=0)
                print(str(len(coords)))
        return coords

    def get_conn_matrix(time_series, conn_model, NETWORK, ID):
        if conn_model == 'corr':
            conn_measure = ConnectivityMeasure(kind='correlation')
            conn_matrix = conn_measure.fit_transform([time_series])[0]
            est_path = dir_path + '/' + ID + '_est_corr.txt'
        elif conn_model == 'cov' or conn_model == 'sps':
                ##Fit estimator to matrix to get sparse matrix
                estimator = GraphLassoCV()

                try:
                    print("Fitting Lasso estimator...")
                    est = estimator.fit(time_series)
                except:
                    print("Error: Lasso sparse matrix modeling failed. Using ledoit-wolf shrinkage...")
                    from sklearn.covariance import LedoitWolf
                    estimator = LedoitWolf()
                    est = estimator.fit(time_series)
                if NETWORK != None:
                    est_path = dir_path + '/' + ID + '_' + NETWORK + '_est%s.txt'%('_sps_inv' if conn_model=='sps' else 'cov')
                else:
                    est_path = dir_path + '/' + ID + '_est%s.txt'%('_sps_inv' if conn_model=='sps' else 'cov')
                if conn_model == 'sps':
                    conn_matrix = -estimator.precision_
                elif conn_model == 'cov':
                    conn_matrix = estimator.covariance_
        np.savetxt(est_path, conn_matrix, delimiter='\t')
        return(conn_matrix, est_path)

    def get_names_and_coords_of_parcels(parlistfile):
        atlas_name = parlistfile.split('/')[-1].split('.')[0]
        ##Code for getting name and coordinates of parcels. Adapted from Dan L. (https://github.com/danlurie/despolab_lesion/blob/master/code/sandbox/Sandbox%20-%20Calculate%20and%20plot%20HCP%20mean%20matrix.ipynb)
###Reindex parcel. schemes with non-contiguous parcels (Andy?)
        bna_img = nib.load(parlistfile)
        bna_data = bna_img.get_data()
        if bna_img.get_data_dtype() != np.dtype(np.int):
            bna_data_for_coords = bna_img.get_data()
            ##Number of parcels:
            par_max = np.ceil(np.max(bna_data_for_coords)).astype('int')
            bna_data = bna_data.astype('int16')
        else:
            par_max = np.max(bna_data)

        img_stack = []
        for idx in range(1, par_max+1):
            roi_img = bna_data == idx
            img_stack.append(roi_img)
        img_stack = np.array(img_stack)
        img_list = []
        for idx in range(par_max):
            roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
            img_list.append(roi_img)

        coords = []
        for roi_img in img_list:
            coords.append(nilearn.plotting.find_xyz_cut_coords(roi_img))
        coords = np.array(coords)
        return(coords, atlas_name, par_max)

    def gen_network_parcels(parlistfile, NETWORK, labels):
        bna_img = nib.load(parlistfile)
        bna_data = bna_img.get_data()
        if bna_img.get_data_dtype() != np.dtype(np.int):
            bna_data_for_coords = bna_img.get_data()
            # Number of parcels:
            par_max = np.ceil(np.max(bna_data_for_coords)).astype('int')
            bna_data = bna_data.astype('int16')
        else:
            par_max = np.max(bna_data)

        img_stack = []
        for idx in range(1, par_max+1):
            roi_img = bna_data == idx
            img_stack.append(roi_img)
        img_stack = np.array(img_stack)
        img_list = []
        for idx in range(par_max):
            roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
            img_list.append(roi_img)

        print('Extracting parcels associated with ' + NETWORK + ' locations...')
        net_parcels = [i for j, i in enumerate(img_list) if j in labels]
        bna_4D = nilearn.image.concat_imgs(net_parcels).get_data()
        index_vec = np.array(range(len(net_parcels))) + 1
        net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
        net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
        out_path = dir_path + '/' + NETWORK + '_parcels.nii.gz'
        nib.save(net_parcels_map_nifti, out_path)
        return(out_path)

    def plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK):
        ##Set title for adj. matrix based on connectivity model used
        if conn_model == 'corr':
            atlast_graph_title = atlas_name + '_Correlation_Graph'
        elif conn_model == 'sps':
            atlast_graph_title = atlas_name + '_Sparse_Covariance_Graph'
        elif conn_model == 'cov':
            atlast_graph_title = atlas_name + '_Covariance_Graph'

        if mask != None:
            atlast_graph_title = atlast_graph_title + '_With_Masked_Nodes'
        if NETWORK != None:
            atlast_graph_title = atlast_graph_title + '_' + NETWORK

        rois_num=conn_matrix.shape[0]
        if NETWORK != None:
            print("Creating plot of dimensions:\n" + str(rois_num) + ' x ' + str(rois_num))
            plt.figure(figsize=(rois_num, rois_num))
        else:
            plt.figure(figsize=(10, 10))
        plt.imshow(conn_matrix, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
        ##And display the labels
        if rois_num < 50:
            x_ticks = plt.xticks(range(rois_num), rotation=90)
            y_ticks = plt.yticks(range(rois_num))
        plt.colorbar()
        plt.title(atlas_name.upper() + ' ' + conn_model.upper() + ' MATRIX')
        out_path_fig=dir_path + '/' + ID + '_adj_mat_' + conn_model + '.png'
        plt.savefig(out_path_fig)
        plt.close()
        return(atlast_graph_title)

    def plot_membership(membership_plotting, conn_matrix, conn_model, coords, edge_threshold, atlast_name, dir_path):
        atlast_connectome_title = atlas_name + '_all_networks'
        n = len(membership_plotting.unique())
        clust_pal = sns.color_palette("Set2", n)
        clust_lut = dict(zip(map(str, np.unique(membership_plotting.astype('category'))), clust_pal))
        clust_colors = colors.to_rgba_array(membership_plotting.map(clust_lut))
        out_path_fig = dir_path + '/' + ID + '_connectome_viz.png'
        plotting.plot_connectome(conn_matrix, coords, node_color = clust_colors, title=atlast_connectome_title, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)

    def plot_timeseries(time_series, NETWORK, ID, dir_path, atlas_name, labels):
        for time_serie, label in zip(time_series.T, labels):
            plt.plot(time_serie, label=label)
        plt.title(NETWORK + ' Network Time Series')
        plt.xlabel('Scan Number')
        plt.ylabel('Normalized Signal')
        plt.legend()
        #plt.tight_layout()
        if NETWORK != None:
            out_path_fig=dir_path + '/' + ID + '_' + NETWORK + '_TS_plot.png'
        else:
            out_path_fig=dir_path + '/' + ID + '_Whole_Brain_TS_plot.png'
        plt.savefig(out_path_fig)
        plt.close()

    def adaptive_thresholding(ts_within_spheres, conn_model, NETWORK, ID, thr):
        thr=0.95
        [conn_matrix, est_path] = get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID)
        i = 1
        zeroes = np.count_nonzero(conn_matrix==0)
        perc_missing_edges = float(zeroes) / float(conn_matrix.size)
        print(str((zeroes)) + " missing edges detected..." + "\n")
        while perc_missing_edges > float(0.25):
            zeroes = np.count_nonzero(conn_matrix==0)
            perc_missing_edges = float(zeroes) / float(conn_matrix.size)
            if perc_missing_edges > float(0.50):
                thr = thr + float(0.05)
            elif perc_missing_edges > float(0.25):
                thr = thr + float(0.01)
            else:
                thr = thr + float(0.001)
            edge_threshold = str(float(thr)*100) +'%'
            print('Adaptively thresholding -- Iteration ' + str(i) + ' -- with thresh: ' + str(thr) + '...')
            print(str((zeroes)) + " missing edges detected..." + "\n")
            [conn_matrix, est_path] = get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID)
            i = i + 1
            if perc_missing_edges < float(0.25) or thr >= float(0.99):
                break
        return(conn_matrix, est_path, edge_threshold, thr)

    ##Case 1: Whole-brain connectome with nilearn atlas
    if '.nii' in input_file and parlistfile == None and NETWORK == None:
        ##Input is nifti file
        func_file=input_file

        ##Fetch nilearn atlas coords
        [coords, atlas_name] = fetch_nilearn_atlas_coords(atlas_select)

        ##Get subject directory path
        dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ##Get coord membership dictionary if all_nets option triggered
        if all_nets != False:
            [membership, membership_plotting] = get_mem_dict(pynets_dir, func_file, coords)

        ##Mask coordinates
        if mask is not None:
            coords = coord_masker_with_tuples(mask, coords)

        ##Extract within-spheres time-series from funct file
        spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), memory='nilearn_cache', memory_level=5, verbose=2, standardize=True)
        ts_within_spheres = spheres_masker.fit_transform(func_file)
        print('\n' + 'Time series has {0} samples'.format(ts_within_spheres.shape[0]) + '\n')

        ##Fit connectivity model
        if adapt_thresh == False:
            edge_threshold = str(float(thr)*100) +'%'
            [conn_matrix, est_path] = get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID)
        else:
            [conn_matrix, est_path, edge_threshold, thr] = adaptive_thresholding(ts_within_spheres, conn_model, NETWORK, ID, thr)

        ##Plot adj. matrix based on determined inputs
        plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK)

        ##Plot connectome viz for all Yeo networks
        ##Tweak edge_threshold to keep only the strongest connections based on thr
        if all_nets != False:
            plot_membership(membership_plotting, conn_matrix, conn_model, coords, edge_threshold, atlas_name, dir_path)
        else:
            out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_connectome_viz.png'
            plotting.plot_connectome(conn_matrix, coords, title=atlas_name, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)

    ##Case 2: Whole-brain connectome with user-specified atlas
    elif '.nii' in input_file and parlistfile != None and NETWORK == None: # block of code for whole brain parcellations
        ##Input is nifti file
        func_file=input_file

        ##Fetch user-specified atlas coords
        [coords, atlas_name, par_max] = get_names_and_coords_of_parcels(parlistfile)

        ##Get subject directory path
        dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ##Get coord membership dictionary if all_nets option triggered
        if all_nets != None:
            [membership, membership_plotting] = get_mem_dict(pynets_dir, func_file, coords)

        ##Describe user atlas coords
        print('\n' + atlas_name + ' comes with {0} '.format(par_max) + 'parcels' + '\n')
        print('\n'+ 'Stacked atlas coordinates in array of shape {0}.'.format(coords.shape) + '\n')

        ##Mask coordinates
        if mask is not None:
            coords = coord_masker(mask, coords)

        ##extract time series from whole brain parcellaions:
        parcellation = nib.load(parlistfile)
        parcel_masker = input_data.NiftiLabelsMasker(labels_img=parcellation, background_label=0, memory='nilearn_cache', memory_level=5, standardize=True)
        ts_within_parcels = parcel_masker.fit_transform(func_file)
        print('\n' + 'Time series has {0} samples'.format(ts_within_parcels.shape[0]) + '\n')

        ##Fit connectivity model
        if adapt_thresh == False:
            edge_threshold = str(float(thr)*100) +'%'
            [conn_matrix, est_path] = get_conn_matrix(ts_within_parcels, conn_model, NETWORK, ID)
        else:
            [conn_matrix, est_path, edge_threshold, thr] = adaptive_thresholding(ts_within_parcels, conn_model, NETWORK, ID, thr)

        ##Plot adj. matrix based on determined inputs
        atlast_graph_title = plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK)

        ##Plot connectome viz for all Yeo networks
        ##Tweak edge_threshold to keep only the strongest connections based on thr
        if all_nets != False:
            plot_membership(membership_plotting, conn_matrix, conn_model, coords, edge_threshold, atlas_name, dir_path)
        else:
            out_path_fig=dir_path + '/' + ID + '_connectome_viz.png'
            plotting.plot_connectome(conn_matrix, coords, title=atlast_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)

    ##Case 3: RSN connectome with nilearn atlas or user-specified atlas
    elif '.nii' in input_file and NETWORK != None:
        ##Input is nifti file
        func_file=input_file

        if parlistfile == None:
            ##Fetch nilearn atlas coords
            [coords_all, atlas_name] = fetch_nilearn_atlas_coords(atlas_select)

            if atlas_name == 'Power 2011 atlas':
                ##Reference RSN list
                load_path = pynets_dir + '/RSN_refs/' + NETWORK + '_coords.csv'
            	df = pd.read_csv(load_path).ix[:,0:4]
            	i=1
            	net_coords = []
            	labels = []
            	for i in range(len(df)):
              	    print("ROI Reference #: " + str(i))
              	    x = int(df.ix[i,1])
              	    y = int(df.ix[i,2])
              	    z = int(df.ix[i,3])
              	    print("X:" + str(x) + " Y:" + str(y) + " Z:" + str(z))
              	    net_coords.append((x, y, z))
              	    labels.append(i)
                    i = i + 1
             	print("-----------------------------------------------------\n")
                print(net_coords)
              	print(labels)
                print("\n-----------------------------------------------------")
            elif atlas_name != 'Power 2011 atlas':
                sys.exit()
                ####Add code for any special RSN reference lists for the nilearn atlases here#####

            ##Get subject directory path
            dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_name
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            ##If masking, remove those coords that fall outside of the mask
            if mask != None:
                net_coords = coord_masker(mask, net_coords)

            ##Grow ROIs
            masker = input_data.NiftiSpheresMasker(seeds=net_coords, radius=float(node_size), allow_overlap=True, memory_level=5, memory='nilearn_cache', verbose=2, standardize=True)
            ts_within_spheres = masker.fit_transform(func_file)
            net_ts = ts_within_spheres
        else:
            ##Fetch user-specified atlas coords
            [coords_all, atlas_name, par_max] = get_names_and_coords_of_parcels(parlistfile)
            coords = list(tuple(x) for x in coords_all)

            ##Get subject directory path
            dir_path = os.path.dirname(os.path.realpath(func_file)) + '/' + atlas_name
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            ##Get coord membership dictionary
            [membership, membership_plotting] = get_mem_dict(pynets_dir, func_file, coords)

            ##Convert to membership dataframe
            mem_df = membership.to_frame().reset_index()

            ##Get coords for network-of-interest
            mem_df.loc[mem_df['index'] == NETWORK]
            net_coords = mem_df.loc[mem_df['index'] == NETWORK][[0]].values[:,0]
            net_coords = list(tuple(x) for x in net_coords)
            labels = mem_df.loc[mem_df['index'] == NETWORK].index.values

            if mask != None:
                net_coords = coord_masker(mask, net_coords)

            ##Grow ROIs
            masker = input_data.NiftiSpheresMasker(seeds=net_coords, radius=float(node_size), allow_overlap=True, memory_level=5, memory='nilearn_cache', verbose=2, standardize=True)
            ts_within_spheres = masker.fit_transform(func_file)
            net_ts = ts_within_spheres

            ##Generate network parcels image (through refinement, this could be used
            ##in place of the 3 lines above)
            #net_parcels_img_path = gen_network_parcels(parlistfile, NETWORK, labels)
            #parcellation = nib.load(net_parcels_img_path)
            #parcel_masker = input_data.NiftiLabelsMasker(labels_img=parcellation, background_label=0, memory='nilearn_cache', memory_level=5, standardize=True)
            #ts_within_parcels = parcel_masker.fit_transform(func_file)
            #net_ts = ts_within_parcels

        ##Plot network time-series
        plot_timeseries(net_ts, NETWORK, ID, dir_path, atlas_name, labels)

        ##Fit connectivity model
        if adapt_thresh == False:
            edge_threshold = str(float(thr)*100) +'%'
            [conn_matrix, est_path] = get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID)
        else:
            [conn_matrix, est_path, edge_threshold, thr] = adaptive_thresholding(ts_within_spheres, conn_model, NETWORK, ID, thr)

        ##Plot adj. matrix based on determined inputs
        plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK)

        ##Plot connectome viz for specific Yeo networks
        ##Tweak edge_threshold to keep only the strongest connections based on thr
        title = "Connectivity Projected on the " + NETWORK
        out_path_fig=dir_path + '/' + ID + '_' + NETWORK + '_connectome_plot.png'
        plotting.plot_connectome(conn_matrix, net_coords, edge_threshold=edge_threshold, title=title, display_mode='lyrz', output_file=out_path_fig)
    return(est_path)

##Extract network metrics interface
def extractnetstats(ID, NETWORK, thr, conn_model, est_path1, out_file=None):
    def threshold_proportional(in_mat, thr):
        ##number of nodes
        n = len(in_mat)
        ##clear diagonal
        np.fill_diagonal(in_mat, 0)
        ##if symmetric matrix
        if np.allclose(in_mat, in_mat.T):
            ##ensure symmetry is preserved
            in_mat[np.tril_indices(n)] = 0
            ##halve number of removed links
            ud = 2
        else:
            ud = 1
        ##find all links
        ind = np.where(in_mat)
        ##sort indices by magnitude
        I = np.argsort(in_mat[ind])[::-1]
        ##number of links to be preserved
        en = int(round((n * n - n) * float(thr) / ud))
        ##apply threshold
        in_mat[(ind[0][I][en:], ind[1][I][en:])] = 0
        ##if symmetric matrix
        if ud == 2:
            ##reconstruct symmetry
            in_mat[:, :] = in_mat + in_mat.T
        return in_mat

    ##Load and threshold matrix
    in_mat_un_thr = np.array(genfromtxt(est_path1))
    in_mat = threshold_proportional(in_mat_un_thr, thr)

    ##Get hyperbolic tangent of matrix if non-sparse (i.e. fischer r-to-z transform), and divide by the variance of the matrix
    if conn_model != 'sps':
        in_mat = np.arctanh(in_mat)/np.var(in_mat)

    ##Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path1))

    ##Load numpy matrix as networkx graph
    G=nx.from_numpy_matrix(in_mat)

    ##Save gephi files
    if NETWORK != None:
        H = nx.write_graphml(G, dir_path + '/' + ID + '_' + NETWORK + '.graphml')
    else:
        H = nx.write_graphml(G, dir_path + '/' + ID + '.graphml')

    ###############################################################
    ########### Calculate graph metrics from graph G ##############
    ###############################################################
    from itertools import permutations
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality

    ##Define missing network functions here. Small-worldness, modularity, and rich-club will also need to be added.
    def efficiency(G, u, v):
        return float(1) / nx.shortest_path_length(G, u, v)

    def global_efficiency(G):
        n = len(G)
        denom = n * (n - 1)
        return float(sum(efficiency(G, u, v) for u, v in permutations(G, 2))) / denom

    def local_efficiency(G):
        return float(sum(global_efficiency(nx.ego_graph(G, v)) for v in G)) / len(G)

    def create_random_graph(G, n, p):
        rG = nx.erdos_renyi_graph(n, p, seed=42)
        return rG

    def smallworldness_measure(G, rG):
        C_g = nx.algorithms.average_clustering(G)
        C_r = nx.algorithms.average_clustering(rG)
        L_g = nx.average_shortest_path_length(G)
        L_r = nx.average_shortest_path_length(rG)
        gam = float(C_g) / float(C_r)
        lam = float(L_g) / float(L_r)
        swm = gam / lam
        return swm

    def smallworldness(G, rep = 1000):
        n = nx.number_of_nodes(G)
        m = nx.number_of_edges(G)

        p = float(m) * 2 /(n*(n-1))
        ss = []
        for bb in range(rep):
        	rG = create_random_graph(G, n, p)
        	swm = smallworldness_measure(G, rG)
        	ss.append(swm)
        mean_s = np.mean(ss)
        return mean_s

    ##For scalar metrics from networkx.algorithms library,
    ##add the name of the function here for it to be automatically calculated.
    ##Because I'm lazy, it will also need to be imported above.
    metric_list = [global_efficiency, local_efficiency, smallworldness, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]

    ##Iteratively run functions from above metric list
    num_mets = len(metric_list)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j=0
    for i in metric_list:
        net_met = '%s' % i.func_name
        try:
            net_met_val = float(i(G))
        except:
            net_met_val = np.nan
        net_met_arr[j,0] = net_met
        net_met_arr[j,1] = net_met_val
        print(net_met)
        print(str(net_met_val))
        print('\n')
        j = j + 1

    ##If an RSN, extract node metrics like centrality measures here.
    if NETWORK != None:
        bc_vector = betweenness_centrality(G)
        bc_vals = bc_vector.values()
        bc_nodes = bc_vector.keys()
        num_nodes = len(bc_nodes)
        bc_arr = np.zeros([num_nodes, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            bc_arr[j,0] = NETWORK + '_' + str(bc_nodes[j]) + '_bet_cent'
            bc_arr[j,1] = bc_vals[j]
            print(NETWORK + '_' + str(bc_nodes[j]))
            print(str(bc_vals[j]))
            j = j + 1
        net_met_val_list = list(net_met_arr[:,1]) + list(bc_arr[:,1])
    else:
        net_met_val_list = list(net_met_arr[:,1])

    ##Create a list of metric names for scalar metrics
    metric_list_names = []
    for i in metric_list:
        metric_list_names.append('%s' % i.func_name)

    ##Create a list of metric names for nodal-type metrics
    if NETWORK != None:
        for i in bc_arr[:,0]:
            metric_list_names.append(i)

    ##Save metric names as pickle
    met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/met_list_pickle'
    cPickle.dump(metric_list_names, open(met_list_picke_path, 'wb'))

    ##Save results to csv
    if 'inv' in est_path1:
        if NETWORK != None:
            out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_inv_sps_cov.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_mets_inv_sps_cov.csv'
    else:
        if NETWORK != None:
            out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_corr.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_mets_corr.csv'
    np.savetxt(out_path, net_met_val_list)

    return(out_path)
    return(metric_list_names)
    ###############################################################
    ###############################################################


class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    sub_id = traits.Str(mandatory=True)
    NETWORK = traits.Any(mandatory=True)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path1 = File(exists=True, mandatory=True, desc="")

class ExtractNetStatsOutputSpec(TraitedSpec):
    out_file = File()

class ExtractNetStats(BaseInterface):
    input_spec = ExtractNetStatsInputSpec
    output_spec = ExtractNetStatsOutputSpec

    def _run_interface(self, runtime):
        out = extractnetstats(
            self.inputs.sub_id,
            self.inputs.NETWORK,
	        self.inputs.thr,
            self.inputs.conn_model,
            self.inputs.est_path1)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(getattr(self, '_outpath'))}

##save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, NETWORK, out_file=None):
    met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/met_list_pickle'
    metric_list_names = cPickle.load(open(met_list_picke_path, 'rb'))
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
    df = df.T
    column_headers={k: v for k, v in enumerate(metric_list_names)}
    df = df.rename(columns=column_headers)
    df['id'] = range(1, len(df) + 1)
    if 'id' in df.columns:
        cols = df.columns.tolist()
        ix = cols.index('id')
        cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
        df = df[cols_ID]
    df['id'] = df['id'].astype('object')
    df['id'].values[0] = ID
    out_file = csv_loc.replace('.', '')[:-3] + '_' + ID
    df.to_pickle(out_file)
    return(out_file)

class Export2PandasInputSpec(BaseInterfaceInputSpec):
    in_csv = File(exists=True, mandatory=True, desc="")
    sub_id = traits.Str(mandatory=True)
    NETWORK = traits.Any(mandatory=True)
    out_file = File('output_export2pandas.csv', usedefault=True)

class Export2PandasOutputSpec(TraitedSpec):
    out_file = File()

class Export2Pandas(BaseInterface):
    input_spec = Export2PandasInputSpec
    output_spec = Export2PandasOutputSpec

    def _run_interface(self, runtime):
        export_to_pandas(
            self.inputs.in_csv,
            self.inputs.sub_id,
            self.inputs.NETWORK,
            out_file=self.inputs.out_file)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(self.inputs.out_file)}

if __name__ == '__main__':
    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'NETWORK', 'pynets_dir', 'thr', 'node_size', 'mask', 'parlistfile', 'all_nets', 'conn_model', 'adapt_thresh']), name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.in_file = input_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.NETWORK = NETWORK
    inputnode.inputs.pynets_dir = pynets_dir
    inputnode.inputs.thr = thr
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.parlistfile = parlistfile
    inputnode.inputs.all_nets = all_nets
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.adapt_thresh = adapt_thresh

    #3) Add variable to function nodes
    ##Create function nodes
    imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select', 'NETWORK', 'pynets_dir', 'node_size', 'mask', 'thr', 'parlistfile', 'all_nets', 'conn_model', 'adapt_thresh'], output_names = ['est_path'], function=mat_funcs, imports=import_list), name = "imp_est")
    net_mets_node = pe.Node(ExtractNetStats(), name = "ExtractNetStats")
    export_to_pandas_node = pe.Node(Export2Pandas(), name = "export_to_pandas")

    ##Create PyNets workflow
    wf = pe.Workflow(name='PyNets_WORKFLOW')
    wf.base_directory='/tmp/pynets'

    ##Create data sink
    #datasink = pe.Node(nio.DataSink(), name='sinker')
    #datasink.inputs.base_directory = dir_path + '/DataSink'

    ##Add variable to workflow
    ##Connect nodes of workflow
    wf.connect([
        (inputnode, imp_est, [('in_file', 'input_file'),
                              ('ID', 'ID'),
                              ('atlas_select', 'atlas_select'),
                              ('NETWORK', 'NETWORK'),
    			              ('pynets_dir', 'pynets_dir'),
    			              ('node_size', 'node_size'),
                              ('mask', 'mask'),
                              ('thr', 'thr'),
                              ('parlistfile', 'parlistfile'),
                              ('all_nets', 'all_nets'),
                              ('conn_model', 'conn_model'),
                              ('adapt_thresh', 'adapt_thresh')]),
        (inputnode, net_mets_node, [('ID', 'sub_id'),
                                   ('NETWORK', 'NETWORK'),
    				               ('thr', 'thr'),
                                   ('conn_model', 'conn_model')]),
        (imp_est, net_mets_node, [('est_path', 'est_path1')]),
        #(net_mets_cov_node, datasink, [('est_path', 'csv_loc')]),
        (inputnode, export_to_pandas_node, [('ID', 'sub_id'),
                                        ('NETWORK', 'NETWORK')]),
        (net_mets_node, export_to_pandas_node, [('out_file', 'in_csv')]),
        #(export_to_pandas1, datasink, [('out_file', 'pandas_df)]),
    ])

    #wf.run(plugin='SLURM')
    #wf.run(plugin='MultiProc')
    wf.run()
