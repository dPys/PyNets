#!/bin/env python -W ignore::DeprecationWarning
#    PyNets: A Python-Powered Workflow for Network Analysis of Resting-State fMRI (rsfMRI)
#    Copyright (C) 2017
#    ORIGINAL AUTHOR: Derek A. Pisner (University of Texas at Austin)
#    DEVELOPERS:
#
#    openDTI is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    openDTI is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the complete GNU Affero General Public
#    License with openDTI in a file called LICENSE.txt. If not, and/or you simply have
#    questions about licensing and copyright/patent restrictions with openDTI, please
#    contact the primary author, Derek Pisner, at dpisner@utexas.edu
import sys
import argparse
from nipype.interfaces.base import isdefined,Undefined
from sklearn.model_selection import train_test_split

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
    parser.add_argument('-g',
        default=False,
        action='store_true',
        help='Optionally use this flag if your input file is a pre-made graph matrix, as opposed to a time-series text file or functional image')
    parser.add_argument('-sps',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to model your graph using lasso sparse inverse covariance')
    args = parser.parse_args()

if len(sys.argv) > 1:
    input_file=args.i
    if input_file is None:
        print("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format or a path to a time-series text/csv file, with the -i flag")
        sys.exit()
    ID=args.ID
    if ID is None:
        print("Error: You must include a subject ID in your command line call")
        sys.exit()
else:
    print("\nMissing command-line inputs! See help options with the -h flag")
    sys.exit()
atlas_select=args.a
NETWORK=args.n
thr=args.thr
node_size=args.ns
mask=args.m
graph=args.g
sps_model=args.sps
all_nets=args.an
#######################

import warnings
warnings.filterwarnings("ignore")
import nilearn
import numpy as np
import os
from numpy import genfromtxt
from sklearn.covariance import GraphLassoCV
from matplotlib import pyplot as plt
from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import input_data
from nilearn import plotting
import networkx as nx
import gzip
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
import numpy.linalg as npl
import seaborn as sns
from matplotlib import colors

import_list=["import nilearn", "import numpy as np", "import os", "from numpy import genfromtxt", "from matplotlib import pyplot as plt", "from nipype import Node, Workflow", "from nipype import Node, Workflow", "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu", "from nipype.interfaces import io as nio", "from nilearn import plotting", "from nilearn import datasets", "from nilearn.input_data import NiftiLabelsMasker", "from nilearn.connectome import ConnectivityMeasure", "from nilearn import datasets", "import gzip", "from nilearn import input_data", "from nilearn import plotting", "import networkx as nx", "import nibabel as nib", "from nipype.interfaces.base import isdefined,Undefined", "import pandas as pd", "import nibabel as nib", "from nibabel.affines import apply_affine", "import numpy.linalg as npl", "import seaborn as sns", "from matplotlib import colors"]

print("\n\n\n")
if graph == True:
    print ("GRAPH INPUT FILE: " + input_file)
else:
    print ("INPUT FILE: " + input_file)
print("\n")
print ("SUBJECT ID: " + str(ID))
if '.nii' in input_file:
    print("\n")
    print ("ATLAS: " + str(atlas_select))
    print("\n")
    if NETWORK != None:
        print ("NETWORK: " + str(NETWORK))
    elif NETWORK == None:
        print("USING WHOLE-BRAIN CONNECTOME...")
print("\n\n\n")

dir_path = os.path.dirname(os.path.realpath(input_file))

pynets_dir = os.path.dirname(os.path.abspath(__file__))
#print(pynets_dir)
#sys.exit()

parlistfile=args.ua

##Import/generate time-series and estimate GLOBAL covariance/sparse inverse covariance matrices
def import_mat_func(input_file, ID, atlas_select, NETWORK, pynets_dir, node_size, mask, thr, graph, parlistfile, sps_model, all_nets):
    if '.nii' in input_file and parlistfile == None and NETWORK == None:
        if graph == False:
            func_file=input_file

            if all_nets != None:
                func_img = nib.load(func_file)
                par_path = pynets_dir + '/RSN_refs/yeo.nii.gz'
                par_img = nib.load(par_path)
                par_data = par_img.get_data()

                ref_dict = {0:'unknown', 1:'VIS', 2:'SM', 3:'DA', 4:'VA', 5:'LIM', 6:'FP', 7:'DEF'}

                def get_ref_net(x, y, z):
                    aff_inv=npl.inv(func_img.affine)
                    # apply_affine(aff, (x,y,z)) # vox to mni
                    vox_coord = apply_affine(aff_inv, (x, y, z)) # mni to vox
                    return ref_dict[int(par_data[int(vox_coord[0]),int(vox_coord[1]),int(vox_coord[2])])]

            dir_path = os.path.dirname(os.path.realpath(func_file))
            atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
            atlas_name = atlas['description'].splitlines()[0]
            print(atlas_name + ' comes with {0}.'.format(atlas.keys()))
            print("\n")
            coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
            if all_nets != None:
                membership = pd.Series([get_ref_net(coord[0],coord[1],coord[2]) for coord in coords])
            print('Stacked atlas coordinates in array of shape {0}.'.format(coords.shape))
            print("\n")
            if mask is not None:
                from nilearn import masking
                mask_data, _ = masking._load_mask_img(mask)
                mask_coords = list(zip(*np.where(mask_data != 0)))
                for coord in coords:
                    if tuple(coord) not in mask_coords:
                        print('Removing coordinate: ' + str(tuple(coord)) + ' since it falls outside of network mask...')
                        ix = np.where(coords == coord)[0][0]
                        coords = np.delete(coords, ix, axis=0)
                        print(str(len(coords)))
                        print("\n")
            spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), memory='nilearn_cache', memory_level=5, verbose=2)
            time_series = spheres_masker.fit_transform(func_file)
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            print("\n")
            print('Time series has {0} samples'.format(time_series.shape[0]))
            print("\n")
        else:
            correlation_matrix = genfromtxt(graph, delimiter='\t')
        plt.imshow(correlation_matrix, vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest')
        plt.colorbar()
        plt.title(atlas_name + ' correlation matrix')
        out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_adj_mat_corr.png'
        plt.savefig(out_path_fig)
        plt.close()
        ##Tweak edge_threshold to keep only the strongest connections.
        atlast_graph_title = atlas_name + ' correlation graph'
        if mask is None:
            atlast_graph_title = atlas_name + ' correlation graph'
        else:
            atlast_graph_title = atlas_name + ' Masked Nodes'
        edge_threshold = str(float(thr)*100) +'%'

        # plot graph:
        if all_nets != None:
            # coloring code:
            n = len(membership.unique())
            clust_pal = sns.color_palette("Set1", n)
            clust_lut = dict(zip(map(str, np.unique(membership.astype('category'))), clust_pal))
            clust_colors = colors.to_rgba_array(membership.map(clust_lut))

            plotting.plot_connectome(correlation_matrix, coords, node_color = clust_colors, title=atlast_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True)
        else:
            plotting.plot_connectome(correlation_matrix, coords, title=atlast_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True)
        out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_connectome_viz.png'
        plt.savefig(out_path_fig)
        plt.close()
        time_series_path = dir_path + '/' + ID + '_ts.txt'
        np.savetxt(time_series_path, time_series, delimiter='\t')
        mx = genfromtxt(time_series_path, delimiter='')

    elif '.nii' in input_file and parlistfile != None and NETWORK == None: # block of code for whole brain parcellations
        if all_nets != None:
            par_path = pynets_dir + '/RSN_refs/yeo.nii.gz'
            par_img = nib.load(par_path)
            par_data = par_img.get_data()

            ref_dict = {0:'unknown', 1:'VIS', 2:'SM', 3:'DA', 4:'VA', 5:'LIM', 6:'FP', 7:'DEF'}

            def get_ref_net(x, y, z):
                aff_inv=npl.inv(bna_img.affine)
                # apply_affine(aff, (x,y,z)) # vox to mni
                vox_coord = apply_affine(aff_inv, (x, y, z)) # mni to vox
                return ref_dict[int(par_data[int(vox_coord[0]),int(vox_coord[1]),int(vox_coord[2])])]

        func_file=input_file
        dir_path = os.path.dirname(os.path.realpath(func_file))

        atlas_name = parlistfile.split('/')[-1].split('.')[0]
        # Code for getting name and coordinates of parcels.
        # Adapted from Dan L. (https://github.com/danlurie/despolab_lesion/blob/master/code/sandbox/Sandbox%20-%20Calculate%20and%20plot%20HCP%20mean%20matrix.ipynb)
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

        bna_4D = nilearn.image.concat_imgs(img_list)
        coords = []
        for roi_img in img_list:
            coords.append(nilearn.plotting.find_xyz_cut_coords(roi_img))
        coords = np.array(coords)
        if all_nets != None:
            membership = pd.Series([get_ref_net(coord[0],coord[1],coord[2]) for coord in coords])
        # atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
        # atlas_name = atlas['description'].splitlines()[0]
        print("\n")
        print(atlas_name + ' comes with {0}.'.format(par_max) + 'parcels')
        print("\n")
        print("\n")
        print('Stacked atlas coordinates in array of shape {0}.'.format(coords.shape))
        print("\n")
        if mask is not None:
            from nilearn import masking
            mask_data, _ = masking._load_mask_img(mask)
            mask_coords = list(zip(*np.where(mask_data != 0)))
            for coord in coords:
                if tuple(coord) not in mask_coords:
                    print('Removing coordinate: ' + str(tuple(coord)) + ' since it falls outside of network mask...')
                    ix = np.where(coords == coord)[0][0]
                    coords = np.delete(coords, ix, axis=0)
                    print(str(len(coords)))

        ##extract time series from whole brain parcellaions:
        parcellation = nib.load(parlistfile)
        parcel_masker = input_data.NiftiLabelsMasker(labels_img=parcellation, background_label=0, memory='nilearn_cache', memory_level=5)
        time_series = parcel_masker.fit_transform(func_file)
        ##old ref code for coordinate parcellations:
        #spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), memory='nilearn_cache', memory_level=2, verbose=2)
        #time_series = spheres_masker.fit_transform(func_file)
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        print("\n")
        print('Time series has {0} samples'.format(time_series.shape[0]))
        print("\n")
        plt.imshow(correlation_matrix, vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest')
        plt.colorbar()
        plt.title(atlas_name + ' correlation matrix')
        out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_adj_mat_corr.png'
        plt.savefig(out_path_fig)
        plt.close()
        ##Tweak edge_threshold to keep only the strongest connections.
        atlast_graph_title = atlas_name + ' correlation graph'
        if mask is None:
            atlast_graph_title = atlas_name + ' correlation graph'
        else:
            atlast_graph_title = atlas_name + ' Masked Nodes'
        edge_threshold = str(float(thr)*100) +'%'

        if all_nets != None:
            # coloring code:
            n = len(membership.unique())
            clust_pal = sns.color_palette("Set1", n)
            clust_lut = dict(zip(map(str, np.unique(membership.astype('category'))), clust_pal))
            clust_colors = colors.to_rgba_array(membership.map(clust_lut))
            plotting.plot_connectome(correlation_matrix, coords, node_color = clust_colors, title=atlast_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True)
        else:
            plotting.plot_connectome(correlation_matrix, coords, title=atlast_graph_title, edge_threshold=edge_threshold, node_size=20, colorbar=True)
        out_path_fig=dir_path + '/' + ID + '_' + atlas_name + '_connectome_viz.png'
        plt.savefig(out_path_fig)
        plt.close()
        time_series_path = dir_path + '/' + ID + '_ts.txt'
        np.savetxt(time_series_path, time_series, delimiter='\t')
        mx = genfromtxt(time_series_path, delimiter='')

    elif '.nii' in input_file and NETWORK != None:
        func_file=input_file

        ##Reference RSN list
    	load_path= pynets_dir + '/RSN_refs/' + NETWORK + '_coords.csv'
    	df = pd.read_csv(load_path).ix[:,0:4]
    	i=1
    	coords = []
    	labels = []
    	for i in range(len(df)):
      	    print("ROI Reference #: " + str(i))
      	    x = int(df.ix[i,1])
      	    y = int(df.ix[i,2])
      	    z = int(df.ix[i,3])
      	    print("X:" + str(x) + " Y:" + str(y) + " Z:" + str(z))
      	    coords.append((x, y, z))
      	    labels.append(i)
      	print("\n")
     	print(coords)
      	print(labels)
      	print("\n")
      	print("-------------------")
      	i + 1
        dir_path = os.path.dirname(os.path.realpath(func_file))

        ##Grow ROIs
        ##If masking, remove those coords that fall outside of the mask
        if mask != None:
            from nilearn import masking
            mask_data, _ = masking._load_mask_img(mask)
            mask_coords = list(zip(*np.where(mask_data != 0)))
            for coord in coords:
                if coord in mask_coords:
                    print('Removing coordinate: ' + str(coord) + ' since it falls outside of network mask...')
                    coords.remove(coord)
        masker = input_data.NiftiSpheresMasker(
            seeds=coords, radius=float(node_size), allow_overlap=True, memory_level=5,
            memory='nilearn_cache', verbose=2)
        time_series = masker.fit_transform(func_file)
        for time_serie, label in zip(time_series.T, labels):
            plt.plot(time_serie, label=label)
        plt.title(NETWORK + ' Network Time Series')
        plt.xlabel('Scan Number')
        plt.ylabel('Normalized Signal')
        plt.legend()
        plt.tight_layout()
        out_path_fig=dir_path + '/' + ID + '_' + NETWORK + '_TS_plot.png'
        plt.savefig(out_path_fig)
        plt.close()
        connectivity_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = connectivity_measure.fit_transform([time_series])[0]
        plot_title = NETWORK + ' Network Time Series'
        plotting.plot_connectome(correlation_matrix, coords,
                                 title=plot_title)
        ##Display connectome with hemispheric projections.
        title = "Connectivity Projected on the " + NETWORK
        out_path_fig=dir_path + '/' + ID + '_' + NETWORK + '_connectome_plot.png'
        plotting.plot_connectome(correlation_matrix, coords, title=title,
        display_mode='lyrz', output_file=out_path_fig)
        time_series_path = dir_path + '/' + ID + '_' + NETWORK + '_ts.txt'
        np.savetxt(time_series_path, time_series, delimiter='\t')
        mx = genfromtxt(time_series_path, delimiter='')
    else:
        DR_st_1=input_file
        dir_path = os.path.dirname(os.path.realpath(DR_st_1))
        mx = genfromtxt(DR_st_1, delimiter='')
    from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso
    estimator = GraphLassoCV()
    try:
        est = estimator.fit(mx)
    except:
#        print("WARNING: Lasso Cross-Validation Failed. Using Shrunk Covariance instead...")
#        emp_cov = covariance.empirical_covariance(mx)
#        shrunk_cov = covariance.shrunk_covariance(emp_cov, shrinkage=0.8) # Set shrinkage closer to 1 for poorly-conditioned data
#
#        alphaRange = 10.0 ** np.arange(-8,0) # 1e-7 to 1e-1 by order of magnitude
#        for alpha in alphaRange:
#            try:
#                estimator = covariance.graph_lasso(shrunk_cov, alpha)
#                print("Calculated graph-lasso covariance matrix for alpha=%s"%alpha)
#            except FloatingPointError:
#                print("Failed at alpha=%s"%alpha)
        estimator = ShrunkCovariance()
        est = estimator.fit(mx)
    if NETWORK != None:
        est_path = dir_path + '/' + ID + '_' + NETWORK + '_est%s.txt'%('_sps_inv' if sps_model else '')
    else:
        est_path = dir_path + '/' + ID + '_est%s.txt'%('_sps_inv' if sps_model else '')
    if sps_model == False:
        if NETWORK != None:
            np.savetxt(est_path, correlation_matrix, delimiter='\t')
        else:
            np.savetxt(est_path, correlation_matrix, delimiter='\t')
    elif sps_model == True:
        np.savetxt(est_path, estimator.precision_, delimiter='\t')
    return(mx, est_path)

##Create adj. plots for matrix interface
def mat_plt_func(mx, est_path, ID, NETWORK, sps_model):
    dir_path = os.path.dirname(os.path.realpath(est_path))
    est = genfromtxt(est_path)
    rois_num=est.shape[0]
    if NETWORK != None:
        if sps_model == False:
            print("Creating Correlation plot of dimensions:\n" + str(rois_num) + ' x ' + str(rois_num))
        elif sps_model == True:
            print("Creating Sparse Inverse Covariance plot of dimensions:\n" + str(rois_num) + ' x ' + str(rois_num))
        plt.figure(figsize=(rois_num, rois_num))
        ##The covariance can be found at estimator.covariance_
        plt.imshow(est, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
        ##And display the labels
        x_ticks = plt.xticks(range(rois_num), rotation=90)
        y_ticks = plt.yticks(range(rois_num))
        if sps_model == False:
            plt.title('Correlation')
        elif sps_model == True:
            plt.title('Sparse inverse covariance')
    A=np.matrix(est)
    G=nx.from_numpy_matrix(A)
    if NETWORK != None:
        G = nx.write_graphml(G, dir_path + '/' + ID + '_' + NETWORK + '.graphml')
        out_path=dir_path + '/' + ID + '_' + NETWORK + '_adj_mat%s.png'%('_sps_inv' if sps_model else '')
    else:
        G = nx.write_graphml(G, dir_path + '/' + ID + '.graphml')
        out_path=dir_path + '/' + ID + '_adj_mat%s.png'%('_sps_inv' if sps_model else '')
    plt.savefig(out_path)
    plt.close()
    return(est_path)

##Extract network metrics interface
def extractnetstats(est_path, ID, NETWORK, thr, sps_model, out_file=None):
    in_mat = np.array(genfromtxt(est_path))
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
    threshold_proportional(in_mat, thr)
    ##Get hyperbolic tangent of graph if non-sparse (i.e. fischer r-to-z transform), and divide by the variance of the matrix
    if sps_model == False:
        in_mat = np.arctanh(in_mat)/np.var(in_mat)
    dir_path = os.path.dirname(os.path.realpath(est_path))
    G=nx.from_numpy_matrix(in_mat)

###############################################################
############Calculate graph metrics from graph G###############
###############################################################
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, rich_club_coefficient, transitivity, betweenness_centrality
    #from networkx.algorithms.matching import min_maximal_matching

    from itertools import permutations
    import cPickle

    def efficiency(G, u, v):
        return float(1) / nx.shortest_path_length(G, u, v)

    def global_efficiency(G):
        n = len(G)
        denom = n * (n - 1)
        return float(sum(efficiency(G, u, v) for u, v in permutations(G, 2))) / denom

    def local_efficiency(G):
        return float(sum(global_efficiency(nx.ego_graph(G, v)) for v in G)) / len(G)

    metric_list = [global_efficiency, local_efficiency, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, rich_club_coefficient, transitivity]

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
        j = j + 1

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

    metric_list_names = []
    for i in metric_list:
        metric_list_names.append('%s' % i.func_name)

    if NETWORK != None:
        for i in bc_arr[:,0]:
            metric_list_names.append(i)

    ##Save metric names as pickle
    met_list_picke_path = os.path.dirname(os.path.abspath(est_path)) + '/met_list_pickle'
    cPickle.dump(metric_list_names, open(met_list_picke_path, 'wb'))

###############################################################
###############################################################
###############################################################

    ##Save results to csv
    if 'inv' in est_path:
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

class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    est_path = File(exists=True, mandatory=True, desc="")
    sub_id = traits.Str(mandatory=True)
    NETWORK = traits.Any(mandatory=True)
    thr = traits.Any(mandatory=True)
    sps_model = traits.Bool(mandatory=True)

class ExtractNetStatsOutputSpec(TraitedSpec):
    out_file = File()

class ExtractNetStats(BaseInterface):
    input_spec = ExtractNetStatsInputSpec
    output_spec = ExtractNetStatsOutputSpec

    def _run_interface(self, runtime):
        out = extractnetstats(
            self.inputs.est_path,
            self.inputs.sub_id,
            self.inputs.NETWORK,
	        self.inputs.thr,
            self.inputs.sps_model)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(getattr(self, '_outpath'))}

##save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, NETWORK, out_file=None):
    import cPickle
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

##Create input/output nodes
#1) Add variable to IdentityInterface if user-set
inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'NETWORK', 'pynets_dir', 'thr', 'node_size', 'mask', 'graph', 'parlistfile', 'sps_model', 'all_nets']), name='inputnode')

#2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
inputnode.inputs.in_file = input_file
inputnode.inputs.ID = ID
inputnode.inputs.atlas_select = atlas_select
inputnode.inputs.NETWORK = NETWORK
inputnode.inputs.pynets_dir = pynets_dir
inputnode.inputs.thr = thr
inputnode.inputs.node_size = node_size
inputnode.inputs.mask = mask
inputnode.inputs.graph = graph
inputnode.inputs.parlistfile = parlistfile
inputnode.inputs.sps_model = sps_model
inputnode.inputs.all_nets = all_nets

#3) Add variable to function nodes
##Create function nodes
imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select', 'NETWORK', 'pynets_dir', 'node_size', 'mask', 'thr', 'graph', 'parlistfile', 'sps_model', 'all_nets'], output_names = ['mx','est_path'], function=import_mat_func, imports=import_list), name = "imp_est")
cov_plt = pe.Node(niu.Function(input_names = ['mx', 'est_path', 'ID', 'NETWORK', 'sps_model'], output_names = ['est_path'], function=mat_plt_func, imports=import_list), name = "cov_plt")
net_mets_corr_node = pe.Node(ExtractNetStats(), name = "ExtractNetStats")
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
                          ('graph', 'graph'),
                          ('parlistfile', 'parlistfile'),
                          ('sps_model', 'sps_model'),
                          ('all_nets', 'all_nets')]),
    (inputnode, cov_plt, [('ID', 'ID'),
                          ('NETWORK', 'NETWORK'),
                          ('sps_model', 'sps_model')]),
    (imp_est, cov_plt, [('mx', 'mx'),
                        ('est_path', 'est_path')]),
    (imp_est, net_mets_corr_node, [('est_path', 'est_path')]),
    (inputnode, net_mets_corr_node, [('ID', 'sub_id'),
                               ('NETWORK', 'NETWORK'),
				               ('thr', 'thr'),
                               ('sps_model', 'sps_model')]),
    #(net_mets_cov_node, datasink, [('est_path', 'csv_loc')]),
    (inputnode, export_to_pandas_node, [('ID', 'sub_id'),
                                    ('NETWORK', 'NETWORK')]),
    (net_mets_corr_node, export_to_pandas_node, [('out_file', 'in_csv')]),
    #(export_to_pandas1, datasink, [('out_file', 'pandas_df)]),
])

#wf.run(plugin='SLURM')
#wf.run(plugin='MultiProc')
wf.run()
