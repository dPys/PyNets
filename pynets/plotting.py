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
#warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from numpy import genfromtxt
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
from scipy.cluster.hierarchy import linkage, fcluster
from nipype.utils.filemanip import load_json, save_json

def plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK, label_names, mask):
    ##Set title for adj. matrix based on connectivity model used
    if conn_model == 'corr':
        atlast_graph_title = atlas_name + '_Correlation_Graph'
    elif conn_model == 'partcorr':
        atlast_graph_title = atlas_name + '_Partial_Correlation_Graph'
    elif conn_model == 'sps':
        atlast_graph_title = atlas_name + '_Sparse_Covariance_Graph'
    elif conn_model == 'cov':
        atlast_graph_title = atlas_name + '_Covariance_Graph'
    if mask != None:
        atlast_graph_title = atlast_graph_title + '_With_Masked_Nodes'
    if NETWORK != None:
        atlast_graph_title = atlast_graph_title + '_' + NETWORK
        out_path_fig=dir_path + '/' + ID + '_' + NETWORK + '_adj_mat_' + conn_model + '_network.png'
    else:
        out_path_fig=dir_path + '/' + ID + '_adj_mat_' + conn_model + '.png'
    rois_num=conn_matrix.shape[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(conn_matrix, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    ##And display the labels
    if rois_num < 50:
        if all(isinstance(item, int) for item in label_names)==False:
            x_ticks = plt.xticks(range(len(label_names)), label_names, size='x-small', rotation=90)
            y_ticks = plt.yticks(range(len(label_names)), label_names, size='x-small')
        else:
            x_ticks = plt.xticks(range(rois_num), rotation=90)
            y_ticks = plt.yticks(range(rois_num))
    plt.title(atlast_graph_title)
    plt.grid(False)
    plt.savefig(out_path_fig)
    plt.close()
    return(atlast_graph_title)

def plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, NETWORK, label_names):
    import json
    from pynets.thresholding import normalize
    from pathlib import Path
    from random import sample
    from string import ascii_uppercase, ascii_lowercase
    link_comm = True

    conn_matrix = normalize(conn_matrix)
    G=nx.from_numpy_matrix(conn_matrix)

    def doClust(X, clust_levels):
        ##get the linkage diagram
        Z = linkage(X, 'ward', )
        ##choose # cluster levels
        cluster_levels = range(1,int(clust_levels))
        ##init array to store labels for each level
        clust_levels_tmp = int(clust_levels) - 1
        label_arr = np.zeros((int(clust_levels_tmp),int(X.shape[0])))
        ##iterate thru levels
        for c in cluster_levels:
            fl = fcluster(Z,c,criterion='maxclust')
            #print(fl)
            label_arr[c-1, :] = fl
        return label_arr, clust_levels_tmp

    if NETWORK is not None:
        clust_levels = 3
        [label_arr, clust_levels_tmp] = doClust(conn_matrix, clust_levels)
    else:
        if link_comm == True:
            from pynets.netstats import link_communities
            #G_lin = nx.line_graph(G)
            ##Plot link communities
            node_comm_aff_mat = link_communities(conn_matrix, type_clustering='single')
            clust_levels = len(node_comm_aff_mat)
            clust_levels_tmp = int(clust_levels) - 1
            mask_mat = np.squeeze(np.array([node_comm_aff_mat == 0]).astype('int'))
            label_arr = node_comm_aff_mat * np.expand_dims(np.arange(1,clust_levels+1),axis=1) + mask_mat
        #else:
            ##Plot node communities
            #from pynets.netstats import community_louvain
            #[ci, q] = community_louvain(conn_matrix, gamma=0.75)
            #clust_levels = len(np.unique(ci))
            #clust_levels_tmp = int(clust_levels) - 1

    def get_node_label(node_idx, labels, clust_levels_tmp):
        def get_letters(n, random=False, uppercase=False):
            """Return n letters of the alphabet."""
            letters = (ascii_uppercase if uppercase else ascii_lowercase)
            return json.dumps((sample(letters, n) if random else list(letters[:n])))
        abet = get_letters(clust_levels_tmp)
        node_labels = labels[:, node_idx]
        return ".".join(["{}{}".format(abet[i],int(l)) for i, l in enumerate(node_labels)])+".{}".format(label_names[node_idx])

    output = []
    for node_idx, connections in enumerate(G.adjacency_list()):
        weight_vec = []
        for i in connections:
            wei = G.get_edge_data(node_idx,int(i))['weight']
            #wei = G_lin.get_edge_data(node_idx,int(i))['weight']
            weight_vec.append(wei)
        entry = {}
        nodes_label = get_node_label(node_idx, label_arr, clust_levels_tmp)
        entry["name"] = nodes_label
        entry["size"] = len(connections)
        entry["imports"] = [get_node_label(int(d)-1, label_arr, clust_levels_tmp) for d in connections]
        entry["weights"] = weight_vec
        output.append(entry)

    if NETWORK != None:
        json_file_name = str(ID) + '_' + NETWORK + '_connectogram_' + conn_model + '_network.json'
        connectogram_plot = dir_path + '/' + json_file_name
        connectogram_js_sub = dir_path + '/' + str(ID) + '_' + NETWORK + '_connectogram_' + conn_model + '_network.js'
        connectogram_js_name = str(ID) + '_' + NETWORK + '_connectogram_' + conn_model + '_network.js'
    else:
        json_file_name = str(ID) + '_connectogram_' + conn_model + '.json'
        connectogram_plot = dir_path + '/' + json_file_name
        connectogram_js_sub = dir_path + '/' + str(ID) + '_connectogram_' + conn_model + '.js'
        connectogram_js_name = str(ID) + '_connectogram_' + conn_model + '.js'
    save_json(connectogram_plot, output)

    ##Copy index.html and json to dir_path
    #conn_js_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/connectogram.js'
    #index_html_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/index.html'
    conn_js_path = Path(__file__).parent/"connectogram.js"
    index_html_path = Path(__file__).parent/"index.html"
    replacements_html = {'connectogram.js': str(connectogram_js_name)}
    with open(index_html_path) as infile, open(str(dir_path + '/index.html'), 'w') as outfile:
        for line in infile:
            for src, target in replacements_html.items():
                line = line.replace(src, target)
            outfile.write(line)
    replacements_js = {'template.json': str(json_file_name)}
    with open(conn_js_path) as infile, open(connectogram_js_sub, 'w') as outfile:
        for line in infile:
            for src, target in replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

def plot_membership(membership_plotting, conn_matrix, conn_model, coords, edge_threshold, atlast_name, dir_path):
    atlast_connectome_title = atlas_name + '_all_networks'
    n = len(membership_plotting.unique())
    clust_pal = sns.color_palette("Set2", n)
    clust_lut = dict(zip(map(str, np.unique(membership_plotting.astype('category'))), clust_pal))
    clust_colors = colors.to_rgba_array(membership_plotting.map(clust_lut))
    out_path_fig = dir_path + '/' + ID + '_connectome_viz.png'
    niplot.plot_connectome(conn_matrix, coords, node_color = clust_colors, title=atlast_connectome_title, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)
    display.close()

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
