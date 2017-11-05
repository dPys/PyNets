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

def plot_conn_mat(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names, mask):
    ##Set title for adj. matrix based on connectivity model used
    if conn_model == 'corr':
        atlast_graph_title = str(atlas_name) + '_Correlation_Graph'
    elif conn_model == 'partcorr':
        atlast_graph_title = str(atlas_name) + '_Partial_Correlation_Graph'
    elif conn_model == 'sps':
        atlast_graph_title = str(atlas_name) + '_Sparse_Covariance_Graph'
    elif conn_model == 'cov':
        atlast_graph_title = str(atlas_name) + '_Covariance_Graph'
    if mask != None:
        atlast_graph_title = str(atlast_graph_title) + '_With_Masked_Nodes'
    if network != None:
        atlast_graph_title = str(atlast_graph_title) + '_' + str(network)
        out_path_fig=dir_path + '/' + str(ID) + '_' + str(network) + '_adj_mat_' + str(conn_model) + '_network.png'
    else:
        out_path_fig=dir_path + '/' + str(ID) + '_adj_mat_' + str(conn_model) + '.png'
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

def plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names):
    import json
    from pynets.thresholding import normalize
    from pathlib import Path
    from random import sample
    from string import ascii_uppercase, ascii_lowercase
    comm = 'nodes'

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

    if comm == 'nodes' and len(conn_matrix) > 80:
        from pynets.netstats import modularity_finetune_und_sign
        if network is not None and len(conn_matrix) > 80:
            gamma=0.3
        else:
            gamma=1.0
        [node_comm_aff_mat, q] = modularity_finetune_und_sign(conn_matrix, gamma=gamma)
        print('Found ' + str(len(node_comm_aff_mat)) + ' communities with gamma=' + str(gamma) + '...')
        clust_levels = len(node_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([node_comm_aff_mat == 0]).astype('int'))
        label_arr = node_comm_aff_mat * np.expand_dims(np.arange(1,clust_levels+1),axis=1) + mask_mat
    elif comm == 'links' and len(conn_matrix) > 80:
        from pynets.netstats import link_communities
        ##Plot link communities
        link_comm_aff_mat = link_communities(conn_matrix, type_clustering='single')
        print('Found ' + str(len(link_comm_aff_mat)) + ' communities...')
        clust_levels = len(link_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([link_comm_aff_mat == 0]).astype('int'))
        label_arr = link_comm_aff_mat * np.expand_dims(np.arange(1,clust_levels+1),axis=1) + mask_mat
    elif len(conn_matrix) <= 80 and len(conn_matrix) > 20:
        print('Graph too small for reliable plotting of communities. Plotting by fcluster instead...')
        if len(conn_matrix) >= 70:
            clust_levels = 7
        elif len(conn_matrix) >= 60:
            clust_levels = 6
        elif len(conn_matrix) >= 50:
            clust_levels = 5
        elif len(conn_matrix) >= 40:
            clust_levels = 4
        elif len(conn_matrix) >= 30:
            clust_levels = 3
        else:
            clust_levels = 2
        [label_arr, clust_levels_tmp] = doClust(conn_matrix, clust_levels)
    else:
        print('Error: Cannot plot connectogram for graphs smaller than 20 x 20!')
        sys.exit()

    def get_node_label(node_idx, labels, clust_levels_tmp):
        from collections import OrderedDict
        def write_roman(num):
            roman = OrderedDict()
            roman[1000] = "M"
            roman[900] = "CM"
            roman[500] = "D"
            roman[400] = "CD"
            roman[100] = "C"
            roman[90] = "XC"
            roman[50] = "L"
            roman[40] = "XL"
            roman[10] = "X"
            roman[9] = "IX"
            roman[5] = "V"
            roman[4] = "IV"
            roman[1] = "I"
            def roman_num(num):
                for r in roman.keys():
                    x, y = divmod(num, r)
                    yield roman[r] * x
                    num -= (r * x)
                    if num > 0:
                        roman_num(num)
                    else:
                        break
            return "".join([a for a in roman_num(num)])
        int_list = list(np.arange(clust_levels_tmp) + 1)[:-1]
        rn_list = []
        node_idx = node_idx - 1
        node_labels = labels[:, node_idx]
        for i in [int(l) for i, l in enumerate(node_labels)]:
            rn_list.append(json.dumps(write_roman(i)))
        abet = rn_list
        return ".".join(["{}{}".format(abet[i],int(l)) for i, l in enumerate(node_labels)])+".{}".format(label_names[node_idx])

    output = []
    for node_idx, connections in enumerate(G.adjacency_list()):
        weight_vec = []
        for i in connections:
            wei = G.get_edge_data(node_idx,int(i))['weight']
            weight_vec.append(wei)
        entry = {}
        nodes_label = get_node_label(node_idx, label_arr, clust_levels_tmp)
        entry["name"] = nodes_label
        entry["size"] = len(connections)
        entry["imports"] = [get_node_label(int(d)-1, label_arr, clust_levels_tmp) for d in connections]
        entry["weights"] = weight_vec
        output.append(entry)

    if network != None:
        json_file_name = str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.json'
        connectogram_plot = dir_path + '/' + json_file_name
        connectogram_js_sub = dir_path + '/' + str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.js'
        connectogram_js_name = str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.js'
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

    #color_scheme = 'interpolateCool'
    #color_scheme = 'interpolateGnBu'
    color_scheme = 'interpolateOrRd'
    #color_scheme = 'interpolatePuRd'
    #color_scheme = 'interpolateYlOrRd'
    #color_scheme = 'interpolateReds'
    #color_scheme = 'interpolateGreens'
    #color_scheme = 'interpolateBlues'
    replacements_js = {'template.json': str(json_file_name), 'interpolateCool': str(color_scheme)}
    with open(conn_js_path) as infile, open(connectogram_js_sub, 'w') as outfile:
        for line in infile:
            for src, target in replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

def plot_timeseries(time_series, network, ID, dir_path, atlas_name, labels):
    for time_serie, label in zip(time_series.T, labels):
        plt.plot(time_serie, label=label)
    plt.title(network + ' Network Time Series')
    plt.xlabel('Scan Number')
    plt.ylabel('Normalized Signal')
    plt.legend()
    #plt.tight_layout()
    if network != None:
        out_path_fig=dir_path + '/' + ID + '_' + network + '_TS_plot.png'
    else:
        out_path_fig=dir_path + '/' + ID + '_Whole_Brain_TS_plot.png'
    plt.savefig(out_path_fig)
    plt.close()
