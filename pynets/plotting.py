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
