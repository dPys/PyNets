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
from pynets import graphestimation

def threshold_absolute(W, thr, copy=True):
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < thr] = 0
    return W

def threshold_proportional(W, p, copy=True):
    if p > 1 or p < 0:
        print('Threshold must be in range [0,1]')
        sys.exit()
    if copy:
        W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)
    if np.allclose(W, W.T):
        W[np.tril_indices(n)] = 0
        ud = 2
    else:
        ud = 1
    ind = np.where(W)
    I = np.argsort(W[ind])[::-1]
    en = int(round((n * n - n) * p / ud))
    W[(ind[0][I][en:], ind[1][I][en:])] = 0
    #W[np.ix_(ind[0][I][en:], ind[1][I][en:])]=0
    if ud == 2:
        W[:, :] = W + W.T
    return W

def normalize(W, copy=True):
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W

def density_thresholding(ts_within_spheres, conn_model, NETWORK, ID, dens_thresh, dir_path):
    thr=0.0
    [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID, dir_path, thr)
    conn_matrix = normalize(conn_matrix)
    np.fill_diagonal(conn_matrix, 0)
    i = 1
    thr_max=0.40
    G=nx.from_numpy_matrix(conn_matrix)
    density=nx.density(G)
    while float(thr) <= float(thr_max) and float(density) > float(dens_thresh):
        thr = float(thr) + float(0.01)
        conn_matrix = threshold_absolute(conn_matrix, thr)
        G=nx.from_numpy_matrix(conn_matrix)
        density=nx.density(G)

        print('Iteratively thresholding -- Iteration ' + str(i) + ' -- with absolute thresh: ' + str(thr) + ' and Density: ' + str(density) + '...')
        i = i + 1
    edge_threshold = str(float(thr)*100) +'%'
    est_path2 = est_path.split('_0.')[0] + '_' + str(dens_thresh) + '.txt'
    os.rename(est_path, est_path2)
    return(conn_matrix, est_path2, edge_threshold, dens_thresh)

##Calculate density
def est_density(func_mat):
    fG=nx.from_numpy_matrix(func_mat)
    density=nx.density(fG)
    return density

def thr2prob(W, copy=True):
    if copy:
        W = W.copy()
    W[W < 0.001] = 0
    return W

def binarize(W, copy=True):
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W

def adaptive_thresholding(ts_within_spheres, conn_model, NETWORK, ID, struct_mat_path, dir_path):
    import collections
    from pynets import binarize, thr2prob, est_density

    def thr_step(func_mat, thr):
        thr = float(thr) + float(0.01)
        func_mat = threshold_absolute(func_mat, thr)
        return func_mat

    ##Calculate # False Connections
    def est_error_rates(func_mat, struct_mat_bin, thr):
        func_mat = thr_step(func_mat, thr)
        func_mat_bin = binarize(func_mat)
        diffs = func_mat_bin - struct_mat_bin
        density = est_density(func_mat)
        unique, counts = np.unique(diffs, return_counts=True)
        accuracy_dict = dict(zip(unique, counts))
        FN = accuracy_dict.get(-1.0)
        FP = accuracy_dict.get(1.0)
        FN_error = float(float(FN)/diffs.size)
        FP_error = float(float(FP)/diffs.size)
        total_err = float(float(FP + FN)/diffs.size)
        return(FP_error, FN_error, total_err, density)

    [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID, dir_path, thr)
    struct_mat = np.genfromtxt(struct_mat_path)
    print('Using reference structural matrix from: ' + struct_mat_path)

    ##Prep functional mx
    conn_matrix = normalize(conn_matrix)
    np.fill_diagonal(conn_matrix, 0)
    func_mat = conn_matrix
    func_mat_bin = binarize(func_mat)
    fG=nx.from_numpy_matrix(func_mat)
    density = est_density(func_mat)

    ##Prep Structural mx
    np.fill_diagonal(struct_mat, 0)
    struct_mat_thr2bin = thr2prob(struct_mat)
    struct_mat_bin = binarize(struct_mat_thr2bin)
    diffs = func_mat_bin - struct_mat_bin
    unique, counts = np.unique(diffs, return_counts=True)
    accuracy_dict = dict(zip(unique, counts))
    FN = accuracy_dict.get(-1.0)
    ACC = accuracy_dict.get(0.0)
    FP = accuracy_dict.get(1.0)
    FN_error = float(float(FN)/float(diffs.size))
    FP_error = float(float(FP)/float(diffs.size))
    print('FN Error: ' + str(FN_error))
    print('FP Error: ' + str(FP_error))
    ACCUR = float(float(ACC)/float(diffs.size))
    total_err = float(float(FP + FN)/diffs.size)
    print('Using Structural Correspondence as Ground Truth. Unthresholded FP Error: ' + str(FP_error*100) + '%' + '; Unthresholded FN Error: ' + str(FN_error*100) + '%' + '; Unthresholded Accuracy: ' + str(ACCUR*100) + '%')
    print('Adaptively thresholding...')

    thr=0.0
    ##Create dictionary
    d = {}
    d[str(thr)] = [FP_error, FN_error, total_err, density]
    print('Creating dictionary of thresholds...')
    while thr < 0.2:
        [FP_error, FN_error, total_err, density] = est_error_rates(func_mat, struct_mat_bin, thr)
        d[str(thr)] = [round(FP_error,2), round(FN_error,2), round(total_err,2), round(density,2)]
        thr = thr + 0.0001

    d = collections.OrderedDict(sorted(d.items()))
    good_threshes=[]
    for key, value in d.items():
        if value[0] == value[1]:
            good_threshes.append(float(key))

    [conn_matrix, est_path] = graphestimation.get_conn_matrix(ts_within_spheres, conn_model, NETWORK, ID, dir_path, thr)
    conn_matrix = normalize(conn_matrix)
    np.fill_diagonal(conn_matrix, 0)
    min_thresh = min(good_threshes)
    FP = d[str(min_thresh)][0]
    FN = d[str(min_thresh)][1]
    FN_error = float(float(FN)/float(diffs.size))
    FP_error = float(float(FP)/float(diffs.size))
    density = est_density(conn_matrix)
    print('\n\n\nBest Threshold: ' + str(min_thresh))
    print('Graph Density: ' + str(density))
    print('Final Thresholded FN Error: ' + str(FN_error))
    print('Final Thresholded FP Error: ' + str(FP_error) + '\n\n\n')
    conn_matrix = threshold_absolute(conn_matrix, min_thresh)
    edge_threshold = str(float(min_thresh)*100) +'%'
    return(conn_matrix, est_path, edge_threshold, min_thresh)

def binarize(W, copy=True):
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W

def invert(W, copy=False):
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W

def weight_conversion(W, wcm, copy=True):
    if wcm == 'binarize':
        return binarize(W, copy)
    elif wcm == 'lengths':
        return invert(W, copy)

def autofix(W, copy=True):
    if copy:
        W = W.copy()
    # zero diagonal
    np.fill_diagonal(W, 0)
    # remove np.inf and np.nan
    try:
        W[np.logical_or(np.where(np.isinf(W)), np.where(np.isnan(W)))] = 0
    except:
        pass
    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimal=5)
    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)
    return W
