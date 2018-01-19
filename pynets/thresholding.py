# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import sys
import os
import numpy as np
import networkx as nx

def threshold_absolute(W, thr, copy=True):
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < thr] = 0
    return W

def threshold_proportional(W, p, copy=True):
    '''##Adapted from bctpy
    '''
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
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W

def density_thresholding(conn_matrix, dens_thresh):
    thr=0.0
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
    return(conn_matrix)

##Calculate density
def est_density(func_mat):
    '''##Adapted from bctpy
    '''
    fG=nx.from_numpy_matrix(func_mat)
    density=nx.density(fG)
    return density

def thr2prob(W, copy=True):
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W[W < 0.001] = 0
    return W

def binarize(W, copy=True):
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W

def invert(W, copy=False):
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W

def weight_conversion(W, wcm, copy=True):
    '''##Adapted from bctpy
    '''
    if wcm == 'binarize':
        return binarize(W, copy)
    elif wcm == 'lengths':
        return invert(W, copy)

def autofix(W, copy=True):
    '''##Adapted from bctpy
    '''
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
        W = np.around(W, decimals=5)
    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)
    return W

def thresh_and_fit(adapt_thresh, dens_thresh, thr, ts_within_nodes, conn_model, network, ID, dir_path, mask, node_size):
    from pynets import utils, thresholding, graphestimation
   
    ##Adaptive thresholding scenario
    if adapt_thresh is not False:
        try:
            est_path2 = dir_path + '/' + ID + '_structural_est.txt'
            if os.path.isfile(est_path2) == True:
                #[conn_matrix_thr, est_path, edge_threshold, thr] = adaptive_thresholding(ts_within_nodes, conn_model, network, ID, est_path2, dir_path)
                ##Save unthresholded
                unthr_path = utils.create_unthr_path(ID, network, conn_model, mask, dir_path)
                #np.savetxt(unthr_path, conn_matrix_thr, delimiter='\t')
                edge_threshold = str(float(thr)*100) +'%'
            else:
                print('No structural mx found! Exiting...')
                sys.exit()
        except:
            print('No structural mx assigned! Exiting...')
            sys.exit()
    else:
        if not dens_thresh:
            print('\nRunning graph estimation and thresholding proportionally at: ' + str(thr) + '% ...\n')
        else:
            print('\nRunning graph estimation and thresholding to achieve density of: ' + str(100*dens_thresh) + '% ...\n')
        ##Fit mat
        conn_matrix = graphestimation.get_conn_matrix(ts_within_nodes, conn_model)
        
        ##Save unthresholded
        unthr_path = utils.create_unthr_path(ID, network, conn_model, mask, dir_path)
        np.savetxt(unthr_path, conn_matrix, delimiter='\t')

        if not dens_thresh:
            ##Save thresholded
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
            edge_threshold = str(float(thr)*100) +'%'
            est_path = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size) 
        else:
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, dens_thresh)
            edge_threshold = str((1-float(dens_thresh))*100) +'%'
            est_path = utils.create_est_path(ID, network, conn_model, dens_thresh, mask, dir_path, node_size)
        np.savetxt(est_path, conn_matrix_thr, delimiter='\t')
    return(conn_matrix_thr, edge_threshold, est_path, thr)