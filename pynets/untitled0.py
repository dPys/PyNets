#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:11:43 2018

@author: PSYC-dap3463
"""

import numpy as np
import networkx as nx
conn_matrix = np.genfromtxt('/Users/PSYC-dap3463/Downloads/coords_dosenbach_2010/035_Default_est_corr_unthresholded_mat.txt')
dens_thresh = 0.40


def threshold_absolute(W, thr, copy=True):
    '''##Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < thr] = 0
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
    thr_max=0.50
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
    
    
    density_thresholding(conn_matrix, dens_thresh)