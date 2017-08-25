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
import random
import itertools
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
from pynets import nodemaker, thresholding, graphestimation
from itertools import permutations
from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, rich_club_coefficient, eigenvector_centrality, communicability_centrality

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

def modularity(W, qtype='sta', seed=None):
    '''
    Input:      W       undirected (weighted or binary) connection matrix
                        with positive and negative weights
                qtype,  modularity type (see Rubinov and Sporns, 2011)
                           'sta',  Q_* (default if qtype is not specified)
                           'pos',  Q_+
                           'smp',  Q_simple
                           'gja',  Q_GJA
                           'neg',  Q_-
                seed,    random seed. Default None, seed from /dev/urandom
    Output:     Ci,     community affiliation vector
                Q,      modularity (qtype dependent)
    Note: Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)
    n = len(W)
    W0 = W * (W > 0)
    W1 = -W * (W < 0)
    s0 = np.sum(W0)
    s1 = np.sum(W1)
    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:
        s0 = 1
        d1 = 0
    if not s1:
        s1 = 1
        d1 = 0
    h = 1
    nh = n
    ci = [None, np.arange(n) + 1]
    q = [-1, 0]
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise KeyError('Modularity Infinite Loop')

        kn0 = np.sum(W0, axis=0)
        kn1 = np.sum(W1, axis=0)
        km0 = kn0.copy()
        km1 = kn1.copy()
        knm0 = W0.copy()
        knm1 = W1.copy()
        m = np.arange(nh) + 1
        flag = True
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise KeyError('Infinite Loop was detected and stopped.')

            flag = False
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                dQ0 = (knm0[u, :] + W0[u, u] - knm0[u, ma]) - kn0[u] * (
                    km0 + kn0[u] - km0[ma]) / s0
                dQ1 = (knm1[u, :] + W1[u, u] - knm1[u, ma]) - kn1[u] * (
                    km1 + kn1[u] - km1[ma]) / s1
                dQ = d0 * dQ0 - d1 * dQ1
                dQ[ma] = 0
                max_dQ = np.max(dQ)
                if max_dQ > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]
                    km1[ma] -= kn1[u]
                    m[u] = mb + 1
        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1
        for u in range(nh):
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]
        nh = np.max(m)
        wn0 = np.zeros((nh, nh))
        wn1 = np.zeros((nh, nh))
        for u in range(nh):
            for v in range(u, nh):
                wn0[u, v] = np.sum(W0[np.ix_(m == u + 1, m == v + 1)])
                wn1[u, v] = np.sum(W1[np.ix_(m == u + 1, m == v + 1)])
                wn0[v, u] = wn0[u, v]
                wn1[v, u] = wn1[u, v]
        W0 = wn0
        W1 = wn1
        q.append(0)
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1
    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1
    return ci_ret, q[-1]
