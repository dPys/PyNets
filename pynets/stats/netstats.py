#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
from __future__ import division
import numpy as np
import warnings
import networkx as nx
import tkinter
import matplotlib
matplotlib.use('agg')
warnings.filterwarnings("ignore")


def timeout(seconds):
    """
    Timeout function for hung calculations during automated graph analysis.
    """
    from functools import wraps
    import errno
    import os
    import signal

    class TimeoutError(Exception):
        pass

    def decorator(func):
        def _handle_timeout(signum, frame):
            error_message = os.strerror(errno.ETIME)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(240)
def average_shortest_path_length_for_all(G):
    """
    Helper function, in the case of graph disconnectedness,
    that returns the average shortest path length, calculated
    iteratively for each distinct subgraph of the graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    average_shortest_path_length : float
        The length of the average shortest path for graph G.

    Notes
    -----
    A helper function for calculating average shortest path length
    in the case that a graph is disconnected. Calculation occurs
    across all subgraphs detected in G.
    """
    import math
    connected_component_subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    subgraphs = [sbg for sbg in connected_component_subgraphs if len(sbg) > 1]

    return math.fsum(nx.average_shortest_path_length(sg) for sg in subgraphs) / len(subgraphs)


@timeout(120)
def global_efficiency(G, weight=None):
    """
    Return the global efficiency of the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    global_efficiency : float

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege exists, construct the
    corresponding weighted graph and calculate its global_efficiency to scale
    the weighted graph.

    Distance between nodes is calculated as the sum of weights. If the graph is
    defined such that a higher weight represents a stronger connection,
    distance should be represented by 1/weight. In this case, use the invert_
    weights function to generate a graph where the weights are set to 1/weight
    and then calculate efficiency

    References
    ----------
    .. Adapted from NetworkX to incorporate weight parameter
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    N = len(G)
    if N < 2:
        return 0

    inv_lengths = []
    for node in G:
        if weight is None:
            lengths = nx.single_source_shortest_path_length(G, node)
        else:
            lengths = nx.single_source_dijkstra_path_length(G, node, weight=weight)

        inv = [1 / x for x in lengths.values() if x is not 0]
        inv_lengths.extend(inv)

    return sum(inv_lengths) / (N * (N - 1))


@timeout(120)
def local_efficiency(G, weight=None):
    """
    Return the local efficiency of each node in the graph G

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    local_efficiency : dict
       The keys of the dict are the nodes in the graph G and the corresponding
       values are local efficiencies of each node

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege exists, construct the
    corresponding weighted graph and calculate its local_efficiency to scale
    the weighted graph.

    References
    ----------
    .. Adapted from NetworkX to incorporate weight parameter
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    if G.is_directed():
        new_graph = nx.DiGraph
    else:
        new_graph = nx.Graph

    efficiencies = dict()
    for node in G:
        temp_G = new_graph()
        temp_G.add_nodes_from(G.neighbors(node))
        for neighbor in G.neighbors(node):
            for (n1, n2) in G.edges(neighbor):
                if (n1 in temp_G) and (n2 in temp_G):
                    temp_G.add_edge(n1, n2)

        if weight is not None:
            for (n1, n2) in temp_G.edges():
                temp_G[n1][n2][weight] = G[n1][n2][weight]

        efficiencies[node] = global_efficiency(temp_G, weight)

    return efficiencies


@timeout(120)
def average_local_efficiency(G, weight=None):
    """
    Return the average local efficiency of all of the nodes in the graph G

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    average_local_efficiency : float
        Average local efficiency of graph G.

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted network, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege existed, a revised
    version of this function would be required.

    References
    ----------
    .. Adapted from NetworkX to incorporate weight parameter
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    eff = local_efficiency(G, weight)
    total = sum(eff.values())
    N = len(eff)
    return total / N


def create_random_graph(G, n, p):
    """
    Creates a random Erdos Renyi Graph

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.

    Returns
    -------
    rG : Obj
        Random networkX graph.
    """
    rG = nx.erdos_renyi_graph(n, p, seed=42)
    return rG


def smallworldness_measure(G, rG):
    """
    Calculates smallworldness measure

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    rG : Obj
        Random networkX graph.

    Returns
    -------
    swm : float
        Smallworldness measure.
    """
    C_g = nx.algorithms.average_clustering(G)
    C_r = nx.algorithms.average_clustering(rG)
    try:
        L_g = nx.average_shortest_path_length(G)
        L_r = nx.average_shortest_path_length(rG)
    except:
        L_g = average_shortest_path_length_for_all(G)
        L_r = average_shortest_path_length_for_all(rG)
    gam = float(C_g) / float(C_r)
    lam = float(L_g) / float(L_r)
    swm = gam / lam
    return swm


def smallworldness(G, rep=1000):
    """
    Calculates smallworldness.

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    rep : int
        Number of repetitions. Default is 1000.

    Returns
    -------
    mean_s : float
        Mean smallworldness measure across repetitions.
    """
    print("%s%s%s" % ('Estimating smallworldness using ', rep, ' random graphs...'))
    #import multiprocessing
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


def create_communities(node_comm_aff_mat, node_num):
    """
    Create a 1D vector of community assignments from a community affiliation matrix.

    Parameters
    ----------
    node_comm_aff_mat : array
        Community affiliation matrix produced from modularity estimation (e.g. Louvain).

    node_num : int
        Number of total connected nodes in the graph used to estimate node_comm_aff_mat.

    Returns
    -------
    com_assign : array
        1D numpy vector of community assignments.
    """
    com_assign = np.zeros((node_num, 1))
    for i in range(len(node_comm_aff_mat)):
        community = node_comm_aff_mat[i, :]
        for j in range(len(community)):
            if community[j] == 1:
                com_assign[j, 0] = i
    return com_assign


@timeout(120)
def participation_coef(W, ci, degree='undirected'):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    degree : str
        Flag to describe nature of graph 'undirected': For undirected graphs
                                         'in': Uses the in-degree
                                         'out': Uses the out-degree
    Returns
    -------
    P : Nx1 np.ndarray
        Participation coefficient

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors
    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P


def participation_coef_sign(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices

    def pcoef(W_):
        S = np.sum(W_, axis=1)  # strength
        # neighbor community affil.
        Gc = np.dot(np.logical_not(W_ == 0), np.diag(ci))
        Sc2 = np.zeros((n,))

        for i in range(1, int(np.max(ci) + 1)):
            Sc2 += np.square(np.sum(W_ * (Gc == i), axis=1))

        P = np.ones((n,)) - Sc2 / np.square(S)
        P[np.where(np.isnan(P))] = 0
        P[np.where(np.logical_not(P))] = 0  # p_ind=0 if no (out)neighbors
        return P

    # explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Ppos = pcoef(W * (W > 0))
        Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg


def diversity_coef_sign(W, ci):
    '''
    The Shannon-entropy based diversity coefficient measures the diversity
    of intermodular connections of individual nodes and ranges from 0 to 1.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector
    Returns
    -------
    Hpos : Nx1 np.ndarray
        diversity coefficient based on positive connections
    Hneg : Nx1 np.ndarray
        diversity coefficient based on negative connections

    References
    ----------
    Adapted from bctpy
    '''

    def entropy(w_):
        # Strength
        S = np.sum(w_, axis=1)
        # Node-to-module degree
        Snm = np.zeros((n, m))
        for i in range(m):
            Snm[:, i] = np.sum(w_[:, ci == i + 1], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0
        pnm[np.logical_not(pnm)] = 1
        return -np.sum(pnm * np.log(pnm), axis=1) / np.log(m)

    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    # Number of modules
    m = np.max(ci)

    # Explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


def link_communities(W, type_clustering='single'):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This algorithm uncovers overlapping community structure via hierarchical
    clustering of network links. This algorithm is generalized for
    weighted/directed/fully-connected networks

    Parameters
    ----------
    W : NxN np.array
        directed weighted/binary adjacency matrix
    type_clustering : str
        type of hierarchical clustering. 'single' for single-linkage,
        'complete' for complete-linkage. Default value='single'

    Returns
    -------
    M : CxN np.ndarray
        nodal community affiliation matrix.

    References
    ----------
    Adapted from bctpy
    '''
    from pynets.core.thresholding import normalize

    n = len(W)
    W = normalize(W)

    if type_clustering not in ('single', 'complete'):
        print('Error: Unrecognized clustering type')

    # Set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[range(n), range(n)] = (np.sum(W, axis=0) / np.sum(np.logical_not(W), axis=0) + np.sum(W.T, axis=0) /
                             np.sum(np.logical_not(W.T), axis=0)) / 2

    # Out/in norm squared
    No = np.sum(W ** 2, axis=1)
    Ni = np.sum(W ** 2, axis=0)

    # Weighted in/out jaccard
    Jo = np.zeros((n, n))
    Ji = np.zeros((n, n))

    for b in range(n):
        for c in range(n):
            Do = np.dot(W[b, :], W[c, :].T)
            Jo[b, c] = Do / (No[b] + No[c] - Do)

            Di = np.dot(W[:, b].T, W[:, c])
            Ji[b, c] = Di / (Ni[b] + Ni[c] - Di)

    # Get link similarity
    A, B = np.where(np.logical_and(np.logical_or(W, W.T), np.triu(np.ones((n, n)), 1)))
    m = len(A)
    # Link nodes
    Ln = np.zeros((m, 2), dtype=np.int32)
    # Link weights
    Lw = np.zeros((m,))

    for i in range(m):
        Ln[i, :] = (A[i], B[i])
        Lw[i] = (W[A[i], B[i]] + W[B[i], A[i]]) / 2

    # Link similarity
    ES = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if Ln[i, 0] == Ln[j, 0]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 1]
            elif Ln[i, 0] == Ln[j, 1]:
                a = Ln[i, 0]
                b = Ln[i, 1]
                c = Ln[j, 0]
            elif Ln[i, 1] == Ln[j, 0]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 1]
            elif Ln[i, 1] == Ln[j, 1]:
                a = Ln[i, 1]
                b = Ln[i, 0]
                c = Ln[j, 0]
            else:
                continue

            ES[i, j] = (W[a, b] * W[a, c] * Ji[b, c] + W[b, a] * W[c, a] * Jo[b, c]) / 2

    np.fill_diagonal(ES, 0)
    # Perform hierarchical clustering
    # Community affiliation matrix
    C = np.zeros((m, m), dtype=np.int32)
    Nc = C.copy()
    Mc = np.zeros((m, m), dtype=np.float32)
    # Community nodes, links, density
    Dc = Mc.copy()
    # Initial community assignments
    U = np.arange(m)
    C[0, :] = np.arange(m)

    for i in range(m - 1):
        print('Hierarchy %i' % i)
        # time1 = time.time()
        # Loop over communities
        for j in range(len(U)):
            # Get link indices
            ixes = C[i, :] == U[j]
            links = np.sort(Lw[ixes])
            # nodes = np.sort(Ln[ixes,:].flat)
            nodes = np.sort(np.reshape(
                Ln[ixes, :], 2 * np.size(np.where(ixes))))
            # Get unique nodes
            nodulo = np.append(nodes[0], (nodes[1:])[nodes[1:] != nodes[:-1]])
            # nodulo = ((nodes[1:])[nodes[1:] != nodes[:-1]])
            nc = len(nodulo)
            # nc = len(nodulo)+1
            mc = np.sum(links)
            # Minimal weight
            min_mc = np.sum(links[:nc - 1])
            # Community density
            dc = (mc - min_mc) / (nc * (nc - 1) / 2 - min_mc)
            if np.array(dc).shape is not ():
                print(dc)
                print(dc.shape)
            Nc[i, j] = nc
            Mc[i, j] = mc
            Dc[i, j] = dc if not np.isnan(dc) else 0
        # time2 = time.time()
        # print('compute densities time', time2-time1)
        # Copy current partition
        C[i + 1, :] = C[i, :]
        # if i in (2693,):
        #    import pdb
        #    pdb.set_trace()
        u1, u2 = np.where(ES[np.ix_(U, U)] == np.max(ES[np.ix_(U, U)]))
        if np.size(u1) > 2:
            wehr, = np.where((u1 == u2[0]))
            uc = np.squeeze((u1[0], u2[0]))
            ud = np.squeeze((u1[wehr], u2[wehr]))
            u1 = uc
            u2 = ud

        # time25 = time.time()
        # print('copy and max time', time25-time2)
        # ugl = np.array((u1,u2))
        ugl = np.sort((u1, u2), axis=1)
        ug_rows = ugl[np.argsort(ugl, axis=0)[:, 0]]
        # implementation of matlab unique(A, 'rows')
        unq_rows = np.vstack({tuple(row) for row in ug_rows})
        V = U[unq_rows]
        # time3 = time.time()
        # print('sortrows time', time3-time25)

        for j in range(len(V)):
            if type_clustering == 'single':
                x = np.max(ES[V[j, :], :], axis=0)
            elif type_clustering == 'complete':
                x = np.min(ES[V[j, :], :], axis=0)
            # Assign distances to whole clusters
            #            import pdb
            #            pdb.set_trace()
            ES[V[j, :], :] = np.array((x, x))
            ES[:, V[j, :]] = np.transpose((x, x))
            # clear diagonal
            ES[V[j, 0], V[j, 0]] = 0
            ES[V[j, 1], V[j, 1]] = 0
            # merge communities
            C[i + 1, C[i + 1, :] == V[j, 1]] = V[j, 0]
            V[V == V[j, 1]] = V[j, 0]

        # time4 = time.time()
        # print('get linkages time', time4-time3)
        U = np.unique(C[i + 1, :])
        if len(U) == 1:
            break
        # time5 = time.time()
        # print('get unique communities time', time5-time4)
    # Dc[ np.where(np.isnan(Dc)) ]=0
    i = np.argmax(np.sum(Dc * Mc, axis=1))
    U = np.unique(C[i, :])
    M = np.zeros((len(U), n))
    for j in range(len(U)):
        M[j, np.unique(Ln[C[i, :] == U[j], :])] = 1

    M = M[np.sum(M, axis=1) > 2, :]
    return M


def prune_disconnected(G):
    """
    Returns a copy of G with isolates pruned.

    Parameters
    ----------
    G : Obj
        NetworkX graph with isolated nodes present.

    Returns
    -------
    G : Obj
        NetworkX graph with isolated nodes pruned.
    pruned_nodes : list
        List of indices of nodes that were pruned from G.
    """
    print('Pruning disconnected...')

    # List because it returns a generator
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    components_isolated = list(components[0])

    # Remove disconnected nodes
    pruned_nodes = []
    s = 0
    for node in list(G.nodes()):
        if node not in components_isolated:
            G.remove_node(node)
            pruned_nodes.append(s)
        s = s + 1

    return G, pruned_nodes


def most_important(G):
    """
    Returns a copy of G with isolates and low-importance nodes pruned

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    G : Obj
        NetworkX graph with isolated and low-importance nodes pruned.
    pruned_nodes : list
       List of indices of nodes that were pruned from G.
    """
    print('Pruning fully disconnected and low importance nodes (3 SD < M)...')
    ranking = nx.betweenness_centrality(G).items()
    # print(ranking)
    r = [x[1] for x in ranking]
    m = sum(r) / len(r) - 3 * np.std(r)
    Gt = G.copy()
    pruned_nodes = []
    i = 0
    # Remove near-zero isolates
    for k, v in ranking:
        if v < m:
            Gt.remove_node(k)
            pruned_nodes.append(i)
        i = i + 1

    # List because it returns a generator
    components = list(nx.connected_components(Gt))
    components.sort(key=len, reverse=True)
    components_isolated = list(components[0])

    # Remove disconnected nodes
    s = 0
    for node in list(Gt.nodes()):
        if node not in components_isolated:
            Gt.remove_node(node)
            pruned_nodes.append(s)
        s = s + 1

    return Gt, pruned_nodes


@timeout(300)
def raw_mets(G, i, custom_weight):
    """
    API that iterates across NetworkX algorithms for a graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    i : str
        Name of the NetworkX algorithm.
    custom_weight : float
        The edge attribute that holds the numerical value used as a weight.
        If None, then each edge has weight 1. Default is None.

    Returns
    -------
    net_met_val : float
        Value of the graph metric i that was calculated from G.
    """
    if str(i) is 'average_shortest_path_length':
        if nx.is_connected(G) is True:
            try:
                net_met_val = float(i(G))
            except:
                net_met_val = float(average_shortest_path_length_for_all(G))
        else:
            [H, _] = prune_disconnected(G)
            net_met_val = float(i(H))

    elif str(i) is 'smallworldness':
        try:
            net_met_val = float(i(G))
        except:
            [H, _] = prune_disconnected(G)
            net_met_val = float(i(H))
    else:
        if (custom_weight is not None) and (str(i) is 'degree_assortativity_coefficient' or str(i) is 'global_efficiency' or str(i) is 'average_local_efficiency' or str(i) is 'average_clustering'):
            custom_weight_param = 'weight = ' + str(custom_weight)
            net_met_val = float(i(G, custom_weight_param))
        else:
            net_met_val = float(i(G))

    return net_met_val


# Extract network metrics interface
def extractnetstats(ID, network, thr, conn_model, est_path, roi, prune, norm, binary, custom_weight=None):
    """
    Function interface for performing fully-automated graph analysis.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    custom_weight : float
        The edge attribute that holds the numerical value used as a weight.
        If None, then each edge has weight 1. Default is None.

    Returns
    -------
    out_path : str
        Path to .csv file where graph analysis results are saved.
    """
    import pandas as pd
    import yaml
    import os
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pathlib import Path
    from pynets.core import thresholding, utils
    from graspy.utils import pass_to_ranks

    # Advanced options
    fmt = 'edgelist_ssv'
    est_path_fmt = "%s%s" % ('.', est_path.split('.')[-1])

    # Load and threshold matrix
    if est_path_fmt == '.txt':
        in_mat_raw = np.array(np.genfromtxt(est_path))
    else:
        in_mat_raw = np.array(np.load(est_path))

    # De-diagnal and remove nan's and inf's, ensure edge weights are positive
    in_mat = np.array(np.abs(np.array(thresholding.autofix(in_mat_raw))))

    # Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix if non-covariance
    if (conn_model == 'corr') or (conn_model == 'partcorr'):
        in_mat = np.arctanh(in_mat)

    # Normalize connectivity matrix
    # By maximum edge weight
    if norm == 1:
        in_mat = thresholding.normalize(in_mat)
    # Apply log10
    elif norm == 2:
        in_mat = np.log10(in_mat)
    # Apply PTR simple-nonzero
    elif norm == 3:
        in_mat = pass_to_ranks(in_mat, method="simple-nonzero")
    # Apply PTR simple-all
    elif norm == 4:
        in_mat = pass_to_ranks(in_mat, method="simple-all")
    # Apply PTR zero-boost
    elif norm == 5:
        in_mat = pass_to_ranks(in_mat, method="zero-boost")
    # Apply standardization [0, 1]
    elif norm == 6:
        in_mat = thresholding.standardize(in_mat)
    else:
        pass

    # Binarize graph
    if binary is True:
        in_mat = thresholding.binarize(in_mat)

    # Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path))

    # Load numpy matrix as networkx graph
    G_pre = nx.from_numpy_matrix(in_mat)

    # Prune irrelevant nodes (i.e. nodes who are fully disconnected from the graph and/or those whose betweenness
    # centrality are > 3 standard deviations below the mean)
    if (prune == 1) or (nx.is_connected(G_pre) is True):
        if nx.is_connected(G_pre) is False:
            print('Warning: Graph is fragmented...\n')
        [G, _] = prune_disconnected(G_pre)
    elif prune == 2:
        print('Pruning to retain only most important nodes...')
        [G, _] = most_important(G_pre)
    else:
        print('Graph is connected...')
        G = G_pre

    # Get corresponding matrix
    in_mat = np.array(nx.to_numpy_matrix(G))

    # Saved pruned
    if (prune != 0) and (prune is not None):
        final_mat_path = "%s%s%s" % (est_path.split(est_path_fmt)[0], '_pruned_mat', est_path_fmt)
        utils.save_mat(in_mat, final_mat_path, fmt)
        print("%s%s" % ('Source File: ', final_mat_path))
    else:
        print("%s%s" % ('Source File: ', est_path))

    # Print graph summary
    print("%s%.2f%s" % ('\n\nThreshold: ', 100 * float(thr), '%'))

    info_list = list(nx.info(G).split('\n'))[2:]
    for i in info_list:
        print(i)

    # Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')

    # Load numpy matrix as networkx graph
    G_len = nx.from_numpy_matrix(mat_len)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # Calculate global and local metrics from graph G # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    import community
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, eigenvector_centrality, communicability_betweenness_centrality, clustering, degree_centrality, rich_club_coefficient
    from pynets.stats.netstats import average_local_efficiency, global_efficiency, participation_coef, participation_coef_sign, diversity_coef_sign, smallworldness_measure, smallworldness, create_random_graph
    # For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the
    # function (with a G-only input) to the netstats module.
    # metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient,
    #                     average_clustering, average_shortest_path_length, graph_number_of_cliques, transitivity,
    #                     smallworldness]
    metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient,
                        average_clustering, average_shortest_path_length, graph_number_of_cliques, transitivity]
    metric_list_comm = ['louvain_modularity']
    # with open("%s%s" % (str(Path(__file__).parent), '/global_graph_measures.yaml'), 'r') as stream:
    #     try:
    #         metric_dict_global = yaml.load(stream)
    #         metric_list_global = metric_dict_global['metric_list_global']
    #         print("%s%s%s" % ('\n\nCalculating global measures:\n', metric_list_global, '\n\n'))
    #     except FileNotFoundError:
    #         print('Failed to parse global_graph_measures.yaml')

    with open("%s%s" % (str(Path(__file__).parent), '/nodal_graph_measures.yaml'), 'r') as stream:
        try:
            metric_dict_nodal = yaml.load(stream)
            metric_list_nodal = metric_dict_nodal['metric_list_nodal']
            print("%s%s%s" % ('\n\nCalculating nodal measures:\n', metric_list_nodal, '\n\n'))
        except FileNotFoundError:
            print('Failed to parse nodal_graph_measures.yaml')

    # Note the use of bare excepts in preceding blocks. Typically, this is considered bad practice in python. Here,
    # we are exploiting it intentionally to facilitate uninterrupted, automated graph analysis even when algorithms are
    # undefined. In those instances, solutions are assigned NaN's.

    # Iteratively run functions from above metric list that generate single scalar output
    num_mets = len(metric_list_glob)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j = 0
    for i in metric_list_glob:
        met_name = str(i).split('<function ')[1].split(' at')[0]
        net_met = met_name
        try:
            try:
                net_met_val = raw_mets(G, i, custom_weight)
                if net_met_val == 1.0:
                    net_met_val = np.nan
                    print("%s%s%s" % ('WARNING: ', str(i), ' is undefined for graph G'))
            except:
                print("%s%s%s" % ('WARNING: ', net_met, ' failed for graph G.'))
                net_met_val = np.nan
        except:
            print("%s%s%s" % ('WARNING: ', str(i), ' is undefined for graph G'))
            net_met_val = np.nan
        net_met_arr[j, 0] = net_met
        net_met_arr[j, 1] = net_met_val
        print(net_met)
        print(str(net_met_val))
        print('\n')
        j = j + 1
    net_met_val_list = list(net_met_arr[:, 1])

    # Create a list of metric names for scalar metrics
    metric_list_names = []
    net_met_val_list_final = net_met_val_list
    for i in net_met_arr[:, 0]:
        metric_list_names.append(i)

    # Run miscellaneous functions that generate multiple outputs
    # Calculate modularity using the Louvain algorithm
    if 'louvain_modularity' in metric_list_comm:
        try:
            ci = community.best_partition(G)
            modularity = community.community_louvain.modularity(ci, G)
            metric_list_names.append('modularity')
            if modularity == 1.0:
                modularity = np.nan
                print('Louvain modularity calculation is undefined for graph G')
            net_met_val_list_final.append(modularity)
        except:
            print('Louvain modularity calculation is undefined for graph G')
            pass

    # Participation Coefficient by louvain community
    if 'participation_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Participation coefficient cannot be calculated for graph G in the absence of a '
                               'community affiliation vector')
            if len(in_mat[in_mat < 0.0]) > 0:
                pc_vector = participation_coef_sign(in_mat, list(ci.values()))[0]
            else:
                pc_vector = participation_coef(in_mat, list(ci.values()))
            print('\nExtracting Participation Coefficient vector for all network nodes...')
            pc_vals = list(pc_vector)
            pc_edges = list(range(len(pc_vector)))
            num_edges = len(pc_edges)
            pc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                pc_arr[j, 0] = "%s%s" % (str(pc_edges[j]), '_partic_coef')
                try:
                    pc_arr[j, 1] = pc_vals[j]
                except:
                    print("%s%s%s" % ('Participation coefficient is undefined for node ', str(j), ' of graph G'))
                    pc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            pc_arr[num_edges, 0] = 'average_participation_coefficient'
            nonzero_arr_partic_coef = np.delete(pc_arr[:, 1], [0])
            pc_arr[num_edges, 1] = np.mean(nonzero_arr_partic_coef)
            print("%s%s" % ('Mean Participation Coefficient across edges: ', str(pc_arr[num_edges, 1])))
            for i in pc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(pc_arr[:, 1])
        except:
            print('Participation coefficient cannot be calculated for graph G')
            pass

    # Diversity Coefficient by louvain community
    if 'diversity_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Diversity coefficient cannot be calculated for graph G in the absence of a community '
                               'affiliation vector')
            dc_vector = diversity_coef_sign(in_mat, list(ci.values()))[0]
            print('\nExtracting Diversity Coefficient vector for all network nodes...')
            dc_vals = list(dc_vector)
            dc_edges = list(range(len(dc_vector)))
            num_edges = len(dc_edges)
            dc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                dc_arr[j, 0] = "%s%s" % (str(dc_edges[j]), '_diversity_coef')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Diversity coefficient is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            dc_arr[num_edges, 0] = 'average_diversity_coefficient'
            nonzero_arr_diversity_coef = np.delete(dc_arr[:, 1], [0])
            dc_arr[num_edges, 1] = np.mean(nonzero_arr_diversity_coef)
            print("%s%s" % ('Mean Diversity Coefficient across edges: ', str(dc_arr[num_edges, 1])))
            for i in dc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
        except:
            print('Diversity coefficient cannot be calculated for graph G')
            pass

    # Local Efficiency
    if 'local_efficiency' in metric_list_nodal:
        try:
            le_vector = local_efficiency(G)
            print('\nExtracting Local Efficiency vector for all network nodes...')
            le_vals = list(le_vector.values())
            le_nodes = list(le_vector.keys())
            num_nodes = len(le_nodes)
            le_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                le_arr[j, 0] = "%s%s" % (str(le_nodes[j]), '_local_efficiency')
                try:
                    le_arr[j, 1] = le_vals[j]
                except:
                    print("%s%s%s" % ('Local efficiency is undefined for node ', str(j), ' of graph G'))
                    le_arr[j, 1] = np.nan
                j = j + 1
            le_arr[num_nodes, 0] = 'average_local_efficiency_nodewise'
            nonzero_arr_le = np.delete(le_arr[:, 1], [0])
            le_arr[num_nodes, 1] = np.mean(nonzero_arr_le)
            print("%s%s" % ('Mean Local Efficiency across nodes: ', str(le_arr[num_nodes, 1])))
            for i in le_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(le_arr[:, 1])
        except:
            print('Local efficiency cannot be calculated for graph G')
            pass

    # Local Clustering
    if 'local_clustering' in metric_list_nodal:
        try:
            cl_vector = clustering(G)
            print('\nExtracting Local Clustering vector for all network nodes...')
            cl_vals = list(cl_vector.values())
            cl_nodes = list(cl_vector.keys())
            num_nodes = len(cl_nodes)
            cl_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                cl_arr[j, 0] = "%s%s" % (str(cl_nodes[j]), '_local_clustering')
                try:
                    cl_arr[j, 1] = cl_vals[j]
                except:
                    print("%s%s%s" % ('Local clustering is undefined for node ', str(j), ' of graph G'))
                    cl_arr[j, 1] = np.nan
                j = j + 1
            cl_arr[num_nodes, 0] = 'average_local_efficiency_nodewise'
            nonzero_arr_cl = np.delete(cl_arr[:, 1], [0])
            cl_arr[num_nodes, 1] = np.mean(nonzero_arr_cl)
            print("%s%s" % ('Mean Local Clustering across nodes: ', str(cl_arr[num_nodes, 1])))
            for i in cl_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(cl_arr[:, 1])
        except:
            print('Local clustering cannot be calculated for graph G')
            pass

    # Degree centrality
    if 'degree_centrality' in metric_list_nodal:
        try:
            dc_vector = degree_centrality(G)
            print('\nExtracting Degree Centrality vector for all network nodes...')
            dc_vals = list(dc_vector.values())
            dc_nodes = list(dc_vector.keys())
            num_nodes = len(dc_nodes)
            dc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                dc_arr[j, 0] = "%s%s" % (str(dc_nodes[j]), '_degree_centrality')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Degree centrality is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                j = j + 1
            dc_arr[num_nodes, 0] = 'average_degree_cent'
            nonzero_arr_dc = np.delete(dc_arr[:, 1], [0])
            dc_arr[num_nodes, 1] = np.mean(nonzero_arr_dc)
            print("%s%s" % ('Mean Degree Centrality across nodes: ', str(dc_arr[num_nodes, 1])))
            for i in dc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
        except:
            print('Degree centrality cannot be calculated for graph G')
            pass

    # Betweenness Centrality
    if 'betweenness_centrality' in metric_list_nodal:
        try:
            bc_vector = betweenness_centrality(G_len, normalized=True)
            print('\nExtracting Betweeness Centrality vector for all network nodes...')
            bc_vals = list(bc_vector.values())
            bc_nodes = list(bc_vector.keys())
            num_nodes = len(bc_nodes)
            bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                bc_arr[j, 0] = "%s%s" % (str(bc_nodes[j]), '_betweenness_centrality')
                try:
                    bc_arr[j, 1] = bc_vals[j]
                except:
                    print("%s%s%s" % ('Betweeness centrality is undefined for node ', str(j), ' of graph G'))
                    bc_arr[j, 1] = np.nan
                j = j + 1
            bc_arr[num_nodes, 0] = 'average_betweenness_centrality'
            nonzero_arr_betw_cent = np.delete(bc_arr[:, 1], [0])
            bc_arr[num_nodes, 1] = np.mean(nonzero_arr_betw_cent)
            print("%s%s" % ('Mean Betweenness Centrality across nodes: ', str(bc_arr[num_nodes, 1])))
            for i in bc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(bc_arr[:, 1])
        except:
            print('Betweenness centrality cannot be calculated for graph G')
            pass

    # Eigenvector Centrality
    if 'eigenvector_centrality' in metric_list_nodal:
        try:
            ec_vector = eigenvector_centrality(G, max_iter=1000)
            print('\nExtracting Eigenvector Centrality vector for all network nodes...')
            ec_vals = list(ec_vector.values())
            ec_nodes = list(ec_vector.keys())
            num_nodes = len(ec_nodes)
            ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                ec_arr[j, 0] = "%s%s" % (str(ec_nodes[j]), '_eigenvector_centrality')
                try:
                    ec_arr[j, 1] = ec_vals[j]
                except:
                    print("%s%s%s" % ('Eigenvector centrality is undefined for node ', str(j), ' of graph G'))
                    ec_arr[j, 1] = np.nan
                j = j + 1
            ec_arr[num_nodes, 0] = 'average_eigenvector_centrality'
            nonzero_arr_eig_cent = np.delete(ec_arr[:, 1], [0])
            ec_arr[num_nodes, 1] = np.mean(nonzero_arr_eig_cent)
            print("%s%s" % ('Mean Eigenvector Centrality across nodes: ', str(ec_arr[num_nodes, 1])))
            for i in ec_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(ec_arr[:, 1])
        except:
            print('Eigenvector centrality cannot be calculated for graph G')
            pass

    # Communicability Centrality
    if 'communicability_centrality' in metric_list_nodal:
        try:
            cc_vector = communicability_betweenness_centrality(G, normalized=True)
            print('\nExtracting Communicability Centrality vector for all network nodes...')
            cc_vals = list(cc_vector.values())
            cc_nodes = list(cc_vector.keys())
            num_nodes = len(cc_nodes)
            cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                cc_arr[j, 0] = "%s%s" % (str(cc_nodes[j]), '_communicability_centrality')
                try:
                    cc_arr[j, 1] = cc_vals[j]
                except:
                    print("%s%s%s" % ('Communicability centrality is undefined for node ', str(j), ' of graph G'))
                    cc_arr[j, 1] = np.nan
                j = j + 1
            cc_arr[num_nodes, 0] = 'average_communicability_centrality'
            nonzero_arr_comm_cent = np.delete(cc_arr[:, 1], [0])
            cc_arr[num_nodes, 1] = np.mean(nonzero_arr_comm_cent)
            print("%s%s" % ('Mean Communicability Centrality across nodes: ', str(cc_arr[num_nodes, 1])))
            for i in cc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(cc_arr[:, 1])
        except:
            print('Communicability centrality cannot be calculated for graph G')
            pass

    # Rich club coefficient
    if 'rich_club_coefficient' in metric_list_nodal:
        try:
            rc_vector = rich_club_coefficient(G, normalized=True)
            print('\nExtracting Rich Club Coefficient vector for all network nodes...')
            rc_vals = list(rc_vector.values())
            rc_edges = list(rc_vector.keys())
            num_edges = len(rc_edges)
            rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                rc_arr[j, 0] = "%s%s" % (str(rc_edges[j]), '_rich_club')
                try:
                    rc_arr[j, 1] = rc_vals[j]
                except:
                    print("%s%s%s" % ('Rich club coefficient is undefined for node ', str(j), ' of graph G'))
                    rc_arr[j, 1] = np.nan
                j = j + 1
            # Add mean
            rc_arr[num_edges, 0] = 'average_rich_club_coefficient'
            nonzero_arr_rich_club = np.delete(rc_arr[:, 1], [0])
            rc_arr[num_edges, 1] = np.mean(nonzero_arr_rich_club)
            print("%s%s" % ('Mean Rich Club Coefficient across edges: ', str(rc_arr[num_edges, 1])))
            for i in rc_arr[:, 0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(rc_arr[:, 1])
        except:
            print('Rich club coefficient cannot be calculated for graph G')
            pass

    # And save results to csv
    out_path_neat = "%s%s" % (utils.create_csv_path(dir_path, est_path).split('.csv')[0], '_neat.csv')
    df = pd.DataFrame.from_dict(dict(zip(metric_list_names, net_met_val_list_final)), orient='index').transpose()
    df.to_csv(out_path_neat, index=False)

    return out_path_neat
