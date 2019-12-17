#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import pandas as pd
import numpy as np
import warnings
import networkx as nx
from pynets.core import thresholding
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


@timeout(720)
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
    """
    import math
    connected_component_subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    subgraphs = [sbg for sbg in connected_component_subgraphs if len(sbg) > 1]

    return math.fsum(nx.average_shortest_path_length(sg, weight='weight') for sg in subgraphs) / len(subgraphs)


@timeout(720)
def subgraph_number_of_cliques_for_all(G):
    """
    Helper function, in the case of graph disconnectedness,
    that returns the number of cliques, calculated
    iteratively for each distinct subgraph of the graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    number of cliques : int
        The average number of cliques for graph G.
    """
    import math
    connected_component_subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    subgraphs = [sbg for sbg in connected_component_subgraphs if len(sbg) > 1]

    return np.rint(math.fsum(nx.graph_number_of_cliques(sg) for sg in subgraphs) / len(subgraphs))


@timeout(720)
def global_efficiency(G, weight='weight'):
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


@timeout(360)
def local_efficiency(G, weight='weight'):
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


@timeout(720)
def average_local_efficiency(G, weight='weight'):
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


@timeout(720)
def smallworldness(G, niter=10, nrand=100):
    """Returns the small-world coefficient (omega) of a graph

    The small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    The small-world coefficient (omega) ranges between -1 and 1. Values close
    to 0 means the G features small-world characteristics. Values close to -1
    means G has a lattice shape whereas values close to 1 means G is a random
    graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=10)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=100)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).


    Returns
    -------
    omega : float
        The small-work coefficient (omega)
    """

    from networkx.algorithms.smallworld import random_reference, lattice_reference

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=i)
        Gl = lattice_reference(G, niter=niter, seed=i)
        randMetrics["C"].append(weighted_transitivity(Gl))
        randMetrics["L"].append(nx.average_shortest_path_length(Gr, weight='weight'))
        del Gr, Gl

    C = weighted_transitivity(G)
    try:
        L = nx.average_shortest_path_length(G, weight='weight')
    except:
        L = average_shortest_path_length_for_all(G)
    Cl = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    return (Lr / L) - (C / Cl)


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


@timeout(360)
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

@timeout(360)
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

@timeout(360)
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


@timeout(360)
def weighted_transitivity(G):
    r"""Compute weighted graph transitivity, the fraction of all possible weighted triangles
    present in G.

    Possible triangles are identified by the number of "triads"
    (two edges with a shared vertex).

    The transitivity is

    .. math::

        T = 3\frac{\#triangles}{\#triads}.

    Parameters
    ----------
    G : graph

    Returns
    -------
    out : float
       Transitivity
    """
    from networkx.algorithms.cluster import _weighted_triangles_and_degree_iter
    triangles = sum(t for v, d, t in _weighted_triangles_and_degree_iter(G))
    contri = sum(d * (d - 1) for v, d, t in _weighted_triangles_and_degree_iter(G))
    return 0 if triangles == 0 else triangles / contri


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
    ranking = nx.betweenness_centrality(G, weight='weight').items()
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


@timeout(1200)
def raw_mets(G, i):
    """
    API that iterates across NetworkX algorithms for a graph G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    i : str
        Name of the NetworkX algorithm.

    Returns
    -------
    net_met_val : float
        Value of the graph metric i that was calculated from G.
    """
    from functools import partial
    if isinstance(i, partial):
        net_name = str(i.func)
    else:
        net_name = str(i)
    if 'average_shortest_path_length' in net_name:
        if nx.is_connected(G) is True:
            try:
                net_met_val = float(i(G))
            except:
                net_met_val = float(average_shortest_path_length_for_all(G))
        else:
            [H, _] = prune_disconnected(G)
            net_met_val = float(i(H))
    elif 'graph_number_of_cliques' in net_name:
        if nx.is_connected(G) is True:
            try:
                net_met_val = float(i(G))
            except:
                net_met_val = float(subgraph_number_of_cliques_for_all(G))
        else:
            [H, _] = prune_disconnected(G)
            net_met_val = float(i(H))
    elif 'smallworldness' in net_name:
        try:
            net_met_val = float(i(G))
        except:
            [H, _] = prune_disconnected(G)
            net_met_val = float(i(H))
    elif 'degree_assortativity_coefficient' in net_name:
            H = G.copy()
            for u, v, d in H.edges(data=True):
                H[u][v]['weight'] = int(np.round(100*H[u][v]['weight'], 1))
            net_met_val = float(i(H))
    else:
        net_met_val = float(i(G))

    return net_met_val


class CleanGraphs(object):
    """
    A Class for cleaning graphs in preparation for network analysis.

    Parameters
    ----------
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    norm : int
        Indicates method of normalizing resulting graph.

    Returns
    -------
    out_path : str
        Path to .csv file where graph analysis results are saved.
    """
    def __init__(self, thr, conn_model, est_path, prune, norm, out_fmt='edgelist_ssv'):
        self.thr = thr
        self.conn_model = conn_model
        self.est_path = est_path
        self.prune = prune
        self.norm = norm
        self.out_fmt = out_fmt
        self.in_mat = None
        self._est_path_fmt = "%s%s" % ('.', self.est_path.split('.')[-1])

        # Load and threshold matrix
        if self._est_path_fmt == '.txt':
            self.in_mat_raw = np.array(np.genfromtxt(self.est_path))
        else:
            self.in_mat_raw = np.array(np.load(self.est_path))

        # De-diagnal and remove nan's and inf's, ensure edge weights are positive
        self.in_mat = np.array(np.abs(np.array(thresholding.autofix(self.in_mat_raw))))

        # Load numpy matrix as networkx graph
        self.G = nx.from_numpy_matrix(self.in_mat)

    def normalize_graph(self):

        # Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix if non-covariance
        if (self.conn_model == 'corr') or (self.conn_model == 'partcorr'):
            self.in_mat = np.arctanh(self.in_mat)

        # Normalize connectivity matrix
        if self.norm == 3 or self.norm == 4 or self.norm == 5:
            from graspy.utils import pass_to_ranks

        # By maximum edge weight
        if self.norm == 1:
            self.in_mat = thresholding.normalize(self.in_mat)
        # Apply log10
        elif self.norm == 2:
            self.in_mat = np.log10(self.in_mat)
        # Apply PTR simple-nonzero
        elif self.norm == 3:
            self.in_mat = pass_to_ranks(self.in_mat, method="simple-nonzero")
        # Apply PTR simple-all
        elif self.norm == 4:
            self.in_mat = pass_to_ranks(self.in_mat, method="simple-all")
        # Apply PTR zero-boost
        elif self.norm == 5:
            self.in_mat = pass_to_ranks(self.in_mat, method="zero-boost")
        # Apply standardization [0, 1]
        elif self.norm == 6:
            self.in_mat = thresholding.standardize(self.in_mat)
        else:
            pass

        self.G = nx.from_numpy_matrix(self.in_mat)

        return self.G

    def prune_graph(self):
        from pynets.core import utils
        # Load numpy matrix as networkx graph
        self.G = nx.from_numpy_matrix(self.in_mat)

        # Prune irrelevant nodes (i.e. nodes who are fully disconnected from the graph and/or those whose betweenness
        # centrality are > 3 standard deviations below the mean)
        if (self.prune == 1) or (nx.is_connected(self.G) is True):
            if nx.is_connected(self.G) is False:
                print('Warning: Graph is fragmented...\n')
            [self.G, _] = prune_disconnected(self.G)
        elif self.prune == 2:
            print('Pruning to retain only most important nodes...')
            [self.G, _] = most_important(self.G)
        else:
            print('Graph is connected...')

        # Get corresponding matrix
        self.in_mat = np.array(nx.to_numpy_matrix(self.G))

        # Saved pruned
        if (self.prune != 0) and (self.prune is not None):
            final_mat_path = "%s%s" % (self.est_path.split('.npy')[0], '_pruned_mat')
            utils.save_mat(self.in_mat, final_mat_path, self.out_fmt)
            print("%s%s" % ('Source File: ', final_mat_path))
        else:
            print("%s%s" % ('Source File: ', self.est_path))
        return self.in_mat, final_mat_path

    def print_summary(self):
        print("%s%.2f%s" % ('\n\nThreshold: ', 100 * float(self.thr), '%'))

        info_list = list(nx.info(self.G).split('\n'))[2:]
        for i in info_list:
            print(i)
        return

    def binarize_graph(self):
        from pynets.core import thresholding
        in_mat_bin = thresholding.binarize(self.in_mat)

        # Load numpy matrix as networkx graph
        G_bin = nx.from_numpy_matrix(in_mat_bin)
        return in_mat_bin, G_bin

    def create_length_matrix(self):
        in_mat_len = thresholding.weight_conversion(self.in_mat, 'lengths')

        # Load numpy matrix as networkx graph
        G_len = nx.from_numpy_matrix(in_mat_len)
        return in_mat_len, G_len


def save_netmets(dir_path, est_path, metric_list_names, net_met_val_list_final):
    from pynets.core import utils

    # And save results to csv
    out_path_neat = "%s%s" % (utils.create_csv_path(dir_path, est_path).split('.csv')[0], '_neat.csv')
    zipped_dict = dict(zip(metric_list_names, net_met_val_list_final))
    df = pd.DataFrame.from_dict(zipped_dict, orient='index', dtype='float32').transpose()
    df.to_csv(out_path_neat, index=False)
    del df, zipped_dict, net_met_val_list_final, metric_list_names

    return out_path_neat


def iterate_nx_global_measures(G, metric_list_glob):
    # import random
    num_mets = len(metric_list_glob)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j = 0
    for i in metric_list_glob:
        net_met = str(i).split('<function ')[1].split(' at')[0]
        try:
            try:
                net_met_val = raw_mets(G, i)
            except:
                print("%s%s%s" % ('WARNING: ', net_met, ' failed for graph G.'))
                # np.save("%s%s%s%s" % ('/tmp/', net_met, random.randint(1, 400), '.npy'),
                #         np.array(nx.to_numpy_matrix(G)))
                net_met_val = np.nan
        except:
            print("%s%s%s" % ('WARNING: ', str(i), ' is undefined for graph G'))
            # np.save("%s%s%s%s" % ('/tmp/', net_met, random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
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
    for i in net_met_arr[:, 0]:
        metric_list_names.append(i)
    return net_met_val_list, metric_list_names


def community_resolution_selection(G):
    import community
    resolution = 1
    ci = np.array(list(community.best_partition(G, resolution=resolution).values()))
    num_comms = len(np.unique(ci))
    if num_comms == 1:
        resolution = 10
        tries = 0
        while num_comms == 1:
            ci = np.array(list(community.best_partition(G, resolution=resolution).values()))
            num_comms = len(np.unique(ci))
            print("%s%s%s%s%s" % ('Found ', num_comms, ' communities at resolution: ', resolution, '...'))
            resolution = resolution + 10
            tries = tries + 1
            if tries > 100:
                print('\nWARNING: Louvain community detection failed. Proceeding with single community affiliation '
                      'vector...')
                break
    elif num_comms > len(G.edges()) / 10:
        resolution = 0.1
        tries = 0
        while num_comms == 1:
            ci = np.array(list(community.best_partition(G, resolution=resolution).values()))
            num_comms = len(np.unique(ci))
            print("%s%s%s%s%s" % ('Found ', num_comms, ' communities at resolution: ', resolution, '...'))
            resolution = resolution / 10
            tries = tries + 1
            if tries > 100:
                print('\nWARNING: Louvain community detection failed. Proceeding with single community affiliation '
                      'vector...')
                break
    else:
        print("%s%s%s%s%s" % ('Found ', num_comms, ' communities at resolution: ', resolution, '...'))
    return dict(zip(G.nodes(), ci)), ci, resolution, num_comms


def get_community(G, net_met_val_list_final, metric_list_names):
    import community
    ci_dict, ci, resolution, num_comms = community_resolution_selection(G)
    modularity = community.community_louvain.modularity(ci_dict, G)
    metric_list_names.append('modularity')
    if modularity == 1.0:
        modularity = np.nan
        print('Louvain modularity calculation is undefined for graph G')
    net_met_val_list_final.append(modularity)
    return net_met_val_list_final, metric_list_names, ci


def get_participation(in_mat, ci, metric_list_names, net_met_val_list_final):
    if len(in_mat[in_mat < 0.0]) > 0:
        pc_vector = participation_coef_sign(in_mat, ci)[0]
    else:
        pc_vector = participation_coef(in_mat, ci)
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
    return metric_list_names, net_met_val_list_final


def get_diversity(in_mat, ci, metric_list_names, net_met_val_list_final):
    dc_vector = diversity_coef_sign(in_mat, ci)[0]
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
    return metric_list_names, net_met_val_list_final


def get_local_efficiency(G, metric_list_names, net_met_val_list_final):
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
    return metric_list_names, net_met_val_list_final


def get_clustering(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import clustering

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
    return metric_list_names, net_met_val_list_final


def get_degree_centrality(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import degree_centrality
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
    return metric_list_names, net_met_val_list_final


def get_betweenness_centrality(G_len, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import betweenness_centrality
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
    return metric_list_names, net_met_val_list_final


def get_eigen_centrality(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import eigenvector_centrality
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
    return metric_list_names, net_met_val_list_final


def get_comm_centrality(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import communicability_betweenness_centrality
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
    return metric_list_names, net_met_val_list_final


@timeout(360)
def get_rich_club_coeff(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import rich_club_coefficient
    rc_vector = rich_club_coefficient(G, normalized=True, seed=42, Q=100)
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
    return metric_list_names, net_met_val_list_final


def extractnetstats(ID, network, thr, conn_model, est_path, roi, prune, norm, binary):
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

    Returns
    -------
    out_path : str
        Path to .csv file where graph analysis results are saved.
    """
    import os.path as op
    import yaml
#    import random
    import networkx
    import pynets.stats.netstats
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pathlib import Path

    cg = CleanGraphs(thr, conn_model, est_path, prune, norm)
    if float(norm) >= 1:
        cg.normalize_graph()

    if float(prune) >= 1:
        cg.prune_graph()

    if binary is True:
        in_mat, G = cg.binarize_graph()
    else:
        in_mat, G = cg.in_mat, cg.G

    in_mat_len, G_len = cg.create_length_matrix()

    cg.print_summary()

    dir_path = op.dirname(op.realpath(est_path))

    # Load netstats config and parse graph algorithms as objects
    with open("%s%s" % (str(Path(__file__).parent), '/global_graph_measures.yaml'), 'r') as stream:
        try:
            nx_algs = ['degree_assortativity_coefficient', 'average_clustering', 'average_shortest_path_length',
                       'graph_number_of_cliques']
            pynets_algs = ['average_local_efficiency', 'global_efficiency', 'smallworldness', 'weighted_transitivity']
            metric_dict_global = yaml.load(stream)
            metric_list_global = metric_dict_global['metric_list_global']
            metric_list_global = [getattr(networkx.algorithms, i) for i in
                                  metric_list_global if i in
                                  nx_algs] + [getattr(pynets.stats.netstats, i)
                                              for i in metric_list_global if i in pynets_algs]
            metric_list_global_names = [str(i).split('<function ')[1].split(' at')[0] for i in metric_list_global]
            if binary is False:
                from functools import partial
                metric_list_global = [partial(i, weight='weight') if 'weight' in i.__code__.co_varnames else i for i in
                                      metric_list_global]
            print("%s%s%s" % ('\n\nCalculating global measures:\n',
                              metric_list_global_names, '\n\n'))
        except FileNotFoundError:
            print('Failed to parse global_graph_measures.yaml')

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
    net_met_val_list_final, metric_list_names = iterate_nx_global_measures(G, metric_list_global)

    # Run miscellaneous functions that generate multiple outputs
    # Calculate modularity using the Louvain algorithm
    if 'louvain_modularity' in metric_list_nodal:
        try:
            net_met_val_list_final, metric_list_names, ci = get_community(G, net_met_val_list_final, metric_list_names)
        except:
            print('Louvain modularity calculation is undefined for graph G')
            # np.save("%s%s%s" % ('/tmp/community_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Participation Coefficient by louvain community
    if 'participation_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Participation coefficient cannot be calculated for graph G in the absence of a '
                               'community affiliation vector')
            metric_list_names, net_met_val_list_final = get_participation(in_mat, ci, metric_list_names,
                                                                          net_met_val_list_final)
        except:
            print('Participation coefficient cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/partic_coeff_failure', random.randint(1, 400), '.npy'), in_mat)
            pass

    # Diversity Coefficient by louvain community
    if 'diversity_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Diversity coefficient cannot be calculated for graph G in the absence of a community '
                               'affiliation vector')
            metric_list_names, net_met_val_list_final = get_diversity(in_mat, ci, metric_list_names,
                                                                      net_met_val_list_final)
        except:
            print('Diversity coefficient cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/div_coeff_failure', random.randint(1, 400), '.npy'), in_mat)
            pass

    # Local Efficiency
    if 'local_efficiency' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_local_efficiency(G, metric_list_names,
                                                                             net_met_val_list_final)
        except:
            print('Local efficiency cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/local_eff_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Local Clustering
    if 'local_clustering' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_clustering(G, metric_list_names, net_met_val_list_final)
        except:
            print('Local clustering cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/local_clust_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Degree centrality
    if 'degree_centrality' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_degree_centrality(G, metric_list_names,
                                                                              net_met_val_list_final)
        except:
            print('Degree centrality cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/degree_cent_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Betweenness Centrality
    if 'betweenness_centrality' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_betweenness_centrality(G_len, metric_list_names,
                                                                                   net_met_val_list_final)
        except:
            print('Betweenness centrality cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/betw_cent_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G_len)))
            pass

    # Eigenvector Centrality
    if 'eigenvector_centrality' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_eigen_centrality(G, metric_list_names,
                                                                             net_met_val_list_final)
        except:
            print('Eigenvector centrality cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/eig_cent_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Communicability Centrality
    if 'communicability_centrality' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_comm_centrality(G, metric_list_names,
                                                                            net_met_val_list_final)
        except:
            print('Communicability centrality cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/comm_cent_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    # Rich club coefficient
    if 'rich_club_coefficient' in metric_list_nodal:
        try:
            metric_list_names, net_met_val_list_final = get_rich_club_coeff(G, metric_list_names,
                                                                            net_met_val_list_final)
        except:
            print('Rich club coefficient cannot be calculated for graph G')
            # np.save("%s%s%s" % ('/tmp/rich_club_failure', random.randint(1, 400), '.npy'),
            #         np.array(nx.to_numpy_matrix(G)))
            pass

    out_path_neat = save_netmets(dir_path, est_path, metric_list_names, net_met_val_list_final)

    # Cleanup
    del net_met_val_list_final, metric_list_names, metric_list_global

    return out_path_neat


def collect_pandas_df_make(net_mets_csv_list, ID, network, plot_switch, nc_collect=False, create_summary=True,
                           sql_out=True):
    """
    Summarize list of pickled pandas dataframes of graph metrics unique to eacho unique combination of hyperparameters.

    Parameters
    ----------
    net_mets_csv_list : list
        List of file paths to pickled pandas dataframes as themselves.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    plot_switch : bool
        Activate summary plotting (histograms, central tendency, AUC, etc.)
    sql_out : bool
        Optionally output data to sql.

    Returns
    -------
    combination_complete : bool
        If True, then data integration completed successfully.
    """
    import os
    import os.path as op
    import pandas as pd
    from pynets.core import utils
    from itertools import groupby
    import re

    # Check for existence of net_mets csv files, condensing final list to only those that were actually produced.
    net_mets_csv_list_exist = []
    for net_mets_csv in list(net_mets_csv_list):
        if op.isfile(net_mets_csv) is True:
            net_mets_csv_list_exist.append(net_mets_csv)

    if len(list(net_mets_csv_list)) > len(net_mets_csv_list_exist):
        raise UserWarning('Warning! Number of actual models produced less than expected. Some graphs were excluded')

    net_mets_csv_list = net_mets_csv_list_exist
    subject_path = op.dirname(op.dirname(op.dirname(net_mets_csv_list[0])))

    if len(net_mets_csv_list) > 1:
        print("%s%s%s" % ('\n\nList of result files to concatenate:\n', str(net_mets_csv_list), '\n\n'))

        models = []
        for file_ in net_mets_csv_list:
            models.append(op.basename(op.dirname(op.dirname(file_))) + '/netmetrics/' + op.basename(file_))

        def sort_thr(model_name):
            return model_name.split('thr-')[1].split('_')[0]

        models.sort(key=sort_thr)

        # Group by secondary attributes
        models_grouped = [list(x) for x in zip(*[list(g) for k, g in
                                                 groupby(models, lambda s: s.split('thr-')[1].split('_')[0])])]

        hyperparam_dict = {}
        hyperparam_dict['id'] = ID
        gen_hyperparams = ['node_type', 'atlas', 'thrtype']
        if max([len(i) for i in models_grouped]) > 1:
            print('Multiple thresholds detected. Computing Area Under the Curve (AUC)...')
            meta = dict()
            non_decimal = re.compile(r'[^\d.]+')
            for thr_set in range(len(models_grouped)):
                meta[thr_set] = dict()
                meta[thr_set]['dataframes'] = dict()
                for i in models_grouped[thr_set]:
                    thr = non_decimal.sub('', i.split('thr-')[1].split('_')[0])
                    _file = subject_path + '/' + i
                    df = pd.read_csv(_file)
                    if nc_collect is False:
                        node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]
                        df = df.drop(node_cols, axis=1)
                    meta[thr_set]['dataframes'][thr] = df

            # For each unique threshold set, for each graph measure, extract AUC
            if sql_out is True:
                try:
                    import sqlalchemy
                    sql_db = utils.build_sql_db(op.dirname(op.dirname(op.dirname(subject_path))), ID)
                except:
                    sql_out = False
            for thr_set in meta.keys():
                df_summary = pd.concat(meta[thr_set]['dataframes'].values())
                df_summary['thr'] = meta[thr_set]['dataframes'].keys()
                meta[thr_set]['summary_dataframe'] = df_summary
                df_summary_auc = df_summary.iloc[[0]]
                df_summary_auc.columns = [col + '_auc' for col in df_summary.columns]

                print("%s%s" % ('\nAUC for threshold group: ', models_grouped[thr_set]))
                file_renamed = list(set([re.sub(r'thr\-\d+\.*\d+', '',
                                                i.split('/netmetrics/')[1]).replace('neat', 'auc') for i in
                                         models_grouped[thr_set]]))[0]
                atlas = models_grouped[thr_set][0].split('/')[0]
                modality = file_renamed.split('modality-')[1].split('_')[0]

                # Build hyperparameter dictionary
                hyperparam_dict, hyperparams = utils.build_hp_dict(file_renamed, atlas, modality, hyperparam_dict,
                                                                   gen_hyperparams)

                for measure in df_summary.columns[:-1]:
                    # Get Area Under the Curve
                    df_summary_nonan = df_summary[pd.notnull(df_summary[measure])]
                    df_summary_auc[measure] = np.trapz(np.array(df_summary_nonan[measure]).astype('float32'),
                                                       np.array(df_summary_nonan['thr']).astype('float32'))
                    print("%s%s%s" % (measure, ': ', df_summary_auc[measure].to_string(index=False)))
                meta[thr_set]['auc_dataframe'] = df_summary_auc
                auc_dir = subject_path + '/' + atlas + '/netmetrics/auc/'
                if not os.path.isdir(auc_dir):
                    os.makedirs(auc_dir, exist_ok=True)
                df_summary_auc = df_summary_auc.drop(columns=['thr_auc'])
                df_summary_auc = df_summary_auc.loc[:, df_summary_auc.columns.str.endswith('auc')]
                auc_outfile = auc_dir + file_renamed
                df_summary_auc.to_csv(auc_outfile, header=True, index=False, chunksize=100000, compression='gzip',
                                      encoding='utf-8')
                if sql_out is True:
                    sql_db.create_modality_table(modality)
                    sql_db.add_hp_columns(list(set(hyperparams)) + list(df_summary_auc.columns))
                    sql_db.add_row_from_df(df_summary_auc, hyperparam_dict)
                    # sql_db.engine.execute("SELECT * FROM func").fetchall()

        if create_summary is True:
            try:
                summary_dir = subject_path + '/summary'
                if not os.path.isdir(summary_dir):
                    os.makedirs(summary_dir, exist_ok=True)

                # Concatenate and find mean across dataframes
                print('Concatenating frames...')
                df_concat = pd.concat([meta[thr_set]['auc_dataframe'] for thr_set in meta.keys()])
                measures = list(df_concat.columns)
                if plot_switch is True:
                    from pynets.plotting import plot_gen
                    plot_gen.plot_graph_measure_hists(df_concat, measures, file_)
                df_concatted_mean = df_concat.loc[:, measures].mean(skipna=True).to_frame().transpose()
                df_concatted_median = df_concat.loc[:, measures].median(skipna=True).to_frame().transpose()
                df_concatted_mode = pd.DataFrame(df_concat.loc[:, measures].mode(axis=0, dropna=True).max()).transpose()
                df_concatted_mean.columns = [str(col) + '_mean' for col in df_concatted_mean.columns]
                df_concatted_median.columns = [str(col) + '_median' for col in df_concatted_median.columns]
                df_concatted_mode.columns = [str(col) + '_maxmode' for col in df_concatted_mode.columns]
                result = pd.concat([df_concatted_mean, df_concatted_median, df_concatted_mode], axis=1)
                df_concatted_final = result.reindex(sorted(result.columns), axis=1)
                print('\nConcatenating dataframes for ' + str(ID) + '...\n')
                net_csv_summary_out_path = "%s%s%s%s%s%s" % (summary_dir, '/', str(ID), '_net_mets',
                                                             '%s' % ('_' + network if network is not None else ''),
                                                             '_mean.csv')
                df_concatted_final.to_csv(net_csv_summary_out_path, index=False)
                combination_complete = True
            except RuntimeWarning:
                combination_complete = False
                print("%s%s%s" % ('\nWARNING: DATAFRAME CONCATENATION FAILED FOR ', str(ID), '!\n'))
                pass
        else:
            combination_complete = True
    else:
        if network is not None:
            print("%s%s%s%s%s" % ('\nSingle dataframe for the ', network, ' network for subject ', ID, '\n'))
        else:
            print("%s%s%s" % ('\nSingle dataframe for subject ', ID, '\n'))
        combination_complete = True

    return combination_complete
