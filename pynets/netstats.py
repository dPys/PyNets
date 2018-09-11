# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
from __future__ import division
import os
import numpy as np
import networkx as nx


def average_shortest_path_length_for_all(G):
    import math
    subgraphs = [sbg for sbg in nx.connected_component_subgraphs(G) if len(sbg) > 1]
    return math.fsum(nx.average_shortest_path_length(sg) for sg in subgraphs) / len(subgraphs)


def global_efficiency(G, weight=None):
    """Return the global efficiency of the graph G

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

        inv = [1/x for x in lengths.values() if x is not 0]
        inv_lengths.extend(inv)

    return sum(inv_lengths)/(N*(N-1))


def local_efficiency(G, weight=None):
    """Return the local efficiency of each node in the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    local_efficiency : dict
       the keys of the dict are the nodes in the graph G and the corresponding
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


def average_local_efficiency(G, weight=None):
    """Return the average local efficiency of all of the nodes in the graph G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    average_local_efficiency : float

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
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
       small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
       in weighted networks. Eur Phys J B 32, 249-263.

    """
    eff = local_efficiency(G, weight)
    total = sum(eff.values())
    N = len(eff)
    return total/N

def create_random_graph(G, n, p):
    rG = nx.erdos_renyi_graph(n, p, seed=42)
    return rG

def smallworldness_measure(G, rG):
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
    #def small_iters(bb):
    #    rG = create_random_graph(G, n, p)
    #    swm = smallworldness_measure(G, rG)
    #    return swm
    #number_processes = int(multiprocessing.cpu_count()-1)
    #pool = multiprocessing.Pool(number_processes)
    #bb = range(rep)
    #result = pool.map_async(small_iters, bb)
    #pool.close()
    #pool.join()
    #ss = result.get()
    mean_s = np.mean(ss)
    return mean_s


def create_communities(node_comm_aff_mat, node_num):
    com_assign = np.zeros((node_num,1))
    for i in range(len(node_comm_aff_mat)):
        community = node_comm_aff_mat[i,:]
        for j in range(len(community)):
            if community[j] == 1:
                com_assign[j,0]=i
    return com_assign


def _compute_rc(G):
    from networkx.utils import accumulate
    deghist = nx.degree_histogram(G)
    total = sum(deghist)
    # Compute the number of nodes with degree greater than `k`, for each
    # degree `k` (omitting the last entry, which is zero).
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    # Create a sorted list of pairs of edge endpoint degrees.
    # The list is sorted in reverse order so that we can pop from the
    # right side of the list later, instead of popping from the left
    # side of the list, which would have a linear time cost.
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()), reverse=True)
    ek = G.number_of_edges()
    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc


def participation_coef(W, ci, degree='undirected'):
    ## ADAPTED FROM BCTPY ##
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
        participation coefficient
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


def modularity(W, qtype='sta', seed=42):
    ## ADAPTED FROM BCTPY ##
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
        raise KeyError('Modularity type unknown')

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
                dQ0 = (knm0[u, :] + W0[u, u] - knm0[u, ma]) - kn0[u] * (km0 + kn0[u] - km0[ma]) / s0
                dQ1 = (knm1[u, :] + W1[u, u] - knm1[u, ma]) - kn1[u] * (km1 + kn1[u] - km1[ma]) / s1
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


def diversity_coef_sign(W, ci):
    ## ADAPTED FROM BCTPY ##
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
    '''
    # Number of nodes
    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    # Number of modules
    m = np.max(ci)
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

    # Explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


def link_communities(W, type_clustering='single'):
    from pynets.thresholding import normalize
    ## ADAPTED FROM BCTPY ##
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
    '''
    n = len(W)
    W = normalize(W)

    if type_clustering not in ('single', 'complete'):
        print('Error: Unrecognized clustering type')

    # Set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[range(n), range(n)] = (np.sum(W, axis=0) / np.sum(np.logical_not(W), axis=0) + np.sum(W.T, axis=0) / np.sum(np.logical_not(W.T), axis=0)) / 2

    # Out/in norm squared
    No = np.sum(W**2, axis=1)
    Ni = np.sum(W**2, axis=0)

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
        #time1 = time.time()
        # Loop over communities
        for j in range(len(U)):
            # Get link indices
            ixes = C[i, :] == U[j]
            links = np.sort(Lw[ixes])
            #nodes = np.sort(Ln[ixes,:].flat)
            nodes = np.sort(np.reshape(
                Ln[ixes, :], 2 * np.size(np.where(ixes))))
            # Get unique nodes
            nodulo = np.append(nodes[0], (nodes[1:])[nodes[1:] != nodes[:-1]])
            #nodulo = ((nodes[1:])[nodes[1:] != nodes[:-1]])
            nc = len(nodulo)
            #nc = len(nodulo)+1
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
        #time2 = time.time()
        #print('compute densities time', time2-time1)
        # Copy current partition
        C[i + 1, :] = C[i, :]
        #if i in (2693,):
        #    import pdb
        #    pdb.set_trace()
        u1, u2 = np.where(ES[np.ix_(U, U)] == np.max(ES[np.ix_(U, U)]))
        if np.size(u1) > 2:
            wehr, = np.where((u1 == u2[0]))
            uc = np.squeeze((u1[0], u2[0]))
            ud = np.squeeze((u1[wehr], u2[wehr]))
            u1 = uc
            u2 = ud

        #time25 = time.time()
        #print('copy and max time', time25-time2)
        #ugl = np.array((u1,u2))
        ugl = np.sort((u1, u2), axis=1)
        ug_rows = ugl[np.argsort(ugl, axis=0)[:, 0]]
        # implementation of matlab unique(A, 'rows')
        unq_rows = np.vstack({tuple(row) for row in ug_rows})
        V = U[unq_rows]
        #time3 = time.time()
        #print('sortrows time', time3-time25)

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

        #time4 = time.time()
        #print('get linkages time', time4-time3)
        U = np.unique(C[i + 1, :])
        if len(U) == 1:
            break
        #time5 = time.time()
        #print('get unique communities time', time5-time4)
    #Dc[ np.where(np.isnan(Dc)) ]=0
    i = np.argmax(np.sum(Dc * Mc, axis=1))
    U = np.unique(C[i, :])
    M = np.zeros((len(U), n))
    for j in range(len(U)):
        M[j, np.unique(Ln[C[i, :] == U[j], :])] = 1

    M = M[np.sum(M, axis=1) > 2, :]
    return M


def modularity_louvain_und_sign(W, gamma=1, qtype='sta', seed=42):
    ## ADAPTED FROM BCTPY ##
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    The Louvain algorithm is a fast and accurate community detection
    algorithm (at the time of writing).
    Use this function as opposed to modularity_louvain_und() only if the
    network contains a mix of positive and negative weights.  If the network
    contains all positive weights, the output will be equivalent to that of
    modularity_louvain_und().
    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector
    Q : float
        optimized modularity metric
    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # weight of positive links
    s1 = np.sum(W1)  # weight of negative links

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-sQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('Modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    h = 1  # hierarchy index
    nh = n  # number of nodes in hierarchy
    ci = [None, np.arange(n) + 1]  # hierarchical module assignments
    q = [-1, 0]  # hierarchical modularity values
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise ValueError('Modularity Infinite Loop')
        kn0 = np.sum(W0, axis=0)  # positive node degree
        kn1 = np.sum(W1, axis=0)  # negative node degree
        km0 = kn0.copy()  # positive module degree
        km1 = kn1.copy()  # negative module degree
        knm0 = W0.copy()  # positive node-to-module degree
        knm1 = W1.copy()  # negative node-to-module degree

        m = np.arange(nh) + 1  # initial module assignments
        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise ValueError('Infinite Loop was detected and stopped')
            flag = False
            # loop over nodes in random order
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                dQ0 = ((knm0[u, :] + W0[u, u] - knm0[u, ma]) -
                       gamma * kn0[u] * (km0 + kn0[u] - km0[ma]) / s0)  # positive dQ
                dQ1 = ((knm1[u, :] + W1[u, u] - knm1[u, ma]) -
                       gamma * kn1[u] * (km1 + kn1[u] - km1[ma]) / s1)  # negative dQ

                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ[ma] = 0  # no changes for same module

                max_dQ = np.max(dQ)  # maximal increase in modularity
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    mb = np.argmax(dQ)

                    # change positive node-to-module degrees
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]  # change positive module degrees
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]  # change negative module degrees
                    km1[ma] -= kn1[u]

                    m[u] = mb + 1  # reassign module

        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1

        for u in range(nh):  # loop through initial module assignments
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]  # assign new modules

        nh = np.max(m)  # number of new nodes
        wn0 = np.zeros((nh, nh))  # new positive weights matrix
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
        # compute modularity
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1

    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1

    return ci_ret, q[-1]


def prune_disconnected(G):
    """ returns a copy of G with
        isolates pruned """
    print('Pruning fully disconnected...')

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
     """ returns a copy of G with
         isolates and low-importance nodes pruned """
     print('Pruning fully disconnected and low importance nodes (3 SD < M)...')
     ranking = nx.betweenness_centrality(G).items()
     #print(ranking)
     r = [x[1] for x in ranking]
     m = sum(r)/len(r) - 3*np.std(r)
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


# Extract network metrics interface
def extractnetstats(ID, network, thr, conn_model, est_path, mask, prune, node_size, smooth):
    import pandas as pd
    import yaml
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pathlib import Path
    from pynets import thresholding, utils

    # Advanced options
    save_gephi = False
    custom_weight = None
    binary = False

    # Load and threshold matrix
    if '.txt' in est_path:
        in_mat_raw = np.array(np.genfromtxt(est_path))
    else:
        in_mat_raw = np.array(np.load(est_path))

    # De-diagnal and Normalize connectivity matrix (weights between 0-1)
    in_mat = np.array(thresholding.normalize(np.array(thresholding.autofix(in_mat_raw))))

    # Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix if non-covariance
    if conn_model == 'corr':
        in_mat = np.arctanh(in_mat)
        in_mat[np.isnan(in_mat)] = 0
        in_mat[np.isinf(in_mat)] = 1

    # Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path))

    # Load numpy matrix as networkx graph
    G_pre = nx.from_numpy_matrix(in_mat)

    # Prune irrelevant nodes (i.e. nodes who are fully disconnected from the graph and/or those whose betweenness centrality are > 3 standard deviations below the mean)
    if prune == 1:
        [G, _] = prune_disconnected(G_pre)
    elif prune == 2:
        [G, _] = most_important(G_pre)
    else:
        G = G_pre

    # Get corresponding matrix
    in_mat = np.array(nx.to_numpy_matrix(G))

    # Binarize graph
    if binary is True:
        in_mat = thresholding.binarize(in_mat)

    # Print graph summary
    print("%s%.2f%s" % ('\n\nThreshold: ', 100*float(thr), '%'))
    print("%s%s" % ('Source File: ', est_path))
    info_list = list(nx.info(G).split('\n'))[2:]
    for i in info_list:
        print(i)

    # try:
    #     G_dir
    #     print('Analyzing DIRECTED graph when applicable...')
    # except:
    #     print('Graph is UNDIRECTED')

    if nx.is_connected(G) is True:
        frag = False
        print('Graph is connected...')
    else:
        frag = True
        print('Warning: Graph is fragmented...\n')

    # Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')
    # Load numpy matrix as networkx graph
    G_len = nx.from_numpy_matrix(mat_len)

    if save_gephi is True:
        # Save G as gephi file
        if mask:
            if network:
                nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_', os.path.basename(mask).split('.')[0], '_', thr, '_', node_size, '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm.graphml') if float(smooth) > 0 else 'nosm.graphml')))
            else:
                nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_', thr, '_', node_size, '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm.graphml') if float(smooth) > 0 else 'nosm.graphml')))
        else:
            if network:
                nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_', thr, '_', node_size, '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm.graphml') if float(smooth) > 0 else 'nosm.graphml')))
            else:
                nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', thr, '_', node_size, '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm.graphml') if float(smooth) > 0 else 'nosm.graphml')))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # Calculate global and local metrics from graph G # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, eigenvector_centrality, communicability_betweenness_centrality, clustering, degree_centrality, rich_club_coefficient
    from pynets.netstats import average_local_efficiency, global_efficiency, local_efficiency, modularity_louvain_und_sign, smallworldness, participation_coef, diversity_coef_sign
    # For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the function  (with a G-only input) to the netstats module.
    metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]
    #metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, smallworldness]
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
            if i is 'average_shortest_path_length':
                if nx.is_connected(G) is True:
                    # try:
                    #     net_met_val = float(i(G_dir))
                    #     print('Calculating from directed graph...')
                    # except:
                    #     net_met_val = float(i(G))
                    net_met_val = float(i(G))
                else:
                    # Case where G is not fully connected
                    print('WARNING: Calculating average shortest path length for a disconnected graph. This might take awhile...')
                    net_met_val = float(average_shortest_path_length_for_all(G))
            if custom_weight is not None and i is 'degree_assortativity_coefficient' or i is 'global_efficiency' or i is 'average_local_efficiency' or i is 'average_clustering':
                custom_weight_param = 'weight = ' + str(custom_weight)
                # try:
                #     net_met_val = float(i(G_dir, custom_weight_param))
                #     print('Calculating from directed graph...')
                # except:
                #     net_met_val = float(i(G, custom_weight_param))
                net_met_val = float(i(G, custom_weight_param))
            else:
                # try:
                #     net_met_val = float(i(G_dir))
                #     print('Calculating from directed graph...')
                # except:
                #     net_met_val = float(i(G))
                net_met_val = float(i(G))
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
            gamma = nx.density(nx.from_numpy_array(in_mat))
            [ci, modularity] = modularity_louvain_und_sign(in_mat, gamma=gamma)
            metric_list_names.append('modularity')
            net_met_val_list_final.append(modularity)
        except:
            print('Louvain modularity calculation is undefined for graph G')
            pass

    # Participation Coefficient by louvain community
    if 'participation_coefficient' in metric_list_nodal:
        try:
            if ci is None:
                raise KeyError('Participation coefficient cannot be calculated for graph G in the absence of a community affiliation vector')
            pc_vector = participation_coef(in_mat, ci)
            print('\nExtracting Participation Coefficient vector for all network nodes...')
            pc_vals = list(pc_vector)
            pc_edges = list(range(len(pc_vector)))
            num_edges = len(pc_edges)
            pc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                pc_arr[j, 0] = "%s%s" % (str(pc_edges[j]), '_partic_coef')
                #print('\n' + str(rc_edges[j]) + '_rich_club')
                try:
                    pc_arr[j, 1] = pc_vals[j]
                except:
                    print("%s%s%s" % ('Participation coefficient is undefined for node ', str(j), ' of graph G'))
                    pc_arr[j, 1] = np.nan
                #print(str(rc_vals[j]))
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
                raise KeyError('Diversity coefficient cannot be calculated for graph G in the absence of a community affiliation vector')
            [dc_vector, _] = diversity_coef_sign(in_mat, ci)
            print('\nExtracting Diversity Coefficient vector for all network nodes...')
            dc_vals = list(dc_vector)
            dc_edges = list(range(len(dc_vector)))
            num_edges = len(dc_edges)
            dc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j = 0
            for i in range(num_edges):
                dc_arr[j, 0] = "%s%s" % (str(dc_edges[j]), '_diversity_coef')
                #print('\n' + str(rc_edges[j]) + '_rich_club')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Diversity coefficient is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                #print(str(rc_vals[j]))
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
            # try:
            #     le_vector = local_efficiency(G_dir)
            # except:
            #     le_vector = local_efficiency(G)
            le_vector = local_efficiency(G)
            print('\nExtracting Local Efficiency vector for all network nodes...')
            le_vals = list(le_vector.values())
            le_nodes = list(le_vector.keys())
            num_nodes = len(le_nodes)
            le_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                le_arr[j, 0] = "%s%s" % (str(le_nodes[j]), '_local_efficiency')
                #print('\n' + str(le_nodes[j]) + '_local_efficiency')
                try:
                    le_arr[j, 1] = le_vals[j]
                except:
                    print("%s%s%s" % ('Local efficiency is undefined for node ', str(j), ' of graph G'))
                    le_arr[j, 1] = np.nan
                #print(str(le_vals[j]))
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
                #print('\n' + str(cl_nodes[j]) + '_local_clustering')
                try:
                    cl_arr[j, 1] = cl_vals[j]
                except:
                    print("%s%s%s" % ('Local clustering is undefined for node ', str(j), ' of graph G'))
                    cl_arr[j, 1] = np.nan
                #print(str(cl_vals[j]))
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
            # try:
            #     dc_vector = degree_centrality(G_dir)
            # except:
            #     dc_vector = degree_centrality(G)
            dc_vector = degree_centrality(G)
            print('\nExtracting Degree Centrality vector for all network nodes...')
            dc_vals = list(dc_vector.values())
            dc_nodes = list(dc_vector.keys())
            num_nodes = len(dc_nodes)
            dc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                dc_arr[j, 0] = "%s%s" % (str(dc_nodes[j]), '_degree_centrality')
                #print('\n' + str(dc_nodes[j]) + '_degree_centrality')
                try:
                    dc_arr[j, 1] = dc_vals[j]
                except:
                    print("%s%s%s" % ('Degree centrality is undefined for node ', str(j), ' of graph G'))
                    dc_arr[j, 1] = np.nan
                #print(str(cl_vals[j]))
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
                #print('\n' + str(bc_nodes[j]) + '_betw_cent')
                try:
                    bc_arr[j, 1] = bc_vals[j]
                except:
                    print("%s%s%s" % ('Betweeness centrality is undefined for node ', str(j), ' of graph G'))
                    bc_arr[j, 1] = np.nan
                #print(str(bc_vals[j]))
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
            # try:
            #     ec_vector = eigenvector_centrality(G_dir, max_iter=1000)
            # except:
            #     ec_vector = eigenvector_centrality(G, max_iter=1000)
            ec_vector = eigenvector_centrality(G, max_iter=1000)
            print('\nExtracting Eigenvector Centrality vector for all network nodes...')
            ec_vals = list(ec_vector.values())
            ec_nodes = list(ec_vector.keys())
            num_nodes = len(ec_nodes)
            ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j = 0
            for i in range(num_nodes):
                ec_arr[j, 0] = "%s%s" % (str(ec_nodes[j]), '_eigenvector_centrality')
                #print('\n' + str(ec_nodes[j]) + '_eig_cent')
                try:
                    ec_arr[j, 1] = ec_vals[j]
                except:
                    print("%s%s%s" % ('Eigenvector centrality is undefined for node ', str(j), ' of graph G'))
                    ec_arr[j, 1] = np.nan
                #print(str(ec_vals[j]))
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
                #print('\n' + str(cc_nodes[j]) + '_comm_cent')
                try:
                    cc_arr[j, 1] = cc_vals[j]
                except:
                    print("%s%s%s" % ('Communicability centrality is undefined for node ', str(j), ' of graph G'))
                    cc_arr[j, 1] = np.nan
                #print(str(cc_vals[j]))
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
                #print('\n' + str(rc_edges[j]) + '_rich_club')
                try:
                    rc_arr[j, 1] = rc_vals[j]
                except:
                    print("%s%s%s" % ('Rich club coefficient is undefined for node ', str(j), ' of graph G'))
                    rc_arr[j, 1] = np.nan
                #print(str(rc_vals[j]))
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

    if mask:
        met_list_picke_path = "%s%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list', "%s" % ("%s%s%s" % ('_', network, '_') if network else "_"), os.path.basename(mask).split('.')[0])
    else:
        if network:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list_', network)
        else:
            met_list_picke_path = "%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list')
    pickle.dump(metric_list_names, open(met_list_picke_path, 'wb'), protocol=2)

    # And save results to csv
    out_path = utils.create_csv_path(ID, network, conn_model, thr, mask, dir_path, node_size, smooth)
    np.savetxt(out_path, net_met_val_list_final, delimiter='\t')

    if frag is True:
        out_path_neat = "%s%s" % (out_path.split('.csv')[0], '_frag_neat.csv')
    else:
        out_path_neat = "%s%s" % (out_path.split('.csv')[0], '_neat.csv')
    df = pd.DataFrame.from_dict(dict(zip(metric_list_names, net_met_val_list_final)), orient='index').transpose()
    df.to_csv(out_path_neat, index=False)

    return out_path
