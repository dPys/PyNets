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
        return 0    # facilitates calculation of local_efficiency although
                    # could reasonably raise nx.NetworkXUnfeasible or
                    # nx.NetworkXPointlessConcept error instead and force
                    # testing to occur in local_efficiency

    inv_lengths = []
    for node in G:
        if weight is None:
            lengths = nx.single_source_shortest_path_length(G, node)
        else:
            lengths = nx.single_source_dijkstra_path_length(G, node,
                                                            weight=weight)

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

def smallworldness(G, rep = 1000):
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

def clustering_coef_wd(W):
    from pynets.utils import cuberoot
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.
    Parameters
    ----------
    W : NxN np.ndarray
        weighted directed connection matrix
    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    Notes
    -----
    Methodological note (also see clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version
    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    '''
    A = np.logical_not(W == 0).astype(float)
    S = cuberoot(W) + cuberoot(W.T)
    K = np.sum(A + A.T, axis=1)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2
    K[np.where(cyc3 == 0)] = np.inf
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3
    return C

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
    #
    # The list is sorted in reverse order so that we can pop from the
    # right side of the list later, instead of popping from the left
    # side of the list, which would have a linear time cost.
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()),
                          reverse=True)
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

def rich_club_coefficient(G, normalized=True, Q=100):
    """Returns the rich-club coefficient of the graph `G`.

    For each degree *k*, the *rich-club coefficient* is the ratio of the
    number of actual to the number of potential edges for nodes with
    degree greater than *k*:

    .. math::

        \phi(k) = \frac{2 E_k}{N_k (N_k - 1)}

    where `N_k` is the number of nodes with degree larger than *k*, and
    `E_k` is the number of edges among those nodes.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph with neither parallel edges nor self-loops.
    normalized : bool (optional)
        Normalize using randomized network as in [1]_
    Q : float (optional, default=100)
        If `normalized` is True, perform `Q * m` double-edge
        swaps, where `m` is the number of edges in `G`, to use as a
        null-model for normalization.

    Returns
    -------
    rc : dictionary
       A dictionary, keyed by degree, with rich-club coefficient values.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    >>> rc = nx.rich_club_coefficient(G, normalized=False)
    >>> rc[0] # doctest: +SKIP
    0.4

    Notes
    -----
    The rich club definition and algorithm are found in [1]_.  This
    algorithm ignores any edge weights and is not defined for directed
    graphs or graphs with parallel edges or self loops.

    Estimates for appropriate values of `Q` are found in [2]_.

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa,
       and Tibério S. Caetano,
       "The rich-club phenomenon across complex network hierarchies",
       Applied Physics Letters Vol 91 Issue 8, August 2007.
       https://arxiv.org/abs/physics/0701290
    .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,
       "Uniform generation of random graphs with arbitrary degree
       sequences", 2006. https://arxiv.org/abs/cond-mat/0312028
    """
    if nx.number_of_selfloops(G) > 0:
        raise Exception('rich_club_coefficient is not implemented for '
                        'graphs with self loops.')
    rc = _compute_rc(G)
    if normalized:
        # make R a copy of G, randomize with Q*|E| double edge swaps
        # and use rich_club coefficient of R to normalize
        R = G.copy()
        E = R.number_of_edges()
        nx.double_edge_swap(R, Q * E, max_tries=Q * E * 10)
        rcran = _compute_rc(R)
        rcran = {key:val for key, val in rcran.items() if val != 0}
        rc = {k: v / rcran[k] for k, v in rcran.items()}
        #rc_norm = {k: v / rcran[k] for k, v in rcran.items()}
        #good_nodes = list(rc_norm.keys())
        #for k in list(rc.items()):
            #if k[0] not in good_nodes:
                #del rc[k[0]]
    return rc

def modularity(W, qtype='sta', seed=42):
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
    '''
    n = len(W)  # number of nodes

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    m = np.max(ci)  # number of modules

    def entropy(w_):
        S = np.sum(w_, axis=1)  # strength
        Snm = np.zeros((n, m))  # node-to-module degree
        for i in range(m):
            Snm[:, i] = np.sum(w_[:, ci == i + 1], axis=1)
        pnm = Snm / (np.tile(S, (m, 1)).T)
        pnm[np.isnan(pnm)] = 0
        pnm[np.logical_not(pnm)] = 1
        return -np.sum(pnm * np.log(pnm), axis=1) / np.log(m)

    #explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg

def core_periphery_dir(W, gamma=1, C0=None):
    '''
    The optimal core/periphery subdivision is a partition of the network
    into two nonoverlapping groups of nodes, a core group and a periphery
    group. The number of core-group edges is maximized, and the number of
    within periphery edges is minimized.
    The core-ness is a statistic which quantifies the goodness of the
    optimal core/periphery subdivision (with arbitrary relative value).
    The algorithm uses a variation of the Kernighan-Lin graph partitioning
    algorithm to optimize a core-structure objective described in
    Borgatti & Everett (2000) Soc Networks 21:375-395
    See Rubinov, Ypma et al. (2015) PNAS 112:10032-7
    Parameters
    ----------
    W : NxN np.ndarray
        directed connection matrix
    gamma : core-ness resolution parameter
        Default value = 1
        gamma > 1 detects small core, large periphery
        0 < gamma < 1 detects large core, small periphery
    C0 : NxN np.ndarray
        Initial core structure
    '''
    n = len(W)
    np.fill_diagonal(W, 0)

    if C0 == None:
        C = np.random.randint(2, size=(n,))
    else:
        C = C0.copy()

    s = np.sum(W)
    p = np.mean(W)
    b = W - gamma * p
    B = (b + b.T) / (2 * s)
    cix, = np.where(C)
    ncix, = np.where(np.logical_not(C))
    q = np.sum(B[np.ix_(cix, cix)]) - np.sum(B[np.ix_(ncix, ncix)])

    #print(q)

    flag = True
    it = 0
    while flag:
        it += 1
        if it > 100:
            raise ValueError('Infinite Loop aborted')

        flag = False
        #initial node indices
        ixes = np.arange(n)

        Ct = C.copy()
        while len(ixes) > 0:
            Qt = np.zeros((n,))
            ctix, = np.where(Ct)
            nctix, = np.where(np.logical_not(Ct))
            q0 = (np.sum(B[np.ix_(ctix, ctix)]) -
                  np.sum(B[np.ix_(nctix, nctix)]))
            Qt[ctix] = q0 - 2 * np.sum(B[ctix, :], axis=1)
            Qt[nctix] = q0 + 2 * np.sum(B[nctix, :], axis=1)

            max_Qt = np.max(Qt[ixes])
            u, = np.where(np.abs(Qt[ixes]-max_Qt) < 1e-10)
            #u = u[np.random.randint(len(u))]
            #print(np.sum(Ct))
            Ct[ixes[u]] = np.logical_not(Ct[ixes[u]])
            #print(np.sum(Ct))

            ixes = np.delete(ixes, u)

            #print(max_Qt - q)
            #print(len(ixes))

            if max_Qt - q > 1e-10:
                flag = True
                C = Ct.copy()
                cix, = np.where(C)
                ncix, = np.where(np.logical_not(C))
                q = (np.sum(B[np.ix_(cix, cix)]) -
                     np.sum(B[np.ix_(ncix, ncix)]))

    cix, = np.where(C)
    ncix, = np.where(np.logical_not(C))
    q = np.sum(B[np.ix_(cix, cix)]) - np.sum(B[np.ix_(ncix, ncix)])
    return C, q

def link_communities(W, type_clustering='single'):
    from pynets.thresholding import normalize
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

    # set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[range(n), range(n)] = (
        np.sum(W, axis=0) / np.sum(np.logical_not(W), axis=0) +
        np.sum(W.T, axis=0) / np.sum(np.logical_not(W.T), axis=0)) / 2

    # out/in norm squared
    No = np.sum(W**2, axis=1)
    Ni = np.sum(W**2, axis=0)

    # weighted in/out jaccard
    Jo = np.zeros((n, n))
    Ji = np.zeros((n, n))

    for b in range(n):
        for c in range(n):
            Do = np.dot(W[b, :], W[c, :].T)
            Jo[b, c] = Do / (No[b] + No[c] - Do)

            Di = np.dot(W[:, b].T, W[:, c])
            Ji[b, c] = Di / (Ni[b] + Ni[c] - Di)

    # get link similarity
    A, B = np.where(np.logical_and(np.logical_or(W, W.T),
                                   np.triu(np.ones((n, n)), 1)))
    m = len(A)
    Ln = np.zeros((m, 2), dtype=np.int32)  # link nodes
    Lw = np.zeros((m,))  # link weights

    for i in range(m):
        Ln[i, :] = (A[i], B[i])
        Lw[i] = (W[A[i], B[i]] + W[B[i], A[i]]) / 2

    ES = np.zeros((m, m), dtype=np.float32)  # link similarity
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

            ES[i, j] = (W[a, b] * W[a, c] * Ji[b, c] +
                        W[b, a] * W[c, a] * Jo[b, c]) / 2

    np.fill_diagonal(ES, 0)

    # perform hierarchical clustering

    C = np.zeros((m, m), dtype=np.int32)  # community affiliation matrix

    Nc = C.copy()
    Mc = np.zeros((m, m), dtype=np.float32)
    Dc = Mc.copy()  # community nodes, links, density

    U = np.arange(m)  # initial community assignments
    C[0, :] = np.arange(m)


    for i in range(m - 1):
        print('Hierarchy %i' % i)

        #time1 = time.time()

        for j in range(len(U)):  # loop over communities
            ixes = C[i, :] == U[j]  # get link indices

            links = np.sort(Lw[ixes])
            #nodes = np.sort(Ln[ixes,:].flat)

            nodes = np.sort(np.reshape(
                Ln[ixes, :], 2 * np.size(np.where(ixes))))

            # get unique nodes
            nodulo = np.append(nodes[0], (nodes[1:])[nodes[1:] != nodes[:-1]])
            #nodulo = ((nodes[1:])[nodes[1:] != nodes[:-1]])

            nc = len(nodulo)
            #nc = len(nodulo)+1
            mc = np.sum(links)
            min_mc = np.sum(links[:nc - 1])  # minimal weight
            dc = (mc - min_mc) / (nc * (nc - 1) /
                                  2 - min_mc)  # community density

            if np.array(dc).shape is not ():
                print(dc)
                print(dc.shape)

            Nc[i, j] = nc
            Mc[i, j] = mc
            Dc[i, j] = dc if not np.isnan(dc) else 0

        #time2 = time.time()
        #print('compute densities time', time2-time1)
        C[i + 1, :] = C[i, :]  # copy current partition
        #if i in (2693,):
        #    import pdb
        #    pdb.set_trace()
        # Profiling and debugging show that this line, finding
        # the max values in this matrix, take about 3x longer than the
        # corresponding matlab version. Can it be improved?

        u1, u2 = np.where(ES[np.ix_(U, U)] == np.max(ES[np.ix_(U, U)]))

        if np.size(u1) > 2:
            # pick one
            wehr, = np.where((u1 == u2[0]))

            uc = np.squeeze((u1[0], u2[0]))
            ud = np.squeeze((u1[wehr], u2[wehr]))

            u1 = uc
            u2 = ud

        #time25 = time.time()
        #print('copy and max time', time25-time2)
        # get unique links (implementation of matlab sortrows)
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

            # assign distances to whole clusters
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

def modularity_louvain_dir(W, gamma=1, hierarchy=False, seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    The Louvain algorithm is a fast and accurate community detection
    algorithm (as of writing). The algorithm may also be used to detect
    hierarchical community structure.
    Parameters
    ----------
    W : NxN np.ndarray
        directed weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    hierarchy : bool
        Enables hierarchical output. Defalut value=False
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.ndarray
        refined community affiliation vector. If hierarchical output enabled,
        it is an NxH np.ndarray instead with multiple iterations
    Q : float
        optimized modularity metric. If hierarchical output enabled, becomes
        an Hx1 array of floats instead.
    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    '''
    np.random.seed(seed)

    n = len(W)  # number of nodes
    s = np.sum(W)  # total weight of edges
    h = 0  # hierarchy index
    ci = []
    ci.append(np.arange(n) + 1)  # hierarchical module assignments
    q = []
    q.append(-1)  # hierarchical modularity index
    n0 = n

    while True:
        if h > 300:
            raise ValueError('Modularity Infinite Loop Style')
        k_o = np.sum(W, axis=1)  # node in/out degrees
        k_i = np.sum(W, axis=0)
        km_o = k_o.copy()  # module in/out degrees
        km_i = k_i.copy()
        knm_o = W.copy()  # node-to-module in/out degrees
        knm_i = W.copy()

        m = np.arange(n) + 1  # initial module assignments

        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise ValueError('Modularity Infinite Loop Style')
            flag = False

            # loop over nodes in random order
            for u in np.random.permutation(n):
                ma = m[u] - 1
                # algorithm condition
                dq_o = ((knm_o[u, :] - knm_o[u, ma] + W[u, u]) -
                        gamma * k_o[u] * (km_i - km_i[ma] + k_i[u]) / s)
                dq_i = ((knm_i[u, :] - knm_i[u, ma] + W[u, u]) -
                        gamma * k_i[u] * (km_o - km_o[ma] + k_o[u]) / s)
                dq = (dq_o + dq_i) / 2
                dq[ma] = 0

                max_dq = np.max(dq)  # find maximal modularity increase
                if max_dq > 1e-10:  # if maximal increase positive
                    mb = np.argmax(dq)  # take only one value

                    knm_o[:, mb] += W[u, :].T  # change node-to-module degrees
                    knm_o[:, ma] -= W[u, :].T
                    knm_i[:, mb] += W[:, u]
                    knm_i[:, ma] -= W[:, u]
                    km_o[mb] += k_o[u]  # change module out-degrees
                    km_o[ma] -= k_o[u]
                    km_i[mb] += k_i[u]
                    km_i[ma] -= k_i[u]

                    m[u] = mb + 1  # reassign module
                    flag = True

        _, m = np.unique(m, return_inverse=True)
        m += 1
        h += 1
        ci.append(np.zeros((n0,)))
        # for i,mi in enumerate(m):		#loop through module assignments
        for i in range(n):
            # ci[h][np.where(ci[h-1]==i)]=mi	#assign new modules
            ci[h][np.where(ci[h - 1] == i + 1)] = m[i]

        n = np.max(m)  # new number of modules
        W1 = np.zeros((n, n))  # new weighted matrix
        for i in range(n):
            for j in range(n):
                # pool weights of nodes in same module
                W1[i, j] = np.sum(W[np.ix_(m == i + 1, m == j + 1)])

        q.append(0)
        # compute modularity
        q[h] = np.trace(W1) / s - gamma * np.sum(np.dot(W1 / s, W1 / s))
        if q[h] - q[h - 1] < 1e-10:  # if modularity does not increase
            break

    ci = np.array(ci, dtype=int)
    if hierarchy:
        ci = ci[1:-1]
        q = q[1:-1]
        return ci, q
    else:
        return ci[h - 1], q[h - 1]

def most_important(G):
     """ returns a copy of G with
         the most important nodes
         according to the pagerank """
     ranking = nx.betweenness_centrality(G).items()
     #print(ranking)
     r = [x[1] for x in ranking]
     m = sum(r)/len(r) - 3*np.std(r)
     Gt = G.copy()
     pruned_nodes = []
     i = 0
     for k, v in ranking:
        if v < m:
            Gt.remove_node(k)
            pruned_nodes.append(i)
        i = i + 1
     pruned_edges = []
     ##Remove near-zero isolates
     s = 0
     components = list(nx.connected_components(Gt)) # list because it returns a generator
     components.sort(key=len, reverse=True)
     components_isolated = list(components[0])

     for node,degree in list(Gt.degree()):
         if degree < 0.001:
             try:
                 Gt.remove_node(node)
                 pruned_edges.append(s)
             except:
                 pass
         if node not in components_isolated:
             try:
                 Gt.remove_node(node)
                 pruned_edges.append(s)
             except:
                 pass
         s = s + 1
     return Gt, pruned_nodes, pruned_edges


##Extract network metrics interface
def extractnetstats(ID, network, thr, conn_model, est_path, mask, prune, node_size, out_file=None):
    from pynets import thresholding, utils

    ##Load and threshold matrix
    in_mat = np.array(np.load(est_path))
    in_mat = thresholding.autofix(in_mat)

    ##Normalize connectivity matrix (weights between 0-1)
    in_mat = thresholding.normalize(in_mat)

    ##Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix if non-covariance
    if conn_model == 'corr':
        in_mat = np.arctanh(in_mat)
        in_mat[np.isnan(in_mat)] = 0
        in_mat[np.isinf(in_mat)] = 1

    ##Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path))

    ##Load numpy matrix as networkx graph
    G_pre=nx.from_numpy_matrix(in_mat)

    ##Prune irrelevant nodes (i.e. nodes who are fully disconnected from the graph and/or those whose betweenness centrality are > 3 standard deviations below the mean)
    if prune is True:
        [G_pruned, _, _] = most_important(G_pre)
    else:
        G_pruned = G_pre

    ##Make directed if model is effective type
    if conn_model == 'effective':
        G_di = nx.DiGraph(G_pruned)
        G_dir = G_di.to_directed()
        G = G_pruned
    else:
        G = G_pruned

    ##Get corresponding matrix
    in_mat = nx.to_numpy_array(G)

    ##Print graph summary
    print("%s%.2f%s" % ('\n\nThreshold: ',100*float(thr),'%'))
    print("%s%s" % ('Source File: ', est_path))
    info_list = list(nx.info(G).split('\n'))[2:]
    for i in info_list:
        print(i)

    try:
        G_dir
        print('Analyzing DIRECTED graph when applicable...')
    except:
        print('Graph is UNDIRECTED')
        try:
            if nx.is_connected(G) == True:
                print('Graph is CONNECTED')
            else:
                print('Graph is DISCONNECTED\n')
        except:
            pass

    ##Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')
    ##Load numpy matrix as networkx graph
    G_len = nx.from_numpy_matrix(mat_len)

    ##Save G as gephi file
    if mask:
        if network:
            nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_', os.path.basename(mask).split('.')[0], '_', thr, '_', node_size, '.graphml'))
        else:
            nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_', thr, '_', node_size, '.graphml'))
    else:
        if network:
            nx.write_graphml(G, "%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_', thr, '_', node_size, '.graphml'))
        else:
            nx.write_graphml(G, "%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', thr, '_', node_size, '.graphml'))

    ###############################################################
    ########### Calculate graph metrics from graph G ##############
    ###############################################################
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, eigenvector_centrality, communicability_betweenness_centrality, clustering, degree_centrality
    from pynets.netstats import average_local_efficiency, global_efficiency, local_efficiency, modularity_louvain_dir, smallworldness
    ##For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the function  (with a G-only input) to the netstats module.
    metric_list = [global_efficiency, average_local_efficiency, smallworldness, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]

    ##Custom Weight Parameter
    #custom_weight = 0.25
    custom_weight = None

    ##Iteratively run functions from above metric list that generate single scalar output
    num_mets = len(metric_list)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j=0
    for i in metric_list:
        met_name = str(i).split('<function ')[1].split(' at')[0]
        net_met = met_name
        try:
            if i is 'average_shortest_path_length':
                try:
                    try:
                        net_met_val = float(i(G_dir))
                        print('Calculating from directed graph...')
                    except:
                        net_met_val = float(i(G))
                except:
                    ##case where G is not fully connected
                    print('WARNING: Calculating average shortest path length for a disconnected graph. This might take awhile...')
                    net_met_val = float(average_shortest_path_length_for_all(G))
            if custom_weight is not None and i is 'degree_assortativity_coefficient' or i is 'global_efficiency' or i is 'average_local_efficiency' or i is 'average_clustering':
                custom_weight_param = 'weight = ' + str(custom_weight)
                try:
                    net_met_val = float(i(G_dir, custom_weight_param))
                    print('Calculating from directed graph...')
                except:
                    net_met_val = float(i(G, custom_weight_param))
            else:
                try:
                    net_met_val = float(i(G_dir))
                    print('Calculating from directed graph...')
                except:
                    net_met_val = float(i(G))
        except:
            net_met_val = np.nan
        net_met_arr[j,0] = net_met
        net_met_arr[j,1] = net_met_val
        print(net_met)
        print(str(net_met_val))
        print('\n')
        j = j + 1
    net_met_val_list = list(net_met_arr[:,1])

    ##Run miscellaneous functions that generate multiple outputs
    ##Calculate modularity using the Louvain algorithm
    try:
        [community_aff, modularity] = modularity_louvain_dir(in_mat)
    except:
        pass

    ##Calculate core-periphery subdivision
    try:
        [Coreness_vec, Coreness_q] = core_periphery_dir(in_mat)
    except:
        pass

    ##Local Efficiency
    try:
        try:
            le_vector = local_efficiency(G_dir)
        except:
            le_vector = local_efficiency(G)
        print('\nExtracting Local Efficiency vector for all network nodes...')
        le_vals = list(le_vector.values())
        le_nodes = list(le_vector.keys())
        num_nodes = len(le_nodes)
        le_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            le_arr[j,0] = str(le_nodes[j]) + '_local_efficiency'
            #print('\n' + str(le_nodes[j]) + '_local_efficiency')
            try:
                le_arr[j,1] = le_vals[j]
            except:
                le_arr[j,1] = np.nan
            #print(str(le_vals[j]))
            j = j + 1
        le_arr[num_nodes,0] = 'MEAN_local_efficiency'
        nonzero_arr_le = np.delete(le_arr[:,1], [0])
        le_arr[num_nodes,1] = np.mean(nonzero_arr_le)
        print('Mean Local Efficiency across nodes: ' + str(le_arr[num_nodes,1]))
    except:
        pass

    ##Local Clustering
    try:
        cl_vector = clustering(G)
        print('\nExtracting Local Clustering vector for all network nodes...')
        cl_vals = list(cl_vector.values())
        cl_nodes = list(cl_vector.keys())
        num_nodes = len(cl_nodes)
        cl_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            cl_arr[j,0] = str(cl_nodes[j]) + '_local_clustering'
            #print('\n' + str(cl_nodes[j]) + '_local_clustering')
            try:
                cl_arr[j,1] = cl_vals[j]
            except:
                cl_arr[j,1] = np.nan
            #print(str(cl_vals[j]))
            j = j + 1
        cl_arr[num_nodes,0] = 'MEAN_local_efficiency'
        nonzero_arr_cl = np.delete(cl_arr[:,1], [0])
        cl_arr[num_nodes,1] = np.mean(nonzero_arr_cl)
        print('Mean Local Clustering across nodes: ' + str(cl_arr[num_nodes,1]))
    except:
        pass

    ##Degree centrality
    try:
        try:
            dc_vector = degree_centrality(G_dir)
        except:
            dc_vector = degree_centrality(G)
        print('\nExtracting Degree Centrality vector for all network nodes...')
        dc_vals = list(dc_vector.values())
        dc_nodes = list(dc_vector.keys())
        num_nodes = len(dc_nodes)
        dc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            dc_arr[j,0] = str(dc_nodes[j]) + '_degree_centrality'
            #print('\n' + str(dc_nodes[j]) + '_degree_centrality')
            try:
                dc_arr[j,1] = dc_vals[j]
            except:
                dc_arr[j,1] = np.nan
            #print(str(cl_vals[j]))
            j = j + 1
        dc_arr[num_nodes,0] = 'MEAN_degree_centrality'
        nonzero_arr_dc = np.delete(dc_arr[:,1], [0])
        dc_arr[num_nodes,1] = np.mean(nonzero_arr_dc)
        print('Mean Degree Centrality across nodes: ' + str(dc_arr[num_nodes,1]))
    except:
        pass

    ##Betweenness Centrality
    try:
        bc_vector = betweenness_centrality(G_len, normalized=True)
        print('\nExtracting Betweeness Centrality vector for all network nodes...')
        bc_vals = list(bc_vector.values())
        bc_nodes = list(bc_vector.keys())
        num_nodes = len(bc_nodes)
        bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            bc_arr[j,0] = str(bc_nodes[j]) + '_betweenness_centrality'
            #print('\n' + str(bc_nodes[j]) + '_betw_cent')
            try:
                bc_arr[j,1] = bc_vals[j]
            except:
                bc_arr[j,1] = np.nan
            #print(str(bc_vals[j]))
            j = j + 1
        bc_arr[num_nodes,0] = 'MEAN_betw_cent'
        nonzero_arr_betw_cent = np.delete(bc_arr[:,1], [0])
        bc_arr[num_nodes,1] = np.mean(nonzero_arr_betw_cent)
        print('Mean Betweenness Centrality across nodes: ' + str(bc_arr[num_nodes,1]))
    except:
        pass

    ##Eigenvector Centrality
    try:
        try:
            ec_vector = eigenvector_centrality(G_dir, max_iter=1000)
        except:
            ec_vector = eigenvector_centrality(G, max_iter=1000)
        print('\nExtracting Eigenvector Centrality vector for all network nodes...')
        ec_vals = list(ec_vector.values())
        ec_nodes = list(ec_vector.keys())
        num_nodes = len(ec_nodes)
        ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            ec_arr[j,0] = str(ec_nodes[j]) + '_eigenvector_centrality'
            #print('\n' + str(ec_nodes[j]) + '_eig_cent')
            try:
                ec_arr[j,1] = ec_vals[j]
            except:
                ec_arr[j,1] = np.nan
            #print(str(ec_vals[j]))
            j = j + 1
        ec_arr[num_nodes,0] = 'MEAN_eig_cent'
        nonzero_arr_eig_cent = np.delete(ec_arr[:,1], [0])
        ec_arr[num_nodes,1] = np.mean(nonzero_arr_eig_cent)
        print('Mean Eigenvector Centrality across nodes: ' + str(ec_arr[num_nodes,1]))
    except:
        pass

    ##Communicability Centrality
    try:
        cc_vector = communicability_betweenness_centrality(G, normalized=True)
        print('\nExtracting Communicability Centrality vector for all network nodes...')
        cc_vals = list(cc_vector.values())
        cc_nodes = list(cc_vector.keys())
        num_nodes = len(cc_nodes)
        cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            cc_arr[j,0] = str(cc_nodes[j]) + '_communicability_centrality'
            #print('\n' + str(cc_nodes[j]) + '_comm_cent')
            try:
                cc_arr[j,1] = cc_vals[j]
            except:
                cc_arr[j,1] = np.nan
            #print(str(cc_vals[j]))
            j = j + 1
        cc_arr[num_nodes,0] = 'MEAN_comm_cent'
        nonzero_arr_comm_cent = np.delete(cc_arr[:,1], [0])
        cc_arr[num_nodes,1] = np.mean(nonzero_arr_comm_cent)
        print('Mean Communicability Centrality across nodes: ' + str(cc_arr[num_nodes,1]))
    except:
        pass

    ##Rich club coefficient
    try:
        rc_vector = rich_club_coefficient(G, normalized=True)
        print('\nExtracting Rich Club Coefficient vector for all network nodes...')
        rc_vals = list(rc_vector.values())
        rc_edges = list(rc_vector.keys())
        num_edges = len(rc_edges)
        rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
        j=0
        for i in range(num_edges):
            rc_arr[j,0] = str(rc_edges[j]) + '_rich_club'
            #print('\n' + str(rc_edges[j]) + '_rich_club')
            try:
                rc_arr[j,1] = rc_vals[j]
            except:
                rc_arr[j,1] = np.nan
            #print(str(rc_vals[j]))
            j = j + 1
        ##Add mean
        rc_arr[num_edges,0] = 'MEAN_rich_club'
        nonzero_arr_rich_club = np.delete(rc_arr[:,1], [0])
        rc_arr[num_edges,1] = np.mean(nonzero_arr_rich_club)
        print('Mean Rich Club Coefficient across edges: ' + str(rc_arr[num_edges,1]))
    except:
        pass

    ##Create a list of metric names for scalar metrics
    metric_list_names = []
    net_met_val_list_final = net_met_val_list
    for i in net_met_arr[:,0]:
        metric_list_names.append(i)

    ##Add modularity measure
    try:
        metric_list_names.append('Modularity')
        net_met_val_list_final.append(modularity)
    except:
        pass

    ##Add Core/Periphery measure
    try:
        metric_list_names.append('Coreness')
        net_met_val_list_final.append(Coreness_q)
    except:
        pass

    ##Add local efficiency measures
    try:
        for i in le_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(le_arr[:,1])
    except:
        pass

    ##Add local clustering measures
    try:
        for i in cl_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(cl_arr[:,1])
    except:
        pass

    ##Add centrality measures
    try:
        for i in dc_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(dc_arr[:,1])
    except:
        pass
    try:
        for i in bc_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(bc_arr[:,1])
    except:
        pass
    try:
        for i in ec_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(ec_arr[:,1])
    except:
        pass
    try:
        for i in cc_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(cc_arr[:,1])
    except:
        pass

    ##Add rich club measure
    try:
        for i in rc_arr[:,0]:
            metric_list_names.append(i)
        net_met_val_list_final = net_met_val_list_final + list(rc_arr[:,1])
    except:
        pass

    ##Save metric names as pickle
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    if mask != None:
        if network != None:
            met_list_picke_path = "%s%s%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list_', network, '_', os.path.basename(mask).split('.')[0])
        else:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list_', os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list_', network)
        else:
            met_list_picke_path = "%s%s" % (os.path.dirname(os.path.abspath(est_path)), '/net_metric_list')
    pickle.dump(metric_list_names, open(met_list_picke_path, 'wb'), protocol=2)

    ##And save results to csv
    out_path = utils.create_csv_path(ID, network, conn_model, thr, mask, dir_path, node_size)
    np.savetxt(out_path, net_met_val_list_final, delimiter='\t')

    return out_path
