# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import sys
import os
import numpy as np
import networkx as nx
from numpy import genfromtxt

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

def create_communities(node_comm_aff_mat, node_num):
    com_assign = np.zeros((node_num,1))
    for i in range(len(node_comm_aff_mat)):
        community = node_comm_aff_mat[i,:]
        for j in range(len(community)):
            if community[j] == 1:
                com_assign[j,0]=i
    return com_assign

def modularity(W, qtype='sta', seed=None):
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
            #print('Infinite Loop aborted')
            sys.exit(0)

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
        print('Unrecognized clustering type')

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
        print('hierarchy %i' % i)

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
            print('Modularity Infinite Loop Style E.  Please '
                                'contact the developer with this error.')
            sys.exit(0)
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
                print('Modularity Infinite Loop Style F.  Please '
                                    'contact the developer with this error.')
                sys.exit(0)
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
    
##Extract network metrics interface
def extractnetstats(ID, network, thr, conn_model, est_path1, mask, out_file=None):
    from pynets import thresholding

    ##Load and threshold matrix
    in_mat = np.array(genfromtxt(est_path1))
    in_mat = thresholding.autofix(in_mat)

    ##Get hyperbolic tangent of matrix if non-sparse (i.e. fischer r-to-z transform)
    if conn_model == 'corr':
        in_mat = np.arctanh(in_mat)

    ##Get dir_path
    dir_path = os.path.dirname(os.path.realpath(est_path1))

    ##Assign Weight matrix
    mat_wei = in_mat
    ##Load numpy matrix as networkx graph
    G=nx.from_numpy_matrix(mat_wei)

    ##Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')
    ##Load numpy matrix as networkx graph
    G_len=nx.from_numpy_matrix(mat_len)

    ##Save gephi files
    if mask != None:
        if network != None:
            nx.write_graphml(G, dir_path + '/' + ID + '_' + network + '_' + str(os.path.basename(mask).split('.')[0]) + '.graphml')
        else:
            nx.write_graphml(G, dir_path + '/' + ID + '_' + str(os.path.basename(mask).split('.')[0]) + '.graphml')        
    else:
        if network != None:
            nx.write_graphml(G, dir_path + '/' + ID + '_' + network + '.graphml')
        else:
            nx.write_graphml(G, dir_path + '/' + ID + '.graphml')

    ###############################################################
    ########### Calculate graph metrics from graph G ##############
    ###############################################################
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, rich_club_coefficient, eigenvector_centrality, communicability_betweenness_centrality, clustering, degree_centrality
    from pynets.netstats import average_local_efficiency, global_efficiency, local_efficiency, modularity_louvain_dir, smallworldness
    ##For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the function  (with a G-only input) to the netstats module.
    metric_list = [global_efficiency, average_local_efficiency, smallworldness, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]

    ##Custom Parameters
    custom_params = 'weight = 0.25'

    ##Iteratively run functions from above metric list that generate single scalar output
    num_mets = len(metric_list)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j=0
    for i in metric_list:
        met_name = str(i).split('<function ')[1].split(' at')[0]
        if network != None:
            net_met = network + '_' + met_name
        else:
            net_met = met_name
        try:
            if custom_params and i is 'degree_assortativity_coefficient':
                net_met_val = float(i(G, custom_params))
            else:
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
    [community_aff, modularity] = modularity_louvain_dir(mat_wei)

    ##Calculate core-periphery subdivision
    [Coreness_vec, Coreness_q] = core_periphery_dir(mat_wei)

    ##Local Efficiency
    try:
        le_vector = local_efficiency(G)
        print('Extracting Local Efficiency vector for all network nodes...')
        le_vals = list(le_vector.values())
        le_nodes = list(le_vector.keys())
        num_nodes = len(le_nodes)
        le_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                le_arr[j,0] = network + '_' + str(le_nodes[j]) + '_local_efficiency'
                print('\n' + network + '_' + str(le_nodes[j]) + '_local_efficiency')
            else:
                le_arr[j,0] = 'WholeBrain_' + str(le_nodes[j]) + '_local_efficiency'
                print('\n' + 'WholeBrain_' + str(le_nodes[j]) + '_local_efficiency')
            try:
                le_arr[j,1] = le_vals[j]
            except:
                le_arr[j,1] = np.nan
            print(str(le_vals[j]))
            j = j + 1
        le_val_list = list(le_arr[:,1])
        le_arr[num_nodes,0] = network + '_MEAN_local_efficiency'
        nonzero_arr_le = np.delete(le_arr[:,1], [0])
        le_arr[num_nodes,1] = np.mean(nonzero_arr_le)
        print('\n' + 'Local Efficiency across all nodes: ' + str(le_arr[num_nodes,1]) + '\n')
    except:
        le_val_list = []

    ##Local Clustering
    try:
        cl_vector = clustering(G)
        print('Extracting Local Clustering vector for all network nodes...')
        cl_vals = list(cl_vector.values())
        cl_nodes = list(cl_vector.keys())
        num_nodes = len(cl_nodes)
        cl_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                cl_arr[j,0] = network + '_' + str(cl_nodes[j]) + '_local_clustering'
                print('\n' + network + '_' + str(cl_nodes[j]) + '_local_clustering')
            else:
                cl_arr[j,0] = 'WholeBrain_' + str(cl_nodes[j]) + '_local_clustering'
                print('\n' + 'WholeBrain_' + str(cl_nodes[j]) + '_local_clustering')
            try:
                cl_arr[j,1] = cl_vals[j]
            except:
                cl_arr[j,1] = np.nan
            print(str(cl_vals[j]))
            j = j + 1
        cl_val_list = list(cl_arr[:,1])
        cl_arr[num_nodes,0] = network + '_MEAN_local_efficiency'
        nonzero_arr_cl = np.delete(cl_arr[:,1], [0])
        cl_arr[num_nodes,1] = np.mean(nonzero_arr_cl)
        print('\n' + 'Local Efficiency across all nodes: ' + str(cl_arr[num_nodes,1]) + '\n')
    except:
        cl_val_list = []

    ##Degree centrality
    try:
        dc_vector = degree_centrality(G)
        print('Extracting Degree Centrality vector for all network nodes...')
        dc_vals = list(dc_vector.values())
        dc_nodes = list(dc_vector.keys())
        num_nodes = len(dc_nodes)
        dc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                dc_arr[j,0] = network + '_' + str(dc_nodes[j]) + '_degree_centrality'
                print('\n' + network + '_' + str(dc_nodes[j]) + '_degree_centrality')
            else:
                dc_arr[j,0] = 'WholeBrain_' + str(dc_nodes[j]) + '_degree_centrality'
                print('\n' + 'WholeBrain_' + str(dc_nodes[j]) + '_degree_centrality')
            try:
                dc_arr[j,1] = dc_vals[j]
            except:
                dc_arr[j,1] = np.nan
            print(str(cl_vals[j]))
            j = j + 1
        dc_val_list = list(dc_arr[:,1])
        dc_arr[num_nodes,0] = network + '_MEAN_degree_centrality'
        nonzero_arr_dc = np.delete(dc_arr[:,1], [0])
        dc_arr[num_nodes,1] = np.mean(nonzero_arr_dc)
        print('\n' + 'Degree Centrality across all nodes: ' + str(dc_arr[num_nodes,1]) + '\n')
    except:
        dc_val_list = []

    ##Betweenness Centrality
    try:
        bc_vector = betweenness_centrality(G_len)
        print('Extracting Betweeness Centrality vector for all network nodes...')
        bc_vals = list(bc_vector.values())
        bc_nodes = list(bc_vector.keys())
        num_nodes = len(bc_nodes)
        bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                bc_arr[j,0] = network + '_' + str(bc_nodes[j]) + '_betw_cent'
                print('\n' + network + '_' + str(bc_nodes[j]) + '_betw_cent')
            else:
                bc_arr[j,0] = 'WholeBrain_' + str(bc_nodes[j]) + '_betw_cent'
                print('\n' + 'WholeBrain_' + str(bc_nodes[j]) + '_betw_cent')
            try:
                bc_arr[j,1] = bc_vals[j]
            except:
                bc_arr[j,1] = np.nan
            print(str(bc_vals[j]))
            j = j + 1
        bc_val_list = list(bc_arr[:,1])
        bc_arr[num_nodes,0] = network + '_MEAN_betw_cent'
        nonzero_arr_betw_cent = np.delete(bc_arr[:,1], [0])
        bc_arr[num_nodes,1] = np.mean(nonzero_arr_betw_cent)
        print('\n' + 'Mean Betweenness Centrality across all nodes: ' + str(bc_arr[num_nodes,1]) + '\n')
    except:
        bc_val_list = []

    ##Eigenvector Centrality
    try:
        ec_vector = eigenvector_centrality(G)
        print('Extracting Eigenvector Centrality vector for all network nodes...')
        ec_vals = list(ec_vector.values())
        ec_nodes = list(ec_vector.keys())
        num_nodes = len(ec_nodes)
        ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                ec_arr[j,0] = network + '_' + str(ec_nodes[j]) + '_eig_cent'
                print('\n' + network + '_' + str(ec_nodes[j]) + '_eig_cent')
            else:
                ec_arr[j,0] = 'WholeBrain_' + str(ec_nodes[j]) + '_eig_cent'
                print('\n' + 'WholeBrain_' + str(ec_nodes[j]) + '_eig_cent')
            try:
                ec_arr[j,1] = ec_vals[j]
            except:
                ec_arr[j,1] = np.nan
            print(str(ec_vals[j]))
            j = j + 1
        ec_val_list = list(ec_arr[:,1])
        ec_arr[num_nodes,0] = network + '_MEAN_eig_cent'
        nonzero_arr_eig_cent = np.delete(ec_arr[:,1], [0])
        ec_arr[num_nodes,1] = np.mean(nonzero_arr_eig_cent)
        print('\n' + 'Mean Eigenvector Centrality across all nodes: ' + str(ec_arr[num_nodes,1]) + '\n')
    except:
        ec_val_list = []

    ##Communicability Centrality
    try:
        cc_vector = communicability_betweenness_centrality(G_len)
        print('Extracting Communicability Centrality vector for all network nodes...')
        cc_vals = list(cc_vector.values())
        cc_nodes = list(cc_vector.keys())
        num_nodes = len(cc_nodes)
        cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if network != None:
                cc_arr[j,0] = network + '_' + str(cc_nodes[j]) + '_comm_cent'
                print('\n' + network + '_' + str(cc_nodes[j]) + '_comm_cent')
            else:
                cc_arr[j,0] = 'WholeBrain_' + str(cc_nodes[j]) + '_comm_cent'
                print('\n' + 'WholeBrain_' + str(cc_nodes[j]) + '_comm_cent')
            try:
                cc_arr[j,1] = cc_vals[j]
            except:
                cc_arr[j,1] = np.nan
            print(str(cc_vals[j]))
            j = j + 1
        cc_val_list = list(cc_arr[:,1])
        cc_arr[num_nodes,0] = network + '_MEAN_comm_cent'
        nonzero_arr_comm_cent = np.delete(cc_arr[:,1], [0])
        cc_arr[num_nodes,1] = np.mean(nonzero_arr_comm_cent)
        print('\n' + 'Mean Communicability Centrality across all nodes: ' + str(cc_arr[num_nodes,1]) + '\n')
    except:
        cc_val_list = []

    ##Rich club coefficient
    try:
        rc_vector = rich_club_coefficient(G, normalized=True)
        print('Extracting Rich Club Coefficient vector for all network nodes...')
        rc_vals = list(rc_vector.values())
        rc_edges = list(rc_vector.keys())
        num_edges = len(rc_edges)
        rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
        j=0
        for i in range(num_edges):
            if network != None:
                rc_arr[j,0] = network + '_' + str(rc_edges[j]) + '_rich_club'
                print('\n' + network + '_' + str(rc_edges[j]) + '_rich_club')
            else:
                rc_arr[j,0] = 'WholeBrain_' + str(rc_edges[j]) + '_rich_club'
                print('\n' + 'WholeBrain_' + str(rc_edges[j]) + '_rich_club')
            try:
                rc_arr[j,1] = rc_vals[j]
            except:
                rc_arr[j,1] = np.nan
            print(str(rc_vals[j]))
            j = j + 1
        ##Add mean
        rc_val_list = list(rc_arr[:,1])
        rc_arr[num_edges,0] = network + '_MEAN_rich_club'
        nonzero_arr_rich_club = np.delete(rc_arr[:,1], [0])
        rc_arr[num_edges,1] = np.mean(nonzero_arr_rich_club)
        print('\n' + 'Mean Rich Club Coefficient across all edges: ' + str(rc_arr[num_edges,1]) + '\n')
    except:
        rc_val_list = []

    ##Create a list of metric names for scalar metrics
    metric_list_names = []
    net_met_val_list_final = net_met_val_list
    for i in net_met_arr[:,0]:
        metric_list_names.append(i)

    ##Add modularity measure
    try:
        if network != None:
            metric_list_names.append(network + '_Modularity')
        else:
            metric_list_names.append('WholeBrain_Modularity')
        net_met_val_list_final.append(modularity)
    except:
        pass

    ##Add Core/Periphery measure
    try:
        if network != None:
            metric_list_names.append(network + '_Coreness')
        else:
            metric_list_names.append('WholeBrain_Coreness')
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
        import cPickle
    except ImportError:
        import _pickle as cPickle
        
    if mask != None:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/net_metric_list_' + network + '_' + str(os.path.basename(mask).split('.')[0])
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/net_metric_list_WB' + '_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/net_metric_list_' + network
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/net_metric_list_WB'
    cPickle.dump(metric_list_names, open(met_list_picke_path, 'wb'))

    ##Save results to csv
    if mask != None:
        if network != None:
            out_path = dir_path + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr) + '_' + str(os.path.basename(mask).split('.')[0]) + '.csv'
    else:
        if network != None:
            out_path = dir_path + '/' + ID + '_' + network + '_net_metrics_' + conn_model + '_' + str(thr) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_metrics_' + conn_model + '_' + str(thr) + '.csv'
    np.savetxt(out_path, net_met_val_list_final)

    return(out_path)
