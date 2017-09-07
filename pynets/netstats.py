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
import multiprocessing
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

def create_communities(node_comm_aff_mat, node_num):
    com_assign = np.zeros((node_num,1))
    for i in range(len(nod_comm_aff_mat)):
        community = nod_comm_aff_mat[i,:]
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

def link_communities(W, type_clustering='single'):
    n = len(W)
    #W = normalize(W)

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

    import time

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

def community_louvain(W, gamma=1, ci=None, B='modularity', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).
    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        NxN objective-function matrix. builtin values
            'modularity' uses Q-metric as objective function
            'potts' uses Potts model Hamiltonian.
            'negative_sym' symmetric treatment of negative weights
            'negative_asym' asymmetric treatment of negative weights
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)
    '''
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)

    if np.min(W) < -1e-10:
        print('adjmat must not contain negative weights')

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            print('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W, axis=0)) / s0

        W1 = W * (W < 0)
        s1 = np.sum(W1)
        if s1:
            B1 = (W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0))
                / s1)
        else:
            B1 = 0

    elif np.min(W) < -1e-10:
        print("Input connection matrix contains negative "
            'weights but objective function dealing with negative weights '
            'was not selected')

    if B == 'potts' and np.any(np.logical_not(np.logical_or(W == 0, W == 1))):
        print('Potts hamiltonian requires binary input matrix')

    if B == 'modularity':
        B = W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s
    elif B == 'potts':
        B = W - gamma * np.logical_not(W)
    elif B == 'negative_sym':
        B = B0 / (s0 + s1) - B1 / (s0 + s1)
    elif B == 'negative_asym':
        B = B0 / s0 - B1 / (s0 + s1)
    else:
        try:
            B = np.array(B)
        except:
            print('unknown objective function type')

        if B.shape != W.shape:
            print('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print ('Warning: objective function matrix not symmetric, '
                   'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                print('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q

##Extract network metrics interface
def extractnetstats(ID, NETWORK, thr, conn_model, est_path1, out_file=None):
    import pynets
    from pynets import netstats, thresholding

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

    ##Create Binary matrix
    #mat_bin = weight_conversion(in_mat, 'binarize')
    ##Load numpy matrix as networkx graph
    #G_bin=nx.from_numpy_matrix(mat_bin)

    ##Create Length matrix
    mat_len = thresholding.weight_conversion(in_mat, 'lengths')
    ##Load numpy matrix as networkx graph
    G_len=nx.from_numpy_matrix(mat_len)

    ##Save gephi files
    if NETWORK != None:
        nx.write_graphml(G, dir_path + '/' + ID + '_' + NETWORK + '.graphml')
    else:
        nx.write_graphml(G, dir_path + '/' + ID + '.graphml')

    ###############################################################
    ########### Calculate graph metrics from graph G ##############
    ###############################################################
    import random
    import itertools
    from itertools import permutations
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, rich_club_coefficient, eigenvector_centrality, communicability_centrality
    from pynets.netstats import efficiency, global_efficiency, local_efficiency, create_random_graph, smallworldness_measure, smallworldness, modularity
    ##For non-nodal scalar metrics from networkx.algorithms library, add the name of the function to metric_list for it to be automatically calculated.
    ##For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the function  (with a G-only input) to the netstats module.
    #metric_list = [global_efficiency, local_efficiency, smallworldness, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]
    metric_list = [global_efficiency, local_efficiency, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]

    ##Iteratively run functions from above metric list
    num_mets = len(metric_list)
    net_met_arr = np.zeros([num_mets, 2], dtype='object')
    j=0
    for i in metric_list:
        met_name = str(i).split('<function ')[1].split(' at')[0]
        if NETWORK != None:
            net_met = NETWORK + '_' + met_name
        else:
            net_met = met_name
        try:
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

    ##Calculate modularity using the Louvain algorithm
    [community_aff, modularity] = modularity(mat_wei)

    ##betweenness_centrality
    try:
        bc_vector = betweenness_centrality(G_len)
        print('Extracting Betweeness Centrality vector for all network nodes...')
        bc_vals = list(bc_vector.values())
        bc_nodes = list(bc_vector.keys())
        num_nodes = len(bc_nodes)
        bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if NETWORK != None:
                bc_arr[j,0] = NETWORK + '_' + str(bc_nodes[j]) + '_betw_cent'
                print('\n' + NETWORK + '_' + str(bc_nodes[j]) + '_betw_cent')
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
        bc_arr[num_nodes,0] = NETWORK + '_MEAN_betw_cent'
        nonzero_arr_betw_cent = np.delete(bc_arr[:,1], [0])
        bc_arr[num_nodes,1] = np.mean(nonzero_arr_betw_cent)
        print('\n' + 'Mean Betweenness Centrality across all nodes: ' + str(bc_arr[num_nodes,1]) + '\n')
    except:
        print('Betweeness Centrality calculation failed. Skipping...')
        bc_val_list = []
        pass

    ##eigenvector_centrality
    try:
        ec_vector = eigenvector_centrality(G_len)
        print('Extracting Eigenvector Centrality vector for all network nodes...')
        ec_vals = list(ec_vector.values())
        ec_nodes = list(ec_vector.keys())
        num_nodes = len(ec_nodes)
        ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if NETWORK != None:
                ec_arr[j,0] = NETWORK + '_' + str(ec_nodes[j]) + '_eig_cent'
                print('\n' + NETWORK + '_' + str(ec_nodes[j]) + '_eig_cent')
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
        ec_arr[num_nodes,0] = NETWORK + '_MEAN_eig_cent'
        nonzero_arr_eig_cent = np.delete(ec_arr[:,1], [0])
        ec_arr[num_nodes,1] = np.mean(nonzero_arr_eig_cent)
        print('\n' + 'Mean Eigenvector Centrality across all nodes: ' + str(ec_arr[num_nodes,1]) + '\n')
    except:
        print('Eigenvector Centrality calculation failed. Skipping...')
        ec_val_list = []
        pass

    ##communicability_centrality
    try:
        cc_vector = communicability_centrality(G_len)
        print('Extracting Communicability Centrality vector for all network nodes...')
        cc_vals = list(cc_vector.values())
        cc_nodes = list(cc_vector.keys())
        num_nodes = len(cc_nodes)
        cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
        j=0
        for i in range(num_nodes):
            if NETWORK != None:
                cc_arr[j,0] = NETWORK + '_' + str(cc_nodes[j]) + '_comm_cent'
                print('\n' + NETWORK + '_' + str(cc_nodes[j]) + '_comm_cent')
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
        cc_arr[num_nodes,0] = NETWORK + '_MEAN_comm_cent'
        nonzero_arr_comm_cent = np.delete(cc_arr[:,1], [0])
        cc_arr[num_nodes,1] = np.mean(nonzero_arr_comm_cent)
        print('\n' + 'Mean Communicability Centrality across all nodes: ' + str(cc_arr[num_nodes,1]) + '\n')
    except:
        print('Communicability Centrality calculation failed. Skipping...')
        cc_val_list = []
        pass

    ##rich_club_coefficient
    try:
        rc_vector = rich_club_coefficient(G, normalized=True)
        print('Extracting Rich Club Coefficient vector for all network nodes...')
        rc_vals = list(rc_vector.values())
        rc_edges = list(rc_vector.keys())
        num_edges = len(rc_edges)
        rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
        j=0
        for i in range(num_edges):
            if NETWORK != None:
                rc_arr[j,0] = NETWORK + '_' + str(rc_edges[j]) + '_rich_club'
                print('\n' + NETWORK + '_' + str(rc_edges[j]) + '_rich_club')
            else:
                cc_arr[j,0] = 'WholeBrain_' + str(rc_nodes[j]) + '_rich_club'
                print('\n' + 'WholeBrain_' + str(rc_nodes[j]) + '_rich_club')
            try:
                rc_arr[j,1] = rc_vals[j]
            except:
                rc_arr[j,1] = np.nan
            print(str(rc_vals[j]))
            j = j + 1
        ##Add mean
        rc_val_list = list(rc_arr[:,1])
        rc_arr[num_edges,0] = NETWORK + '_MEAN_rich_club'
        nonzero_arr_rich_club = np.delete(rc_arr[:,1], [0])
        rc_arr[num_edges,1] = np.mean(nonzero_arr_rich_club)
        print('\n' + 'Mean Rich Club Coefficient across all edges: ' + str(rc_arr[num_edges,1]) + '\n')
    except:
        print('Rich Club calculation failed. Skipping...')
        rc_val_list = []
        pass

    ##Create a list of metric names for scalar metrics
    metric_list_names = []
    net_met_val_list_final = net_met_val_list
    for i in net_met_arr[:,0]:
        metric_list_names.append(i)

    ##Add modularity measure
    try:
        if NETWORK != None:
            metric_list_names.append(NETWORK + '_Modularity')
        else:
            metric_list_names.append('WholeBrain_Modularity')
        net_met_val_list_final.append(modularity)
    except:
        pass

    ##Add centrality and rich club measures
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
    if NETWORK != None:
        met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/met_list_pickle_' + NETWORK
    else:
        met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/met_list_pickle_WB'
    cPickle.dump(metric_list_names, open(met_list_picke_path, 'wb'))

    ##Save results to csv
    if 'inv' in est_path1:
        if NETWORK != None:
            out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_sps_cov_' + str(thr) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_mets_sps_cov_' + str(thr) + '.csv'
    else:
        if NETWORK != None:
            out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_corr_' + str(thr) + '.csv'
        else:
            out_path = dir_path + '/' + ID + '_net_mets_corr_' + str(thr) + '.csv'
    np.savetxt(out_path, net_met_val_list_final)

    return(out_path)
