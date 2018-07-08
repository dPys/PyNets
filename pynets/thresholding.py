# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np
import networkx as nx


def threshold_absolute(W, thr, copy=True):
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < thr] = 0
    return W


def threshold_proportional(W, p, copy=True):
    '''# Adapted from bctpy
    '''
    if p > 1 or p < 0:
        raise ValueError('Threshold must be in range [0,1]')
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
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W


def density_thresholding(conn_matrix, thr):
    from pynets import thresholding
    work_thr = 0.0
    conn_matrix = thresholding.normalize(conn_matrix)
    np.fill_diagonal(conn_matrix, 0)
    i = 1
    thr_max = 0.50
    G = nx.from_numpy_matrix(conn_matrix)
    density = nx.density(G)
    while float(work_thr) <= float(thr_max) and float(density) > float(thr):
        work_thr = float(work_thr) + float(0.01)
        conn_matrix = thresholding.threshold_proportional(conn_matrix, work_thr)
        G = nx.from_numpy_matrix(conn_matrix)
        density = nx.density(G)
        print("%s%d%s%.2f%s%.2f%s" % ('Iteratively thresholding -- Iteration ', i, ' -- with thresh: ', float(work_thr), ' and Density: ', float(density), '...'))
        i = i + 1
    return conn_matrix


# Calculate density
def est_density(func_mat):
    '''# Adapted from bctpy
    '''
    fG=nx.from_numpy_matrix(func_mat)
    density=nx.density(fG)
    return density


def thr2prob(W, copy=True):
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W[W < 0.001] = 0
    return W


def binarize(W, copy=True):
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def invert(W, copy=False):
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return W


def weight_conversion(W, wcm, copy=True):
    '''# Adapted from bctpy
    '''
    if wcm == 'binarize':
        return binarize(W, copy)
    elif wcm == 'lengths':
        return invert(W, copy)


def autofix(W, copy=True):
    '''# Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    # zero diagonal
    np.fill_diagonal(W, 0)
    # remove np.inf and np.nan
    try:
        W[np.logical_or(np.where(np.isinf(W)), np.where(np.isnan(W)))] = 0
    except:
        print('Failed to removed undefined entries!')
        pass
    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimals=5)
    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W


def weight_to_distance(G):
    """
    inverts all the edge weights so they become equivalent to distance measure.
    """
    edge_list = [v[2]['weight'] for v in G.edges(data=True)]
    # maximum edge value
    emax = np.max(edge_list) + 1 / float(G.number_of_nodes())
    for edge in G.edges():
        G.edges[edge[0], edge[1]]['distance'] = emax - G.edges[edge[0], edge[1]]['weight']

    return G


def knn(conn_matrix, k):
    """
    Creating a k-nearest neighbour graph
    """
    gra = nx.Graph()
    nodes = list(range(len(conn_matrix[0])))
    gra.add_nodes_from(nodes)
    for i in nodes:
        line = np.ma.masked_array(conn_matrix[i, :], mask=np.isnan(conn_matrix[i]))
        line.mask[i] = True
        for _ in range(k):
            node = np.argmax(line)
            if not np.isnan(conn_matrix[i, node]):
                gra.add_edge(i, node)

            line.mask[node] = True

    return gra


def local_thresholding_prop(conn_matrix, thr):
    from pynets import netstats, thresholding
    '''
    Threshold the adjacency matrix by building from the minimum spanning tree (MST) and adding
    successive N-nearest neighbour degree graphs to achieve target proportional threshold.
    '''

    fail_tol = 10
    G = nx.from_numpy_matrix(conn_matrix)
    if not nx.is_connected(G):
        [G, _] = netstats.prune_disconnected(G)

    maximum_edges = G.number_of_edges()
    G = thresholding.weight_to_distance(G)
    min_t = nx.minimum_spanning_tree(G, weight="distance")
    len_edges = min_t.number_of_edges()
    upper_values = np.triu_indices(np.shape(conn_matrix)[0], k=1)
    weights = np.array(conn_matrix[upper_values])
    weights = weights[~np.isnan(weights)]
    edgenum = int(thr * len(weights))

    if len_edges > edgenum:
        print("%s%s%s" % ('Warning: The minimum spanning tree already has: ', len_edges, ' edges, select more edges. Local Threshold will be applied by just retaining the Minimum Spanning Tree'))
        conn_matrix_thr = nx.to_numpy_array(G)
        return conn_matrix_thr

    k = 1
    len_edge_list = []
    while len_edges < edgenum and k <= np.shape(conn_matrix)[0] and (len(len_edge_list[-fail_tol:]) - len(set(len_edge_list[-fail_tol:]))) < (fail_tol-1):
        print(k)
        print(len_edges)
        len_edge_list.append(len_edges)
        # Create nearest neighbour graph
        nng = thresholding.knn(conn_matrix, k)
        number_before = nng.number_of_edges()
        # Remove edges from the NNG that exist already in the new graph/MST
        nng.remove_edges_from(min_t.edges())
        if nng.number_of_edges() == 0 and number_before >= maximum_edges:
            break

        # Add weights to NNG
        for e in nng.edges():
            nng.edges[e[0], e[1]]['weight'] = float(conn_matrix[e[0], e[1]])

        # Obtain list of edges from the NNG in order of weight
        edge_list = sorted(nng.edges(data=True), key=lambda t: t[2]['weight'], reverse=True)
        # Add edges in order of connectivity strength
        for edge in edge_list:
            #print("%s%s" % ('Adding edge to mst: ', edge))
            min_t.add_edges_from([edge])
            min_t_mx = nx.to_numpy_array(min_t)
            len_edges = nx.from_numpy_matrix(min_t_mx).number_of_edges()
            if len_edges >= edgenum:
                #print(len_edges)
                break

        print('\n')
        if (len(len_edge_list[-fail_tol:]) - len(set(len_edge_list[-fail_tol:]))) >= (fail_tol-1):
            print("%s%s%s" % ('Cannot apply local thresholding to achieve threshold of: ', thr, '. Using maximally saturated connected matrix instead...'))

        k += 1

    conn_matrix_thr = nx.to_numpy_array(min_t, nodelist=sorted(min_t.nodes()), dtype=np.float64)

    if len(min_t.nodes()) < conn_matrix.shape[0]:
        raise RuntimeWarning("%s%s%s" % ('Cannot apply local thresholding to achieve threshold of: ', thr, '. Try a higher -thr or -min_thr'))

    return conn_matrix_thr


def local_thresholding_dens(conn_matrix, thr):
    from pynets import netstats, thresholding
    '''
    Threshold the adjacency matrix by building from the minimum spanning tree (MST) and adding
    successive N-nearest neighbour degree graphs to achieve target density.
    '''

    fail_tol = 10

    G = nx.from_numpy_matrix(conn_matrix)
    if not nx.is_connected(G):
        [G, _] = netstats.prune_disconnected(G)

    maximum_edges = G.number_of_edges()
    G = thresholding.weight_to_distance(G)
    min_t = nx.minimum_spanning_tree(G, weight="distance")
    mst_density = nx.density(min_t)
    G_density = nx.density(G)

    if mst_density > G_density:
        print("%s%s%s" % ('Warning: The minimum spanning tree already has: ', thr, ' density. Local Threshold will be applied by just retaining the Minimum Spanning Tree'))
        conn_matrix_thr = nx.to_numpy_array(G)
        return conn_matrix_thr

    k = 1
    dense_list = []
    while mst_density < float(thr) and (len(dense_list[-fail_tol:]) - len(set(dense_list[-fail_tol:]))) < (fail_tol - 1):
        print(k)
        print(mst_density)
        dense_list.append(mst_density)
        # Create nearest neighbour graph
        nng = thresholding.knn(conn_matrix, k)
        number_before = nng.number_of_edges()
        # Remove edges from the NNG that exist already in the new graph/MST
        nng.remove_edges_from(min_t.edges())
        if nng.number_of_edges() == 0 and number_before >= maximum_edges:
            break

        # Add weights to NNG
        for e in nng.edges():
            nng.edges[e[0], e[1]]['weight'] = float(conn_matrix[e[0], e[1]])

        # Obtain list of edges from the NNG in order of weight
        edge_list = sorted(nng.edges(data=True), key=lambda t: t[2]['weight'], reverse=True)
        # Add edges in order of connectivity strength
        for edge in edge_list:
            min_t.add_edges_from([edge])
            mst_density = thresholding.est_density((nx.to_numpy_array(min_t)))
            #print("%s%s" % ('Adding edge to mst: ', edge))
            if mst_density >= G_density or mst_density >= float(thr):
                #print(mst_density)
                break
        print('\n')
        if (len(dense_list[-fail_tol:]) - len(set(dense_list[-fail_tol:]))) >= (fail_tol - 1):
            print("%s%s%s" % ('Cannot apply local thresholding to achieve density of: ', thr, '. Using maximally saturated connected matrix instead...'))

        k += 1

    conn_matrix_thr = nx.to_numpy_array(min_t, nodelist=sorted(min_t.nodes()), dtype=np.float64)
    if len(min_t.nodes()) < conn_matrix.shape[0]:
        raise RuntimeWarning("%s%s%s" % ('Cannot apply local thresholding to achieve density of: ', thr, '. Try a higher -thr or -min_thr'))

    return conn_matrix_thr


def thresh_and_fit(dens_thresh, thr, ts_within_nodes, conn_model, network, ID, dir_path, mask, node_size, min_span_tree):
    from pynets import utils, thresholding, graphestimation

    thr_perc = 100 * float(thr)
    edge_threshold = "%s%s" % (str(thr_perc), '%')

    # Fit mat
    conn_matrix = graphestimation.get_conn_matrix(ts_within_nodes, conn_model)

    # Save unthresholded
    unthr_path = utils.create_unthr_path(ID, network, conn_model, mask, dir_path)
    np.save(unthr_path, conn_matrix)

    if min_span_tree is True:
        print('Using local thresholding option from the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            conn_matrix_thr = thresholding.local_thresholding_prop(conn_matrix, thr)
        else:
            conn_matrix_thr = thresholding.local_thresholding_dens(conn_matrix, thr)
    else:
        if dens_thresh is False:
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('WARNING: Fragmented graph!')

    # Save thresholded mat
    est_path = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
    np.save(est_path, conn_matrix_thr)
    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network


def thresh_diff(dens_thresh, thr, conn_model, network, ID, dir_path, mask, node_size, conn_matrix, parc, min_span_tree):
    from pynets import utils, thresholding

    thr_perc = 100 * float(thr)
    edge_threshold = "%s%s" % (str(thr_perc), '%')
    if parc is True:
        node_size = 'parc'

    if min_span_tree is True:
        print('Using local thresholding option from the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            conn_matrix_thr = thresholding.local_thresholding_prop(conn_matrix, thr)
        else:
            conn_matrix_thr = thresholding.local_thresholding_dens(conn_matrix, thr)
    else:
        if dens_thresh is False:
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('WARNING: Fragmented graph!')

    # Save thresholded mat
    est_path = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size)
    np.save(est_path, conn_matrix_thr)
    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network
