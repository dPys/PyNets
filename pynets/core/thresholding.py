#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import networkx as nx
warnings.filterwarnings("ignore")


def threshold_absolute(W, thr, copy=True):
    '''
    This function thresholds the connectivity matrix by absolute weight
    magnitude. All weights below the given threshold, and all weights
    on the main diagonal (self-self connections) are set to 0.
    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    thr : float
        absolute weight threshold
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)
    W[W < thr] = 0
    return W


def threshold_proportional(W, p, copy=True):
    '''
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights, and
    all weights on the main diagonal (self-self connections) are set to 0.
    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    p : float
        proportional weight threshold (0<p<1)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy

    Notes
    -----
    The proportion of elements set to 0 is a fraction of all elements
    in the matrix, whether or not they are already 0. That is, this function
    has the following behavior:
    >> x = np.random.random_sample((10,10))
    >> x_25 = threshold_proportional(x, .25)
    >> np.size(np.where(x_25)) #note this double counts each nonzero element
    46
    >> x_125 = threshold_proportional(x, .125)
    >> np.size(np.where(x_125))
    22
    >> x_test = threshold_proportional(x_25, .5)
    >> np.size(np.where(x_test))
    46
    That is, the 50% thresholding of x_25 does nothing because >=50% of the
    elements in x_25 are aleady <=0. This behavior is the same as in BCT. Be
    careful with matrices that are both signed and sparse.
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
    if ud == 2:
        W[:, :] = W + W.T
    return W


def normalize(W):
    '''
    Normalizes an input weighted connection matrix.

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix

    Returns
    -------
    W : np.ndarray
        normalized connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    W /= np.max(np.abs(W))
    return W


def standardize(W):
    '''
    Normalizes an input weighted connection matrix [0, 1]

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix

    Returns
    -------
    W : np.ndarray
        standardized connectivity matrix
    '''
    W = (W - np.min(W)) / np.ptp(W)
    return W


def density_thresholding(conn_matrix, thr, max_iters=10000, interval=0.01):
    """
    Iteratively apply an absolute threshold to achieve a target density.

    Parameters
    ----------
    conn_matrix : np.ndarray
        Weighted connectivity matrix
    thr : float
        Density value between 0-1.
    max_iters : int
        Maximum number of iterations for performing absolute thresholding. Default is 1000.
    interval : float
        Interval for increasing the absolute threshold for each iteration. Default is 0.01.

    Returns
    -------
    conn_matrix : np.ndarray
        Thresholded connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy
    """
    from pynets.core import thresholding
    np.fill_diagonal(conn_matrix, 0)

    work_thr = 0
    i = 1
    density = nx.density(nx.from_numpy_matrix(conn_matrix))
    if float(thr) < float(density):
        while float(i) < max_iters and float(work_thr) < float(1):
            work_thr = float(work_thr) + float(interval)
            density = nx.density(nx.from_numpy_matrix(thresholding.threshold_absolute(conn_matrix, work_thr)))
            print("%s%d%s%.2f%s%.2f%s" % ('Iteration ', i, ' -- with Thresh: ', float(work_thr), ' and Density: ',
                                          float(density), '...'))
            if float(thr) >= float(density):
                conn_matrix = thresholding.threshold_absolute(conn_matrix, work_thr)
                break
            i = i + 1
    else:
        print('Density of raw matrix is already greater than or equal to the target density requested')

    return conn_matrix


# Calculate density
def est_density(in_mat):
    """
    Calculates the density of a given graph.

    Parameters
    ----------
    in_mat : NxN np.ndarray
        weighted connectivity matrix.

    Returns
    -------
    density : float
        Density of the graph.
    """
    fG = nx.from_numpy_matrix(in_mat)
    density = nx.density(fG)
    return density


def thr2prob(W, copy=True):
    """
    Thresholds the near-zero ranks of a ranked graph.

    Parameters
    ----------
    W : NxN np.ndarray
        Weighted connectivity matrix of ranks.

    Returns
    -------
    W : NxN np.ndarray
        Weighted connectivity matrix of ranks with no near-zero entries.

    References
    ----------
    .. Adapted from Adapted from bctpy
    """
    if copy:
        W = W.copy()
    W[W < 0.001] = 0
    return W


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def invert(W, copy=False):
    '''
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.
    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]

    return W


def weight_conversion(W, wcm, copy=True):
    '''
    W_bin = weight_conversion(W, 'binarize');
    W_nrm = weight_conversion(W, 'normalize');
    L = weight_conversion(W, 'lengths');
    This function may either binarize an input weighted connection matrix,
    normalize an input weighted connection matrix or convert an input
    weighted connection matrix to a weighted connection-length matrix.
    Binarization converts all present connection weights to 1.
    Normalization scales all weight magnitudes to the range [0,1] and
    should be done prior to computing some weighted measures, such as the
    weighted clustering coefficient.
    Conversion of connection weights to connection lengths is needed
    prior to computation of weighted distance-based measures, such as
    distance and betweenness centrality. In a weighted connection network,
    higher weights are naturally interpreted as shorter lengths. The
    connection-lengths matrix here is defined as the inverse of the
    connection-weights matrix.
    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    wcm : str
        weight conversion command.
        'binarize' : binarize weights
        'normalize' : normalize weights
        'lengths' : convert weights to lengths (invert matrix)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        connectivity matrix with specified changes

    References
    ----------
    .. Adapted from Adapted from bctpy

    Notes
    -----
    This function is included for compatibility with BCT. But there are
    other functions binarize(), normalize() and invert() which are simpler to
    call directly.
    '''
    if wcm == 'binarize':
        return binarize(W, copy)
    elif wcm == 'lengths':
        return invert(W, copy)


def autofix(W, copy=True):
    '''
    Fix a bunch of common problems. More specifically, remove Inf and NaN,
    ensure exact binariness and symmetry (i.e. remove floating point
    instability), and zero diagonal.

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix.
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : np.ndarray
        connectivity matrix with fixes applied.

    References
    ----------
    .. Adapted from Adapted from bctpy
    '''
    if copy:
        W = W.copy()
    # zero diagonal
    np.fill_diagonal(W, 0)
    # remove np.inf and np.nan
    W[np.where(np.isinf(W))] = 0
    W[np.where(np.isnan(W))] = 0

    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimals=5)
    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W


def disparity_filter(G, weight='weight'):
    """
    Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009.

    Parameters
    ----------
    G : Object
        Weighted NetworkX graph.

    weight : str
        Key for edge data used as the edge weight w_ij. Default is 'weight'.

    Returns
    -------
    B : Object
        Weighted NetworkX graph with a significance score (alpha) assigned to each edge.

    References
    ----------
    .. [1] M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks.
       PNAS, 106:16, pp. 6483-6488.
    """
    from scipy import integrate

    if nx.is_directed(G):  # directed case
        N = nx.DiGraph()
        for u in G:

            k_out = G.out_degree(u)
            k_in = G.in_degree(u)

            if k_out > 1:
                sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
                for v in G.successors(u):
                    w = G[u][v][weight]
                    p_ij_out = float(np.absolute(w)) / sum_w_out
                    alpha_ij_out = 1 - (k_out - 1) * integrate.quad(lambda x: (1 - x) ** (k_out - 2), 0, p_ij_out)[0]
                    N.add_edge(u, v, weight=w, alpha_out=float('%.4f' % alpha_ij_out))

            elif k_out == 1 and G.in_degree(list(G.successors(u))[0]) == 1:
                # we need to keep the connection as it is the only way to maintain the connectivity of the network
                v = list(G.successors(u))[0]
                w = G[u][v][weight]
                N.add_edge(u, v, weight=w, alpha_out=0., alpha_in=0.)
                # there is no need to do the same for the k_in, since the link is built already from the tail

            if k_in > 1:
                sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
                for v in G.predecessors(u):
                    w = G[v][u][weight]
                    p_ij_in = float(np.absolute(w)) / sum_w_in
                    alpha_ij_in = 1 - (k_in - 1) * integrate.quad(lambda x: (1 - x) ** (k_in - 2), 0, p_ij_in)[0]
                    N.add_edge(v, u, weight=w, alpha_in=float('%.4f' % alpha_ij_in))
        return N

    else:  # undirected case
        B = nx.Graph()
        for u in G:
            k = len(G[u])
            if k > 1:
                sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
                for v in G[u]:
                    w = G[u][v][weight]
                    p_ij = float(np.absolute(w)) / sum_w
                    alpha_ij = 1 - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
                    B.add_edge(u, v, weight=w, alpha=float('%.4f' % alpha_ij))
            else:
                B.add_node(u)
        return B


def disparity_filter_alpha_cut(G, weight='weight', alpha_t=0.4, cut_mode='or'):
    """
    Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009.

    Parameters
    ----------
    G : Object
        Weighted NetworkX graph.

    weight : str
        Key for edge data used as the edge weight w_ij. Default is 'weight'.

    alpha_t : float
            The threshold, between 0 and 1, for the alpha parameter
            used to select the surviving edges. Default is 0.4.

    cut_mode : str
            In the case of directed graphs. It represents the logic operation to filter out edges
            that do not pass the threshold value, combining the alpha_in and alpha_out attributes
            resulting from the disparity_filter function. Default is 'or'. Possible strings: 'or', 'and'.
    Returns
    -------
    B : Object
        Weighted NetworkX graph with a significance score (alpha) assigned to each edge. The resulting
        graph contains only edges that survived from the filtering with the alpha_t threshold.

    References
    ----------
    .. [1] M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks.
       PNAS, 106:16, pp. 6483-6488.
    """

    if nx.is_directed(G):  # Directed case:
        B = nx.DiGraph()
        for u, v, w in G.edges(data=True):
            try:
                alpha_in = w['alpha_in']
            except KeyError:  # there is no alpha_in, so we assign 1. It will never pass the cut
                alpha_in = 1
            try:
                alpha_out = w['alpha_out']
            except KeyError:  # there is no alpha_out, so we assign 1. It will never pass the cut
                alpha_out = 1

            if cut_mode == 'or':
                if alpha_in < alpha_t or alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
            elif cut_mode == 'and':
                if alpha_in < alpha_t and alpha_out < alpha_t:
                    B.add_edge(u, v, weight=w[weight])
        return B

    else:
        B = nx.Graph()  # Undirected case:
        for u, v, w in G.edges(data=True):
            try:
                alpha = w['alpha']
            except KeyError:  # there is no alpha, so we assign 1. It will never pass the cut
                alpha = 1

            if alpha < alpha_t:
                B.add_edge(u, v, weight=w[weight])

        return B


def weight_to_distance(G):
    """
    Inverts all the edge weights so they become equivalent to distance measure.
    With a weight, the higher the value the stronger the connection. With a distance,
    the higher the value the "weaker" the connection. In this case there is no
    measurement unit for the distance, as it is just a conversion from the weights.
    The distances can be accessed in each node's property with constants.

    Parameters
    ----------
    G : Object
        Weighted NetworkX graph.

    Returns
    -------
    G : Object
        Inverted NetworkX graph equivalent to the distance measure.
    """
    edge_list = [v[2]['weight'] for v in G.edges(data=True)]
    # maximum edge value
    emax = np.max(edge_list) + 1 / float(G.number_of_nodes())
    for edge in G.edges():
        G.edges[edge[0], edge[1]]['distance'] = emax - G.edges[edge[0], edge[1]]['weight']

    return G


def knn(conn_matrix, k):
    """
    Creates a k-nearest neighbour graph.

    Parameters
    ----------
    conn_matrix : array
        Weighted NxN matrix.
    k : int
        Number of nearest neighbours to include in the knn estimation.

    Returns
    -------
    gra : Obj
        KNN Weighted NetworkX graph.
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


def local_thresholding_prop(conn_matrix, coords, labels, thr):
    """
    Threshold the adjacency matrix by building from the minimum spanning tree (MST) and adding
    successive N-nearest neighbour degree graphs to achieve target proportional threshold.

    Parameters
    ----------
    conn_matrix : array
        Weighted NxN matrix.
    thr : float
        A proportional threshold, between 0 and 1, to achieve through local thresholding.

    Returns
    -------
    conn_matrix_thr : array
        Weighted, MST local-thresholded, NxN matrix.
    """
    from pynets.core import thresholding
    from pynets.stats import netstats

    fail_tol = 10
    conn_matrix = np.nan_to_num(conn_matrix)
    G = nx.from_numpy_matrix(conn_matrix)
    if not nx.is_connected(G):
        [G, pruned_nodes] = netstats.prune_disconnected(G)
        pruned_nodes.sort(reverse=True)
        coords_pre = list(coords)
        labels_pre = list(labels)
        if len(pruned_nodes) > 0:
            for j in pruned_nodes:
                labels_pre.pop(j)
                coords_pre.pop(j)
            conn_matrix = nx.to_numpy_array(G)
            labels = labels_pre
            coords = coords_pre

    maximum_edges = G.number_of_edges()
    G = thresholding.weight_to_distance(G)
    min_t = nx.minimum_spanning_tree(G, weight="distance")
    len_edges = min_t.number_of_edges()
    upper_values = np.triu_indices(np.shape(conn_matrix)[0], k=1)
    weights = np.array(conn_matrix[upper_values])
    weights = weights[~np.isnan(weights)]
    edgenum = int(float(thr) * float(len(weights)))
    if len_edges > edgenum:
        print("%s%s%s" % ('Warning: The minimum spanning tree already has: ', len_edges,
                          ' edges, select more edges. Local Threshold will be applied by just retaining the Minimum '
                          'Spanning Tree'))
        conn_matrix_thr = nx.to_numpy_array(G)
        return conn_matrix_thr, coords, labels

    k = 1
    len_edge_list = []
    while len_edges < edgenum and k <= np.shape(conn_matrix)[0] and (len(len_edge_list[-fail_tol:]) -
                                                                     len(set(len_edge_list[-fail_tol:]))) < (fail_tol -
                                                                                                             1):
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
            # print("%s%s" % ('Adding edge to mst: ', edge))
            min_t.add_edges_from([edge])
            min_t_mx = nx.to_numpy_array(min_t)
            len_edges = nx.from_numpy_matrix(min_t_mx).number_of_edges()
            if len_edges >= edgenum:
                # print(len_edges)
                break
        k += 1

    conn_matrix_thr = nx.to_numpy_array(min_t, nodelist=sorted(min_t.nodes()), dtype=np.float64)

    return conn_matrix_thr, coords, labels


def thresh_func(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                smooth, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                hpass):
    """
    Threshold a functional connectivity matrix using any of a variety of methods.

    Parameters
    ----------
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for thresholding.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : float
        High-pass filter values (Hz) to apply to node-extracted time-series.

    Returns
    -------
    conn_matrix_thr : array
        Weighted, thresholded, NxN matrix.
    edge_threshold : str
        The string percentage representation of thr.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : float
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    from pynets.core import utils, thresholding

    thr_perc = 100 * float(thr)
    if parc is True:
        node_size = 'parc'

    if np.count_nonzero(conn_matrix) == 0:
        raise ValueError('ERROR: Raw connectivity matrix contains only zeros.')

    # Save unthresholded
    utils.save_mat(conn_matrix, utils.create_raw_path_func(ID, network, conn_model, roi, dir_path, node_size, smooth,
                                                           c_boot, hpass, parc))

    if min_span_tree is True:
        print('Using local thresholding option with the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            print('Ignoring -dt flag since local density thresholding is not currently supported.')
        thr_type = 'MST_thr'
        edge_threshold = "%s%s" % (str(np.abs(1 - thr_perc)), '%')
        [conn_matrix_thr, coords, labels] = thresholding.local_thresholding_prop(conn_matrix, coords, labels, thr)
    elif disp_filt is True:
        thr_type = 'DISP_alpha'
        edge_threshold = "%s%s" % (str(np.abs(1 - thr_perc)), '%')
        G1 = thresholding.disparity_filter(nx.from_numpy_array(conn_matrix))
        # G2 = nx.Graph([(u, v, d) for u, v, d in G1.edges(data=True) if d['alpha'] < thr])
        print('Computing edge disparity significance with alpha = %s' % thr)
        print('Filtered graph: nodes = %s, edges = %s' % (G1.number_of_nodes(), G1.number_of_edges()))
        # print('Backbone graph: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))
        # print(G2.edges(data=True))
        conn_matrix_thr = nx.to_numpy_array(G1)
    else:
        if dens_thresh is False:
            thr_type = 'prop'
            edge_threshold = "%s%s" % (str(np.abs(thr_perc)), '%')
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            thr_type = 'dens'
            edge_threshold = None
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('Warning: Fragmented graph')

    # Save thresholded mat
    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot,
                                          thr_type, hpass, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, smooth, prune, ID, dir_path, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass


def thresh_struct(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                  disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary,
                  target_samples, track_type, atlas_mni, streams, directget):
    """
    Threshold a structural connectivity matrix using any of a variety of methods.

    Parameters
    ----------
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).

    Returns
    -------
    conn_matrix_thr : array
        Weighted, thresholded, NxN matrix.
    edge_threshold : str
        The string percentage representation of thr.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    """
    from pynets.core import utils, thresholding

    thr_perc = 100 * float(thr)
    if parc is True:
        node_size = 'parc'

    if np.count_nonzero(conn_matrix) == 0:
        raise ValueError('ERROR: Raw connectivity matrix contains only zeros.')

    # Save unthresholded
    utils.save_mat(conn_matrix, utils.create_raw_path_diff(ID, network, conn_model, roi, dir_path, node_size,
                                                           target_samples, track_type, parc))

    if min_span_tree is True:
        print('Using local thresholding option with the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            print('Ignoring -dt flag since local density thresholding is not currently supported.')
        thr_type = 'MST_thr'
        edge_threshold = "%s%s" % (str(np.abs(1 - thr_perc)), '%')
        [conn_matrix_thr, coords, labels] = thresholding.local_thresholding_prop(conn_matrix, coords, labels, thr)
    elif disp_filt is True:
        thr_type = 'DISP_alpha'
        edge_threshold = "%s%s" % (str(np.abs(1 - thr_perc)), '%')
        G1 = thresholding.disparity_filter(nx.from_numpy_array(conn_matrix))
        # G2 = nx.Graph([(u, v, d) for u, v, d in G1.edges(data=True) if d['alpha'] < thr])
        print('Computing edge disparity significance with alpha = %s' % thr)
        print('Filtered graph: nodes = %s, edges = %s' % (G1.number_of_nodes(), G1.number_of_edges()))
        # print('Backbone graph: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))
        # print(G2.edges(data=True))
        conn_matrix_thr = nx.to_numpy_array(G1)
    else:
        if dens_thresh is False:
            thr_type = 'prop'
            edge_threshold = "%s%s" % (str(np.abs(thr_perc)), '%')
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            thr_type = 'dens'
            edge_threshold = "%s%s" % (str(np.abs(1 - thr_perc)), '%')
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('Warning: Fragmented graph')

    # Save thresholded mat
    est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples,
                                          track_type, thr_type, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, prune, ID, dir_path, atlas, uatlas, labels, coords, norm, binary, target_samples, track_type, atlas_mni, streams, directget
