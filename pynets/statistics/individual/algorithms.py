#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner
"""
import matplotlib
from pynets.core.utils import load_runconfig
import numpy as np
import warnings
import networkx as nx
from pynets.core.utils import timeout

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


try:
    hardcoded_params = load_runconfig()
    DEFAULT_TIMEOUT = hardcoded_params["graph_analysis_timeout"][0]
    DEFAULT_ENGINE = hardcoded_params["graph_analysis_engine"][0]
except FileNotFoundError as e:
    import sys
    print(e, "Failed to parse advanced.yaml")


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    import six

    if isinstance(key, six.string_types):
        key = key.encode('ascii', errors='replace')

    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, six.string_types):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    try:
        import graph_tool.all as gt
    except ImportWarning as e:
        print(e, "Graph Tool not installed!")
    gtG = gt.Graph(directed=nxG.is_directed())

    for key, value in nxG.graph.items():
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)
        gtG.graph_properties[key] = prop
        gtG.graph_properties[key] = value

    nprops = set()
    for node, data in nxG.nodes(data=True):
        for key, val in data.items():
            if key in nprops:
                continue

            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)
            gtG.vertex_properties[key.decode("utf-8")] = prop

            nprops.add(key.decode("utf-8"))

    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    eprops = set()
    for src, dst, data in nxG.edges(data=True):
        for key, val in data.items():
            if key in eprops:
                continue

            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)
            gtG.edge_properties[key.decode("utf-8")] = prop

            eprops.add(key.decode("utf-8"))

    vertices = {}
    for node, data in nxG.nodes(data=True):
        v = gtG.add_vertex()
        vertices[node] = v

        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value

    # Add the edges
    for src, dst, data in nxG.edges(data=True):
        e = gtG.add_edge(vertices[src], vertices[dst])
        for key, value in data.items():
            gtG.ep[key][e] = value

    return gtG


def np2gt(adj):
    try:
        import graph_tool.all as gt
    except ImportError as e:
        print(e, "Graph Tool not installed!")
    g = gt.Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    nnz = np.nonzero(np.triu(adj, 1))
    nedges = len(nnz[0])
    g.add_edge_list(np.hstack([np.transpose(nnz), np.reshape(adj[nnz],
                                                             (nedges, 1))]),
                    eprops=[edge_weights])
    return g


def countmotifs(A, N=4):
    """
    Counts number of motifs with size N from A.

    Parameters
    ----------
    A : ndarray
        M x M Connectivity matrix
    N : int
        Size of motif type. Default is N=4, only 3 or 4 supported.

    Returns
    -------
    umotifs : int
        Total count of size N motifs for graph A.

    References
    ----------
    .. [1] Sporns, O., & Kötter, R. (2004). Motifs in Brain Networks.
      PLoS Biology. https://doi.org/10.1371/journal.pbio.0020369

    """
    from copy import copy
    from collections import Counter

    assert N in [3, 4], "Only motifs of size N=3,4 currently supported"
    X2 = np.array([[k] for k in range(A.shape[0] - 1)])
    for n in range(N - 1):
        X = copy(X2)
        X2 = []
        for vsub in X:
            # in_matind list of nodes neighboring vsub with a larger index than
            # root v
            idx = np.where(np.any(A[(vsub[0] + 1):, vsub], 1))[0] + vsub[0] + 1
            # Only keep node indices not in vsub
            idx = idx[[k not in vsub for k in idx]]
            if len(idx) > 0:
                # If new neighbors found, add all new vsubs to list
                X2.append([np.append(vsub, ik) for ik in idx])
        if len(X2) > 0:
            X2 = np.vstack(X2)
        else:
            return 0

    X2 = np.sort(X2, 1)
    X2 = X2[
        np.unique(
            np.ascontiguousarray(X2).view(
                np.dtype((np.void, X2.dtype.itemsize * X2.shape[1]))
            ),
            return_index=True,
        )[1]
    ]
    return Counter(["".join(np.sort(np.sum(A[x, :][:, x], 1)
                                    ).astype(int).astype(str)) for x in X2])


def average_shortest_path_length_fast(G, weight="weight"):
    try:
        import graph_tool.all as gt
    except ImportWarning as e:
        print(e, "Graph Tool not installed!")
    if type(G) == nx.classes.graph.Graph:
        n = len(G)
        g = nx2gt(G)
    else:
        g = G
        n = len(g.get_vertices())
    if weight == "weight":
        dist = gt.shortest_distance(g, weights=g.edge_properties['weight'],
                                    directed=False)
    else:
        dist = gt.shortest_distance(g, directed=False)
    sum_of_all_dists = sum(
        [sum(i.a[(i.a > 1e-9) & (i.a < 1e9)]) for i in dist])
    return sum_of_all_dists / (n * (n - 1))


@timeout(DEFAULT_TIMEOUT)
def average_shortest_path_length_for_all(G):
    """
    Helper function, in the case of graph disconnectedness,
    that returns the average shortest path length, calculated
    iteratively for each distinct subgraph of the G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    average_shortest_path_length : float
        The length of the average shortest path for G.

    """
    import math

    connected_component_subgraphs = [
        G.subgraph(c) for c in nx.connected_components(G)]
    subgraphs = [sbg for sbg in connected_component_subgraphs if len(sbg) > 1]

    return math.fsum(nx.average_shortest_path_length(
        sg, weight="weight") for sg in subgraphs) / len(subgraphs)


@timeout(DEFAULT_TIMEOUT)
def subgraph_number_of_cliques_for_all(G):
    """
    Helper function, in the case of graph disconnectedness,
    that returns the number of cliques, calculated
    iteratively for each distinct subgraph of the G.

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    number of cliques : int
        The average number of cliques for G.

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
      "Algorithm 457: finding all cliques of an undirected graph".
      *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
      <http://portal.acm.org/citation.cfm?doid=362342.362367>
    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
      "The worst-case time complexity for generating all maximal
      cliques and computational experiments",
      *Theoretical Computer Science*, Volume 363, Issue 1,
      Computing and Combinatorics,
      10th Annual International Conference on
      Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28-42
      <https://doi.org/10.1016/j.tcs.2006.06.015>
    .. [3] F. Cazals, C. Karande,
      "A note on the problem of reporting maximal cliques",
      *Theoretical Computer Science*,
      Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
      <https://doi.org/10.1016/j.tcs.2008.05.010>

    """
    import math

    connected_component_subgraphs = [
        G.subgraph(c) for c in nx.connected_components(G)]
    subgraphs = [sbg for sbg in connected_component_subgraphs if len(sbg) > 1]

    return np.rint(math.fsum(nx.graph_number_of_cliques(sg)
                             for sg in subgraphs) / len(subgraphs))


def global_efficiency(G, weight="weight", engine=DEFAULT_ENGINE):
    """
    Return the global efficiency of the G

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    global_efficiency : float

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted subnet, the scaling factor
    is 1 and can be ignored. In the case of a weighted graph, calculating the
    scaling factor requires somehow knowing the weights of the edges required
    to make a completely connected graph. Since that knowlege may not exist,
    the scaling factor is not included. If that knowlege exists, construct the
    corresponding weighted graph and calculate its global_efficiency to scale
    the weighted graph. Distance between nodes is calculated as the sum of
    weights.
    If the graph is defined such that a higher weight represents a stronger
    connection, distance should be represented by 1/weight. In this case, use
    the invert weights function to generate a graph where the weights are set
    to 1/weight and then calculate efficiency

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
        return np.nan

    if engine.upper() == 'NX' or engine.upper() == 'NETWORKX':
        lengths = list(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    elif engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")
        g = nx2gt(G)
        vertices = list(g.get_vertices())
        all_shortest_dist = [dict(zip(vertices,
                                      list(i))) for
                             i in gt.shortest_distance(
            g, weights=g.edge_properties['weight'], directed=False)]
        lengths = tuple(dict(zip(vertices, all_shortest_dist)).items())
    else:
        raise ValueError(f"Engine {engine} not recognized.")
    inv_lengths = []
    for length in lengths:
        inv = [1 / x for x in length[1].values() if float(x) != float(0)]
        inv_lengths.extend(inv)
    return sum(inv_lengths) / (N * (N - 1))


def local_efficiency(G, weight="weight", engine=DEFAULT_ENGINE):
    """
    Return the local efficiency of each node in the G

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    local_efficiency : dict
       The keys of the dict are the nodes in the G and the corresponding
       values are local efficiencies of each node

    Notes
    -----
    The published definition includes a scale factor based on a completely
    connected graph. In the case of an unweighted subnet, the scaling factor
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
    from graspologic.utils import largest_connected_component

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
                temp_G[n1][n2][weight] = np.abs(G[n1][n2][weight])

        temp_G = largest_connected_component(temp_G, return_inds=False)

        if nx.is_empty(temp_G) is True or len(temp_G) < 2 or \
                nx.number_of_edges(temp_G) == 0:
            efficiencies[node] = 0
        else:
            try:
                if engine.upper() == 'GT' or \
                        engine.upper() == 'GRAPH_TOOL' or \
                        engine.upper() == 'GRAPHTOOL':
                    efficiencies[node] = global_efficiency(temp_G, weight,
                                                           engine='gt')
                else:
                    efficiencies[node] = global_efficiency(temp_G, weight,
                                                           engine='nx')
            except BaseException:
                efficiencies[node] = np.nan
    return efficiencies


@timeout(DEFAULT_TIMEOUT)
def average_local_efficiency(G, weight="weight", engine=DEFAULT_ENGINE):
    """
    Return the average local efficiency of all of the nodes in the G

    Parameters
    ----------
    G : Obj
        NetworkX graph.

    Returns
    -------
    average_local_efficiency : float
        Average local efficiency of G.

    Notes
    -----
    Adapted from NetworkX to incorporate weight parameter.

    References
    ----------
    .. [1] Latora, V., and Marchiori, M. (2001). Efficient behavior of
      small-world networks. Physical Review Letters 87.
    .. [2] Latora, V., and Marchiori, M. (2003). Economic small-world behavior
      in weighted networks. Eur Phys J B 32, 249-263.

    """
    N = len(G)

    if N < 2:
        return np.nan

    if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        eff = local_efficiency(G, weight, engine='gt')
    else:
        eff = local_efficiency(G, weight, engine='nx')

    e_loc_vec = np.array(list(eff.values()))
    e_loc_vec = np.array(e_loc_vec[e_loc_vec != 0.])
    return np.nanmean(e_loc_vec)


@timeout(DEFAULT_TIMEOUT)
def smallworldness(
        G,
        niter=5,
        nrand=10,
        approach="clustering",
        reference="lattice",
        engine=DEFAULT_ENGINE):
    """
    Returns the small-world coefficient of a graph

    The small-world coefficient of a G is:

    omega/sigma = Lr/L - C/Cl

    where C and L are respectively the average clustering
    coefficient/ transitivity and average shortest path length of G. Lr is
    the average shortest path length of an equivalent random graph and Cl is
    the average clustering coefficient/transitivity of an equivalent
    lattice/random graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    approach : str
        Specifies whether to use clustering coefficient directly `clustering`
        or `transitivity` method of counting weighted triangles.
        Default is `clustering`.
    reference : str
        Specifies whether to use a random `random` or lattice
        `lattice` reference. Default is `lattice`.

    Returns
    -------
    omega/sigma : float
        The smallworld coefficient

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
      "The Ubiquity of Small-World Networks".
      Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
      doi:10.1089/brain.2011.0038.

    """
    from networkx.algorithms.smallworld import random_reference, \
        lattice_reference

    N = len(G)

    if N < 2:
        return np.nan

    if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")

    def get_random(G, reference, engine, niter, i):
        nnodes = len(G)
        nedges = nx.number_of_edges(G)
        shape = np.array([nnodes, nnodes])
        if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                engine.upper() == 'GRAPHTOOL':
            if reference == "random":
                def sample_k(max):
                    accept = False
                    while not accept:
                        k = np.random.randint(1, max + 1)
                        accept = np.random.random() < 1.0 / k
                    return k

                G_rand = gt.random_graph(nnodes, lambda: sample_k(nedges),
                                         model="configuration",
                                         directed=False,
                                         n_iter=niter)
            else:
                raise NotImplementedError(f"{reference}' graph type not yet"
                                          f" available using graph_tool "
                                          f"engine")
        else:
            if reference == "random":
                G_rand = random_reference(G, niter=niter, seed=i)
            elif reference == "lattice":
                G_rand = lattice_reference(G, niter=niter, seed=i)
            else:
                raise NotImplementedError(f"{reference}' graph type not "
                                          f"recognized!")
        return G_rand

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}
    for i in range(nrand):
        Gr = get_random(G, "random", engine, niter, i)
        if reference == "lattice":
            Gl = get_random(G, reference, "nx", niter, i)
            if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                    engine.upper() == 'GRAPHTOOL':
                Gl = nx2gt(Gl)
        else:
            Gl = Gr
        if approach == "clustering":
            if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                    engine.upper() == 'GRAPHTOOL':
                clust_coef_ = gt.global_clustering(
                    Gl, weight=Gl.edge_properties['weight'])[0]
            else:
                clust_coef_ = nx.average_clustering(Gl, weight='weight')
            randMetrics["C"].append(clust_coef_)
        elif approach == "transitivity" and engine == 'nx':
            randMetrics["C"].append(weighted_transitivity(Gl))
        else:
            raise ValueError(f"{approach}' approach not recognized!")

        if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                engine.upper() == 'GRAPHTOOL':
            randMetrics["L"].append(
                average_shortest_path_length_fast(Gr, weight=None))
        else:
            randMetrics["L"].append(
                nx.average_shortest_path_length(Gr, weight=None))
        del Gr, Gl

    if approach == "clustering":
        if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                engine.upper() == 'GRAPHTOOL':
            g = nx2gt(G)
            C = gt.global_clustering(g, weight=g.edge_properties['weight'])[0]
        else:
            C = nx.average_clustering(G, weight='weight')
    elif approach == "transitivity" and engine == 'nx':
        C = weighted_transitivity(G)
    else:
        raise ValueError(f"{approach}' approach not recognized!")

    if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        L = average_shortest_path_length_fast(G, weight=None)
    else:
        L = nx.average_shortest_path_length(G, weight=None)

    Cl = np.nanmean(randMetrics["C"], dtype=np.float32)
    Lr = np.nanmean(randMetrics["L"], dtype=np.float32)

    return np.nan_to_num(Lr / L) - np.nan_to_num(C / Cl)


def rich_club_coefficient(G, engine=DEFAULT_ENGINE):
    if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")

        g = nx2gt(G)

        deghist = gt.vertex_hist(g, 'total')[0]
        total = sum(deghist)
        rc = {}
        # Compute the number of nodes with degree greater than `k`, for each
        # degree `k` (omitting the last entry, which is zero).
        nks = (total - cs for cs in np.cumsum(deghist) if total - cs > 1)
        deg = g.degree_property_map('total')
        for k, nk in enumerate(nks):
            if nk == 0:
                continue
            sub_g = gt.GraphView(g, vfilt=lambda v: deg[v] > k)
            ek = sub_g.num_edges()
            rc[k] = 2 * ek / (nk * (nk - 1))
    else:
        from networkx.algorithms import rich_club_coefficient
        rc = rich_club_coefficient(G, seed=42, Q=100)

    return rc


def create_communities(node_comm_aff_mat, node_num):
    """
    Create a 1D vector of community assignments from a community affiliation
    matrix.

    Parameters
    ----------
    node_comm_aff_mat : array
        Community affiliation matrix produced from modularity estimation
        (e.g. Louvain).
    node_num : int
        Number of total connected nodes in the graph used to estimate
        node_comm_aff_mat.

    Returns
    -------
    com_assign : array
        1D numpy vector of community assignments.

    References
    ----------
    .. [1] Newman, M.E.J. & Girvan, M. Finding and evaluating community
      structure in networks. Physical Review E 69, 26113(2004).
    .. [2] Blondel, V.D. et al. Fast unfolding of communities in large
      networks. J. Stat. Mech 10008, 1-12(2008).

    """
    com_assign = np.zeros((node_num, 1))
    for i in range(len(node_comm_aff_mat)):
        community = node_comm_aff_mat[i, :]
        for j in range(len(community)):
            if community[j] == 1:
                com_assign[j, 0] = i
    return com_assign


@timeout(DEFAULT_TIMEOUT)
def participation_coef(W, ci, degree="undirected"):
    """
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
    .. [1] Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of
      complex metabolic networks. Nature, 433, 895-900.
    .. [2] Rubinov, M., & Sporns, O. (2010). Complex subnet measures of brain
      connectivity: Uses and interpretations. NeuroImage, 52, 1059-1069.

    """

    if degree == "in":
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


@timeout(DEFAULT_TIMEOUT)
def participation_coef_sign(W, ci):
    """
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
    .. [1] Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of
      complex metabolic networks. Nature, 433, 895-900.
    .. [2] Rubinov, M., & Sporns, O. (2010). Complex subnet measures of brain
      connectivity: Uses and interpretations. NeuroImage, 52, 1059-1069.

    """
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
    with np.errstate(invalid="ignore"):
        Ppos = pcoef(W * (W > 0))
        Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg


# @timeout(DEFAULTTIMEOUT_)
def diversity_coef_sign(W, ci):
    """
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
    .. [1] Rubinov, M., & Sporns, O. (2010). Complex subnet measures of brain
      connectivity: Uses and interpretations. NeuroImage, 52, 1059-1069.

    """

    def entropy(w_):
        # Strength
        S = np.sum(w_, axis=1)
        # Node-to-module degree
        Snm = np.zeros((n, m))
        for i in range(m):
            Snm[:, i] = np.sum(w_[:, ci == i + 1], axis=1)
        pnm = Snm / np.tile(S, (m, 1)).T
        pnm[np.isnan(pnm)] = 0
        pnm[np.logical_not(pnm)] = 1
        return -np.sum(pnm * np.log(pnm), axis=1) / np.log(m)

    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    # Number of modules
    m = np.max(ci)

    # Explicitly ignore compiler warning for division by zero
    with np.errstate(invalid="ignore"):
        Hpos = entropy(W * (W > 0))
        Hneg = entropy(-W * (W < 0))

    return Hpos, Hneg


def link_communities(W, type_clustering="single"):
    """
    The optimal community structure is a subdivision of the subnet into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This algorithm uncovers overlapping community structure via hierarchical
    clustering of subnet links. This algorithm is generalized for
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
    .. [1] de Reus, M. A., Saenger, V. M., Kahn, R. S., & van den Heuvel,
      M. P. (2014). An edge-centric perspective on the human connectome:
      Link communities in the brain. Philosophical Transactions of the Royal
      Society B: Biological Sciences. https://doi.org/10.1098/rstb.2013.0527

    """

    from pynets.core.thresholding import normalize

    n = len(W)
    W = normalize(W)

    if type_clustering not in ("single", "complete"):
        raise ValueError("Unrecognized clustering type")

    # Set diagonal to mean weights
    np.fill_diagonal(W, 0)
    W[range(n), range(n)] = (
        np.sum(W, axis=0) / np.sum(np.logical_not(W), axis=0)
        + np.sum(W.T, axis=0) / np.sum(np.logical_not(W.T), axis=0)
    ) / 2

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
    A, B = np.where(
        np.logical_and(
            np.logical_or(
                W, W.T), np.triu(
                np.ones(
                    (n, n)), 1)))
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

            ES[i, j] = (W[a, b] * W[a, c] * Ji[b, c] +
                        W[b, a] * W[c, a] * Jo[b, c]) / 2

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
        print(f"Hierarchy {i:d}")
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
            min_mc = np.sum(links[: nc - 1])
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
            (wehr,) = np.where((u1 == u2[0]))
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
            if type_clustering == "single":
                x = np.max(ES[V[j, :], :], axis=0)
            elif type_clustering == "complete":
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


def weighted_transitivity(G):
    r"""
    Compute weighted graph transitivity, the fraction of all possible
    weighted triangles present in G.

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

    References
    ----------
    .. [1] Wasserman, S., and Faust, K. (1994). Social subnet Analysis:
      Methods and Applications. Cambridge: Cambridge University Press.
    .. [2] Alain Barrat, Marc Barthelemy, Romualdo Pastor-Satorras, Alessandro
      Vespignani: The architecture of complex weighted networks, Proc. Natl.
      Acad. Sci. USA 101, 3747 (2004)

    """

    from networkx.algorithms.cluster import _weighted_triangles_and_degree_iter

    triangles = sum(t for v, d, t in _weighted_triangles_and_degree_iter(G))
    contri = sum(d * (d - 1)
                 for v, d, t in _weighted_triangles_and_degree_iter(G))

    return 0 if triangles == 0 else triangles / contri


def prune_small_components(G, min_nodes):
    """
    Returns a recomposed graph of all connected components of a minimum size

    Parameters
    ----------
    G : Obj
        NetworkX graph with isolated nodes present.
    min_nodes: int
        Minimum number of nodes permitted in a connected subgraph

    Returns
    -------
    G : Obj
        NetworkX graph with isolated nodes pruned.
    pruned_nodes : list
        List of indices of nodes that were pruned from G.

    References
    ----------
    .. [1] Hayasaka, S. (2017). Anti-Fragmentation of Resting-State
      Functional Magnetic Resonance Imaging Connectivity Networks
      with Node-Wise Thresholding. Brain Connectivity.
      https://doi.org/10.1089/brain.2017.0523
    .. [2] Fornito, A., Zalesky, A., & Bullmore, E. T. (2016).
      Fundamentals of Brain subnet Analysis. In Fundamentals of Brain
      subnet Analysis. https://doi.org/10.1016/C2012-0-06036-X

    """

    G_tmp = G.copy()

    # List because it returns a generator
    components = list(nx.connected_components(G_tmp))
    del G_tmp

    components.sort(key=len, reverse=True)
    print(f"{len(components)} connected component(s) detected...")

    # Iterate across cc subgraphs
    good_components = []
    c = 0
    for comp in components:
        print(f"Component {c}: {len(comp)} nodes")
        if len(comp) > min_nodes:
            good_components.append(nx.subgraph(G_tmp, comp))
        c = c + 1

    del components

    if len(good_components) == 0:
        raise ValueError(f"No components with a minimum of {min_nodes} found")

    return nx.compose_all(good_components)


def most_important(G, method="betweenness", sd=1, engine=DEFAULT_ENGINE):
    """
    Returns a copy of G with hubs as defined by centrality,
    core topology, or rich-club topology.

    Parameters
    ----------
    G : Obj
        NetworkX graph.
    method : str
        Determines method for defining hubs. Valid inputs are coreness,
        richclub, and eigenvector centrality. Default is coreness.
    sd : int
        Number of standard errors as cutoff for low-importance pruning.

    Returns
    -------
    G : Obj
        NetworkX graph with isolated and low-importance nodes pruned.
    pruned_nodes : list
       List of indices of nodes that were pruned from G.

    References
    ----------
    .. [1] Power, J. D., Schlaggar, B. L., Lessov-Schlaggar,
      C. N., & Petersen, S. E. (2013). Evidence for hubs
      in human functional brain networks. Neuron.
      https://doi.org/10.1016/j.neuron.2013.07.035

    """

    print(f"Detecting hubs using {method} with SE: {sd}...")
    if method == "eigenvector":
        ranking = nx.eigenvector_centrality(G, weight="weight").items()
    elif method == "richclub" and len(G.nodes()) > 4:
        ranking = rich_club_coefficient(G).items()
    else:
        ranking = nx.betweenness_centrality(G, weight="weight").items()

    # print(ranking)
    r = [x[1] for x in ranking]
    m = sum(r) / len(r) - sd * np.std(r)
    Gt = G.copy()
    pruned_nodes = []
    i = 0
    for k, v in ranking:
        if v < m:
            Gt.remove_node(k)
            pruned_nodes.append(i)
        i = i + 1

    return defragment(G)


def defragment(G):
    G_tmp = G.copy()

    isolates = [n for (n, d) in G_tmp.degree() if d == 0]

    # Remove any lingering isolates
    s = 0
    pruned_nodes = []
    for node in list(G_tmp.nodes()):
        if node in isolates:
            G_tmp.remove_node(node)
            pruned_nodes.append(s)
        s = s + 1

    for edge in list(G_tmp.edges()):
        if (str(edge[0]) in pruned_nodes) or (str(edge[1]) in pruned_nodes):
            G_tmp.remove_edge(edge)

    return G_tmp, pruned_nodes


def raw_mets(G, i, engine=DEFAULT_ENGINE):
    """
    API that iterates across NetworkX algorithms for a G.

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

    # import random
    from functools import partial
    if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")

    if isinstance(i, partial):
        net_name = str(i.func)
    else:
        net_name = str(i)
    if "average_shortest_path_length" in net_name:
        if engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
                engine.upper() == 'GRAPHTOOL':
            try:
                net_met_val = average_shortest_path_length_fast(
                    G, weight='weight')
            except BaseException:
                net_met_val = np.nan
        else:
            try:
                net_met_val = float(i(G))
            except BaseException:
                try:
                    net_met_val = float(
                        average_shortest_path_length_for_all(G))
                except BaseException as e:
                    print(e, f"WARNING: {net_name} failed for G.")
                    # np.save(f"/tmp/average_shortest_path_length_
                    # {random.randint(1, 400)}.npy",
                    #         np.array(nx.to_numpy_matrix(H)))
                    net_met_val = np.nan
    elif "average_clustering" in net_name and \
        (engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL'
         or engine.upper() == 'GRAPHTOOL'):
        try:
            g = nx2gt(G)
            net_met_val = gt.global_clustering(
                g, weight=g.edge_properties['weight'])[0]
        except BaseException as e:
            print(e, f"WARNING: {net_name} failed for G.")
            net_met_val = np.nan
    elif "graph_number_of_cliques" in net_name:
        if nx.is_connected(G) is True:
            try:
                net_met_val = float(i(G))
            except BaseException:
                try:
                    net_met_val = float(subgraph_number_of_cliques_for_all(G))
                except BaseException as e:
                    print(e, f"WARNING: {net_name} failed for G.")
                    # np.save(f"/tmp/graph_num_cliques_
                    # {random.randint(1, 400)}.npy",
                    #         np.array(nx.to_numpy_matrix(H)))
                    net_met_val = np.nan
        else:
            net_met_val = np.nan

    elif "smallworldness" in net_name:
        try:
            net_met_val = float(i(G))
        except BaseException as e:
            print(e, f"WARNING: {net_name} failed for G.")
            # np.save(f"/tmp/smallworldness{random.randint(1, 400)}.npy",
            #         np.array(nx.to_numpy_matrix(H)))
            net_met_val = np.nan
    elif "degree_assortativity_coefficient" in net_name:
        H = G.copy()
        for u, v, d in H.edges(data=True):
            H[u][v]["weight"] = int(np.round(100 * H[u][v]["weight"], 1))
        if nx.is_connected(H) is True:
            try:
                net_met_val = float(i(H))
            except BaseException as e:
                print(UserWarning(f"Warning {e}: "
                                  f"Degree assortativity coefficient measure "
                                  f"failed!"))
                net_met_val = np.nan
        else:
            try:
                from networkx.algorithms.assortativity import (
                    degree_pearson_correlation_coefficient,
                )

                net_met_val = float(
                    degree_pearson_correlation_coefficient(
                        H, weight="weight"))
            except BaseException as e:
                print(e, f"WARNING: {net_name} failed for G.")
                # np.save(f"/tmp/degree_assortativity_coefficient_
                # {random.randint(1, 400)}.npy",
                #         np.array(nx.to_numpy_matrix(H)))
                net_met_val = np.nan
    else:
        try:
            net_met_val = float(i(G))
        except BaseException as e:
            print(e, f"WARNING: {net_name} failed for G.")
            net_met_val = np.nan

    return net_met_val


def iterate_nx_global_measures(G, metric_list_glob):
    import time

    # import random
    num_mets = len(metric_list_glob)
    net_met_arr = np.zeros([num_mets, 2], dtype="object")
    j = 0
    for i in metric_list_glob:
        start_time = time.time()
        net_met = str(i).split("<function ")[1].split(" at")[0]
        try:
            try:
                net_met_val = raw_mets(G, i)
            except BaseException:
                print(f"{'WARNING: '}{net_met}{' failed for G.'}")
                # np.save("%s%s%s%s" % ('/tmp/', net_met,
                # random.randint(1, 400), '.npy'),
                #         np.array(nx.to_numpy_matrix(G)))
                net_met_val = np.nan
        except BaseException:
            print(f"{'WARNING: '}{str(i)}{' is undefined for G'}")
            # np.save("%s%s%s%s" % ('/tmp/', net_met, random.randint(1,
            # 400), '.npy'), np.array(nx.to_numpy_matrix(G)))
            net_met_val = np.nan
        net_met_arr[j, 0] = net_met
        net_met_arr[j, 1] = net_met_val
        print(net_met.replace("_", " ").title())
        print(str(net_met_val))
        print(f"{np.round(time.time() - start_time, 3)}{'s'}")
        print("\n")
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
    ci = np.array(
        list(
            community.best_partition(
                G,
                resolution=resolution).values()))
    num_comms = len(np.unique(ci))
    if num_comms == 1:
        resolution = 10
        tries = 0
        while num_comms == 1:
            ci = np.array(
                list(
                    community.best_partition(
                        G,
                        resolution=resolution).values()))
            num_comms = len(np.unique(ci))
            print(
                f"{'Found '}{num_comms}{' communities at resolution: '}"
                f"{resolution}{'...'}"
            )
            resolution = resolution + 10
            tries = tries + 1
            if tries > 100:
                print(
                    "\nWARNING: Louvain community detection failed. "
                    "Proceeding with single community affiliation vector...")
                break
    else:
        print(
            f"{'Found '}{num_comms}{' communities at resolution: '}"
            f"{resolution}{'...'}"
        )
    return dict(zip(G.nodes(), ci)), ci, resolution, num_comms


@timeout(DEFAULT_TIMEOUT)
def get_community(G, net_met_val_list_final, metric_list_names):
    import community

    ci_dict, ci, resolution, num_comms = community_resolution_selection(G)
    modularity = community.community_louvain.modularity(ci_dict, G)
    metric_list_names.append("modularity")
    if modularity == 1 or modularity == 0:
        modularity = np.nan
        print("Louvain modularity is undefined for G")
    net_met_val_list_final.append(modularity)
    return net_met_val_list_final, metric_list_names, ci


@timeout(DEFAULT_TIMEOUT)
def get_participation(in_mat, ci, metric_list_names, net_met_val_list_final):
    if len(in_mat[in_mat < 0]) > 0:
        pc_vector = participation_coef_sign(in_mat, ci)[0]
    else:
        pc_vector = participation_coef(in_mat, ci)
    print("\nExtracting Participation Coefficients...")
    pc_vals = list(pc_vector)
    pc_edges = list(range(len(pc_vector)))
    num_edges = len(pc_edges)
    pc_arr = np.zeros([num_edges + 1, 2], dtype="object")
    j = 0
    for i in range(num_edges):
        pc_arr[j, 0] = f"{str(pc_edges[j])}{'_participation_coefficient'}"
        try:
            pc_arr[j, 1] = pc_vals[j]
        except BaseException:
            print(
                f"{'Participation coefficient is undefined for node '}"
                f"{str(j)}{' of G'}"
            )
            pc_arr[j, 1] = np.nan
        j = j + 1
    # Add mean
    pc_arr[num_edges, 0] = "average_participation_coefficient"
    nonzero_arr_partic_coef = np.delete(pc_arr[:, 1], [0])
    pc_arr[num_edges, 1] = np.nanmean(
        nonzero_arr_partic_coef.astype('float32'), dtype=np.float32)
    print(f"{'Mean Participation Coefficient: '}{str(pc_arr[num_edges, 1])}")
    for i in pc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(pc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_diversity(in_mat, ci, metric_list_names, net_met_val_list_final):
    dc_vector = diversity_coef_sign(in_mat, ci)[0]
    print("\nExtracting Diversity Coefficients...")
    dc_vals = list(dc_vector)
    dc_edges = list(range(len(dc_vector)))
    num_edges = len(dc_edges)
    dc_arr = np.zeros([num_edges + 1, 2], dtype="object")
    j = 0
    for i in range(num_edges):
        dc_arr[j, 0] = f"{str(dc_edges[j])}{'_diversity_coefficient'}"
        try:
            dc_arr[j, 1] = dc_vals[j]
        except BaseException:
            print(
                f"{'Diversity coefficient is undefined for node '}{str(j)}"
                f"{' of G'}")
            dc_arr[j, 1] = np.nan
        j = j + 1
    # Add mean
    dc_arr[num_edges, 0] = "average_diversity_coefficient"
    nonzero_arr_diversity_coef = np.delete(dc_arr[:, 1], [0])
    dc_arr[num_edges, 1] = np.nanmean(
        nonzero_arr_diversity_coef.astype('float32'), dtype=np.float32)
    print(f"{'Mean Diversity Coefficient: '}{str(dc_arr[num_edges, 1])}")
    for i in dc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_local_efficiency(G, metric_list_names, net_met_val_list_final):
    le_vector = local_efficiency(G)
    print("\nExtracting Local Efficiencies...")
    le_vals = list(le_vector.values())
    le_nodes = list(le_vector.keys())
    num_nodes = len(le_nodes)
    le_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        le_arr[j, 0] = f"{str(le_nodes[j])}{'_local_efficiency'}"
        try:
            le_arr[j, 1] = le_vals[j]
        except BaseException:
            print(f"{'Local efficiency is undefined for node '}{str(j)}"
                  f"{' of G'}")
            le_arr[j, 1] = np.nan
        j = j + 1
    le_arr[num_nodes, 0] = "average_local_efficiency_nodewise"
    nonzero_arr_le = np.delete(le_arr[:, 1], [0])
    le_arr[num_nodes, 1] = np.nanmean(nonzero_arr_le.astype('float32'),
                                      dtype=np.float32)
    print(f"{'Mean Local Efficiency: '}{str(le_arr[num_nodes, 1])}")
    for i in le_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(le_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_clustering(G, metric_list_names, net_met_val_list_final,
                   engine=DEFAULT_ENGINE):

    if engine.upper() == 'NX' or engine.upper() == 'NETWORKX':
        cl_vector = nx.clustering(G, weight="weight")
    elif engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")
        g = nx2gt(G)
        cl_vector = dict(zip(list(g.get_vertices()), list(
            gt.local_clustering(g, weight=g.ep["weight"]).get_array())))
    else:
        raise ValueError(f"Engine {engine} not recognized.")

    print("\nExtracting Local Clusterings...")
    cl_vals = list(cl_vector.values())
    cl_nodes = list(cl_vector.keys())
    num_nodes = len(cl_nodes)
    cl_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        cl_arr[j, 0] = f"{str(cl_nodes[j])}{'_local_clustering'}"
        try:
            cl_arr[j, 1] = cl_vals[j]
        except BaseException:
            print(f"{'Local clustering is undefined for node '}{str(j)}"
                  f"{' of G'}")
            cl_arr[j, 1] = np.nan
        j = j + 1
    cl_arr[num_nodes, 0] = "average_local_clustering_nodewise"
    nonzero_arr_cl = np.delete(cl_arr[:, 1], [0])
    cl_arr[num_nodes, 1] = np.nanmean(nonzero_arr_cl.astype('float32'),
                                      dtype=np.float32)
    print(f"{str(cl_arr[num_nodes, 1])}")
    for i in cl_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(cl_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_degree_centrality(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import degree_centrality

    dc_vector = degree_centrality(G)
    print("\nExtracting Local Degree Centralities...")
    dc_vals = list(dc_vector.values())
    dc_nodes = list(dc_vector.keys())
    num_nodes = len(dc_nodes)
    dc_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        dc_arr[j, 0] = f"{str(dc_nodes[j])}{'_degree_centrality'}"
        try:
            dc_arr[j, 1] = dc_vals[j]
        except BaseException:
            print(f"{'Degree centrality is undefined for node '}{str(j)}"
                  f"{' of G'}")
            dc_arr[j, 1] = np.nan
        j = j + 1
    dc_arr[num_nodes, 0] = "average_degree_centrality"
    nonzero_arr_dc = np.delete(dc_arr[:, 1], [0])
    dc_arr[num_nodes, 1] = np.nanmean(nonzero_arr_dc.astype('float32'),
                                      dtype=np.float32)
    print(
        f"{str(dc_arr[num_nodes, 1])}")
    for i in dc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(dc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_betweenness_centrality(
        G_len,
        metric_list_names,
        net_met_val_list_final, engine=DEFAULT_ENGINE):
    from networkx.algorithms import betweenness_centrality

    if engine.upper() == 'NX' or engine.upper() == 'NETWORKX':
        bc_vector = betweenness_centrality(G_len, normalized=True)
    elif engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")
        g = nx2gt(G_len)
        bc_vector = dict(zip(list(g.get_vertices()), list(
            gt.betweenness(g, weight=g.ep["weight"])[0].get_array())))
    else:
        raise ValueError(f"Engine {engine} not recognized.")

    print("\nExtracting Local Betweenness Centralities...")
    bc_vals = list(bc_vector.values())
    bc_nodes = list(bc_vector.keys())
    num_nodes = len(bc_nodes)
    bc_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        bc_arr[j, 0] = f"{str(bc_nodes[j])}{'_betweenness_centrality'}"
        try:
            bc_arr[j, 1] = bc_vals[j]
        except BaseException:
            print(
                f"{'betweennesss centrality is undefined for node '}"
                f"{str(j)}{' of G'}"
            )
            bc_arr[j, 1] = np.nan
        j = j + 1
    bc_arr[num_nodes, 0] = "average_betweenness_centrality"
    nonzero_arr_betw_cent = np.delete(bc_arr[:, 1], [0])
    bc_arr[num_nodes, 1] = np.nanmean(nonzero_arr_betw_cent.astype('float32'),
                                      dtype=np.float32)
    print(
        f"{'Mean Betweenness Centrality: '}"
        f"{str(bc_arr[num_nodes, 1])}")
    for i in bc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(bc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_eigen_centrality(G, metric_list_names, net_met_val_list_final,
                         engine=DEFAULT_ENGINE):

    if engine.upper() == 'NX' or engine.upper() == 'NETWORKX':
        from networkx.algorithms import eigenvector_centrality
        ec_vector = eigenvector_centrality(G, max_iter=1000)
    elif engine.upper() == 'GT' or engine.upper() == 'GRAPH_TOOL' or \
            engine.upper() == 'GRAPHTOOL':
        try:
            import graph_tool.all as gt
        except ImportWarning as e:
            print(e, "Graph Tool not installed!")
        g = nx2gt(G)
        ec_vector = dict(zip(list(g.get_vertices()), list(
            gt.eigenvector(g, weight=g.ep["weight"])[1].get_array())))
    else:
        raise ValueError(f"Engine {engine} not recognized.")

    print("\nExtracting Local Eigenvector Centralities...")
    ec_vals = list(ec_vector.values())
    ec_nodes = list(ec_vector.keys())
    num_nodes = len(ec_nodes)
    ec_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        ec_arr[j, 0] = f"{str(ec_nodes[j])}{'_eigenvector_centrality'}"
        try:
            ec_arr[j, 1] = ec_vals[j]
        except BaseException:
            print(
                f"{'Eigenvector centrality is undefined for node '}"
                f"{str(j)}{' of G'}")
            ec_arr[j, 1] = np.nan
        j = j + 1
    ec_arr[num_nodes, 0] = "average_eigenvector_centrality"
    nonzero_arr_eig_cent = np.delete(ec_arr[:, 1], [0])
    ec_arr[num_nodes, 1] = np.nanmean(nonzero_arr_eig_cent.astype('float32'),
                                      dtype=np.float32)
    print(
        f"{'Mean Eigenvector Centrality: '}"
        f"{str(ec_arr[num_nodes, 1])}")
    for i in ec_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(ec_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_comm_centrality(G, metric_list_names, net_met_val_list_final):
    from networkx.algorithms import communicability_betweenness_centrality

    cc_vector = communicability_betweenness_centrality(G)
    print("\nExtracting Local Communicability Centralities...")
    cc_vals = list(cc_vector.values())
    cc_nodes = list(cc_vector.keys())
    num_nodes = len(cc_nodes)
    cc_arr = np.zeros([num_nodes + 1, 2], dtype="object")
    j = 0
    for i in range(num_nodes):
        cc_arr[j, 0] = f"{str(cc_nodes[j])}{'_communicability_centrality'}"
        try:
            cc_arr[j, 1] = cc_vals[j]
        except BaseException:
            print(
                f"{'Communicability centrality is undefined for node '}"
                f"{str(j)}{' of G'}"
            )
            cc_arr[j, 1] = np.nan
        j = j + 1
    cc_arr[num_nodes, 0] = "average_communicability_centrality"
    nonzero_arr_comm_cent = np.delete(cc_arr[:, 1], [0])
    cc_arr[num_nodes, 1] = np.nanmean(nonzero_arr_comm_cent.astype('float32'),
                                      dtype=np.float32)
    print(
        f"{'Mean Communicability Centrality: '}"
        f"{str(cc_arr[num_nodes, 1])}"
    )
    for i in cc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(cc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


@timeout(DEFAULT_TIMEOUT)
def get_rich_club_coeff(G, metric_list_names, net_met_val_list_final,
                        engine=DEFAULT_ENGINE):

    print("\nExtracting Rich Club Coefficient...")
    rc_vector = rich_club_coefficient(G, engine=engine)
    rc_vals = list(rc_vector.values())
    rc_edges = list(rc_vector.keys())
    num_edges = len(rc_edges)
    rc_arr = np.zeros([num_edges + 1, 2], dtype="object")
    j = 0
    for i in range(num_edges):
        rc_arr[j, 0] = f"{str(rc_edges[j])}{'_rich_club_coefficient'}"
        try:
            rc_arr[j, 1] = rc_vals[j]
        except BaseException:
            print(
                f"{'Rich club coefficient is undefined for node '}"
                f"{str(j)}{' of G'}")
            rc_arr[j, 1] = np.nan
        j = j + 1
    # Add mean
    rc_arr[num_edges, 0] = "average_rich_club_coefficient"
    nonzero_arr_rich_club = np.delete(rc_arr[:, 1], [0])
    rc_arr[num_edges, 1] = np.nanmean(nonzero_arr_rich_club.astype('float32'),
                                      dtype=np.float32)
    print(
        f"{'Mean Rich Club Coefficient: '}"
        f"{str(rc_arr[num_edges, 1])}")
    for i in rc_arr[:, 0]:
        metric_list_names.append(i)
    net_met_val_list_final = net_met_val_list_final + list(rc_arr[:, 1])
    return metric_list_names, net_met_val_list_final


def collect_pandas_df_make(
    net_mets_csv_list,
    ID,
    subnet,
    plot_switch,
    embed=False,
    create_summary=False,
):
    """
    Summarize list of pickled pandas dataframes of graph metrics unique to
    each unique combination of metaparameters.

    Parameters
    ----------
    net_mets_csv_list : list
        List of file paths to pickled pandas dataframes as themselves.
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    plot_switch : bool
        Activate summary plotting (histograms, central tendency, AUC, etc.)

    Returns
    -------
    combination_complete : bool
        If True, then data integration completed successfully.

    References
    ----------
    .. [1] Drakesmith, M., Caeyenberghs, K., Dutt, A., Lewis, G., David,
      A. S., & Jones, D. K. (2015). Overcoming the effects of false positives
      and threshold bias in graph theoretical analyses of neuroimaging data.
      NeuroImage. https://doi.org/10.1016/j.neuroimage.2015.05.011

    """
    import gc
    import os
    import os.path as op
    import pandas as pd
    from pynets.core import utils
    from pynets.statistics.utils import build_mp_dict
    from itertools import groupby
    import re
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    embedding_methods = hardcoded_params["embed"]

    # from sklearn.decomposition import PCA

    # Check for existence of net_mets csv files, condensing final list to only
    # those that were actually produced.
    net_mets_csv_list_exist = []
    for net_mets_csv in list(utils.flatten(net_mets_csv_list)):
        if op.isfile(net_mets_csv) is True:
            if net_mets_csv.endswith('.csv'):
                net_mets_csv_list_exist.append(net_mets_csv)
            else:
                print(UserWarning('Warning: File not .csv format'))
                continue
        else:
            print(UserWarning(f"Warning: {net_mets_csv} not found. "
                              f"Skipping..."))

    if len(list(net_mets_csv_list)) > len(net_mets_csv_list_exist):
        raise UserWarning(
            "Warning! Number of actual graphs produced less than expected. "
            "Some were excluded"
        )

    net_mets_csv_list = net_mets_csv_list_exist
    if len(net_mets_csv_list) >= 1:
        subject_path = op.dirname(op.dirname(op.dirname(net_mets_csv_list[0])))
    else:
        print("No topology files found!")
        combination_complete = True

    hyperparam_dict = {}
    dfs_non_auc = []
    hyperparam_dict["id"] = ID
    gen_hyperparams = ["nodetype", "model", "template"]

    if len(net_mets_csv_list) > 1:
        print(f"\n\nAll graph analysis results:\n{str(net_mets_csv_list)}\n\n")

        models = []
        for file_ in net_mets_csv_list:
            models.append(
                f"{op.basename(op.dirname(op.dirname(file_)))}/topology/"
                f"{op.basename(file_)}"
            )

        if any('thr-' in i for i in net_mets_csv_list_exist):
            def sort_thr(model_name):
                return model_name.split("thr-")[1].split("_")[0]

            models.sort(key=sort_thr)

            # Group by secondary attributes
            models_grouped = [
                list(x)
                for x in zip(
                    *[
                        list(g)
                        for k, g in groupby(
                            models, lambda s: s.split("thr-")[1].split("_")[0]
                        )
                    ]
                )
            ]
            node_cols = None
            if max([len(i) for i in models_grouped]) > 1:
                print(
                    "Multiple thresholds detected. Computing AUC..."
                )
                meta = dict()
                non_decimal = re.compile(r"[^\d.]+")
                for thr_set in range(len(models_grouped)):
                    meta[thr_set] = dict()
                    meta[thr_set]["dataframes"] = dict()
                    for i in models_grouped[thr_set]:
                        thr = non_decimal.sub("",
                                              i.split("thr-")[1].split("_")[0])
                        _file = subject_path + "/" + i
                        if os.path.isfile(_file):
                            df = pd.read_csv(_file, memory_map=True,
                                             chunksize=100000,
                                             encoding="utf-8",
                                             skip_blank_lines=False,
                                             warn_bad_lines=True,
                                             error_bad_lines=False
                                             ).read()
                            node_cols = [
                                s
                                for s in list(df.columns)
                                if isinstance(s, int) or any(c.isdigit() for
                                                             c in s)
                            ]
                            if embed is False:
                                df = df.drop(node_cols, axis=1)
                            meta[thr_set]["dataframes"][thr] = df
                        else:
                            print(f"File {_file} not found...")
                            continue
                # For each unique threshold set, for each graph measure,
                # extract AUC
                for thr_set in meta.keys():
                    if len(meta[thr_set]["dataframes"].values()) > 1:
                        df_summary = pd.concat(
                            meta[thr_set]["dataframes"].values())
                    else:
                        print(f"No values to concatenate at {thr_set}...")
                        continue
                    df_summary["thr"] = meta[thr_set]["dataframes"].keys()
                    meta[thr_set]["summary_dataframe"] = df_summary
                    df_summary_auc = df_summary.iloc[[0]]
                    df_summary_auc.columns = [
                        col + "_auc" for col in df_summary.columns]

                    print(f"\nAUC for threshold group: "
                          f"{models_grouped[thr_set]}")
                    file_renamed = list(
                        set(
                            [
                                re.sub(
                                    r"thr\-\d+\.*\d+\_", "",
                                    i.split("/topology/")[1]
                                ).replace("neat", "auc")
                                for i in models_grouped[thr_set]
                            ]
                        )
                    )[0]
                    atlas = models_grouped[thr_set][0].split("/")[0]
                    modality = file_renamed.split("modality-")[1].split("_")[0]

                    # Build hyperparameter dictionary
                    hyperparam_dict, hyperparams = build_mp_dict(
                        file_renamed, modality, hyperparam_dict,
                        gen_hyperparams)

                    for measure in df_summary.columns[:-1]:
                        # Get Area Under the Curve
                        df_summary_nonan = df_summary[pd.notnull(
                            df_summary[measure])]
                        df_summary_auc[measure] = np.trapz(
                            np.array(df_summary_nonan[measure]
                                     ).astype("float32")
                        )
                        print(
                            f"{measure}: "
                            f"{df_summary_auc[measure].to_string(index=False)}"
                        )
                    meta[thr_set]["auc_dataframe"] = df_summary_auc
                    auc_dir = f"{subject_path}{'/'}{atlas}{'/topology/auc/'}"
                    if not os.path.isdir(auc_dir):
                        os.makedirs(auc_dir, exist_ok=True)
                    df_summary_auc = df_summary_auc.drop(columns=["thr_auc"])
                    df_summary_auc = df_summary_auc.loc[
                        :, df_summary_auc.columns.str.endswith("auc")
                    ]
                    auc_outfile = auc_dir + file_renamed
                    if os.path.isfile(auc_outfile):
                        try:
                            os.remove(auc_outfile)
                        except BaseException:
                            continue
                    df_summary_auc.to_csv(
                        auc_outfile,
                        header=True,
                        index=False,
                        chunksize=100000,
                        compression="gzip",
                        encoding="utf-8",
                    )
                    node_cols_embed = [i for i in node_cols if i in
                                       embedding_methods]

                    from pathlib import Path
                    parent_dir = str(
                        Path(os.path.dirname(net_mets_csv_list[0])).parent)
                    base_name = \
                        os.path.basename(net_mets_csv_list[0]).split(
                            'metrics_')[
                            1].split('_thr-')[0]

                    if embed is True and len(node_cols_embed) > 0:
                        embed_dir = f"{parent_dir}/embeddings"
                        if not os.path.isdir(embed_dir):
                            os.makedirs(embed_dir, exist_ok=True)

                        node_cols_auc = [f"{i}_auc" for i in node_cols_embed if
                                         f"{i}_auc" in df_summary_auc.columns]
                        df_summary_auc_nodes = df_summary_auc[node_cols_auc]
                        node_embeddings_grouped = [{k: list(g)} for k, g in
                                                   groupby(
                                                       df_summary_auc_nodes,
                            lambda s:
                            s.split("_")[1])]
                        for node_dict in node_embeddings_grouped:
                            node_top_type = list(node_dict.keys())[0]
                            node_top_cols = list(node_dict.values())[0]
                            embedding_frame = \
                                df_summary_auc_nodes[node_top_cols]
                            out_path = f"{embed_dir}/gradient-" \
                                       f"{node_top_type}_" \
                                       f"subnet-{atlas}_auc_nodes_" \
                                       f"{base_name}.csv"
                            embedding_frame.to_csv(out_path, index=False)
        else:
            models_grouped = None
            meta = {}
            for file_ in net_mets_csv_list:
                df = pd.read_csv(file_, memory_map=True,
                                 chunksize=100000, encoding="utf-8",
                                 skip_blank_lines=False,
                                 warn_bad_lines=True,
                                 error_bad_lines=False
                                 ).read()
                node_cols = [
                    s
                    for s in list(df.columns)
                    if isinstance(s, int) or any(c.isdigit() for c in
                                                 s)
                ]
                if embed is False:
                    df.drop(node_cols, axis=1, inplace=True)
                elif len(node_cols) > 1:
                    from pathlib import Path
                    parent_dir = str(
                        Path(os.path.dirname(net_mets_csv_list[0])).parent)
                    node_cols_embed = [i for i in node_cols if
                                       any(map(i.__contains__,
                                               embedding_methods))]
                    if len(node_cols_embed) > 0:
                        embed_dir = f"{parent_dir}/embeddings"
                        if not os.path.isdir(embed_dir):
                            os.makedirs(embed_dir, exist_ok=True)
                        df_nodes = df[node_cols_embed]
                        node_embeddings_grouped = [{k: list(g)} for k, g in
                                                   groupby(df_nodes,
                                                           lambda s: s.split(
                                                               "_")[1])]
                        atlas = os.path.dirname(file_
                                                ).split('subnet-')[1].split(
                            '_')[0]
                        if 'thr-' in os.path.basename(file_):
                            base_name = os.path.basename(file_).split(
                                'metrics_')[1].split(
                                '_thr-')[0] + '.csv'
                        else:
                            base_name = os.path.basename(file_).split(
                                'metrics_')[1]
                        for node_dict in node_embeddings_grouped:
                            node_top_type = list(node_dict.keys())[0]
                            node_top_cols = list(node_dict.values())[0]
                            embedding_frame = df_nodes[node_top_cols]
                            out_path = f"{embed_dir}/gradient-" \
                                       f"{node_top_type}_" \
                                       f"subnet-{atlas}_nodes_" \
                                       f"{base_name}"
                            embedding_frame.to_csv(out_path, index=False)
                dfs_non_auc.append(df)

        if create_summary is True:
            try:
                summary_dir = f"{subject_path}/summary"
                if not os.path.isdir(summary_dir):
                    os.makedirs(summary_dir, exist_ok=True)

                # Concatenate and find mean across dataframes
                print("Concatenating frames...")
                if models_grouped:
                    if max([len(i) for i in models_grouped]) > 1:
                        df_concat = pd.concat(
                            [meta[thr_set]["auc_dataframe"] for
                             thr_set in meta.keys() if "auc_dataframe" in
                             meta[thr_set].keys()]
                        )
                        del meta
                    else:
                        df_concat = pd.concat(dfs_non_auc)
                else:
                    df_concat = pd.concat(dfs_non_auc)
                measures = list(df_concat.columns)
                df_concatted_mean = (df_concat.loc[:, measures].mean(
                    skipna=True).to_frame().transpose())
                df_concatted_median = (
                    df_concat.loc[:, measures]
                    .median(skipna=True)
                    .to_frame()
                    .transpose()
                )
                df_concatted_mode = pd.DataFrame(
                    df_concat.loc[:, measures].mode(axis=0, dropna=True).max()
                ).transpose()

                # PCA across AUC node measures
                # node_measures_grouped = [list(y) for x, y in
                # groupby(node_cols, lambda s: s.split('_')[1])]
                # for node_measures in node_measures_grouped:
                #     pca = PCA(n_components=2)
                #     df_concatted_pca = pd.Series(pca.fit_transform(
                #     df_concat.loc[:,
                #     node_measures])[1]).to_frame().transpose()
                #     df_concatted_pca.columns = [str(col) + '_PCA' for col in
                #     df_concatted_pca.columns]
                df_concatted_mean.columns = [
                    str(col) + "_mean" for col in df_concatted_mean.columns
                ]
                df_concatted_median.columns = [
                    str(col) + "_median" for col in df_concatted_median.columns
                ]
                df_concatted_mode.columns = [
                    str(col) + "_maxmode" for col in df_concatted_mode.columns
                ]
                result = pd.concat(
                    [df_concatted_mean, df_concatted_median,
                     df_concatted_mode], axis=1
                )
                df_concatted_final = result.reindex(
                    sorted(result.columns), axis=1)
                print(f"\nConcatenating dataframes for {str(ID)}...\n")
                net_csv_summary_out_path = (
                    f"{summary_dir}/avg_topology_sub-{str(ID)}"
                    f"{'%s' % ('_' + subnet if subnet is not None else '')}"
                    f".csv")
                if os.path.isfile(net_csv_summary_out_path):
                    try:
                        os.remove(net_csv_summary_out_path)
                    except BaseException:
                        pass
                df_concatted_final.to_csv(
                    net_csv_summary_out_path, index=False)
                del (
                    result,
                    df_concat,
                    df_concatted_mean,
                    df_concatted_median,
                    df_concatted_mode,
                    df_concatted_final,
                )

                combination_complete = True
            except RuntimeWarning:
                combination_complete = False
                print(
                    f"\nWARNING: DATAFRAME CONCATENATION FAILED FOR "
                    f"{str(ID)}!\n")
                pass
        else:
            combination_complete = True
    else:
        if subnet is not None:
            print(
                f"\nSingle dataframe for the {subnet} subnet for subject "
                f"{ID}\n")
        else:
            print(f"\nSingle dataframe for subject {ID}\n")
        combination_complete = True

    gc.collect()

    return combination_complete
