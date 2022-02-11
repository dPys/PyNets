#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
@authors: Derek Pisner & Ryan Hammonds
"""
import pytest
import os
import numpy as np
import networkx as nx
import time
from pathlib import Path
from pynets.statistics.individual import algorithms
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("value", [True, 2, 3.14, "word", {}])
def test_get_prop_type(value):
    """
    Test for get_prop_type() functionality
    """
    start_time = time.time()
    tname, value, key = algorithms.get_prop_type(value)
    if value == 2:
        value = int(value)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))

    if type(value) == bool:
        assert tname is 'bool'

    elif type(value) == float or type(value) == int:
        assert tname is 'float'

    elif type(value) == dict:
        assert tname is 'object'

    elif type(value) == bytes:
        assert tname is 'string'


def test_nx2gt(gen_mat_data):
    """
    Test for nx2gt() functionality
    """
    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    nxG = nx.from_numpy_array(in_mat)

    start_time = time.time()
    gtG = algorithms.nx2gt(nxG)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))
    try:
        import graph_tool.all as gt
        assert type(gtG) is gt.graph_tool.Graph
    except ImportError as e:
        print(e, "graph_tool not installed!")
        assert gtG is not None


def test_np2gt(gen_mat_data):
    """
    Test for np2gt() functionality
    """
    in_mat = gen_mat_data(asfile=False)['mat_list'][0]

    start_time = time.time()
    Gt = algorithms.np2gt(in_mat)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))
    try:
        import graph_tool.all as gt
        assert type(Gt) is gt.graph_tool.Graph
    except ImportError as e:
        print(e, "graph_tool not installed!")
        assert Gt is not None


def test_average_shortest_path_length_for_all(gen_mat_data):
    """
    Test for average_shortest_path_length_for_all functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    avgest_path_len = algorithms.average_shortest_path_length_for_all(G)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))
    assert avgest_path_len > 0
    assert type(avgest_path_len) == float


@pytest.mark.parametrize("weight", ["weight", "not_weight"])
def test_average_shortest_path_length_fast(gen_mat_data, weight):
    """
    Test for average_shortest_path_length_fast functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    avgest_path_len = algorithms.average_shortest_path_length_fast(
        G, weight=weight)
    print("%s%s%s" % ('test_average_shortest_path_length_fast --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert avgest_path_len > 0
    assert type(avgest_path_len) == np.float64


@pytest.mark.parametrize("engine", ["GT", "NX"])
def test_average_local_efficiency(gen_mat_data, engine):
    """
    Test for average_local_efficiency functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    average_local_efficiency = algorithms.average_local_efficiency(
        G, engine=engine)
    print("%s%s%s" % ('test_average_local_efficiency --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert average_local_efficiency > 0
    assert type(average_local_efficiency) == np.float64


# used random node_comm_aff_mat
def test_create_communities():
    """
    Test for create_communities functionality
    """
    base_dir = str(Path(__file__).parent / "examples")
    node_comm_aff_mat = np.random.rand(5, 5)
    node_num = 3

    start_time = time.time()
    com_assign = algorithms.create_communities(node_comm_aff_mat, node_num)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))
    assert len(com_assign) > 0


@pytest.mark.parametrize("degree", ['undirected', 'in', 'out'])
def test_participation_coef(gen_mat_data, degree):
    """
    Test for participation_coef functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    P = algorithms.participation_coef(W, ci, degree=degree)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert P.size > 0


def test_modularity(gen_mat_data):
    """
    Test for modularity functionality
    """
    import community

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_matrix(in_mat)
    start_time = time.time()
    ci_dict = community.best_partition(G)
    mod = community.community_louvain.modularity(ci_dict, G)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert type(ci_dict) == dict
    assert type(mod) == float


def test_diversity_coef_sign(gen_mat_data):
    """
    Test for diversity_coef_sign functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]

    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    [Hpos, Hneg] = algorithms.diversity_coef_sign(W, ci)
    print("%s%s%s" % (
    'thresh_and_fit (Functional, proportional thresholding) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))
    assert Hpos.size > 0
    assert Hneg.size > 0


@pytest.mark.parametrize("clustering",
                         [
                             'single',
                             pytest.param('complete', marks=pytest.mark.xfail(
                                 raises=ValueError)),
                             pytest.param(None, marks=pytest.mark.xfail(
                                 raises=ValueError))
                         ]
                         )
def test_link_communities(gen_mat_data, clustering):
    """
    Test for link_communities functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    start_time = time.time()
    M = algorithms.link_communities(in_mat, type_clustering=clustering)
    print("%s%s%s" % ('Link Communities --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert type(M) is np.ndarray
    assert np.sum(M) == 20


# @pytest.mark.parametrize("min_nodes",
#                          [0, 1, 10, pytest.param(30, marks=pytest.mark.xfail(
#                                  raises=ValueError))])
# def test_prune_small_components(gen_mat_data, min_nodes):
#     """
#     Test pruning functionality
#     """
#     return


@pytest.mark.parametrize("method", ["betweenness", "coreness",
                                    "eigenvector"])
@pytest.mark.parametrize("engine", ["GT", "NX"])
def test_most_important(gen_mat_data, method, engine):
    """
    Test pruning for most important nodes functionality
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    Gt, pruned_nodes = algorithms.most_important(G, method=method,
                                                 engine=engine)
    print("%s%s%s" % ('test_most_important --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert Gt is not None
    assert pruned_nodes is not None


@pytest.mark.parametrize("engine", ["GT", "NX"])
def test_raw_mets(gen_mat_data, engine):
    """
    Test raw_mets extraction functionality
    """
    from pynets.statistics.individual.algorithms import global_efficiency, \
        average_local_efficiency, smallworldness
    from networkx.algorithms import degree_assortativity_coefficient, \
        average_clustering, average_shortest_path_length, \
        degree_pearson_correlation_coefficient, graph_number_of_cliques, \
        transitivity

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    metric_list_glob = [global_efficiency, average_local_efficiency,
                        degree_assortativity_coefficient,
                        average_clustering, average_shortest_path_length,
                        degree_pearson_correlation_coefficient,
                        graph_number_of_cliques, transitivity,
                        smallworldness]
    for i in metric_list_glob:
        net_met_val = algorithms.raw_mets(G, i, engine=engine)
        assert net_met_val is not np.nan
        if engine == 'nx':
            assert type(net_met_val) == np.float64
        elif engine == 'gt':
            assert type(net_met_val) == np.float64


def test_subgraph_number_of_cliques_for_all(gen_mat_data):
    """
    Test cliques computation
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    cliques = algorithms.subgraph_number_of_cliques_for_all(G)

    assert cliques > 0


@pytest.mark.parametrize("approach",
                         [
                             'clustering',
                             'transitivity',
                             pytest.param('impossible',
                                          marks=pytest.mark.xfail(
                                              raises=ValueError))
                         ]
                         )
@pytest.mark.parametrize("reference",
                         [
                             'random',
                             'lattice',
                             # 'fast',
                             pytest.param('impossible',
                                          marks=pytest.mark.xfail(
                                              raises=ValueError))
                         ]
                         )
def test_smallworldness(gen_mat_data, approach, reference):
    """
    Test small-world coefficient (omega) computation
    """

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    sigma = algorithms.smallworldness(G, niter=5, nrand=5,
                                      approach=approach,
                                      reference=reference, engine='nx')

    # A subnet is smallworld if sigma > 1
    assert sigma < 1


def test_participation_coef_sign(gen_mat_data):
    """
    Test participation coefficient computation
    """
    in_mat = gen_mat_data(asfile=False)['mat_list'][0]

    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    Ppos, Pneg = algorithms.participation_coef_sign(W, ci)

    assert len(Ppos) == ci_dim and len(Pneg) == ci_dim


@pytest.mark.parametrize("binarize", [True, False])
def test_weighted_transitivity(gen_mat_data, binarize):
    """ Test weighted_transitivity computation
    """
    from pynets.core.thresholding import binarize

    if binarize:
        in_mat = gen_mat_data(asfile=False, binary=True)['mat_list'][0]
    else:
        in_mat = gen_mat_data(asfile=False)['mat_list'][0]

    G = nx.from_numpy_array(in_mat)

    transitivity = algorithms.weighted_transitivity(G)

    assert transitivity <= 3 and transitivity >= 0


@pytest.mark.parametrize("true_metric", [True, False])
def test_iterate_nx_global_measures(gen_mat_data, true_metric):
    """ Test iterating over net metric list
    """
    from networkx.algorithms import average_shortest_path_length

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]
    G = nx.from_numpy_array(in_mat)

    if true_metric:
        metric_list_glob = [average_shortest_path_length]
    else:
        metric_list_glob = ['<function fake_func at 0x7f8b7129b700>']

    algorithms.iterate_nx_global_measures(G, metric_list_glob)


@pytest.mark.parametrize("sim_num_comms", [1, 5, 10])
@pytest.mark.parametrize("sim_size", [1, 5, 10])
def test_community_resolution_selection(sim_num_comms, sim_size):
    """ Test community resolution selection
    Note: It is impossible to enter or cover the second while loop in
          netstats.community_resolution_selection.
    """
    G = nx.caveman_graph(sim_num_comms, sim_size)
    node_ci, ci, resolution, num_comms = \
        algorithms.community_resolution_selection(G)

    assert len(node_ci) == len(ci)
    assert num_comms == sim_num_comms
    assert resolution is not None


@pytest.mark.parametrize("metric", ['participation', 'diversity',
                                    'local_efficiency',
                                    'comm_centrality'])
@pytest.mark.parametrize("engine", ["GT", "NX"])
def test_get_metrics(gen_mat_data, metric, engine):
    """
    Test various wrappers for getting nx graph metrics
    """

    binary = False

    in_mat = gen_mat_data(asfile=False, mat_type='sb',
                          binary=binary)['mat_list'][0]

    G = nx.from_numpy_array(in_mat)
    ci = np.ones(in_mat.shape[0])
    metric_list_names = []
    net_met_val_list_final = []

    if metric == 'participation':
        metric_list_names, net_met_val_list_final = \
            algorithms.get_participation(in_mat, ci, metric_list_names,
                                         net_met_val_list_final)
        assert len(metric_list_names) == len(
            algorithms.participation_coef(in_mat, ci)) + 1
        assert len(net_met_val_list_final) == len(
            algorithms.participation_coef(in_mat, ci)) + 1
    elif metric == 'diversity':
        metric_list_names, net_met_val_list_final = \
            algorithms.get_diversity(in_mat, ci, metric_list_names,
                                     net_met_val_list_final)
        assert len(metric_list_names) == \
               np.shape(algorithms.diversity_coef_sign(in_mat, ci))[1] + 1
        assert len(net_met_val_list_final) == \
               np.shape(algorithms.diversity_coef_sign(in_mat, ci))[1] + 1
    elif metric == 'local_efficiency':
        metric_list_names, net_met_val_list_final = \
            algorithms.get_local_efficiency(G, metric_list_names,
                                            net_met_val_list_final)
        assert len(metric_list_names) == \
               len(algorithms.local_efficiency(G, engine=engine)) + 1
        assert len(net_met_val_list_final) == len(algorithms.local_efficiency(
            G, engine=engine)) + 1
    elif metric == 'comm_centrality':
        metric_list_names, net_met_val_list_final = \
            algorithms.get_comm_centrality(G, metric_list_names,
                                           net_met_val_list_final)
        assert len(metric_list_names) == len(
            nx.algorithms.communicability_betweenness_centrality(G)) + 1
        assert len(net_met_val_list_final) == len(
            nx.algorithms.communicability_betweenness_centrality(G)) + 1
    # elif metric == 'rich_club_coeff':
    #     metric_list_names, net_met_val_list_final = \
    #         algorithms.get_rich_club_coeff(G, metric_list_names,
    #                                        net_met_val_list_final)
    #     assert len(metric_list_names) == len(
    #         nx.algorithms.rich_club_coefficient(G)) + 1
    #     assert len(net_met_val_list_final) == len(
    #         nx.algorithms.rich_club_coefficient(G)) + 1
