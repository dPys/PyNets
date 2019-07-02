#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import networkx as nx
import time
from pathlib import Path
from pynets.stats import netstats


def test_average_shortest_path_length_for_all():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    avg_shortest_path_len = netstats.average_shortest_path_length_for_all(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert avg_shortest_path_len is not None


def test_average_local_efficiency():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    average_local_efficiency = netstats.average_local_efficiency(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert average_local_efficiency is not None


# used random node_comm_aff_mat
def test_create_communities():
    base_dir = str(Path(__file__).parent/"examples")
    node_comm_aff_mat = np.random.rand(5,5)
    node_num = 3

    start_time = time.time()
    com_assign = netstats.create_communities(node_comm_aff_mat, node_num)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert com_assign is not None


def test_participation_coef():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    P = netstats.participation_coef(W, ci, degree='undirected')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert P is not None


def test_modularity():
    import community
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    G = nx.from_numpy_matrix(in_mat)
    start_time = time.time()
    ci = community.best_partition(G)
    mod = community.community_louvain.modularity(ci, G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert ci is not None
    assert mod is not None


def test_diversity_coef_sign():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    [Hpos, Hneg] = netstats.diversity_coef_sign(W, ci)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert Hpos is not None
    assert Hneg is not None


def test_link_communities():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')

    start_time = time.time()
    M = netstats.link_communities(in_mat, type_clustering='single')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert M is not None


def test_prune_disconnected():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [G, pruned_nodes] = netstats.prune_disconnected(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert G is not None
    assert pruned_nodes is not None


def test_most_important():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [Gt, pruned_nodes] = netstats.most_important(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert Gt is not None
    assert pruned_nodes is not None


def test_extractnetstats():
    base_dir = str(Path(__file__).parent/"examples")
    ID = '002'
    network = 'Default'
    thr = 0.95
    conn_model = 'cov'
    est_path = base_dir + '/002/fmri/002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy'
    prune = 1
    node_size = 'parc'
    norm = 1
    binary = False
    roi = None

    start_time = time.time()
    out_path = netstats.extractnetstats(ID, network, thr, conn_model, est_path, roi, prune, node_size, norm, binary)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert out_path is not None


def test_raw_mets():
    from pynets.stats.netstats import global_efficiency, average_local_efficiency
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, sigma
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.genfromtxt(base_dir + '/002/fmri/whole_brain_cluster_labels_PCA200/002_est_sps_raw_mat.txt')
    G = nx.from_numpy_array(in_mat)
    [G, _] = netstats.prune_disconnected(G)
    custom_weight = None
    metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient,
                        average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient,
                        graph_number_of_cliques, transitivity]
    for i in metric_list_glob:
        net_met_val = netstats.raw_mets(G, i, custom_weight)
        print(net_met_val)
        assert net_met_val is not np.nan
