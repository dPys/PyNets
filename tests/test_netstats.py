#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import networkx as nx
import time
from pathlib import Path
from pynets.stats import netstats
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_average_shortest_path_length_for_all():
    """
    Test for average_shortest_path_length_for_all functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    avgest_path_len = netstats.average_shortest_path_length_for_all(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert avgest_path_len is not None


def test_average_local_efficiency():
    """
    Test for average_local_efficiency functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    average_local_efficiency = netstats.average_local_efficiency(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert average_local_efficiency is not None


# used random node_comm_aff_mat
def test_create_communities():
    """
    Test for create_communities functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    node_comm_aff_mat = np.random.rand(5,5)
    node_num = 3

    start_time = time.time()
    com_assign = netstats.create_communities(node_comm_aff_mat, node_num)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))
    assert com_assign is not None


def test_participation_coef():
    """
    Test for participation_coef functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    P = netstats.participation_coef(W, ci, degree='undirected')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert P is not None


def test_modularity():
    """
    Test for modularity functionality
    """
    import community
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    G = nx.from_numpy_matrix(in_mat)
    start_time = time.time()
    ci = community.best_partition(G)
    mod = community.community_louvain.modularity(ci, G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert ci is not None
    assert mod is not None


def test_diversity_coef_sign():
    """
    Test for diversity_coef_sign functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
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
    """
    Test for link_communities functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")

    start_time = time.time()
    M = netstats.link_communities(in_mat, type_clustering='single')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert M is not None


def test_prune_disconnected():
    """
    Test pruning functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [G, pruned_nodes] = netstats.prune_disconnected(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert G is not None
    assert pruned_nodes is not None


def test_most_important():
    """
    Test pruning for most important nodes functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy")
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [Gt, pruned_nodes] = netstats.most_important(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert Gt is not None
    assert pruned_nodes is not None


@pytest.mark.parametrize("binary", ['True', 'False'])
@pytest.mark.parametrize("prune", ['0', '1', '2'])
@pytest.mark.parametrize("norm", ['0', '1', '2', '3', '4', '5', '6'])
def test_extractnetstats(binary, prune, norm):
    """
    Test extractnetstats functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    ID = '002'
    network = 'Default'
    thr = 0.95
    conn_model = 'cov'
    est_path = f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-sps_thrtype-DENS_thr-0.19.npy"
    prune = 1
    norm = 1
    binary = False
    roi = None

    start_time = time.time()
    out_path = netstats.extractnetstats(ID, network, thr, conn_model, est_path, roi, prune, norm, binary)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    assert out_path is not None


def test_raw_mets():
    """
    Test raw_mets extraction functionality
    """
    from pynets.stats.netstats import global_efficiency, average_local_efficiency
    from networkx.algorithms import degree_assortativity_coefficient, average_clustering, \
        average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, \
        sigma
    base_dir = str(Path(__file__).parent/"examples")
    est_path = f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-sps_thrtype-DENS_thr-0.19.npy"
    in_mat = np.load(est_path)
    G = nx.from_numpy_array(in_mat)
    [G, _] = netstats.prune_disconnected(G)
    metric_list_glob = [global_efficiency, average_local_efficiency, degree_assortativity_coefficient,
                        average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient,
                        graph_number_of_cliques, transitivity]
    for i in metric_list_glob:
        net_met_val = netstats.raw_mets(G, i)
        print(i)
        print(net_met_val)
        assert net_met_val is not np.nan
