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
from pynets import netstats


def test_average_shortest_path_length_for_all():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    avg_shortest_path_len = netstats.average_shortest_path_length_for_all(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert avg_shortest_path_len is not None

def test_global_efficiency():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    global_eff = netstats.global_efficiency(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert global_eff is not None

def test_local_efficiency():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)
    efficiencies = netstats.local_efficiency(G, weight=None)

    start_time = time.time()
    netstats.local_efficiency(G, weight=None)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    # for i in efficiencies:
    #     assert i is not None
    assert efficiencies is not None

def test_average_local_efficiency():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    average_local_efficiency = netstats.average_local_efficiency(G, weight=None)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert average_local_efficiency is not None

def test_create_random_graph():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)
    n = 10
    p = 0.5

    start_time = time.time()
    rG = netstats.create_random_graph(G, n, p)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert rG is not None

def test_create_random_graph():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)
    n = 10
    p = 0.5

    start_time = time.time()
    rG = netstats.create_random_graph(G, n, p)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert rG is not None

def test_smallworldness_measure():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)
    n = 10
    p = 0.5

    start_time = time.time()
    rG = netstats.create_random_graph(G, n, p)
    swm = netstats.smallworldness_measure(G, rG)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert swm is not None

# def test_smallworldness():
#     base_dir = str(Path(__file__).parent/"examples")
#     in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
#     G = nx.from_numpy_array(in_mat)
#
#     start_time = time.time()
#     mean_s = netstats.smallworldness(G, rep=1000)
#     print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
#     assert mean_s is not None

# used random node_comm_aff_mat
def test_create_communities():
    base_dir = str(Path(__file__).parent/"examples")
    node_comm_aff_mat = np.random.rand(5,5)
    node_num = 3

    start_time = time.time()
    com_assign = netstats.create_communities(node_comm_aff_mat, node_num)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert com_assign is not None

def test_compute_rc():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    rc = netstats._compute_rc(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert rc is not None

def test_participation_coef():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    P = netstats.participation_coef(W, ci, degree='undirected')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert P is not None

def test_modularity():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')

    start_time = time.time()
    [ci, mod] = netstats.modularity(in_mat, qtype='sta', seed=42)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert ci is not None
    assert mod is not None

def test_diversity_coef_sign():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    ci = np.ones(in_mat.shape[0])
    ci_dim = int(np.shape(ci)[0])
    W = np.random.rand(ci_dim, ci_dim)

    start_time = time.time()
    [Hpos, Hneg] = netstats.diversity_coef_sign(W, ci)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert Hpos is not None
    assert Hneg is not None

#Rerun, too slow
def test_link_communities():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')

    start_time = time.time()
    M = netstats.link_communities(in_mat, type_clustering='single')
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert M is not None

def test_modularity_louvain_und_sign():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')

    start_time = time.time()
    [ci, mod] = netstats.modularity_louvain_und_sign(in_mat, gamma=1, qtype='sta', seed=42)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert ci is not None
    assert mod is not None

def test_prune_disconnected():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [G, pruned_nodes] = netstats.prune_disconnected(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert G is not None
    assert pruned_nodes is not None

def test_most_important():
    base_dir = str(Path(__file__).parent/"examples")
    in_mat = np.load(base_dir + '/997/997_Default_est_cov_0.1_4.npy')
    G = nx.from_numpy_array(in_mat)

    start_time = time.time()
    [Gt, pruned_nodes] = netstats.most_important(G)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert Gt is not None
    assert pruned_nodes is not None

def test_extractnetstats():
    base_dir = str(Path(__file__).parent/"examples")
    ID = '997'
    network = 'Default'
    thr = 0.95
    conn_model = 'cov'
    est_path = base_dir + '/997/997_Default_est_cov_0.95prop_TESTmm_2fwhm.npy'
    mask = None
    prune = 1
    node_size = 'parc'
    smooth = 2
    c_boot = 0

    start_time = time.time()
    out_path = netstats.extractnetstats(ID, network, thr, conn_model, est_path, mask, prune, node_size, smooth, c_boot)
    print("%s%s%s" % ('thresh_and_fit (Functional, proportional thresholding) --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))
    assert out_path is not None
