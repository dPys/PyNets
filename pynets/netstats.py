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
