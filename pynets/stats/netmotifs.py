#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner & James Kunert-Graf
"""
import numpy as np
from copy import copy
import warnings
from collections import Counter
import itertools
warnings.filterwarnings("ignore")


def countmotifs(A, N=4):
    '''
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
    '''
    import gc
    assert N in [3, 4], "Only motifs of size N=3,4 currently supported"
    X2 = np.array([[k] for k in range(A.shape[0]-1)])
    for n in range(N-1):
        X = copy(X2)
        X2 = []
        for vsub in X:
            # in_matind list of nodes neighboring vsub with a larger index than root v
            idx=np.where(np.any(A[(vsub[0]+1):, vsub], 1))[0]+vsub[0]+1
            # Only keep node indices not in vsub
            idx=idx[[k not in vsub for k in idx]]
            if len(idx)>0:
                # If new neighbors found, add all new vsubs to list
                X2.append([np.append(vsub,ik) for ik in idx])
        if len(X2)>0:
            X2 = np.vstack(X2)
        else:
            umotifs = 0
            return umotifs

    X2 = np.sort(X2,1)
    X2 = X2[np.unique(np.ascontiguousarray(X2).view(np.dtype((np.void,
                                                              X2.dtype.itemsize * X2.shape[1]))),
                      return_index=True)[1]]
    umotifs = Counter([''.join(np.sort(np.sum(A[x, :][:, x],
                                              1)).astype(int).astype(str)) for x in X2])
    del X2
    gc.collect()
    return umotifs


def adaptivethresh(in_mat, thr, mlib, N):
    '''
    Counts number of motifs with a given absolute threshold.

    Parameters
    ----------
    in_mat : ndarray
        M x M Connectivity matrix
    thr : float
        Absolute threshold [0, 1].
    mlib : list
        List of motif classes.

    Returns
    -------
    mf : ndarray
        1D vector listing the total motifs of size N for each
        class of mlib.
    '''
    from pynets.stats.netmotifs import countmotifs
    mf = countmotifs((in_mat > thr).astype(int), N=N)
    try:
        mf = np.array([mf[k] for k in mlib])
    except:
        mf = np.zeros(len(mlib))
    return mf


def compare_motifs(struct_mat, func_mat, name, bins=20, N=4):
    from pynets.stats.netmotifs import adaptivethresh
    from pynets.core.thresholding import standardize
    from scipy import spatial
    import pandas as pd
    import gc

    mlib = ['1113', '1122', '1223', '2222', '2233', '3333']

    # Standardize structural graph
    struct_mat = standardize(struct_mat)
    dims_struct = struct_mat.shape[0]
    struct_mat[range(dims_struct), range(dims_struct)] = 0
    at_struct = adaptivethresh(struct_mat, float(0.0), mlib, N)
    print("%s%s%s" % ('Layer 1 (structural) has: ', np.sum(at_struct), ' total motifs'))

    # Functional graph threshold window
    func_mat = standardize(func_mat)
    dims_func = func_mat.shape[0]
    func_mat[range(dims_func), range(dims_func)] = 0
    tmin_func = func_mat.min()
    tmax_func = func_mat.max()
    threshes_func = np.linspace(tmin_func, tmax_func, bins)

    assert np.all(struct_mat == struct_mat.T), "Structural Matrix must be symmetric"
    assert np.all(func_mat == func_mat.T), "Functional Matrix must be symmetric"

    # Count motifs
    print("%s%s%s%s" % ('Mining ', N, '-node motifs: ', mlib))
    motif_dict = {}
    motif_dict['struct'] = {}
    motif_dict['func'] = {}
    for thr_func in threshes_func:
        # Count
        at_func = adaptivethresh(func_mat, float(thr_func), mlib, N)
        motif_dict['struct']["%s%s" % ('struct_func_', np.round(thr_func, 4))] = at_struct
        motif_dict['func']["%s%s" % ('struct_func_', np.round(thr_func, 4))] = at_func

        print("%s%s%s%s%s" % ('Layer 2 (functional) with absolute threshold of: ',
                              np.round(thr_func, 2), ' yields ',
                              np.sum(at_func), ' total motifs'))
        gc.collect()

    df = pd.DataFrame(motif_dict)

    for idx in range(len(df)):
        df.set_value(df.index[idx], 'dist', spatial.distance.cosine(df['struct'][idx], df['func'][idx]))

    df = df[pd.notnull(df['dist'])]

    df['struct_func_3333'] = np.zeros(len(df))
    df['struct_func_2233'] = np.zeros(len(df))
    df['struct_func_2222'] = np.zeros(len(df))
    df['struct_func_1223'] = np.zeros(len(df))
    df['struct_func_1122'] = np.zeros(len(df))
    df['struct_func_1113'] = np.zeros(len(df))
    df['struct_3333'] = np.zeros(len(df))
    df['func_3333'] = np.zeros(len(df))
    df['struct_2233'] = np.zeros(len(df))
    df['func_2233'] = np.zeros(len(df))
    df['struct_2222'] = np.zeros(len(df))
    df['func_2222'] = np.zeros(len(df))
    df['struct_1223'] = np.zeros(len(df))
    df['func_1223'] = np.zeros(len(df))
    df['struct_1122'] = np.zeros(len(df))
    df['func_1122'] = np.zeros(len(df))
    df['struct_1113'] = np.zeros(len(df))
    df['func_1113'] = np.zeros(len(df))

    for idx in range(len(df)):
        df.set_value(df.index[idx], 'struct_3333', df['struct'][idx][-1])
        df.set_value(df.index[idx], 'func_3333', df['func'][idx][-1])

        df.set_value(df.index[idx], 'struct_2233', df['struct'][idx][-2])
        df.set_value(df.index[idx], 'func_2233', df['func'][idx][-2])

        df.set_value(df.index[idx], 'struct_2222', df['struct'][idx][-3])
        df.set_value(df.index[idx], 'func_2222', df['func'][idx][-3])

        df.set_value(df.index[idx], 'struct_1223', df['struct'][idx][-4])
        df.set_value(df.index[idx], 'func_1223', df['func'][idx][-4])

        df.set_value(df.index[idx], 'struct_1122', df['struct'][idx][-5])
        df.set_value(df.index[idx], 'func_1122', df['func'][idx][-5])

        df.set_value(df.index[idx], 'struct_1113', df['struct'][idx][-6])
        df.set_value(df.index[idx], 'func_1113', df['func'][idx][-6])

    df['struct_func_3333'] = np.abs(df['struct_3333'] - df['func_3333'])
    df['struct_func_2233'] = np.abs(df['struct_2233'] - df['func_2233'])
    df['struct_func_2222'] = np.abs(df['struct_2222'] - df['func_2222'])
    df['struct_func_1223'] = np.abs(df['struct_1223'] - df['func_1223'])
    df['struct_func_1122'] = np.abs(df['struct_1122'] - df['func_1122'])
    df['struct_func_1113'] = np.abs(df['struct_1113'] - df['func_1113'])

    df = df.drop(columns=['struct', 'func'])

    df = df.loc[~(df==0).all(axis=1)]

    df = df.sort_values(by=['dist', 'struct_func_3333', 'struct_func_2233', 'struct_func_2222', 'struct_func_1223',
                            'struct_func_1122', 'struct_func_1113', 'struct_3333', 'func_3333', 'struct_2233',
                            'func_2233', 'struct_2222', 'func_2222', 'struct_1223', 'func_1223', 'struct_1122',
                            'func_1122', 'struct_1113', 'func_1113'], ascending=[True, True, True, True, True, True,
                                                                                 True, False, False, False, False,
                                                                                 False, False, False, False, False,
                                                                                 False, False, False])

    # Take the top 25th percentile
    df = df[df['dist'] <= df['dist'].quantile(0.25)]
    best_threshes = []
    best_mats = []
    #best_graphs = []
    best_multigraphs = []
    for key in list(df.index):
        func_mat_tmp = func_mat.copy()
        struct_mat_tmp = struct_mat.copy()
        struct_thr = float(key.split('_')[-1])
        func_thr = float(key.split('_')[-1])
        best_threshes.append((struct_thr, func_thr))

        func_mat_tmp[func_mat_tmp < func_thr] = 0
        struct_mat_tmp[struct_mat_tmp < struct_thr] = 0
        best_mats.append((func_mat_tmp, struct_mat_tmp))

        G = build_nx_multigraph(func_mat, struct_mat, key)
        best_multigraphs.append(G)

    mg_dict = dict(zip(best_threshes, best_multigraphs))

    return mg_dict


def build_nx_multigraph(func_mat, struct_mat, name):
    import networkx as nx
    G_struct = nx.from_numpy_matrix(struct_mat)
    G_func = nx.from_numpy_matrix(func_mat)
    G = nx.MultiGraph()
    G.name = name
    G.add_weighted_edges_from(G_struct.edges(data=True), color='green')
    G.add_weighted_edges_from(G_func.edges(data=True), color='red')
    return G


def build_multigraphs(est_path_iterlist, ID):
    """
    Constructs a multimodal multigraph for each available resolution of vertices.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy file containing graph.
    ID : str
        A subject id or other unique identifier.

    Returns
    -------
    ml_graph_path_list : list
        List of path strings to multilayer graph edgelists.
    """
    import yaml
    import re
    import os
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    from pathlib import Path
    from pynets.stats.netstats import community_resolution_selection

    # Available functional and structural connectivity models
    # with open('/Users/derekpisner/Applications/PyNets/pynets/runconfig.yaml', 'r') as stream:
    with open("%s%s" % (str(Path(__file__).parent.parent), '/runconfig.yaml'), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        try:
            func_models = hardcoded_params['available_models']['func_models']
        except KeyError:
            print('ERROR: available functional models not sucessfully extracted from runconfig.yaml')
        try:
            struct_models = hardcoded_params['available_models']['struct_models']
        except KeyError:
            print('ERROR: available structural models not sucessfully extracted from runconfig.yaml')

    atlases = list(set([x.split('/')[-3].split('/')[0] for x in est_path_iterlist]))
    parcel_dict_func = dict.fromkeys(atlases)
    parcel_dict_dwi = dict.fromkeys(atlases)
    est_path_iterlist_dwi = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                      struct_models]))
    est_path_iterlist_func = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                       func_models]))

    func_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_func]))
    dwi_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_dwi]))

    multigraph_list_all = []
    for atlas in atlases:
        if len(func_subnets) > 1:
            parcel_dict_func[atlas] = {}
            for sub_net in func_subnets:
                parcel_dict_func[atlas][sub_net] = []
        else:
            parcel_dict_func[atlas] = []

        if len(dwi_subnets) > 1:
            parcel_dict_dwi[atlas] = {}
            for sub_net in dwi_subnets:
                parcel_dict_dwi[atlas][sub_net] = []
        else:
            parcel_dict_dwi[atlas] = []

        for graph_path in est_path_iterlist_dwi:
            if atlas in graph_path:
                if len(dwi_subnets) > 1:
                    for sub_net in dwi_subnets:
                        if sub_net in graph_path:
                            parcel_dict_dwi[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_dwi[atlas].append(graph_path)

        for graph_path in est_path_iterlist_func:
            if atlas in graph_path:
                if len(func_subnets) > 1:
                    for sub_net in func_subnets:
                        if sub_net in graph_path:
                            parcel_dict_func[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_func[atlas].append(graph_path)

        parcel_dict = {}
        # Create dictionary of all possible pairs of structural-functional graphs for each unique resolution
        # of vertices
        for res in list(set([i for i in parcel_dict_dwi.keys() if i in parcel_dict_func.keys()])):
            parcel_dict[res] = list(set(itertools.product(parcel_dict_dwi[res], parcel_dict_func[res])))

        dir_path = str(Path(os.path.dirname(est_path_iterlist_dwi[0])).parent.parent.parent)
        namer_dir = f"{dir_path}/graphs_multilayer"
        if not os.path.isdir(namer_dir):
            os.mkdir(namer_dir)
        ml_graph_path_list = []
        multigraph_list = []
        for res in list(parcel_dict.keys()):
            for struct_graph_path, func_graph_path in parcel_dict[res]:
                struct_mat = np.load(struct_graph_path)
                func_mat = np.load(func_graph_path)
                func_mat[~struct_mat.astype('bool')] = 0
                struct_mat[~func_mat.astype('bool')] = 0

                struct_mat = nx.to_numpy_array(sorted(nx.connected_component_subgraphs(nx.from_numpy_matrix(
                    struct_mat)), key=len, reverse=True)[0])

                func_mat = nx.to_numpy_array(sorted(nx.connected_component_subgraphs(nx.from_numpy_matrix(
                    func_mat)), key=len, reverse=True)[0])

                struct_node_comm_aff_mat = community_resolution_selection(
                    nx.from_numpy_matrix(np.abs(struct_mat)))[1]

                func_node_comm_aff_mat = community_resolution_selection(
                    nx.from_numpy_matrix(np.abs(func_mat)))[1]

                struct_comms = []
                for i in np.unique(struct_node_comm_aff_mat):
                    struct_comms.append(struct_node_comm_aff_mat==i)

                func_comms = []
                for i in np.unique(func_node_comm_aff_mat):
                    func_comms.append(func_node_comm_aff_mat==i)

                sims = cosine_similarity(struct_comms, func_comms)
                struct_comm = struct_comms[np.argmax(sims, axis=1)[0]]
                func_comm = func_comms[np.argmax(sims, axis=0)[0]]

                comm_mask = np.equal.outer(struct_comm, func_comm).astype(bool)
                struct_mat[~comm_mask] = 0
                func_mat[~comm_mask] = 0
                # Truncate names
                struct_name = re.sub(r'^(.{25}).*$', '\g<1>...',
                                     struct_graph_path.split('/')[-1].split('_raw.npy')[0])
                func_name = re.sub(r'^(.{25}).*$', '\g<1>...',
                                   func_graph_path.split('/')[-1].split('_raw.npy')[0])
                name = f"{ID}_{res}_multigraph_LAYER1_{struct_name}_LAYER2_{func_name}"

                struct_mat = np.maximum(struct_mat, struct_mat.T)
                func_mat = np.maximum(func_mat, func_mat.T)
                mldict = compare_motifs(struct_mat, func_mat, name)
                multigraph_list.append(mldict)
                for thr in list(mldict.keys()):
                    multigraph = mldict[thr]
                    out_path = f"{namer_dir}/struct_func_mtlayer_{atlas}_{name}_motif-{thr[0]}_{thr[1]}.edgelist"
                    nx.write_edgelist(multigraph, out_path)
                    ml_graph_path_list.append(out_path)
        multigraph_list_all.append(ml_graph_path_list)

    return multigraph_list_all
