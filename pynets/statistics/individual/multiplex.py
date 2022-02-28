#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner & James Kunert-Graf
"""
import matplotlib
import numpy as np
import warnings
import os
import gc
from pathlib import Path

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def optimize_mutual_info(dwi_mat, func_mat, bins=20):
    """

    """
    import itertools
    from pynets.core.thresholding import threshold_absolute
    import pandas as pd

    # Functional graph threshold window
    threshes_dwi = np.linspace(dwi_mat.min(), dwi_mat.max(), bins)

    # Functional graph threshold window
    threshes_func = np.linspace(func_mat.min(), func_mat.max(), bins)

    mutual_info_dict = {}

    all_thr_combos = list(itertools.product(threshes_func, threshes_dwi))
    for thr_func, thr_dwi in all_thr_combos:
        X = threshold_absolute(dwi_mat, thr_dwi)
        Y = threshold_absolute(func_mat, thr_func)
        mutual_info_dict[f"func-{round(thr_func, 2)}_" \
                         f"dwi-{round(thr_dwi, 2)}"] = mutual_information_2d(
            X.ravel(), Y.ravel())

    df = pd.DataFrame(mutual_info_dict, index=range(1))
    df = df.loc[:, (df > 0.01).any(axis=0)]
    df = df.T.sort_values(by=0, ascending=False)
    best_thresh_combo = list(df.index)[0]

    best_dwi_thr = float(best_thresh_combo.split("-")[1].split('_')[0])
    best_func_thr = float(best_thresh_combo.split("-")[2].split('_')[0])

    return {best_dwi_thr: threshold_absolute(dwi_mat, best_dwi_thr)}, \
           {best_func_thr: threshold_absolute(func_mat, best_func_thr)}, \
           df.head(1)[0].values[0]


def build_mx_multigraph(func_mat, dwi_mat, name, namer_dir):
    """
    It creates a symmetric (undirected) MultilayerGraph object from
    vertex-aligned structural and functional connectivity matrices.

    Parameters:
    -----------
    func_mat : ndarray
        Functional adjacency matrix of size N x N.
    dwi_mat : ndarray
        Structural adjacency matrix of size N x N.
    name : str
        Intended name of the multiplex object.
    namer_dir : str
        Path to output directory.

    Returns:
    --------
    mg_path : str
        A filepath to a gpickled MultilayerGraph object

    References
    ----------
    .. [1] R. Amato, N. E Kouvaris, M. San Miguel and A. Diaz-Guilera,
      Opinion competition dynamics on multiplex networks, New J. Phys.
      DOI: https://doi.org/10.1088/1367-2630/aa936a
    .. [2] N. E. Kouvaris, S. Hata and A. Diaz-Guilera, Pattern formation
      in multiplex networks, Scientific Reports 5, 10840 (2015).
      http://www.nature.com/srep/2015/150604/srep10840/full/srep10840.html
    .. [3] A. Sole-Ribata, M. De Domenico, N. E. Kouvaris, A. Diaz-Guilera, S.
      Gomez and A. Arenas, Spectral properties of the Laplacian of a multiplex
      network, Phys. Rev. E 88, 032807 (2013).
      http://journals.aps.org/pre/abstract/10.1103/PhysRevE.88.032807

    """
    import networkx as nx
    import multinetx as mx

    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    mg = mx.MultilayerGraph()
    N = dwi_mat.shape[0]
    adj_block = mx.lil_matrix(np.zeros((N * 2, N * 2)))
    adj_block[0:N, N: 2 * N] = np.identity(N)
    adj_block += adj_block.T
    G_dwi = nx.from_numpy_matrix(dwi_mat)
    G_func = nx.from_numpy_matrix(func_mat)
    mg.add_layer(G_dwi)
    mg.add_layer(G_func)
    mg.layers_interconnect(inter_adjacency_matrix=adj_block)
    mg.name = name

    # Save mG to pickle
    graph_dir = f"{namer_dir}/mplx_graphs"
    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)
    mG_path = f"{graph_dir}/{name[:200]}_mG.pkl"
    nx.write_gpickle(mg, mG_path, protocol=2)

    return mG_path


def mutual_information_2d(x, y, sigma=1, normalized=True):
    """
    Computes (normalized) mutual information between two 2D variate from a
    joint histogram.

    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram

    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    from scipy import ndimage
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + np.finfo(float).eps
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi


def matching(
    paths,
    atlas,
    namer_dir,
):
    import networkx as nx
    import numpy as np
    import glob
    from pynets.core import thresholding
    from pynets.statistics.utils import parse_closest_ixs
    from graspologic.utils import remove_loops, symmetrize, \
        multigraph_lcc_intersection

    [dwi_graph_path, func_graph_path] = paths
    dwi_mat = np.load(dwi_graph_path)
    func_mat = np.load(func_graph_path)
    dwi_mat = thresholding.autofix(symmetrize(remove_loops(dwi_mat)))
    func_mat = thresholding.autofix(symmetrize(remove_loops(func_mat)))
    dwi_mat = thresholding.standardize(dwi_mat)
    func_mat = thresholding.standardize(func_mat)

    node_dict_dwi = parse_closest_ixs(
        glob.glob(f"{str(Path(dwi_graph_path).parent.parent)}"
                  f"/nodes/*"), dwi_mat.shape[0])[1]

    node_dict_func = parse_closest_ixs(
        glob.glob(f"{str(Path(func_graph_path).parent.parent)}"
                  f"/nodes/*"), func_mat.shape[0])[1]

    G_dwi = nx.from_numpy_array(dwi_mat)
    nx.set_edge_attributes(G_dwi, 'structural',
                            nx.get_edge_attributes(G_dwi, 'weight').values())
    nx.set_node_attributes(G_dwi, dict(node_dict_dwi), name='dwi')

    #G_dwi.nodes(data=True)

    G_func = nx.from_numpy_array(func_mat)
    nx.set_edge_attributes(G_func, 'functional',
                           nx.get_edge_attributes(G_func, 'weight').values())
    nx.set_node_attributes(G_func, dict(node_dict_func), name='func')
    #G_func.nodes(data=True)

    R = G_dwi.copy()
    R.remove_nodes_from(n for n in G_dwi if n not in G_func)
    R.remove_edges_from(e for e in G_dwi.edges if e not in G_func.edges)
    G_dwi = R.copy()

    R = G_func.copy()
    R.remove_nodes_from(n for n in G_func if n not in G_dwi)
    R.remove_edges_from(e for e in G_func.edges if e not in G_dwi.edges)
    G_func = R.copy()

    [G_dwi, G_func] = multigraph_lcc_intersection([G_dwi, G_func])

    dwi_name = dwi_graph_path.split("/rawgraph_"
                                          )[-1].split(".npy")[0]
    func_name = func_graph_path.split("/rawgraph_")[-1].split(".npy")[0]
    name = f"{atlas}_mplx_Layer-1_{dwi_name[0:30]}_" \
           f"Layer-2_{func_name[0:30]}"

    dwi_opt, func_opt, best_mi = optimize_mutual_info(
        nx.to_numpy_array(G_dwi), nx.to_numpy_array(G_func), bins=50)

    func_mat_final = list(func_opt.values())[0]
    dwi_mat_final = list(dwi_opt.values())[0]
    G_dwi_final = nx.from_numpy_array(dwi_mat_final)
    G_func_final = nx.from_numpy_array(func_mat_final)

    G_multi = nx.OrderedMultiGraph(nx.compose(G_dwi_final, G_func_final))

    mG = build_mx_multigraph(
        nx.to_numpy_array(G_func_final),
        nx.to_numpy_array(G_dwi_final),
        f"{name}_{list(dwi_opt.keys())[0]}_{list(func_opt.keys())[0]}",
        namer_dir)

    mG_nx = f"{namer_dir}/{name}_dwiThr-{list(dwi_opt.keys())[0]}_" \
            f"funcThr-{list(func_opt.keys())[0]}.gpickle"
    nx.write_gpickle(G_multi, mG_nx)

    out_dwi_mat = f"{namer_dir}/dwi-{name[0:30]}thr-" \
                  f"{list(dwi_opt.keys())[0]}.npy"
    out_func_mat = f"{namer_dir}/func-{name[0:30]}thr-" \
                   f"{list(func_opt.keys())[0]}.npy"
    np.save(out_dwi_mat, dwi_mat_final)
    np.save(out_func_mat, func_mat_final)
    return mG_nx, mG, out_dwi_mat, out_func_mat


def build_multigraphs(est_path_iterlist):
    """
    Constructs a multimodal multigraph for each available resolution of
    vertices.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy file containing graph.

    Returns
    -------
    multigraph_list_all : list
        List of multiplex graph dictionaries corresponding to
        each unique node resolution.
    graph_path_list_top : list
        List of lists consisting of pairs of most similar
        structural and functional connectomes for each unique node resolution.

    References
    ----------
    .. [1] Bullmore, E., & Sporns, O. (2009). Complex brain networks: Graph
      theoretical analysis of structural and functional systems.
      Nature Reviews Neuroscience. https://doi.org/10.1038/nrn2575
    .. [2] Vaiana, M., & Muldoon, S. F. (2018). Multilayer Brain Networks.
      Journal of Nonlinear Science. https://doi.org/10.1007/s00332-017-9436-8

    """
    import os
    import itertools
    import numpy as np
    from pathlib import Path
    from pynets.core.utils import flatten, load_runconfig

    raw_est_path_iterlist = list(
        set(
            [
                os.path.dirname(i) + '/raw' + os.path.basename(i).split(
                    "_thrtype")[0] + ".npy"
                for i in list(flatten(est_path_iterlist))
            ]
        )
    )

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    try:
        func_models = hardcoded_params["available_models"]["func_models"]
    except KeyError:
        print(
            "ERROR: available functional models not sucessfully extracted"
            " from advanced.yaml"
        )
    try:
        dwi_models = hardcoded_params["available_models"][
            "dwi_models"]
    except KeyError:
        print(
            "ERROR: available structural models not sucessfully extracted"
            " from advanced.yaml"
        )

    atlases = list(set([x.split("/")[-3].split("/")[0]
                        for x in raw_est_path_iterlist]))
    parcel_dict_func = dict.fromkeys(atlases)
    parcel_dict_dwi = dict.fromkeys(atlases)
    est_path_iterlist_dwi = list(
        set(
            [
                i
                for i in raw_est_path_iterlist
                if i.split("model-")[1].split("_")[0] in dwi_models
            ]
        )
    )
    est_path_iterlist_func = list(
        set(
            [
                i
                for i in raw_est_path_iterlist
                if i.split("model-")[1].split("_")[0] in func_models
            ]
        )
    )

    if "_subnet" in ";".join(est_path_iterlist_func):
        func_subnets = list(
            set([i.split("_subnet-")[1].split("_")[0] for i in
                 est_path_iterlist_func])
        )
    else:
        func_subnets = []
    if "_subnet" in ";".join(est_path_iterlist_dwi):
        dwi_subnets = list(
            set([i.split("_subnet-")[1].split("_")[0] for i in
                 est_path_iterlist_dwi])
        )
    else:
        dwi_subnets = []

    dir_path = str(
        Path(
            os.path.dirname(
                est_path_iterlist_dwi[0])).parent.parent.parent)
    namer_dir = f"{dir_path}/graphs_multilayer"
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    multigraph_list_all = []
    graph_path_list_all = []
    for atlas in atlases:
        if len(func_subnets) >= 1:
            parcel_dict_func[atlas] = {}
            for sub_net in func_subnets:
                parcel_dict_func[atlas][sub_net] = []
        else:
            parcel_dict_func[atlas] = []

        if len(dwi_subnets) >= 1:
            parcel_dict_dwi[atlas] = {}
            for sub_net in dwi_subnets:
                parcel_dict_dwi[atlas][sub_net] = []
        else:
            parcel_dict_dwi[atlas] = []

        for graph_path in est_path_iterlist_dwi:
            if atlas in graph_path:
                if len(dwi_subnets) >= 1:
                    for sub_net in dwi_subnets:
                        if sub_net in graph_path:
                            parcel_dict_dwi[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_dwi[atlas].append(graph_path)

        for graph_path in est_path_iterlist_func:
            if atlas in graph_path:
                if len(func_subnets) >= 1:
                    for sub_net in func_subnets:
                        if sub_net in graph_path:
                            parcel_dict_func[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_func[atlas].append(graph_path)

        parcel_dict = {}
        # Create dictionary of all possible pairs of structural-functional
        # graphs for each unique resolution of vertices
        if len(dwi_subnets) >= 1 and len(func_subnets) >= 1:
            parcel_dict[atlas] = {}
            subnets = np.intersect1d(dwi_subnets, func_subnets).tolist()
            for subnet in subnets:
                parcel_dict[atlas][subnet] = list(set(itertools.product(
                    parcel_dict_dwi[atlas][subnet],
                    parcel_dict_func[atlas][subnet])))
                for paths in list(parcel_dict[atlas][subnet]):
                    [
                        mG_nx,
                        mG,
                        out_dwi_mat,
                        out_func_mat
                    ] = matching(
                        paths,
                        atlas,
                        namer_dir,
                    )
        else:
            parcel_dict[atlas] = list(set(itertools.product(
                parcel_dict_dwi[atlas], parcel_dict_func[atlas])))
            for paths in list(parcel_dict[atlas]):
                [
                    mG_nx,
                    mG,
                    out_dwi_mat,
                    out_func_mat
                ] = matching(
                    paths,
                    atlas,
                    namer_dir,
                )
        multigraph_list_all.append((mG_nx, mG))
        graph_path_list_all.append((out_dwi_mat, out_func_mat))

    return (
        multigraph_list_all,
        graph_path_list_all,
        len(multigraph_list_all) * [namer_dir],
    )
