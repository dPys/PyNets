"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def optimize_mutual_info(
    dwi_mat: np.ndarray, func_mat: np.ndarray, bins: int = 20
):
    """ """
    import itertools
    import pandas as pd
    from pynets.core.thresholding import threshold_proportional

    # Functional graph threshold window
    threshes_dwi = np.linspace(dwi_mat.min(), dwi_mat.max(), bins)

    # Functional graph threshold window
    threshes_func = np.linspace(func_mat.min(), func_mat.max(), bins)

    mutual_info_dict = {}

    all_thr_combos = list(itertools.product(threshes_func, threshes_dwi))
    for thr_func, thr_dwi in all_thr_combos:
        X = threshold_proportional(dwi_mat, thr_dwi)
        Y = threshold_proportional(func_mat, thr_func)
        mutual_info_dict[
            f"func-{round(thr_func, 2)}_" f"dwi-{round(thr_dwi, 2)}"
        ] = mutual_information_2d(X.ravel(), Y.ravel())

    df = pd.DataFrame(mutual_info_dict, index=range(1))
    df = df.loc[:, (df > 0.01).any(axis=0)]
    df = df.T.sort_values(by=0, ascending=False)
    if len(list(df.index)) == 0:
        raise ValueError(
            "No connections found in either or both of func or " "dwi graphs!"
        )

    best_thresh_combo = list(df.index)[0]

    best_dwi_thr = float(best_thresh_combo.split("-")[1].split("_")[0])
    best_func_thr = float(best_thresh_combo.split("-")[2].split("_")[0])

    return (
        {best_dwi_thr: threshold_proportional(dwi_mat, best_dwi_thr)},
        {best_func_thr: threshold_proportional(func_mat, best_func_thr)},
        df.head(1)[0].values[0],
    )


def build_mx_multigraph(
    func_mat: np.ndarray, dwi_mat: np.ndarray, name: str, namer_dir: str
):
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
    import multinetx as mx
    import networkx as nx

    mg = mx.MultilayerGraph()
    N = dwi_mat.shape[0]
    adj_block = mx.lil_matrix(np.zeros((N * 2, N * 2)))
    adj_block[0:N, N : 2 * N] = np.identity(N)
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
    mG_path = f"{graph_dir}/{name}.pkl"
    nx.write_gpickle(mg, mG_path, protocol=2)

    return mG_path


def mutual_information_2d(
    x: np.ndarray, y: np.ndarray, sigma: int = 1, normalized: bool = True
):
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
    ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

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
        mi = (
            (np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
            / np.sum(jh * np.log(jh))
        ) - 1
    else:
        mi = (
            np.sum(jh * np.log(jh))
            - np.sum(s1 * np.log(s1))
            - np.sum(s2 * np.log(s2))
        )

    return mi


def matching(
    paths: list,
    atlas: str,
    namer_dir: str,
):
    import glob
    import networkx as nx
    import numpy as np
    from graspologic.utils import (
        multigraph_lcc_intersection,
        remove_loops,
        symmetrize,
    )
    from pynets.core import thresholding
    from pynets.core.nodemaker import parse_closest_ixs

    [dwi_graph_path, func_graph_path] = paths
    dwi_mat = np.load(dwi_graph_path)
    func_mat = np.load(func_graph_path)
    dwi_mat = thresholding.autofix(symmetrize(remove_loops(dwi_mat)))
    func_mat = thresholding.autofix(symmetrize(remove_loops(func_mat)))
    dwi_mat = thresholding.standardize(dwi_mat)
    func_mat = thresholding.standardize(func_mat)

    node_dict_dwi = parse_closest_ixs(
        glob.glob(
            f"{str(Path(dwi_graph_path).parent.parent)}" f"/nodes/*.json"
        ),
        dwi_mat.shape[0],
    )[1]

    node_dict_func = parse_closest_ixs(
        glob.glob(
            f"{str(Path(func_graph_path).parent.parent)}" f"/nodes/*.json"
        ),
        func_mat.shape[0],
    )[1]

    G_dwi = nx.from_numpy_array(dwi_mat)
    nx.set_edge_attributes(
        G_dwi, "structural", nx.get_edge_attributes(G_dwi, "weight").values()
    )
    nx.set_node_attributes(G_dwi, dict(node_dict_dwi), name="dwi")
    # G_dwi.nodes(data=True)

    G_func = nx.from_numpy_array(func_mat)
    nx.set_edge_attributes(
        G_func, "functional", nx.get_edge_attributes(G_func, "weight").values()
    )
    nx.set_node_attributes(G_func, dict(node_dict_func), name="func")
    # G_func.nodes(data=True)

    R = G_dwi.copy()
    R.remove_nodes_from(n for n in G_dwi if n not in G_func)
    R.remove_edges_from(e for e in G_dwi.edges if e not in G_func.edges)
    G_dwi = R.copy()

    R = G_func.copy()
    R.remove_nodes_from(n for n in G_func if n not in G_dwi)
    R.remove_edges_from(e for e in G_func.edges if e not in G_dwi.edges)
    G_func = R.copy()

    [G_dwi, G_func] = multigraph_lcc_intersection([G_dwi, G_func])

    def writeJSON(metadata_str, outputdir):
        import json
        import uuid

        modality = metadata_str.split("modality-")[1].split("_")[0]
        metadata_list = [
            i for i in metadata_str.split("modality-")[1].split("_") if "-" in i
        ]
        hash = str(uuid.uuid4())
        filename = f"{outputdir}/sidecar_modality-{modality}_{hash}.json"
        metadata_dict = {}
        for meta in metadata_list:
            k, v = meta.split("-")
            metadata_dict[k] = v
        with open(filename, "w+") as jsonfile:
            json.dump(metadata_dict, jsonfile, indent=4)
        jsonfile.close()
        return hash

    dwi_name = dwi_graph_path.split("/")[-1].split(".npy")[0]
    func_name = func_graph_path.split("/")[-1].split(".npy")[0]

    dwi_hash = writeJSON(dwi_name, namer_dir)
    func_hash = writeJSON(func_name, namer_dir)

    name = (
        f"{atlas}_mplx_layer1-dwi_ensemble-{dwi_hash}_"
        f"layer2-func_ensemble-{func_hash}"
    )

    dwi_opt, func_opt, best_mi = optimize_mutual_info(
        nx.to_numpy_array(G_dwi), nx.to_numpy_array(G_func), bins=50
    )

    func_mat_final = list(func_opt.values())[0]
    dwi_mat_final = list(dwi_opt.values())[0]
    G_dwi_final = nx.from_numpy_array(dwi_mat_final)
    G_func_final = nx.from_numpy_array(func_mat_final)

    G_multi = nx.OrderedMultiGraph(nx.compose(G_dwi_final, G_func_final))

    out_name = (
        f"{name}_matchthr-{list(dwi_opt.keys())[0]}_"
        f"{list(func_opt.keys())[0]}"
    )
    mG = build_mx_multigraph(
        nx.to_numpy_array(G_func_final),
        nx.to_numpy_array(G_dwi_final),
        out_name,
        namer_dir,
    )

    mG_nx = f"{namer_dir}/{out_name}.gpickle"
    nx.write_gpickle(G_multi, mG_nx)

    dwi_file_out = f"{namer_dir}/{dwi_name}.npy"
    func_file_out = f"{namer_dir}/{func_name}.npy"
    np.save(dwi_file_out, dwi_mat_final)
    np.save(func_file_out, func_mat_final)

    return mG_nx, mG, dwi_file_out, func_file_out


def build_multigraphs(est_path_iterlist: list):
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
    import itertools
    import os
    from pathlib import Path

    import numpy as np

    from pynets.core.utils import flatten
    from pynets.statistics.individual.multiplex import matching

    raw_est_path_iterlist = list(flatten(est_path_iterlist))

    atlases = list(
        set([x.split("/")[-3].split("/")[0] for x in raw_est_path_iterlist])
    )
    parcel_dict_func = dict.fromkeys(atlases)
    parcel_dict_dwi = dict.fromkeys(atlases)
    est_path_iterlist_dwi = list(
        set([i for i in raw_est_path_iterlist if "dwi" in i])
    )
    est_path_iterlist_func = list(
        set([i for i in raw_est_path_iterlist if "func" in i])
    )

    if "_subnet" in ";".join(est_path_iterlist_func):
        func_subnets = list(
            set(
                [
                    i.split("_subnet-")[1].split("_")[0]
                    for i in est_path_iterlist_func
                ]
            )
        )
    else:
        func_subnets = []
    if "_subnet" in ";".join(est_path_iterlist_dwi):
        dwi_subnets = list(
            set(
                [
                    i.split("_subnet-")[1].split("_")[0]
                    for i in est_path_iterlist_dwi
                ]
            )
        )
    else:
        dwi_subnets = []

    dir_path = str(
        Path(os.path.dirname(est_path_iterlist_dwi[0])).parent.parent.parent
    )
    namer_dir = f"{dir_path}/dwi-func"
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
                parcel_dict[atlas][subnet] = list(
                    set(
                        itertools.product(
                            parcel_dict_dwi[atlas][subnet],
                            parcel_dict_func[atlas][subnet],
                        )
                    )
                )
                for paths in list(parcel_dict[atlas][subnet]):
                    [mG_nx, mG, out_dwi_mat, out_func_mat] = matching(
                        paths,
                        atlas,
                        namer_dir,
                    )
        else:
            parcel_dict[atlas] = list(
                set(
                    itertools.product(
                        parcel_dict_dwi[atlas], parcel_dict_func[atlas]
                    )
                )
            )
            for paths in list(parcel_dict[atlas]):
                [mG_nx, mG, out_dwi_mat, out_func_mat] = matching(
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
