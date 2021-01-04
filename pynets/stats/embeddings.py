#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner
"""
from pathlib import Path
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def _omni_embed(pop_array, atlas, graph_path_list, ID,
                subgraph_name="all_nodes", n_components=None, norm=1):
    """
    Omnibus embedding of arbitrary number of input graphs with matched vertex
    sets.

    Given :math:`A_1, A_2, ..., A_m` a collection of (possibly weighted) adjacency
    matrices of a collection :math:`m` undirected graphs with matched vertices.
    Then the :math:`(mn \times mn)` omnibus matrix, :math:`M`, has the subgraph where
    :math:`M_{ij} = \frac{1}{2}(A_i + A_j)`. The omnibus matrix is then embedded
    using adjacency spectral embedding.


    Parameters
    ----------
    graphs : list of nx.Graph or ndarray, or ndarray
        If list of nx.Graph, each Graph must contain same number of nodes.
        If list of ndarray, each array must have shape (n_vertices, n_vertices).
        If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
    atlas : str
    graph_pathlist : list
    ID : str
    subgraph_name : str

    Returns
    -------
    out_path : str
        File path to .npy file containing omni embedding tensor.

    References
    ----------
    .. [1] Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017,
      November). A central limit theorem for an omnibus embedding of multiple random
      dot product graphs. In Data Mining Workshops (ICDMW), 2017 IEEE International
      Conference on (pp. 964-967). IEEE.
    .. [2] Chung, J., Pedigo, B. D., Bridgeford, E. W., Varjavand, B. K., Helm, H. S.,
      & Vogelstein, J. T. (2019). Graspy: Graph statistics in python.
      Journal of Machine Learning Research.

    """
    import networkx as nx
    import numpy as np
    from pynets.core.utils import flatten
    from graspy.embed import OmnibusEmbed, ClassicalMDS
    from joblib import dump
    from pynets.stats.netstats import CleanGraphs

    clean_mats = []
    i = 0
    for graph_path in graph_path_list:
        cg = CleanGraphs(None, None, graph_path, 0, norm)

        if float(norm) >= 1:
            G = cg.normalize_graph()
            mat_clean = nx.to_numpy_array(G)
        else:
            mat_clean = pop_array[i]

        clean_mats.append(mat_clean)
        i += 1

    # Omnibus embedding
    print(
        f"{'Embedding unimodal omnetome for atlas: '}{atlas} and "
        f"{subgraph_name}{'...'}"
    )
    omni = OmnibusEmbed(n_components=n_components, check_lcc=False)
    mds = ClassicalMDS(n_components=n_components)
    omni_fit = omni.fit_transform(pop_array)

    # Transform omnibus tensor into dissimilarity feature
    mds_fit = mds.fit_transform(omni_fit.reshape(omni_fit.shape[1],
                                                 omni_fit.shape[2],
                                                 omni_fit.shape[0]))

    dir_path = str(Path(os.path.dirname(graph_path_list[0])).parent)

    namer_dir = f"{dir_path}/embeddings"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = (
        f"{namer_dir}/gradient-OMNI_{atlas}_{subgraph_name}_"
        f"{os.path.basename(graph_path_list[0]).split('_thrtype')[0]}.npy"
    )

    # out_path_est_omni = f"{namer_dir}/gradient-OMNI_{atlas}_" \
    #                     f"{subgraph_name}_" \
    #                     f"{os.path.basename(graph_path).split('_thrtype')[0]}" \
    #                     f"_MDS.joblib"
    # out_path_est_mds = f"{namer_dir}/gradient-OMNI_{atlas}_" \
    #                    f"{subgraph_name}_" \
    #                    f"{os.path.basename(graph_path).split('_thrtype')[0]}" \
    #                    f"_MDS.joblib"

    # dump(omni, out_path_est_omni)
    # dump(omni, out_path_est_mds)

    print("Saving...")
    np.save(out_path, mds_fit)
    del mds, mds_fit, omni, omni_fit
    return out_path


def _mase_embed(pop_array, atlas, graph_path, ID, subgraph_name="all_nodes", n_components=None):
    """
    Multiple Adjacency Spectral Embedding (MASE) embeds arbitrary number of input
    graphs with matched vertex sets.

    For a population of undirected graphs, MASE assumes that the population of graphs
    is sampled from :math:`VR^{(i)}V^T` where :math:`V \in \mathbb{R}^{n\times d}` and
    :math:`R^{(i)} \in \mathbb{R}^{d\times d}`. Score matrices, :math:`R^{(i)}`, are
    allowed to vary for each graph, but are symmetric. All graphs share a common a
    latent position matrix :math:`V`.

    For a population of directed graphs, MASE assumes that the population is sampled
    from :math:`UR^{(i)}V^T` where :math:`U \in \mathbb{R}^{n\times d_1}`,
    :math:`V \in \mathbb{R}^{n\times d_2}`, and
    :math:`R^{(i)} \in \mathbb{R}^{d_1\times d_2}`. In this case, score matrices
    :math:`R^{(i)}` can be assymetric and non-square, but all graphs still share a
    common latent position matrices :math:`U` and :math:`V`.


    Parameters
    ----------
    graphs : list of nx.Graph or ndarray, or ndarray
        If list of nx.Graph, each Graph must contain same number of nodes.
        If list of ndarray, each array must have shape (n_vertices, n_vertices).
        If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
    atlas : str
    graph_path : str
    ID : str
    subgraph_name : str

    Returns
    -------
    out_path : str
        File path to .npy file containing MASE embedding tensor.

    References
    ----------
    .. [1] Inference for multiple heterogeneous networks with a common invariant subspace
      J Arroyo, A Athreya, J Cape, G Chen, CE Priebe, JT Vogelstein
      arXiv preprint arXiv:1906.10026
    .. [2] Chung, J., Pedigo, B. D., Bridgeford, E. W., Varjavand, B. K., Helm, H. S.,
      & Vogelstein, J. T. (2019). Graspy: Graph statistics in python.
      Journal of Machine Learning Research.

    """
    import numpy as np
    from graspy.embed import MultipleASE
    from joblib import dump

    # Multiple Adjacency Spectral embedding
    print(
        f"{'Embedding multimodal masetome for atlas: '}{atlas} and "
        f"{subgraph_name}{'...'}"
    )
    mase = MultipleASE(n_components=n_components)
    mase_fit = mase.fit_transform(pop_array)

    dir_path = str(Path(os.path.dirname(graph_path)))
    namer_dir = f"{dir_path}/mplx_embeddings"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = (
        f"{namer_dir}/gradient-MASE_{atlas}_{subgraph_name}"
        f"_{os.path.basename(graph_path)}"
    )
    out_path_est = f"{namer_dir}/gradient-MASE_{atlas}_" \
                   f"{subgraph_name}" \
                   f"_{os.path.basename(graph_path).split('.npy')[0]}.joblib"

    dump(mase, out_path_est)

    print("Saving...")
    np.save(out_path, mase.scores_)
    del mase, mase_fit

    return out_path


def _ase_embed(mat, atlas, graph_path, ID, subgraph_name="all_nodes", n_components=None, prune=0, norm=1):
    """

    Class for computing the adjacency spectral embedding of a graph.

    The adjacency spectral embedding (ASE) is a k-dimensional Euclidean representation
    of the graph based on its adjacency matrix. It relies on an SVD to reduce
    the dimensionality to the specified k, or if k is unspecified, can find a number of
    dimensions automatically

    Parameters
    ----------
    graphs : list of nx.Graph or ndarray, or ndarray
        If list of nx.Graph, each Graph must contain same number of nodes.
        If list of ndarray, each array must have shape (n_vertices, n_vertices).
        If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
    atlas : str
    graph_path : str
    ID : str
    subgraph_name : str

    Returns
    -------
    out_path : str
        File path to .npy file containing ASE embedding tensor.

    Notes
    -----
    The singular value decomposition:

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the
    adjacency matrix of the graph. These basis vectors (in the matrices U or V) are
    ordered according to the amount of variance they explain in the original matrix.
    By selecting a subset of these basis vectors (through our choice of dimensionality
    reduction) we can find a lower dimensional space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
      Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
      Journal of the American Statistical Association, Vol. 107(499), 2012

    """
    import os
    import networkx as nx
    import numpy as np
    from pynets.core.utils import flatten
    from graspy.embed import AdjacencySpectralEmbed
    from joblib import dump
    from pynets.stats.netstats import CleanGraphs
    #from graspy.utils import get_lcc

    # Adjacency Spectral embedding
    print(
        f"{'Embedding unimodal asetome for atlas: '}{atlas} and "
        f"{subgraph_name}{'...'}"
    )
    ase = AdjacencySpectralEmbed(n_components=n_components)
    cg = CleanGraphs(None, None, graph_path, prune, norm)

    if float(norm) >= 1:
        G = cg.normalize_graph()
        mat_clean = nx.to_numpy_array(G)
    else:
        mat_clean = mat

    if float(prune) >= 1:
        graph_path_tmp = cg.prune_graph()[1]
        mat_clean = np.load(graph_path_tmp)
    else:
        mat_clean = mat

    ase_fit = ase.fit_transform(mat_clean)

    dir_path = str(Path(os.path.dirname(graph_path)).parent)

    namer_dir = f"{dir_path}/embeddings"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = f"{namer_dir}/gradient-ASE" \
               f"_{atlas}_{subgraph_name}_{os.path.basename(graph_path)}"
    # out_path_est = f"{namer_dir}/gradient-ASE_{atlas}" \
    #                f"_{subgraph_name}" \
    #                f"_{os.path.basename(graph_path).split('.npy')[0]}.joblib"

    #dump(ase, out_path_est)

    print("Saving...")
    np.save(out_path, ase_fit)
    del ase, ase_fit

    return out_path


def build_asetomes(est_path_iterlist, ID):
    """
    Embeds single graphs using the ASE algorithm.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy files, each containing a graph.
    ID : str
        A subject id or other unique identifier.

    """
    import numpy as np
    from pynets.core.utils import prune_suffices, flatten
    from pynets.stats.embeddings import _ase_embed
    from pynets.core.utils import load_runconfig

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    try:
        n_components = hardcoded_params["gradients"][
            "n_components"][0]
    except KeyError:
        import sys
        print(
            "ERROR: available gradient dimensionality presets not "
            "sucessfully extracted from runconfig.yaml"
        )
        sys.exit(1)

    if isinstance(est_path_iterlist, list):
        est_path_iterlist = list(flatten(est_path_iterlist))
    else:
        est_path_iterlist = [est_path_iterlist]

    out_paths = []
    for file_ in est_path_iterlist:
        mat = np.load(file_)
        atlas = prune_suffices(file_.split("/")[-3])
        res = prune_suffices("_".join(file_.split(
            "/")[-1].split("modality")[1].split("_")[1:]).split("_est")[0])
        if "rsn" in res:
            subgraph = res.split("rsn-")[1]
        else:
            subgraph = "all_nodes"
        out_path = _ase_embed(mat, atlas, file_, ID, subgraph_name=subgraph,
                              n_components=n_components)
        out_paths.append(out_path)

    return out_paths


def build_masetome(est_path_iterlist, ID):
    """
    Embeds structural-functional graph pairs into a common invariant subspace.

    Parameters
    ----------
    est_path_iterlist : list
        List of list of pairs of file paths (.npy) corresponding to
        structural and functional connectomes matched at a given node
        resolution.
    ID : str
        A subject id or other unique identifier.

    References
    ----------
    .. [1] Rosenthal, G., Váša, F., Griffa, A., Hagmann, P., Amico, E., Goñi, J.,
      Sporns, O. (2018). Mapping higher-order relations between brain structure
      and function with embedded vector representations of connectomes.
      Nature Communications. https://doi.org/10.1038/s41467-018-04614-w

    """
    import numpy as np
    from pynets.core.utils import prune_suffices
    from pynets.stats.embeddings import _mase_embed
    from pynets.core.utils import load_runconfig

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    try:
        n_components = hardcoded_params["gradients"][
            "n_components"][0]
    except KeyError:
        import sys
        print(
            "ERROR: available gradient dimensionality presets not "
            "sucessfully extracted from runconfig.yaml"
        )
        sys.exit(1)

    out_paths = []
    for pairs in est_path_iterlist:
        pop_list = []
        for _file in pairs:
            pop_list.append(np.load(_file))
        atlas = prune_suffices(pairs[0].split("/")[-3])
        res = prune_suffices("_".join(pairs[0].split(
            "/")[-1].split("modality")[1].split("_")[1:]).split("_est")[0])
        if "rsn" in res:
            subgraph = res.split("rsn-")[1]
        else:
            subgraph = "all_nodes"
        out_path = _mase_embed(
            pop_list,
            atlas,
            pairs[0],
            ID,
            subgraph_name=subgraph, n_components=n_components)
        out_paths.append(out_path)

    return out_paths


def build_omnetome(est_path_iterlist, ID):
    """
    Embeds ensemble population of graphs into an embedded ensemble feature
    vector.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy file containing graph.
    ID : str
        A subject id or other unique identifier.

    References
    ----------
    .. [1] Liu, Y., He, L., Cao, B., Yu, P. S., Ragin, A. B., & Leow, A. D. (2018).
      Multi-view multi-graph embedding for brain network clustering analysis.
      32nd AAAI Conference on Artificial Intelligence, AAAI 2018.
    .. [2] Levin, K., Athreya, A., Tang, M., Lyzinski, V., & Priebe, C. E. (2017,
      November). A central limit theorem for an omnibus embedding of multiple random
      dot product graphs. In Data Mining Workshops (ICDMW), 2017 IEEE International
      Conference on (pp. 964-967). IEEE.

    """
    import sys
    import numpy as np
    from pynets.core.utils import flatten
    from pynets.stats.embeddings import _omni_embed
    from pynets.core.utils import load_runconfig

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()

    try:
        func_models = hardcoded_params["available_models"]["func_models"]
    except KeyError:
        print(
            "ERROR: available functional models not sucessfully extracted"
            " from runconfig.yaml"
        )
        sys.exit(1)
    try:
        struct_models = hardcoded_params["available_models"][
            "struct_models"]
    except KeyError:
        print(
            "ERROR: available structural models not sucessfully extracted"
            " from runconfig.yaml"
        )
        sys.exit(1)
    try:
        n_components = hardcoded_params["gradients"][
            "n_components"][0]
    except KeyError:
        print(
            "ERROR: available gradient dimensionality presets not "
            "sucessfully extracted from runconfig.yaml"
        )
        sys.exit(1)

    if isinstance(est_path_iterlist, list):
        est_path_iterlist = list(flatten(est_path_iterlist))
    else:
        est_path_iterlist = [est_path_iterlist]

    if len(est_path_iterlist) > 1:
        atlases = list(set([x.split("/")[-3].split("/")[0]
                            for x in est_path_iterlist]))
        parcel_dict_func = dict.fromkeys(atlases)
        parcel_dict_dwi = dict.fromkeys(atlases)

        est_path_iterlist_dwi = list(
            set(
                [
                    i
                    for i in est_path_iterlist
                    if i.split("model-")[1].split("_")[0] in struct_models
                ]
            )
        )
        est_path_iterlist_func = list(
            set(
                [
                    i
                    for i in est_path_iterlist
                    if i.split("model-")[1].split("_")[0] in func_models
                ]
            )
        )

        if "_rsn" in ";".join(est_path_iterlist_func):
            func_subnets = list(
                set([i.split("_rsn-")[1].split("_")[0] for i in
                     est_path_iterlist_func])
            )
        else:
            func_subnets = []
        if "_rsn" in ";".join(est_path_iterlist_dwi):
            dwi_subnets = list(
                set([i.split("_rsn-")[1].split("_")[0] for i in
                     est_path_iterlist_dwi])
            )
        else:
            dwi_subnets = []

        out_paths_func = []
        out_paths_dwi = []
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
                                parcel_dict_dwi[atlas][sub_net].append(
                                    graph_path)
                    else:
                        parcel_dict_dwi[atlas].append(graph_path)

            for graph_path in est_path_iterlist_func:
                if atlas in graph_path:
                    if len(func_subnets) >= 1:
                        for sub_net in func_subnets:
                            if sub_net in graph_path:
                                parcel_dict_func[atlas][sub_net].append(
                                    graph_path)
                    else:
                        parcel_dict_func[atlas].append(graph_path)
            if len(parcel_dict_func[atlas]) > 0:
                if isinstance(parcel_dict_func[atlas], dict):
                    # RSN case
                    for rsn in parcel_dict_func[atlas]:
                        pop_rsn_list = []
                        graph_path_list = []
                        for graph in parcel_dict_func[atlas][rsn]:
                            pop_rsn_list.append(np.load(graph))
                            graph_path_list.append(graph)
                        if len(pop_rsn_list) > 1:
                            if len(
                                    list(set([i.shape for i in
                                              pop_rsn_list]))) > 1:
                                raise RuntimeWarning(
                                    "Inconsistent number of"
                                    " vertices in graph population "
                                    "that precludes embedding...")
                            out_path = _omni_embed(
                                pop_rsn_list, atlas, graph_path_list, ID, rsn,
                                n_components
                            )
                            out_paths_func.append(out_path)
                        else:
                            print(
                                "WARNING: Only one graph sampled, omnibus"
                                " embedding not appropriate."
                            )
                            pass
                else:
                    pop_list = []
                    graph_path_list = []
                    for pop_ref in parcel_dict_func[atlas]:
                        pop_list.append(np.load(pop_ref))
                        graph_path_list.append(pop_ref)
                    if len(pop_list) > 1:
                        if len(list(set([i.shape for i in pop_list]))) > 1:
                            raise RuntimeWarning(
                                "Inconsistent number of vertices in "
                                "graph population that precludes embedding")
                        out_path = _omni_embed(pop_list, atlas,
                                               graph_path_list, ID,
                                               n_components=n_components)
                        out_paths_func.append(out_path)
                    else:
                        print(
                            "WARNING: Only one graph sampled, omnibus embedding"
                            " not appropriate."
                        )
                        pass

            if len(parcel_dict_dwi[atlas]) > 0:
                if isinstance(parcel_dict_dwi[atlas], dict):
                    # RSN case
                    graph_path_list = []
                    for rsn in parcel_dict_dwi[atlas]:
                        pop_rsn_list = []
                        for graph in parcel_dict_dwi[atlas][rsn]:
                            pop_rsn_list.append(np.load(graph))
                            graph_path_list.append(graph)
                        if len(pop_rsn_list) > 1:
                            if len(
                                    list(set([i.shape for i in
                                              pop_rsn_list]))) > 1:
                                raise RuntimeWarning(
                                    "Inconsistent number of"
                                    " vertices in graph population "
                                    "that precludes embedding")
                            out_path = _omni_embed(
                                pop_rsn_list, atlas, graph_path_list, ID,
                                rsn, n_components
                            )
                            out_paths_dwi.append(out_path)
                        else:
                            print(
                                "WARNING: Only one graph sampled, omnibus"
                                " embedding not appropriate."
                            )
                            pass
                else:
                    pop_list = []
                    graph_path_list = []
                    for pop_ref in parcel_dict_dwi[atlas]:
                        pop_list.append(np.load(pop_ref))
                        graph_path_list.append(pop_ref)
                    if len(pop_list) > 1:
                        if len(list(set([i.shape for i in pop_list]))) > 1:
                            raise RuntimeWarning(
                                "Inconsistent number of vertices in graph"
                                " population that precludes embedding")
                        out_path = _omni_embed(pop_list, atlas, graph_path_list,
                                               ID, n_components=n_components)
                        out_paths_dwi.append(out_path)
                    else:
                        print(
                            "WARNING: Only one graph sampled, omnibus embedding"
                            " not appropriate."
                        )
                        pass
    else:
        print("At least two graphs required to build an omnetome...")
        out_paths_func = []
        out_paths_dwi = []
        pass

    return out_paths_dwi, out_paths_func
