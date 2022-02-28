#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import matplotlib
import warnings
import os
import sys
import os.path as op
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
import numpy as np
import time
import logging
import threading
import traceback
import signal

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def get_file():
    """Get a file's base directory path."""
    base_path = str(__file__)
    return base_path


def checkConsecutive(l):
    n = len(l) - 1
    return sum(np.diff(sorted(l)) == 1) >= n


def prune_suffices(res):
    import re

    if "reor-RAS" in str(res):
        res = re.sub(r"_reor\-*[A-Z][A-Z][A-Z]", "", str(res))
    if "res-" in str(res):
        res = re.sub(r"_res\-*[0-4]mm", "", str(res))
    if "noreor-RAS" in str(res):
        res = re.sub(r"_noreor\-*[A-Z][A-Z][A-Z]", "", str(res))
    if "nores-" in str(res):
        res = re.sub(r"_nores\-*[0-4]mm", "", str(res))
    return res


def do_dir_path(atlas, outdir):
    """
    Creates an atlas subdirectory from the base directory of the given
    subject's input file.

    Parameters
    ----------
    atlas : str
        Name of atlas parcellation used.
    outdir : str
        Path to base derivatives directory.

    Returns
    -------
    dir_path : str
        Path to directory containing subject derivative data for given run.

    """
    if atlas:
        if os.path.isfile(atlas):
            atlas = os.path.basename(atlas)
        atlas = prune_suffices(atlas)
        if atlas.endswith(".nii.gz"):
            atlas = atlas.replace(".nii.gz", "")

    dir_path = f"{outdir}/{atlas}"
    if not op.exists(dir_path) and atlas is not None:
        os.makedirs(dir_path, exist_ok=True)
    elif atlas is None:
        raise ValueError("cannot create directory for a null "
                         "atlas!")

    return dir_path


def as_directory(dir_, remove=False, return_as_path=False):
    """
    Convenience function to make a directory while returning it.

    Parameters
    ----------
    dir_ : str, Path
        File location to directory.
    remove : bool, optional
        Whether to remove a previously existing directory, by default False

    Returns
    -------
    str
        Directory string.

    """
    import shutil
    from pathlib import Path

    p = Path(dir_).absolute()

    if remove:
        print(f"Previous directory found at {dir_}. Removing.")
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

    if return_as_path:
        return p

    return str(p)


def create_est_path_func(
    ID,
    subnet,
    conn_model,
    thr,
    roi,
    dir_path,
    node_radius,
    smooth,
    thr_type,
    hpass,
    parc,
    signal,
):
    """
    Name the thresholded functional connectivity matrix file based on
    relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for partial
       correlation). sps type is used by default.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    thr_type : str
        Type of thresholding performed (e.g. prop, abs, dens, mst, disp)
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    signal : str
        The name of a valid function used to reduce the time-series region
        extraction.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with all specified
        combinations of hyperparameter characteristics.

    """
    import os
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e,
              "No template specified in advanced.yaml"
              )

    namer_dir = f"{dir_path}/graphs"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    if hpass is None:
        hpass = 0

    if smooth is None:
        smooth = 0

    subnet_suff = f"_rsn-{subnet}" if subnet is not None else ""
    roi_suff = f"_roi-{op.basename(roi).split('.')[0]}" if roi is not None \
        else ""
    nodetype_suff = f"_nodetype-spheres-{node_radius}mm" if \
        ((node_radius is not None) and (node_radius != 'parc')) \
        else "_nodetype-parc"

    return f"{namer_dir}/graph_sub-{ID}_modality-func{subnet_suff}" \
           f"{roi_suff}_model-{conn_model}_template-{template_name}" \
           f"{nodetype_suff}_tol-{smooth}fwhm_hpass-{hpass}Hz_" \
           f"signal-{signal}_thrtype-{thr_type}_thr-{thr}.npy"


def create_est_path_diff(
    ID,
    subnet,
    conn_model,
    thr,
    roi,
    dir_path,
    node_radius,
    track_type,
    thr_type,
    parc,
    traversal,
    min_length,
    error_margin,
):
    """
    Name the thresholded structural connectivity matrix file based on
    relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for
       partial correlation). sps type is used by default.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    thr_type : str
        Type of thresholding performed (e.g. prop, abs, dens, mst, disp)
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    traversal : str
        The statistical approach to tracking. Options are:
        det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with thresholding applied.

    """
    import os
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e,
              "No template specified in advanced.yaml"
              )

    namer_dir = f"{dir_path}/graphs"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    subnet_suff = f"_rsn-{subnet}" if subnet is not None else ""
    roi_suff = f"_roi-{op.basename(roi).split('.')[0]}" if roi is not None \
        else ""
    nodetype_suff = f"_nodetype-spheres-{node_radius}mm" if \
        ((node_radius is not None) and (node_radius != 'parc')) \
        else "_nodetype-parc"

    return f"{namer_dir}/graph_sub-{ID}_modality-dwi{subnet_suff}" \
           f"{roi_suff}_model-{conn_model}_template-{template_name}" \
           f"{nodetype_suff}_tracktype-{track_type}_" \
           f"traversal-{traversal}_minlength-{min_length}_" \
           f"tol-{error_margin}_thrtype-{thr_type}_thr-{thr}.npy"


def create_raw_path_func(
    ID,
    subnet,
    conn_model,
    roi,
    dir_path,
    node_radius,
    smooth,
    hpass,
    parc,
    signal,
):
    """
    Name the raw functional connectivity matrix file based on relevant
    graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for
       partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    signal : str
        The name of a valid function used to reduce the time-series region
        extraction.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with all specified
        combinations of hyperparameter characteristics.

    """
    import os
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e,
              "No template specified in advanced.yaml"
              )

    namer_dir = f"{dir_path}/graphs"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    if hpass is None:
        hpass = 0

    if smooth is None:
        smooth = 0

    subnet_suff = f"_rsn-{subnet}" if subnet is not None else ""
    roi_suff = f"_roi-{op.basename(roi).split('.')[0]}" if roi is not None \
        else ""
    nodetype_suff = f"_nodetype-spheres-{node_radius}mm" if \
        ((node_radius is not None) and (node_radius != 'parc')) \
        else "_nodetype-parc"

    return f"{namer_dir}/rawgraph_sub-{ID}_modality-func{subnet_suff}" \
           f"{roi_suff}_model-{conn_model}_template-{template_name}" \
           f"{nodetype_suff}_tol-{smooth}fwhm_hpass-{hpass}Hz_" \
           f"signal-{signal}.npy"


def create_raw_path_diff(
    ID,
    subnet,
    conn_model,
    roi,
    dir_path,
    node_radius,
    track_type,
    parc,
    traversal,
    min_length,
    error_margin
):
    """
    Name the raw structural connectivity matrix file based on relevant
    graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for
       partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    traversal : str
        The statistical approach to tracking. Options are:
        det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with thresholding applied.

    """
    import os
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e,
              "No template specified in advanced.yaml"
              )

    namer_dir = f"{dir_path}/graphs"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    subnet_suff = f"_rsn-{subnet}" if subnet is not None else ""
    roi_suff = f"_roi-{op.basename(roi).split('.')[0]}" if roi is not None \
        else ""
    nodetype_suff = f"_nodetype-spheres-{node_radius}mm" if \
        ((node_radius is not None) and (node_radius != 'parc')) \
        else "_nodetype-parc"

    return f"{namer_dir}/rawgraph_sub-{ID}_modality-dwi{subnet_suff}" \
           f"{roi_suff}_model-{conn_model}_template-{template_name}" \
           f"{nodetype_suff}_tracktype-{track_type}_" \
           f"traversal-{traversal}_minlength-{min_length}_" \
           f"tol-{error_margin}.npy"


def create_csv_path(dir_path, est_path):
    """

    Create a csv path to save graph metrics.

    Parameters
    ----------
    dir_path : str
        Path to directory containing subject derivative data for given run.
    est_path : str
        File path to .npy file containing graph with thresholding applied.

    Returns
    -------
    out_path : str
        File path to .csv with graph metrics.

    """
    import os
    from pathlib import Path

    namer_dir = f"{str(Path(dir_path).parent)}/topology"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    return f"{namer_dir}/metrics_" \
           f"{est_path.split('/')[-1].split('.npy')[0]}.csv"


def load_mat(est_path):
    """
    Load an adjacency matrix using any of a variety of methods.

    Parameters
    ----------
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    """
    import numpy as np
    import networkx as nx
    import os.path as op

    fmt = op.splitext(est_path)[1]

    if fmt == ".edgelist_csv" or fmt == ".csv":
        with open(est_path, "rb") as stream:
            G = nx.read_weighted_edgelist(stream, delimiter=",")
        stream.close()
    elif fmt == ".edgelist_ssv" or fmt == ".ssv":
        with open(est_path, "rb") as stream:
            G = nx.read_weighted_edgelist(stream, delimiter=" ")
        stream.close()
    elif fmt == ".edgelist_tsv" or fmt == ".tsv":
        with open(est_path, "rb") as stream:
            G = nx.read_weighted_edgelist(stream, delimiter="\t")
    elif fmt == ".gpickle":
        G = nx.read_gpickle(est_path)
    elif fmt == ".graphml":
        G = nx.read_graphml(est_path)
    elif fmt == ".txt":
        G = nx.from_numpy_array(np.genfromtxt(est_path))
    elif fmt == ".npy":
        G = nx.from_numpy_array(np.load(est_path, allow_pickle=True))
    else:
        raise ValueError("\nFile format not supported!")

    G.graph["ecount"] = nx.number_of_edges(G)
    G = nx.convert_node_labels_to_integers(G, first_label=1)

    return nx.to_numpy_matrix(G, weight="weight")


def load_mat_ext(
    est_path,
    ID,
    subnet,
    conn_model,
    roi,
    prune,
    norm,
    binary,
    min_span_tree,
    dens_thresh,
    disp_filt,
):

    return (
        load_mat(est_path),
        est_path,
        ID,
        subnet,
        conn_model,
        roi,
        prune,
        norm,
        binary,
        min_span_tree,
        dens_thresh,
        disp_filt,
    )


def save_mat(conn_matrix, est_path, fmt=None):
    """
    Save an adjacency matrix using any of a variety of methods.

    Parameters
    ----------
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    est_path : str
        File path to .npy file containing graph.
    fmt : str
        Format to save connectivity matrix/graph (e.g. .npy, .pkl, .graphml,
         .txt, .ssv, .csv).

    """
    import numpy as np
    import networkx as nx
    from pynets.core.utils import load_runconfig

    if fmt is None:
        hardcoded_params = load_runconfig()
        fmt = hardcoded_params["graph_file_format"][0]

    G = nx.from_numpy_array(conn_matrix)
    G.graph["ecount"] = nx.number_of_edges(G)
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    if fmt == "edgelist_csv":
        if os.path.isfile(f"{est_path.split('.npy')[0]}.csv"):
            os.remove(f"{est_path.split('.npy')[0]}.csv")
        nx.write_weighted_edgelist(
            G, f"{est_path.split('.npy')[0]}.csv", encoding="utf-8"
        )
    elif fmt == "gpickle":
        if os.path.isfile(f"{est_path.split('.npy')[0]}.pkl"):
            os.remove(f"{est_path.split('.npy')[0]}.pkl")
        nx.write_gpickle(G, f"{est_path.split('.npy')[0]}.pkl")
    elif fmt == "graphml":
        if os.path.isfile(f"{est_path.split('.npy')[0]}.graphml"):
            os.remove(f"{est_path.split('.npy')[0]}.graphml")
        nx.write_graphml(G, f"{est_path.split('.npy')[0]}.graphml")
    elif fmt == "txt":
        if os.path.isfile(f"{est_path.split('.npy')[0]}{'.txt'}"):
            os.remove(f"{est_path.split('.npy')[0]}{'.txt'}")
        np.savetxt(
            f"{est_path.split('.npy')[0]}{'.txt'}",
            nx.to_numpy_matrix(G))
    elif fmt == "npy":
        if os.path.isfile(est_path):
            os.remove(est_path)
        np.save(est_path, nx.to_numpy_matrix(G))
    elif fmt == "edgelist_ssv":
        if os.path.isfile(f"{est_path.split('.npy')[0]}.ssv"):
            os.remove(f"{est_path.split('.npy')[0]}.ssv")
        nx.write_weighted_edgelist(
            G,
            f"{est_path.split('.npy')[0]}.ssv",
            delimiter=" ",
            encoding="utf-8")
    else:
        raise ValueError("\nFile format not supported!")

    return


def mergedicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and \
                    isinstance(dict2[k], dict):
                yield k, dict(mergedicts(dict1[k],
                                          dict2[k]))
            else:
                yield k, dict2[k]
        elif k in dict1:
            yield k, dict1[k]
        else:
            yield k, dict2[k]


def save_mat_thresholded(
    conn_matrix,
    est_path_orig,
    thr_type,
    ID,
    subnet,
    thr,
    conn_model,
    roi,
    prune,
    norm,
    binary,
):
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    est_path = fname_presuffix(est_path_orig,
                               suffix=f"_thrtype-{thr_type}_thr-{thr}")

    if (np.abs(conn_matrix) < 0.0000001).all():
        print(UserWarning(f"Empty graph detected for: {est_path}"))

    save_mat(conn_matrix, est_path, fmt="npy")

    return est_path, ID, subnet, thr, conn_model, roi, prune, norm, binary


def pass_meta_outs(
    conn_model_iterlist,
    est_path_iterlist,
    network_iterlist,
    thr_iterlist,
    prune_iterlist,
    ID_iterlist,
    roi_iterlist,
    norm_iterlist,
    binary_iterlist,
):
    """
    Passes lists of iterable parameters as metadata.

    Parameters
    ----------
    conn_model_iterlist : list
       List of connectivity estimation model parameters (e.g. corr for
       correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding
        applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using any
        variety of methods triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of
        disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest
        Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an
        unweighted graph were binarized.

    Returns
    -------
    conn_model_iterlist : list
       List of connectivity estimation model parameters (e.g. corr for
       correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding
        applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using
        any variety of methods triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of
        disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest
        Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an
        unweighted graph were binarized.
    embed_iterlist : list
        List of booleans indicating whether omnibus embedding of graph
        population was performed.
    multimodal_iterlist : list
        List of booleans indicating whether multiple modalities of input data
        have been specified.
    """

    return (
        conn_model_iterlist,
        est_path_iterlist,
        network_iterlist,
        thr_iterlist,
        prune_iterlist,
        ID_iterlist,
        roi_iterlist,
        norm_iterlist,
        binary_iterlist,
    )


def pass_meta_ins(
        conn_model,
        est_path,
        subnet,
        thr,
        prune,
        ID,
        roi,
        norm,
        binary):
    """
    Passes parameters as metadata.

    Parameters
    ----------
    conn_model : str
        Connectivity estimation model (e.g. corr for correlation, cov for
        covariance, sps for precision covariance, partcorr for partial
        correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.

    Returns
    -------
    conn_model : str
        Connectivity estimation model (e.g. corr for correlation, cov for
        covariance, sps for precision covariance, partcorr for partial
        correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    """
    est_path_iterlist = est_path
    conn_model_iterlist = conn_model
    network_iterlist = subnet
    thr_iterlist = thr
    prune_iterlist = prune
    ID_iterlist = ID
    roi_iterlist = roi
    norm_iterlist = norm
    binary_iterlist = binary
    # print('\n\nParam-iters:\n')
    # print(conn_model_iterlist)
    # print(est_path_iterlist)
    # print(network_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(roi_iterlist)
    # print(norm_iterlist)
    # print(binary_iterlist)
    # print('\n\n')
    return (
        conn_model_iterlist,
        est_path_iterlist,
        network_iterlist,
        thr_iterlist,
        prune_iterlist,
        ID_iterlist,
        roi_iterlist,
        norm_iterlist,
        binary_iterlist,
    )


def pass_meta_ins_multi(
    conn_model_func,
    est_path_func,
    network_func,
    thr_func,
    prune_func,
    ID_func,
    roi_func,
    norm_func,
    binary_func,
    conn_model_struct,
    est_path_struct,
    network_struct,
    thr_struct,
    prune_struct,
    ID_struct,
    roi_struct,
    norm_struct,
    binary_struct,
):
    """
    Passes multimodal iterable parameters as metadata.

    Parameters
    ----------
    conn_model_func : str
        Functional connectivity estimation model (e.g. corr for correlation, cov
        for covariance, sps for precision covariance, partcorr for partial
        correlation). sps type is used by default.
    est_path_func : str
        File path to .npy file containing functional graph with thresholding
        applied.
    network_func : str
        Functional resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    thr_func : float
        A value, between 0 and 1, to threshold the functional graph using any
        variety of methods triggered through other options.
    prune_func : bool
        Indicates whether to prune final functional graph of disconnected
        nodes/isolates.
    ID_func : str
        A subject id or other unique identifier for the functional workflow.
    roi_func : str
        File path to binarized/boolean region-of-interest Nifti1Image file
        applied to the functional data.
    norm_func : int
        Indicates method of normalizing resulting functional graph.
    binary_func : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted functional graph.
    conn_model_struct : str
        Diffusion structural connectivity estimation model (e.g. corr for
        correlation, cov for covariance, sps for precision covariance, partcorr
        for partial correlation). sps type is used by default.
    est_path_struct : str
        File path to .npy file containing diffusion structural graph with
        thresholding applied.
    network_struct : str
        Diffusion structural resting-state subnet based on Yeo-7 and Yeo-17
        naming (e.g. 'Default') used to filter nodes in the study of brain
        subgraphs.
    thr_struct : float
        A value, between 0 and 1, to threshold the diffusion structural graph
        using any variety of methods triggered through other options.
    prune_struct : bool
        Indicates whether to prune final diffusion structural graph of
        disconnected nodes/isolates.
    ID_struct : str
        A subject id or other unique identifier for the diffusion structural
        workflow.
    roi_struct : str
        File path to binarized/boolean region-of-interest Nifti1Image file
        applied too the dwi data.
    norm_struct : int
        Indicates method of normalizing resulting diffusion structural graph.
    binary_struct : bool
        Indicates whether to binarize resulting diffusion structural graph
        edges to form an unweighted graph.

    Returns
    -------
    conn_model_iterlist : list
        List of connectivity estimation model parameters (e.g. corr for
        correlation, cov for covariance, sps for precision covariance, partcorr
        for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding
        applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using any
        variety of methods triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of
        disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest
        Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an
        unweighted graph were binarized.
    embed_iterlist : list
        List of booleans indicating whether omnibus embedding of graph
        population was performed.
    multimodal_iterlist : list
        List of booleans indicating whether multiple modalities of input data
        have been specified.
    """
    est_path_iterlist = [est_path_func, est_path_struct]
    conn_model_iterlist = [conn_model_func, conn_model_struct]
    network_iterlist = [network_func, network_struct]
    thr_iterlist = [thr_func, thr_struct]
    prune_iterlist = [prune_func, prune_struct]
    ID_iterlist = [ID_func, ID_struct]
    roi_iterlist = [roi_func, roi_struct]
    norm_iterlist = [norm_func, norm_struct]
    binary_iterlist = [binary_func, binary_struct]
    # print('\n\nParam-iters:\n')
    # print(conn_model_iterlist)
    # print(est_path_iterlist)
    # print(network_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(roi_iterlist)
    # print(norm_iterlist)
    # print(binary_iterlist)
    # print('\n\n')
    return (
        conn_model_iterlist,
        est_path_iterlist,
        network_iterlist,
        thr_iterlist,
        prune_iterlist,
        ID_iterlist,
        roi_iterlist,
        norm_iterlist,
        binary_iterlist,
    )


def collectpandasjoin(net_mets_csv):
    """
    Passes csv pandas dataframe as metadata.

    Parameters
    ----------
    net_mets_csv : str
        File path to csv pandas dataframe.

    Returns
    -------
    net_mets_csv_out : str
        File path to csv pandas dataframe as itself.

    """
    net_mets_csv_out = net_mets_csv
    return net_mets_csv_out


def flatten(l):
    """
    Flatten list of lists.
    """
    import collections

    for el in l:
        if isinstance(
                el, collections.Iterable) and not isinstance(
                el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def decompress_nifti(infile):
    from nipype.utils.filemanip import split_filename
    import gzip
    import os
    import shutil
    from time import sleep

    _, base, ext = split_filename(infile)

    if ext[-3:].lower() == ".gz":
        ext = ext[:-3]

    with gzip.open(infile, "rb") as in_file:
        with open(os.path.abspath(base + ext), "wb") as out_file:
            shutil.copyfileobj(in_file, out_file, 128*1024)

    sleep(5)
    # in_file.close()
    # out_file.close()
    os.remove(infile)
    return out_file.name


def collect_pandas_df(
    subnet, ID, net_mets_csv_list, plot_switch, multi_nets, multimodal, embed
):
    """
    API for summarizing independent lists of pickled pandas dataframes of
     graph metrics for each modality, RSN, and roi.

    Parameters
    ----------
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    net_mets_csv_list : list
        List of file paths to pickled pandas dataframes as themselves.
    plot_switch : bool
        Activate summary plotting (histograms, ROC curves, etc.)
    multi_nets : list
        List of Yeo RSN's specified in workflow(s).
    multimodal : bool
        Indicates whether multiple modalities of input data have been
        specified.

    Returns
    -------
    combination_complete : bool
        If True, then collect_pandas_df completed successfully.

    """
    from pathlib import Path
    from pynets.statistics.individual.algorithms import collect_pandas_df_make
    from pynets.core.utils import load_runconfig

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    try:
        func_models = hardcoded_params["available_models"]["func_models"]
    except KeyError as e:
        print(e,
              "available functional models not sucessfully extracted"
              " from advanced.yaml"
              )
    try:
        dwi_models = hardcoded_params["available_models"][
            "dwi_models"]
    except KeyError as e:
        print(e,
              "available structural models not sucessfully extracted"
              " from advanced.yaml"
              )

    net_mets_csv_list = list(flatten(net_mets_csv_list))

    if multi_nets is not None:
        net_mets_csv_list_nets = net_mets_csv_list
        for subnet in multi_nets:
            net_mets_csv_list = list(
                set([i for i in net_mets_csv_list_nets if subnet in i])
            )
            if multimodal is True:
                net_mets_csv_list_dwi = list(
                    set(
                        [
                            i
                            for i in net_mets_csv_list
                            if i.split("model-")[1].split("_")[0] in
                            dwi_models
                        ]
                    )
                )
                combination_complete_dwi = collect_pandas_df_make(
                    net_mets_csv_list_dwi, ID, subnet, plot_switch, embed
                )
                net_mets_csv_list_func = list(
                    set(
                        [
                            i
                            for i in net_mets_csv_list
                            if i.split("model-")[1].split("_")[0] in
                            func_models
                        ]
                    )
                )
                combination_complete_func = collect_pandas_df_make(
                    net_mets_csv_list_func, ID, subnet, plot_switch, embed
                )

                if (
                    combination_complete_dwi is True
                    and combination_complete_func is True
                ):
                    combination_complete = True
                else:
                    combination_complete = False
            else:
                combination_complete = collect_pandas_df_make(
                    net_mets_csv_list, ID, subnet, plot_switch, embed
                )
    else:
        if multimodal is True:
            net_mets_csv_list_dwi = list(
                set(
                    [
                        i
                        for i in net_mets_csv_list
                        if i.split("model-")[1].split("_")[0] in dwi_models
                    ]
                )
            )
            combination_complete_dwi = collect_pandas_df_make(
                net_mets_csv_list_dwi, ID, subnet, plot_switch, embed
            )
            net_mets_csv_list_func = list(
                set(
                    [
                        i
                        for i in net_mets_csv_list
                        if i.split("model-")[1].split("_")[0] in func_models
                    ]
                )
            )
            combination_complete_func = collect_pandas_df_make(
                net_mets_csv_list_func, ID, subnet, plot_switch, embed
            )

            if combination_complete_dwi is \
                    True and combination_complete_func is True:
                combination_complete = True
            else:
                combination_complete = False
        else:
            combination_complete = collect_pandas_df_make(
                net_mets_csv_list, ID, subnet, plot_switch, embed
            )

    return combination_complete


def check_est_path_existence(est_path_list):
    """
    Checks for the existence of each graph estimated and saved to disk.

    Parameters
    ----------
    est_path_list : list
        List of file paths to .npy file containing graph with thresholding
        applied.

    Returns
    -------
    est_path_list_ex : list
        List of existing file paths to .npy file containing graph with
        thresholding applied.
    bad_ixs : int
        List of indices in est_path_list with non-existent and/or corrupt
        files.

    """
    est_path_list_ex = []
    bad_ixs = []
    i = -1

    for est_path in est_path_list:
        i = i + 1
        if op.isfile(est_path) is True:
            est_path_list_ex.append(est_path)
        else:
            print(f"\n\nWarning: Missing {est_path}...\n\n")
            bad_ixs.append(i)
            continue
    return est_path_list_ex, bad_ixs


def load_runconfig(location=None):
    import time
    import psutil
    import yaml
    import pkg_resources

    if not location:
        location = pkg_resources.resource_filename("pynets", "advanced.yaml")

    # asynchronous config parsing
    def proc_access(location, proc):
        try:
            return [location == f for f in proc.open_files()]
        except psutil.NoSuchProcess as e:
            # Catches race condition
            return [False]
        except psutil.AccessDenied as e:
            # If we're not root/admin sometimes we can't query processes
            return [False]

    while sum(list(flatten([proc_access(location, p) for p in
         psutil.process_iter(attrs=['name']) if ('python' in p.info['name'])
                                                and p.is_running() and
                                                p.username() != 'root']))
              ) > 0:
        time.sleep(1)

    with open(location, mode='r') as f:
        stream = f.read()
    f.close()

    return yaml.load(stream, Loader=yaml.FullLoader)


def save_coords_and_labels_to_json(coords, labels, dir_path,
                                   subnet='all_nodes', indices=None):
    """
    Save coordinates and labels to json.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    subnet : str
        Restricted sub-subnet name.

    Returns
    -------
    nodes_path : str
        Path to nodes json metadata file.

    """
    import json
    import os
    from pynets.core.utils import prune_suffices

    namer_dir = f"{dir_path}/nodes"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    if not isinstance(coords, list):
        coords = list(tuple(x) for x in coords)

    if not isinstance(labels, list):
        labels = list(labels)

    assert len(coords) == len(labels)

    if any(isinstance(sub, dict) for sub in labels):
        consensus_labs = True
    else:
        consensus_labs = False

    i = 0
    node_list = []
    for node in labels:
        node_dict = {}
        if consensus_labs is True and isinstance(node, tuple):
            lab, ix = node
            node_dict['index'] = str(ix)
            node_dict['label'] = str(lab)
        elif indices is not None:
            node_dict['index'] = str(indices[i])
            node_dict['label'] = str(node)
        else:
            node_dict['index'] = str(i)
            node_dict['label'] = str(node)
        node_dict['coord'] = coords[i]
        node_list.append(node_dict)
        i += 1

    nodes_path = f"{namer_dir}/nodes-{prune_suffices(subnet)}_" \
                 f"count-{len(labels)}.json"

    with open(nodes_path, 'w') as f:
        json.dump(node_list, f)

    return nodes_path


def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))


def get_template_tf(template_name, vox_size):
    from pathlib import Path
    from templateflow.api import get as get_template

    templateflow_home = Path(
        os.getenv(
            "TEMPLATEFLOW_HOME",
            os.path.join(os.getenv("HOME"), ".cache", "templateflow"),
        )
    )
    res = int(vox_size.strip("mm"))
    # str(get_template(
    # template_name, resolution=res, desc=None, suffix='T1w',
    # extension=['.nii', '.nii.gz']))

    template = str(
        get_template(
            template_name,
            resolution=res,
            desc="brain",
            suffix="T1w",
            extension=[".nii", ".nii.gz"],
        )
    )

    template_mask = str(
        get_template(
            template_name,
            resolution=res,
            desc="brain",
            suffix="mask",
            extension=[".nii", ".nii.gz"],
        )
    )

    return template, template_mask, templateflow_home


def save_nifti_parcels_map(ID, dir_path, subnet, net_parcels_map_nifti,
                           vox_size):
    """
    This function takes a Nifti1Image parcellation object resulting from some
    form of masking and saves it to disk.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer
        voxel intensities corresponding to ROI membership.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).

    Returns
    -------
    net_parcels_nii_path : str
        File path to Nifti1Image consisting of a 3D array with integer voxel
        intensities corresponding to ROI membership.

    """
    import os
    import pkg_resources
    import sys
    from pynets.core.utils import load_runconfig
    from nilearn.image import resample_to_img

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e,
              "No template specified in advanced.yaml"
              )

    namer_dir = f"{dir_path}/parcellations"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    net_parcels_nii_path = "%s%s%s%s%s" % (
        namer_dir,
        "/parcellation_space-",
        template_name,
        "%s" % ("%s%s" % ("_rsn-", subnet) if subnet is not None else ""),
        ".nii.gz",
    )

    template_brain = pkg_resources.resource_filename(
        "pynets", f"templates/standard/{template_name}_brain_{vox_size}.nii.gz"
    )

    if sys.platform.startswith('win') is False:
        try:
            template_img = nib.load(template_brain)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load MNI template. Do you have git-lfs "
                  f"installed?")
    else:
        try:
            template_img = nib.load(template_brain)
        except ImportError as e:
            print(e, f"\nCannot load MNI template. Do you have git-lfs "
                  f"installed?")

    net_parcels_map_nifti = resample_to_img(
        net_parcels_map_nifti, template_img, interpolation="nearest"
    )

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return net_parcels_nii_path


def save_ts_to_file(
    roi,
    subnet,
    ID,
    dir_path,
    ts_within_nodes,
    smooth,
    hpass,
    node_radius,
    signal,
):
    """
    This function saves the time-series 4D numpy array to disk as a .npy file.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ts_within_nodes : array
        2D m x n array consisting of the time-series signal for each ROI node
        where m = number of scans and n = number of ROI's, where ROI's are
        parcel volumes.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for time-series extraction.
    signal : str
        The name of a valid function used to reduce the time-series region
        extraction.

    Returns
    -------
    out_path_ts : str
        Path to .npy file containing array of fMRI time-series extracted from
        nodes.

    """
    import os

    namer_dir = f"{dir_path}/timeseries"
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    if hpass is None:
        hpass = 0

    if smooth is None:
        smooth = 0

    subnet_suff = f"_rsn-{subnet}" if subnet is not None else ""
    roi_suff = f"_roi-{op.basename(roi).split('.')[0]}" if roi is not None \
        else ""
    nodetype_suff = f"_nodetype-spheres-{node_radius}mm" if \
        ((node_radius is not None) and (node_radius != 'parc')) \
        else "_nodetype-parc"

    out_path_ts = f"{namer_dir}/nodetimeseries_sub-{ID}_" \
                  f"modality-func{subnet_suff}" \
                  f"{roi_suff}{nodetype_suff}_tol-{smooth}fwhm_hpass-" \
                  f"{hpass}Hz_signal-{signal}.npy"

    np.save(out_path_ts, ts_within_nodes)
    return out_path_ts


def as_list(x):
    """
    A function to convert an item to a list if it is not, or pass
    it through otherwise.
    """
    if not isinstance(x, list):
        return [x]
    else:
        return x


def merge_dicts(x, y):
    """
    A function to merge two dictionaries, making it easier for us to make
    modality specific queries for dwi images (since they have variable
    extensions due to having an nii, bval, and bvec file).
    """
    z = x.copy()
    z.update(y)
    return z


def timeout(seconds):
    """
    Timeout function for hung calculations.
    """
    from functools import wraps
    import errno
    import os
    import signal

    class TimeoutWarning(Exception):
        pass

    def decorator(func):
        def _handle_timeout(signum, frame):
            error_message = os.strerror(errno.ETIME)
            raise TimeoutWarning(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def filter_cols_from_targets(df, targets):
    base = r'^{}'
    expr = '(?=.*{})'
    out = df.columns[
        df.columns.str.contains(
            base.format(
                ''.join(
                    expr.format(w) for w in
                    targets)))]

    return out


def build_args_from_config(modality, arg_dict):
    import ast
    from pynets.core.utils import load_runconfig

    modalities = ["func", "dwi"]

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    try:
        func_models = hardcoded_params["available_models"]["func_models"]
    except KeyError as e:
        print(e,
              "available functional models not successfully extracted"
              " from advanced.yaml"
              )
    try:
        dwi_models = hardcoded_params["available_models"][
            "dwi_models"]
    except KeyError as e:
        print(e,
              "available structural models not successfully extracted"
              " from advanced.yaml"
              )

    arg_list = []
    for mod_ in modalities:
        arg_list.append(arg_dict[mod_])

    arg_list.append(arg_dict["gen"])

    args_dict_all = {}
    models = []
    for d in arg_list:
        if "mod" in d.keys():
            if d["mod"] == "None":
                del d["mod"]
                args_dict_all.update(d)
                continue
            if len(modality) == 1:
                if any(x in d["mod"] for x in
                       func_models) and ("dwi" in modality):
                    del d["mod"]
                elif any(x in d["mod"] for x in
                         dwi_models) and ("func" in modality):
                    del d["mod"]
            else:
                if any(x in d["mod"] for x in func_models) or any(
                    x in d["mod"] for x in dwi_models
                ):
                    models.append(ast.literal_eval(d["mod"]))
        args_dict_all.update(d)

    if len(modality) > 1:
        args_dict_all["mod"] = str(list(set(flatten(models))))

    print("Arguments parsed from config .json:\n")
    print(args_dict_all)

    for key, val in args_dict_all.items():
        if isinstance(val, str):
            args_dict_all[key] = ast.literal_eval(val)
    return args_dict_all


def check_template_loads(template, template_mask, template_name):
    import sys
    if sys.platform.startswith('win') is False:
        try:
            nib.load(template)
            nib.load(template_mask)
            return print('Local template detected...')
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load template {template_name} image or template "
                  f"mask. Do you have git-lfs installed?")
    else:
        try:
            nib.load(template)
            nib.load(template_mask)
            return print('Local template detected...')
        except ImportError as e:
            print(e, f"\nCannot load template {template_name} image or "
                     f"template mask. Do you have git-lfs installed?")


def save_4d_to_3d(in_file):
    from nipype.utils.filemanip import fname_presuffix

    files_3d = nib.four_to_three(nib.load(in_file))
    out_files = []
    for i, file_3d in enumerate(files_3d):
        out_file = fname_presuffix(in_file, suffix="_tmp_{}".format(i))
        file_3d.to_filename(out_file)
        out_files.append(out_file)
    del files_3d
    return out_files


def save_3d_to_4d(in_files):
    from nipype.utils.filemanip import fname_presuffix
    from nilearn.image import concat_imgs

    img_4d = concat_imgs([nib.load(img_3d) for img_3d in in_files],
                         auto_resample=True, ensure_ndim=4)
    out_file = fname_presuffix(in_files[0], suffix="_merged")
    img_4d.affine[3][3] = len(in_files)
    img_4d.to_filename(out_file)
    del img_4d
    return out_file


def kill_process_family(parent_pid):
    import os
    import psutil
    import signal

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signal.SIGTERM)
    os.kill(int(parent_pid), signal.SIGTERM)
    return


def dumpstacks(signal, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" %
                    (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' %
                        (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


class watchdog(object):
    def run(self):
        self.shutdown = threading.Event()
        watchdog_thread = threading.Thread(target=self._watchdog,
                                           name="watchdog")
        try:
            watchdog_thread.start()
            self._run()
        finally:
            self.shutdown.set()
            watchdog_thread.join()
        return 0

    # Default timeout to 2 hours of inactivity
    def _watchdog(self, watchdog_timeout=600):

        self.last_progress_time = time.time()

        while self.last_progress_time == time.time():
            if self.shutdown.wait(timeout=5):
                return
            last_progress_delay = time.time() - self.last_progress_time
            if last_progress_delay < watchdog_timeout:
                continue
            signal.signal(signal.SIGQUIT, dumpstacks)
            print(f"WATCHDOG: No progress in {last_progress_delay} "
                  f"seconds...")
            time.sleep(1)
            os.kill(0, 9)

    def _run(self):
        from pynets.cli.pynets_run import main
        while self.last_progress_time == time.time():
            main()
