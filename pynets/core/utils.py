#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import os
import os.path as op
import indexed_gzip
import nibabel as nib
import numpy as np
warnings.filterwarnings("ignore")


def get_file():
    """Get a file's base directory path."""
    base_path = str(__file__)
    return base_path


def do_dir_path(atlas, in_file):
    """
    Creates an atlas subdirectory from the base directory of the given subject's input file.

    Parameters
    ----------
    atlas : str
        Name of atlas parcellation used.
    in_file : str
        File path to dwi or func Nifti1Image object input.

    Returns
    -------
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    dir_path = "%s%s%s" % (op.dirname(op.realpath(in_file)), '/', atlas)
    if not op.exists(dir_path) and atlas is not None:
        os.makedirs(dir_path, exist_ok=True)
    elif atlas is None:
        raise ValueError("Error: cannot create directory for a null atlas!")
    return dir_path


def create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot, thr_type, hpass, parc):
    """
    Name the thresholded functional connectivity matrix file based on relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    thr_type : str
        Type of thresholding performed (e.g. prop, abs, dens, mst, disp)
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with all specified combinations of hyperparameter characteristics.
    """
    import os
    if (node_size is None) and (parc is True):
        node_size = '_parc'

    namer_dir = '{}/graphs'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-func_',
                                                         '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                                 network is not None else ''),
                                                         '%s' % ("%s%s%s" % ('roi-', op.basename(roi).split('.')[0],
                                                                             '_') if roi is not None else ''),
                                                         'est-', conn_model, '_',
                                                         '%s' % ("%s%s%s" % ('nodetype-spheres-', node_size, 'mm_') if
                                                                 ((node_size != 'parc') and (node_size is not None))
                                                                 else 'nodetype-parc_'),
                                                         "%s" % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if
                                                                 float(c_boot) > 0 else ''),
                                                         "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if
                                                                 float(smooth) > 0 else ''),
                                                         "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if
                                                                 hpass is not None else ''),
                                                         'thrtype-', thr_type, '_thr-', thr,
                                                         '.npy')

    return est_path


def create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples, track_type, thr_type,
                         parc):
    """
    Name the thresholded structural connectivity matrix file based on relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    thr_type : str
        Type of thresholding performed (e.g. prop, abs, dens, mst, disp)
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    """
    import os
    if (node_size is None) and (parc is True):
        node_size = '_parc'

    namer_dir = '{}/graphs'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-dwi_',
                                                         '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                                 network is not None
                                                                 else ''),
                                                         '%s' % ("%s%s%s" % ('roi-',
                                                                             op.basename(roi).split('.')[0], '_') if
                                                                 roi is not None else ''),
                                                         'est-', conn_model, '_',
                                                         '%s' % ("%s%s%s" % ('nodetype-spheres-', node_size, 'mm_') if
                                                                 ((node_size != 'parc') and (node_size is not None))
                                                                 else 'nodetype-parc_'),
                                                         "%s" % ("%s%s%s" % ('samples-', int(target_samples),
                                                                             'streams_') if float(target_samples) > 0
                                                                 else '_'),
                                                         'tt-', track_type, '_thrtype-', thr_type, '_thr-', thr,
                                                         '.npy')
    return est_path


def create_raw_path_func(ID, network, conn_model, roi, dir_path, node_size, smooth, c_boot, hpass, parc):
    """
    Name the raw functional connectivity matrix file based on relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with all specified combinations of hyperparameter characteristics.
    """
    import os
    if (node_size is None) and (parc is True):
        node_size = '_parc'

    namer_dir = '{}/graphs'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-func_',
                                                 '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                         network is not None else ''),
                                                 '%s' % ("%s%s%s" % ('roi-', op.basename(roi).split('.')[0], '_') if
                                                         roi is not None else ''),
                                                 'raw_', conn_model, '_',
                                                 '%s' % ("%s%s%s" % ('nodetype-spheres-', node_size, 'mm_') if
                                                         ((node_size != 'parc') and (node_size is not None)) else
                                                         'nodetype-parc_'),
                                                 "%s" % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if
                                                         float(c_boot) > 0 else ''),
                                                 "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if
                                                         float(smooth) > 0 else ''),
                                                 "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if
                                                         hpass is not None else ''),
                                                 '.npy')

    return est_path


def create_raw_path_diff(ID, network, conn_model, roi, dir_path, node_size, target_samples, track_type, parc):
    """
    Name the raw structural connectivity matrix file based on relevant graph-generating parameters.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.

    Returns
    -------
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    """
    import os
    if (node_size is None) and (parc is True):
        node_size = '_parc'

    namer_dir = '{}/graphs'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-dwi_',
                                                 '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                         network is not None else ''),
                                                 '%s' % ("%s%s%s" % ('roi-', op.basename(roi).split('.')[0], '_') if
                                                         roi is not None else ''), 'raw_', conn_model, '_',
                                                 '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if
                                                         ((node_size != 'parc') and (node_size is not None)) else
                                                         'parc_'),
                                                 "%s" % ("%s%s%s" % ('samples-', int(target_samples), 'streams_') if
                                                         float(target_samples) > 0 else ''), 'tt-', track_type, '.npy')
    return est_path


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

    namer_dir = '{}/netmetrics'.format(str(Path(dir_path).parent))
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = "%s%s%s%s" % (namer_dir, '/', est_path.split('/')[-1].split('.npy')[0], '_net_mets.csv')

    return out_path


def save_mat(conn_matrix, est_path, fmt='npy'):
    """
    Threshold a diffusion structural connectivity matrix using any of a variety of methods.

    Parameters
    ----------
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    fmt : str
        Format to save connectivity matrix/graph (e.g. .npy, .pkl, .graphml, .txt, .ssv, .csv). Default is .npy.
    """
    import networkx as nx
    G = nx.from_numpy_array(conn_matrix)
    G.graph['ecount'] = nx.number_of_edges(G)
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    if fmt == 'edgelist_csv':
        nx.write_weighted_edgelist(G, "%s%s" % (est_path.split('.npy')[0], '.csv'), encoding='utf-8')
    elif fmt == 'gpickle':
        nx.write_gpickle(G, "%s%s" % (est_path.split('.npy')[0], '.pkl'))
    elif fmt == 'graphml':
        nx.write_graphml(G, "%s%s" % (est_path.split('.npy')[0], '.graphml'))
    elif fmt == 'txt':
        np.savetxt("%s%s" % (est_path.split('.npy')[0], '.txt'), nx.to_numpy_matrix(G))
    elif fmt == 'npy':
        np.save(est_path, nx.to_numpy_matrix(G))
    elif fmt == 'edgelist_ssv':
        nx.write_weighted_edgelist(G, "%s%s" % (est_path.split('.npy')[0], '.ssv'), delimiter=" ", encoding='utf-8')
    else:
        raise ValueError('\nERROR: File format not supported!')

    return


def pass_meta_outs(conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist,
                   prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist, embed,
                   multimodal, multiplex):
    """
    Passes lists of iterable parameters as metadata.

    Parameters
    ----------
    conn_model_iterlist : list
       List of connectivity estimation model parameters (e.g. corr for correlation, cov for covariance,
       sps for precision covariance, partcorr for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an unweighted graph were binarized.
    embed : str
        Embed the ensemble(s) produced into feature vector(s). Options include: omni or mase.
    multimodal : bool
        Boolean indicating whether multiple modalities of input data have been specified.
    multiplex : int
        Switch indicating approach to multiplex graph analysis if multimodal is also True.

    Returns
    -------
    conn_model_iterlist : list
       List of connectivity estimation model parameters (e.g. corr for correlation, cov for covariance,
       sps for precision covariance, partcorr for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an unweighted graph were binarized.
    embed_iterlist : list
        List of booleans indicating whether omnibus embedding of graph population was performed.
    multimodal_iterlist : list
        List of booleans indicating whether multiple modalities of input data have been specified.
    """
    from pynets.core.utils import flatten
    from pynets.stats import netmotifs, embeddings

    if embed is not None:
        embeddings.build_embedded_connectome(list(flatten(est_path_iterlist)), list(flatten(ID_iterlist))[0],
                                             multimodal, embed)

    if (multiplex > 0) and (multimodal is True):
        multigraph_list_all = netmotifs.build_multigraphs(est_path_iterlist, list(flatten(ID_iterlist))[0])

    return conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist


def pass_meta_ins(conn_model, est_path, network, thr, prune, ID, roi, norm, binary):
    """
    Passes parameters as metadata.

    Parameters
    ----------
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
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
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
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
    network_iterlist = network
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
    return conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist


def pass_meta_ins_multi(conn_model_func, est_path_func, network_func, thr_func, prune_func, ID_func,
                        roi_func, norm_func, binary_func, conn_model_struct, est_path_struct, network_struct,
                        thr_struct, prune_struct, ID_struct, roi_struct, norm_struct, binary_struct):
    """
    Passes multimodal iterable parameters as metadata.

    Parameters
    ----------
    conn_model_func : str
       Functional connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision
       covariance, partcorr for partial correlation). sps type is used by default.
    est_path_func : str
        File path to .npy file containing functional graph with thresholding applied.
    network_func : str
        Functional resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    thr_func : float
        A value, between 0 and 1, to threshold the functional graph using any variety of methods
        triggered through other options.
    prune_func : bool
        Indicates whether to prune final functional graph of disconnected nodes/isolates.
    ID_func : str
        A subject id or other unique identifier for the functional workflow.
    roi_func : str
        File path to binarized/boolean region-of-interest Nifti1Image file applied to the functional data.
    norm_func : int
        Indicates method of normalizing resulting functional graph.
    binary_func : bool
        Indicates whether to binarize resulting graph edges to form an unweighted functional graph.
    conn_model_struct : str
       Diffusion structural connectivity estimation model (e.g. corr for correlation, cov for covariance,
       sps for precision covariance, partcorr for partial correlation). sps type is used by default.
    est_path_struct : str
        File path to .npy file containing diffusion structural graph with thresholding applied.
    network_struct : str
        Diffusion structural resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter
        nodes in the study of brain subgraphs.
    thr_struct : float
        A value, between 0 and 1, to threshold the diffusion structural graph using any variety of methods
        triggered through other options.
    prune_struct : bool
        Indicates whether to prune final diffusion structural graph of disconnected nodes/isolates.
    ID_struct : str
        A subject id or other unique identifier for the diffusion structural workflow.
    roi_struct : str
        File path to binarized/boolean region-of-interest Nifti1Image file applied too the dwi data.
    norm_struct : int
        Indicates method of normalizing resulting diffusion structural graph.
    binary_struct : bool
        Indicates whether to binarize resulting diffusion structural graph edges to form an unweighted graph.


    Returns
    -------
    conn_model_iterlist : list
       List of connectivity estimation model parameters (e.g. corr for correlation, cov for covariance,
       sps for precision covariance, partcorr for partial correlation). sps type is used by default.
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding applied.
    network_iterlist : list
        List of resting-state networks based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    thr_iterlist : list
        List of values, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune_iterlist : list
        List of booleans indicating whether final graphs were pruned of disconnected nodes/isolates.
    ID_iterlist : list
        List of repeated subject id strings.
    roi_iterlist : list
        List of file paths to binarized/boolean region-of-interest Nifti1Image files.
    norm_iterlist : list
        Indicates method of normalizing resulting graph.
    binary_iterlist : list
        List of booleans indicating whether resulting graph edges to form an unweighted graph were binarized.
    embed_iterlist : list
        List of booleans indicating whether omnibus embedding of graph population was performed.
    multimodal_iterlist : list
        List of booleans indicating whether multiple modalities of input data have been specified.
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
    return conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist


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
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def collect_pandas_df(network, ID, net_mets_csv_list, plot_switch, multi_nets, multimodal):
    """
    API for summarizing independent lists of pickled pandas dataframes of graph metrics for each modality, RSN, and roi.

    Parameters
    ----------
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    net_mets_csv_list : list
        List of file paths to pickled pandas dataframes as themselves.
    plot_switch : bool
        Activate summary plotting (histograms, ROC curves, etc.)
    multi_nets : list
        List of Yeo RSN's specified in workflow(s).
    multimodal : bool
        Indicates whether multiple modalities of input data have been specified.

    Returns
    -------
    combination_complete : bool
        If True, then collect_pandas_df completed successfully
    """
    from pathlib import Path
    import yaml
    from pynets.core.utils import flatten
    from pynets.stats.netstats import collect_pandas_df_make

    # Available functional and structural connectivity models
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

    net_mets_csv_list = list(flatten(net_mets_csv_list))

    if multi_nets is not None:
        net_mets_csv_list_nets = net_mets_csv_list
        for network in multi_nets:
            net_mets_csv_list = list(set([i for i in net_mets_csv_list_nets if network in i]))
            if multimodal is True:
                net_mets_csv_list_dwi = list(set([i for i in net_mets_csv_list if i.split('mets_')[1].split('_')[0]
                                                   in struct_models]))
                combination_complete_dwi = collect_pandas_df_make(net_mets_csv_list_dwi, ID, network, plot_switch)
                net_mets_csv_list_func = list(set([i for i in net_mets_csv_list if
                                                    i.split('mets_')[1].split('_')[0] in func_models]))
                combination_complete_func = collect_pandas_df_make(net_mets_csv_list_func, ID, network, plot_switch)

                if combination_complete_dwi is True and combination_complete_func is True:
                    combination_complete = True
                else:
                    combination_complete = False
            else:
                combination_complete = collect_pandas_df_make(net_mets_csv_list, ID, network, plot_switch)
    else:
        if multimodal is True:
            net_mets_csv_list_dwi = list(set([i for i in net_mets_csv_list if i.split('mets_')[1].split('_')[0] in
                                               struct_models]))
            combination_complete_dwi = collect_pandas_df_make(net_mets_csv_list_dwi, ID, network, plot_switch)
            net_mets_csv_list_func = list(set([i for i in net_mets_csv_list if i.split('mets_')[1].split('_')[0]
                                                in func_models]))
            combination_complete_func = collect_pandas_df_make(net_mets_csv_list_func, ID, network, plot_switch)

            if combination_complete_dwi is True and combination_complete_func is True:
                combination_complete = True
            else:
                combination_complete = False
        else:
            combination_complete = collect_pandas_df_make(net_mets_csv_list, ID, network, plot_switch)

    return combination_complete


def check_est_path_existence(est_path_list):
    """
    Checks for the existence of each graph estimated and saved to disk.

    Parameters
    ----------
    est_path_list : list
        List of file paths to .npy file containing graph with thresholding applied.
    Returns
    -------
    est_path_list_ex : list
        List of existing file paths to .npy file containing graph with thresholding applied.
    bad_ixs : int
        List of indices in est_path_list with non-existent and/or corrupt files.
    """
    est_path_list_ex = []
    bad_ixs = []
    i = -1

    for est_path in est_path_list:
        i = i + 1
        if op.isfile(est_path) is True:
            est_path_list_ex.append(est_path)
        else:
            print("%s%s%s" % ('\n\nWarning: Missing ', est_path, '...\n\n'))
            bad_ixs.append(i)
            continue
    return est_path_list_ex, bad_ixs


def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """
    Save RSN coordinates and labels to pickle files.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.

    Returns
    -------
    coord_path : str
        Path to pickled coordinates list.
    labels_path : str
        Path to pickled labels list.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    import os

    namer_dir = '{}/nodes'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    # Save coords to pickle
    coord_path = "%s%s%s%s" % (namer_dir, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)

    # Save labels to pickle
    labels_path = "%s%s%s%s" % (namer_dir, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)

    return coord_path, labels_path


def save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti):
    """
    This function takes a Nifti1Image parcellation object resulting from some form of masking and saves it to disk.

    Parameters
    ----------
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.

    Returns
    -------
    net_parcels_nii_path : str
        File path to Nifti1Image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    """
    import os

    namer_dir = '{}/parcellations'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    net_parcels_nii_path = "%s%s%s%s%s%s%s" % (namer_dir, '/', str(ID), '_parcels_masked',
                                               '%s' % ("%s%s" % ('_rsn-', network) if network is not None else ''),
                                               '%s' % ("%s%s" % ('_roi-', op.basename(roi).split('.')[0])
                                                       if roi is not None else ''),
                                               '.nii.gz')

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return net_parcels_nii_path


def save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot, smooth, hpass, node_size):
    """
    This function saves the time-series 4D numpy array to disk as a .npy file.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ts_within_nodes : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's, where ROI's are parcel volumes.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for time-series extraction.

    Returns
    -------
    out_path_ts : str
        Path to .npy file containing array of fMRI time-series extracted from nodes.
    """
    import os
    namer_dir = '{}/timeseries'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    # Save time series as npy file
    out_path_ts = "%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_',
                                              '%s' % ("%s%s%s" % ('rsn-', network, '_') if network is not None else ''),
                                              '%s' % ("%s%s%s" % ('roi-', op.basename(roi).split('.')[0], '_') if
                                                      roi is not None else ''),
                                              '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if
                                                      ((node_size != 'parc') and (node_size is not None)) else
                                                      'parc_'),
                                              '%s' % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if
                                                      float(c_boot) > 0 else ''),
                                              "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if
                                                      float(smooth) > 0 else ''),
                                              "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if
                                                      hpass is not None else ''),
                                              'node_ts.npy')

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


def build_hp_dict(file_renamed, atlas, modality, hyperparam_dict, hyperparams):
    """
    A function to build a hyperparameter dictionary by parsing a given net_mets file path.
    """
    hyperparam_dict['node_type'] = file_renamed.split('nodetype-')[1].split('_')[0]
    hyperparam_dict['atlas'] = atlas
    hyperparam_dict['thrtype'] = file_renamed.split('thrtype-')[1].split('_')[0]

    if modality == 'func':
        hyperparam_dict['estimator'] = file_renamed.split('est-')[1].split('_')[0]
        hyperparams.append('estimator')
        if 'smooth-' in file_renamed:
            hyperparam_dict['smooth'] = file_renamed.split('smooth-')[1].split('_')[0].split('fwhm')[0]
            hyperparams.append('smooth')
        if 'hpass-' in file_renamed:
            hyperparam_dict['hpass'] = file_renamed.split('hpass-')[1].split('_')[0].split('Hz')[0]
            hyperparams.append('hpass')
    return hyperparam_dict, hyperparams


class build_sql_db(object):
    """
    A SQL exporter for AUC metrics.
    """
    def __init__(self, dir_path, ID):
        from sqlalchemy import create_engine
        self.ID = ID
        db_file = dir_path + '/' + self.ID + '_auc_db.sql'
        self.engine = create_engine('sqlite:///' + db_file, echo=False, encoding='utf-8')
        self.hyperparams = None
        self.modality = None
        return

    def create_modality_table(self, modality):
        from sqlalchemy.sql import text
        self.modality = modality
        statement = """CREATE TABLE IF NOT EXISTS """ + self.modality + """(id TEXT);"""
        self.engine.execute(text(statement.replace("'", "")))

    def add_hp_columns(self, hyperparams):
        from sqlalchemy.sql import text
        self.hyperparams = hyperparams
        for hp in self.hyperparams:
            try:
                statement = """ALTER TABLE """ + self.modality + """ ADD COLUMN """ + hp + """;"""
                self.engine.execute(text(statement.replace("'", "")))
            except:
                continue
        return

    def add_row_from_df(self, df_summary_auc, hyperparam_dict):
        import pandas as pd
        df_summary_auc_ext = pd.concat([pd.DataFrame.from_dict(hyperparam_dict, orient='index').transpose(),
                                        df_summary_auc], axis=1)
        df_summary_auc_ext.to_sql(self.modality, con=self.engine, index=False, chunksize=1000, if_exists='append')
        return

