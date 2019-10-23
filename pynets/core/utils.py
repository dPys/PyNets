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
import nibabel as nib
import numpy as np
from pynets.stats.netstats import extractnetstats
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface
warnings.filterwarnings("ignore")


def get_file():
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
        File path to -dwi or -func Nifti1Image object input.

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
    Name the thresholded functional connectivity matrix file based on relevant graph-generating parameters

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

    namer_dir = dir_path + '/graphs'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_',
                                                       '%s' % (network + '_' if network is not None else ''),
                                                       '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                       'est-', conn_model, '_thr-', thr, thr_type, '_',
                                                       '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if ((node_size != 'parc') and (node_size is not None)) else 'parc_'),
                                                       "%s" % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if float(c_boot) > 0 else ''),
                                                       "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if float(smooth) > 0 else ''),
                                                       "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if hpass is not None else ''),
                                                       'func.npy')

    return est_path


def create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples, track_type, thr_type,
                         parc):
    """
    Name the thresholded structural connectivity matrix file based on relevant graph-generating parameters

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

    namer_dir = dir_path + '/graphs'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_',
                                                     '%s' % (network + '_' if network is not None else ''),
                                                     '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                     'est-', conn_model, '_thr-', thr, thr_type, '_',
                                                     '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if ((node_size != 'parc') and (node_size is not None)) else 'parc_'),
                                                     "%s" % ("%s%s%s" % ('samples-', int(target_samples), 'streams_') if float(target_samples) > 0 else '_'),
                                                     track_type, '_dwi.npy')
    return est_path


def create_raw_path_func(ID, network, conn_model, roi, dir_path, node_size, smooth, c_boot, hpass, parc):
    """
    Name the raw functional connectivity matrix file based on relevant graph-generating parameters

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

    namer_dir = dir_path + '/graphs'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_',
                                                 '%s' % (network + '_' if network is not None else ''),
                                                 '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                 'raw_', conn_model, '_',
                                                 '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if ((node_size != 'parc') and (node_size is not None)) else 'parc_'),
                                                 "%s" % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if float(c_boot) > 0 else ''),
                                                 "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if float(smooth) > 0 else ''),
                                                 "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if hpass is not None else ''),
                                                 'func.npy')

    return est_path


def create_raw_path_diff(ID, network, conn_model, roi, dir_path, node_size, target_samples, track_type, parc):
    """
    Name the raw structural connectivity matrix file based on relevant graph-generating parameters

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

    namer_dir = dir_path + '/graphs'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_',
                                               '%s' % (network + '_' if network is not None else ''),
                                               '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                               'raw_', conn_model, '_',
                                               '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if ((node_size != 'parc') and (node_size is not None)) else 'parc_'),
                                               "%s" % ("%s%s%s" % ('samples-', int(target_samples), 'streams_') if float(target_samples) > 0 else '_'),
                                               track_type, '_dwi.npy')
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

    namer_dir = str(Path(dir_path).parent) + '/netmetrics'
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
    from pynets.core.utils import build_embedded_connectome, flatten
    from pynets.stats import netmotifs

    if embed is not None:
        build_embedded_connectome(list(flatten(est_path_iterlist)), list(flatten(ID_iterlist))[0], multimodal, embed)

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


def CollectPandasJoin(net_mets_csv):
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


def build_embedded_connectome(est_path_iterlist, ID, multimodal, embed):
    import os
    import yaml
    from pathlib import Path
    from pynets.core.utils import flatten
    from sklearn.feature_selection import VarianceThreshold
    from graspy.embed import OmnibusEmbed, ClassicalMDS, MultipleASE
    """
    Embeds ensemble population of graphs into an embedded ensemble feature vector.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy file containing graph with thresholding applied.
    ID : str
        A subject id or other unique identifier.
    multimodal_iterlist : list
        List of booleans indicating whether multiple modalities of input data have been specified.
    type : str
        Specifies which type of ensemble embedding will be performed. Options include `omni` and `mase`.
    """

    def _omni_embed(pop_array, subgraph_name='whole_brain'):
        variance_threshold = VarianceThreshold(threshold=0.00001)
        diags = np.array([np.triu(pop_array[i]) for i in range(len(pop_array))])
        diags_red = diags.reshape(diags.shape[0], diags.shape[1] * diags.shape[2])
        var_thr = variance_threshold.fit(diags_red.T)
        graphs_ix_keep = var_thr.get_support(indices=True)
        pop_array_red = [pop_array[i] for i in graphs_ix_keep]

        # Omnibus embedding -- random dot product graph (rdpg)
        print("%s%s%s%s%s" % ('Embedding ensemble for atlas: ', atlas, ' and ', subgraph_name, '...'))
        omni = OmnibusEmbed(check_lcc=False)
        mds = ClassicalMDS()
        try:
            omni_fit = omni.fit_transform(pop_array_red)
        except:
            omni_fit = omni.fit_transform(pop_array)

        # Transform omnibus tensor into dissimilarity feature
        mds_fit = mds.fit_transform(omni_fit)

        dir_path = str(Path(os.path.dirname(graph_path)).parent)

        namer_dir = dir_path + '/embeddings'
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        out_path = "%s%s%s%s%s%s%s%s" % (namer_dir, '/', list(flatten(ID))[0], '_omnetome_', atlas, '_', subgraph_name,
                                         '.npy')
        print('Saving...')
        np.save(out_path, mds_fit)
        del mds, mds_fit, omni, omni_fit
        return out_path

    def _mase_embed(pop_array, subgraph_name='whole_brain'):
        variance_threshold = VarianceThreshold(threshold=0.00001)
        diags = np.array([np.triu(pop_array[i]) for i in range(len(pop_array))])
        diags_red = diags.reshape(diags.shape[0], diags.shape[1] * diags.shape[2])
        var_thr = variance_threshold.fit(diags_red.T)
        graphs_ix_keep = var_thr.get_support(indices=True)
        pop_array_red = [pop_array[i] for i in graphs_ix_keep]

        # Omnibus embedding -- random dot product graph (rdpg)
        print("%s%s%s%s%s" % ('Embedding ensemble for atlas: ', atlas, ' and ', subgraph_name, '...'))
        mase = MultipleASE()
        try:
            mase_fit = mase.fit_transform(pop_array_red)
        except:
            mase_fit = mase.fit_transform(pop_array)

        dir_path = str(Path(os.path.dirname(graph_path)).parent)
        namer_dir = dir_path + '/embeddings'
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        out_path = "%s%s%s%s%s%s%s%s" % (namer_dir, '/', list(flatten(ID))[0], '_masetome_', atlas, '_', subgraph_name,
                                         '.npy')
        print('Saving...')
        np.save(out_path, mase.scores_)
        del mase, mase_fit
        return out_path

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

    if multimodal is True:
        out_path = None
        atlases = list(set([x.split('/')[-3].split('/')[0] for x in est_path_iterlist]))
        parcel_dict_func = dict.fromkeys(atlases)
        parcel_dict_dwi = dict.fromkeys(atlases)

        est_path_iterlist_dwi = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                          struct_models]))
        est_path_iterlist_func = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                           func_models]))

        func_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_func]))

        dwi_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_dwi]))

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

            pop_list = []
            for pop_ref in parcel_dict_func[atlas]:
                # RSN case
                if isinstance(pop_ref, dict):
                    rsns = [i.split('_')[1] for i in list(pop_ref.keys())]
                    i = 0
                    for rsn in rsns:
                        pop_rsn_list = []
                        for graph in pop_ref[rsn]:
                            pop_list.append(np.load(graph))
                        if len(pop_rsn_list) > 1:
                            if len(list(set([i.shape for i in pop_rsn_list]))) > 1:
                                raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population '
                                                     'that precludes embedding')
                            if embed == 'omni':
                                out_path = _omni_embed(pop_list, rsns[i])
                            elif embed == 'mase':
                                out_path = _mase_embed(pop_list, rsns[i])
                            else:
                                raise ValueError('Embedding type not recognized. Presently supported options include: '
                                                 'omni or mase')
                        else:
                            print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                            pass
                        i = i + 1
                else:
                    pop_list.append(np.load(pop_ref))
            if len(pop_list) > 1:
                if len(list(set([i.shape for i in pop_list]))) > 1:
                    raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population that '
                                         'precludes embedding')
                if embed == 'omni':
                    out_path = _omni_embed(pop_list)
                elif embed == 'mase':
                    out_path = _mase_embed(pop_list)
                else:
                    raise ValueError('Embedding type not recognized. Presently supported options include: '
                                     'omni or mase')
            else:
                print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                pass

            pop_list = []
            for pop_ref in parcel_dict_dwi[atlas]:
                # RSN case
                if isinstance(pop_ref, dict):
                    rsns = [i.split('_')[1] for i in list(pop_ref.keys())]
                    i = 0
                    for rsn in rsns:
                        pop_rsn_list = []
                        for graph in pop_ref[rsn]:
                            pop_list.append(np.load(graph))
                        if len(pop_rsn_list) > 1:
                            if len(list(set([i.shape for i in pop_rsn_list]))) > 1:
                                raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population '
                                                     'that precludes embedding')
                            if embed == 'omni':
                                out_path = _omni_embed(pop_list, rsns[i])
                            elif embed == 'mase':
                                out_path = _mase_embed(pop_list, rsns[i])
                            else:
                                raise ValueError('Embedding type not recognized. Presently supported options include: '
                                                 'omni or mase')
                        else:
                            print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                            pass
                        i = i + 1
                else:
                    pop_list.append(np.load(pop_ref))
            if len(pop_list) > 1:
                if len(list(set([i.shape for i in pop_list]))) > 1:
                    raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population that '
                                         'precludes embedding')
                if embed == 'omni':
                    out_path = _omni_embed(pop_list)
                elif embed == 'mase':
                    out_path = _mase_embed(pop_list)
                else:
                    raise ValueError('Embedding type not recognized. Presently supported options include: '
                                     'omni or mase')
            else:
                print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                pass

    elif (multimodal is False) and (len(est_path_iterlist) > 1):
        atlases = list(set([x.split('/')[-3].split('/')[0] for x in est_path_iterlist]))
        parcel_dict = dict.fromkeys(atlases)
        subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist if i.split('_est')[0].split('/')[-1] != ID]))
        out_path = None
        for atlas in atlases:
            if len(subnets) > 1:
                parcel_dict[atlas] = {}
                for sub_net in subnets:
                    parcel_dict[atlas][sub_net] = []
            else:
                parcel_dict[atlas] = []

            for graph_path in est_path_iterlist:
                if atlas in graph_path:
                    if len(subnets) > 1:
                        for sub_net in subnets:
                            if sub_net in graph_path:
                                parcel_dict[atlas][sub_net].append(graph_path)
                    else:
                        parcel_dict[atlas].append(graph_path)
            pop_list = []
            for pop_ref in parcel_dict[atlas]:
                # RSN case
                if isinstance(pop_ref, dict):
                    rsns = [i.split('_')[1] for i in list(pop_ref.keys())]
                    i = 0
                    for rsn in rsns:
                        pop_rsn_list = []
                        for graph in pop_ref[rsn]:
                            pop_list.append(np.load(graph))
                        if len(pop_rsn_list) > 1:
                            if len(list(set([i.shape for i in pop_rsn_list]))) > 1:
                                raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population '
                                                     'that precludes embedding')
                            out_path = _omni_embed(pop_rsn_list, rsns[i])
                        else:
                            print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                            pass
                        i = i + 1
                else:
                    pop_list.append(np.load(pop_ref))
            if len(pop_list) > 1:
                if len(list(set([i.shape for i in pop_list]))) > 1:
                    raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population that '
                                         'precludes embedding')
                if embed == 'omni':
                    out_path = _omni_embed(pop_list)
                elif embed == 'mase':
                    out_path = _mase_embed(pop_list)
                else:
                    raise ValueError('Embedding type not recognized. Presently supported options include: '
                                     'omni or mase')
            else:
                print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                pass
    else:
        raise RuntimeWarning('ERROR: Only one graph sampled, omnibus embedding not appropriate.')

    return out_path


def collect_pandas_df_make(net_mets_csv_list, ID, network, plot_switch, nc_collect=False, create_summary=True):
    """
    Summarize list of pickled pandas dataframes of graph metrics unique to eacho unique combination of hyperparameters.

    Parameters
    ----------
    net_mets_csv_list : list
        List of file paths to pickled pandas dataframes as themselves.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the
        study of brain subgraphs.
    plot_switch : bool
        Activate summary plotting (histograms, central tendency, AUC, etc.)

    Returns
    -------
    combination_complete : bool
        If True, then data integration completed successfully.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from itertools import groupby
    import re

    # Check for existence of net_mets csv files, condensing final list to only those that were actually produced.
    net_mets_csv_list_exist = []
    for net_mets_csv in list(net_mets_csv_list):
        if op.isfile(net_mets_csv) is True:
            net_mets_csv_list_exist.append(net_mets_csv)

    if len(list(net_mets_csv_list)) > len(net_mets_csv_list_exist):
        raise UserWarning('Warning! Number of actual models produced less than expected. Some graphs were excluded')
        combination_complete = False

    net_mets_csv_list = net_mets_csv_list_exist
    subject_path = op.dirname(op.dirname(net_mets_csv_list[0]))

    if len(net_mets_csv_list) > 1:
        print("%s%s%s" % ('\n\nList of result files to concatenate:\n', str(net_mets_csv_list), '\n\n'))
        net_mets_csv_list.sort()

        models = []
        for file_ in net_mets_csv_list:
            models.append(op.basename(file_))

        # Group by secondary attributes
        models_grouped = [list(x) for x in zip(*[list(g) for k, g in groupby(models, lambda s: s.split('thr-')[1].split('_')[0])])]

        # group by estimator type
        models_all = []
        for mod in models_grouped:
            models_all.extend([list(i) for j, i in groupby(mod, lambda s: s.split('est-')[1].split('_')[0])])

        meta = dict()
        non_decimal = re.compile(r'[^\d.]+')
        for thr_set in range(len(models_all)):
            meta[thr_set] = dict()
            meta[thr_set]['dataframes'] = dict()
            for i in models_all[thr_set]:
                thr = non_decimal.sub('', i.split('thr-')[1].split('_')[0])
                _file = subject_path + '/netmetrics/' + i
                df = pd.read_csv(_file)
                if nc_collect is False:
                    node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]
                    df = df.drop(node_cols, axis=1)
                meta[thr_set]['dataframes'][thr] = df

        # For each unique threshold set, for each graph measure, extract AUC
        auc_dir = subject_path + '/netmetrics/auc/'
        if not os.path.isdir(auc_dir):
            os.makedirs(auc_dir, exist_ok=True)

        for thr_set in meta.keys():
            df_summary = pd.concat(meta[thr_set]['dataframes'].values())
            df_summary['thr'] = meta[thr_set]['dataframes'].keys()
            meta[thr_set]['summary_dataframe'] = df_summary
            df_summary_auc = df_summary.iloc[[0]]
            df_summary_auc.columns = [col + '_auc' for col in df_summary.columns]

            print('\nCalculating AUC for:')
            for measure in df_summary.columns[:-1]:
                # Get Area Under the Curve
                print(measure)
                df_summary_nonan = df_summary.dropna(subset=[measure], axis=0)
                df_summary_auc[measure] = np.trapz(np.array(df_summary_nonan[measure]).astype('float64'),
                                                   np.array(df_summary_nonan['thr']).astype('float64'))
            meta[thr_set]['auc_dataframe'] = df_summary_auc

            auc_outfile = auc_dir + list(set([re.sub(r'thr\-\d+\.*\d+', '', i).replace('neat', 'auc') for i in models_all[thr_set]]))[0]
            df_summary_auc.to_csv(auc_outfile, header=True, index=False, chunksize=100000, compression='gzip',
                                  encoding='utf-8')

        if create_summary is True:
            try:
                summary_dir = subject_path + '/netmetrics/summary/'
                if not os.path.isdir(summary_dir):
                    os.makedirs(summary_dir, exist_ok=True)

                # Concatenate and find mean across dataframes
                print('Concatenating frames...')
                df_concat = pd.concat([meta[thr_set]['auc_dataframe'] for thr_set in meta.keys()])
                measures = list(df_concat.columns)
                if plot_switch is True:
                    from pynets.plotting import plot_gen
                    plot_gen.plot_graph_measure_hists(df_concat, measures, file_)
                df_concatted_mean = df_concat.loc[:, measures].mean(skipna=True).to_frame().transpose()
                df_concatted_median = df_concat.loc[:, measures].median(skipna=True).to_frame().transpose()
                df_concatted_mode = df_concat.loc[:, measures].mode(dropna=True)
                df_concatted_mean.columns = [str(col) + '_mean' for col in df_concatted_mean.columns]
                df_concatted_median.columns = [str(col) + '_median' for col in df_concatted_median.columns]
                df_concatted_mode.columns = [str(col) + '_mode' for col in df_concatted_mode.columns]
                result = pd.concat([df_concatted_mean, df_concatted_median, df_concatted_mode], axis=1)
                df_concatted_final = result.reindex(sorted(result.columns), axis=1)
                print('\nConcatenating dataframes for ' + str(ID) + '...\n')
                net_csv_summary_out_path = "%s%s%s%s%s%s" % (summary_dir, '/', str(ID), '_net_mets',
                                                             '%s' % ('_' + network if network is not None else ''),
                                                             '_mean.csv')
                df_concatted_final.to_csv(net_csv_summary_out_path, index=False)
                combination_complete = True
            except RuntimeWarning:
                combination_complete = False
                print("%s%s%s" % ('\nWARNING: DATAFRAME CONCATENATION FAILED FOR ', str(ID), '!\n'))
                pass
        else:
            combination_complete = True
    else:
        if network is not None:
            print("%s%s%s%s%s" % ('\nSingle dataframe for the ', network, ' network for subject ', ID, '\n'))
        else:
            print("%s%s%s" % ('\nSingle dataframe for subject ', ID, '\n'))
        combination_complete = True

    return combination_complete


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
    from pynets.core.utils import collect_pandas_df_make, flatten

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

    namer_dir = dir_path + '/nodes'
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

    namer_dir = dir_path + '/parcellations'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    net_parcels_nii_path = "%s%s%s%s%s%s%s" % (namer_dir, '/', str(ID), '_parcels_masked',
                                               '%s' % ('_' + network if network is not None else ''),
                                               '%s' % ('_' + op.basename(roi).split('.')[0] if roi is not None else ''),
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
    namer_dir = dir_path + '/timeseries'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    # Save time series as npy file
    out_path_ts = "%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_', '%s' % (network + '_' if network is not None else ''),
                                              '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                              '%s' % ("%s%s%s" % ('spheres-', node_size, 'mm_') if ((node_size != 'parc') and (node_size is not None)) else 'parc_'),
                                              '%s' % ("%s%s%s" % ('boot-', int(c_boot), 'iter_') if float(c_boot) > 0 else ''),
                                              "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if float(smooth) > 0 else ''),
                                              "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if hpass is not None else ''),
                                              'ts_from_nodes.npy')

    np.save(out_path_ts, ts_within_nodes)
    return out_path_ts


def timeseries_bootstrap(tseries, block_size):
    """
    Generates a bootstrap sample derived from the input time-series.
    Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries

    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
       changes in brain connectivity for functional MRI. Statistica Sinica,
       special issue on Statistical Challenges and Advances in Brain Science,
       2008, 18: 1253-1268.
    """
    np.random.seed(int(42))

    # calculate number of blocks
    k = int(np.ceil(float(tseries.shape[0]) / block_size))

    # generate random indices of blocks
    r_ind = np.floor(np.random.rand(1, k) * tseries.shape[0])
    blocks = np.dot(np.arange(0, block_size)[:, np.newaxis], np.ones([1, k]))

    block_offsets = np.dot(np.ones([block_size, 1]), r_ind)
    block_mask = (blocks + block_offsets).flatten('F')[:tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])

    return tseries[block_mask.astype('int'), :], block_mask.astype('int')


def rescale_bvec(bvec, bvec_rescaled):
    """
    Normalizes b-vectors to be of unit length for the non-zero b-values. If the
    b-value is 0, the vector is untouched.

    Parameters
    ----------
    bvec : str
        File name of the original b-vectors file.
    bvec_rescaled : str
        File name of the new (normalized) b-vectors file. Must have extension `.bvec`.

    Returns
    -------
    bvec_rescaled : str
        File name of the new (normalized) b-vectors file. Must have extension `.bvec`.
    """
    bv1 = np.array(np.loadtxt(bvec))
    # Enforce proper dimensions
    bv1 = bv1.T if bv1.shape[0] == 3 else bv1

    # Normalize values not close to norm 1
    bv2 = [b / np.linalg.norm(b) if not np.isclose(np.linalg.norm(b), 0)
           else b for b in bv1]
    np.savetxt(bvec_rescaled, bv2)
    return bvec_rescaled


def make_mean_b0(in_file):
    import time

    b0_img = nib.load(in_file)
    b0_img_data = b0_img.get_data()
    mean_b0 = np.mean(b0_img_data, axis=3, dtype=b0_img_data.dtype)
    mean_file_out = in_file.split(".nii.gz")[0] + "_mean_b0.nii.gz"
    nib.save(
        nib.Nifti1Image(mean_b0, affine=b0_img.affine, header=b0_img.header),
        mean_file_out,
    )
    while os.path.isfile(mean_file_out) is False:
        time.sleep(1)
    return mean_file_out


def make_gtab_and_bmask(fbval, fbvec, dwi_file, network, node_size, atlas, b0_thr=100):
    """
    Create gradient table from bval/bvec, and a mean B0 brain mask.

    Parameters
    ----------
    fbval : str
        File name of the b-values file.
    fbvec : str
        File name of the b-vectors file.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.

    Returns
    -------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    nodif_b0_bet : str
        File path to mean brain-extracted B0 image.
    B0_mask : str
        File path to mean B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import os
    from dipy.io import save_pickle
    import os.path as op
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from pynets.core.utils import rescale_bvec, make_mean_b0

    outdir = op.dirname(dwi_file)

    namer_dir = outdir + '/dmri_tmp'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    B0_bet = "%s%s" % (namer_dir, "/mean_B0_bet.nii.gz")
    B0_mask = "%s%s" % (namer_dir, "/mean_B0_bet_mask.nii.gz")
    bvec_rescaled = "%s%s" % (namer_dir, "/bvec_scaled.bvec")
    gtab_file = "%s%s" % (namer_dir, "/gtab.pkl")
    all_b0s_file = "%s%s" % (namer_dir, "/all_b0s.nii.gz")

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    # Correct any corrupted
    bvecs[np.where(np.any(abs(bvecs) >= 10, axis=1) == True)] = [1, 0, 0]
    bvecs[np.where(bvals == 0)] = 0
    if len(bvecs[np.where(np.logical_and(bvals > 50, np.all(abs(bvecs) == np.array([0, 0, 0]), axis=1)))]) > 0:
        raise ValueError('WARNING: Encountered potentially corrupted bval/bvecs. Check to ensure volumes with a '
                         'diffusion weighting are not being treated as B0\'s along the bvecs')
    # Save corrected
    np.savetxt(fbval, bvals)
    np.savetxt(fbvec, bvecs)
    bvec_rescaled = rescale_bvec(fbvec, bvec_rescaled)
    if fbval and bvec_rescaled:
        bvals, bvecs = read_bvals_bvecs(fbval, bvec_rescaled)
    else:
        raise ValueError('Either bvals or bvecs files not found (or rescaling failed)!')

    # Creating the gradient table
    gtab = gradient_table(bvals, bvecs)

    # Correct b0 threshold
    gtab.b0_threshold = b0_thr

    # Get b0 indices
    b0s = np.where(gtab.bvals <= gtab.b0_threshold)[0]
    print("%s%s" % ('b0\'s found at: ', b0s))

    # Correct b0 mask
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    # Extract and Combine all b0s collected, make mean b0
    print("Extracting b0's...")
    b0_vols = []
    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_data()
    for b0 in b0s:
        print(b0)
        b0_vols.append(dwi_data[:, :, :, b0])

    all_b0s = np.stack(b0_vols, axis=3)
    all_b0s_aff = dwi_img.affine.copy()
    all_b0s_aff[3][3] = len(b0_vols)
    nib.save(nib.Nifti1Image(all_b0s, affine=all_b0s_aff), all_b0s_file)
    mean_b0_file = make_mean_b0(all_b0s_file)

    # Create mean b0 brain mask
    cmd = 'bet ' + mean_b0_file + ' ' + B0_bet + ' -m -f 0.2'
    os.system(cmd)

    return gtab_file, B0_bet, B0_mask, dwi_file


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
    extensions due to having an nii.gz, bval, and bvec file).
    """
    z = x.copy()
    z.update(y)
    return z


def create_temporary_copy(path, temp_file_name, fmt):
    """
    A function to create temporary file equivalents
    """
    import tempfile, shutil
    from time import strftime
    import uuid
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    temp_dir = tempfile.gettempdir() + '/' + run_uuid
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = "%s%s%s%s" % (temp_dir, '/', temp_file_name, fmt)
    shutil.copy2(path, temp_path)
    return temp_path


class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    """
    Input interface wrapper for ExtractNetStats
    """
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path = File(exists=True, mandatory=True)
    roi = traits.Any(mandatory=False)
    prune = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=False)
    binary = traits.Any(mandatory=False)


class ExtractNetStatsOutputSpec(TraitedSpec):
    """
    Output interface wrapper for ExtractNetStats
    """
    out_path_neat = File(exists=True, mandatory=True)


class ExtractNetStats(BaseInterface):
    """
    Interface wrapper for ExtractNetStats
    """
    input_spec = ExtractNetStatsInputSpec
    output_spec = ExtractNetStatsOutputSpec

    def _run_interface(self, runtime):
        out = extractnetstats(
            self.inputs.ID,
            self.inputs.network,
            self.inputs.thr,
            self.inputs.conn_model,
            self.inputs.est_path,
            self.inputs.roi,
            self.inputs.prune,
            self.inputs.norm,
            self.inputs.binary)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_path_neat': op.abspath(getattr(self, '_outpath'))}


class CombinePandasDfsInputSpec(BaseInterfaceInputSpec):
    """
    Input interface wrapper for CombinePandasDfs
    """
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    net_mets_csv_list = traits.List(mandatory=True)
    plot_switch = traits.Any(mandatory=True)
    multi_nets = traits.Any(mandatory=True)
    multimodal = traits.Any(mandatory=True)


class CombinePandasDfsOutputSpec(TraitedSpec):
    """
    Output interface wrapper for CombinePandasDfs
    """
    combination_complete = traits.Bool()


class CombinePandasDfs(SimpleInterface):
    """
    Interface wrapper for CombinePandasDfs
    """
    input_spec = CombinePandasDfsInputSpec
    output_spec = CombinePandasDfsOutputSpec

    def _run_interface(self, runtime):
        combination_complete = collect_pandas_df(
            self.inputs.network,
            self.inputs.ID,
            self.inputs.net_mets_csv_list,
            self.inputs.plot_switch,
            self.inputs.multi_nets,
            self.inputs.multimodal)
        setattr(self, '_combination_complete', combination_complete)
        return runtime

    def _list_outputs(self):
        return {'combination_complete': getattr(self, '_combination_complete')}
