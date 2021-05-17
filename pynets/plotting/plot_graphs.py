#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import tkinter
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("agg")


def plot_conn_mat(conn_matrix, labels, out_path_fig, cmap, binarized=False,
                  dpi_resolution=300):
    """
    Plot a connectivity matrix.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    labels : list
        List of string labels corresponding to ROI nodes.
    out_path_fig : str
        File path to save the connectivity matrix image as a .png figure.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use('Agg')
    import mplcyberpunk
    from matplotlib import pyplot as plt
    plt.style.use("cyberpunk")
    from matplotlib import pyplot as plt
    from nilearn.plotting import plot_matrix
    from pynets.core import thresholding
    import matplotlib.ticker as mticker

    conn_matrix = thresholding.standardize(conn_matrix)
    conn_matrix_bin = thresholding.binarize(conn_matrix)
    conn_matrix_plt = np.nan_to_num(np.multiply(conn_matrix, conn_matrix_bin))

    try:
        plot_matrix(
            conn_matrix_plt,
            figure=(10, 10),
            labels=labels,
            vmax=np.percentile(conn_matrix_plt[conn_matrix_plt > 0], 95),
            vmin=0,
            reorder="average",
            auto_fit=True,
            grid=False,
            colorbar=False,
            cmap=cmap,
        )
    except RuntimeWarning:
        print("Connectivity matrix too sparse for plotting...")

    if len(labels) > 500:
        tick_interval = 5
    elif len(labels) > 100:
        tick_interval = 4
    elif len(labels) > 50:
        tick_interval = 2
    else:
        tick_interval = 1

    plt.axes().yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    plt.axes().xaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#000000'
    plt.savefig(out_path_fig, dpi=dpi_resolution)
    plt.close()
    return


def plot_community_conn_mat(
        conn_matrix,
        labels,
        out_path_fig_comm,
        community_aff,
        cmap,
        dpi_resolution=300):
    """
    Plot a community-parcellated connectivity matrix.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    labels : list
        List of string labels corresponding to ROI nodes.
    out_path_fig_comm : str
        File path to save the community-parcellated connectivity matrix image
        as a .png figure.
    community_aff : array
        Community-affiliation vector.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use("agg")
    import mplcyberpunk
    plt.style.use("cyberpunk")
    import matplotlib.patches as patches
    import matplotlib.ticker as mticker
    from nilearn.plotting import plot_matrix
    from pynets.core import thresholding

    plt.style.use("cyberpunk")

    conn_matrix_bin = thresholding.binarize(conn_matrix)
    conn_matrix = thresholding.standardize(conn_matrix)
    conn_matrix_plt = np.nan_to_num(np.multiply(conn_matrix, conn_matrix_bin))

    sorting_array = sorted(
        range(len(community_aff)),
        key=lambda k: community_aff[k])
    sorted_conn_matrix = conn_matrix[sorting_array, :]
    sorted_conn_matrix = sorted_conn_matrix[:, sorting_array]
    rois_num = sorted_conn_matrix.shape[0]
    if rois_num < 100:
        try:
            plot_matrix(
                conn_matrix_plt,
                figure=(10, 10),
                labels=labels,
                vmax=np.percentile(conn_matrix_plt[conn_matrix_plt > 0], 95),
                vmin=0,
                reorder=False,
                auto_fit=True,
                grid=False,
                colorbar=False,
                cmap=cmap,
            )
        except RuntimeWarning:
            print("Connectivity matrix too sparse for plotting...")
    else:
        try:
            plot_matrix(
                conn_matrix_plt,
                figure=(10, 10),
                vmax=np.abs(np.max(conn_matrix_plt)),
                vmin=0,
                auto_fit=True,
                grid=False,
                colorbar=False,
                cmap=cmap,
            )
        except RuntimeWarning:
            print("Connectivity matrix too sparse for plotting...")

    ax = plt.gca()
    total_size = 0
    for community in np.unique(community_aff):
        size = sum(sorted(community_aff) == community)
        ax.add_patch(
            patches.Rectangle(
                (total_size, total_size),
                size,
                size,
                fill=False,
                edgecolor="white",
                alpha=None,
                linewidth=1,
            )
        )
        total_size += size

    if len(labels) > 500:
        tick_interval = 5
    elif len(labels) > 100:
        tick_interval = 4
    elif len(labels) > 50:
        tick_interval = 2
    else:
        tick_interval = 1

    plt.axes().yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    plt.axes().xaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#000000'
    plt.savefig(out_path_fig_comm, dpi=dpi_resolution)
    plt.close()
    return


def plot_conn_mat_func(
    conn_matrix,
    conn_model,
    atlas,
    dir_path,
    ID,
    network,
    labels,
    roi,
    thr,
    node_size,
    smooth,
    hpass,
    extract_strategy,
):
    """
    API for selecting among various functional connectivity matrix plotting
    approaches.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for partial
       correlation). sps type is used by default.
    atlas : str
        Name of atlas parcellation used.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g.
        'Default') used to filter nodes in the study of brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    extract_strategy : str
        The name of a valid function used to reduce the time-series region
        extraction.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pynets.core.utils import load_runconfig
    import sys
    import networkx as nx
    import os.path as op
    from pynets.plotting import plot_graphs

    out_path_fig = \
        "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % \
        (dir_path,
         "/adjacency_",
         ID,
         "_modality-func_",
         "%s" % ("%s%s%s" % ("rsn-",
                             network,
                             "_") if network is not None else ""),
         "%s" % ("%s%s%s" % ("roi-",
                             op.basename(roi).split(".")[0],
                             "_") if roi is not None else ""),
         "model-",
         conn_model,
         "_",
         "%s" % ("%s%s%s" % ("nodetype-spheres-",
                             node_size,
                             "mm_") if (
             (node_size != "parc") and (
                 node_size is not None)) else "nodetype-parc_"),
         "%s" % ("%s%s%s" % ("smooth-",
                             smooth,
                             "fwhm_") if float(smooth) > 0 else ""),
         "%s" % ("%s%s%s" % ("hpass-",
                             hpass,
                             "Hz_") if hpass is not None else ""),
         "%s" % ("%s%s%s" % ("extract-",
                             extract_strategy,
                             "") if extract_strategy is not None else ""),
         "_thr-",
         thr,
         ".png",
         )

    hardcoded_params = load_runconfig()
    try:
        cmap_name = hardcoded_params["plotting"]["functional"][
            "adjacency"]["color_theme"][0]
    except KeyError as e:
        print(e,
              "Plotting configuration not successfully extracted from"
              " runconfig.yaml"
              )

    plot_graphs.plot_conn_mat(
        conn_matrix, labels, out_path_fig, cmap=plt.get_cmap(cmap_name)
    )

    # Plot community adj. matrix
    try:
        from pynets.stats.netstats import community_resolution_selection

        G = nx.from_numpy_matrix(np.abs(conn_matrix))
        _, node_comm_aff_mat, resolution, num_comms = \
            community_resolution_selection(G)
        out_path_fig_comm = \
            "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % \
            (dir_path,
             "/adjacency-communities_",
             ID,
             "_modality-func_",
             "%s" % ("%s%s%s" % ("rsn-",
                                 network,
                                 "_") if network is not None else ""),
             "%s" % ("%s%s%s" % ("roi-",
                                 op.basename(roi).split(".")[0],
                                 "_") if roi is not None else ""),
             "model-",
             conn_model,
             "_",
             "%s" % ("%s%s%s" % ("nodetype-spheres-",
                                 node_size,
                                 "mm_") if (
                 (node_size != "parc") and (
                     node_size is not None)) else "nodetype-parc_"),
             "%s" % ("%s%s%s" % ("smooth-",
                                 smooth,
                                 "fwhm_") if float(smooth) > 0 else ""),
             "%s" % ("%s%s%s" % ("hpass-",
                                 hpass,
                                 "Hz_") if hpass is not None else ""),
             "%s" % ("%s%s%s" % ("extract-",
                                 extract_strategy,
                                 "") if extract_strategy is not None else ""),
             "_thr-",
             thr,
             ".png",
             )
        plot_graphs.plot_community_conn_mat(
            conn_matrix,
            labels,
            out_path_fig_comm,
            node_comm_aff_mat,
            cmap=plt.get_cmap(cmap_name),
        )
    except BaseException:
        print(
            "\nWARNING: Louvain community detection failed. Cannot plot "
            "community matrix..."
        )

    return


def plot_conn_mat_struct(
    conn_matrix,
    conn_model,
    atlas,
    dir_path,
    ID,
    network,
    labels,
    roi,
    thr,
    node_size,
    target_samples,
    track_type,
    directget,
    min_length,
    error_margin
):
    """
    API for selecting among various structural connectivity matrix plotting
    approaches.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for partial
       correlation). sps type is used by default.
    atlas : str
        Name of atlas parcellation used.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    directget : str
        The statistical approach to tracking. Options are:
        det (deterministic), closest (clos), boot (bootstrapped), and prob
        (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pynets.core.utils import load_runconfig
    import sys
    from pynets.plotting import plot_graphs
    import networkx as nx
    import os.path as op

    out_path_fig = \
        "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % \
        (dir_path,
         "/adjacency_",
         ID,
         "_modality-dwi_",
         "%s" % ("%s%s%s" % ("rsn-",
                             network,
                             "_") if network is not None else ""),
         "%s" % ("%s%s%s" % ("roi-",
                             op.basename(roi).split(".")[0],
                             "_") if roi is not None else ""),
         "model-",
         conn_model,
         "_",
         "%s" % ("%s%s%s" % ("nodetype-spheres-",
                             node_size,
                             "mm_") if (
             (node_size != "parc") and (
                 node_size is not None)) else "nodetype-parc_"),
         "%s" % ("%s%s%s" % ("samples-",
                             int(target_samples),
                             "streams_") if float(target_samples) > 0
                 else "_"),
         "tracktype-",
         track_type,
         "_directget-",
         directget,
         "_minlength-",
         min_length,
         "_tol-",
         error_margin,
         "_thr-",
         thr,
         ".png",
         )

    hardcoded_params = load_runconfig()
    try:
        cmap_name = hardcoded_params["plotting"]["structural"][
            "adjacency"]["color_theme"][0]
    except KeyError as e:
        print(e,
              "Plotting configuration not successfully extracted from"
              " runconfig.yaml"
              )

    plot_graphs.plot_conn_mat(
        conn_matrix, labels, out_path_fig, cmap=plt.get_cmap(cmap_name)
    )

    # Plot community adj. matrix
    try:
        from pynets.stats.netstats import community_resolution_selection

        G = nx.from_numpy_matrix(np.abs(conn_matrix))
        _, node_comm_aff_mat, resolution, num_comms = \
            community_resolution_selection(G)
        out_path_fig_comm = \
            "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" \
            % (dir_path,
               "/adjacency-communities_",
               ID,
               "_modality-dwi_",
               "%s" % ("%s%s%s" % ("rsn-",
                                   network,
                                   "_") if network is not None else ""),
               "%s" % ("%s%s%s" % ("roi-",
                                   op.basename(roi).split(".")[0],
                                   "_") if roi is not None else ""),
               "model-",
               conn_model,
               "_",
               "%s" % ("%s%s%s" % ("nodetype-spheres-",
                                   node_size,
                                   "mm_") if (
                   (node_size != "parc") and (
                       node_size is not None)) else "nodetype-parc_"),
               "%s" % ("%s%s%s" % ("samples-",
                                   int(target_samples),
                                   "streams_") if float(target_samples) > 0
                       else "_"),
               "tracktype-",
               track_type,
               "_directget-",
               directget,
               "_minlength-",
               min_length,
               "_tol-",
               error_margin,
               "_thr-",
               thr,
               ".png",
               )
        plot_graphs.plot_community_conn_mat(
            conn_matrix,
            labels,
            out_path_fig_comm,
            node_comm_aff_mat,
            cmap=plt.get_cmap(cmap_name),
        )
    except BaseException:
        print(
            "\nWARNING: Louvain community detection failed. Cannot plot"
            " community matrix..."
        )

    return
