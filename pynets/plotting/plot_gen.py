#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import networkx as nx
import os.path as op
import tkinter
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.use('agg')


def plot_connectogram(conn_matrix, conn_model, atlas, dir_path, ID, network, labels):
    """
    Plot a connectogram for a given connectivity matrix.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    atlas : str
        Name of atlas parcellation used.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    """
    import json
    from pathlib import Path
    from networkx.readwrite import json_graph
    from pynets.core.thresholding import normalize
    from pynets.stats.netstats import most_important
    # from scipy.cluster.hierarchy import linkage, fcluster
    from nipype.utils.filemanip import save_json

    # Advanced Settings
    comm = 'nodes'
    pruned = False
    #color_scheme = 'interpolateCool'
    #color_scheme = 'interpolateGnBu'
    #color_scheme = 'interpolateOrRd'
    #color_scheme = 'interpolatePuRd'
    #color_scheme = 'interpolateYlOrRd'
    #color_scheme = 'interpolateReds'
    #color_scheme = 'interpolateGreens'
    color_scheme = 'interpolateBlues'
    # Advanced Settings

    conn_matrix = normalize(conn_matrix)
    G = nx.from_numpy_matrix(np.abs(conn_matrix))
    if pruned is True:
        [G, pruned_nodes] = most_important(G)
        conn_matrix = nx.to_numpy_array(G)

        pruned_nodes.sort(reverse=True)
        for j in pruned_nodes:
            del labels[labels.index(labels[j])]

    # def _doClust(X, clust_levels):
    #     """
    #     Create Ward cluster linkages.
    #     """
    #     # get the linkage diagram
    #     Z = linkage(X, 'ward')
    #     # choose # cluster levels
    #     cluster_levels = range(1, int(clust_levels))
    #     # init array to store labels for each level
    #     clust_levels_tmp = int(clust_levels) - 1
    #     label_arr = np.zeros((int(clust_levels_tmp), int(X.shape[0])))
    #     # iterate thru levels
    #     for c in cluster_levels:
    #         fl = fcluster(Z, c, criterion='maxclust')
    #         #print(fl)
    #         label_arr[c-1, :] = fl
    #     return label_arr, clust_levels_tmp

    if comm == 'nodes' and len(conn_matrix) > 40:
        from pynets.stats.netstats import community_resolution_selection
        G = nx.from_numpy_matrix(np.abs(conn_matrix))
        _, node_comm_aff_mat, resolution, num_comms = community_resolution_selection(G)
        clust_levels = len(node_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([node_comm_aff_mat == 0]).astype('int'))
        label_arr = node_comm_aff_mat * np.expand_dims(np.arange(1, clust_levels+1), axis=1) + mask_mat
    elif comm == 'links' and len(conn_matrix) > 40:
        from pynets.stats.netstats import link_communities
        # Plot link communities
        link_comm_aff_mat = link_communities(conn_matrix, type_clustering='single')
        print("%s%s%s" % ('Found ', str(len(link_comm_aff_mat)), ' communities...'))
        clust_levels = len(link_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([link_comm_aff_mat == 0]).astype('int'))
        label_arr = link_comm_aff_mat * np.expand_dims(np.arange(1, clust_levels+1), axis=1) + mask_mat
    else:
        return
    # elif len(conn_matrix) > 20:
    #     print('Graph too small for reliable plotting of communities. Plotting by fcluster instead...')
    #     if len(conn_matrix) >= 250:
    #         clust_levels = 7
    #     elif len(conn_matrix) >= 200:
    #         clust_levels = 6
    #     elif len(conn_matrix) >= 150:
    #         clust_levels = 5
    #     elif len(conn_matrix) >= 100:
    #         clust_levels = 4
    #     elif len(conn_matrix) >= 50:
    #         clust_levels = 3
    #     else:
    #         clust_levels = 2
    #     [label_arr, clust_levels_tmp] = _doClust(conn_matrix, clust_levels)

    def _get_node_label(node_idx, labels, clust_levels_tmp):
        """
        Tag a label to a given node based on its community/cluster assignment
        """
        from collections import OrderedDict

        def _write_roman(num):
            """
            Create community/cluster assignments using a Roman-Numeral generator.
            """
            roman = OrderedDict()
            roman[1000] = "M"
            roman[900] = "CM"
            roman[500] = "D"
            roman[400] = "CD"
            roman[100] = "C"
            roman[90] = "XC"
            roman[50] = "L"
            roman[40] = "XL"
            roman[10] = "X"
            roman[9] = "IX"
            roman[5] = "V"
            roman[4] = "IV"
            roman[1] = "I"

            def roman_num(num):
                """

                :param num:
                """
                for r in roman.keys():
                    x, y = divmod(num, r)
                    yield roman[r] * x
                    num -= (r * x)
                    if num > 0:
                        roman_num(num)
                    else:
                        break
            return "".join([a for a in roman_num(num)])
        rn_list = []
        node_idx = node_idx - 1
        node_labels = labels[:, node_idx]
        for k in [int(l) for i, l in enumerate(node_labels)]:
            rn_list.append(json.dumps(_write_roman(k)))
        abet = rn_list
        node_lab_alph = ".".join(["{}{}".format(abet[i], int(l)) for i, l in enumerate(node_labels)]) + ".{}".format(
            labels[node_idx])
        return node_lab_alph

    output = []

    adj_dict = {}
    for i in list(G.adjacency()):
        source = list(i)[0]
        target = list(list(i)[1])
        adj_dict[source] = target

    for node_idx, connections in adj_dict.items():
        weight_vec = []
        for i in connections:
            wei = G.get_edge_data(node_idx,int(i))['weight']
            weight_vec.append(wei)
        entry = {}
        nodes_label = _get_node_label(node_idx, label_arr, clust_levels_tmp)
        entry["name"] = nodes_label
        entry["size"] = len(connections)
        entry["imports"] = [_get_node_label(int(d)-1, label_arr, clust_levels_tmp) for d in connections]
        entry["weights"] = weight_vec
        output.append(entry)

    if network:
        json_file_name = "%s%s%s%s%s%s" % (str(ID), '_', network, '_connectogram_', conn_model, '_network.json')
        json_fdg_file_name = "%s%s%s%s%s%s" % (str(ID), '_', network, '_fdg_', conn_model, '_network.json')
        connectogram_plot = "%s%s%s" % (dir_path, '/', json_file_name)
        fdg_js_sub = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_fdg_', conn_model, '_network.js')
        fdg_js_sub_name = "%s%s%s%s%s%s" % (str(ID), '_', network, '_fdg_', conn_model, '_network.js')
        connectogram_js_sub = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_connectogram_', conn_model,
                                                    '_network.js')
        connectogram_js_name = "%s%s%s%s%s%s" % (str(ID), '_', network, '_connectogram_', conn_model, '_network.js')
    else:
        json_file_name = "%s%s%s%s" % (str(ID), '_connectogram_', conn_model, '.json')
        json_fdg_file_name = "%s%s%s%s" % (str(ID), '_fdg_', conn_model, '.json')
        connectogram_plot = "%s%s%s" % (dir_path, '/', json_file_name)
        connectogram_js_sub = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_connectogram_', conn_model, '.js')
        fdg_js_sub = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_fdg_', conn_model, '.js')
        fdg_js_sub_name = "%s%s%s%s" % (str(ID), '_fdg_', conn_model, '.js')
        connectogram_js_name = "%s%s%s%s" % (str(ID), '_connectogram_', conn_model, '.js')
    save_json(connectogram_plot, output)

    # Force-directed graphing
    G = nx.from_numpy_matrix(np.round(np.abs(conn_matrix).astype('float64'), 6))
    data = json_graph.node_link_data(G)
    data.pop('directed', None)
    data.pop('graph', None)
    data.pop('multigraph', None)
    for k in range(len(data['links'])):
        data['links'][k]['value'] = data['links'][k].pop('weight')
    for k in range(len(data['nodes'])):
        data['nodes'][k]['id'] = str(data['nodes'][k]['id'])
    for k in range(len(data['links'])):
        data['links'][k]['source'] = str(data['links'][k]['source'])
        data['links'][k]['target'] = str(data['links'][k]['target'])

    # Add community structure
    for k in range(len(data['nodes'])):
        data['nodes'][k]['group'] = str(label_arr[0][k])

    # Add node labels
    for k in range(len(data['nodes'])):
        data['nodes'][k]['name'] = str(labels[k])

    out_file = "%s%s%s" % (dir_path, '/', str(json_fdg_file_name))
    save_json(out_file, data)

    # Copy index.html and json to dir_path
    conn_js_path = str(Path(__file__).parent/"connectogram.js")
    index_html_path = str(Path(__file__).parent/"index.html")
    fdg_replacements_js = {"FD_graph.json": str(json_fdg_file_name)}
    replacements_html = {'connectogram.js': str(connectogram_js_name), 'fdg.js': str(fdg_js_sub_name)}
    fdg_js_path = str(Path(__file__).parent/"fdg.js")
    with open(index_html_path) as infile, open(str(dir_path + '/index.html'), 'w') as outfile:
        for line in infile:
            for src, target in replacements_html.items():
                line = line.replace(src, target)
            outfile.write(line)

    replacements_js = {'template.json': str(json_file_name), 'interpolateCool': str(color_scheme)}
    with open(conn_js_path) as infile, open(connectogram_js_sub, 'w') as outfile:
        for line in infile:
            for src, target in replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

    with open(fdg_js_path) as infile, open(fdg_js_sub, 'w') as outfile:
        for line in infile:
            for src, target in fdg_replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

    return


def plot_timeseries(time_series, network, ID, dir_path, atlas, labels):
    """
    Plot time-series.

    Parameters
    ----------
    time-series : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    atlas : str
        Name of atlas parcellation used.
    labels : list
        List of string labels corresponding to ROI nodes.
    """
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt

    for time_serie, label in zip(time_series.T, labels):
        plt.plot(time_serie, label=label)
    plt.xlabel('Scan Number')
    plt.ylabel('Normalized Signal')
    plt.legend()
    #plt.tight_layout()
    if network:
        plt.title("%s%s" % (network, ' Time Series'))
        out_path_fig = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, 'rsn_ts_plot.png')
    else:
        plt.title('Time Series')
        out_path_fig = "%s%s%s%s" % (dir_path, '/', ID, '_wb_ts_plot.png')
    plt.savefig(out_path_fig)
    plt.close('all')
    return


def plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, coords, thr,
                  node_size, edge_threshold, smooth, prune, uatlas, c_boot, norm, binary, hpass):
    """
    Plot adjacency matrix, connectogram, and glass brain for functional connectome.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    atlas : str
        Name of atlas parcellation used.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set (e.g. a coordinate atlas).
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    edge_threshold : float
        The actual value, between 0 and 1, that the graph was thresholded (can differ from thr if target was not
        successfully obtained.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    import os
    import os.path as op
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    import pkg_resources
    import networkx as nx
    from pynets.core import thresholding
    from pynets.plotting import plot_gen, plot_graphs
    from pynets.stats.netstats import most_important, prune_disconnected
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")

    coords = list(coords)
    labels = list(labels)
    if len(coords) > 0:
        dpi_resolution = 500
        if '\'b' in atlas:
            atlas = atlas.decode('utf-8')
        if (prune == 1 or prune == 2) and len(coords) == conn_matrix.shape[0]:
            G_pre = nx.from_numpy_matrix(np.abs(conn_matrix))
            if prune == 1:
                [G, pruned_nodes] = prune_disconnected(G_pre)
            elif prune == 2:
                [G, pruned_nodes] = most_important(G_pre)
            else:
                G = G_pre
                pruned_nodes = []
            pruned_nodes.sort(reverse=True)
            print('(Display)')
            coords_pre = list(coords)
            labels_pre = list(labels)
            if len(pruned_nodes) > 0:
                for j in pruned_nodes:
                    labels_pre.pop(j)
                    coords_pre.pop(j)
                conn_matrix = nx.to_numpy_array(G)
                labels = labels_pre
                coords = coords_pre
            else:
                print('No nodes to prune for plot...')

        coords = list(tuple(x) for x in coords)

        namer_dir = dir_path + '/figures'
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        # Plot connectogram
        if len(conn_matrix) > 20:
            try:
                plot_gen.plot_connectogram(conn_matrix, conn_model, atlas, namer_dir, ID, network, labels)
            except RuntimeWarning:
                print('\n\n\nWarning: Connectogram plotting failed!')
        else:
            print('Warning: Cannot plot connectogram for graphs smaller than 20 x 20!')

        # Plot adj. matrix based on determined inputs
        if not node_size or node_size == 'None':
            node_size = 'parc'
        plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas, namer_dir, ID, network, labels, roi, thr,
                                       node_size, smooth, c_boot, hpass)

        # Plot connectome
        if roi:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_', atlas, '_', conn_model,
                                                                     '_', op.basename(roi).split('.')[0],
                                                                     "%s" % ("%s%s%s" % ('_', network, '_thr-') if
                                                                             network else "_thr-"),
                                                                     thr, '_', node_size,
                                                                     '%s' % ("mm_" if node_size != 'parc' else
                                                                             "_"),
                                                                     "%s" % ("%s%s" % (int(c_boot), 'nb_') if
                                                                             float(c_boot) > 0 else 'nb_'),
                                                                     "%s" % ("%s%s" % (smooth, 'fwhm_') if
                                                                             float(smooth) > 0 else ''),
                                                                     "%s" % ("%s%s" % (hpass, 'Hz_') if
                                                                             hpass is not None else ''),
                                                                     'func_glass_viz.png')

            # Save coords to pickle
            coord_path = "%s%s%s%s" % (namer_dir, '/coords_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = "%s%s%s%s" % (namer_dir, '/labelnames_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)

        else:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_', atlas, '_', conn_model,
                                                                 "%s" % ("%s%s%s" % ('_', network, '_thr-') if
                                                                         network else "_thr-"),
                                                                 thr, '_', node_size, '%s' % ("mm_" if
                                                                                              node_size != 'parc' else
                                                                                              "_"),
                                                                 "%s" % ("%s%s" % (int(c_boot), 'nb_') if
                                                                         float(c_boot) > 0 else 'nb_'),
                                                                 "%s" % ("%s%s" % (smooth, 'fwhm_') if
                                                                         float(smooth) > 0 else ''),
                                                                 "%s" % ("%s%s" % (hpass, 'Hz_') if hpass is not None
                                                                         else ''),
                                                                 'func_glass_viz.png')
            # Save coords to pickle
            coord_path = "%s%s" % (namer_dir, '/coords_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = "%s%s" % (namer_dir, '/labelnames_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)

        connectome = niplot.plot_connectome(np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001, black_bg=True)
        connectome.add_overlay(ch2better_loc, alpha=0.45, cmap=plt.cm.gray)
        #connectome.add_overlay(ch2better_loc, alpha=0.35, cmap=plt.cm.gray)
        conn_matrix = np.array(np.array(thresholding.autofix(conn_matrix)))
        [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
        if node_size == 'parc':
            node_size_plot = int(6)
        else:
            node_size_plot = int(node_size)
        if len(coords) != conn_matrix.shape[0]:
            raise RuntimeWarning('\nWARNING: Number of coordinates does not match conn_matrix dimensions. If you are '
                                 'using disparity filtering, try relaxing the α threshold.')
        else:
            color_theme = 'Blues'
            #color_theme = 'Greens'
            #color_theme = 'Reds'
            node_color = 'auto'
            connectome.add_graph(conn_matrix, coords, edge_threshold=edge_threshold, edge_cmap=color_theme,
                                 edge_vmax=float(z_max), edge_vmin=float(z_min), node_size=node_size_plot,
                                 node_color='auto')
            connectome.savefig(out_path_fig, dpi=dpi_resolution)
    else:
        raise RuntimeError('\nERROR: no coordinates to plot! Are you running plotting outside of pynets\'s internal '
                           'estimation schemes?')

    plt.close('all')

    return


def plot_all_struct(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, coords, thr,
                    node_size, edge_threshold, prune, uatlas, target_samples, norm, binary, track_type, directget):
    """
    Plot adjacency matrix, connectogram, and glass brain for functional connectome.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    atlas : str
        Name of atlas parcellation used.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set (e.g. a coordinate atlas).
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    edge_threshold : float
        The actual value, between 0 and 1, that the graph was thresholded (can differ from thr if target was not
        successfully obtained.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    """
    import matplotlib
    matplotlib.use('agg')
    import os
    import os.path as op
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    import pkg_resources
    import networkx as nx
    from matplotlib import colors
    import seaborn as sns
    from pynets.core import thresholding
    from pynets.plotting import plot_gen, plot_graphs
    from pynets.stats.netstats import most_important, prune_disconnected
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")

    coords = list(coords)
    labels = list(labels)
    if len(coords) > 0:
        dpi_resolution = 500
        if '\'b' in atlas:
            atlas = atlas.decode('utf-8')
        if (prune == 1 or prune == 2) and len(coords) == conn_matrix.shape[0]:
            G_pre = nx.from_numpy_matrix(np.abs(conn_matrix))
            if prune == 1:
                [G, pruned_nodes] = prune_disconnected(G_pre)
            elif prune == 2:
                [G, pruned_nodes] = most_important(G_pre)
            else:
                G = G_pre
                pruned_nodes = []
            pruned_nodes.sort(reverse=True)
            coords_pre = list(coords)
            labels_pre = list(labels)
            if len(pruned_nodes) > 0:
                for j in pruned_nodes:
                    labels_pre.pop(j)
                    coords_pre.pop(j)
                conn_matrix = nx.to_numpy_array(G)
                labels = labels_pre
                coords = coords_pre
            else:
                print('No nodes to prune for plot...')

        coords = list(tuple(x) for x in coords)

        namer_dir = dir_path + '/figures'
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        # Plot connectogram
        if len(conn_matrix) > 20:
            try:
                plot_gen.plot_connectogram(conn_matrix, conn_model, atlas, namer_dir, ID, network, labels)
            except RuntimeWarning:
                print('\n\n\nWarning: Connectogram plotting failed!')
        else:
            print('Warning: Cannot plot connectogram for graphs smaller than 20 x 20!')

        # Plot adj. matrix based on determined inputs
        if not node_size or node_size == 'None':
            node_size = 'parc'
        plot_graphs.plot_conn_mat_struct(conn_matrix, conn_model, atlas, namer_dir, ID, network, labels, roi, thr,
                                         node_size, target_samples, track_type, directget)

        # Plot connectome
        if roi:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_', atlas, '_', conn_model,
                                                                     '_', op.basename(roi).split('.')[0],
                                                                     "%s" % ("%s%s%s" % ('_', network, '_thr-') if
                                                                             network else "_thr-"), thr, '_', node_size,
                                                                     '%s' % ("mm_" if node_size != 'parc' else "_"),
                                                                     "%s" % ("%s%s" %
                                                                             (int(target_samples), '_samples') if
                                                                             float(target_samples) > 0 else ''),
                                                                     "%s%s%s" % ('_', track_type, '_track'),
                                                                     "%s%s" % ('_', directget),
                                                                     'struct_glass_viz.png')

            # Save coords to pickle
            coord_path = "%s%s%s%s" % (namer_dir, '/coords_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = "%s%s%s%s" % (namer_dir, '/labelnames_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)

        else:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_', atlas, '_', conn_model,
                                                                 "%s" % ("%s%s%s" % ('_', network, '_thr-') if
                                                                         network else "_thr-"),
                                                                 thr, '_', node_size, '%s' % ("mm_" if
                                                                                              node_size != 'parc' else
                                                                                              "_"),
                                                                 "%s" % ("%s%s" % (int(target_samples), '_samples') if
                                                                         float(target_samples) > 0 else ''),
                                                                 "%s%s%s" % ('_', track_type, '_track'),
                                                                 "%s%s" % ('_', directget),
                                                                 'struct_glass_viz.png')
            # Save coords to pickle
            coord_path = "%s%s" % (namer_dir, '/coords_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = "%s%s" % (namer_dir, '/labelnames_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)

        connectome = niplot.plot_connectome(np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001, black_bg=True)
        connectome.add_overlay(ch2better_loc, alpha=0.45, cmap=plt.cm.gray)
        #connectome.add_overlay(ch2better_loc, alpha=0.35, cmap=plt.cm.gray)
        conn_matrix = np.array(np.array(thresholding.autofix(conn_matrix)))
        [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
        if node_size == 'parc':
            node_size_plot = int(6)
        else:
            node_size_plot = int(node_size)
        if len(coords) != conn_matrix.shape[0]:
            raise RuntimeWarning('\nWARNING: Number of coordinates does not match conn_matrix dimensions.')
        else:
            norm = colors.Normalize(vmin=-1, vmax=1)
            clust_pal = sns.color_palette("Blues_r", conn_matrix.shape[0])
            clust_colors = colors.to_rgba_array(clust_pal)
            fa_path = dir_path + '/../reg_dmri/dmri_tmp/DSN/Warped.nii.gz'
            if os.path.isfile(fa_path):
                connectome.add_overlay(img=fa_path,
                                       threshold=0.01, alpha=0.25, cmap=plt.cm.copper)

            connectome.add_graph(conn_matrix, coords, edge_threshold=edge_threshold, edge_cmap=plt.cm.binary,
                                 edge_vmax=float(z_max), edge_vmin=float(z_min), node_size=node_size_plot,
                                 node_color=clust_colors)
            connectome.savefig(out_path_fig, dpi=dpi_resolution)
    else:
        raise RuntimeError('\nERROR: no coordinates to plot! Are you running plotting outside of pynets\'s internal '
                           'estimation schemes?')

    plt.close('all')

    return


def show_template_bundles(final_streamlines, template_path, fname):
    import nibabel as nib
    from fury import actor, window
    renderer = window.Renderer()
    template_img_data = nib.load(template_path).get_data().astype('bool')
    template_actor = actor.contour_from_roi(template_img_data,
                                            color=(50, 50, 50), opacity=0.05)
    renderer.add(template_actor)
    lines_actor = actor.streamtube(final_streamlines, window.colors.orange,
                                   linewidth=0.3)
    renderer.add(lines_actor)
    window.record(renderer, n_frames=1, out_path=fname, size=(900, 900))
    return


def plot_graph_measure_hists(df_concat, measures, net_pick_file):
    """
    Plot histograms for each graph theoretical measure.

    Parameters
    ----------
    df_concat : DataFrame
        Pandas dataframe of concatenated graph measures across ensemble.
    measures : list
        List of string names for graph measures whose order corresponds to headers/values in df_concat.
    net_pick_file : st
        File path to .pkl file of network measures used to generate df_concat.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    print('Saving model plots...')

    namer_dir = op.dirname(op.dirname(net_pick_file)) + '/figures'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    for name in measures:
        try:
            x = np.array(df_concat[name][np.isfinite(df_concat[name])])
        except:
            pass
        try:
            x = np.delete(x, np.argwhere(x == '')).astype('float')
        except:
            pass
        fig, ax = plt.subplots(tight_layout=True)
        if True in pd.isnull(x):
            x = x[~pd.isnull(x)]
            if len(x) > 0:
                print("%s%s%s" % ('NaNs encountered for ', name,
                                  '. Plotting and averaging across non-missing values. Checking output is '
                                  'recommended...'))
                ax.hist(x)
            else:
                print("%s%s" % ('Warning: No numeric data to plot for ', name))
                continue
        else:
            try:
                ax.hist(x)
            except:
                print("%s%s" % ('Warning: Inf or NaN values encounterd. No numeric data to plot for ', name))
                pass
        out_path_fig = "%s%s%s%s" % (namer_dir, '/', name, '_mean_plot.png')
        fig.savefig(out_path_fig)
        plt.close('all')
    return
