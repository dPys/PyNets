#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import nibabel as nib
import networkx as nx
import os.path as op
import tkinter
import matplotlib
import matplotlib.pyplot as plt
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
        print(f"{'Found '}{str(len(link_comm_aff_mat))}{' communities...'}")
        clust_levels = len(link_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([link_comm_aff_mat == 0]).astype('int'))
        label_arr = link_comm_aff_mat * np.expand_dims(np.arange(1, clust_levels+1), axis=1) + mask_mat
    else:
        return

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
        json_file_name = f"{str(ID)}{'_'}{network}{'_connectogram_'}{conn_model}{'_network.json'}"
        json_fdg_file_name = f"{str(ID)}{'_'}{network}{'_fdg_'}{conn_model}{'_network.json'}"
        connectogram_plot = f"{dir_path}{'/'}{json_file_name}"
        fdg_js_sub = f"{dir_path}{'/'}{str(ID)}{'_'}{network}{'_fdg_'}{conn_model}{'_network.js'}"
        fdg_js_sub_name = f"{str(ID)}{'_'}{network}{'_fdg_'}{conn_model}{'_network.js'}"
        connectogram_js_sub = f"{dir_path}/{str(ID)}_{network}_connectogram_{conn_model}_network.js"
        connectogram_js_name = f"{str(ID)}{'_'}{network}{'_connectogram_'}{conn_model}{'_network.js'}"
    else:
        json_file_name = f"{str(ID)}{'_connectogram_'}{conn_model}{'.json'}"
        json_fdg_file_name = f"{str(ID)}{'_fdg_'}{conn_model}{'.json'}"
        connectogram_plot = f"{dir_path}{'/'}{json_file_name}"
        connectogram_js_sub = f"{dir_path}{'/'}{str(ID)}{'_connectogram_'}{conn_model}{'.js'}"
        fdg_js_sub = f"{dir_path}{'/'}{str(ID)}{'_fdg_'}{conn_model}{'.js'}"
        fdg_js_sub_name = f"{str(ID)}{'_fdg_'}{conn_model}{'.js'}"
        connectogram_js_name = f"{str(ID)}{'_connectogram_'}{conn_model}{'.js'}"
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

    out_file = f"{dir_path}{'/'}{str(json_fdg_file_name)}"
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
        plt.title(f"{network}{' Time Series'}")
        out_path_fig = f"{dir_path}{'/'}{ID}{'_'}{network}{'rsn_ts_plot.png'}"
    else:
        plt.title('Time Series')
        out_path_fig = f"{dir_path}{'/'}{ID}{'_wb_ts_plot.png'}"
    plt.savefig(out_path_fig)
    plt.close('all')
    return


def plot_network_clusters(graph, communities, out_path, figsize=(8, 8), node_size=50, plot_overlaps=False,
                          plot_labels=False):
    """
    Plot a graph with node color coding for communities.

    Parameters
    ----------
    graph : NetworkX graph
    communities : array
        Community affiliation vector
    out_path : str
        Path to save figure.
    figsize : Tuple of integers
        The figure size; it is a pair of float, default (8, 8).
    node_size: int
        Default 200.
    plot_overlaps : bool
        Flag to control if multiple algorithms memberships are plotted. Default is False.
    plot_labels : bool
        Flag to control if node labels are plotted. Default is False.
    """

    COLOR = ['r', 'b', 'g', 'c', 'm', 'y', 'k',
             '0.8', '0.2', '0.6', '0.4', '0.7', '0.3', '0.9', '0.1', '0.5']

    def getIndexPositions(listOfElements, element):
        ''' Returns the indexes of all occurrences of give element in
        the list- listOfElements '''
        indexPosList = []
        indexPos = 0
        while True:
            try:
                indexPos = listOfElements.index(element, indexPos)
                indexPosList.append(indexPos)
                indexPos += 1
            except ValueError as e:
                break

        return indexPosList

    partition = [getIndexPositions(communities.tolist(), i) for i in set(communities.tolist())]

    n_communities = min(len(partition), len(COLOR))
    plt.figure(figsize=figsize)
    plt.axis('off')

    position = nx.fruchterman_reingold_layout(graph)

    fig = nx.draw_networkx_nodes(graph, position, node_size=node_size, node_color='w')
    fig.set_edgecolor('k')
    nx.draw_networkx_edges(graph, position, alpha=.5)
    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            fig = nx.draw_networkx_nodes(graph, position, node_size=size,
                                         nodelist=partition[i], node_color=COLOR[i])
            fig.set_edgecolor('k')
    if plot_labels:
        nx.draw_networkx_labels(graph, position, labels={node: str(node) for node in graph.nodes()})

    fig.savefig(out_path)
    fig.close('all')

    return


def plot_all_func(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, coords, thr,
                  node_size, edge_threshold, smooth, prune, uatlas, norm, binary, hpass):
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

    if not isinstance(coords, list):
        coords = list(coords)

    if not isinstance(labels, list):
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
                                       node_size, smooth, hpass)

        # Plot connectome
        out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-func_',
                                                           '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                                   network is not None else ''),
                                                           '%s' % ("%s%s%s" % ('roi-', op.basename(roi).split('.')[0],
                                                                               '_') if roi is not None else ''),
                                                           'est-', conn_model, '_',
                                                           '%s' % (
                                                               "%s%s%s" % ('nodetype-spheres-', node_size, 'mm_') if
                                                               ((node_size != 'parc') and (node_size is not None))
                                                               else 'nodetype-parc_'),
                                                           "%s" % ("%s%s%s" % ('smooth-', smooth, 'fwhm_') if
                                                                   float(smooth) > 0 else ''),
                                                           "%s" % ("%s%s%s" % ('hpass-', hpass, 'Hz_') if
                                                                   hpass is not None else ''),
                                                           '_thr-', thr, '_glass_viz.png')
        if roi:
            # Save coords to pickle
            coord_path = f"{namer_dir}{'/coords_'}{op.basename(roi).split('.')[0]}{'_plotting.pkl'}"
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = f"{namer_dir}{'/labelnames_'}{op.basename(roi).split('.')[0]}{'_plotting.pkl'}"
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)

        else:
            # Save coords to pickle
            coord_path = f"{namer_dir}{'/coords_plotting.pkl'}"
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = f"{namer_dir}{'/labelnames_plotting.pkl'}"
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
                                 node_color=node_color, edge_kwargs={'alpha': 0.45})
            connectome.savefig(out_path_fig, dpi=dpi_resolution)
    else:
        raise RuntimeError('\nERROR: no coordinates to plot! Are you running plotting outside of pynets\'s internal '
                           'estimation schemes?')

    plt.close('all')

    return


def plot_all_struct(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, coords, thr,
                    node_size, edge_threshold, prune, uatlas, target_samples, norm, binary, track_type, directget,
                    min_length):
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
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
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

    if not isinstance(coords, list):
        coords = list(coords)

    if not isinstance(labels, list):
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
                                         node_size, target_samples, track_type, directget, min_length)

        # Plot connectome
        out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/', ID, '_modality-dwi_',
                                                                     '%s' % ("%s%s%s" % ('rsn-', network, '_') if
                                                                             network is not None else ''),
                                                                     '%s' % ("%s%s%s" % ('roi-',
                                                                                         op.basename(roi).split(
                                                                                             '.')[0],
                                                                                         '_') if roi is not
                                                                                                 None else ''),
                                                                     'est-', conn_model, '_',
                                                                     '%s' % (
                                                                         "%s%s%s" % ('nodetype-spheres-', node_size,
                                                                                     'mm_')
                                                                         if ((node_size != 'parc') and
                                                                             (node_size is not None))
                                                                         else 'nodetype-parc_'),
                                                                     "%s" % ("%s%s%s" % (
                                                                         'samples-', int(target_samples),
                                                                         'streams_')
                                                                             if float(target_samples) > 0 else '_'),
                                                                     'tt-', track_type, '_dg-', directget,
                                                                     '_ml-', min_length,
                                                                     '_thr-', thr, '_glass_viz.png')
        if roi:
            # Save coords to pickle
            coord_path = f"{namer_dir}{'/coords_'}{op.basename(roi).split('.')[0]}{'_plotting.pkl'}"
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = f"{namer_dir}{'/labelnames_'}{op.basename(roi).split('.')[0]}{'_plotting.pkl'}"
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f, protocol=2)
        else:
            # Save coords to pickle
            coord_path = f"{namer_dir}{'/coords_plotting.pkl'}"
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)

            # Save labels to pickle
            labels_path = f"{namer_dir}{'/labelnames_plotting.pkl'}"
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
                                 edge_vmax=float(z_max), edge_vmin=0, node_size=node_size_plot,
                                 node_color=clust_colors, edge_kwargs={'alpha': 0.10})
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


def pad_im(image, max_dim, pad_val=255, rgb=False):
    """
    Pads an image to be same dimensions as given max_dim

    Parameters
    -----------
    image: np array
        image object can be multiple dimensional or a slice.
    max_dim: int
        dimension to pad up to
    pad_val: int
        value to pad with. default is 255 (white) background
    rgb: boolean
        flag to indicate if RGB and last dimension should not be padded
    Returns
    -----------
    padded_image: np array
        image with padding
    """
    pad_width = []
    for i in range(image.ndim):
        pad_width.append(((max_dim - image.shape[i]) // 2, (max_dim - image.shape[i]) // 2))
    if rgb:
        pad_width[-1] = (0, 0)

    pad_width = tuple(pad_width)
    padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=pad_val)

    return padded_image


def qa_fast_png(csf, gm, wm, outdir):
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from matplotlib.colors import LinearSegmentedColormap
    """
    FAST (FMRIB's Automated Segmentation Tool)
    segments a 3D image of the brain into different tissue types (Grey Matter, White Matter, CSF, etc.)
    Mark different colors of white matter, gray matter, cerebrospinal fluid in a '3 by 3' picture, i.e. QA for FAST

    Parameters
    ---------------
    csf: str
    the path of csf nifti image
    gm: str
    the path of gm nifti image
    wm: str
    the path of wm nifti image
    outdir: str
    the path to save QA graph
    """

    # load data
    gm_data = nib.load(gm).get_data()
    csf_data = nib.load(csf).get_data()
    wm_data = nib.load(wm).get_data()

    # set Color map
    cmap1 = LinearSegmentedColormap.from_list('mycmap1', ['white', 'blue'])
    cmap2 = LinearSegmentedColormap.from_list('mycmap2', ['white', 'magenta'])
    cmap3 = LinearSegmentedColormap.from_list('mycmap2', ['white', 'green'])

    overlay = plt.figure()
    overlay.set_size_inches(12.5, 10.5, forward=True)
    plt.title(
        f'Qa for FAST(segments a 3D image of the brain into different tissue types)\n (scan volume:{gm_data.shape}) \n',
        fontsize=22)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # Set the 3D matrix cutting position in three directions
    shape = csf_data.shape
    index = [0.35, 0.51, 0.65]
    x = [int(shape[0] * index[0]), int(shape[0] * index[1]), int(shape[0] * index[2])]
    y = [int(shape[1] * index[0]), int(shape[1] * index[1]), int(shape[1] * index[2])]
    z = [int(shape[2] * index[0]), int(shape[2] * index[1]), int(shape[2] * index[2])]
    coords = (x, y, z)

    # Set labels for the y-axis
    labs = [
        "Sagittal Slice",
        "Coronal Slice",
        "Axial Slice",
    ]

    var = ["X", "Y", "Z"]

    # Generate 3 by 3 picture
    idx = 0
    for i, coord in enumerate(coords):
        for pos in coord:
            idx += 1
            ax = overlay.add_subplot(3, 3, idx)
            ax.set_title(var[i] + " = " + str(pos))
            if i == 0:
                csf_slice = ndimage.rotate(csf_data[pos, :, :], 90)
                gm_slice = ndimage.rotate(gm_data[pos, :, :], 90)
                wm_slice = ndimage.rotate(wm_data[pos, :, :], 90)
            elif i == 1:
                csf_slice = ndimage.rotate(csf_data[:, pos, :], 90)
                gm_slice = ndimage.rotate(gm_data[:, pos, :], 90)
                wm_slice = ndimage.rotate(wm_data[:, pos, :], 90)
            else:
                csf_slice = ndimage.rotate(csf_data[:, :, pos], 90)
                gm_slice = ndimage.rotate(gm_data[:, :, pos], 90)
                wm_slice = ndimage.rotate(wm_data[:, :, pos], 90)

            # set y labels
            if idx % 3 == 1:
                plt.ylabel(labs[i])

            #  padding pictures to make them the same size
            csf_slice = (csf_slice * 255).astype(np.uint8)
            gm_slice = (gm_slice * 255).astype(np.uint8)
            wm_slice = (wm_slice * 255).astype(np.uint8)
            csf_slice = pad_im(csf_slice, max(shape), 0, False)
            gm_slice = pad_im(gm_slice, max(shape), 0, False)
            wm_slice = pad_im(wm_slice, max(shape), 0, False)

            # hide axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # display image
            ax.imshow(csf_slice, interpolation="none", cmap=cmap1, alpha=1)
            ax.imshow(gm_slice, interpolation="none", cmap=cmap2, alpha=0.5)
            ax.imshow(wm_slice, interpolation="none", cmap=cmap3, alpha=0.3)

            # Legend of white matter(WM), gray matter(GM) and cerebrospinal fluid(csf)
            if idx == 3:
                plt.plot(0, 0, "-", c='green', label='wm')
                plt.plot(0, 0, "-", c='pink', label='gm')
                plt.plot(0, 0, "-", c='blue', label='csf')
                plt.legend(loc='upper right', fontsize=15, bbox_to_anchor=(1.5, 1.2))

    # save figure
    overlay.savefig(f"{outdir}", format="png")
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
    from sklearn.preprocessing import scale
    print('Saving model plots...')

    namer_dir = op.dirname(op.dirname(op.dirname(net_pick_file)))
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    def nearest_square_root(limit):
        answer = 0
        while (answer + 1) ** 2 < limit:
            answer += 1
        return int(np.sqrt(answer ** 2))

    global_measures = [meas for meas in measures if not meas.split('_')[0].isdigit() and
                       meas.endswith('_auc') and not meas.startswith('thr_')]

    if len(df_concat) >= 30:
        fig, axes = plt.subplots(ncols=nearest_square_root(len(global_measures)),
                                 nrows=nearest_square_root(len(global_measures)),
                                 sharex=True, sharey=True, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            try:
                x = np.array(df_concat[global_measures[i]][np.isfinite(df_concat[global_measures[i]])])
            except:
                continue
            try:
                x = np.delete(x, np.argwhere(x == '')).astype('float')
            except:
                continue
            try:
                x = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
            except:
                continue
            if True in pd.isnull(x):
                try:
                    x = x[~pd.isnull(x)]
                    if len(x) > 0:
                        print(f"NaNs encountered for {global_measures[i]}. Plotting and averaging across non-missing "
                              f"values. Checking output is recommended...")
                        ax.hist(x, density=True, bins='auto', alpha=0.8)
                        ax.set_title(global_measures[i])
                    else:
                        print(f"{'Warning: No numeric data to plot for '}{global_measures[i]}")
                        continue
                except:
                    continue
            else:
                try:
                    ax.hist(x, density=True, bins='auto', alpha=0.8)
                    ax.set_title(global_measures[i])
                except:
                    print(f"Warning: Inf or NaN values encounterd. No numeric data to plot for {global_measures[i]}")
                    continue

        plt.tight_layout()
        out_path_fig = f"{namer_dir}{'/mean_global_topology_distribution_multiplot.png'}"
        fig.savefig(out_path_fig)
        plt.close('all')
    else:
        print('At least 30 iterations needed to produce multiplot of global graph topology distributions. '
              'Continuing...')
        pass

    return
