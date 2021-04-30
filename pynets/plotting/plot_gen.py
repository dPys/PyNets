#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import nibabel as nib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import networkx as nx
import os.path as op
import tkinter
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.use("agg")


def plot_timeseries(time_series, network, ID, dir_path, atlas, labels):
    """
    Plot time-series.

    Parameters
    ----------
    time-series : array
        2D m x n array consisting of the time-series signal for each ROI node
        where m = number of scans and n = number of ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g.
        'Default') used to filter nodes in the study of brain subgraphs.
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

    matplotlib.use("agg")
    from matplotlib import pyplot as plt

    for time_serie, label in zip(time_series.T, labels):
        plt.plot(time_serie, label=label)
    plt.xlabel("Scan Number")
    plt.ylabel("Normalized Signal")
    plt.legend()
    # plt.tight_layout()
    if network:
        plt.title(f"{network}{' Time Series'}")
        out_path_fig = f"{dir_path}/timseries_sub-{ID}_rsn-{network}.png"
    else:
        plt.title("Time Series")
        out_path_fig = f"{dir_path}/timseries_sub-{ID}.png"
    plt.savefig(out_path_fig)
    plt.close("all")
    return


def plot_network_clusters(
    graph,
    communities,
    out_path,
    figsize=(8, 8),
    node_size=50,
    plot_overlaps=False,
    plot_labels=False,
):
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
        Default 50.
    plot_overlaps : bool
        Flag to control if multiple algorithms memberships are plotted.
        Default is False.
    plot_labels : bool
        Flag to control if node labels are plotted. Default is False.

    """

    COLOR = [
        "r",
        "b",
        "g",
        "c",
        "m",
        "y",
        "k",
        "0.8",
        "0.2",
        "0.6",
        "0.4",
        "0.7",
        "0.3",
        "0.9",
        "0.1",
        "0.5",
    ]

    def getIndexPositions(listOfElements, element):
        """ Returns the indexes of all occurrences of give element in
        the list- listOfElements """
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

    partition = [
        getIndexPositions(
            communities.tolist(),
            i) for i in set(
            communities.tolist())]

    n_communities = min(len(partition), len(COLOR))
    fig = plt.figure(figsize=figsize)
    plt.axis("off")

    position = nx.fruchterman_reingold_layout(graph)

    nx.draw_networkx_nodes(
        graph, position, node_size=node_size, node_color="w", edgecolors="k"
    )
    nx.draw_networkx_edges(graph, position, alpha=0.5)

    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size

            nx.draw_networkx_nodes(
                graph,
                position,
                node_size=size,
                nodelist=partition[i],
                node_color=COLOR[i],
                edgecolors="k",
            )

    if plot_labels:
        nx.draw_networkx_labels(
            graph, position, labels={node: str(node) for node in graph.nodes()}
        )

    plt.savefig(out_path)
    plt.close("all")

    return


def create_gb_palette(
        mat,
        edge_cmap,
        coords,
        labels,
        node_size="auto",
        node_cmap=None,
        prune=True,
        centrality_type='eig',
        max_node_size=None,
        node_aff_mat=None):
    """
    Create connectome color palette based on graph topography.

    Parameters
    ----------
    mat : array
        NxN matrix.
    edge_cmap: colormap
        colormap used for representing the weight of the edges.
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set
        (e.g. a coordinate atlas).
    labels : list
        List of string labels corresponding to ROI nodes.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    node_size: scalar or array_like
        size(s) of the nodes in points^2.
    node_cmap: colormap
        colormap used for representing the community assignment of the nodes.

    """
    import matplotlib
    matplotlib.use('Agg')
    import warnings
    warnings.filterwarnings("ignore")
    import random
    import seaborn as sns
    import networkx as nx
    from pynets.core import thresholding
    import matplotlib.pyplot as plt
    import mplcyberpunk
    from matplotlib import colors
    from sklearn.preprocessing import minmax_scale
    from pynets.stats.netstats import community_resolution_selection, \
        prune_disconnected

    plt.style.use("cyberpunk")

    mat = np.array(np.array(thresholding.autofix(mat)))
    if prune is True:
        [G, pruned_nodes] = prune_disconnected(
            nx.from_numpy_matrix(np.abs(mat)), fallback_lcc=False)
        pruned_nodes.sort(reverse=True)
        coords_pre = list(coords)
        labels_pre = list(labels)
        if len(pruned_nodes) > 0:
            for j in pruned_nodes:
                del labels_pre[j], coords_pre[j]
            mat = nx.to_numpy_array(G)
            labels = labels_pre
            coords = coords_pre
        else:
            print("No nodes to prune for plotting...")
    else:
        G = nx.from_numpy_matrix(np.abs(mat))

    # Node centralities
    try:
        if centrality_type == 'eig':
            node_centralities = list(
                nx.algorithms.eigenvector_centrality_numpy(
                    G, weight="weight").values())
        elif centrality_type == 'bet':
            node_centralities = list(
                nx.algorithms.betweenness_centrality(
                    G, weight="weight").values())
        elif centrality_type == 'deg':
            node_centralities = list(
                nx.algorithms.degree_centrality(
                    G).values())
        elif isinstance(centrality_type, list):
            node_centralities = centrality_type

    except BaseException:
        node_centralities = len(coords) * [1]

    if not max_node_size:
        max_node_size = 1 / mat.shape[0] * \
            1e3 + 0.5 if node_size == "auto" else node_size

    node_sizes = np.array(
        minmax_scale(node_centralities, feature_range=(1, max_node_size))
    )

    # Node communities
    if node_aff_mat is None:
        _, node_aff_mat, _, num_comms = community_resolution_selection(G)
    else:
        num_comms = len(set(node_aff_mat))

    # Path lengths
    edge_lengths = []
    for edge_dict in [i[1] for i in nx.all_pairs_shortest_path_length(G)]:
        edge_lengths.extend(list(edge_dict.values()))

    edge_sizes = np.array(minmax_scale(edge_lengths, feature_range=(0.5, 2)))

    # Nodes
    if not node_cmap:
        # Generate as many randomly distinct colors as num_comms
        def random_color(n):
            ret = []
            r = int(random.random() * 256)
            g = int(random.random() * 256)
            b = int(random.random() * 256)
            step = 256 / n
            for i in range(n):
                r += step
                g += step
                b += step
                r = int(r) % 256
                g = int(g) % 256
                b = int(b) % 256
                ret.append((r, g, b))
            return ret

        flatui = [
            "#{:02x}{:02x}{:02x}".format(i[0], i[1], i[2])
            for i in random_color(num_comms)
        ]

        try:
            ls_cmap = colors.LinearSegmentedColormap.from_list(
                node_aff_mat, sns.color_palette(flatui,
                                                n_colors=num_comms)
            )
            matplotlib.cm.register_cmap("community", ls_cmap)
            clust_pal = sns.color_palette("community", n_colors=mat.shape[0])
        except BaseException:
            clust_pal = sns.color_palette("Set2", n_colors=mat.shape[0])
    else:
        clust_pal = sns.color_palette(node_cmap, n_colors=mat.shape[0])
    clust_pal_nodes = colors.to_rgba_array(clust_pal)

    # Edges
    z_max = np.max(mat)
    z_min = 0
    edge_cmap_pl = sns.color_palette(edge_cmap)
    clust_pal_edges = colors.ListedColormap(edge_cmap_pl.as_hex())

    return (
        mat,
        clust_pal_edges,
        clust_pal_nodes,
        node_sizes,
        edge_sizes,
        z_min,
        z_max,
        coords,
        labels,
    )


def plot_all_func(
    conn_matrix,
    conn_model,
    atlas,
    dir_path,
    ID,
    network,
    labels,
    roi,
    coords,
    thr,
    node_size,
    edge_threshold,
    smooth,
    prune,
    uatlas,
    norm,
    binary,
    hpass,
    extract_strategy,
    edge_color_override=False,
):
    """
    Plot adjacency matrix, connectogram, and glass brain for functional
    connectome.

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
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set
        (e.g. a coordinate atlas).
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    edge_threshold : float
        The actual value, between 0 and 1, that the graph was thresholded
        (can differ from thr if target was not successfully obtained.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    uatlas : str
        File path to atlas parcellation Nifti1Image.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    extract_strategy : str
        The name of a valid function used to reduce the time-series region
        extraction.
    edge_color_override : bool
        Switch that enables random sequential colormap selection for edges.

    """
    import os
    from pynets.core.utils import load_runconfig
    import sys
    import os.path as op
    import random
    import matplotlib
    matplotlib.use("agg")
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    import pkg_resources
    import pickle
    from scipy.spatial import distance
    from pynets.core.utils import load_mat
    from pynets.plotting import plot_gen, plot_graphs
    from pynets.plotting.plot_gen import create_gb_palette

    ch2better_loc = pkg_resources.resource_filename(
        "pynets", "templates/ch2better.nii.gz"
    )

    if sys.platform.startswith('win') is False:
        try:
            nib.load(ch2better_loc)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")
    else:
        try:
            nib.load(ch2better_loc)
        except ImportError as e:
            print(e, f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")

    hardcoded_params = load_runconfig()

    try:
        if edge_color_override is False:
            color_theme = hardcoded_params["plotting"]["functional"
                                                       ]["glassbrain"][
                "color_theme"][0]
        else:
            color_theme = random.choice(
                [
                    "Purples_d",
                    "Blues_d",
                    "Greens_d",
                    "Oranges_d",
                    "Reds_d",
                    "YlOrBr_d",
                    "YlOrRd_d",
                    "OrRd_d",
                    "PuRd_d",
                    "RdPu_d",
                    "BuPu_d",
                    "GnBu_d",
                    "PuBu_d",
                    "YlGnBu_d",
                    "PuBuGn_d",
                    "BuGn_d",
                    "YlGn_d",
                ]
            )

        connectogram = hardcoded_params["plotting"]["connectogram"][0]
        glassbrain = hardcoded_params["plotting"]["glassbrain"][0]
        adjacency = hardcoded_params["plotting"]["adjacency"][0]
        dpi_resolution = hardcoded_params["plotting"]["dpi"][0]
        labeling_atlas = hardcoded_params["plotting"]["labeling_atlas"][0]
    except KeyError as e:
        print(e,
              "Plotting configuration not successfully extracted "
              "from runconfig.yaml"
              )

    if not isinstance(coords, list):
        coords = list(tuple(x) for x in coords)

    if any(isinstance(sub, dict) for sub in labels):
        labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, str):
        import ast
        if any(isinstance(sub, dict) for sub in ast.literal_eval(labels)):
            labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, list):
        if isinstance(labels[0], dict):
            labels = [lab[labeling_atlas] for lab in labels]
    else:
        if not isinstance(labels, list):
            labels = list(labels)

    if not isinstance(conn_matrix, np.ndarray):
        if isinstance(conn_matrix, str):
            if os.path.isfile(conn_matrix):
                conn_matrix = load_mat(conn_matrix)
            else:
                raise ValueError(
                    f"{conn_matrix} type {type(conn_matrix)} not recognized!")
        else:
            raise ValueError(f"{conn_matrix} type {type(conn_matrix)} "
                             f"not recognized!")

    if len(coords) > 0:
        if isinstance(atlas, bytes):
            atlas = atlas.decode("utf-8")

        namer_dir = dir_path + "/figures"
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        # Plot connectogram
        if connectogram is True:
            if len(conn_matrix) > 20:
                try:
                    plot_gen.plot_connectogram(
                        conn_matrix, conn_model, atlas, namer_dir, ID,
                        network, labels)
                except RuntimeWarning:
                    print("\n\n\nWarning: Connectogram plotting failed!")
            else:
                print(
                    "Warning: Cannot plot connectogram for graphs smaller than"
                    " 20 x 20!"
                )

        # Plot adj. matrix based on determined inputs
        if not node_size or node_size == "None":
            node_size = "parc"

        if adjacency is True:
            plot_graphs.plot_conn_mat_func(
                conn_matrix,
                conn_model,
                atlas,
                namer_dir,
                ID,
                network,
                labels,
                roi,
                thr,
                node_size,
                smooth,
                hpass,
                extract_strategy,
            )

        if glassbrain is True:
            views = ["x", "y", "z"]
            # Plot connectome
            out_path_fig = \
                "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % \
                (namer_dir,
                 "/glassbrain_",
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
                 "%s" % ("%s%s" % ("extract-",
                                   extract_strategy) if extract_strategy is
                         not None else ""),
                 "_thr-",
                 thr,
                 ".png",
                 )

            connectome = niplot.plot_connectome(
                np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001,
                black_bg=True
            )
            connectome.add_overlay(ch2better_loc, alpha=0.45, cmap=plt.cm.gray)
            [
                conn_matrix,
                clust_pal_edges,
                clust_pal_nodes,
                node_sizes,
                edge_sizes,
                z_min,
                z_max,
                coords,
                labels,
            ] = create_gb_palette(conn_matrix, color_theme, coords, labels)

            if roi:
                # Save coords to pickle
                coord_paths = f"{namer_dir}/coords_" \
                    f"{op.basename(roi).split('.')[0]}_plotting.pkl"
                with open(coord_paths, "wb") as f:
                    pickle.dump(coords, f, protocol=2)

                # Save labels to pickle
                labels_path = f"{namer_dir}/labelnames_" \
                              f"{op.basename(roi).split('.')[0]}_plotting.pkl"
                with open(labels_path, "wb") as f:
                    pickle.dump(labels, f, protocol=2)

            else:
                # Save coords to pickle
                coord_paths = f"{namer_dir}{'/coords_plotting.pkl'}"
                with open(coord_paths, "wb") as f:
                    pickle.dump(coords, f, protocol=2)

                # Save labels to pickle
                labels_path = f"{namer_dir}{'/labelnames_plotting.pkl'}"
                with open(labels_path, "wb") as f:
                    pickle.dump(labels, f, protocol=2)

            connectome.add_graph(
                conn_matrix,
                [tuple(x) for x in coords],
                edge_cmap=clust_pal_edges,
                edge_vmax=float(z_max),
                edge_vmin=float(z_min),
                node_size=node_sizes,
                node_color=clust_pal_nodes,
                edge_kwargs={"alpha": 0.45, 'zorder': 1},
                node_kwargs={'zorder': 1000}
            )
            for view in views:
                mod_lines = []
                for line, edge_size in list(
                    zip(connectome.axes[view].ax.lines, edge_sizes)
                ):
                    line.set_lw(edge_size)
                    mod_lines.append(line)
                connectome.axes[view].ax.lines = mod_lines

            zorder = 10000
            for view in views:
                coord_anns = []
                for coord, label in list(zip(coords, labels)):
                    if view == 'x':
                        coord_ann = (coord[1], coord[2])
                    if view == 'y':
                        coord_ann = (coord[0], coord[2])
                    if view == 'z':
                        coord_ann = (coord[0], coord[1])

                    if len(coord_anns) > 0:
                        dists = []
                        for c in coord_anns:
                            dists.append(distance.euclidean(coord_ann, c))
                        if any([i < 20 for i in dists]):
                            continue
                    if label == 'Unlabeled':
                        continue
                    coord_anns.append(coord_ann)
                    connectome.axes[view].ax.annotate(
                        label,
                        coord_ann,
                        xycoords='data',
                        textcoords='offset points',
                        xytext=(-0.0001, -0.0001),
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize='2.7',
                        fontweight='extra bold',
                        zorder=zorder,
                        color='black')
                    zorder += 10

                    connectome.axes[view].ax.annotate(
                        label,
                        coord_ann,
                        xycoords='data',
                        textcoords='offset points',
                        xytext=(0, 0),
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize='2.65',
                        fontweight='bold',
                        zorder=zorder,
                        color='white')
                    zorder += 100

            connectome.savefig(out_path_fig, dpi=dpi_resolution)
        else:
            raise RuntimeError(
                "\nno coordinates to plot! Are you running plotting "
                "outside of pynets's internal estimation schemes?")

        plt.close("all")

    return


def plot_all_struct(
    conn_matrix,
    conn_model,
    atlas,
    dir_path,
    ID,
    network,
    labels,
    roi,
    coords,
    thr,
    node_size,
    edge_threshold,
    prune,
    uatlas,
    target_samples,
    norm,
    binary,
    track_type,
    directget,
    min_length,
    error_margin
):
    """
    Plot adjacency matrix, connectogram, and glass brain for functional
    connectome.

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
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set
        (e.g. a coordinate atlas).
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of
        methods triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    edge_threshold : float
        The actual value, between 0 and 1, that the graph was thresholded
        (can differ from thr if target was not successfully obtained.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    uatlas : str
        File path to atlas parcellation Nifti1Image.
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
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    """
    import matplotlib
    matplotlib.use("agg")
    import os
    from pynets.core.utils import load_runconfig
    import sys
    import os.path as op
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    import pkg_resources
    import pickle
    from scipy.spatial import distance
    from pynets.core.utils import load_mat
    from pynets.plotting import plot_gen, plot_graphs
    from pynets.plotting.plot_gen import create_gb_palette
    import mplcyberpunk

    plt.style.use("cyberpunk")

    ch2better_loc = pkg_resources.resource_filename(
        "pynets", "templates/ch2better.nii.gz"
    )

    if sys.platform.startswith('win') is False:
        try:
            nib.load(ch2better_loc)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")
    else:
        try:
            nib.load(ch2better_loc)
        except ImportError as e:
            print(e, f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")

    hardcoded_params = load_runconfig()

    try:
        color_theme = hardcoded_params["plotting"]["structural"][
            "glassbrain"]["color_theme"][0]
        connectogram = hardcoded_params["plotting"]["connectogram"][0]
        glassbrain = hardcoded_params["plotting"]["glassbrain"][0]
        adjacency = hardcoded_params["plotting"]["adjacency"][0]
        dpi_resolution = hardcoded_params["plotting"]["dpi"][0]
        labeling_atlas = hardcoded_params["plotting"]["labeling_atlas"][0]
    except KeyError as e:
        print(
            e, "Plotting configuration not successfully extracted from"
            " runconfig.yaml"
        )

    if any(isinstance(sub, dict) for sub in labels):
        labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, str):
        import ast
        if any(isinstance(sub, dict) for sub in ast.literal_eval(labels)):
            labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, list):
        if isinstance(labels[0], list):
            labels = [lab[0][labeling_atlas] for lab in labels]
    else:
        if not isinstance(labels, list):
            labels = list(labels)

    if not isinstance(conn_matrix, np.ndarray):
        if isinstance(conn_matrix, str):
            if os.path.isfile(conn_matrix):
                conn_matrix = load_mat(conn_matrix)
            else:
                raise ValueError(
                    f"{conn_matrix} type {type(conn_matrix)} not recognized!")
        else:
            raise ValueError(f"{conn_matrix} type {type(conn_matrix)} "
                             f"not recognized!")

    if len(coords) > 0:
        if isinstance(atlas, bytes):
            atlas = atlas.decode("utf-8")

        namer_dir = f"{dir_path}/figures"
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        # Plot connectogram
        if connectogram is True:
            if len(conn_matrix) > 20:
                try:
                    plot_gen.plot_connectogram(
                        conn_matrix, conn_model, atlas, namer_dir, ID,
                        network, labels)
                except RuntimeWarning:
                    print("\n\n\nWarning: Connectogram plotting failed!")
            else:
                print(
                    "Warning: Cannot plot connectogram for graphs smaller than"
                    " 20 x 20!"
                )

        # Plot adj. matrix based on determined inputs
        if not node_size or node_size == "None":
            node_size = "parc"

        if adjacency is True:
            plot_graphs.plot_conn_mat_struct(
                conn_matrix,
                conn_model,
                atlas,
                namer_dir,
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
            )

        if glassbrain is True:
            views = ["x", "y", "z"]
            # Plot connectome
            out_path_fig = \
                "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % \
                (namer_dir,
                 "/glassbrain_",
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

            connectome = niplot.plot_connectome(
                np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001,
                black_bg=True
            )
            connectome.add_overlay(ch2better_loc, alpha=0.45, cmap=plt.cm.gray)

            [
                conn_matrix,
                clust_pal_edges,
                clust_pal_nodes,
                node_sizes,
                edge_sizes,
                _,
                _,
                coords,
                labels,
            ] = create_gb_palette(conn_matrix, color_theme, coords, labels)
            if roi:
                # Save coords to pickle
                coord_paths = f"{namer_dir}/coords_" \
                              f"{op.basename(roi).split('.')[0]}_plotting.pkl"
                with open(coord_paths, "wb") as f:
                    pickle.dump(coords, f, protocol=2)

                # Save labels to pickle
                labels_path = f"{namer_dir}/labelnames_" \
                              f"{op.basename(roi).split('.')[0]}_plotting.pkl"
                with open(labels_path, "wb") as f:
                    pickle.dump(labels, f, protocol=2)
            else:
                # Save coords to pickle
                coord_paths = f"{namer_dir}{'/coords_plotting.pkl'}"
                with open(coord_paths, "wb") as f:
                    pickle.dump(coords, f, protocol=2)

                # Save labels to pickle
                labels_path = f"{namer_dir}{'/labelnames_plotting.pkl'}"
                with open(labels_path, "wb") as f:
                    pickle.dump(labels, f, protocol=2)

            connectome.add_graph(
                conn_matrix,
                [tuple(x) for x in coords],
                edge_cmap=clust_pal_edges,
                edge_vmax=float(1),
                edge_vmin=float(1),
                node_size=node_sizes,
                node_color=clust_pal_nodes,
                edge_kwargs={"alpha": 0.30, 'zorder': 1},
                node_kwargs={'zorder': 1000}
            )

            for view in views:
                mod_lines = []
                for line, edge_size in list(
                    zip(connectome.axes[view].ax.lines, edge_sizes)
                ):
                    line.set_lw(edge_size * 0.20)
                    mod_lines.append(line)
                connectome.axes[view].ax.lines = mod_lines
                mplcyberpunk.make_lines_glow(connectome.axes[view].ax,
                                             n_glow_lines=10,
                                             diff_linewidth=0.80,
                                             alpha_line=0.075)
                connectome.axes[view].ax.set_axisbelow(True)

            zorder = 10000
            for view in views:
                coord_anns = []
                for coord, label in list(zip(coords, labels)):
                    if view == 'x':
                        coord_ann = (coord[1], coord[2])
                    if view == 'y':
                        coord_ann = (coord[0], coord[2])
                    if view == 'z':
                        coord_ann = (coord[0], coord[1])

                    if len(coord_anns) > 0:
                        dists = []
                        for c in coord_anns:
                            dists.append(distance.euclidean(coord_ann, c))
                        if any([i < 20 for i in dists]):
                            continue
                    if label == 'Unlabeled':
                        continue
                    coord_anns.append(coord_ann)
                    connectome.axes[view].ax.set_axisbelow(False)
                    connectome.axes[view].ax.annotate(
                        label,
                        coord_ann,
                        xycoords='data',
                        textcoords='offset points',
                        xytext=(-0.0001, -0.0001),
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize='2.7',
                        fontweight='extra bold',
                        zorder=zorder,
                        color='black')
                    zorder += 10

                    connectome.axes[view].ax.annotate(
                        label,
                        coord_ann,
                        xycoords='data',
                        textcoords='offset points',
                        xytext=(0, 0),
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize='2.65',
                        fontweight='bold',
                        zorder=zorder,
                        color='orange')
                    zorder += 100

            connectome.savefig(out_path_fig, dpi=dpi_resolution)
        else:
            raise RuntimeError(
                "\nNo coordinates to plot! Are you running plotting "
                "outside of pynets's internal estimation schemes?")

        plt.close("all")

    return


def plot_all_struct_func(mG_path, namer_dir, name, modality_paths, metadata):
    """
    Plot adjacency matrix and glass brain for structural-functional multiplex
    connectome.

    Parameters
    ----------
    mG_path : str
        A gpickle file containing a a MultilayerGraph object
        (See https://github.com/nkoub/multinetx).
    namer_dir : str
        Path to output directory for multiplex data.
    name : str
        Concatenation of multimodal graph filenames.
    modality_paths : tuple
        A tuple of filepath strings to the raw structural and raw functional
        connectome graph files (.npy).
    metadata : dict
        Dictionary coontaining coords and labels shared by each layer of the
        multilayer graph.

    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import sys
    if sys.platform.startswith('win') is False:
        import indexed_gzip
    import nibabel as nib
    import multinetx as mx
    import matplotlib
    matplotlib.use("agg")
    import pkg_resources
    import networkx as nx
    from pynets.core.utils import load_runconfig
    import sys
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    from pynets.core import thresholding
    from pynets.plotting.plot_gen import create_gb_palette
    import mplcyberpunk
    from scipy.spatial import distance

    coords = metadata["coords"]
    labels = metadata["labels"]

    ch2better_loc = pkg_resources.resource_filename(
        "pynets", "templates/ch2better.nii.gz"
    )

    if sys.platform.startswith('win') is False:
        try:
            nib.load(ch2better_loc)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")
    else:
        try:
            nib.load(ch2better_loc)
        except ImportError as e:
            print(e, f"\nCannot load plotting template. Do you have git-lfs "
                  f"installed?")

    hardcoded_params = load_runconfig()

    try:
        color_theme_func = hardcoded_params["plotting"]["functional"][
            "glassbrain"]["color_theme"][0]
        color_theme_struct = hardcoded_params["plotting"]["structural"][
            "glassbrain"
        ]["color_theme"][0]
        glassbrain = hardcoded_params["plotting"]["glassbrain"][0]
        adjacency = hardcoded_params["plotting"]["adjacency"][0]
        dpi_resolution = hardcoded_params["plotting"]["dpi"][0]
        labeling_atlas = hardcoded_params["plotting"]["labeling_atlas"][0]
    except KeyError as e:
        print(e,
              "Plotting configuration not successfully extracted from"
              " runconfig.yaml"
              )

    if any(isinstance(sub, dict) for sub in labels):
        labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, str):
        import ast
        if any(isinstance(sub, dict) for sub in ast.literal_eval(labels)):
            labels = [lab[labeling_atlas] for lab in labels]
    elif isinstance(labels, list):
        if isinstance(labels[0], list):
            labels = [lab[0][labeling_atlas] for lab in labels]
    else:
        if not isinstance(labels, list):
            labels = list(labels)

    [struct_mat, func_mat] = [
        np.load(modality_paths[0]), np.load(modality_paths[1])]

    if adjacency is True:
        # Multiplex adjacency
        mG = nx.read_gpickle(mG_path)

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        adj = thresholding.standardize(
            mx.adjacency_matrix(mG, weight="weight").todense()
        )
        [z_min, z_max] = np.abs(adj).min(), np.abs(adj).max()

        adj[adj == 0] = np.nan

        ax1.imshow(
            adj,
            origin="lower",
            interpolation="nearest",
            cmap=plt.cm.RdBu,
            vmin=0.01,
            vmax=z_max,
        )
        ax1.set_title("Supra-Adjacency Matrix")

        ax2 = fig.add_subplot(122)
        ax2.axis("off")
        ax2.set_title(f"Functional-Structural Multiplex Connectome")

        pos = mx.get_position(
            mG,
            mx.fruchterman_reingold_layout(mG.get_layer(0)),
            layer_vertical_shift=1.0,
            layer_horizontal_shift=0.0,
            proj_angle=7,
        )
        edge_intensities = []
        for a, b, w in mG.edges(data=True):
            if w != {}:
                edge_intensities.append(w["weight"])
            else:
                edge_intensities.append(0)

        node_centralities = list(
            nx.algorithms.eigenvector_centrality(mG, weight="weight").values()
        )
        mx.draw_networkx(
            mG,
            pos=pos,
            ax=ax2,
            node_size=100,
            with_labels=True,
            edge_color=edge_intensities,
            node_color=node_centralities,
            edge_vmin=z_min,
            edge_vmax=z_max,
            dim=3,
            font_size=6,
            widths=3,
            alpha=0.7,
            cmap=plt.cm.RdBu,
        )
        plt.savefig(
            f"{namer_dir}/adjacency-supra_{name[:200]}.png",
            dpi=dpi_resolution)

    if glassbrain is True:
        plt.style.use("cyberpunk")

        # Multiplex glass brain
        views = ["x", "y", "z"]
        connectome = niplot.plot_connectome(
            np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001,
            black_bg=True
        )
        connectome.add_overlay(ch2better_loc, alpha=0.50, cmap=plt.cm.gray)

        [
            struct_mat,
            _,
            _,
            _,
            edge_sizes_struct,
            _,
            _,
            coords,
            labels,
        ] = create_gb_palette(
            struct_mat, color_theme_struct, coords, labels, prune=False
        )

        connectome.add_graph(
            struct_mat,
            [tuple(x) for x in coords],
            edge_threshold="10%",
            edge_cmap=plt.cm.binary,
            node_size=1,
            #edge_kwargs={"alpha": 0.10, "lineStyle": "dashed"},
            edge_kwargs={"alpha": 0.10},
            node_kwargs={"alpha": 0.95},
            edge_vmax=float(1),
            edge_vmin=float(1),
        )

        for view in views:
            mod_lines = []
            for line, edge_size in list(
                zip(connectome.axes[view].ax.lines, edge_sizes_struct)
            ):
                line.set_lw(edge_size*0.5)
                mod_lines.append(line)
            connectome.axes[view].ax.lines = mod_lines
            mplcyberpunk.make_lines_glow(connectome.axes[view].ax,
                                         n_glow_lines=10, diff_linewidth=0.20,
                                         alpha_line=0.10)
        [func_mat,
         clust_pal_edges,
         clust_pal_nodes,
         node_sizes,
         edge_sizes_func,
         z_min,
         z_max,
         coords,
         labels,
         ] = create_gb_palette(func_mat,
                               color_theme_func,
                               coords,
                               labels,
                               prune=False)
        connectome.add_graph(
            func_mat,
            [tuple(x) for x in coords],
            edge_threshold="0%",
            edge_cmap=clust_pal_edges,
            edge_kwargs={"alpha": 0.50, 'zorder': 500},
            edge_vmax=float(z_max),
            edge_vmin=float(z_min),
            node_size=node_sizes,
            node_color=clust_pal_nodes,
            node_kwargs={'zorder': 1000}
        )

        for view in views:
            mod_lines = []
            for line, edge_size in list(
                zip(
                    connectome.axes[view].ax.lines[len(edge_sizes_struct):],
                    edge_sizes_func,
                )
            ):
                line.set_lw(edge_size)
                mod_lines.append(line)
            connectome.axes[view].ax.lines[len(edge_sizes_struct):] = mod_lines
            connectome.axes[view].ax.set_axisbelow(True)

        zorder = 10000
        for view in views:
            coord_anns = []
            for coord, label in list(zip(coords, labels)):
                if view == 'x':
                    coord_ann = (coord[1], coord[2])
                if view == 'y':
                    coord_ann = (coord[0], coord[2])
                if view == 'z':
                    coord_ann = (coord[0], coord[1])

                if len(coord_anns) > 0:
                    dists = []
                    for c in coord_anns:
                        dists.append(distance.euclidean(coord_ann, c))
                    if any([i < 15 for i in dists]):
                        continue
                if label == 'Unlabeled':
                    continue
                coord_anns.append(coord_ann)
                connectome.axes[view].ax.set_axisbelow(False)
                connectome.axes[view].ax.annotate(label,
                                                  coord_ann,
                                                  xycoords='data',
                                                  textcoords='offset points',
                                                  xytext=(-0.0001, -0.0001),
                                                  horizontalalignment='center',
                                                  verticalalignment='top',
                                                  fontsize='2.75',
                                                  fontweight='extra bold',
                                                  zorder=zorder,
                                                  color='black')
                zorder += 10

                connectome.axes[view].ax.annotate(label,
                                                  coord_ann,
                                                  xycoords='data',
                                                  textcoords='offset points',
                                                  xytext=(0, 0),
                                                  horizontalalignment='center',
                                                  verticalalignment='top',
                                                  fontsize='2.7',
                                                  fontweight='bold',
                                                  zorder=zorder,
                                                  color='orange')
                zorder += 100

        connectome.savefig(
            f"{namer_dir}/glassbrain-mplx_{name[:200]}.png", dpi=dpi_resolution
        )

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
    # window.show(renderer)
    window.record(renderer, n_frames=1, out_path=fname, size=(900, 900))
    return


def view_tractogram(streams, atlas):
    import nibabel as nib
    import pkg_resources
    from nibabel.affines import apply_affine
    from dipy.io.streamline import load_tractogram
    from fury import actor, window, colormap
    from dipy.tracking.utils import streamline_near_roi
    from nilearn.image import resample_to_img
    from pynets.registration.utils import rescale_affine_to_center
    from dipy.tracking.streamline import transform_streamlines
    from dipy.align.imaffine import (
        transform_origins,
    )

    FA_template_path = pkg_resources.resource_filename("pynets",
                                                       "templates/"
                                                       "FA_2mm.nii.gz")
    ch2_better_path = pkg_resources.resource_filename("pynets",
                                                      "templates/"
                                                      "ch2better.nii.gz")
    FA_template_img = nib.load(FA_template_path)
    clean_template_img = nib.load(ch2_better_path)

    tractogram = load_tractogram(
        streams,
        'same',
        bbox_valid_check=False,
    )

    affine_map = transform_origins(
        clean_template_img.get_fdata(), clean_template_img.affine,
        FA_template_img.get_fdata(), FA_template_img.affine)
    warped_aff = affine_map.affine_inv.copy()
    warped_aff_scaled = rescale_affine_to_center(
        warped_aff, voxel_dims=[4, 4, 4],
        target_center_coords=clean_template_img.affine[:3, 3]*np.array([0.5,
                                                                        0.5,
                                                                        1]))
    streamlines = transform_streamlines(
        tractogram.streamlines, warped_aff_scaled)

    # Load atlas rois
    atlas_img = nib.load(atlas)
    resampled_img = resample_to_img(atlas_img, clean_template_img,
                                    interpolation='nearest', clip=False)
    atlas_img_data = resampled_img.get_fdata().astype('uint32')

    # Collapse list of connected streamlines for visualization

    clean_template_data = clean_template_img.get_data()
    mean, std = clean_template_data[clean_template_data > 0].mean(), \
        clean_template_data[clean_template_data > 0].std()
    value_range = (mean - 3 * std, mean + 3 * std)
    clean_template_data[clean_template_data < 0.01] = 0
    template_actor = actor.slicer(clean_template_data, np.eye(4),
                                  value_range)

    renderer = window.Renderer()
    renderer.add(template_actor)
    template_actor2 = template_actor.copy()
    template_actor2.display(template_actor2.shape[0] // 2, None, None)
    renderer.add(template_actor2)

    # renderer.add(actor.contour_from_roi(atlas_img_data.astype('bool')))

    # Creat palette of roi colors and add them to the scene as faint contours
    roi_colors = np.random.rand(int(np.max(atlas_img_data)), 3)
    parcel_contours = []

    i = 0
    for roi in np.unique(atlas_img_data)[1:]:
        include_roi_coords = np.array(np.where(atlas_img_data == roi)).T
        x_include_roi_coords = apply_affine(np.eye(4), include_roi_coords)
        bool_list = []
        for sl in streamlines:
            bool_list.append(streamline_near_roi(sl, x_include_roi_coords,
                                                 tol=1.0, mode='either_end'))
        if sum(bool_list) > 0:
            print('ROI: ' + str(i))
            parcel_contours.append(actor.contour_from_roi(atlas_img_data ==
                                                          roi,
                                                          color=roi_colors[i],
                                                          opacity=0.8))
        else:
            pass
        i = i + 1

    for vol_actor in parcel_contours:
        renderer.add(vol_actor)

    lines_actor = actor.line(streamlines,
                             colormap.create_colormap(
                                 np.ones([len(streamlines)]),
                                 name='Greys_r', auto=True),
                             lod_points=10000, depth_cue=True, linewidth=0.3,
                             fake_tube=True, opacity=1.0)
    renderer.add(lines_actor)

    window.show(renderer)
    return


def plot_graph_measure_hists(csv_all_metrics):
    """
    Plot histograms for each graph theoretical measure for a given
    subject.

    Parameters
    ----------
    csv_all_metrics : str
        CSV file of concatenated graph measures across ensemble.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import scale

    df_concat = pd.read_csv(csv_all_metrics)
    df_concat = df_concat.drop(columns=['id'])
    measures = df_concat.columns
    print("Making model plots...")

    def nearest_square_root(limit):
        answer = 0
        while (answer + 1) ** 2 < limit:
            answer += 1
        return int(np.sqrt(answer ** 2))

    try:
        global_measures = list(set([
            meas.split('auc_')[1] for meas in measures
        ]))
    except ValueError as e:
        print(e, measures)

    fig, axes = plt.subplots(
        ncols=nearest_square_root(len(global_measures)),
        nrows=nearest_square_root(len(global_measures)),
        sharex=True,
        sharey=True,
        figsize=(10, 10),
    )
    for i, ax in enumerate(axes.flatten()):
        try:
            ensemble_metric_df = df_concat.loc[:,
                                               df_concat.columns.str.contains(
                                                   global_measures[i])]
            x = np.asarray(
                ensemble_metric_df[
                    np.isfinite(ensemble_metric_df)
                ]
            )[0]
        except BaseException:
            continue
        try:
            x = scale(x, axis=0, with_mean=True, with_std=True, copy=True)
        except BaseException:
            continue
        if True in pd.isnull(x):
            try:
                x = x[~pd.isnull(x)]
                if len(x) > 0:
                    print(
                        f"NaNs encountered for {global_measures[i]}. Plotting "
                        f"and averaging across non-missing "
                        f"values. Checking output is recommended...")
                    ax.hist(x, density=True, bins="auto", alpha=0.8)
                    ax.set_title(global_measures[i])
                else:
                    print(
                        f"Warning: No numeric data to plot for "
                        f"{global_measures[i]}"
                    )
                    continue
            except BaseException:
                continue
        else:
            try:
                ax.hist(x, density=True, bins="auto", alpha=0.8)
                ax.set_title(global_measures[i])
            except BaseException:
                print(
                    f"Inf or NaN values encounterd. No numeric data to plot "
                    f"for {global_measures[i]}"
                )
                continue

    plt.tight_layout()
    return plt
