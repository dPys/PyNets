#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import warnings
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import networkx as nx
import tkinter

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def plot_connectogram(
    conn_matrix,
    conn_model,
    dir_path,
    ID,
    subnet,
    labels,
    comm="nodes",
    color_scheme="interpolateBlues",
    prune=False,
):
    """
    Plot a connectogram for a given connectivity matrix.

    Parameters
    ----------
    conn_matrix : array
        NxN matrix.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for
       partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    subnet : str
        Resting-state network based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    comm : str, optional default: 'nodes'
        Communitity setting, either 'nodes' or 'links'
    color_scheme : str, optional, default: 'interpolateBlues'
        Color scheme in json.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.

    """
    import json
    from pathlib import Path
    from networkx.readwrite import json_graph
    from pynets.core.thresholding import normalize
    from pynets.statistics.individual.algorithms import most_important, \
        link_communities, community_resolution_selection

    # from scipy.cluster.hierarchy import linkage, fcluster
    from nipype.utils.filemanip import save_json

    conn_matrix = normalize(conn_matrix)
    G = nx.from_numpy_matrix(np.abs(conn_matrix))
    if prune is True:
        [G, pruned_nodes] = most_important(G)
        conn_matrix = nx.to_numpy_array(G)

        pruned_nodes.sort(reverse=True)
        for j in pruned_nodes:
            del labels[labels.index(labels[j])]

    if comm == "nodes" and len(conn_matrix) > 40:
        G = nx.from_numpy_matrix(np.abs(conn_matrix))
        _, node_comm_aff_mat, resolution, num_comms = \
            community_resolution_selection(G)
        clust_levels = len(node_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([node_comm_aff_mat == 0]).astype("int"))
        label_arr = (
            node_comm_aff_mat *
            np.expand_dims(
                np.arange(
                    1,
                    clust_levels +
                    1),
                axis=1) +
            mask_mat)
    elif comm == "links" and len(conn_matrix) > 40:
        # Plot link communities
        link_comm_aff_mat = link_communities(
            conn_matrix, type_clustering="single")[0]
        print(f"{'Found '}{str(len(link_comm_aff_mat))}{' communities...'}")
        clust_levels = len(link_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([link_comm_aff_mat == 0]).astype("int"))
        label_arr = (
            link_comm_aff_mat *
            np.expand_dims(
                np.arange(
                    1,
                    clust_levels +
                    1),
                axis=1) +
            mask_mat)
    else:
        return

    def _get_node_label(node_idx, labels, clust_levels_tmp):
        """
        Tag a label to a given node based on its community/cluster assignment
        """
        from collections import OrderedDict

        def _write_roman(num):
            """
            Create community/cluster assignments using a Roman-Numeral
            generator.
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
                    num -= r * x
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
        node_lab_alph = ".".join(
            ["{}{}".format(abet[i], int(l)) for i, l in enumerate(node_labels)]
        ) + ".{}".format(labels[node_idx])
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
            wei = G.get_edge_data(node_idx, int(i))["weight"]
            weight_vec.append(wei)
        entry = {}
        nodes_label = _get_node_label(node_idx, label_arr, clust_levels_tmp)
        entry["name"] = nodes_label
        entry["size"] = len(connections)
        entry["imports"] = [
            _get_node_label(int(d) - 1, label_arr, clust_levels_tmp)
            for d in connections
        ]
        entry["weights"] = weight_vec
        output.append(entry)

    if subnet:
        json_file_name = (
            f"{str(ID)}{'_'}{subnet}{'_connectogram_'}{conn_model}"
            f"{'_network.json'}"
        )
        json_fdg_file_name = (
            f"{str(ID)}{'_'}{subnet}{'_fdg_'}{conn_model}{'_network.json'}"
        )
        connectogram_plot = f"{dir_path}{'/'}{json_file_name}"
        fdg_js_sub = f"{dir_path}{'/'}{str(ID)}{'_'}{subnet}{'_fdg_'}" \
                     f"{conn_model}{'_network.js'}"
        fdg_js_sub_name = f"{str(ID)}{'_'}{subnet}{'_fdg_'}{conn_model}" \
                          f"{'_network.js'}"
        connectogram_js_sub = (
            f"{dir_path}/{str(ID)}_{subnet}_connectogram_{conn_model}"
            f"_network.js"
        )
        connectogram_js_name = (
            f"{str(ID)}{'_'}{subnet}{'_connectogram_'}{conn_model}"
            f"{'_network.js'}"
        )
    else:
        json_file_name = f"{str(ID)}{'_connectogram_'}{conn_model}{'.json'}"
        json_fdg_file_name = f"{str(ID)}{'_fdg_'}{conn_model}{'.json'}"
        connectogram_plot = f"{dir_path}{'/'}{json_file_name}"
        connectogram_js_sub = (
            f"{dir_path}{'/'}{str(ID)}{'_connectogram_'}{conn_model}{'.js'}"
        )
        fdg_js_sub = f"{dir_path}{'/'}{str(ID)}{'_fdg_'}{conn_model}{'.js'}"
        fdg_js_sub_name = f"{str(ID)}{'_fdg_'}{conn_model}{'.js'}"
        connectogram_js_name = f"{str(ID)}{'_connectogram_'}{conn_model}" \
                               f"{'.js'}"
    save_json(connectogram_plot, output)

    # Force-directed graphing
    G = nx.from_numpy_matrix(
        np.round(
            np.abs(conn_matrix).astype("float64"),
            6))
    data = json_graph.node_link_data(G)
    data.pop("directed", None)
    data.pop("graph", None)
    data.pop("multigraph", None)
    for k in range(len(data["links"])):
        data["links"][k]["value"] = data["links"][k].pop("weight")
    for k in range(len(data["nodes"])):
        data["nodes"][k]["id"] = str(data["nodes"][k]["id"])
    for k in range(len(data["links"])):
        data["links"][k]["source"] = str(data["links"][k]["source"])
        data["links"][k]["target"] = str(data["links"][k]["target"])

    # Add community structure
    for k in range(len(data["nodes"])):
        data["nodes"][k]["group"] = str(label_arr[0][k])

    # Add node labels
    for k in range(len(data["nodes"])):
        data["nodes"][k]["name"] = str(labels[k])

    out_file = f"{dir_path}{'/'}{str(json_fdg_file_name)}"
    save_json(out_file, data)

    # Copy index.html and json to dir_path
    conn_js_path = str(Path(__file__).parent / "connectogram.js")
    index_html_path = str(Path(__file__).parent / "index.html")
    fdg_replacements_js = {"FD_graph.json": str(json_fdg_file_name)}
    replacements_html = {
        "connectogram.js": str(connectogram_js_name),
        "fdg.js": str(fdg_js_sub_name),
    }
    fdg_js_path = str(Path(__file__).parent / "fdg.js")
    with open(index_html_path) as infile, open(
        str(dir_path + "/index.html"), "w"
    ) as outfile:
        for line in infile:
            for src, target in replacements_html.items():
                line = line.replace(src, target)
            outfile.write(line)

    replacements_js = {
        "template.json": str(json_file_name),
        "interpolateCool": str(color_scheme),
    }
    with open(conn_js_path) as infile, open(connectogram_js_sub, "w") as \
        outfile:
        for line in infile:
            for src, target in replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

    with open(fdg_js_path) as infile, open(fdg_js_sub, "w") as outfile:
        for line in infile:
            for src, target in fdg_replacements_js.items():
                line = line.replace(src, target)
            outfile.write(line)

    return
