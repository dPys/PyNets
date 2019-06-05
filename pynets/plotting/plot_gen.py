# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import numpy as np
import networkx as nx
import os
import os.path as op
import warnings
warnings.filterwarnings("ignore")


def plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names):
    """

    :param conn_matrix:
    :param conn_model:
    :param atlas_select:
    :param dir_path:
    :param ID:
    :param network:
    :param label_names:
    :return:
    """
    import json
    from pathlib import Path
    from networkx.readwrite import json_graph
    from pynets.thresholding import normalize
    from pynets.stats.netstats import most_important
    from scipy.cluster.hierarchy import linkage, fcluster
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
    G = nx.from_numpy_matrix(conn_matrix)
    if pruned is True:
        [G, pruned_nodes] = most_important(G)
        conn_matrix = nx.to_numpy_array(G)

        pruned_nodes.sort(reverse=True)
        for j in pruned_nodes:
            del label_names[label_names.index(label_names[j])]

    def doClust(X, clust_levels):
        """

        :param X:
        :param clust_levels:
        :return:
        """
        # get the linkage diagram
        Z = linkage(X, 'ward')
        # choose # cluster levels
        cluster_levels = range(1, int(clust_levels))
        # init array to store labels for each level
        clust_levels_tmp = int(clust_levels) - 1
        label_arr = np.zeros((int(clust_levels_tmp), int(X.shape[0])))
        # iterate thru levels
        for c in cluster_levels:
            fl = fcluster(Z, c, criterion='maxclust')
            #print(fl)
            label_arr[c-1, :] = fl
        return label_arr, clust_levels_tmp

    if comm == 'nodes' and len(conn_matrix) > 40:
        from pynets.stats.netstats import modularity_louvain_und_sign

        gamma = nx.density(nx.from_numpy_array(conn_matrix))
        try:
            if network or len(conn_matrix) < 100:
                [node_comm_aff_mat, q] = modularity_louvain_und_sign(conn_matrix, gamma=float(gamma * 0.001))
            else:
                [node_comm_aff_mat, q] = modularity_louvain_und_sign(conn_matrix, gamma=float(gamma * 0.01))
            print("%s%s%s%s%s" % ('Found ', str(len(np.unique(node_comm_aff_mat))), ' communities using γ=', str(gamma), '...'))
        except:
            print('\nWARNING: Louvain community detection failed. Proceeding with single community affiliation vector...')
            node_comm_aff_mat = np.ones(conn_matrix.shape[0]).astype('int')
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
    elif len(conn_matrix) > 20:
        print('Graph too small for reliable plotting of communities. Plotting by fcluster instead...')
        if len(conn_matrix) >= 250:
            clust_levels = 7
        elif len(conn_matrix) >= 200:
            clust_levels = 6
        elif len(conn_matrix) >= 150:
            clust_levels = 5
        elif len(conn_matrix) >= 100:
            clust_levels = 4
        elif len(conn_matrix) >= 50:
            clust_levels = 3
        else:
            clust_levels = 2
        [label_arr, clust_levels_tmp] = doClust(conn_matrix, clust_levels)

    def get_node_label(node_idx, labels, clust_levels_tmp):
        """

        :param node_idx:
        :param labels:
        :param clust_levels_tmp:
        :return:
        """
        from collections import OrderedDict

        def write_roman(num):
            """

            :param num:
            :return:
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
            rn_list.append(json.dumps(write_roman(k)))
        abet = rn_list
        node_lab_alph = ".".join(["{}{}".format(abet[i], int(l)) for i, l in enumerate(node_labels)]) + ".{}".format(
            label_names[node_idx])
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
        nodes_label = get_node_label(node_idx, label_arr, clust_levels_tmp)
        entry["name"] = nodes_label
        entry["size"] = len(connections)
        entry["imports"] = [get_node_label(int(d)-1, label_arr, clust_levels_tmp) for d in connections]
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
    G = nx.from_numpy_matrix(np.round(conn_matrix.astype('float64'), 6))
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
        data['nodes'][k]['name'] = str(label_names[k])

    out_file = "%s%s%s" % (dir_path, '/', str(json_fdg_file_name))
    save_json(out_file, data)

    # Copy index.html and json to dir_path
    #conn_js_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/plotting/connectogram.js'
    #index_html_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/plotting/index.html'
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


def plot_timeseries(time_series, network, ID, dir_path, atlas_select, labels):
    """

    :param time_series:
    :param network:
    :param ID:
    :param dir_path:
    :param atlas_select:
    :param labels:
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
    plt.close()


def plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, roi, coords, thr,
             node_size, edge_threshold, smooth, prune, uatlas_select, c_boot, norm, binary):
    """

    :param conn_matrix:
    :param conn_model:
    :param atlas_select:
    :param dir_path:
    :param ID:
    :param network:
    :param label_names:
    :param roi:
    :param coords:
    :param thr:
    :param node_size:
    :param edge_threshold:
    :param smooth:
    :param prune:
    :param uatlas_select:
    :param c_boot:
    :param norm:
    :param binary:
    :return:
    """
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    import pkg_resources
    import networkx as nx
    from pynets import plotting, thresholding
    from pynets.plotting import plot_gen, plot_graphs
    from pynets.stats.netstats import most_important, prune_disconnected
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    coords = list(coords)
    label_names = list(label_names)
    if len(coords) > 0:
        dpi_resolution = 500
        if '\'b' in atlas_select:
            atlas_select = atlas_select.decode('utf-8')
        if (prune == 1 or prune == 2) and len(coords) == conn_matrix.shape[0]:
            G_pre = nx.from_numpy_matrix(conn_matrix)
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
            label_names_pre = list(label_names)
            if len(pruned_nodes) > 0:
                for j in pruned_nodes:
                    label_names_pre.pop(j)
                    coords_pre.pop(j)
                conn_matrix = nx.to_numpy_array(G)
                label_names = label_names_pre
                coords = coords_pre
            else:
                print('No nodes to prune for plot...')

        coords = list(tuple(x) for x in coords)
        # Plot connectogram
        if len(conn_matrix) > 20:
            try:
                plot_gen.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)
            except RuntimeWarning:
                print('\n\n\nWarning: Connectogram plotting failed!')
        else:
            print('Warning: Cannot plot connectogram for graphs smaller than 20 x 20!')

        # Plot adj. matrix based on determined inputs
        if not node_size or node_size == 'None':
            node_size = 'parc'
        plot_graphs.plot_conn_mat_func(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, roi,
                                    thr, node_size, smooth, c_boot)

        # Plot connectome
        if roi:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', str(atlas_select), '_', str(conn_model), '_', str(op.basename(roi).split('.')[0]), "%s" % ("%s%s%s" % ('_', network, '_') if network else "_"), str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'), "%s" % ("%s%s" % (smooth, 'fwhm_') if float(smooth) > 0 else 'nosm_'), 'func_glass_viz.png')
            # Save coords to pickle
            coord_path = "%s%s%s%s" % (dir_path, '/coords_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)
            # Save labels to pickle
            labels_path = "%s%s%s%s" % (dir_path, '/labelnames_', op.basename(roi).split('.')[0], '_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(label_names, f, protocol=2)
        else:
            out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', str(atlas_select), '_', str(conn_model), "%s" % ("%s%s%s" % ('_', network, '_') if network else "_"), str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'), "%s" % ("%s%s" % (smooth, 'fwhm_') if float(smooth) > 0 else 'nosm_'), 'func_glass_viz.png')
            # Save coords to pickle
            coord_path = "%s%s" % (dir_path, '/coords_plotting.pkl')
            with open(coord_path, 'wb') as f:
                pickle.dump(coords, f, protocol=2)
            # Save labels to pickle
            labels_path = "%s%s" % (dir_path, '/labelnames_plotting.pkl')
            with open(labels_path, 'wb') as f:
                pickle.dump(label_names, f, protocol=2)

        ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")
        connectome = niplot.plot_connectome(np.zeros(shape=(1, 1)), [(0, 0, 0)], node_size=0.0001, black_bg=True)
        connectome.add_overlay(ch2better_loc, alpha=0.45, cmap=plt.cm.gray)
        #connectome.add_overlay(ch2better_loc, alpha=0.35, cmap=plt.cm.gray)
        conn_matrix = np.array(np.array(thresholding.autofix(conn_matrix)))
        [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
        if node_size == 'parc':
            node_size_plot = int(2)
            if uatlas_select:
                connectome.add_contours(uatlas_select, filled=True, alpha=0.20, cmap=plt.cm.gist_rainbow)
        else:
            node_size_plot = int(node_size)
        if len(coords) != conn_matrix.shape[0]:
            raise RuntimeWarning('\nWARNING: Number of coordinates does not match conn_matrix dimensions. If you are using disparity filtering, try relaxing the α threshold.')
        else:
            color_theme = 'Blues'
            #color_theme = 'Greens'
            #color_theme = 'Reds'
            node_color = 'auto'
            connectome.add_graph(conn_matrix, coords, edge_threshold=edge_threshold, edge_cmap=color_theme, edge_vmax=float(z_max),
                                 edge_vmin=float(z_min), node_size=node_size_plot, node_color='auto')
            connectome.savefig(out_path_fig, dpi=dpi_resolution)
    else:
        raise RuntimeError('\nERROR: no coordinates to plot! Are you running plotting outside of pynets\'s internal estimation schemes?')
    return


def structural_plotting(conn_matrix, uatlas_select, streamlines_mni, template_mask, interactive=False):
    """

    :param conn_matrix:
    :param uatlas_select:
    :param streamlines_mni:
    :param template_mask:
    :param interactive:
    :return:
    """
    import nibabel as nib
    import numpy as np
    import networkx as nx
    import os
    import pkg_resources
    from nibabel.affines import apply_affine
    from fury import actor, window, colormap, ui
    from dipy.tracking.utils import streamline_near_roi
    from nilearn.plotting import find_parcellation_cut_coords
    from nilearn.image import resample_to_img
    from pynets.thresholding import normalize

    ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")

    # Instantiate scene
    r = window.Renderer()

    # Set camera
    r.set_camera(position=(-176.42, 118.52, 128.20),
                 focal_point=(113.30, 128.31, 76.56),
                 view_up=(0.18, 0.00, 0.98))

    # Load atlas rois
    atlas_img = nib.load(uatlas_select)
    atlas_img_data = atlas_img.get_data()

    # Collapse list of connected streamlines for visualization
    streamlines = nib.streamlines.load(streamlines_mni).streamlines
    parcels = []
    i = 0
    for roi in np.unique(atlas_img_data)[1:]:
        parcels.append(atlas_img_data == roi)
        i = i + 1

    # Add streamlines as cloud of 'white-matter'
    streamlines_actor = actor.line(streamlines,
                                   colormap.create_colormap(np.ones([len(streamlines)]), name='Greys_r', auto=True),
                                   lod_points=10000, depth_cue=True, linewidth=0.2, fake_tube=True, opacity=1.0)
    r.add(streamlines_actor)

    # Creat palette of roi colors and add them to the scene as faint contours
    roi_colors = np.random.rand(int(np.max(atlas_img_data)), 3)
    parcel_contours = []
    i = 0
    for roi in np.unique(atlas_img_data)[1:]:
        include_roi_coords = np.array(np.where(atlas_img_data == roi)).T
        x_include_roi_coords = apply_affine(np.eye(4), include_roi_coords)
        bool_list = []
        for sl in streamlines:
            bool_list.append(streamline_near_roi(sl, x_include_roi_coords, tol=1.0, mode='either_end'))
        if sum(bool_list) > 0:
            print('ROI: ' + str(i))
            parcel_contours.append(actor.contour_from_roi(atlas_img_data == roi, color=roi_colors[i], opacity=0.2))
        else:
            pass
        i = i + 1

    for vol_actor in parcel_contours:
        r.add(vol_actor)

    # Get voxel coordinates of parcels and add them as 3d spherical centroid nodes
    [coords, label_names] = find_parcellation_cut_coords(atlas_img, background_label=0, return_label_names=True)

    def mmToVox(nib_nifti, mmcoords):
        """

        :param nib_nifti:
        :param mmcoords:
        :return:
        """
        return nib.affines.apply_affine(np.linalg.inv(nib_nifti.affine), mmcoords)

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(atlas_img, i))
    coords_vox = list(set(list(tuple(x) for x in coords_vox)))

    # Build an edge list of 3d lines
    G = nx.from_numpy_array(normalize(conn_matrix))
    for i in G.nodes():
        nx.set_node_attributes(G, {i: coords_vox[i]}, label_names[i])

    G.remove_nodes_from(list(nx.isolates(G)))
    G_filt = nx.Graph()
    fedges = filter(lambda x: G.degree()[x[0]] > 0 and G.degree()[x[1]] > 0, G.edges())
    G_filt.add_edges_from(fedges)

    coord_nodes = []
    for i in range(len(G.edges())):
        edge = list(G.edges())[i]
        [x, y] = edge
        x_coord = list(G.nodes[x].values())[0]
        x_label = list(G.nodes[x].keys())[0]
        l_x = actor.label(text=str(x_label), pos=x_coord, scale=(1, 1, 1), color=(50, 50, 50))
        r.add(l_x)
        y_coord = list(G.nodes[y].values())[0]
        y_label = list(G.nodes[y].keys())[0]
        l_y = actor.label(text=str(y_label), pos=y_coord, scale=(1, 1, 1), color=(50, 50, 50))
        r.add(l_y)
        coord_nodes.append(x_coord)
        coord_nodes.append(y_coord)
        c = actor.line([(x_coord, y_coord)], window.colors.coral,
                       linewidth=100 * (float(G.get_edge_data(x, y)['weight'])) ^ 2)
        r.add(c)

    point_actor = actor.point(list(set(coord_nodes)), window.colors.grey, point_radius=0.75)
    r.add(point_actor)

    # Load glass brain template and resample to MNI152_2mm brain
    template_img = nib.load(ch2better_loc)
    template_target_img = nib.load(template_mask)
    res_brain_img = resample_to_img(template_img, template_target_img)
    template_img_data = res_brain_img.get_data().astype('bool')
    template_actor = actor.contour_from_roi(template_img_data, color=(50, 50, 50), opacity=0.05)
    r.add(template_actor)

    # Show scene
    if interactive is True:
        window.show(r, size=(600, 600), reset_camera=False)
    else:
        fig_path = os.path.dirname(streamlines_mni) + '/3d_connectome_fig.png'
        window.record(r, out_path=fig_path, size=(600, 600))

    return


def plot_graph_measure_hists(df_concat, measures, net_pick_file):
    """

    :param df_concat:
    :param measures:
    :param net_pick_file:
    :return:
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    print('Saving model plots...')
    df_concat = df_concat.drop(columns=['id'])
    model_list = list(df_concat['Model'])
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
                                  '. Plotting and averaging across non-missing values. Checking output is recommended...'))
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
        out_path_fig = "%s%s%s%s" % (op.dirname(op.dirname(net_pick_file)), '/', name, '_mean_plot.png')
        fig.savefig(out_path_fig)
        plt.close('all')
    return
