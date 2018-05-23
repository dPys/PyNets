# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np
import networkx as nx
import os


def plot_conn_mat(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, thr, node_size):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if mask:
        if network:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + network + '_' + str(os.path.basename(mask).split('.')[0]) + '_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_network.png'
        else:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + str(os.path.basename(mask).split('.')[0]) + '_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.png'
    else:
        if network:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + network + '_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_network.png'
        else:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.png'

    rois_num=conn_matrix.shape[0]
    plt.figure(figsize=(10, 10))
    [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
    plt.imshow(conn_matrix, interpolation="nearest", vmax=z_max, vmin=z_min, cmap=plt.cm.RdBu_r)
    ##And display the labels
    if rois_num < 50:
        if all(isinstance(item, int) for item in label_names)==False:
            plt.xticks(range(len(label_names)), label_names, size='x-small', rotation=90)
            plt.yticks(range(len(label_names)), label_names, size='x-small')
        else:
            plt.xticks(range(rois_num), rotation=90)
            plt.yticks(range(rois_num))
    plt.grid(False)
    plt.savefig(out_path_fig)
    plt.close()
    return

def plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names):
    import json
    from networkx.readwrite import json_graph
    from pathlib import Path
    from pynets.thresholding import normalize
    from pynets.netstats import most_important
    from scipy.cluster.hierarchy import linkage, fcluster
    from nipype.utils.filemanip import save_json

    ##Advanced Settings
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
    ##Advanced Settings

    conn_matrix = normalize(conn_matrix)
    G=nx.from_numpy_matrix(conn_matrix)
    if pruned == True:
        [G, pruned_nodes, pruned_edges] = most_important(G)
        conn_matrix = nx.to_numpy_array(G)

        pruned_nodes.sort(reverse = True)
        for j in pruned_nodes:
            del label_names[label_names.index(label_names[j])]

        pruned_edges.sort(reverse = True)
        for j in pruned_edges:
            del label_names[label_names.index(label_names[j])]

    def doClust(X, clust_levels):
        ##get the linkage diagram
        Z = linkage(X, 'ward', )
        ##choose # cluster levels
        cluster_levels = range(1,int(clust_levels))
        ##init array to store labels for each level
        clust_levels_tmp = int(clust_levels) - 1
        label_arr = np.zeros((int(clust_levels_tmp),int(X.shape[0])))
        ##iterate thru levels
        for c in cluster_levels:
            fl = fcluster(Z,c,criterion='maxclust')
            #print(fl)
            label_arr[c-1, :] = fl
        return label_arr, clust_levels_tmp

    if comm == 'nodes' and len(conn_matrix) > 40:
        from pynets.netstats import modularity_louvain_dir
        if len(conn_matrix) < 50:
            gamma=0.00001
        elif len(conn_matrix) < 100:
            gamma=0.0001
        elif len(conn_matrix) < 200:
            gamma=0.001
        elif len(conn_matrix) < 500:
            gamma=0.01
        elif len(conn_matrix) < 1000:
            gamma=0.5
        else:
            gamma=1

        [node_comm_aff_mat, q] = modularity_louvain_dir(conn_matrix, hierarchy=True, gamma=gamma)
        print('Found ' + str(len(np.unique(node_comm_aff_mat))) + ' communities with gamma=' + str(gamma) + '...')
        clust_levels = len(node_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([node_comm_aff_mat == 0]).astype('int'))
        label_arr = node_comm_aff_mat * np.expand_dims(np.arange(1,clust_levels+1),axis=1) + mask_mat
    elif comm == 'links' and len(conn_matrix) > 40:
        from pynets.netstats import link_communities
        ##Plot link communities
        link_comm_aff_mat = link_communities(conn_matrix, type_clustering='single')
        print('Found ' + str(len(link_comm_aff_mat)) + ' communities...')
        clust_levels = len(link_comm_aff_mat)
        clust_levels_tmp = int(clust_levels) - 1
        mask_mat = np.squeeze(np.array([link_comm_aff_mat == 0]).astype('int'))
        label_arr = link_comm_aff_mat * np.expand_dims(np.arange(1,clust_levels+1),axis=1) + mask_mat
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
        from collections import OrderedDict
        def write_roman(num):
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
        for i in [int(l) for i, l in enumerate(node_labels)]:
            rn_list.append(json.dumps(write_roman(i)))
        abet = rn_list
        return ".".join(["{}{}".format(abet[i],int(l)) for i, l in enumerate(node_labels)])+".{}".format(label_names[node_idx])

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
        json_file_name = str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.json'
        json_fdg_file_name = str(ID) + '_' + network + '_fdg_' + conn_model + '_network.json'
        connectogram_plot = dir_path + '/' + json_file_name
        fdg_js_sub = dir_path + '/' + str(ID) + '_' + network + '_fdg_' + conn_model + '_network.js'
        fdg_js_sub_name = str(ID) + '_' + network + '_fdg_' + conn_model + '_network.js'
        connectogram_js_sub = dir_path + '/' + str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.js'
        connectogram_js_name = str(ID) + '_' + network + '_connectogram_' + conn_model + '_network.js'
    else:
        json_file_name = str(ID) + '_connectogram_' + conn_model + '.json'
        json_fdg_file_name = str(ID) + '_fdg_' + conn_model + '.json'
        connectogram_plot = dir_path + '/' + json_file_name
        connectogram_js_sub = dir_path + '/' + str(ID) + '_connectogram_' + conn_model + '.js'
        fdg_js_sub = dir_path + '/' + str(ID) + '_fdg_' + conn_model + '.js'
        fdg_js_sub_name = str(ID) + '_fdg_' + conn_model + '.js'
        connectogram_js_name = str(ID) + '_connectogram_' + conn_model + '.js'
    save_json(connectogram_plot, output)

    ##Force-directed graphing
    G=nx.from_numpy_matrix(np.round(conn_matrix.astype('float64'),6))
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

    ##Add community structure
    for k in range(len(data['nodes'])):
        data['nodes'][k]['group'] = str(label_arr[0][k])

    ##Add node labels
    for k in range(len(data['nodes'])):
        data['nodes'][k]['name'] = str(label_names[k])

    out_file = str(dir_path + '/' + json_fdg_file_name)
    save_json(out_file, data)

    ##Copy index.html and json to dir_path
    #conn_js_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/connectogram.js'
    #index_html_path = '/Users/PSYC-dap3463/Applications/PyNets/pynets/index.html'
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


def plot_timeseries(time_series, network, ID, dir_path, atlas_select, labels):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    for time_serie, label in zip(time_series.T, labels):
        plt.plot(time_serie, label=label)
    plt.xlabel('Scan Number')
    plt.ylabel('Normalized Signal')
    plt.legend()
    #plt.tight_layout()
    if network:
        plt.title(network + ' Time Series')
        out_path_fig=dir_path + '/' + ID + '_' + network + '_TS_plot.png'
    else:
        plt.title('Time Series')
        out_path_fig=dir_path + '/' + ID + '_Whole_Brain_TS_plot.png'
    plt.savefig(out_path_fig)
    plt.close()


def plot_all(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, coords, thr, node_size, edge_threshold):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from nilearn import plotting as niplot
    pruning=True
    dpi_resolution=500
    import pkg_resources
    import networkx as nx
    from pynets import plotting
    from pynets.netstats import most_important
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    G_pre=nx.from_numpy_matrix(conn_matrix)
    if pruning == True:
        [G, pruned_nodes, pruned_edges] = most_important(G_pre)
    else:
        G = G_pre
    conn_matrix = nx.to_numpy_array(G)

    pruned_nodes.sort(reverse=True)
    for j in pruned_nodes:
        del label_names[label_names.index(label_names[j])]
        del coords[coords.index(coords[j])]

    pruned_edges.sort(reverse=True)
    for j in pruned_edges:
        del label_names[label_names.index(label_names[j])]
        del coords[coords.index(coords[j])]

    ##Plot connectogram
    if len(conn_matrix) > 20:
        try:
            plotting.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)
        except RuntimeError:
            print('\n\n\nError: Connectogram plotting failed!')
    else:
        print('Error: Cannot plot connectogram for graphs smaller than 20 x 20!')

    ##Plot adj. matrix based on determined inputs
    plotting.plot_conn_mat(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names, mask, thr, node_size)

    ##Plot connectome
    if mask:
        if network:
            out_path_fig=dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(network) + '_' + str(thr) + '_' + str(node_size) + '_functional_connectome_viz.png'
        else:
            out_path_fig=dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(thr) + '_' + str(node_size) + '_functional_connectome_viz.png'
        ##Save coords to pickle
        coord_path = "%s%s%s%s" % (dir_path, '/coords_', os.path.basename(mask).split('.')[0], '_plotting.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        net_parcels_map_nifti = None
        ##Save labels to pickle
        labels_path = "%s%s%s%s" % (dir_path, '/labelnames_', os.path.basename(mask).split('.')[0], '_plotting.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)
    else:
        if network:
            out_path_fig=dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(network) + '_' + str(thr) + '_' + str(node_size) + '_functional_connectome_viz.png'
        else:
            out_path_fig=dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_functional_connectome_viz.png'
        ##Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/coords_plotting.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        ##Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/labelnames_plotting.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)
    #niplot.plot_connectome(conn_matrix, coords, edge_threshold=edge_threshold, node_size=20, colorbar=True, output_file=out_path_fig)
    ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")
    connectome = niplot.plot_connectome(np.zeros(shape=(1, 1)), [(0, 0, 0)], black_bg=True, node_size=0.0001)
    connectome.add_overlay(ch2better_loc, alpha=0.4, cmap=plt.cm.gray)
    [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
    connectome.add_graph(conn_matrix, coords, edge_threshold=edge_threshold, black_bg=True, edge_cmap='Greens', edge_vmax=z_max, edge_vmin=z_min, node_size=4)
    connectome.savefig(out_path_fig, dpi=dpi_resolution)
    return

def structural_plotting(conn_matrix_symm, label_names, atlas_select, ID, bedpostx_dir, network, parc, mask, coords, dir_path, conn_model, thr, node_size):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    import seaborn as sns
    import pkg_resources
    from matplotlib import colors
    from nilearn import plotting as niplot
    from pynets.netstats import most_important
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    edge_threshold = 0.10
    connectome_fdt_thresh = 90
    dpi_resolution = 500
    #pruning = False

    ####Auto-set INPUTS####
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    if network:
        probtrackx_output_dir_path = dir_path + '/probtrackx_' + str(network)
    else:
        probtrackx_output_dir_path = dir_path + '/probtrackx_WB'
    ####Auto-set INPUTS####
    if parc is True:
        node_size = 'parc'

    # G_pre=nx.from_numpy_matrix(conn_matrix_symm)
    # if pruning is True:
    #     [G, pruned_nodes, pruned_edges] = most_important(G_pre)
    # else:
    #     G = G_pre
    # conn_matrix = nx.to_numpy_array(G)
    #
    # pruned_nodes.sort(reverse=True)
    # for j in pruned_nodes:
    #     del label_names[label_names.index(label_names[j])]
    #     del coords[coords.index(coords[j])]
    #
    # pruned_edges.sort(reverse=True)
    # for j in pruned_edges:
    #     del label_names[label_names.index(label_names[j])]
    #     del coords[coords.index(coords[j])]

    plt.figure(figsize=(10, 10))
    plt.imshow(conn_matrix_symm, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    if label_names:
        plt.xticks(range(len(label_names)), label_names, size='xx-small', rotation=90)
        plt.yticks(range(len(label_names)), label_names, size='xx-small')
    else:
        plt.xticks(range(conn_matrix_symm.shape[0]), rotation=90)
        plt.yticks(range(conn_matrix_symm.shape[0]))
    plt_title = str(os.path.basename(dir_path)) + ' Structural Connectivity of: ' + str(ID)
    plt.title(plt_title)
    plt.grid(False)
    #plt.gcf().subplots_adjust(left=0.8)

    if mask:
        if network:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + network + '_' + str(os.path.basename(mask).split('.')[0]) + '_struct_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_network.png'
        else:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + str(os.path.basename(mask).split('.')[0]) + '_struct_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.png'
    else:
        if network:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_' + network + '_struct_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_network.png'
        else:
            out_path_fig=dir_path + '/' + str(ID) + '_' + str(atlas_select) + '_struct_adj_mat_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '.png'

    plt.savefig(out_path_fig, dpi=dpi_resolution)
    plt.close()

    ##Prepare glass brain figure
    fdt_paths_loc = probtrackx_output_dir_path + '/fdt_paths.nii.gz'

    ##Create transform matrix between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    flirt.inputs.in_file = nodif_brain_mask_path
    flirt.inputs.out_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
    flirt.inputs.out_file = '/tmp/out_flirt.nii.gz'
    flirt.run()

    ##Apply transform between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    flirt.inputs.in_file = nodif_brain_mask_path
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
    flirt.inputs.out_file = bedpostx_dir + '/xfms/diff2MNI_affine.nii.gz'
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.run()

    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    flirt.inputs.in_file = fdt_paths_loc
    out_file_MNI = fdt_paths_loc.split('.nii')[0] + '_MNI.nii.gz'
    flirt.inputs.out_file = out_file_MNI
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
    flirt.run()

    fdt_paths_MNI_loc = probtrackx_output_dir_path + '/fdt_paths_MNI.nii.gz'

    colors.Normalize(vmin=-1, vmax=1)
    clust_pal = sns.color_palette("Blues_r", 4)
    clust_colors = colors.to_rgba_array(clust_pal)

    ##Plotting with glass brain
    ch2better_loc = pkg_resources.resource_filename("pynets", "templates/ch2better.nii.gz")
    connectome = niplot.plot_connectome(np.zeros(shape=(1,1)), [(0,0,0)], black_bg=False, node_size=0.0001)
    connectome.add_overlay(ch2better_loc, alpha=0.5, cmap=plt.cm.gray)
    [z_min, z_max] = -np.abs(conn_matrix_symm).max(), np.abs(conn_matrix_symm).max()
    connectome.add_graph(conn_matrix_symm, coords, edge_threshold=edge_threshold, node_color=clust_colors, edge_cmap=plt.cm.binary, edge_vmax=z_max, edge_vmin=z_min, node_size=4)
    #connectome.add_overlay(img=fdt_paths_MNI_loc, threshold=connectome_fdt_thresh, cmap=niplot.cm.cyan_copper_r, alpha=0.6)

    ##Plot connectome
    if mask:
        if network:
            out_path_fig = dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(network) + '_' + str(thr) + '_' + str(node_size) + '_struct_viz.png'
        else:
            out_path_fig = dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(os.path.basename(mask).split('.')[0]) + '_' + str(thr) + '_' + str(node_size) + '_struct_viz.png'
        ##Save coords to pickle
        coord_path = "%s%s%s%s" % (dir_path, '/struct_coords_', os.path.basename(mask).split('.')[0], '_plotting.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        net_parcels_map_nifti = None
        ##Save labels to pickle
        labels_path = "%s%s%s%s" % (dir_path, '/struct_labelnames_', os.path.basename(mask).split('.')[0], '_plotting.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)
    else:
        if network:
            out_path_fig = dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(network) + '_' + str(thr) + '_' + str(node_size) + '_struct_viz.png'
        else:
            out_path_fig = dir_path + '/' + ID + '_' + str(atlas_select) + '_' + str(conn_model) + '_' + str(thr) + '_' + str(node_size) + '_struct_viz.png'
        ##Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/struct_coords_plotting.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        ##Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/struct_labelnames_plotting.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)
    connectome.savefig(out_path_fig, dpi=dpi_resolution)
    connectome.close()
    return
