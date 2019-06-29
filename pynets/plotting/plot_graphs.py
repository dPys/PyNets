# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def plot_conn_mat(conn_matrix, labels, out_path_fig):
    """

    :param conn_matrix:
    :param labels:
    :param out_path_fig:
    :return:
    """
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    #from pynets import thresholding
    from nilearn.plotting import plot_matrix

    dpi_resolution = 300

    # conn_matrix = np.array(np.array(thresholding.autofix(conn_matrix)))
    [z_min, z_max] = -np.abs(conn_matrix).max(), np.abs(conn_matrix).max()
    rois_num = conn_matrix.shape[0]
    if rois_num < 100:
        try:
            plot_matrix(conn_matrix, figure=(10, 10), labels=labels, vmax=z_max*0.5, vmin=z_min*0.5, reorder=True,
                        auto_fit=True, grid=False, colorbar=False)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')
    else:
        try:
            plot_matrix(conn_matrix, figure=(10, 10), vmax=z_max*0.5, vmin=z_min*0.5, auto_fit=True, grid=False,
                        colorbar=False)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')
    plt.savefig(out_path_fig, dpi=dpi_resolution)
    plt.close()
    return


def plot_community_conn_mat(conn_matrix, labels, out_path_fig_comm, community_aff):
    """

    :param conn_matrix:
    :param labels:
    :param out_path_fig_comm:
    :param community_aff:
    :return:
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    matplotlib.use('agg')
    #from pynets import thresholding
    from nilearn.plotting import plot_matrix

    dpi_resolution = 300

    #conn_matrix = np.array(np.array(thresholding.autofix(conn_matrix)))
    sorting_array = sorted(range(len(community_aff)), key=lambda k: community_aff[k])
    sorted_conn_matrix = conn_matrix[sorting_array, :]
    sorted_conn_matrix = sorted_conn_matrix[:, sorting_array]
    [z_min, z_max] = -np.abs(sorted_conn_matrix).max(), np.abs(sorted_conn_matrix).max()
    rois_num = sorted_conn_matrix.shape[0]
    if rois_num < 100:
        try:
            plot_matrix(conn_matrix, figure=(10, 10), labels=labels, vmax=z_max, vmin=z_min,
                        reorder=False, auto_fit=True, grid=False, colorbar=False)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')
    else:
        try:
            plot_matrix(conn_matrix, figure=(10, 10), vmax=z_max, vmin=z_min, auto_fit=True, grid=False, colorbar=False)
        except RuntimeWarning:
            print('Connectivity matrix too sparse for plotting...')

    ax = plt.gca()
    total_size = 0
    for community in np.unique(community_aff):
        size = sum(sorted(community_aff) == community)
        ax.add_patch(patches.Rectangle(
                (total_size, total_size),
                size,
                size,
                fill=False,
                edgecolor='black',
                alpha=None,
                linewidth=1
            )
        )
        total_size += size

    plt.savefig(out_path_fig_comm, dpi=dpi_resolution)
    plt.close()
    return


def plot_conn_mat_func(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, thr, node_size, smooth, c_boot, hpass):
    """

    :param conn_matrix:
    :param conn_model:
    :param atlas:
    :param dir_path:
    :param ID:
    :param network:
    :param labels:
    :param roi:
    :param thr:
    :param node_size:
    :param smooth:
    :param c_boot:
    :param hpass:
    :return:
    """
    import networkx as nx
    import os.path as op
    import community
    from pynets.plotting import plot_graphs

    out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', atlas,
                                                           '%s' % ("%s%s%s" % ('_', network, '_') if network else "_"),
                                                           '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                           'func_adj_mat_', conn_model, '_', thr, '_', node_size,
                                                           '%s' % ("mm_" if node_size != 'parc' else "_"),
                                                           '%s' % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'),
                                                           '%s' % ("%s%s" % (smooth, 'fwhm.png') if float(smooth) > 0 else ''),
                                                           '%s' % ("%s%s" % (hpass, 'Hz.png') if hpass is not None else '.png'))

    plot_graphs.plot_conn_mat(conn_matrix, labels, out_path_fig)

    # Plot community adj. matrix
    G = nx.from_numpy_matrix(conn_matrix)
    try:
        node_comm_aff_mat = community.best_partition(G)
        print("%s%s%s" % ('Found ', str(len(np.unique(node_comm_aff_mat))), ' communities...'))
        out_path_fig_comm = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', atlas,
                                                                    '%s' % ("%s%s%s" % ('_', network, '_') if network else "_"),
                                                                    '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                                    'func_adj_mat_comm_', conn_model, '_', thr, '_',
                                                                    node_size, '%s' % ("mm_" if node_size != 'parc' else "_"),
                                                                    '%s' % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'),
                                                                    '%s' % ("%s%s" % (smooth, 'fwhm.png') if float(smooth) > 0 else ''),
                                                                    '%s' % ("%s%s" % (hpass, 'Hz.png') if hpass is not None else '.png'))
        plot_graphs.plot_community_conn_mat(conn_matrix, labels, out_path_fig_comm, node_comm_aff_mat)
    except:
        print('\nWARNING: Louvain community detection failed. Cannot plot community matrix...')

    return


def plot_conn_mat_struct(conn_matrix, conn_model, atlas, dir_path, ID, network, labels, roi, thr, node_size, smooth, c_boot, hpass):
    """

    :param conn_matrix:
    :param conn_model:
    :param atlas:
    :param dir_path:
    :param ID:
    :param network:
    :param labels:
    :param roi:
    :param thr:
    :param node_size:
    :param smooth:
    :param c_boot:
    :param hpass:
    :return:
    """
    from pynets.plotting import plot_graphs
    import networkx as nx
    import community
    import os.path as op
    out_path_fig = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', atlas,
                                                           '%s' % ("%s%s%s" % ('_', network, '_') if network else "_"),
                                                           '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                           'struct_adj_mat_',
                                                           conn_model, '_', thr, '_', node_size,
                                                           '%s' % ("mm_" if node_size != 'parc' else "_"),
                                                           '%s' % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'),
                                                           '%s' % ("%s%s" % (smooth, 'fwhm.png') if float(smooth) > 0 else ''),
                                                           '%s' % ("%s%s" % (hpass, 'Hz.png') if hpass is not None else '.png'))
    plot_graphs.plot_conn_mat(conn_matrix, labels, out_path_fig)

    # Plot community adj. matrix
    G = nx.from_numpy_matrix(conn_matrix)
    try:
        node_comm_aff_mat = community.best_partition(G)
        print("%s%s%s" % ('Found ', str(len(np.unique(node_comm_aff_mat))), ' communities...'))
        out_path_fig_comm = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', atlas,
                                                                    '%s' % ("%s%s%s" % ('_', network, '_') if network else "_"),
                                                                    '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                                    'struct_adj_mat_comm_', conn_model, '_', thr, '_', node_size,
                                                                    '%s' % ("mm_" if node_size != 'parc' else "_"),
                                                                    '%s' % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else 'nb_'),
                                                                    '%s' % ("%s%s" % (smooth, 'fwhm.png') if float(smooth) > 0 else ''),
                                                                    '%s' % ("%s%s" % (hpass, 'Hz.png') if hpass is not None else '.png'))
        plot_graphs.plot_community_conn_mat(conn_matrix, labels, out_path_fig_comm, node_comm_aff_mat)
    except:
        print('\nWARNING: Louvain community detection failed. Cannot plot community matrix...')

    return
