#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import os.path as op
import nibabel as nib
import numpy as np
import shutil
from pynets.stats.netstats import extractnetstats
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface


def get_file():
    """

    :return:
    """
    base_path = str(__file__)
    return base_path


# Save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, network, roi):
    """

    :param csv_loc:
    :param ID:
    :param network:
    :param roi:
    :return:
    """
    import pandas as pd
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    # Check for existence of csv_loc
    if op.isfile(csv_loc) is False:
        raise FileNotFoundError('\nERROR: Missing netmetrics csv file output. Cannot export to pandas df!')

    if roi is not None:
        if network is not None:
            met_list_picke_path = "%s%s%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_metric_list_', network, '_',
                                                  str(op.basename(roi).split('.')[0]))
        else:
            met_list_picke_path = "%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_metric_list_',
                                              str(op.basename(roi).split('.')[0]))
    else:
        if network is not None:
            met_list_picke_path = "%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_metric_list_', network)
        else:
            met_list_picke_path = "%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_metric_list')

    metric_list_names = pickle.load(open(met_list_picke_path, 'rb'))
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
    df = df.T
    column_headers = {k: v for k, v in enumerate(metric_list_names)}
    df = df.rename(columns=column_headers)
    df['id'] = range(1, len(df) + 1)
    cols = df.columns.tolist()
    ix = cols.index('id')
    cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
    df = df[cols_ID]
    df['id'] = df['id'].astype('object')
    df.id = df.id.replace(1, ID)
    net_pickle_mt = csv_loc.split('.csv')[0]
    df.to_pickle(net_pickle_mt, protocol=2)
    return net_pickle_mt


def do_dir_path(atlas_select, in_file):
    """

    :param atlas_select:
    :param in_file:
    :return:
    """
    dir_path = "%s%s%s" % (op.dirname(op.realpath(in_file)), '/', atlas_select)
    if not op.exists(dir_path) and atlas_select is not None:
        os.makedirs(dir_path)
    elif atlas_select is None:
        raise ValueError("Error: cannot create directory for a null atlas!")
    return dir_path


def create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot, thr_type):
    """

    :param ID:
    :param network:
    :param conn_model:
    :param thr:
    :param roi:
    :param dir_path:
    :param node_size:
    :param smooth:
    :param c_boot:
    :param thr_type:
    :return:
    """
    if node_size is None:
        node_size = 'parc'
    if roi is not None:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_',
                                                                 str(conn_model), '_', str(thr), thr_type, '_',
                                                                 str(op.basename(roi).split('.')[0]), '_', str(node_size),
                                                                 '%s' % ("mm_" if node_size != 'parc' else ''),
                                                                 "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0
                                                                         else ''),
                                                                 "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0
                                                                         else ''),
                                                                 '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_',
                                                             str(thr), thr_type, '_', str(op.basename(roi).split('.')[0]),
                                                             '_', str(node_size), '%s' % ("mm_" if node_size != 'parc'
                                                                                          else ''),
                                                             "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0
                                                                     else ''),
                                                             "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0
                                                                     else ''),
                                                             '.npy')
    else:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model),
                                                             '_', str(thr), thr_type, '_', str(node_size),
                                                             '%s' % ("mm_" if node_size != 'parc' else ''),
                                                             "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0
                                                                     else ''),
                                                             "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0
                                                                     else ''), '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_', str(thr),
                                                         thr_type, '_', str(node_size),
                                                         '%s' % ("mm_" if node_size != 'parc' else ''),
                                                         "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0
                                                                 else ''),
                                                         "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0
                                                                 else ''), '.npy')
    return est_path


def create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples, track_type, thr_type):
    """

    :param ID:
    :param network:
    :param conn_model:
    :param thr:
    :param roi:
    :param dir_path:
    :param node_size:
    :param target_samples:
    :param track_type:
    :param thr_type:
    :return:
    """
    if node_size is None:
        node_size = 'parc'
    if roi is not None:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_',
                                                                 str(conn_model), '_', str(thr), thr_type, '_',
                                                                 str(op.basename(roi).split('.')[0]), '_', str(node_size),
                                                                 '%s' % ("mm_" if node_size != 'parc' else ''),
                                                                 "%s" % ("%s%s" % (int(target_samples), 'samples_') if
                                                                         float(target_samples) > 0 else ''),
                                                                 "%s%s" % (track_type, '_track'), '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_',
                                                             str(thr), thr_type, '_', str(op.basename(roi).split('.')[0]),
                                                             '_', str(node_size), '%s' % ("mm_" if node_size != 'parc'
                                                                                          else ''),
                                                             "%s" % ("%s%s" % (int(target_samples), 'samples_') if
                                                                     float(target_samples) > 0 else ''),
                                                             "%s%s" % (track_type, '_track'), '.npy')
    else:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model),
                                                             '_', str(thr), thr_type, '_', str(node_size),
                                                             '%s' % ("mm_" if node_size != 'parc' else ''),
                                                             "%s" % ("%s%s" % (int(target_samples), 'samples_') if
                                                                     float(target_samples) > 0 else ''),
                                                             "%s%s" % (track_type, '_track'), '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_', str(thr),
                                                         thr_type, '_', str(node_size),
                                                         '%s' % ("mm_" if node_size != 'parc' else ''),
                                                         "%s" % ("%s%s" % (int(target_samples), 'samples_') if
                                                                 float(target_samples) > 0 else ''),
                                                         "%s%s" % (track_type, '_track'), '.npy')
    return est_path


def create_unthr_path(ID, network, conn_model, roi, dir_path):
    """

    :param ID:
    :param network:
    :param conn_model:
    :param roi:
    :param dir_path:
    :return:
    """
    if roi is not None:
        if network is not None:
            unthr_path = "%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model), '_',
                                                   str(op.basename(roi).split('.')[0]), '_unthresh_mat.npy')
        else:
            unthr_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_',
                                               str(op.basename(roi).split('.')[0]), '_unthresh_mat.npy')
    else:
        if network is not None:
            unthr_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model),
                                               '_unthresholded_mat.npy')
        else:
            unthr_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_unthresh_mat.npy')
    return unthr_path


def create_csv_path(ID, network, conn_model, thr, roi, dir_path, node_size):
    """

    :param ID:
    :param network:
    :param conn_model:
    :param thr:
    :param roi:
    :param dir_path:
    :param node_size:
    :return:
    """
    if node_size is None:
        node_size = 'parc'
    if roi is not None:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_net_metrics_',
                                                           conn_model, '_', str(thr), '_',
                                                           str(op.basename(roi).split('.')[0]), '_', str(node_size),
                                                           '%s' % ("mm" if node_size != 'parc' else ''), '.csv')
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_net_metrics_', conn_model, '_', str(thr),
                                                       '_', str(op.basename(roi).split('.')[0]), '_', str(node_size),
                                                       '%s' % ("mm" if node_size != 'parc' else ''), '.csv')
    else:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_net_metrics_', conn_model,
                                                       '_', str(thr), '_', str(node_size),
                                                       '%s' % ("mm" if node_size != 'parc' else ''), '.csv')
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_net_metrics_', conn_model, '_', str(thr),
                                                   '_', str(node_size), '%s' % ("mm" if node_size != 'parc' else ''),
                                                   '.csv')
    return out_path


def save_mat(conn_matrix, est_path, fmt='npy'):
    """

    :param conn_matrix:
    :param est_path:
    :param fmt:
    :return:
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


def pass_meta_outs(conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist,
                  prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist):
    """

    :param conn_model_iterlist:
    :param est_path_iterlist:
    :param network_iterlist:
    :param node_size_iterlist:
    :param thr_iterlist:
    :param prune_iterlist:
    :param ID_iterlist:
    :param roi_iterlist:
    :param norm_iterlist:
    :param binary_iterlist:
    :return:
    """
    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist


def pass_meta_ins(conn_model, est_path, network, node_size, thr, prune, ID, roi, norm, binary):
    """

    :param conn_model:
    :param est_path:
    :param network:
    :param node_size:
    :param thr:
    :param prune:
    :param ID:
    :param roi:
    :param norm:
    :param binary:
    :return:
    """
    est_path_iterlist = est_path
    conn_model_iterlist = conn_model
    network_iterlist = network
    node_size_iterlist = node_size
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
    # print(node_size_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(roi_iterlist)
    # print(norm_iterlist)
    # print(binary_iterlist)
    # print('\n\n')
    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist


def CollectPandasJoin(net_pickle_mt):
    """

    :param net_pickle_mt:
    :return:
    """
    net_pickle_mt_out = net_pickle_mt
    return net_pickle_mt_out


def flatten(l):
    """

    :param l:
    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def random_forest_ensemble(df_in):
    """

    :param df_in:
    :return:
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    #from sklearn.model_selection import cross_val_score
    from random import randint
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    df_train = df_in.drop(columns=['modularity', 'average_diversity_coefficient'])

    for column in list(df_train.columns):
        try:
            df_train[column] = (df_train[column].str.split()).apply(lambda x: float(x[0]))
        except:
            continue

    y = df_train.T[randint(1, len(df_train.T))]
    df_train_in = df_train.T
    #full_scores = cross_val_score(estimator, np.array(df_train_in), np.array(y), scoring='neg_mean_squared_error', cv=5)
    fit = estimator.fit(df_train_in, y)
    df = pd.DataFrame(fit.predict(df_train_in)).T
    df.columns = list(df_train.T.index)
    return df


def collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch):
    """

    :param net_pickle_mt_list:
    :param ID:
    :param network:
    :param plot_switch:
    :return:
    """
    import pandas as pd
    import numpy as np
    import os
    import matplotlib
    matplotlib.use('Agg')
    from itertools import chain
    rand_forest = False

    # Check for existence of net_pickle files, condensing final list to only those that were actually produced.
    net_pickle_mt_list_exist = []
    for net_pickle_mt in list(net_pickle_mt_list):
        if op.isfile(net_pickle_mt) is True:
            net_pickle_mt_list_exist.append(net_pickle_mt)

    if len(list(net_pickle_mt_list)) > len(net_pickle_mt_list_exist):
        raise UserWarning('Warning! Number of actual models produced less than expected. Some graphs were excluded')

    net_pickle_mt_list = net_pickle_mt_list_exist

    if len(net_pickle_mt_list) > 1:
        print("%s%s%s" % ('\n\nList of result files to concatenate:\n', str(net_pickle_mt_list), '\n\n'))
        subject_path = op.dirname(op.dirname(net_pickle_mt_list[0]))
        name_of_network_pickle = "%s%s" % ('net_metrics_', net_pickle_mt_list[0].split('_0.')[0].split('net_metrics_')[1])
        net_pickle_mt_list.sort()

        list_ = []
        models = []
        for file_ in net_pickle_mt_list:
            df = pd.read_pickle(file_)
            try:
                node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]
                df = df.drop(node_cols, axis=1)
                models.append(op.basename(file_))
            except RuntimeError:
                print('Error: Node column removal failed for mean stats file...')
            list_.append(df)

        try:
            # Concatenate and find mean across dataframes
            list_of_dicts = [cur_df.T.to_dict().values() for cur_df in list_]
            #df_concat = pd.concat(list_, axis=1)
            df_concat = pd.DataFrame(list(chain(*list_of_dicts)))
            df_concat["Model"] = np.array([i.replace('_net_metrics', '') for i in models])
            measures = list(df_concat.columns)
            measures.remove('id')
            measures.remove('Model')
            if plot_switch is True:
                from pynets import plotting
                plotting.plot_graph_measure_hists(df_concat, measures, file_)
            df_concatted = df_concat.loc[:, measures].mean().to_frame().transpose()
            df_concatted_std = df_concat.loc[:, measures].std().to_frame().transpose()
            df_concatted.columns = [str(col) + '_mean' for col in df_concatted.columns]
            df_concatted_std.columns = [str(col) + '_std_dev' for col in df_concatted_std.columns]
            result = pd.concat([df_concatted, df_concatted_std], axis=1)
            df_concatted_final = result.reindex(sorted(result.columns), axis=1)
            print('\nConcatenating dataframes for ' + str(ID) + '...\n')
            if network:
                net_pick_out_path = "%s%s%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle, '_',
                                                          network, '_mean')
            else:
                net_pick_out_path = "%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle, '_mean')
            df_concatted_final.to_pickle(net_pick_out_path)
            df_concatted_final.to_csv(net_pick_out_path + '.csv', index=False)

            if rand_forest is True:
                df_rnd_forest = random_forest_ensemble(df_concat.loc[:, measures])
                if network:
                    net_pick_out_path = "%s%s%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle,
                                                              '_', network, '_rand_forest')
                else:
                    net_pick_out_path = "%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle,
                                                          '_rand_forest')
                df_rnd_forest.to_pickle(net_pick_out_path)
                df_rnd_forest.to_csv(net_pick_out_path + '.csv', index=False)

        except RuntimeWarning:
            print("%s%s%s" % ('\nWARNING: DATAFRAME CONCATENATION FAILED FOR ', str(ID), '!\n'))
            pass
    else:
        if network is not None:
            print("%s%s%s" % ('\nSingle dataframe for the ' + network + ' network for: ', str(ID), '\n'))
        else:
            print("%s%s%s" % ('\nSingle dataframe for: ', str(ID), '\n'))
        pass

    return


def collect_pandas_df(network, ID, net_pickle_mt_list, plot_switch, multi_nets):
    """

    :param network:
    :param ID:
    :param net_pickle_mt_list:
    :param plot_switch:
    :param multi_nets:
    :return:
    """
    from pynets.utils import collect_pandas_df_make, flatten

    net_pickle_mt_list = list(flatten(net_pickle_mt_list))

    if multi_nets is not None:
        net_pickle_mt_list_nets = net_pickle_mt_list
        for network in multi_nets:
            net_pickle_mt_list = list(set([i for i in net_pickle_mt_list_nets if network in i]))
            collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)
    else:
        collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)

    return


def list_first_mems(est_path, network, thr, dir_path, node_size, smooth, c_boot):
    """

    :param est_path:
    :param network:
    :param thr:
    :param dir_path:
    :param node_size:
    :param smooth:
    :param c_boot:
    :return:
    """
    est_path = est_path[0]
    network = network[0]
    thr = thr[0]
    dir_path = dir_path[0]
    node_size = node_size[0]
    print('\n\n\n\n')
    print(est_path)
    print(network)
    print(thr)
    print(dir_path)
    print(node_size)
    print(smooth)
    print(c_boot)
    print('\n\n\n\n')
    return est_path, network, thr, dir_path, node_size, smooth, c_boot


def check_est_path_existence(est_path_list):
    """

    :param est_path_list:
    :return:
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


def save_RSN_coords_and_labels_to_pickle(coords, label_names, dir_path, network):
    """

    :param coords:
    :param label_names:
    :param dir_path:
    :param network:
    :return:
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    # Save coords to pickle
    coord_path = "%s%s%s%s" % (dir_path, '/', network, '_func_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/', network, '_func_labelnames_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f, protocol=2)
    return


def save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti):
    """

    :param ID:
    :param dir_path:
    :param roi:
    :param network:
    :param net_parcels_map_nifti:
    :return:
    """
    if roi:
        if network:
            net_parcels_nii_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_masked_', network, '_',
                                                         str(op.basename(roi).split('.')[0]), '.nii.gz')
        else:
            net_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_masked_',
                                                     str(op.basename(roi).split('.')[0]), '.nii.gz')
    else:
        if network:
            net_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_', network, '.nii.gz')
        else:
            net_parcels_nii_path = "%s%s%s%s" % (dir_path, '/', str(ID), '_parcels.nii.gz')

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return net_parcels_nii_path


def cuberoot(x):
    """

    :param x:
    :return:
    """
    return np.sign(x) * np.abs(x)**(1 / 3)


def save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot):
    """

    :param roi:
    :param network:
    :param ID:
    :param dir_path:
    :param ts_within_nodes:
    :param c_boot:
    :return:
    """
    # Save time series as txt file
    if roi is None:
        if network is not None:
            out_path_ts = "%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, "%s" % ("%s%s" % (int(c_boot), 'nb_') if
                                                                                       float(c_boot) > 0 else ''),
                                              '_rsn_net_ts.npy')
        else:
            out_path_ts = "%s%s%s%s%s" % (dir_path, '/', ID, "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0
                                                                     else ''), '_wb_net_ts.npy')
    else:
        if network is not None:
            out_path_ts = "%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', op.basename(roi).split('.')[0], '_', network,
                                                  "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else ''),
                                                  '_rsn_net_ts.npy')
        else:
            out_path_ts = "%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', op.basename(roi).split('.')[0],
                                              "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else ''),
                                              '_wb_net_ts.npy')
    np.save(out_path_ts, ts_within_nodes)
    return


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

    Positional Arguments:
            - bvec:
                    File name of the original b-vectors file
            - bvec_new:
                    File name of the new (normalized) b-vectors file. Must have
                    extension `.bvec`
    """
    bv1 = np.array(np.loadtxt(bvec))
    # Enforce proper dimensions
    bv1 = bv1.T if bv1.shape[0] == 3 else bv1

    # Normalize values not close to norm 1
    bv2 = [b/np.linalg.norm(b) if not np.isclose(np.linalg.norm(b), 0)
           else b for b in bv1]
    np.savetxt(bvec_rescaled, bv2)
    return bvec_rescaled


def make_gtab_and_bmask(fbval, fbvec, dwi_file):
    """

    :param fbval:
    :param fbvec:
    :param dwi_file:
    :return:
    """
    import os
    from dipy.io import save_pickle
    import os.path as op
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from nilearn.image import mean_img
    from pynets.utils import rescale_bvec
    """
    Takes bval and bvec files and produces a structure in dipy format
    **Positional Arguments:**
    """
    # Use b0's from the DWI to create a more stable DWI image for registration
    outdir = op.dirname(dwi_file)

    nodif_b0 = "{}/nodif_b0.nii.gz".format(outdir)
    nodif_b0_bet = "{}/nodif_b0_bet.nii.gz".format(outdir)
    nodif_b0_mask = "{}/nodif_b0_bet_mask.nii.gz".format(outdir)
    bvec_rescaled = "{}/bvec_scaled.bvec".format(outdir)
    gtab_file = "{}/gtab.pkl".format(outdir)

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[np.where(np.any(abs(bvecs) >= 10, axis=1) == True)] = [1, 0, 0]
    bvecs[np.where(bvals == 0)] = 0
    if len(bvecs[np.where(np.logical_and(bvals > 50, np.all(abs(bvecs)==np.array([0, 0, 0]), axis=1)))]) > 0:
        raise ValueError('WARNING: Encountered potentially corrupted bval/bvecs. Check to ensure volumes with a '
                         'diffusion weighting are not being treated as B0\'s along the bvecs')
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
    gtab.b0_threshold = min(bvals)

    # Get b0 indices
    b0s = np.where(gtab.bvals == gtab.b0_threshold)[0]
    print("%s%s" % ('b0\'s found at: ', b0s))

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    # Extract and Combine all b0s collected
    print('Extracting b0\'s...')
    cmds = []
    b0s_bbr = []
    for b0 in b0s:
        print(b0)
        b0_bbr = "{}/{}_b0.nii.gz".format(outdir, str(b0))
        cmd = 'fslroi ' + dwi_file + ' ' + b0_bbr + ' ' + str(b0) + ' 1'
        cmds.append(cmd)
        b0s_bbr.append(b0_bbr)

    for cmd in cmds:
        os.system(cmd)

    # Get mean b0
    mean_b0 = mean_img(b0s_bbr)
    nib.save(mean_b0, nodif_b0)

    # Get mean b0 brain mask
    cmd = 'bet ' + nodif_b0 + ' ' + nodif_b0_bet + ' -m -f 0.2'
    os.system(cmd)
    return gtab_file, nodif_b0_bet, nodif_b0_mask, dwi_file


def check_orient_and_dims(infile, vox_size, bvecs=None):
    """

    :param infile:
    :param vox_size:
    :param bvecs:
    :return:
    """
    import os.path as op
    from pynets.utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        outfile = match_target_vox_res(infile, vox_size, outdir, sens='dwi')
    elif (vols > 1) and (bvecs is None):
        # func case
        infile = reorient_img(infile, outdir)
        # Check dimensions
        outfile = match_target_vox_res(infile, vox_size, outdir, sens='func')
    else:
        # t1w case
        infile = reorient_img(infile, outdir)
        # Check dimensions
        outfile = match_target_vox_res(infile, vox_size, outdir, sens='t1w')

    print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs


def reorient_dwi(dwi_prep, bvecs, out_dir):
    """

    :param dwi_prep:
    :param bvecs:
    :param out_dir:
    :return:
    """
    import shutil
    # Check orientation (dwi_prep)
    cmd = 'fslorient -getorient ' + dwi_prep
    orient = os.popen(cmd).read().strip('\n')
    dwi_orig = dwi_prep
    dwi_prep = "{}/{}_pre_reor.nii.gz".format(out_dir, dwi_prep.split('/')[-1].split('.nii.gz')[0])
    shutil.copyfile(dwi_orig, dwi_prep)
    bvecs_orig = bvecs
    bvecs = "{}/bvecs.bvec".format(out_dir)
    shutil.copyfile(bvecs_orig, bvecs)
    bvecs_mat = np.genfromtxt(bvecs)
    cmd = 'fslorient -getqform ' + dwi_prep
    qform = os.popen(cmd).read().strip('\n')
    reoriented = False
    if orient == 'NEUROLOGICAL':
        reoriented = True
        print('Neurological (dwi), reorienting to radiological...')
        # Orient dwi to RADIOLOGICAL
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            dwi_prep_PA = "{}/dwi_reor_PA.nii.gz".format(out_dir)
            print('Reorienting P-A flip (dwi)...')
            cmd = 'fslswapdim ' + dwi_prep + ' -x -y z ' + dwi_prep_PA
            os.system(cmd)
            bvecs_mat[:,1] = -bvecs_mat[:,1]
            cmd = 'fslorient -getqform ' + dwi_prep_PA
            qform = os.popen(cmd).read().strip('\n')
            dwi_prep = dwi_prep_PA
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            dwi_prep_IS = "{}/dwi_reor_IS.nii.gz".format(out_dir)
            print('Reorienting I-S flip (dwi)...')
            cmd = 'fslswapdim ' + dwi_prep + ' -x y -z ' + dwi_prep_IS
            os.system(cmd)
            bvecs_mat[:,2] = -bvecs_mat[:,2]
            dwi_prep = dwi_prep_IS
        bvecs_mat[:, 0] = -bvecs_mat[:, 0]
        cmd = 'fslorient -forceradiological ' + dwi_prep
        os.system(cmd)
        np.savetxt(bvecs, bvecs_mat)
    else:
        print('Radiological (dwi)...')
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            dwi_prep_PA = "{}/dwi_reor_PA.nii.gz".format(out_dir)
            print('Reorienting P-A flip (dwi)...')
            cmd = 'fslswapdim ' + dwi_prep + ' -x -y z ' + dwi_prep_PA
            os.system(cmd)
            bvecs_mat[:,1] = -bvecs_mat[:,1]
            cmd = 'fslorient -getqform ' + dwi_prep_PA
            qform = os.popen(cmd).read().strip('\n')
            dwi_prep = dwi_prep_PA
            reoriented = True
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            dwi_prep_IS = "{}/dwi_reor_IS.nii.gz".format(out_dir)
            print('Reorienting I-S flip (dwi)...')
            cmd = 'fslswapdim ' + dwi_prep + ' -x y -z ' + dwi_prep_IS
            os.system(cmd)
            bvecs_mat[:,2] = -bvecs_mat[:,2]
            dwi_prep = dwi_prep_IS
            reoriented = True
        np.savetxt(bvecs, bvecs_mat)

    if reoriented is True:
        imgg = nib.load(dwi_prep)
        data = imgg.get_fdata()
        affine = imgg.affine
        hdr = imgg.header
        imgg = nib.Nifti1Image(data, affine=affine, header=hdr)
        imgg.set_sform(affine)
        imgg.set_qform(affine)
        imgg.update_header()
        nib.save(imgg, dwi_prep)

        print('Reoriented affine: ')
        print(affine)
    else:
        dwi_prep = dwi_orig
        print('Image already in RAS+')

    return dwi_prep, bvecs


def reorient_img(img, out_dir):
    """

    :param img:
    :param out_dir:
    :return:
    """
    import shutil
    cmd = 'fslorient -getorient ' + img
    orient = os.popen(cmd).read().strip('\n')
    img_orig = img
    img = "{}/{}_pre_reor.nii.gz".format(out_dir, img.split('/')[-1].split('.nii.gz')[0])
    shutil.copyfile(img_orig, img)
    cmd = 'fslorient -getqform ' + img
    qform = os.popen(cmd).read().strip('\n')
    reoriented = False
    if orient == 'NEUROLOGICAL':
        reoriented = True
        print('Neurological (img), reorienting to radiological...')
        # Orient img to std
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            img_PA = "{}/img_reor_PA.nii.gz".format(out_dir)
            print('Reorienting P-A flip (img)...')
            cmd = 'fslswapdim ' + img + ' -x -y z ' + img_PA
            os.system(cmd)
            cmd = 'fslorient -getqform ' + img_PA
            qform = os.popen(cmd).read().strip('\n')
            img = img_PA
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            img_IS = "{}/img_reor_IS.nii.gz".format(out_dir)
            print('Reorienting I-S flip (img)...')
            cmd = 'fslswapdim ' + img + ' -x y -z ' + img_IS
            os.system(cmd)
            img = img_IS
        cmd = 'fslorient -forceradiological ' + img
        os.system(cmd)
    else:
        print('Radiological (img)...')
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            img_PA = "{}/img_reor_PA.nii.gz".format(out_dir)
            print('Reorienting P-A flip (img)...')
            cmd = 'fslswapdim ' + img + ' -x -y z ' + img_PA
            os.system(cmd)
            cmd = 'fslorient -getqform ' + img_PA
            qform = os.popen(cmd).read().strip('\n')
            img = img_PA
            reoriented = True
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            img_IS = "{}/img_reor_IS.nii.gz".format(out_dir)
            print('Reorienting I-S flip (img)...')
            cmd = 'fslswapdim ' + img + ' -x y -z ' + img_IS
            os.system(cmd)
            img = img_IS
            reoriented = True

    if reoriented is True:
        imgg = nib.load(img)
        data = imgg.get_fdata()
        affine = imgg.affine
        hdr = imgg.header
        imgg = nib.Nifti1Image(data, affine=affine, header=hdr)
        imgg.set_sform(affine)
        imgg.set_qform(affine)
        imgg.update_header()
        nib.save(imgg, img)

        print('Reoriented affine: ')
        print(affine)
    else:
        img = img_orig
        print('Image already in RAS+')

    return img


def match_target_vox_res(img_file, vox_size, out_dir, sens):
    """

    :param img_file:
    :param vox_size:
    :param out_dir:
    :param sens:
    :return:
    """
    from dipy.align.reslice import reslice
    # Check dimensions
    img = nib.load(img_file)
    data = img.get_fdata()
    affine = img.affine
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    if vox_size == '1mm':
        new_zooms = (1., 1., 1.)
    elif vox_size == '2mm':
        new_zooms = (2., 2., 2.)

    if (abs(zooms[0]), abs(zooms[1]), abs(zooms[2])) != new_zooms:
        print('Reslicing image ' + img_file + ' to ' + vox_size + '...')
        img_file_pre = "{}/{}_pre_res.nii.gz".format(out_dir, os.path.basename(img_file).split('.nii.gz')[0])
        shutil.copyfile(img_file, img_file_pre)

        data2, affine2 = reslice(data, affine, zooms, new_zooms)
        if sens == 'dwi':
            affine2[0:3,3] = np.zeros(3)
            affine2[0:3, 0:3] = np.eye(3) * np.array(new_zooms) * np.sign(affine2[0:3, 0:3])
        img2 = nib.Nifti1Image(data2, affine=affine2, header=hdr)
        img2.set_qform(affine2)
        img2.set_sform(affine2)
        img2.update_header()
        nib.save(img2, img_file)
        print('Resliced affine: ')
        print(nib.load(img_file).affine)
    else:
        if sens == 'dwi':
            affine[0:3,3] = np.zeros(3)
        img = nib.Nifti1Image(data, affine=affine, header=hdr)
        img.set_sform(affine)
        img.set_qform(affine)
        img.update_header()
        nib.save(img, img_file)

    return img_file


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


class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path = File(exists=True, mandatory=True, desc="")
    roi = traits.Any(mandatory=False)
    prune = traits.Any(mandatory=False)
    node_size = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=False)
    binary = traits.Any(mandatory=False)


class ExtractNetStatsOutputSpec(TraitedSpec):
    out_file = File()


class ExtractNetStats(BaseInterface):
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
            self.inputs.node_size,
            self.inputs.norm,
            self.inputs.binary)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(getattr(self, '_outpath'))}


class Export2PandasInputSpec(BaseInterfaceInputSpec):
    csv_loc = File(exists=True, mandatory=True, desc="")
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    roi = traits.Any(mandatory=False)


class Export2PandasOutputSpec(TraitedSpec):
    net_pickle_mt = traits.Any(mandatory=True)


class Export2Pandas(BaseInterface):
    input_spec = Export2PandasInputSpec
    output_spec = Export2PandasOutputSpec

    def _run_interface(self, runtime):
        out = export_to_pandas(
            self.inputs.csv_loc,
            self.inputs.ID,
            self.inputs.network,
            self.inputs.roi)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'net_pickle_mt': op.abspath(getattr(self, '_outpath'))}


class CollectPandasDfsInputSpec(BaseInterfaceInputSpec):
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    net_pickle_mt_list = traits.List(mandatory=True)
    plot_switch = traits.Any(mandatory=True)
    multi_nets = traits.Any(mandatory=True)


class CollectPandasDfs(SimpleInterface):
    input_spec = CollectPandasDfsInputSpec

    def _run_interface(self, runtime):
        collect_pandas_df(
            self.inputs.network,
            self.inputs.ID,
            self.inputs.net_pickle_mt_list,
            self.inputs.plot_switch,
            self.inputs.multi_nets)
        return runtime
