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
np.warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def get_file():
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
            met_list_picke_path = "%s%s%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_met_list_', network, '_',
                                                  str(op.basename(roi).split('.')[0]))
        else:
            met_list_picke_path = "%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_met_list_',
                                              str(op.basename(roi).split('.')[0]))
    else:
        if network is not None:
            met_list_picke_path = "%s%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_met_list_', network)
        else:
            met_list_picke_path = "%s%s" % (op.dirname(op.abspath(csv_loc)), '/net_met_list')

    metric_list_names = pickle.load(open(met_list_picke_path, 'rb'))
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
    df = df.T
    column_headers = {k: v for k, v in enumerate(metric_list_names)}
    df = df.rename(columns=column_headers)
    df['id'] = range(1, len(df) + 1)
    cols = df.columns.tolist()
    ix = cols.index('id')
    cols_ID = cols[ix:ix + 1] + cols[:ix] + cols[ix + 1:]
    df = df[cols_ID]
    df['id'] = df['id'].astype('object')
    df.id = df.id.replace(1, ID)
    net_pickle_mt = csv_loc.split('.csv')[0]
    df.to_pickle(net_pickle_mt, protocol=2)
    return net_pickle_mt


def do_dir_path(atlas, in_file):
    """

    :param atlas:
    :param in_file:
    :return:
    """
    dir_path = "%s%s%s" % (op.dirname(op.realpath(in_file)), '/', atlas)
    if not op.exists(dir_path) and atlas is not None:
        os.makedirs(dir_path)
    elif atlas is None:
        raise ValueError("Error: cannot create directory for a null atlas!")
    return dir_path


def create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot, thr_type, hpass, parc):
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
    :param hpass:
    :return:
    """
    if (node_size is None) and (parc is True):
        node_size = 'parc'

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_',
                                                       '%s' % (network + '_' if network is not None else ''),
                                                       '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                       'est_', conn_model, '_', thr, thr_type, '_',
                                                       '%s' % ("%s%s" % (node_size, 'mm_') if node_size != 'parc' else ''),
                                                       "%s" % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else ''),
                                                       "%s" % ("%s%s" % (smooth, 'fwhm_') if float(smooth) > 0 else ''),
                                                       "%s" % ("%s%s" % (hpass, 'Hz') if hpass is not None else ''),
                                                       '.npy')

    return est_path


def create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples, track_type, thr_type,
                         parc):
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
    if (node_size is None) and (parc is True):
        node_size = 'parc'

    est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_',
                                                     '%s' % (network + '_' if network is not None else ''),
                                                     '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                     'est_', conn_model, '_', thr, thr_type, '_',
                                                     '%s' % ("%s%s" % (node_size, 'mm_') if node_size != 'parc' else ''),
                                                     "%s" % ("%s%s" % (int(target_samples), 'samples_') if float(target_samples) > 0 else ''),
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
    unthr_path = "%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', '%s' % (network + '_' if network is not None else ''),
                                         '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                         'est_', conn_model, '_raw_mat.npy')
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

    out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_net_mets_',
                                               '%s' % (network + '_' if network is not None else ''),
                                               '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                               conn_model, '_', thr, '_', node_size,
                                               '%s' % ("mm" if node_size != 'parc' else ''), '.csv')
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
                   prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist, embed=True,
                   multimodal=False):
    from pynets.utils import build_omnetome, flatten
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
    if embed is True:
        build_omnetome(list(flatten(est_path_iterlist)), list(flatten(ID_iterlist))[0], multimodal)

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


def pass_meta_ins_multi(conn_model_func, est_path_func, network_func, node_size_func, thr_func, prune_func, ID_func,
                        roi_func, norm_func, binary_func, conn_model_struct, est_path_struct, network_struct,
                        node_size_struct, thr_struct, prune_struct, ID_struct, roi_struct, norm_struct, binary_struct):
    est_path_iterlist = [est_path_func, est_path_struct]
    conn_model_iterlist = [conn_model_func, conn_model_struct]
    network_iterlist = [network_func, network_struct]
    node_size_iterlist = [node_size_func, node_size_struct]
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


def build_omnetome(est_path_iterlist, ID, multimodal):
    import os
    from pynets.utils import flatten
    from sklearn.feature_selection import VarianceThreshold
    from graspy.embed import OmnibusEmbed, ClassicalMDS
    """

    :param net_pickle_mt_lis:
    :return:
    """

    def omni_embed(pop_array):
        variance_threshold = VarianceThreshold(threshold=0.05)
        diags = np.array([np.triu(pop_array[i]) for i in range(len(pop_array))])
        diags_red = diags.reshape(diags.shape[0], diags.shape[1] * diags.shape[2])
        var_thr = variance_threshold.fit(diags_red.T)
        graphs_ix_keep = var_thr.get_support(indices=True)
        pop_array_red = [pop_array[i] for i in graphs_ix_keep]

        # Omnibus embedding -- random dot product graph (rdpg)
        print("%s%s%s" % ('Embedding ensemble for atlas: ', atlas, '...'))
        omni = OmnibusEmbed(check_lcc=False)
        try:
            omni_fit = omni.fit_transform(pop_array_red)
            mds = ClassicalMDS()
            mds_fit = mds.fit_transform(omni_fit)
        except:
            omni_fit = omni.fit_transform(pop_array)
            mds = ClassicalMDS()
            mds_fit = mds.fit_transform(omni_fit)

        # Transform omnibus tensor into dissimilarity feature
        dir_path = os.path.dirname(graph_path)
        out_path = "%s%s%s%s%s%s" % (dir_path, '/', list(flatten(ID))[0], '_omnetome_', atlas, '.npy')
        print('Saving...')
        np.save(out_path, mds_fit)
        del mds, mds_fit, omni, omni_fit
        return

    atlases = list(set([x.split('/')[-2].split('/')[0] for x in est_path_iterlist]))
    parcel_dict = dict.fromkeys(atlases)
    for key in parcel_dict:
        parcel_dict[key] = []

    func_models = ['corr', 'sps', 'cov', 'partcorr', 'QuicGraphicalLasso', 'QuicGraphicalLassoCV',
                   'QuicGraphicalLassoEBIC', 'AdaptiveQuicGraphicalLasso']

    struct_models = ['csa', 'tensor', 'csd']

    if multimodal is True:
        est_path_iterlist_dwi = list(set([i for i in est_path_iterlist if i.split('est_')[1].split('_')[0] in
                                          struct_models]))
        est_path_iterlist_func = list(set([i for i in est_path_iterlist if i.split('est_')[1].split('_')[0] in
                                           func_models]))

        for atlas in atlases:
            for graph_path in est_path_iterlist_dwi:
                if atlas in graph_path:
                    parcel_dict[atlas].append(graph_path)
            for graph_path in est_path_iterlist_func:
                if atlas in graph_path:
                    parcel_dict[atlas].append(graph_path)
            pop_array = []
            for graph in parcel_dict[atlas]:
                pop_array.append(np.load(graph))
            if len(pop_array) > 1:
                omni_embed(pop_array)
            else:
                print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                pass
    elif (multimodal is False) and (len(est_path_iterlist) > 1):
        for atlas in atlases:
            for graph_path in est_path_iterlist:
                if atlas in graph_path:
                    parcel_dict[atlas].append(graph_path)
            pop_array = []
            for graph in parcel_dict[atlas]:
                pop_array.append(np.load(graph))
            omni_embed(pop_array)
    else:
        raise RuntimeError('ERROR: Only one graph sampled, omnibus embedding not appropriate.')
    return


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
    import matplotlib
    matplotlib.use('Agg')
    from itertools import chain

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
        name_of_network_pickle = "%s%s" % ('net_mets_',
                                           net_pickle_mt_list[0].split('_0.')[0].split('net_mets_')[1])
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
            df_concat = pd.DataFrame(list(chain(*list_of_dicts)))
            df_concat["Model"] = np.array([i.replace('_net_mets', '') for i in models])
            measures = list(df_concat.columns)
            measures.remove('id')
            measures.remove('Model')
            if plot_switch is True:
                from pynets.plotting import plot_gen
                plot_gen.plot_graph_measure_hists(df_concat, measures, file_)
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
            df_concatted_final.to_csv("%s%s" % (net_pick_out_path, '.csv'), index=False)

        except RuntimeWarning:
            print("%s%s%s" % ('\nWARNING: DATAFRAME CONCATENATION FAILED FOR ', str(ID), '!\n'))
            pass
    else:
        if network is not None:
            print("%s%s%s%s%s" % ('\nSingle dataframe for the ', network, ' network for: ', ID, '\n'))
        else:
            print("%s%s%s" % ('\nSingle dataframe for: ', ID, '\n'))
        pass

    return


def collect_pandas_df(network, ID, net_pickle_mt_list, plot_switch, multi_nets, multimodal):
    """

    :param network:
    :param ID:
    :param net_pickle_mt_list:
    :param plot_switch:
    :param multi_nets:
    :return:
    """
    from pynets.utils import collect_pandas_df_make, flatten

    func_models = ['corr', 'sps', 'cov', 'partcorr', 'QuicGraphicalLasso', 'QuicGraphicalLassoCV',
                   'QuicGraphicalLassoEBIC', 'AdaptiveQuicGraphicalLasso']

    struct_models = ['csa', 'tensor', 'csd']

    net_pickle_mt_list = list(flatten(net_pickle_mt_list))

    if multi_nets is not None:
        net_pickle_mt_list_nets = net_pickle_mt_list
        for network in multi_nets:
            net_pickle_mt_list = list(set([i for i in net_pickle_mt_list_nets if network in i]))
            if multimodal is True:
                net_pickle_mt_list_dwi = list(set([i for i in net_pickle_mt_list if i.split('metrics_')[1].split('_')[0]
                                                   in struct_models]))
                collect_pandas_df_make(net_pickle_mt_list_dwi, ID, network, plot_switch)
                net_pickle_mt_list_func = list(set([i for i in net_pickle_mt_list if
                                                    i.split('metrics_')[1].split('_')[0] in func_models]))
                collect_pandas_df_make(net_pickle_mt_list_func, ID, network, plot_switch)
            else:
                collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)
    else:
        if multimodal is True:
            net_pickle_mt_list_dwi = list(set([i for i in net_pickle_mt_list if i.split('metrics_')[1].split('_')[0] in
                                               struct_models]))
            collect_pandas_df_make(net_pickle_mt_list_dwi, ID, network, plot_switch)
            net_pickle_mt_list_func = list(set([i for i in net_pickle_mt_list if i.split('metrics_')[1].split('_')[0]
                                                in func_models]))
            collect_pandas_df_make(net_pickle_mt_list_func, ID, network, plot_switch)
        else:
            collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)

    return


def list_first_mems(est_path, network, thr, dir_path, node_size, smooth, c_boot, hpass):
    """

    :param est_path:
    :param network:
    :param thr:
    :param dir_path:
    :param node_size:
    :param smooth:
    :param c_boot:
    :param hpass:
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
    print(hpass)
    print('\n\n\n\n')
    return est_path, network, thr, dir_path, node_size, smooth, c_boot, hpass


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


def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """

    :param coords:
    :param labels:
    :param dir_path:
    :param network:
    :return:
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    # Save coords to pickle
    coord_path = "%s%s%s%s" % (dir_path, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)
    return coord_path, labels_path


def save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti):
    import os.path as op
    """

    :param ID:
    :param dir_path:
    :param roi:
    :param network:
    :param net_parcels_map_nifti:
    :return:
    """

    net_parcels_nii_path = "%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_masked_',
                                               '%s' % (network + '_' if network is not None else ''),
                                               '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                               '.nii.gz')

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return net_parcels_nii_path


def cuberoot(x):
    """

    :param x:
    :return:
    """
    return np.sign(x) * np.abs(x) ** (1 / 3)


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
    out_path_ts = "%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', '%s' % (network + '_' if network is not None else ''),
                                        '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                        '%s' % ("%s%s" % (int(c_boot), 'nb_') if float(c_boot) > 0 else ''),
                                        'rsn_net_ts.npy')

    np.save(out_path_ts, ts_within_nodes)
    return out_path_ts


def timeseries_bootstrap(tseries, block_size):
    # """
    # Generates a bootstrap sample derived from the input time-series.
    # Utilizes Circular-block-bootstrap method described in [1]_.
    # Parameters
    # ----------
    # tseries : array_like
    #     A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    # block_size : integer
    #     Size of the bootstrapped blocks
    # Returns
    # -------
    # bseries : array_like
    #     Bootstrap sample of the input timeseries
    # References
    # ----------
    # .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
    #    changes in brain connectivity for functional MRI. Statistica Sinica,
    #    special issue on Statistical Challenges and Advances in Brain Science,
    #    2008, 18: 1253-1268.
    # """
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
    # """
    # Normalizes b-vectors to be of unit length for the non-zero b-values. If the
    # b-value is 0, the vector is untouched.
    #
    # Positional Arguments:
    #         - bvec:
    #                 File name of the original b-vectors file
    #         - bvec_new:
    #                 File name of the new (normalized) b-vectors file. Must have
    #                 extension `.bvec`
    # """
    bv1 = np.array(np.loadtxt(bvec))
    # Enforce proper dimensions
    bv1 = bv1.T if bv1.shape[0] == 3 else bv1

    # Normalize values not close to norm 1
    bv2 = [b / np.linalg.norm(b) if not np.isclose(np.linalg.norm(b), 0)
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
    # """
    # Takes bval and bvec files and produces a structure in dipy format
    # **Positional Arguments:**
    # """
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
    if len(bvecs[np.where(np.logical_and(bvals > 50, np.all(abs(bvecs) == np.array([0, 0, 0]), axis=1)))]) > 0:
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
        cmds.append('fslroi {} {} {} 1'.format(dwi_file, b0_bbr, str(b0), ' 1'))
        b0s_bbr.append(b0_bbr)

    for cmd in cmds:
        os.system(cmd)

    # Get mean b0
    mean_b0 = mean_img(b0s_bbr)
    nib.save(mean_b0, nodif_b0)

    # Get mean b0 brain mask
    os.system("bet {} {} -m -f 0.2".format(nodif_b0, nodif_b0_bet))
    return gtab_file, nodif_b0_bet, nodif_b0_mask, dwi_file


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
    """
    Input interface wrapper for ExtractNetStats
    """
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
    """
    Output interface wrapper for ExtractNetStats
    """
    out_file = File()


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
            self.inputs.node_size,
            self.inputs.norm,
            self.inputs.binary)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(getattr(self, '_outpath'))}


class Export2PandasInputSpec(BaseInterfaceInputSpec):
    """
    Input interface wrapper for Export2Pandas
    """
    csv_loc = File(exists=True, mandatory=True, desc="")
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    roi = traits.Any(mandatory=False)


class Export2PandasOutputSpec(TraitedSpec):
    """
    Output interface wrapper for Export2Pandas
    """
    net_pickle_mt = traits.Any(mandatory=True)


class Export2Pandas(BaseInterface):
    """
    Interface wrapper for Export2Pandas
    """
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
    """
    Input interface wrapper for CollectPandasDfs
    """
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    net_pickle_mt_list = traits.List(mandatory=True)
    plot_switch = traits.Any(mandatory=True)
    multi_nets = traits.Any(mandatory=True)
    multimodal = traits.Any(mandatory=True)


class CollectPandasDfs(SimpleInterface):
    """
    Interface wrapper for CollectPandasDfs
    """
    input_spec = CollectPandasDfsInputSpec

    def _run_interface(self, runtime):
        collect_pandas_df(
            self.inputs.network,
            self.inputs.ID,
            self.inputs.net_pickle_mt_list,
            self.inputs.plot_switch,
            self.inputs.multi_nets,
            self.inputs.multimodal)
        return runtime
