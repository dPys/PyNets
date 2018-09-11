#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import nibabel as nib
import numpy as np
from pynets.netstats import extractnetstats
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface

nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = 'auto'


def get_file():
    base_path = str(__file__)
    return base_path


# Save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, network, mask):
    import pandas as pd
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    # Check for existence of csv_loc
    if os.path.isfile(csv_loc) is False:
        raise FileNotFoundError('\n\n\nERROR: Missing netmetrics csv file output. Cannot export to pandas df!')

    if mask is not None:
        if network is not None:
            met_list_picke_path = "%s%s%s%s%s" % (os.path.dirname(os.path.abspath(csv_loc)), '/net_metric_list_', network, '_', str(os.path.basename(mask).split('.')[0]))
        else:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(csv_loc)), '/net_metric_list_', str(os.path.basename(mask).split('.')[0]))
    else:
        if network is not None:
            met_list_picke_path = "%s%s%s" % (os.path.dirname(os.path.abspath(csv_loc)), '/net_metric_list_', network)
        else:
            met_list_picke_path = "%s%s" % (os.path.dirname(os.path.abspath(csv_loc)), '/net_metric_list')

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
    dir_path = "%s%s%s" % (os.path.dirname(os.path.realpath(in_file)), '/', atlas_select)
    if not os.path.exists(dir_path) and atlas_select is not None:
        os.makedirs(dir_path)
    elif atlas_select is None:
        raise ValueError("Error: cannot create directory for a null atlas!")
    return dir_path


def create_est_path(ID, network, conn_model, thr, mask, dir_path, node_size, smooth, thr_type):
    if mask is not None:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model), '_', str(thr), thr_type, '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_', str(thr), thr_type, '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.npy')
    else:
        if network is not None:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model), '_', str(thr), thr_type, '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.npy')
        else:
            est_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_', str(thr), thr_type, '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.npy')
    return est_path


def create_unthr_path(ID, network, conn_model, mask, dir_path):
    if mask is not None:
        if network is not None:
            unthr_path = "%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model), '_', str(os.path.basename(mask).split('.')[0]), '_unthresh_mat.npy')
        else:
            unthr_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_', str(os.path.basename(mask).split('.')[0]), '_unthresh_mat.npy')
    else:
        if network is not None:
            unthr_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_est_', str(conn_model), '_unthresholded_mat.npy')
        else:
            unthr_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_est_', str(conn_model), '_unthresh_mat.npy')
    return unthr_path


def create_csv_path(ID, network, conn_model, thr, mask, dir_path, node_size, smooth):
    if mask is not None:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_net_metrics_', conn_model, '_', str(thr), '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.csv')
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_net_metrics_', conn_model, '_', str(thr), '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.csv')
    else:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_', network, '_net_metrics_', conn_model, '_', str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.csv')
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_net_metrics_', conn_model, '_', str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'), '.csv')
    return out_path


def nil_parcellate(func_file, clust_mask, k, clust_type, ID, dir_path, uatlas_select):
    import time
    import nibabel as nib
    from nilearn.regions import Parcellations
    detrending = True

    start = time.time()
    func_img = nib.load(func_file)
    mask_img = nib.load(clust_mask)
    clust_est = Parcellations(method=clust_type, detrend=detrending, n_parcels=int(k),
                              mask=mask_img)
    clust_est.fit(func_img)
    nib.save(clust_est.labels_img_, uatlas_select)
    print("%s%s%s" % (clust_type, k, " clusters: %.2fs" % (time.time() - start)))
    return


def individual_tcorr_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5):
    import os
    from pynets import utils
    nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']

    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    atlas_select = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', str(k))
    print("%s%s%s%s%s%s%s" % ('\nCreating atlas using ', clust_type, ' at cluster level ', str(k),
                              ' for ', str(atlas_select), '...\n'))
    dir_path = utils.do_dir_path(atlas_select, func_file)
    uatlas_select = "%s%s%s%s%s%s%s%s" % (dir_path, '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz')

    if clust_type in nilearn_clust_list:
        utils.nil_parcellate(func_file, clust_mask, k, clust_type, ID, dir_path, uatlas_select)
    elif clust_type == 'ncut':
        from pynets.clustools import make_image_from_bin_renum, binfile_parcellate, make_local_connectivity_tcorr
        working_dir = "%s%s%s" % (os.path.dirname(func_file), '/', atlas_select)
        outfile = "%s%s%s%s" % (working_dir, '/rm_tcorr_conn_', str(ID), '.npy')
        outfile_parc = "%s%s%s" % (working_dir, '/rm_tcorr_indiv_cluster_', str(ID))
        binfile = "%s%s%s%s%s%s" % (working_dir, '/rm_tcorr_indiv_cluster_', str(ID), '_', str(k), '.npy')
        make_local_connectivity_tcorr(func_file, clust_mask, outfile, thresh)
        binfile_parcellate(outfile, outfile_parc, int(k))
        make_image_from_bin_renum(uatlas_select, binfile, clust_mask)

    clustering = True
    return uatlas_select, atlas_select, clustering, clust_mask, k, clust_type


def assemble_mt_path(ID, input_file, atlas_select, network, conn_model, thr, mask, node_size, smooth):
    #nilearn_parc_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    #nilearn_coord_atlases=['harvard_oxford', 'msdl', 'coords_power_2011', 'smith_2009', 'basc_multiscale_2015', 'allen_2011', 'coords_dosenbach_2010']
    if conn_model == 'prob':
        ID_dir = str(os.path.dirname(input_file))
    else:
        ID_dir = str(os.path.dirname(input_file).split('.')[0])
    if mask is not None:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (ID_dir, '/', str(atlas_select), '/', str(ID), '_', network, '_net_metrics_', conn_model, '_', str(thr), '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'))
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (ID_dir, '/', str(atlas_select), '/', str(ID), '_net_metrics_', conn_model, '_', str(thr), '_', str(os.path.basename(mask).split('.')[0]), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'))
    else:
        if network is not None:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (ID_dir, '/', str(atlas_select), '/', str(ID), '_', network, '_net_metrics_', conn_model, '_', str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'))
        else:
            out_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (ID_dir, '/', str(atlas_select), '/', str(ID), '_net_metrics_', conn_model, '_', str(thr), '_', str(node_size), '%s' % ("mm_" if node_size != 'parc' else "_"), "%s" % ("%s%s" % (smooth, 'fwhm') if float(smooth) > 0 else 'nosm'))
    return out_path


def pass_meta_outs(conn_model, est_path, network, node_size, smooth, thr, prune, ID, mask):
    est_path_iterlist = est_path
    conn_model_iterlist = conn_model
    network_iterlist = network
    node_size_iterlist = node_size
    smooth_iterlist = smooth
    thr_iterlist = thr
    prune_iterlist = prune
    ID_iterlist = ID
    mask_iterlist = mask
    # print('\n\nParam-iters:\n')
    # print(conn_model_iterlist)
    # print(est_path_iterlist)
    # print(network_iterlist)
    # print(node_size_iterlist)
    # print(smooth_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(mask_iterlist)
    # print('\n\n')
    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, smooth_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, mask_iterlist


def collect_pandas_join(net_pickle_mt):
    net_pickle_mt_out = net_pickle_mt
    return net_pickle_mt_out


def flatten(l):
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            for ell in flatten(el):
                yield ell
        else:
            yield el


def collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch):
    import pandas as pd
    import numpy as np
    import os
    import matplotlib
    matplotlib.use('Agg')
    from itertools import chain

    # Check for existence of net_pickle files, condensing final list to only those that were actually produced.
    net_pickle_mt_list_exist = []
    for net_pickle_mt in list(net_pickle_mt_list):
        if os.path.isfile(net_pickle_mt) is True:
            net_pickle_mt_list_exist.append(net_pickle_mt)

    if len(list(net_pickle_mt_list)) > len(net_pickle_mt_list_exist):
        raise UserWarning('Warning! Number of actual models produced less than expected. Some graphs were excluded')

    net_pickle_mt_list = net_pickle_mt_list_exist

    if len(net_pickle_mt_list) > 1:
        print("%s%s%s" % ('\n\nList of result files to concatenate:\n', str(net_pickle_mt_list), '\n\n'))
        subject_path = os.path.dirname(os.path.dirname(net_pickle_mt_list[0]))
        name_of_network_pickle = "%s%s" % ('net_metrics_', net_pickle_mt_list[0].split('_0.')[0].split('net_metrics_')[1])
        net_pickle_mt_list.sort()

        list_ = []
        models = []
        for file_ in net_pickle_mt_list:
            df = pd.read_pickle(file_)
            try:
                node_cols = [s for s in list(df.columns) if isinstance(s, int) or any(c.isdigit() for c in s)]
                df = df.drop(node_cols, axis=1)
                models.append(os.path.basename(file_))
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
            weighted_means = []
            for measure in measures:
                if measure in df_concatted_std.columns:
                    std_val = float(df_concatted_std[measure])
                    weighted_means.append((df_concat.loc[:, measures][measure] *
                                           np.array(1/(df_concat.loc[:, measures][measure])/std_val)).sum() /
                                          np.sum(np.array(1/(df_concat.loc[:, measures][measure])/std_val)))
                else:
                    weighted_means.append(np.nan)
            df_concatted_weight_means = pd.DataFrame(weighted_means).transpose()
            df_concatted_weight_means.columns = [str(col) + '_weighted_mean' for col in measures]
            df_concatted_std.columns = [str(col) + '_std_dev' for col in df_concatted_std.columns]
            result = pd.concat([df_concatted, df_concatted_std, df_concatted_weight_means], axis=1)
            df_concatted_final = result.reindex(sorted(result.columns), axis=1)
            print('\nConcatenating dataframes for ' + str(ID) + '...\n')
            if network:
                net_pick_out_path = "%s%s%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle, '_', network, '_mean')
            else:
                net_pick_out_path = "%s%s%s%s%s%s" % (subject_path, '/', str(ID), '_', name_of_network_pickle, '_mean')
            df_concatted_final.to_pickle(net_pick_out_path)
            df_concatted_final.to_csv(net_pick_out_path + '.csv', index=False)
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


def list_first_mems(est_path, network, thr, dir_path, node_size, smooth):
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
    print('\n\n\n\n')
    return est_path, network, thr, dir_path, node_size, smooth


def check_est_path_existence(est_path_list):
    import os
    est_path_list_ex = []
    bad_ixs = []
    i = -1

    for est_path in est_path_list:
        i = i + 1
        if os.path.isfile(est_path) is True:
            est_path_list_ex.append(est_path)
        else:
            print("%s%s%s" % ('\n\nWarning: Missing ', est_path, '...\n\n'))
            bad_ixs.append(i)
            continue
    return est_path_list_ex, bad_ixs


def save_RSN_coords_and_labels_to_pickle(coords, label_names, dir_path, network):
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


def save_nifti_parcels_map(ID, dir_path, mask, network, net_parcels_map_nifti):
    if mask:
        if network:
            net_parcels_nii_path = "%s%s%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_masked_', network, '_', str(os.path.basename(mask).split('.')[0]), '.nii.gz')
        else:
            net_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_masked_', str(os.path.basename(mask).split('.')[0]), '.nii.gz')
    else:
        if network:
            net_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', str(ID), '_parcels_', network, '.nii.gz')
        else:
            net_parcels_nii_path = "%s%s%s%s" % (dir_path, '/', str(ID), '_parcels.nii.gz')

    nib.save(net_parcels_map_nifti, net_parcels_nii_path)
    return


def cuberoot(x):
    return np.sign(x) * np.abs(x)**(1 / 3)


def save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes):
    import os
    # Save time series as txt file
    if mask is None:
        if network is not None:
            out_path_ts = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_rsn_net_ts.npy')
        else:
            out_path_ts = "%s%s%s%s" % (dir_path, '/', ID, '_wb_net_ts.npy')
    else:
        if network is not None:
            out_path_ts = "%s%s%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_', network, '_rsn_net_ts.npy')
        else:
            out_path_ts = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_', os.path.basename(mask).split('.')[0], '_wb_net_ts.npy')
    np.save(out_path_ts, ts_within_nodes)
    return


class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path = File(exists=True, mandatory=True, desc="")
    mask = traits.Any(mandatory=False)
    prune = traits.Any(mandatory=False)
    node_size = traits.Any(mandatory=False)
    smooth = traits.Any(mandatory=False)


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
            self.inputs.mask,
            self.inputs.prune,
            self.inputs.node_size,
            self.inputs.smooth)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_file': op.abspath(getattr(self, '_outpath'))}


class Export2PandasInputSpec(BaseInterfaceInputSpec):
    csv_loc = File(exists=True, mandatory=True, desc="")
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    mask = traits.Any(mandatory=False)


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
            self.inputs.mask)
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

