#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@authors: Derek Pisner
"""
import pandas as pd
import numpy as np
import itertools
import re
from sklearn.metrics.pairwise import (cosine_distances, haversine_distances, manhattan_distances,
                                      nan_euclidean_distances)
from sklearn.utils import check_X_y
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
warnings.filterwarnings("ignore")


def build_hp_dict(file_renamed, atlas, modality, hyperparam_dict, hyperparams):
    """
    A function to build a hyperparameter dictionary by parsing a given net_mets file path.
    """

    if 'atlas' not in hyperparam_dict.keys():
        hyperparam_dict['atlas'] = [atlas]
    else:
        hyperparam_dict['atlas'].append(atlas)

    for hyperparam in hyperparams:
        if hyperparam != 'smooth' and hyperparam != 'hpass' and hyperparam != 'track_type' and \
                hyperparam != 'directget' and hyperparam != 'min_length':
            if hyperparam not in hyperparam_dict.keys():
                hyperparam_dict[hyperparam] = [file_renamed.split(hyperparam + '-')[1].split('_')[0]]
            else:
                hyperparam_dict[hyperparam].append(file_renamed.split(hyperparam + '-')[1].split('_')[0])

    if modality == 'func':
        if 'smooth-' in file_renamed:
            if 'smooth' not in hyperparam_dict.keys():
                hyperparam_dict['smooth'] = [file_renamed.split('smooth-')[1].split('_')[0].split('fwhm')[0]]
            else:
                hyperparam_dict['smooth'].append(file_renamed.split('smooth-')[1].split('_')[0].split('fwhm')[0])
            hyperparams.append('smooth')
        if 'hpass-' in file_renamed:
            if 'hpass' not in hyperparam_dict.keys():
                hyperparam_dict['hpass'] = [file_renamed.split('hpass-')[1].split('_')[0].split('Hz')[0]]
            else:
                hyperparam_dict['hpass'].append(file_renamed.split('hpass-')[1].split('_')[0].split('Hz')[0])
            hyperparams.append('hpass')
    elif modality == 'dwi':
        if 'tt-' in file_renamed:
            if 'track_type' not in hyperparam_dict.keys():
                hyperparam_dict['track_type'] = [file_renamed.split('tt-')[1].split('_')[0]]
            else:
                hyperparam_dict['track_type'].append(file_renamed.split('tt-')[1].split('_')[0])
            hyperparams.append('track_type')
        if 'dg-' in file_renamed:
            if 'directget' not in hyperparam_dict.keys():
                hyperparam_dict['directget'] = [file_renamed.split('dg-')[1].split('_')[0]]
            else:
                hyperparam_dict['directget'].append(file_renamed.split('dg-')[1].split('_')[0])
            hyperparams.append('directget')
        if 'ml-' in file_renamed:
            if 'min_length' not in hyperparam_dict.keys():
                hyperparam_dict['min_length'] = [file_renamed.split('ml-')[1].split('_')[0]]
            else:
                hyperparam_dict['min_length'].append(file_renamed.split('ml-')[1].split('_')[0])
            hyperparams.append('min_length')
    return hyperparam_dict, hyperparams


def discr_stat(X, Y, dissimilarity="euclidean", remove_isolates=True, return_rdfs=True):
    """
    Computes the discriminability statistic.
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        Input data. If dissimilarity=='precomputed', the input should be the dissimilarity matrix.
    Y : 1d-array, shape (n_samples)
        Input labels.
    dissimilarity : str, {"euclidean" (default), "precomputed"}
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities.
    remove_isolates : bool, optional, default=True
        Whether to remove data that have single label.
    return_rdfs : bool, optional, default=False
        Whether to return rdf for all data points.
    Returns
    -------
    stat : float
        Discriminability statistic.
    rdfs : array, shape (n_samples, max{len(id)})
        Rdfs for each sample. Only returned if ``return_rdfs==True``.
    """
    check_X_y(X, Y, accept_sparse=True)

    uniques, counts = np.unique(Y, return_counts=True)
    if (counts != 1).sum() <= 1:
        msg = "You have passed a vector containing only a single unique sample id."
        raise ValueError(msg)
    if remove_isolates:
        idx = np.isin(Y, uniques[counts != 1])
        labels = Y[idx]

        if dissimilarity == "euclidean" or dissimilarity == "cosine" or dissimilarity == "haversine" or \
            dissimilarity == "manhattan" or dissimilarity == "mahalanobis":
            X = X[idx]
        else:
            X = X[np.ix_(idx, idx)]
    else:
        labels = Y

    if dissimilarity == "euclidean":
        dissimilarities = nan_euclidean_distances(X)
    elif dissimilarity == "cosine":
        dissimilarities = cosine_distances(X)
    elif dissimilarity == "haversine":
        dissimilarities = haversine_distances(X)
    elif dissimilarity == "manhattan":
        dissimilarities = manhattan_distances(X)
    else:
        dissimilarities = X

    rdfs = _discr_rdf(dissimilarities, labels)
    rdfs[rdfs<0.5] = np.nan
    stat = np.nanmean(rdfs)

    if return_rdfs:
        return stat, rdfs
    else:
        return stat


def _discr_rdf(dissimilarities, labels):
    """
    A function for computing the reliability density function of a dataset.
    Parameters
    ----------
    dissimilarities : array, shape (n_samples, n_features) or (n_samples, n_samples)
        Input data. If dissimilarity=='precomputed', the input should be the
        dissimilarity matrix.
    labels : 1d-array, shape (n_samples)
        Input labels.
    Returns
    -------
    out : array, shape (n_samples, max{len(id)})
        Rdfs for each sample. Only returned if ``return_rdfs==True``.
    """
    check_X_y(dissimilarities, labels, accept_sparse=True)
    rdfs = []

    for i, label in enumerate(labels):
        di = dissimilarities[i]

        # All other samples except its own label
        idx = labels == label
        Dij = di[~idx]

        # All samples except itself
        idx[i] = False
        Dii = di[idx]

        rdf = [
            1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii
        ]
        rdfs.append(rdf)

    out = np.full((len(rdfs), max(map(len, rdfs))), np.nan)
    for i, rdf in enumerate(rdfs):
        out[i, : len(rdf)] = rdf

    return out


def reshape_graphs(graphs):
    n, v1, v2 = np.shape(graphs)
    return np.reshape(graphs, (n, v1 * v2))


def CronbachAlpha(itemscores):
    itemscores = np.asarray([i for i in itemscores if np.nan not in i])
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    calpha = nitems / float(nitems-1) * (1 - itemvars.sum() / float(tscores.var(ddof=1)))

    return calpha


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    working_dir = '/scratch/04171/dpisner/HNU/HNU_outs'
    thr_type = 'MST'
    icc = True
    disc = True

    mets = ['global_efficiency', 'average_shortest_path_length',
            'average_betweenness_centrality', 'average_eigenvector_centrality', 'average_degree_centrality',
            'modularity']
    modality = 'func'

    if modality == 'dwi':
        naughty_list = ['fwhm', 'partcorr']
    else:
        naughty_list = ['triple_net_ICA_overlap_3', 'streams', 'PROP']
    df = pd.read_csv(working_dir + '/all_subs_neat.csv')
    for bad_col in naughty_list:
        df = df.loc[:, ~df.columns.str.contains(bad_col, regex=True)]

    df = df[df['id'].str.contains('run')]
    #df = df.loc[:, df.columns.str.contains('MST', regex=True)]

    if modality == 'func':
        df = df.rename(columns=lambda x: re.sub('_partcorr_', '_est-partcorr_', x))
        df = df.rename(columns=lambda x: re.sub('_sps_', '_est-sps_', x))
        df = df.rename(columns=lambda x: re.sub('_sig_bin_nores-2mm', '', x))
    elif modality == 'dwi':
        df = df.rename(columns=lambda x: re.sub('csa_', 'est-csa_', x))
        df = df.rename(columns=lambda x: re.sub('csd_', 'est-csd_', x))
    df = df.rename(columns=lambda x: re.sub('_k', '_k-', x))
    df = df.rename(columns=lambda x: re.sub('_thr_', '', x))
    df = df.rename(columns=lambda x: re.sub('_est-est-', '_est-', x))
    df = df.rename(columns=lambda x: re.sub('__', '_', x))
    df = df.dropna(subset=['id'])

    cols = [j for j in set([i.split('_thrtype-' + thr_type + '_')[0] for i in list(set(df.columns))]) if j != 'id']

    for col in [i for i in df.columns if i != 'id']:
        df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
        df[col] = df[col][df[col]>0]
        df[col] = df[col][df[col]<1]
        #df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3)]

    df = df.drop(df.loc[:, list((100 * (df.isnull().sum() / len(df.index)) > 20))].columns, 1)

    hyperparam_dict = {}

    if icc is True and disc is False:
        df_summary = pd.DataFrame(columns=['grid', 'icc'])
    elif icc is False and disc is True:
        df_summary = pd.DataFrame(columns=['grid', 'discriminability'])
    elif icc is True and disc is True:
        df_summary = pd.DataFrame(columns=['grid', 'discriminability', 'icc'])

    if modality == 'func':
        gen_hyperparams = ['est', 'clust', '_k']
        for col in cols:
            build_hp_dict(col, col.split('_clust')[0], 'func', hyperparam_dict, gen_hyperparams)

        for key in hyperparam_dict:
            hyperparam_dict[key] = list(set(hyperparam_dict[key]))

        grid = list(itertools.product(*(hyperparam_dict[param] for param in hyperparam_dict.keys())))

        subject_dict = {}
        for id in df['id']:
            print(id)
            ID = id.split('_')[0].split('sub-')[1]
            ses = id.split('_')[1].split('ses-')[1]
            if ID not in subject_dict.keys():
                subject_dict[ID] = {}
            subject_dict[ID][ses] = dict.fromkeys(grid , np.nan)
            for atlas, est, clust, _k, smooth, hpass in subject_dict[ID][ses]:
                subject_dict[ID][ses][(atlas, est, clust, _k, smooth, hpass)] = {}
                met_vals = np.empty([len(mets), 1], dtype=np.float32)
                met_vals[:] = np.nan
                i = 0
                for met in mets:
                    col = atlas + '_clust-' + clust + '_k-' + str(_k) + '_est-' + est + \
                          '_nodetype-parc_' + 'smooth-' + str(smooth) + 'fwhm_hpass-' + str(hpass) + 'Hz_' + \
                          'thrtype-' + thr_type + '_net_mets_auc_' + met + '_auc'
                    try:
                        met_vals[i] = df[df['id'] == 'sub-' + ID + '_ses-' + ses][col].values[0]
                    except:
                        print('No values found for: ' + met + ' in column: ' + col + '\n')
                        met_vals[i] = np.nan
                    del col
                    i += 1
                subject_dict[ID][ses][(atlas, est, clust, _k, smooth, hpass)]['topology'] = met_vals
                del i, atlas, est, clust, _k, smooth, hpass
            del ID, ses

        if icc is True:
            i = 0
            for atlas, est, clust, _k, smooth, hpass in grid:
                df_summary.at[i,'grid'] = (atlas, est, clust, _k, smooth, hpass)
                print(atlas, est, clust, _k, smooth, hpass)
                id_list = []
                icc_list = []
                for ID in subject_dict.keys():
                    ses_list = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        ses_list.append(subject_dict[ID][ses][(atlas, est, clust, _k, smooth, hpass)]['topology'])
                    meas = np.hstack(ses_list)
                    try:
                        icc_out = CronbachAlpha(meas)
                        icc_list.append(icc_out)
                        df_summary.at[i,'icc'] = np.nanmean(icc_list)
                        del icc_out, ses_list
                    except:
                        continue
                del icc_list
                i += 1

        if disc is True:
            i = 0
            for atlas, est, clust, _k, smooth, hpass in grid:
                print(atlas, est, clust, _k, smooth, hpass)
                id_list = []
                vect_all = []
                for ID in subject_dict.keys():
                    vects = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        vects.append(subject_dict[ID][ses][(atlas, est, clust, _k, smooth, hpass)]['topology'])
                    vect_all.append(np.concatenate(vects, axis=1))
                    del vects
                X_top = np.swapaxes(np.hstack(vect_all),0,1)

                Y = np.array(id_list)
                try:
                    df_summary.at[i,'grid'] = (atlas, est, clust, _k, smooth, hpass)
                    bad_ixs = [i[1] for i in np.argwhere(np.isnan(X_top))]
                    for m in set(bad_ixs):
                        if (X_top.shape[0] - bad_ixs.count(m))/X_top.shape[0] < 0.50:
                            X_top = np.delete(X_top, m, axis=1)
                    imp = IterativeImputer(max_iter=50, random_state=42)
                    X_top = imp.fit_transform(X_top)
                    scaler = StandardScaler()
                    X_top = scaler.fit_transform(X_top)
                    discr_stat_val, rdf = discr_stat(X_top, Y)
                    df_summary.at[i, 'discriminability'] = discr_stat_val
                    print(discr_stat_val)
                    #print(rdf)
                    del discr_stat_val
                    i += 1
                except:
                    i += 1
                    continue
    elif modality == 'dwi':
        gen_hyperparams = ['est', 'clust', '_k']
        for col in cols:
            build_hp_dict(col, col.split('_clust')[0], 'dwi', hyperparam_dict, gen_hyperparams)

        for key in hyperparam_dict:
            hyperparam_dict[key] = list(set(hyperparam_dict[key]))

        grid = list(itertools.product(*(hyperparam_dict[param] for param in hyperparam_dict.keys())))

        subject_dict = {}
        for id in df['id']:
            print(id)
            ID = id.split('_')[0].split('sub-')[1]
            ses = id.split('_')[1].split('ses-')[1]
            if ID not in subject_dict.keys():
                subject_dict[ID] = {}
            subject_dict[ID][ses] = dict.fromkeys(grid , np.nan)
            for atlas, est, clust, _k, track_type, directget, min_length in subject_dict[ID][ses]:
                subject_dict[ID][ses][(atlas, est, clust, _k, track_type, directget, min_length)] = {}
                met_vals = np.empty([len(mets), 1], dtype = np.float32)
                met_vals[:] = np.nan
                i = 0
                for met in mets:
                    col = atlas + '_clust-' + clust + '_k-' + str(_k) + '_est-' + est + \
                          '_nodetype-parc_samples-50000streams_tt-' + track_type + '_dg-' + directget + '_ml-' + \
                          min_length + '_thrtype-' + thr_type + '_net_mets_auc_' + met + '_auc'
                    try:
                        met_vals[i] = df[df['id'] == 'sub-' + ID + '_ses-' + ses][col].values[0]
                    except:
                        print('No values found for: ' + met + ' in column: ' + col + '\n')
                        met_vals[i] = np.nan
                    del col
                    i += 1
                subject_dict[ID][ses][(atlas, est, clust, _k, track_type, directget, min_length)]['topology'] = met_vals
                del i, atlas, est, clust, _k, track_type, directget, min_length
            del ID, ses

        if icc is True:
            i = 0
            for atlas, est, clust, _k, track_type, directget, min_length in grid:
                df_summary.at[i,'grid'] = (atlas, est, clust, _k, track_type, directget, min_length)
                print(atlas, est, clust, _k, track_type, directget, min_length)
                id_list = []
                icc_list = []
                for ID in subject_dict.keys():
                    ses_list = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        ses_list.append(subject_dict[ID][ses][(atlas, est, clust, _k, track_type, directget,
                                                               min_length)]['topology'])
                    meas = np.hstack(ses_list)
                    try:
                        icc_out = CronbachAlpha(meas)
                        icc_list.append(icc_out)
                        df_summary.at[i,'icc'] = np.nanmean(icc_list)
                        del icc_out, ses_list
                    except:
                        continue
                del icc_list
                i += 1

        if disc is True:
            i = 0
            for atlas, est, clust, _k, track_type, directget, min_length in grid:
                print(atlas, est, clust, _k, track_type, directget, min_length)
                id_list = []
                vect_all = []
                for ID in subject_dict.keys():
                    vects = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        vects.append(subject_dict[ID][ses][(atlas, est, clust, _k, track_type, directget,
                                                            min_length)]['topology'])
                    vect_all.append(np.concatenate(vects, axis=1))
                    del vects
                X_top = np.swapaxes(np.hstack(vect_all),0,1)

                Y = np.array(id_list)
                try:
                    df_summary.at[i, 'grid'] = (atlas, est, clust, _k, track_type, directget, min_length)
                    bad_ixs = [i[1] for i in np.argwhere(np.isnan(X_top))]
                    for m in set(bad_ixs):
                        if (X_top.shape[0] - bad_ixs.count(m))/X_top.shape[0] < 0.50:
                            X_top = np.delete(X_top, m, axis=1)
                    imp = IterativeImputer(max_iter=50, random_state=42)
                    X_top = imp.fit_transform(X_top)
                    scaler = StandardScaler()
                    X_top = scaler.fit_transform(X_top)
                    discr_stat_val, rdf = discr_stat(X_top, Y)
                    df_summary.at[i, 'discriminability'] = discr_stat_val
                    print(discr_stat_val)
                    #print(rdf)
                    del discr_stat_val
                    i += 1
                except:
                    i += 1
                    continue

    if icc is True and disc is False:
        df_summary = df_summary.sort_values('icc', ascending=False)
        #df_summary = df_summary[df_summary.topological_icc > df_summary.icc.quantile(.50)]
    elif icc is False and disc is True:
        df_summary = df_summary.sort_values('discriminability', ascending=False)
        # df_summary = df_summary[df_summary.discriminability >
        #                         df_summary.discriminability.quantile(.50)]
    elif icc is True and disc is True:
        df_summary = df_summary.sort_values(by=['discriminability', 'icc'], ascending=False)

    df_summary.to_csv(working_dir + '/grid_clean_' + modality + '.csv')
