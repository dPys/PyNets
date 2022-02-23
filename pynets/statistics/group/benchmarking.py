#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import (
    cosine_distances,
    haversine_distances,
    manhattan_distances,
    euclidean_distances,
)
from sklearn.utils import check_X_y
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from pynets.statistics.utils import mahalanobis_distances
from pynets.core.utils import flatten

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def discr_stat(
        X,
        Y,
        dissimilarity='mahalanobis',
        remove_isolates=True,
        return_rdfs=True):
    """
    Computes the discriminability statistic.

    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        Input data. If dissimilarity=='precomputed', the input should be the
         dissimilarity matrix.
    Y : 1d-array, shape (n_samples)
        Input labels.
    dissimilarity : str, {"euclidean" (default), "precomputed"} Dissimilarity
        measure can be 'euclidean' (pairwise Euclidean distances between points
        in the dataset) or 'precomputed' (pre-computed dissimilarities).
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
    if remove_isolates:
        idx = np.isin(Y, uniques[counts != 1])
        labels = Y[idx]

        if (
            dissimilarity == "euclidean"
            or dissimilarity == "cosine"
            or dissimilarity == "haversine"
            or dissimilarity == "manhattan"
            or dissimilarity == "mahalanobis"
        ):
            X = X[idx]
        else:
            X = X[np.ix_(idx, idx)]
    else:
        labels = Y

    if dissimilarity == "euclidean":
        dissimilarities = euclidean_distances(X)
    elif dissimilarity == "cosine":
        dissimilarities = cosine_distances(X)
    elif dissimilarity == "haversine":
        dissimilarities = haversine_distances(X)
    elif dissimilarity == "manhattan":
        dissimilarities = manhattan_distances(X)
    elif dissimilarity == "mahalanobis":
        dissimilarities = mahalanobis_distances(X)
    else:
        dissimilarities = X

    rdfs = _discr_rdf(dissimilarities, labels)
    # rdfs[rdfs < 0.5] = np.nan
    stat = np.nanmean(rdfs)

    if return_rdfs:
        return stat, rdfs
    else:
        return stat


def _discr_rdf(dissimilarities, labels):
    """
    A function for computing the int_consist density function of a dataset.

    Parameters
    ----------
    dissimilarities : array, shape (n_samples, n_features)
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

        rdf = [1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) /
               Dij.size for d in Dii]
        rdfs.append(rdf)

    out = np.full((len(rdfs), max(map(len, rdfs))), np.nan)
    for i, rdf in enumerate(rdfs):
        out[i, : len(rdf)] = rdf

    return out


def benchmark_reproducibility(base_dir, comb, modality, alg, par_dict, disc,
                              final_missingness_summary, icc_tmps_dir, icc,
                              mets, ids, template):
    import gc
    import json
    import glob
    from pathlib import Path
    import ast
    import matplotlib
    from pynets.statistics.utils import gen_sub_vec
    matplotlib.use('Agg')

    df_summary = pd.DataFrame(
        columns=['grid', 'modality', 'embedding',
                 'discriminability'])
    print(comb)
    df_summary.at[0, "modality"] = modality
    df_summary.at[0, "embedding"] = alg

    if modality == 'func':
        try:
            signal, hpass, model, granularity, parcellation, smooth = comb
        except BaseException:
            print(f"Missing {comb}...")
            signal, hpass, model, granularity, parcellation = comb
            smooth = '0'
        # comb_tuple = (parcellation, signal, hpass, model, granularity, smooth)
        comb_tuple = comb
    else:
        traversal, minlength, model, granularity, parcellation, error_margin = comb
        # comb_tuple = (parcellation, traversal, minlength, model, granularity, error_margin)
        comb_tuple = comb

    df_summary.at[0, "grid"] = comb_tuple

    # missing_sub_seshes = \
    #     final_missingness_summary.loc[(final_missingness_summary['alg']==alg)
    #                                   & (final_missingness_summary[
    #                                          'modality']==modality) &
    #                                   (final_missingness_summary[
    #                                        'grid']==comb_tuple)
    #                                   ].drop_duplicates(subset='id')

    # icc
    if icc is True:
        from pynets.statistics.utils import parse_closest_ixs
        try:
            import pingouin as pg
        except ImportError:
            print(
                "Cannot evaluate ICC. pingouin"
                " must be installed!")
        dfs = []
        coords_frames = []
        labels_frames = []
        for ses in [str(i) for i in range(1, 11)]:
            for ID in ids:
                if ses in par_dict[ID].keys():
                    if comb_tuple in par_dict[ID][str(ses)][modality][alg
                                                                      ].keys():
                        if 'data' in par_dict[ID][str(ses)][modality][alg][
                                comb_tuple].keys():
                            if par_dict[ID][str(ses)][modality][alg][
                                    comb_tuple]['data'] is not None:
                                if isinstance(par_dict[ID][str(ses)][
                                        modality][alg][comb_tuple][
                                        'data'], str):
                                    data_path = par_dict[ID][str(ses)][
                                        modality][alg][comb_tuple]['data']
                                    parent_dir = Path(os.path.dirname(
                                        par_dict[ID][str(ses)][modality][alg][
                                            comb_tuple]['data'])).parent
                                    if os.path.isfile(data_path):
                                        try:
                                            if data_path.endswith('.npy'):
                                                emb_data = np.load(data_path)
                                            elif data_path.endswith('.csv'):
                                                emb_data = np.array(
                                                    pd.read_csv(data_path)
                                                ).reshape(-1, 1)
                                            else:
                                                emb_data = np.nan
                                            node_files = glob.glob(
                                                f"{parent_dir}/nodes/*.json")
                                        except:
                                            print(f"Failed to load data from "
                                                  f"{data_path}..")
                                            continue
                                    else:
                                        continue
                                else:
                                    node_files = glob.glob(
                                        f"{base_dir}/pynets/sub-{ID}/ses-"
                                        f"{ses}/{modality}/subnet-"
                                        f"{parcellation}_granularity-{granularity}/nodes/*.json")
                                    emb_data = par_dict[ID][str(ses)][
                                        modality][alg][comb_tuple]['data']

                                emb_shape = emb_data.shape[0]

                                if len(node_files) > 0:
                                    ixs, node_dict = parse_closest_ixs(
                                        node_files, emb_shape,
                                        template=template)
                                    if len(ixs) != emb_shape:
                                        ixs, node_dict = parse_closest_ixs(
                                            node_files, emb_shape)
                                    if isinstance(node_dict, dict):
                                        try:
                                            coords = [node_dict[i]['coord']
                                                      for i
                                                      in node_dict.keys()]
                                            try:
                                                labels = [node_dict[i][
                                                    'label'][
                                                    'BrainnetomeAtlas'
                                                    'Fan2016'] for i in
                                                    node_dict.keys()]
                                            except:
                                                labels = [node_dict[i][
                                                    'label'] for i in
                                                    node_dict.keys()]
                                        except ValueError:
                                            print(comb)
                                    elif isinstance(node_dict[0], list):
                                        try:
                                            coords = \
                                                [node_dict[i]['coord'] for i
                                                 in range(len(node_dict))]
                                            try:
                                                labels = \
                                                    [node_dict[i]['label']
                                                     ['BrainnetomeAtlas' \
                                                      'Fan2016'] for i
                                                     in range(len(node_dict))]
                                            except:
                                                labels = [node_dict[i][
                                                    'label'] for i in
                                                    node_dict.keys()]
                                        except ValueError:
                                            print(comb)
                                    else:
                                        print(f"Failed to parse coords/"
                                              f"labels from {node_files}. "
                                              f"Skipping...")
                                        continue
                                    df_coords = pd.DataFrame(
                                        [str(tuple(x)) for x in
                                         coords]).T
                                    df_coords.columns = [
                                        f"subnet-{parcellation}_granularity-"
                                        f"{granularity}_{i}"
                                        for i in ixs]
                                    # labels = [
                                    #     list(i['label'])[7] for i
                                    #     in
                                    #     node_dict]
                                    df_labels = pd.DataFrame(
                                        labels).T
                                    df_labels.columns = [
                                        f"subnet-{parcellation}_granularity-"
                                        f"{granularity}_{i}"
                                        for i in ixs]
                                    coords_frames.append(df_coords)
                                    labels_frames.append(df_labels)
                                else:
                                    print(f"No node files detected for "
                                          f"{comb_tuple} and {ID}-{ses}...")
                                    ixs = [i for i in par_dict[ID][str(ses)][
                                        modality][alg][
                                        comb_tuple]['index'] if i is not None]
                                    coords_frames.append(pd.Series())
                                    labels_frames.append(pd.Series())

                                if len(ixs) == emb_shape:
                                    df_pref = pd.DataFrame(emb_data.T,
                                                           columns=[
                                                               f"{alg}_{i}_rsn"
                                                               f"-{parcellation}_granularity-"
                                                               f"{granularity}"
                                                               for i in ixs])
                                    df_pref['id'] = ID
                                    df_pref['ses'] = ses
                                    df_pref.replace(0, np.nan, inplace=True)
                                    df_pref.reset_index(drop=True,
                                                        inplace=True)
                                    dfs.append(df_pref)
                                else:
                                    print(
                                        f"Embedding shape {emb_shape} for "
                                        f"{comb_tuple} does not correspond to "
                                        f"{len(ixs)} indices found for "
                                        f"{ID}-{ses}. Skipping...")
                                    continue
                        else:
                            print(
                                f"data not found in {comb_tuple}. Skipping...")
                            continue
                else:
                    continue

        if len(dfs) == 0:
            return df_summary

        if len(coords_frames) > 0 and len(labels_frames) > 0:
            coords_frames_icc = pd.concat(coords_frames)
            labels_frames_icc = pd.concat(labels_frames)
            nodes = True
        else:
            nodes = False

        df_long = pd.concat(dfs, axis=0)
        df_long = df_long.dropna(axis='columns', thresh=0.75 * len(df_long))
        df_long = df_long.dropna(axis='rows', how='all')

        dict_sum = df_summary.drop(columns=['grid', 'modality', 'embedding',
                                            'discriminability']).to_dict()

        for lp in [i for i in df_long.columns if 'ses' not in i and 'id' not
                                                 in i]:
            ix = int(lp.split(f"{alg}_")[1].split('_')[0])
            parcellation = lp.split(f"{alg}_{ix}_")[1]
            df_long_clean = df_long[['id', 'ses', lp]]
            # df_long_clean = df_long[['id', 'ses', lp]].loc[(df_long[['id',
            # 'ses', lp]]['id'].duplicated() == True) & (df_long[['id', 'ses',
            # lp]]['ses'].duplicated() == True) & (df_long[['id', 'ses',
            # lp]][lp].isnull()==False)]
            # df_long_clean[lp] = np.abs(df_long_clean[lp].round(6))
            # df_long_clean['ses'] = df_long_clean['ses'].astype('int')
            # g = df_long_clean.groupby(['ses'])
            # df_long_clean = pd.DataFrame(g.apply(
            #     lambda x: x.sample(g.size().min()).reset_index(drop=True))
            #     ).reset_index(drop=True)
            try:
                c_icc = pg.intraclass_corr(data=df_long_clean, targets='id',
                                           raters='ses', ratings=lp,
                                           nan_policy='omit').round(3)
                c_icc = c_icc.set_index("Type")
                c_icc3 = c_icc.drop(
                    index=['ICC1', 'ICC2', 'ICC1k', 'ICC2k', 'ICC3'])
                icc_val = c_icc3['ICC'].values[0]
                if nodes is True:
                    coord_in = np.array(ast.literal_eval(
                        coords_frames_icc[f"{parcellation}_"
                                          f"{ix}"].mode().values[0]),
                        dtype=np.dtype("O"))
                    label_in = np.array(
                        labels_frames_icc[f"{parcellation}_"
                                          f"{ix}"].mode().values[0],
                        dtype=np.dtype("O"))
                else:
                    coord_in = np.nan
                    label_in = np.nan
                dict_sum[f"{lp}_icc"] = icc_val
                del c_icc, c_icc3, icc_val
            except BaseException:
                print(f"FAILED for {lp}...")
                # print(df_long)
                #df_summary.at[0, f"{lp}_icc"] = np.nan
                coord_in = np.nan
                label_in = np.nan

            dict_sum[f"{lp}_coord"] = coord_in
            dict_sum[f"{lp}_label"] = label_in

        df_summary = pd.concat([df_summary,
                                pd.DataFrame(pd.Series(dict_sum).T).T], axis=1)

        print(df_summary)

        tup_name = str(comb_tuple).replace('\', \'', '_').replace('(',
                                                                  '').replace(
            ')', '').replace('\'', '')
        df_summary.to_csv(f"{icc_tmps_dir}/{alg}_{tup_name}.csv",
                          index=False, header=True)
        del df_long

    # discriminability
    if disc is True:
        vect_all = []
        for ID in ids:
            try:
                out = gen_sub_vec(base_dir, par_dict, ID, modality, alg,
                                  comb_tuple)
            except BaseException:
                print(f"{ID} {modality} {alg} {comb_tuple} failed...")
                continue
            # print(out)
            vect_all.append(out)
        # ## TODO: Remove the .iloc below to include global efficiency.
        # vect_all = [pd.DataFrame(i).iloc[1:] for i in vect_all if i is not
        #             None and not np.isnan(np.array(i)).all()]
        vect_all = [pd.DataFrame(i) for i in vect_all if i is not
                    None and not np.isnan(np.array(i)).all()]

        if len(vect_all) > 0:
            if len(vect_all) > 0:
                X_top = pd.concat(vect_all, axis=0, join="outer")
                X_top = np.array(X_top.dropna(axis='columns',
                                              thresh=0.50 * len(X_top)))
            else:
                print('Empty dataframe!')
                return df_summary

            shapes = []
            for ix, i in enumerate(vect_all):
                shapes.append(i.shape[0] * [list(ids)[ix]])
            Y = np.array(list(flatten(shapes)))
            if alg == 'topology':
                imp = IterativeImputer(max_iter=50,
                                       random_state=42)
            else:
                imp = SimpleImputer()
            X_top = imp.fit_transform(X_top)
            scaler = StandardScaler()
            X_top = scaler.fit_transform(X_top)
            try:
                discr_stat_val, rdf = discr_stat(X_top, Y)
                df_summary.at[0, "discriminability"] = discr_stat_val
                print(discr_stat_val)
                print("\n")
                del discr_stat_val
            except BaseException:
                print('Discriminability calculation failed...')
                return df_summary
            # print(rdf)
        del vect_all

    gc.collect()
    return df_summary
