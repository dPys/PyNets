#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@authors: Derek Pisner
"""
import os
from sklearn.metrics.pairwise import (
    cosine_distances,
    haversine_distances,
    manhattan_distances,
    euclidean_distances,
)
from sklearn.utils import check_X_y
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from pynets.stats.prediction import flatten_latent_positions
from pynets.core.utils import flatten
warnings.filterwarnings("ignore")


def build_hp_dict(file_renamed, modality, hyperparam_dict, hyperparams):
    """
    A function to build a hyperparameter dictionary by parsing a given
    file path.
    """

    for hyperparam in hyperparams:
        if (
            hyperparam != "smooth"
            and hyperparam != "hpass"
            and hyperparam != "track_type"
            and hyperparam != "directget"
            and hyperparam != "tol"
            and hyperparam != "minlength"
            and hyperparam != "samples"
            and hyperparam != "nodetype"
            and hyperparam != "template"
            and hyperparam != "extract"

        ):
            if hyperparam not in hyperparam_dict.keys():
                hyperparam_dict[hyperparam] = [
                    str(file_renamed.split(hyperparam + "-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict[hyperparam].append(
                    str(file_renamed.split(hyperparam + "-")[1].split("_")[0])
                )

    if modality == "func":
        if "smooth-" in file_renamed:
            if "smooth" not in hyperparam_dict.keys():
                hyperparam_dict["smooth"] = [str(file_renamed.split(
                    "smooth-")[1].split("_")[0].split("fwhm")[0])]
            else:
                hyperparam_dict["smooth"].append(str(file_renamed.split(
                    "smooth-")[1].split("_")[0].split("fwhm")[0]))
        else:
            if 'smooth' not in hyperparam_dict.keys():
                hyperparam_dict['smooth'] = [str(0)]
            hyperparam_dict["smooth"].append(str(0))
            hyperparams.append("smooth")
        if "hpass-" in file_renamed:
            if "hpass" not in hyperparam_dict.keys():
                hyperparam_dict["hpass"] = [str(file_renamed.split(
                    "hpass-")[1].split("_")[0].split("Hz")[0])]
            else:
                hyperparam_dict["hpass"].append(
                    str(file_renamed.split("hpass-"
                                       )[1].split("_")[0].split("Hz")[0]))
            hyperparams.append("hpass")
        if "extract-" in file_renamed:
            if "extract" not in hyperparam_dict.keys():
                hyperparam_dict["extract"] = [
                    str(file_renamed.split("extract-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["extract"].append(
                    str(file_renamed.split("extract-")[1].split("_")[0])
                )
            hyperparams.append("extract")

    elif modality == "dwi":
        if "directget-" in file_renamed:
            if "directget" not in hyperparam_dict.keys():
                hyperparam_dict["directget"] = [
                    str(file_renamed.split("directget-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["directget"].append(
                    str(file_renamed.split("directget-")[1].split("_")[0])
                )
            hyperparams.append("directget")
        if "minlength-" in file_renamed:
            if "minlength" not in hyperparam_dict.keys():
                hyperparam_dict["minlength"] = [
                    str(file_renamed.split("minlength-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["minlength"].append(
                    str(file_renamed.split("minlength-")[1].split("_")[0])
                )
            hyperparams.append("minlength")
        if "tol-" in file_renamed:
            if "tol" not in hyperparam_dict.keys():
                hyperparam_dict["tol"] = [
                    str(file_renamed.split("tol-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["tol"].append(
                    str(file_renamed.split("tol-")[1].split("_")[0])
                )
            hyperparams.append("tol")

    for key in hyperparam_dict:
        hyperparam_dict[key] = list(set(hyperparam_dict[key]))

    return hyperparam_dict, hyperparams


def discr_stat(
        X,
        Y,
        dissimilarity="euclidean",
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
    else:
        dissimilarities = X

    rdfs = _discr_rdf(dissimilarities, labels)
    rdfs[rdfs < 0.5] = np.nan
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


def beta_lin_comb(beta, GVDAT, meta):
    """
    This function calculates linear combinations of graph vectors stored in
    GVDAT for all subjects and all sessions given the weights vector beta.
    This was adapted from a function of the same name, written by Kamil Bonna
    and Miriam Kosik 10.09.2018.

    Parameters
    ----------
    beta : list
        List of metaparameter weights.
    GVDAT : ndarray
        5d data structure storing graph vectors.

    Returns
    -------
    gv_array : ndarray
        2d array of aggregated graph vectors for all sessions, all subjects.
    """
    import numpy as np
    import math

    def normalize_beta(beta):
        sum_weight = sum([b1 * b2 * b3 for b1 in beta[:N_atl] for b2 in
                          beta[N_atl:N_atl + N_mod] for b3 in beta[-N_thr:]])
        return [b / math.pow(sum_weight, 1 / 3) for b in beta]

    # Dataset dimensionality
    N_sub = meta['N_sub']
    N_ses = meta['N_ses']
    N_gvm = meta['N_gvm']
    N_thr = len(meta['thr'])
    N_atl = len(meta['atl'])
    N_mod = len(meta['mod'])

    # Normalize and split full beta vector
    beta = normalize_beta(beta)
    beta_atl = beta[:N_atl]
    beta_mod = beta[N_atl:N_atl + N_mod]
    beta_thr = beta[-N_thr:]

    # Calculate linear combinations
    gv_array = np.zeros((N_sub * N_ses, N_gvm), dtype='float')
    for sesh in range(N_sub * N_ses):
        gvlc = 0  # Graph-Vector Linear Combination (GVLC)
        for atl in range(N_atl):
            for mod in range(N_mod):
                for thr in range(N_thr):
                    gvlc += GVDAT[sesh][atl][mod][thr] * beta_atl[atl] * \
                            beta_mod[mod] * beta_thr[thr]
        gv_array[sesh] = gvlc
    return gv_array


def gen_sub_vec(base_dir, sub_dict_clean, ID, modality, alg, comb_tuple):
    vects = []
    for ses in sub_dict_clean[ID].keys():
        #print(ses)
        if comb_tuple in sub_dict_clean[ID][ses][modality][alg].keys():
            if alg == 'topology':
                vect = sub_dict_clean[ID][ses][modality][alg][comb_tuple]
            else:
                vect = flatten_latent_positions(base_dir, sub_dict_clean, ID,
                                                ses, modality, comb_tuple, alg)
            vects.append(vect)
    vects = [i for i in vects if i is not None and not np.isnan(np.array(i)).all()]
    if len(vects) > 0 and alg == 'topology':
        out = np.concatenate(vects, axis=1)
    elif len(vects) > 0:
        out = pd.concat(vects, axis=0)
        del vects
    else:
        out = None
    #print(out)
    return out


def benchmark_reproducibility(base_dir, comb, modality, alg, par_dict, disc,
                              final_missingness_summary, icc_tmps_dir, icc,
                              mets, ids):
    import gc
    import json
    import glob
    import ast
    import matplotlib
    matplotlib.use('Agg')

    df_summary = pd.DataFrame(
        columns=['grid', 'modality', 'embedding',
                 'discriminability'])
    print(comb)
    df_summary.at[0, "modality"] = modality
    df_summary.at[0, "embedding"] = alg

    if modality == 'func':
        try:
            extract, hpass, model, res, atlas, smooth = comb
        except BaseException:
            print(f"Missing {comb}...")
            extract, hpass, model, res, atlas = comb
            smooth = '0'
        comb_tuple = (atlas, extract, hpass, model, res, smooth)
    else:
        directget, minlength, model, res, atlas, tol = comb
        comb_tuple = (atlas, directget, minlength, model, res, tol)

    df_summary.at[0, "grid"] = comb_tuple

    # missing_sub_seshes = \
    #     final_missingness_summary.loc[(final_missingness_summary['alg']==alg)
    #                                   & (final_missingness_summary[
    #                                          'modality']==modality) &
    #                                   (final_missingness_summary[
    #                                        'grid']==comb_tuple)
    #                                   ].drop_duplicates(subset='id')

    # icc
    if icc is True and alg == 'topology':
        try:
            import pingouin as pg
        except ImportError:
            print(
                "Cannot evaluate ICC. pingouin"
                " must be installed!")
        for met in mets:
            id_dict = {}
            dfs = []
            for ses in [str(i) for i in range(1, 11)]:
                for ID in ids:
                    id_dict[ID] = {}
                    if comb_tuple in par_dict[ID][str(ses)][modality][alg].keys():
                        id_dict[ID][str(ses)] = \
                            par_dict[ID][ses][modality][alg][comb_tuple][mets.index(met)][0]
                    df = pd.DataFrame(id_dict).T
                    if df.empty:
                        del df
                        return df_summary
                    df.columns.values[0] = f"{met}"
                    df.replace(0, np.nan, inplace=True)
                    df['id'] = df.index
                    df['ses'] = ses
                    df.reset_index(drop=True, inplace=True)
                    dfs.append(df)
            df_long = pd.concat(dfs, names=['id', 'ses', f"{met}"]).drop(
                columns=[str(i) for i in range(1, 10)])
            if '10' in df_long.columns:
                df_long[f"{met}"] = df_long[f"{met}"].fillna(df_long['10'])
                df_long = df_long.drop(columns='10')
            try:
                c_icc = pg.intraclass_corr(data=df_long, targets='id',
                                           raters='ses', ratings=f"{met}",
                                           nan_policy='omit').round(3)
                c_icc = c_icc.set_index("Type")
                c_icc3 = c_icc.drop(index=['ICC1', 'ICC2', 'ICC1k', 'ICC2k', 'ICC3'])
                df_summary.at[0, f"icc_{met}"] = c_icc3['ICC'].values[0]
                df_summary.at[0, f"icc_{met}_CI95%_L"] = c_icc3['CI95%'].values[0][0]
                df_summary.at[0, f"icc_{met}_CI95%_U"] = c_icc3['CI95%'].values[0][1]
            except BaseException:
                print('FAILED...')
                print(df_long)
                del df_long
                return df_summary
            del df_long
    elif icc is True and alg != 'topology':
        import re
        from pynets.stats.prediction import parse_closest_ixs
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
                if comb_tuple in par_dict[ID][str(ses)][modality][alg].keys():
                    if 'data' in par_dict[ID][ses][modality][alg][comb_tuple].keys():
                        if par_dict[ID][ses][modality][alg][comb_tuple]['data'] is not None:
                            if isinstance(par_dict[ID][ses][modality][alg][comb_tuple]['data'], str):
                                if os.path.isfile(par_dict[ID][ses][modality][alg][comb_tuple]['data']):
                                    try:
                                        emb_data = np.load(par_dict[ID][ses][modality][alg][comb_tuple]['data'])
                                        node_files = glob.glob(f"{os.path.dirname(par_dict[ID][ses][modality][alg][comb_tuple]['data'])}/nodes/*.json")
                                    except:
                                        continue
                                else:
                                    continue
                            else:
                                node_files = glob.glob(
                                    f"{base_dir}/embeddings_all_{modality}/"
                                    f"sub-{ID}/ses-{ses}/rsn-{comb_tuple[0]}"
                                    f"_res-{comb_tuple[-2]}/nodes/*.json")
                                emb_data = par_dict[ID][ses][modality][alg][comb_tuple]['data']

                            emb_shape = emb_data.shape[0]
                            ixs = [i for i in par_dict[ID][ses][modality][alg][
                                comb_tuple]['index'] if i is not None]

                            if len(node_files) > 0:
                                ixs, node_dict = parse_closest_ixs(node_files,
                                                                   emb_shape)

                                coords = [(i['coord']) for
                                          i in node_dict]
                                if isinstance(node_dict[0]['label'], str):
                                    labels = [
                                        ast.literal_eval(
                                        re.search('({.+})',
                                                  i['label']).group(0))[
                                            'BrainnetomeAtlasFan2016'] for i in
                                        node_dict]
                                else:
                                    try:
                                        labels = [
                                            list(i['label'])[0][
                                                'BrainnetomeAtlasFan2016'] for i in
                                            node_dict]
                                    except:
                                        try:
                                            labels = [ast.literal_eval(
                                                i['label'].replace('[',
                                                                   '').replace(
                                                    '\n 1]', ''))[
                                                          'BrainnetomeAtlasFan2016']
                                                      for i in node_dict]
                                        except:
                                            continue

                                df_coords = pd.DataFrame(
                                    [str(tuple(x)) for x in
                                     coords]).T
                                df_coords.columns = [
                                    f"rsn-{comb_tuple[0]}_res-" \
                                    f"{comb_tuple[-2]}_{i}"
                                    for i in ixs]
                                # labels = [
                                #     list(i['label'])[7] for i
                                #     in
                                #     node_dict]
                                df_labels = pd.DataFrame(
                                    labels).T
                                df_labels.columns = [
                                    f"rsn-{comb_tuple[0]}_res-" \
                                    f"{comb_tuple[-2]}_{i}"
                                    for i in ixs]
                                coords_frames.append(df_coords)
                                labels_frames.append(df_labels)
                            else:
                                print(f"No node files detected for "
                                      f"{comb_tuple} and {ID}-{ses}...")
                                coords_frames.append(pd.Series())
                                labels_frames.append(pd.Series())

                            if len(ixs) == emb_shape:
                                df_pref = pd.DataFrame(emb_data.T, columns=[
                                    f"{alg}_{i}_rsn-{comb_tuple[0]}_res-"
                                    f"{comb_tuple[-2]}"
                                    for i in ixs])
                                df_pref['id'] = ID
                                df_pref['ses'] = ses
                                df_pref.replace(0, np.nan, inplace=True)
                                df_pref.reset_index(drop=True, inplace=True)
                                dfs.append(df_pref)
                            else:
                                print(
                                    f"Embedding shape {emb_shape} for "
                                    f"{comb_tuple} does not correspond to "
                                    f"{len(ixs)} indices found for {ID}-{ses}."
                                    f" Skipping...")
                                continue
                    else:
                        print(
                            f"data not found in {comb_tuple}. Skipping...")
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

        for lp in [i for i in df_long.columns if 'ses' not in i and 'id' not in i]:
            ix = int(lp.split(f"{alg}_")[1].split('_')[0])
            rsn = lp.split(f"{alg}_{ix}_")[1]
            df_long_clean = df_long[['id', 'ses', lp]]
            # df_long_clean = df_long[['id', 'ses', lp]].loc[(df_long[['id', 'ses', lp]]['id'].duplicated() == True) & (df_long[['id', 'ses', lp]]['ses'].duplicated() == True) & (df_long[['id', 'ses', lp]][lp].isnull()==False)]
            # df_long_clean[lp] = np.abs(df_long_clean[lp].round(6))
            # df_long_clean['ses'] = df_long_clean['ses'].astype('int')
            # g = df_long_clean.groupby(['ses'])
            # df_long_clean = pd.DataFrame(g.apply(
            #     lambda x: x.sample(g.size().min()).reset_index(drop=True))).reset_index(drop=True)
            try:
                c_icc = pg.intraclass_corr(data=df_long_clean, targets='id',
                                           raters='ses', ratings=lp,
                                           nan_policy='omit').round(3)
                c_icc = c_icc.set_index("Type")
                c_icc3 = c_icc.drop(
                    index=['ICC1', 'ICC2', 'ICC1k', 'ICC2k', 'ICC3'])
                icc_val = c_icc3['ICC'].values[0]
                if nodes is True:
                    coord_in = np.array(ast.literal_eval(coords_frames_icc[f"{rsn}_{ix}"].mode().values[0]), dtype=np.dtype("O"))
                    label_in = np.array(labels_frames_icc[f"{rsn}_{ix}"].mode().values[0], dtype=np.dtype("O"))
                else:
                    coord_in = np.nan
                    label_in = np.nan
                dict_sum[f"{lp}_icc"] = icc_val
                del c_icc, c_icc3
            except BaseException:
                print(f"FAILED for {lp}...")
                #print(df_long)
                #df_summary.at[0, f"{lp}_icc"] = np.nan
                coord_in = np.nan
                label_in = np.nan

            dict_sum[f"{lp}_coord"] = coord_in
            dict_sum[f"{lp}_label"] = label_in

        df_summary = pd.concat([df_summary, pd.DataFrame(pd.Series(dict_sum).T).T], axis=1)

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
                X_top = np.array(X_top.dropna(axis='columns', thresh=0.50 * len(X_top)))
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


def tuple_insert(tup, pos, ele):
    tup = tup[:pos] + (ele,) + tup[pos:]
    return tup

