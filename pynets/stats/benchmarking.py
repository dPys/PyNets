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
from datetime import datetime
from sklearn.utils import check_X_y
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import pandas as pd
import dill
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from pynets.stats.prediction import make_subject_dict, cleanNullTerms, \
    get_ensembles_top, get_ensembles_embedding, \
    build_grid, flatten_latent_positions
from joblib import Parallel, delayed
import tempfile
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
    This was adapted from a function of the same name, written by Kamil Bonna,
    10.09.2018.

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


def gen_sub_vec(sub_dict_clean, ID, modality, alg, comb_tuple):
    vects = []
    for ses in sub_dict_clean[ID].keys():
        #print(ses)
        if comb_tuple in sub_dict_clean[ID][ses][modality][alg].keys():
            if alg == 'topology':
                vect = sub_dict_clean[ID][ses][modality][alg][comb_tuple]
            else:
                vect = flatten_latent_positions(sub_dict_clean, ID, ses,
                                                modality, comb_tuple, alg)
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


def benchmark_reproducibility(comb, modality, alg, par_dict, disc,
                              final_missingness_summary, icc_tmps_dir):
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
                        id_dict[ID][str(ses)] = par_dict[ID][ses][modality][alg][comb_tuple][mets.index(met)][0]
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
                                emb_data = par_dict[ID][ses][modality][alg][comb_tuple]['data']
                                node_files = []
                            ixs = par_dict[ID][ses][modality][alg][comb_tuple]['index']
                            if len(ixs) == emb_data.shape[0]:
                                df_pref = pd.DataFrame(emb_data.T, columns=[
                                    f"{alg}_{i}_rsn-{comb_tuple[0]}_res-{comb_tuple[-2]}"
                                    for i in ixs])
                                df_pref['id'] = ID
                                df_pref['ses'] = ses
                                df_pref.replace(0, np.nan, inplace=True)
                                df_pref.reset_index(drop=True, inplace=True)
                                dfs.append(df_pref)
                                if len(node_files) > 0:
                                    label_file = node_files[0]
                                    with open(label_file, 'r+') as f:
                                        node_dict = json.load(f)
                                    indices = [i['index'] for i in
                                                      node_dict]
                                    if indices == ixs:
                                        coords = [i['coord'] for i in
                                                         node_dict]

                                        df_coords = pd.DataFrame(
                                            [str(tuple(x)) for x in
                                             coords]).T
                                        df_coords.columns = [f"rsn-{comb_tuple[0]}_res-{comb_tuple[-2]}_{i}" for i in ixs]
                                        labels = [
                                            list(i['label'])[7] for i
                                            in
                                            node_dict]

                                        df_labels = pd.DataFrame(
                                            labels).T
                                        df_labels.columns = [f"rsn-{comb_tuple[0]}_res-{comb_tuple[-2]}_{i}" for i in ixs]
                                        coords_frames.append(df_coords)
                                        labels_frames.append(df_labels)
                                    else:
                                        coords_frames.append(pd.Series())
                                        labels_frames.append(pd.Series())
                            else:
                                print(
                                    f"{comb_tuple} does not correspond to indices {ixs} for {ID}-{ses}. Skipping...")
                                continue
                    else:
                        print(
                            f"data not found in {par_dict[ID][ses][modality][alg][comb_tuple]}. Skipping...")
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

        dict_sum = df_summary.drop(columns=['grid', 'modality', 'embedding',
                                            'discriminability']).to_dict()

        for lp in [i for i in df_long.columns if 'ses' not in i and 'id' not in i]:
            ix = int(lp.split(f"{alg}_")[1].split('_')[0])
            rsn = lp.split(f"{alg}_{ix}_")[1]
            try:
                c_icc = pg.intraclass_corr(data=df_long[['id', 'ses', lp]], targets='id',
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
                print('FAILED...')
                print(df_long)
                df_summary.at[0, f"{lp}_icc"] = np.nan
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
                out = gen_sub_vec(par_dict, ID, modality, alg,
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


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    #base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/triple'
    base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/outputs_language'
    thr_type = "MST"
    icc = True
    disc = False
    int_consist = False
    target_modality = 'dwi'

    #embedding_types = ['ASE']
    #embedding_types = ['topology']
    #embedding_types = ['OMNI']
    embedding_types = ['OMNI', 'ASE']
    modalities = ['func', 'dwi']
    #rsns = ['kmeans', 'triple']
    #rsns = ['triple']
    #rsns = ['kmeans']
    rsns = ['language']
    #template = 'CN200'
    template = 'MNI152_T1'
    mets = ["global_efficiency",
            "average_shortest_path_length",
            "degree_assortativity_coefficient",
            "average_betweenness_centrality",
            "average_eigenvector_centrality",
            "smallworldness",
            "modularity"]

    modalities = [i for i in modalities if target_modality == i]

    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength', 'tol']

    sessions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    subject_dict_file_path = (
        f"{base_dir}/pynets_subject_dict_{target_modality}_{'_'.join(embedding_types)}.pkl"
    )
    subject_mod_grids_file_path = (
        f"{base_dir}/pynets_modality_grids_{target_modality}_{'_'.join(embedding_types)}.pkl"
    )
    missingness_summary = (
        f"{base_dir}/pynets_missingness_summary_{target_modality}_{'_'.join(embedding_types)}.csv"
    )
    icc_tmps_dir = f"{base_dir}/icc_tmps/{target_modality}_{'_'.join(embedding_types)}"
    os.makedirs(icc_tmps_dir, exist_ok=True)
    if not os.path.isfile(subject_dict_file_path):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            modalities, base_dir, thr_type, mets, embedding_types, template,
            sessions, rsns
        )
        sub_dict_clean = cleanNullTerms(subject_dict)
        missingness_frames = [i for i in missingness_frames if
                              isinstance(i, pd.DataFrame)]
        if len(missingness_frames) != 0:
            if len(missingness_frames) > 0:
                if len(missingness_frames) > 1:
                    final_missingness_summary = pd.concat(missingness_frames)
                    final_missingness_summary.to_csv(missingness_summary,
                                                     index=False)
                    final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
                elif len(missingness_frames) == 1:
                    final_missingness_summary = missingness_frames[0]
                    final_missingness_summary.to_csv(missingness_summary, index=False)
                    final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
                else:
                    final_missingness_summary = pd.Series()
            else:
                final_missingness_summary = pd.Series()
        else:
            final_missingness_summary = pd.Series()
        with open(subject_dict_file_path, "wb") as f:
            dill.dump(sub_dict_clean, f)
        f.close()
        with open(subject_mod_grids_file_path, "wb") as f:
            dill.dump(modality_grids, f)
        f.close()
    else:
        with open(subject_dict_file_path, 'rb') as f:
            sub_dict_clean = dill.load(f)
        f.close()
        with open(subject_mod_grids_file_path, "rb") as f:
            modality_grids = dill.load(f)
        f.close()
        if os.path.isfile(missingness_summary):
            final_missingness_summary = pd.read_csv(missingness_summary)
            final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
        else:
            final_missingness_summary = pd.Series()
    ids = sub_dict_clean.keys()

    def tuple_insert(tup, pos, ele):
        tup = tup[:pos] + (ele,) + tup[pos:]
        return tup

    for modality in modalities:
        print(f"MODALITY: {modality}")
        hyperparams = eval(f"hyperparams_{modality}")
        hyperparam_dict = {}

        for alg in embedding_types:
            print(f"EMBEDDING TYPE: {alg}")
            # if os.path.isfile(f"{base_dir}/grid_clean_{modality}_{alg}.csv"):
            #     continue

            if alg == 'topology':
                ensembles, df_top = get_ensembles_top(modality, thr_type,
                                                      f"{base_dir}/pynets")
            else:
                ensembles = get_ensembles_embedding(modality, alg,
                                                    base_dir)
            grid = build_grid(
                modality, hyperparam_dict, sorted(list(set(hyperparams))),
                ensembles)[1]

            # In the case that we are using all of the 3 RSN connectomes
            # (pDMN, coSN, and fECN) in the feature-space,
            # rather than varying them as hyperparameters (i.e. we assume
            # they each add distinct variance
            # from one another) Create an abridged grid, where

            if modality == "func":
                modality_grids[modality] = grid
            else:
                modality_grids[modality] = grid

            cache_dir = tempfile.mkdtemp()

            with Parallel(
                n_jobs=-1, require="sharedmem", backend='threading',
                verbose=10, max_nbytes='20000M',
                temp_folder=cache_dir
            ) as parallel:
                outs = parallel(
                    delayed(benchmark_reproducibility)(
                        comb, modality, alg, sub_dict_clean,
                        disc, final_missingness_summary, icc_tmps_dir
                    )
                    for comb in grid
                )

            # outs = []
            # for comb in grid:
            #     outs.append(benchmark_reproducibility(
            #         comb, modality, alg, sub_dict_clean,
            #         disc, final_missingness_summary, icc_tmps_dir,
            #     ))
            df_summary = pd.concat([i for i in outs if i is not None and not i.empty], axis=0)
            df_summary = df_summary.dropna(axis=0, how='all')
            print(f"Saving to {base_dir}/grid_clean_{modality}_{alg}_"
                  f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
            df_summary.to_csv(f"{base_dir}"
                              f"/grid_clean_{modality}_{alg}_"
                              f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv", index=False)

            # int_consist
            if int_consist is True and alg == 'topology':
                try:
                    import pingouin as pg
                except ImportError:
                    print(
                        "Cannot evaluate test-retest int_consist. pingouin"
                        " must be installed!")

                df_summary_cronbach = pd.DataFrame(
                    columns=['modality', 'embedding', 'cronbach'])
                df_summary_cronbach.at[0, "modality"] = modality
                df_summary_cronbach.at[0, "embedding"] = alg

                for met in mets:
                    cronbach_ses_list = []
                    for ses in range(1, 10):
                        id_dict = {}
                        for ID in ids:
                            id_dict[ID] = {}
                            for comb in grid:
                                if modality == 'func':
                                    try:
                                        extract, hpass, model, res, atlas, smooth = comb
                                    except BaseException:
                                        print(f"Missing {comb}...")
                                        extract, hpass, model, res, atlas = comb
                                        smooth = '0'
                                    comb_tuple = (
                                    atlas, extract, hpass, model, res,
                                    smooth)
                                else:
                                    directget, minlength, model, res, atlas, tol = comb
                                    comb_tuple = (
                                    atlas, directget, minlength, model,
                                    res, tol)
                                if comb_tuple in sub_dict_clean[ID][str(ses)][modality][alg].keys():
                                    if isinstance(sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple], np.ndarray):
                                        id_dict[ID][comb] = sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple][mets.index(met)][0]
                                    else:
                                        continue
                                else:
                                    continue
                        df_wide = pd.DataFrame(id_dict)
                        if df_wide.empty is True:
                            continue
                        else:
                            df_wide = df_wide.add_prefix(f"{met}_comb_")
                            df_wide.replace(0, np.nan, inplace=True)
                            print(df_wide)
                        try:
                            c_alpha = pg.cronbach_alpha(data=df_wide.dropna(axis=1, how='all'), nan_policy='listwise')
                            cronbach_ses_list.append(c_alpha[0])
                        except BaseException:
                            print('FAILED...')
                            print(df_wide)
                            del df_wide
                        del df_wide
                    df_summary_cronbach.at[0, f"average_cronbach_{met}"] = np.nanmean(cronbach_ses_list)
                print(f"Saving to {base_dir}/grid_clean_{modality}_{alg}_cronbach_"
                      f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
                df_summary_cronbach.to_csv(f"{base_dir}/grid_clean_{modality}_{alg}_cronbach{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv", index=False)
