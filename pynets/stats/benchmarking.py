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
import re
import glob
import dill
import numpy as np
import itertools
import warnings
from sklearn.preprocessing import StandardScaler
from pynets.stats.prediction import make_subject_dict, cleanNullTerms, \
    get_ensembles_top, build_grid, flatten_latent_positions
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
                vect = flatten_latent_positions('triple', sub_dict_clean, ID,
                                                ses,
                                                modality, comb_tuple[1:], alg)
            vects.append(vect)
    vects = [i for i in vects if i is not None and not np.isnan(i).all()]
    if len(vects) > 0 and alg == 'topology':
        out = np.concatenate(vects, axis=1)
    elif len(vects) > 0:
        out = pd.concat(vects, axis=0)
        del vects
    else:
        out = None
    #print(out)
    return out


def benchmark_reproducibility(comb, modality, alg, sub_dict_clean, disc,
                              int_consist, final_missingness_summary):
    df_summary = pd.DataFrame(
        columns=['grid', 'modality', 'embedding',
                 'discriminability'])
    print(comb)
    df_summary.at[0, "modality"] = modality
    df_summary.at[0, "embedding"] = alg

    if modality == 'func':
        try:
            extract, hpass, model, res, atlas, smooth = comb
        except:
            print(f"Missing {comb}...")
            extract, hpass, model, res, atlas = comb
            smooth = '0'
        comb_tuple = (atlas, extract, hpass, model, res, smooth)
    else:
        directget, minlength, model, res, atlas, tol = comb
        comb_tuple = (atlas, directget, minlength, model, res, tol)

    df_summary.at[0, "grid"] = comb_tuple

    missing_sub_seshes = \
        final_missingness_summary.loc[(final_missingness_summary['alg']==alg)
                                      & (final_missingness_summary[
                                             'modality']==modality) &
                                      (final_missingness_summary[
                                           'grid']==comb_tuple)
                                      ].drop_duplicates(subset='id')

    # int_consist
    if int_consist is True and alg == 'topology':
        try:
            import pingouin as pg
        except ImportError:
            print(
                "Cannot evaluate test-retest int_consist. pingouin"
                " must be installed!")
        for met in mets:
            id_dict = {}
            for ID in ids:
                id_dict[ID] = {}
                for ses in sub_dict_clean[ID].keys():
                    if comb_tuple in sub_dict_clean[ID][ses][
                        modality][alg].keys():
                        id_dict[ID][ses] = \
                        sub_dict_clean[ID][ses][modality][alg][comb_tuple][
                            mets.index(met)][0]
            df_wide = pd.DataFrame(id_dict).T
            if df_wide.empty:
                del df_wide
                return pd.Series()
            df_wide = df_wide.add_prefix(f"{met}_visit_")
            df_wide.replace(0, np.nan, inplace=True)
            try:
                c_alpha = pg.cronbach_alpha(data=df_wide)
            except:
                print('FAILED...')
                print(df_wide)
                del df_wide
                return pd.Series()
            df_summary.at[0, f"cronbach_alpha_{met}"] = c_alpha[0]
            del df_wide

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
                    if comb_tuple in sub_dict_clean[ID][ses][
                        modality][alg].keys():
                        id_dict[ID][ses] = \
                        sub_dict_clean[ID][ses][modality][alg][comb_tuple][
                            mets.index(met)][0]
                    df = pd.DataFrame(id_dict).T
                    if df.empty:
                        del df_long
                        return pd.Series()
                    df.columns.values[0] = f"{met}"
                    df.replace(0, np.nan, inplace=True)
                    df['id'] = df.index
                    df['ses'] = ses
                    df.reset_index(drop=True, inplace=True)
                    dfs.append(df)
            df_long = pd.concat(dfs, names=['id', 'ses', f"{met}"]).drop(
                columns=[str(i) for i in range(1, 10)])
            try:
                c_icc = pg.intraclass_corr(data=df_long, targets='id',
                                           raters='ses', ratings=f"{met}",
                                           nan_policy='omit').round(3)
                c_icc = c_icc.set_index("Type")
                df_summary.at[0, f"icc_{met}"] = pd.DataFrame(
                    c_icc.drop(index=['ICC1', 'ICC2', 'ICC3'])['ICC']).mean()[
                    0]
            except:
                print('FAILED...')
                print(df_long)
                del df_long
                return pd.Series()
            del df_long

    if disc is True:
        vect_all = []
        for ID in ids:
            try:
                out = gen_sub_vec(sub_dict_clean, ID, modality, alg,
                                  comb_tuple)
            except:
                print(f"{ID} {modality} {alg} {comb_tuple} failed...")
                continue
            # print(out)
            vect_all.append(out)
        vect_all = [i for i in vect_all if i is not None and not np.isnan(i).all()]
        if len(vect_all) > 0:
            if alg == 'topology':
                X_top = np.swapaxes(np.hstack(vect_all), 0, 1)
                bad_ixs = [i[1] for i in
                           np.argwhere(np.isnan(X_top))]
                for m in set(bad_ixs):
                    if (X_top.shape[0] - bad_ixs.count(m)) / \
                        X_top.shape[0] < 0.50:
                        X_top = np.delete(X_top, m, axis=1)
            else:
                if len(vect_all) > 0:
                    X_top = np.array(pd.concat(vect_all, axis=0))
                else:
                    return pd.Series()
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
            except:
                return pd.Series()
            df_summary.at[0, "discriminability"] = discr_stat_val
            print(discr_stat_val)
            print("\n")
            # print(rdf)
            del discr_stat_val
        del vect_all
    return df_summary


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    #base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/triple'
    base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/visual'
    thr_type = "MST"
    icc = True
    disc = True
    int_consist = True

    #embedding_types = ['topology']
    embedding_types = ['topology', 'OMNI', 'ASE']
    #embedding_types = ['ASE']
    #embedding_types = ['OMNI', 'ASE']
    modalities = ['func', 'dwi']
    #modalities = ['dwi']
    template = 'MNI152_T1'
    mets = ["global_efficiency",
            "average_shortest_path_length",
            "degree_assortativity_coefficient",
            "average_betweenness_centrality",
            "average_eigenvector_centrality",
            "smallworldness",
            "modularity"]

    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength', 'tol']

    sessions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    subject_dict_file_path = f"{base_dir}/pynets_subject_dict_{'_'.join(embedding_types)}.pkl"
    missingness_summary = f"{base_dir}/benchmarking_missingness_summary_{'_'.join(embedding_types)}.csv"
    if not os.path.isfile(subject_dict_file_path):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            modalities, base_dir, thr_type, mets, embedding_types, template,
            sessions)
        sub_dict_clean = cleanNullTerms(subject_dict)
        final_missingness_summary = pd.concat([i for i in missingness_frames if isinstance(i, pd.DataFrame)])
        final_missingness_summary.to_csv(missingness_summary, index=False)
        with open(subject_dict_file_path, 'wb') as f:
            dill.dump(sub_dict_clean, f)
        f.close()
    else:
        with open(subject_dict_file_path, 'rb') as f:
            sub_dict_clean = dill.load(f)
        f.close()

    rsns = ['triple']
    # rsns = ['SalVentAttnA', 'DefaultA', 'ContB']

    ids = sub_dict_clean.keys()

    for modality in modalities:
        print(f"MODALITY: {modality}")
        hyperparams = eval(f"hyperparams_{modality}")
        hyperparam_dict = {}

        ensembles, df_top = get_ensembles_top(modality, thr_type,
                                              f"{base_dir}/pynets")

        grid = build_grid(modality, hyperparam_dict,
                          sorted(list(set(hyperparams))), ensembles)[1]

        for alg in embedding_types:
            print(f"EMBEDDING TYPE: {alg}")
            # if os.path.isfile(f"{base_dir}/grid_clean_{modality}_{alg}.csv"):
            #     continue

            par_dict = sub_dict_clean.copy()
            cache_dir = tempfile.mkdtemp()

            # with Parallel(
            #     n_jobs=-1,
            #     backend='loky',
            #     verbose=10,
            #     max_nbytes=None,
            #     temp_folder=cache_dir,
            # ) as parallel:
            #     outs = parallel(
            #         delayed(benchmark_reproducibility)(
            #             comb, modality, alg, par_dict,
            #             disc, int_consist,
            #         )
            #         for comb in grid
            #     )

            outs = []
            for comb in grid:
                outs.append(benchmark_reproducibility(
                    comb, modality, alg, par_dict,
                    disc, int_consist, final_missingness_summary,
                ))

            df_summary = pd.concat(outs, axis=0)
            df_summary = df_summary.dropna(axis=0, how='all')
            print(f"Saving to {base_dir}/grid_clean_{modality}_{alg}_"
                  f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
            df_summary.to_csv(f"{base_dir}/grid_clean"
                              f"_{modality}_{alg}_"
                              f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv", index=False)
