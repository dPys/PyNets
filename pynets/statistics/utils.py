#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017
"""
import matplotlib
import pandas as pd
import os
import glob
import numpy as np
import itertools
import warnings
from collections import OrderedDict
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

matplotlib.use('Agg')
warnings.simplefilter("ignore")


def mahalanobis_distances(X, y=None):
    """
    Compute the Mahalanobis distances between the training samples and the
    test samples.

    Parameters
    ----------
    X : array-like, shape (n_train_samples, n_features)
        Training data.

    y : array-like, shape (n_test_samples, n_features)
        Test data.

    Returns
    -------
    distances : array, shape (n_test_samples, n_train_samples)
        Mahalanobis distances between the test samples and the training samples.
    """
    from sklearn.metrics.pairwise import check_pairwise_arrays
    X, Y = check_pairwise_arrays(X, y)
    if y is None:
        y = X

    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("Different number of samples in X and y")
    X_train = X
    X_test = y
    # Subtract mean
    X_mean = np.mean(X_train, axis=0)
    X_train = X_train - X_mean
    X_test = X_test - X_mean
    # Compute the inverse of the covariance matrix
    # If the covariance matrix is singular, use the pseudo-inverse instead
    try:
        inv_cov = np.linalg.inv(np.cov(X_train.T))
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(np.cov(X_train.T))
    # Compute the dissimilarity matrix of the squared Mahalanobis distances
    sq_mahal_dist = np.zeros((X_test.shape[0], X_test.shape[0]))
    for i in range(n_samples):
        sq_mahal_dist[i, :] = np.sum(
            (np.dot(X_test, inv_cov) * X_test[i, :]) ** 2, axis=1)
    return np.sqrt(sq_mahal_dist)


class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=10.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=10.0):
        from statsmodels.stats.outliers_influence import \
            variance_inflation_factor
        dropped = True
        vif_cols = []
        while dropped:
            # Loop repeatedly until we find that all columns within our dataset
            # have a VIF value less than the threshold
            variables = X.columns
            dropped = False
            vif = []
            new_vif = 0
            for var in X.columns:
                new_vif = variance_inflation_factor(X[variables].values,
                                                    X.columns.get_loc(var))
                vif.append(new_vif)
                if np.isinf(new_vif):
                    break
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                vif_cols.append(X.columns.tolist()[maxloc])
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X, vif_cols


def preprocess_x_y(X, y, nuisance_cols, nodrop_columns=[],
                   var_thr=.85, remove_multi=True,
                   remove_outliers=True, standardize=True,
                   std_dev=3, vif_thr=10, missingness_thr=0.50,
                   zero_thr=0.50, oversample=False):
    from colorama import Fore, Style

    # Replace all near-zero with zeros
    # Drop excessively sparse columns with >zero_thr zeros
    if zero_thr > 0:
        X = X.apply(lambda x: np.where(np.abs(x) < 0.000001, 0, x))
        X_tmp = X.T.loc[(X == 0).sum() < (float(zero_thr)) * X.shape[0]].T

        if len(nodrop_columns) > 0:
            X = pd.concat([X_tmp, X[[i for i in X.columns if i in
                                     nodrop_columns and i not in
                                     X_tmp.columns]]], axis=1)
        else:
            X = X_tmp
        del X_tmp

        if X.empty or len(X.columns) < 5:
            print(f"\n\n{Fore.RED}Empty feature-space (Zero Columns): "
                  f"{X}{Style.RESET_ALL}\n\n")
            return X, y

    # Remove columns with excessive missing values
    X = X.dropna(thresh=len(X) * (1 - missingness_thr), axis=1)
    if X.empty:
        print(f"\n\n{Fore.RED}Empty feature-space (missingness): "
              f"{X}{Style.RESET_ALL}\n\n")
        return X, y

    # Apply a simple imputer (note that this assumes extreme cases of
    # missingness have already been addressed). The SimpleImputer is better
    # for smaller datasets, whereas the IterativeImputer performs best on
    # larger sets.

    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer
    # imp = IterativeImputer(random_state=0, sample_posterior=True)
    # X = pd.DataFrame(imp.fit_transform(X, y), columns=X.columns)
    imp1 = SimpleImputer()
    X = pd.DataFrame(imp1.fit_transform(X.astype('float32')),
                     columns=X.columns)

    # Deconfound X by any non-connectome columns present, and then remove them
    if len(nuisance_cols) > 0:
        deconfounder = DeConfounder()
        net_cols = [i for i in X.columns if i not in nuisance_cols]
        z_cols = [i for i in X.columns if i in nuisance_cols]
        if len(z_cols) > 0:
            deconfounder.fit(X[net_cols].values, X[z_cols].values)
            X[net_cols] = pd.DataFrame(deconfounder.transform(
                X[net_cols].values, X[z_cols].values),
                             columns=X[net_cols].columns)
            print(f"Deconfounding with {z_cols}...")
            X.drop(columns=z_cols, inplace=True)

    # Standardize X
    if standardize is True:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Remove low-variance columns
    sel = VarianceThreshold(threshold=var_thr)
    sel.fit(X)
    if len(nodrop_columns) > 0:
        good_var_cols = X.columns[np.concatenate(
            [sel.get_support(indices=True), np.array([X.columns.get_loc(c)
                                                      for c in
                                                      nodrop_columns if
                                                      c in X])])]
    else:
        good_var_cols = X.columns[sel.get_support(indices=True)]

    low_var_cols = [i for i in X.columns if i not in list(good_var_cols)]
    if len(low_var_cols) > 0:
        print(f"Dropping {low_var_cols} for low variance...")
    X = X[good_var_cols]

    if X.empty:
        print(f"\n\n{Fore.RED}Empty feature-space (low-variance): "
              f"{X}{Style.RESET_ALL}\n\n")
        return X, y

    # Remove outliers
    if remove_outliers is True:
        X, y = de_outlier(X, y, std_dev, 'IF')
        if X.empty and len(y) < 50:
            print(f"\n\n{Fore.RED}Empty feature-space (outliers): "
                  f"{X}{Style.RESET_ALL}\n\n")
            return X, y

    # Remove missing y
    # imp2 = SimpleImputer()
    # y = imp2.fit_transform(np.array(y).reshape(-1, 1))

    # y_missing_mask = np.invert(np.isnan(y))
    # X = X[y_missing_mask]
    # y = y[y_missing_mask]
    if X.empty or len(y) < 50:
        print(f"\n\n{Fore.RED}Empty feature-space (missing y): "
              f"{X}, {y}{Style.RESET_ALL}\n\n")
        return X, y

    # Remove multicollinear columns
    if remove_multi is True:
        try:
            rvif = ReduceVIF(thresh=vif_thr)
            X = rvif.fit_transform(X)[0]
            if X.empty or len(X.columns) < 5:
                print(f"\n\n{Fore.RED}Empty feature-space "
                      f"(multicollinearity): "
                      f"{X}{Style.RESET_ALL}\n\n")
                return X, y
        except:
            print(f"\n\n{Fore.RED}Empty feature-space (multicollinearity): "
                  f"{X}{Style.RESET_ALL}\n\n")
            return X, y

    if oversample is True:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42, sampling_strategy='auto')
            X, y = sm.fit_resample(X, y)
        except:
            pass

    print(f"\nX: {X}\ny: {y}\n")
    print(f"Features: {list(X.columns)}\n")
    return X, y


class DeConfounder(BaseEstimator, TransformerMixin):
    """ A transformer removing the effect of y on X using
    sklearn.linear_model.LinearRegression.

    References
    ----------
    D. Chyzhyk, G. Varoquaux, B. Thirion and M. Milham,
        "Controlling a confound in predictive models with a test set minimizing
        its effect," 2018 International Workshop on Pattern Recognition in
        Neuroimaging (PRNI), Singapore, 2018,
        pp. 1-4. doi: 10.1109/PRNI.2018.8423961
    """

    def __init__(self, confound_model=LinearRegression()):
        self.confound_model = confound_model

    def fit(self, X, z):
        if z.ndim == 1:
            z = z[:, np.newaxis]
        confound_model = clone(self.confound_model)
        confound_model.fit(z, X)
        self.confound_model_ = confound_model

        return self

    def transform(self, X, z):
        if z.ndim == 1:
            z = z[:, np.newaxis]
        X_confounds = self.confound_model_.predict(z)
        return X - X_confounds


def make_param_grids():
    param_space = {}
    param_space['Cs'] = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
    param_space['l1_ratios'] = [0, 0.25, 0.50, 0.75, 1]
    param_space['alphas'] = [1e-8, 1e-4, 1e-2, 1e-1, 0.25, 0.5, 1]
    return param_space


def get_ensembles_embedding(modality, alg, base_dir):
    if alg == "OMNI" or alg == "ASE":
        ensembles_pre = list(
            set(
                [
                    "subnet-"
                    + i.split('subnet-')[1].split("_")[0]
                    + "_granularity-"
                    + i.split('granularity-')[1].split("/")[0]
                    + "_"
                    + os.path.basename(i).split(modality + "_")[1].replace(
                        ".npy", "")
                    for i in glob.glob(
                        f"{base_dir}/pynets/sub-*/ses-*/{modality}/subnet-*/"
                        f"embeddings/gradient-{alg}*.npy"
                    )
                ]
            )
        )
        ensembles = []
        for i in ensembles_pre:
            if '_thrtype' in i:
                ensembles.append(i.split('_thrtype')[0])
            else:
                ensembles.append(i)
    elif alg == 'eigenvector' or alg == 'betweenness' or alg == 'degree' or \
            alg == 'local_efficiency' or alg == 'local_clustering':
        ensembles_pre = list(
            set(
                [
                    "subnet-"
                    + i.split('subnet-')[1].split("_")[0]
                    + "_granularity-"
                    + i.split('granularity-')[1].split("/")[0]
                    + "_"
                    + os.path.basename(i).split(modality + "_")[1].replace(
                        ".csv", "")
                    for i in glob.glob(
                        f"{base_dir}/pynets/sub-*/ses-*/{modality}/subnet-*/"
                        f"embeddings/gradient-{alg}*.csv"
                    )
                ]
            )
        )
        ensembles = []
        for i in ensembles_pre:
            if '_thrtype' in i:
                ensembles.append(i.split('_thrtype')[0])
            else:
                ensembles.append(i)
    else:
        ensembles = None
    return ensembles


def get_ensembles_top(modality, thr_type, base_dir, drop_thr=0.50):
    topology_file = f"{base_dir}/all_subs_neat_{modality}.csv"
    if os.path.isfile(topology_file):
        df_top = pd.read_csv(topology_file)
        if "Unnamed: 0" in df_top.columns:
            df_top.drop(df_top.filter(regex="Unnamed: 0"), axis=1,
                        inplace=True)
        df_top = df_top.dropna(subset=["id"])
        # df_top = df_top.rename(
        #     columns=lambda x: re.sub("_partcorr", "_model-partcorr", x)
        # )
        # df_top = df_top.rename(columns=lambda x: re.sub("_corr",
        #                                                 "_model-corr", x))
        # df_top = df_top.rename(columns=lambda x: re.sub("_cov",
        #                                                 "_model-cov", x))
        # df_top = df_top.rename(columns=lambda x: re.sub("_sfm",
        #                                                 "_model-sfm", x))
        # df_top = df_top.rename(columns=lambda x: re.sub("_csa",
        #                                                 "_model-csa", x))
        # df_top = df_top.rename(columns=lambda x: re.sub("_tensor",
        #                                                 "_model-tensor", x))
        # df_top = df_top.rename(columns=lambda x: re.sub("_csd",
        #                                                 "_model-csd", x))
        # df_top = df_top.rename(
        #     columns=lambda x: re.sub("thrtype-PROP", "thrtype-MST", x))
        # df_top = df_top.dropna(how='all')
        # df_top = df_top.dropna(axis='columns',
        #                        thresh=drop_thr * len(df_top)
        #                        )
        if not df_top.empty and len(df_top.columns) > 1:
            [df_top, ensembles] = graph_theory_prep(df_top, thr_type)
            # print(df_top)
            ensembles = [i for i in ensembles if i != "id"]
        else:
            ensembles = None
            df_top = None
    else:
        ensembles = None
        df_top = None
    return ensembles, df_top


def make_feature_space_dict(
    base_dir,
    ml_dfs,
    df,
    target_modality,
    subject_dict,
    ses,
    modality_grids,
    target_embedding_type,
    mets=None,
):
    from joblib import Parallel, delayed
    import tempfile
    import gc

    cache_dir = tempfile.mkdtemp()

    if target_modality not in ml_dfs.keys():
        ml_dfs[target_modality] = {}
    if target_embedding_type not in ml_dfs[target_modality].keys():
        ml_dfs[target_modality][target_embedding_type] = {}
    grid_params = list(set(modality_grids[target_modality]))

    grid_params_mod = []
    if target_modality == "func":
        for comb in grid_params:
            try:
                signal, hpass, model, granularity, parcellation, smooth = comb
                grid_params_mod.append((signal, hpass, model, granularity,
                                        parcellation, str(smooth)))
            except:
                try:
                    signal, hpass, model, granularity, parcellation = comb
                    smooth = "0"
                    grid_params_mod.append((signal, hpass, model, granularity,
                                            parcellation, str(smooth)))
                except:
                    print(f"Failed to parse: {comb}")

    elif target_modality == "dwi":
        for comb in grid_params:
            try:
                traversal, minlength, model, granularity, parcellation, \
                error_margin = comb
                grid_params_mod.append((traversal, minlength, model,
                                        granularity, parcellation,
                                        error_margin))
            except:
                print(f"Failed to parse: {comb}")


    with Parallel(
        n_jobs=-1, backend='loky', verbose=10, temp_folder=cache_dir
    ) as parallel:
        outs = parallel(
            delayed(create_feature_space)(
                base_dir,
                df,
                grid_param,
                subject_dict.copy(),
                ses,
                target_modality,
                target_embedding_type,
                mets
            )
            for grid_param in grid_params_mod
        )
    for fs, grid_param in outs:
        ml_dfs[target_modality][target_embedding_type][grid_param] = fs
        del fs, grid_param
    gc.collect()
    return ml_dfs


def build_grid(modality, hyperparam_dict, metaparams, ensembles):
    for ensemble in ensembles:
        try:
            build_mp_dict(ensemble, modality, hyperparam_dict, metaparams)
        except:
            print(f"Failed to parse ensemble {ensemble}...")

    if "subnet" in hyperparam_dict.keys():
        hyperparam_dict["subnet"] = [i for i in
                                     hyperparam_dict["subnet"] if "granularity"
                                     not in i]

    hyperparam_dict = OrderedDict(sorted(hyperparam_dict.items(),
                                         key=lambda x: x[0]))
    grid = list(
        itertools.product(*(hyperparam_dict[param] for param in
                            hyperparam_dict.keys()))
    )

    return hyperparam_dict, grid


def get_index_labels(base_dir, ID, ses, modality, parcellation, granularity,
                     emb_shape):

    node_files = glob.glob(
        f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/subnet-"
        f"{parcellation}_granularity-{granularity}/nodes/*.json")

    if len(node_files) > 0:
        ixs, node_dict = parse_closest_ixs(node_files, emb_shape)
    else:
        return [None]

    if emb_shape == len(ixs):
        return ixs
    else:
        return [None]


def save_netmets(
        dir_path,
        est_path,
        metric_list_names,
        net_met_val_list_final):
    from pynets.core import utils
    import os
    # And save results to csv
    out_path_neat = (
        f"{utils.create_csv_path(dir_path, est_path).split('.csv')[0]}"
        f"{'_neat.csv'}"
    )
    zipped_dict = dict(zip(metric_list_names, net_met_val_list_final))
    df = pd.DataFrame.from_dict(
        zipped_dict, orient="index", dtype="float32"
    ).transpose()
    if os.path.isfile(out_path_neat):
        os.remove(out_path_neat)
    df.to_csv(out_path_neat, index=False)
    del df, zipped_dict, net_met_val_list_final, metric_list_names

    return out_path_neat


def get_ixs_from_node_dict(node_dict):
    import ast
    if isinstance(node_dict, list):
        if all(v is None for v in
               [i['label'] for i in node_dict]):
            node_dict_revised = {}
            for i in range(len(node_dict)):
                node_dict_revised[i] = {}
                node_dict_revised[i]['label'], \
                    node_dict_revised[i]['index'] = ast.literal_eval(
                    node_dict[i]['index'].replace('\n', ','))
            ixs_corr = [int(k['index']) for k in
                        node_dict_revised.values()]
        elif all(isinstance(v, str) for v in
                 [i['label'] for i in node_dict]):
            node_dict_revised = {}
            for i in range(len(node_dict)):
                node_dict_revised[i] = {}
                node_dict_revised[i]['label'] = ast.literal_eval(
                    node_dict[i]['label'].replace('\n', ','))[0]
                node_dict_revised[i]['index'] = ast.literal_eval(
                    node_dict[i]['label'].replace('\n', ','))[1]
            ixs_corr = [int(k['index']) for k in
                        node_dict_revised.values()]
        elif all(isinstance(v, tuple) for v in
                 [i['label'] for i in node_dict]):
            node_dict_revised = {}
            for i in range(len(node_dict)):
                node_dict_revised[i] = {}
                node_dict_revised[i]['label'] = node_dict[i]['label'][0]
                node_dict_revised[i]['index'] = node_dict[i]['label'][1]
            ixs_corr = [int(k['index']) for k in
                        node_dict_revised.values()]
        else:
            ixs_corr = [int(i['index'])
                        for i
                        in node_dict]
            node_dict_revised = node_dict
    else:
        ixs_corr = [int(i['index'])
                    for i
                    in node_dict.values()]
        node_dict_revised = node_dict

    for i in range(len(node_dict)):
        node_dict_revised[i]['coord'] = node_dict[i]['coord']
    return ixs_corr, node_dict_revised


def node_files_search(node_files, emb_shape):
    import os
    import gc
    import json

    if len(node_files) == 1:
        with open(node_files[0],
                  'r+') as f:
            node_dict = json.load(f)
        f.close()
        ixs_corr, node_dict_revised = get_ixs_from_node_dict(node_dict)
    else:
        node_files = sorted(node_files, key=os.path.getmtime)
        try:
            with open(node_files[0],
                      'r+') as f:
                node_dict = json.load(
                    f)
            f.close()
            j = 0
        except:
            with open(node_files[1], 'r+') as f:
                node_dict = json.load(f)
            f.close()
            j = 1

        ixs_corr, node_dict_revised = get_ixs_from_node_dict(node_dict)

        while len(ixs_corr) != emb_shape and j < len(node_files):
            try:
                with open(node_files[j],
                          'r+') as f:
                    node_dict = json.load(
                        f)
                f.close()
            except:
                j += 1
                continue
            ixs_corr, node_dict_revised = get_ixs_from_node_dict(node_dict)
            j += 1
    del f
    gc.collect()

    return ixs_corr, node_dict_revised


def retrieve_indices_from_parcellation(node_files, emb_shape, template,
                                       vox_size='2mm'):
    from pathlib import Path
    dir_path = str(Path(node_files[0]).parent.parent)
    if template == 'any':
        import glob
        template_parcs = glob.glob(f"{dir_path}/parcellations/parcellation_"
                                  f"space-*.nii.gz")
        if len(template_parcs) > 1:
            sorted_template_parcs = sorted(template_parcs,
                                       key=os.path.getmtime)
            template_parc = sorted_template_parcs[0]
        else:
            template_parc = [0]
    else:
        template_parc = f"{dir_path}/parcellations/parcellation_space-" \
                        f"{template}.nii.gz"
    node_file = make_node_dict_from_parcellation(template_parc, dir_path,
                                                 vox_size)
    if os.path.isfile(template_parc):
        ixs_corr, node_dict = node_files_search([node_file], emb_shape)
        return ixs_corr, node_dict
    else:
        return [], {}


def make_node_dict_from_parcellation(parcellation, dir_path, vox_size='2mm'):
    from pynets.core.nodemaker import get_names_and_coords_of_parcels, \
        parcel_naming
    from pynets.core.utils import save_coords_and_labels_to_json
    coords, _, _, label_intensities = \
        get_names_and_coords_of_parcels(parcellation)
    labels = parcel_naming(coords, vox_size)
    node_file = save_coords_and_labels_to_json(coords, labels,
                                               dir_path, subnet='regen',
                                               indices=label_intensities)
    return node_file


def parse_closest_ixs(node_files, emb_shape, vox_size='2mm',
                      template='any'):
    if len(node_files) > 0:
        node_files_named = [i for i in node_files if f"{emb_shape}" in i]
        if len(node_files_named) > 0:
            node_files_named = sorted(node_files_named, key=os.path.getmtime)
            ixs_corr, node_dict = node_files_search(node_files_named,
                                                    emb_shape)
        else:
            ixs_corr, node_dict = node_files_search(node_files, emb_shape)

        if len(ixs_corr) != emb_shape:
            ixs_corr, node_dict = retrieve_indices_from_parcellation(
                node_files, emb_shape, template, vox_size)
        return ixs_corr, node_dict
    else:
        print(UserWarning('Node files empty. Attempting to retrieve manually '
                          'from parcellations...'))
        ixs_corr, node_dict = retrieve_indices_from_parcellation(node_files,
                                                                 emb_shape,
                                                                 template,
                                                                 vox_size)
        return ixs_corr, node_dict


def flatten_latent_positions(base_dir, subject_dict, ID, ses, modality,
                             grid_param, alg):

    if grid_param in subject_dict[ID][str(ses)][modality][alg].keys():
        rsn_dict = subject_dict[ID][str(ses)][modality][alg][grid_param]

        if 'data' in rsn_dict.keys():
            ixs = [i for i in rsn_dict['index'] if i is not None]

            if not isinstance(rsn_dict["data"], np.ndarray):
                if rsn_dict["data"].endswith('.npy'):
                    rsn_dict["data"] = np.load(rsn_dict["data"],
                                               allow_pickle=True)
                elif rsn_dict["data"].endswith('.csv'):
                    rsn_dict["data"] = np.array(pd.read_csv(rsn_dict["data"])
                                                ).reshape(-1, 1)

            emb_shape = rsn_dict["data"].shape[0]

            if len(ixs) != emb_shape:
                node_files = glob.glob(
                    f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/subnet-"
                    f"{grid_param[-2]}_granularity-{grid_param[-3]}/nodes/"
                    f"*.json")
                ixs, node_dict = parse_closest_ixs(node_files, emb_shape)

            if len(ixs) > 0:
                if len(ixs) == emb_shape:
                    rsn_arr = rsn_dict["data"].T.reshape(
                        1, rsn_dict["data"].T.shape[0] *
                        rsn_dict["data"].T.shape[1]
                    )
                    if rsn_dict["data"].shape[1] == 1:
                        df_lps = pd.DataFrame(rsn_arr,
                                              columns=[f"{i}_subnet-"
                                                       f"{grid_param[-2]}_"
                                                       f"granularity-"
                                                       f"{grid_param[-3]}_"
                                                       f"dim1" for i in ixs])
                    elif rsn_dict["data"].shape[1] == 3:
                        df_lps = pd.DataFrame(
                            rsn_arr,
                            columns=[f"{i}_subnet-{grid_param[-2]}_"
                                     f"granularity-{grid_param[-3]}_dim1" for
                                     i in ixs]
                            + [f"{i}_subnet-{grid_param[-2]}_"
                               f"granularity-{grid_param[-3]}_dim2" for
                               i in ixs]
                            + [f"{i}_subnet-{grid_param[-2]}_"
                               f"granularity-{grid_param[-3]}_dim3" for
                               i in ixs],
                        )
                    else:
                        df_lps = None
                    # else:
                    #     raise ValueError(
                    #         f"Number of dimensions {
                    #         rsn_dict['data'].shape[1]} "
                    #         f"not supported. See flatten_latent_positions "
                    #         f"function..."
                    #     )
                    # print(df_lps)
                else:
                    print(
                        f"Length of indices {len(ixs)} does not equal the "
                        f"number of rows {rsn_dict['data'].shape[0]} in the "
                        f"embedding-space for {ID} {ses} {modality} "
                        f"{grid_param}. This means that at some point a"
                        f" node index was dropped from the parcellation, but "
                        f"not from the final graph..."
                    )
                    df_lps = None
            else:
                print(UserWarning(f"Missing indices for "
                                  f"{grid_param} universe..."))
                df_lps = None
        else:
            print(UserWarning(f"Missing {grid_param} universe..."))
            df_lps = None
    else:
        print(UserWarning(f"Missing {grid_param} universe..."))
        df_lps = None

    return df_lps


def create_feature_space(base_dir, df, grid_param, subject_dict, ses,
                         modality, alg, mets=None):
    from colorama import Fore, Style
    from pynets.core.utils import load_runconfig
    df_tmps = []

    hardcoded_params = load_runconfig()
    embedding_methods = hardcoded_params["embed"]

    for ID in df["participant_id"]:
        if ID not in subject_dict.keys():
            print(f"ID: {ID} not found...")
            continue

        if str(ses) not in subject_dict[ID].keys():
            print(f"Session: {ses} not found for ID {ID}...")
            continue

        if modality not in subject_dict[ID][str(ses)].keys():
            print(f"Modality: {modality} not found for ID {ID}, "
                  f"ses-{ses}...")
            continue

        if alg not in subject_dict[ID][str(ses)][modality].keys():
            print(
                f"Modality: {modality} not found for ID {ID}, ses-{ses}, "
                f"{alg}..."
            )
            continue

        if grid_param not in subject_dict[ID][str(ses)][modality][alg].keys():
            print(
                f"Grid param {grid_param} not found for ID {ID}, ses-{ses}, "
                f"{alg} and {modality}..."
            )
            continue

        if alg != "topology" and alg in embedding_methods:
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Grid Param: {grid_param} "
                  f"found for {ID}")
            df_lps = flatten_latent_positions(
                base_dir, subject_dict, ID, ses, modality, grid_param, alg
            )
        else:
            if grid_param in subject_dict[ID][str(ses)][modality][alg].keys():
                df_lps = pd.DataFrame(
                    subject_dict[ID][str(ses)][modality][alg][grid_param].T,
                    columns=mets,
                )
            else:
                df_lps = None

        if df_lps is not None:
            df_tmp = (
                df[df["participant_id"] == ID]
                .reset_index()
                .drop(columns="index")
                .join(df_lps, how="right")
            )
            df_tmps.append(df_tmp)
            del df_tmp
        else:
            print(f"Feature-space null for ID {ID} & ses-{ses}, modality: "
                  f"{modality}, embedding: {alg}...")
            continue

    if len(df_tmps) > 0:
        dfs = [dff.set_index("participant_id"
                             ) for dff in df_tmps if not dff.empty]
        df_all = pd.concat(dfs, axis=0)
        # df_all = df_all.replace({0: np.nan})
        # df_all = df_all.apply(lambda x: np.where(x < 0.00001, np.nan, x))
        # print(len(df_all))
        del df_tmps
        return df_all, grid_param
    else:
        return pd.Series(np.nan), grid_param


def graph_theory_prep(df, thr_type, drop_thr=0.50):
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler
    cols = [
        j
        for j in set(
            [i.split("_thrtype-" + thr_type + "_")[0] for i in
             list(set(df.columns))]
        )
        if j != "id"
    ]

    id_col = df["id"]

    df = df.dropna(thresh=len(df) * drop_thr, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    imp = KNNImputer(n_neighbors=7)
    df = pd.DataFrame(
        imp.fit_transform(scaler.fit_transform(df[[i for i in
                                                   df.columns if i != "id"]])),
        columns=[i for i in df.columns if i != "id"],
    )

    df = pd.concat([id_col, df], axis=1)

    return df, cols


def split_df_to_dfs_by_prefix(df, prefixes=[]):
    from pynets.core.utils import flatten

    df_splits = []
    for p in prefixes:
        df_splits.append(df[list(set(list(flatten([c for c in df.columns if
                                                   c.startswith(p)]))))])
    # pref_selected = list(set(list(flatten([i.columns for i in df_splits]))))
    # df_other = df[[j for j in df.columns if j not in pref_selected]]
    #return df_splits + [df_other]

    return df_splits


def de_outlier(X, y, sd, deoutlier_type='IF'):
    """
    Remove any gross outlier row in X whose linear residual
    when regressing y against X is > sd standard deviations
    away from the mean residual. For classifiers, use a NaiveBayes estimator
    since it does not require tuning, and for regressors, use simple
    linear regression.

    """
    if deoutlier_type == 'IF':
        model = IsolationForest(random_state=42,
                                bootstrap=True, contamination='auto')
        outlier_mask = model.fit_predict(X)
        outlier_mask[outlier_mask == -1] = 0
        outlier_mask = outlier_mask.astype('bool')
    elif deoutlier_type == 'LOF':
        mask = np.zeros(X.shape, dtype=np.bool)
        model = LocalOutlierFactor(n_neighbors=10, metric='mahalanobis')
        model.fit_predict(X)
        X_scores = model.negative_outlier_factor_
        X_scores = (X_scores.max() - X_scores) / (
                X_scores.max() - X_scores.min())
        median_score = np.median(X_scores)
        outlier_mask = np.logical_or(
            [X_scores[i] > median_score for i in range(len(X.shape[0]))], mask)
        outlier_mask[outlier_mask == -1] = 0
        outlier_mask = outlier_mask.astype('bool')
    else:
        if deoutlier_type == 'NB':
            model = GaussianNB()
        elif deoutlier_type == 'LR':
            model = LinearRegression(normalize=True)
        else:
            raise ValueError('predict_type not recognized!')
        model.fit(X, y)
        predicted_y = model.predict(X)

        resids = (y - predicted_y)**2

        outlier_mask = (np.abs(stats.zscore(np.array(resids).reshape(-1, 1))) <
                        float(sd)).all(axis=1)

    return X[outlier_mask], y[outlier_mask]


def get_scorer_ens(scorer_name):
    import importlib
    import sklearn.metrics
    found = False

    try:
        scoring = sklearn.metrics.get_scorer(scorer_name)
        found = True
    except ValueError:
        pass

    if not found:
        i = scorer_name.rfind('.')
        if i < 0:
            raise ValueError(
                'Invalid scorer import path: {}'.format(scorer_name))
        module_name, scorer_name_ = scorer_name[:i], scorer_name[i + 1:]
        mod = importlib.import_module(module_name)
        scoring = getattr(mod, scorer_name_)
        found = True

    return scoring


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X.iloc[bootstrap_indices.tolist(), :], \
           y.iloc[bootstrap_indices.tolist(), :]


def bias_variance_decomp(estimator, X, y, loss='0-1_loss', num_rounds=200,
                         random_seed=None, **fit_params):
    """
    # Nonparametric Permutation Test
    # Author: Sebastian Raschka <sebastianraschka.com> from mlxtend
    (soon to be replaced by a formal dependency)

    Parameters
    ----------
    estimator : object
        A classifier or regressor object or class implementing both a
        `fit` and `predict` method similar to the scikit-learn API.
    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.
    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.
    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.
    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.
    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.
    num_rounds : int (default=200)
        Number of bootstrap rounds (sampling from the training set)
        for performing the bias-variance decomposition. Each bootstrap
        sample has the same size as the original training set.
    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.
    fit_params : additional parameters
        Additional parameters to be passed to the .fit() function of the
        estimator when it is fit to the bootstrap samples.

    Returns
    ----------
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        average bias, and average bias (all floats), where the average
        is computed over the data points in the test set.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/

    """
    supported = ['0-1_loss', 'mse']
    if loss not in supported:
        raise NotImplementedError('loss must be one of the following: %s' %
                                  supported)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.10,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=y)

    rng = np.random.RandomState(random_seed)

    if loss == '0-1_loss':
        dtype = np.int
    elif loss == 'mse':
        dtype = np.float

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=dtype)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)

        # Keras support
        if estimator.__class__.__name__ in ['Sequential', 'Functional']:

            # reset model
            for ix, layer in enumerate(estimator.layers):
                if hasattr(estimator.layers[ix], 'kernel_initializer') and \
                        hasattr(estimator.layers[ix], 'bias_initializer'):
                    weight_initializer = \
                        estimator.layers[ix].kernel_initializer
                    bias_initializer = estimator.layers[ix].bias_initializer

                    old_weights, old_biases = \
                        estimator.layers[ix].get_weights()

                    estimator.layers[ix].set_weights([
                        weight_initializer(shape=old_weights.shape),
                        bias_initializer(shape=len(old_biases))])

            estimator.fit(X_boot, y_boot, **fit_params)
            pred = estimator.predict(X_test).reshape(1, -1)
        else:
            pred = estimator.fit(
                X_boot, y_boot, **fit_params).predict(X_test)
        all_pred[i] = pred

    if loss == '0-1_loss':
        main_predictions = np.apply_along_axis(lambda x:
                                               np.argmax(np.bincount(x)),
                                               axis=0,
                                               arr=all_pred)

        avg_expected_loss = np.apply_along_axis(lambda x:
                                                (x != y_test.values).mean(),
                                                axis=1,
                                                arr=all_pred).mean()

        avg_bias = np.sum(main_predictions != y_test.values) / \
                   y_test.values.size

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int)
        var /= num_rounds

        avg_var = var.sum()/y_test.shape[0]

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x:
            ((x - y_test.values)**2).mean(),
            axis=1,
            arr=all_pred).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test.values)**2) / \
                   y_test.values.size
        avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var

def make_subject_dict(
        modalities, base_dir, thr_type, mets, embedding_types, template,
        sessions, rsns, IDS=None):
    from joblib.externals.loky import get_reusable_executor
    from joblib import Parallel, delayed
    from pynets.core.utils import mergedicts
    from pynets.core.utils import load_runconfig
    import tempfile
    import psutil
    import shutil
    import gc

    hardcoded_params = load_runconfig()
    embedding_methods = hardcoded_params["embed"]
    metaparams_func = hardcoded_params["metaparams_func"]
    metaparams_dwi = hardcoded_params["metaparams_dwi"]

    miss_frames_all = []
    subject_dict_all = {}
    modality_grids = {}
    for modality in modalities:
        print(f"MODALITY: {modality}")
        metaparams = eval(f"metaparams_{modality}")
        for alg in embedding_types:
            print(f"EMBEDDING TYPE: {alg}")
            for ses_name in sessions:
                if not IDS:
                    ids = [
                        f"{os.path.basename(i)}_ses-{ses_name}"
                        for i in glob.glob(f"{base_dir}/pynets/*")
                        if os.path.basename(i).startswith("sub")
                    ]
                else:
                    ids = IDS

                if alg != "topology" and alg in embedding_methods:
                    df_top = None
                    ensembles = get_ensembles_embedding(modality, alg,
                                                        base_dir)
                    if ensembles is None:
                        print("No ensembles found.")
                        continue
                elif alg == "topology":
                    ensembles, df_top = get_ensembles_top(
                        modality, thr_type, f"{base_dir}/pynets"
                    )
                    if "missing" in df_top.columns:
                        df_top.drop(columns="missing", inplace=True)

                    if ensembles is None or df_top is None:
                        print("Missing topology outputs.")
                        continue
                else:
                    continue

                ensembles = list(set([i for i in ensembles if i is not None]))

                hyperparam_dict = {}

                grid = build_grid(
                    modality, hyperparam_dict, sorted(list(set(metaparams))),
                    ensembles)[1]

                grid = list(set([i for i in grid if i != () and
                                 len(list(i)) > 0]))

                grid = [i for i in grid if any(n in i for n in rsns)]

                modality_grids[modality] = grid

                par_dict = subject_dict_all.copy()
                cache_dir = tempfile.mkdtemp()

                max_bytes = int(float(list(
                    psutil.virtual_memory())[4]/len(ids)))
                with Parallel(
                    n_jobs=-1,
                    backend='loky',
                    verbose=1,
                    max_nbytes=f"{max_bytes}M",
                    temp_folder=cache_dir,
                ) as parallel:
                    outs_tup = parallel(
                        delayed(populate_subject_dict)(
                            id,
                            modality,
                            grid,
                            par_dict,
                            alg,
                            base_dir,
                            template,
                            thr_type,
                            embedding_methods,
                            mets,
                            df_top
                        )
                        for id in ids
                    )
                outs = [i[0] for i in outs_tup]
                miss_frames = [i[1] for i in outs_tup if not i[1].empty]
                if len(miss_frames) > 1:
                    miss_frames = pd.concat(miss_frames)
                miss_frames_all.append(miss_frames)
                for d in outs:
                    subject_dict_all = dict(mergedicts(subject_dict_all, d))
                shutil.rmtree(cache_dir, ignore_errors=True)
                get_reusable_executor().shutdown(wait=True)
                del par_dict, outs_tup, outs, df_top, miss_frames, ses_name, \
                    grid, hyperparam_dict, parallel
                gc.collect()
            del alg
        del metaparams
    del modality
    gc.collect()

    return subject_dict_all, modality_grids, miss_frames_all


def populate_subject_dict(
    id,
    modality,
    grid,
    subject_dict,
    alg,
    base_dir,
    template,
    thr_type,
    embedding_methods,
    mets=None,
    df_top=None,
):
    from colorama import Fore, Style
    import gc

    # print(id)
    ID = id.split("_")[0].split("sub-")[1]
    ses = id.split("_")[1].split("ses-")[1]

    completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"

    if ID not in subject_dict.keys():
        subject_dict[ID] = {}

    if ses not in subject_dict[ID].keys():
        subject_dict[ID][str(ses)] = {}

    if modality not in subject_dict[ID][str(ses)].keys():
        subject_dict[ID][str(ses)][modality] = {}

    if alg not in subject_dict[ID][str(ses)][modality].keys():
        subject_dict[ID][str(ses)][modality][alg] = {}

    subject_dict[ID][str(ses)][modality][alg] = dict.fromkeys(grid, np.nan)

    missingness_frame = pd.DataFrame(columns=["id", "ses", "modality", "alg",
                                              "grid"])

    # Functional case
    if modality == "func":
        # with Parallel(
        #     n_jobs=4,
        #     require='sharedmem',
        #     verbose=1,
        # ) as parallel:
        #     parallel(
        #         delayed(func_grabber)(comb, subject_dict, missingness_frame,
        #                               ID, ses, modality, alg, mets, thr_type,
        #                               base_dir,
        #                               template,
        #                               df_top, embedding_methods)
        #         for comb in grid
        #     )
        for comb in grid:
            [subject_dict, missingness_frame] = func_grabber(comb,
                                                             subject_dict,
                                                             missingness_frame,
                                                             ID, ses,
                                                             modality, alg,
                                                             mets,
                                                             thr_type,
                                                             base_dir,
                                                             template, df_top,
                                                             embedding_methods)
            gc.collect()
    # Structural case
    elif modality == "dwi":
        # with Parallel(
        #     n_jobs=4,
        #     require='sharedmem',
        #     verbose=1,
        # ) as parallel:
        #     parallel(
        #         delayed(dwi_grabber)(comb, subject_dict, missingness_frame,
        #                               ID, ses, modality, alg, mets, thr_type,
        #                               base_dir,
        #                               template,
        #                               df_top, embedding_methods)
        #         for comb in grid
        #     )
        for comb in grid:
            [subject_dict, missingness_frame] = dwi_grabber(comb, subject_dict,
                                                            missingness_frame,
                                                            ID, ses, modality,
                                                            alg, mets,
                                                            thr_type,
                                                            base_dir,
                                                            template, df_top,
                                                            embedding_methods)
            gc.collect()
    del modality, ID, ses, df_top
    gc.collect()
    return subject_dict, missingness_frame


def dwi_grabber(comb, subject_dict, missingness_frame,
                ID, ses, modality, alg, mets, thr_type, base_dir, template,
                df_top, embedding_methods):
    import gc
    from pynets.core.utils import filter_cols_from_targets
    from colorama import Fore, Style

    try:
        traversal, minlength, model, granularity, parcellation, \
        error_margin = comb
    except BaseException:
        print(UserWarning(f"{Fore.YELLOW}Failed to parse: "
                          f"{comb}{Style.RESET_ALL}"))
        return subject_dict, missingness_frame

    #comb_tuple = (parcellation, traversal, minlength, model, granularity, error_margin)
    comb_tuple = comb

    # print(comb_tuple)
    subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
    if alg != "topology" and alg in embedding_methods:
        embeddings = glob.glob(
            f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/subnet-"
            f"{parcellation}_granularity-"
            f"{granularity}/"
            f"embeddings/gradient-{alg}*"
        )

        if template == 'any':
            embeddings = [i for i in embeddings if (alg in i) and
                          (f"granularity-{granularity}" in i) and
                          (f"subnet-{parcellation}" in i) and
                          (f"model-{model}" in i) and
                          (f"traversal-{traversal}" in i) and
                          (f"minlength-{minlength}" in i) and
                          (f"tol-{error_margin}" in i) and ('_NULL' not in i)]
        else:
            embeddings = [i for i in embeddings if (alg in i) and
                          (f"granularity-{granularity}" in i) and
                          (f"subnet-{parcellation}" in i) and
                          (f"template-{template}" in i) and
                          (f"model-{model}" in i) and
                          (f"traversal-{traversal}" in i) and
                          (f"minlength-{minlength}" in i) and
                          (f"tol-{error_margin}" in i) and ('_NULL' not in i)]

        if len(embeddings) == 0:
            print(
                f"{Fore.YELLOW}Structural embedding not found for ID: {ID}, "
                f"SESSION: {ses}, EMBEDDING: {alg}, and UNIVERSE: "
                f"{comb_tuple}...{Style.RESET_ALL}"
            )
            missingness_frame = missingness_frame.append(
                {
                    "id": ID,
                    "ses": ses,
                    "modality": modality,
                    "alg": alg,
                    "grid": comb_tuple,
                },
                ignore_index=True,
            )
            return subject_dict, missingness_frame
        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            embeddings_raw = [i for i in embeddings if "thrtype" not
                              in i or 'thr-1.0' in i or 'thr-' not in i]
            if len(embeddings_raw) == 1:
                embedding = embeddings_raw[0]
            else:
                embeddings_raw = [i for i in embeddings_raw if
                                  (f"/subnet-{parcellation}_"
                                   f"granularity-{granularity}/" in i) and
                                  (parcellation in os.path.basename(i)) and
                                  (granularity in os.path.basename(i))]
                if len(embeddings_raw) > 0:
                    sorted_embeddings = sorted(embeddings_raw,
                                               key=lambda x: int(
                        x.partition('samples-')[2].partition('streams')[0]),
                           reverse=False)
                    # TODO: Change "reverse" above to True to grab the MOST
                    #  number of samples (ideal).

                    sorted_embeddings = sorted(sorted_embeddings,
                                               key=os.path.getmtime)
                    embedding = sorted_embeddings[0]
                    print(
                        f"Multiple structural embeddings found for {ID} and"
                        f" {comb_tuple}:\n{sorted_embeddings}\nTaking the most"
                        f" recent with the largest number of samples "
                        f"{embedding}..."
                    )
                else:
                    return subject_dict, missingness_frame
        if os.path.isfile(embedding):
            # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
            try:
                if embedding.endswith('.npy'):
                    emb_shape = np.load(embedding, allow_pickle=True,
                                        mmap_mode=None).shape[0]
                elif embedding.endswith('.csv'):
                    with open(embedding, "r+") as a:
                        emb_shape = len(pd.read_csv(a).columns)
                    a.close()
                else:
                    raise NotImplementedError(f"Format of {embedding} "
                                              f"not recognized! "
                                              f"Only .npy and .csv "
                                              f"currently supported.")
                gc.collect()
            except:
                print(
                    f"{Fore.RED}Failed to load structural embeddings found "
                    f"for ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, and "
                    f"UNIVERSE: {comb_tuple}...{Style.RESET_ALL}"
                )
                missingness_frame = missingness_frame.append(
                    {
                        "id": ID,
                        "ses": ses,
                        "modality": modality,
                        "alg": alg,
                        "grid": comb_tuple,
                    },
                    ignore_index=True,
                )
                return subject_dict, missingness_frame
            try:
                ixs = get_index_labels(base_dir, ID, ses, modality,
                                       parcellation, granularity, emb_shape)
            except BaseException:
                print(f"{Fore.LIGHTYELLOW_EX}Failed to load indices for "
                      f"{embedding}{Style.RESET_ALL}")
                return subject_dict, missingness_frame

            if not isinstance(
                    subject_dict[ID][str(ses)][modality][alg][comb_tuple],
                dict):
                subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
            subject_dict[ID][str(ses)][modality][alg][comb_tuple]["index"] = \
                ixs
            # subject_dict[ID][str(ses)][modality][alg][comb_tuple]["labels"]
            # = labels
            subject_dict[ID][str(ses)][modality][alg][comb_tuple][
                "data"] = embedding
            # print(data)
            completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}")
        else:
            print(
                f"{Fore.YELLOW}Structural embedding not found for ID: {ID}, "
                f"SESSION: {ses}, EMBEDDING: {alg}, and UNIVERSE: "
                f"{comb_tuple}...{Style.RESET_ALL}"
            )
            missingness_frame = missingness_frame.append(
                {
                    "id": ID,
                    "ses": ses,
                    "modality": modality,
                    "alg": alg,
                    "grid": comb_tuple,
                },
                ignore_index=True,
            )
            return subject_dict, missingness_frame
    elif alg == "topology":
        data = np.empty([len(mets), 1], dtype=np.float32)
        data[:] = np.nan
        targets = [
            f"minlength-{minlength}",
            f"traversal-{traversal}",
            f"model-{model}",
            f"granularity-{granularity}",
            f"subnet-{parcellation}",
            f"tol-{error_margin}",
            f"thrtype-{thr_type}",
        ]

        cols = filter_cols_from_targets(df_top, targets)
        i = 0
        for met in mets:
            col_met = [j for j in cols if met in j]
            if len(col_met) == 1:
                col = col_met[0]
            elif len(col_met) > 1:
                print(f"Multiple columns detected: {col_met}")
                col = col_met[0]
            else:
                data[i] = np.nan
                i += 1
                missingness_frame = missingness_frame.append(
                    {
                        "id": ID,
                        "ses": ses,
                        "modality": modality,
                        "alg": alg,
                        "grid": comb_tuple,
                    },
                    ignore_index=True,
                )
                continue
            out = df_top[df_top["id"] == "sub-" + ID + "_ses-" + ses][
                col
            ].values
            if len(out) == 0:
                print(
                    f"Structural topology not found for {ID}, "
                    f"{met}, and {comb_tuple}..."
                )
                print(f"{Fore.YELLOW}Missing metric {met} for ID: {ID}, "
                      f"SESSION: {ses}{Style.RESET_ALL}")
                data[i] = np.nan
            else:
                data[i] = out

            del col, out
            i += 1
        if (np.abs(data) < 0.0000001).all():
            data[:] = np.nan
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, COMPLETENESS: {completion_status}")
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
            completion_status = f"{Fore.YELLOW}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}")
        subject_dict[ID][str(ses)][modality][alg][comb_tuple] = data
        # save_embed_data_to_sql(data, ixs, ID, str(ses), modality, alg,
        # comb_tuple)
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame


def func_grabber(comb, subject_dict, missingness_frame,
                 ID, ses, modality, alg, mets, thr_type, base_dir, template,
                 df_top, embedding_methods):
    import gc
    from pynets.core.utils import filter_cols_from_targets
    from colorama import Fore, Style

    try:
        signal, hpass, model, granularity, parcellation, smooth = comb
    except:
        try:
            signal, hpass, model, granularity, parcellation = comb
            smooth = "0"
        except BaseException:
            print(UserWarning(f"{Fore.YELLOW}Failed to parse: "
                              f"{comb}{Style.RESET_ALL}"))
            return subject_dict, missingness_frame
    # comb_tuple = (parcellation, signal, hpass, model, granularity, str(smooth))
    comb_tuple = comb

    # print(comb_tuple)
    subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
    if alg != "topology" and alg in embedding_methods:
        embeddings = glob.glob(
            f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/"
            f"subnet-{parcellation}_granularity-{granularity}/"
            f"embeddings/gradient-{alg}*"
        )

        if template == 'any':
            embeddings = [i for i in embeddings if ((alg in i) and
                                                    (f"granularity-"
                                                     f"{granularity}" in i)
                                                    and
                                                    (f"subnet-{parcellation}"
                                                     in i) and
                                                    (f"model-{model}" in i)
                                                    and (f"hpass-{hpass}Hz"
                                                         in i)
                                                    and (f"signal-{signal}" in
                                                         i)) and ('_NULL' not
                                                                  in i)]
        else:
            embeddings = [i for i in embeddings if ((alg in i) and
                                                    (f"granularity-"
                                                     f"{granularity}" in i)
                                                    and
                                                    (f"subnet-{parcellation}"
                                                     in i) and
                                                    (f"template-{template}"
                                                     in i) and
                                                    (f"model-{model}" in i)
                                                    and (f"hpass-{hpass}Hz"
                                                         in i)
                                                    and (f"signal-{signal}" in
                                                         i)) and ('_NULL' not
                                                                  in i)]

        if smooth == "0":
            embeddings = [
                i
                for i in embeddings
                if "smooth" not in i
            ]
        else:
            embeddings = [
                i
                for i in embeddings
                if f"tol-{smooth}fwhm" in i
            ]

        if len(embeddings) == 0:
            print(
                f"{Fore.YELLOW}No functional embeddings found for ID: {ID}, "
                f"SESSION: {ses}, EMBEDDING: {alg}, and UNIVERSE: "
                f"{comb_tuple}...{Style.RESET_ALL}"
            )
            missingness_frame = missingness_frame.append(
                {
                    "id": ID,
                    "ses": ses,
                    "modality": modality,
                    "alg": alg,
                    "grid": comb_tuple,
                },
                ignore_index=True,
            )
            return subject_dict, missingness_frame

        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            embeddings_raw = [i for i in embeddings if "thrtype"
                              not in i or 'thr-1.0' in i or 'thr-' not in i]
            if len(embeddings_raw) == 1:
                embedding = embeddings_raw[0]
            else:
                embeddings_raw = [i for i in embeddings_raw if
                                  f"/subnet-{parcellation}_" \
                                  f"granularity-{granularity}/"
                                  in i and (parcellation in
                                            os.path.basename(i))
                                  and (granularity in os.path.basename(i))]
                if len(embeddings_raw) > 0:
                    sorted_embeddings = sorted(embeddings_raw,
                                               key=os.path.getmtime)
                    embedding = sorted_embeddings[0]
                    print(
                        f"Multiple structural embeddings found for {ID} and"
                        f" {comb_tuple}:\n{sorted_embeddings}\nTaking the most"
                        f" recent with the largest number of samples "
                        f"{embedding}..."
                    )
                else:
                    return subject_dict, missingness_frame
        if os.path.isfile(embedding):
            # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
            try:
                if embedding.endswith('.npy'):
                    emb_shape = np.load(embedding, allow_pickle=True,
                                        mmap_mode=None).shape[0]
                elif embedding.endswith('.csv'):
                    with open(embedding, "r+") as a:
                        emb_shape = len(pd.read_csv(a).columns)
                    a.close()
                else:
                    raise NotImplementedError(f"Format of {embedding} "
                                              f"not recognized! "
                                              f"Only .npy and .csv "
                                              f"currently supported.")
                gc.collect()
            except:
                print(
                    f"{Fore.RED}Failed to load functional embeddings found "
                    f"for ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, and "
                    f"UNIVERSE: {comb_tuple}...{Style.RESET_ALL}"
                )
                missingness_frame = missingness_frame.append(
                    {
                        "id": ID,
                        "ses": ses,
                        "modality": modality,
                        "alg": alg,
                        "grid": comb_tuple,
                    },
                    ignore_index=True,
                )
                return subject_dict, missingness_frame
            try:
                ixs = get_index_labels(base_dir, ID, ses, modality,
                                       parcellation, granularity, emb_shape)
            except BaseException:
                print(f"{Fore.LIGHTYELLOW_EX}Failed to load indices for "
                      f"{embedding} {Style.RESET_ALL}")
                return subject_dict, missingness_frame
            if not isinstance(
                    subject_dict[ID][str(ses)][modality][alg][comb_tuple],
                dict):
                subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
            subject_dict[ID][str(ses)][modality][alg][comb_tuple]["index"] = \
                ixs
            # subject_dict[ID][str(ses)][modality][alg][comb_tuple]["labels"]
            # = labels
            subject_dict[ID][str(ses)][modality][alg][comb_tuple][
                "data"] = embedding
            # print(data)
            completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}")
        else:
            print(
                f"{Fore.YELLOW}Functional embedding not found for ID: {ID}, "
                f"SESSION: {ses}, EMBEDDING: {alg}, and UNIVERSE: "
                f"{comb_tuple}...{Style.RESET_ALL}"
            )
            missingness_frame = missingness_frame.append(
                {
                    "id": ID,
                    "ses": ses,
                    "modality": modality,
                    "alg": alg,
                    "grid": comb_tuple,
                },
                ignore_index=True,
            )
            return subject_dict, missingness_frame

    elif alg == "topology":
        data = np.empty([len(mets), 1], dtype=np.float32)
        data[:] = np.nan
        if smooth == '0':
            targets = [
                f"signal-{signal}",
                f"hpass-{hpass}Hz",
                f"model-{model}",
                f"granularity-{granularity}",
                f"subnet-{parcellation}",
                f"thrtype-{thr_type}",
            ]
        else:
            targets = [
                f"signal-{signal}",
                f"hpass-{hpass}Hz",
                f"model-{model}",
                f"granularity-{granularity}",
                f"subnet-{parcellation}",
                f"tol-{smooth}fwhm",
                f"thrtype-{thr_type}",
            ]

        cols = filter_cols_from_targets(df_top, targets)
        i = 0
        for met in mets:
            col_met = [j for j in cols if met in j]
            if len(col_met) == 1:
                col = col_met[0]
            elif len(col_met) > 1:
                if comb_tuple[-1] == '0':
                    col = [i for i in col_met if "fwhm" not in i][0]
                else:
                    print(f"Multiple columns detected: {col_met}")
                    col = col_met[0]
            else:
                data[i] = np.nan
                i += 1
                missingness_frame = missingness_frame.append(
                    {
                        "id": ID,
                        "ses": ses,
                        "modality": modality,
                        "alg": alg,
                        "grid": comb_tuple,
                    },
                    ignore_index=True,
                )
                continue
            out = df_top[df_top["id"] == f"sub-{ID}_ses-{ses}"][col].values
            if len(out) == 0:
                print(
                    f"Functional topology not found for {ID}, {met}, "
                    f"and {comb_tuple}..."
                )
                print(f"{Fore.YELLOW}Missing metric {met} for ID: {ID}, "
                      f"SESSION: {ses}{Style.RESET_ALL}")
                data[i] = np.nan
            else:
                data[i] = out

            del col, out
            i += 1
        if (np.abs(data) < 0.0000001).all():
            data[:] = np.nan
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}")
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
            completion_status = f"{Fore.YELLOW}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}")
        subject_dict[ID][str(ses)][modality][alg][comb_tuple] = data
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame


def cleanNullTerms(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = cleanNullTerms(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif v is not None and v is not np.nan and not \
                isinstance(v, pd.Series):
            clean[k] = v
    return clean


def gen_sub_vec(base_dir, sub_dict_clean, ID, modality, alg, comb_tuple):
    vects = []
    for ses in sub_dict_clean[ID].keys():
        # print(ses)
        if comb_tuple in sub_dict_clean[ID][str(ses)][modality][alg].keys():
            if alg == 'topology':
                vect = sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple]
            else:
                vect = flatten_latent_positions(base_dir, sub_dict_clean, ID,
                                                ses, modality, comb_tuple, alg)
            vects.append(vect)
    vects = [i for i in vects if i is not None and
             not np.isnan(np.array(i)).all()]
    if len(vects) > 0 and alg == 'topology':
        out = np.concatenate(vects, axis=1)
    elif len(vects) > 0:
        out = pd.concat(vects, axis=0)
        del vects
    else:
        out = None
    # print(out)
    return out


def tuple_insert(tup, pos, ele):
    tup = tup[:pos] + (ele,) + tup[pos:]
    return tup


def build_mp_dict(file_renamed, modality, hyperparam_dict, metaparams):
    """
    A function to build a metaparameter dictionary by parsing a given
    file path.
    """

    for hyperparam in metaparams:
        if (
            hyperparam != "smooth"
            and hyperparam != "hpass"
            and hyperparam != "track_type"
            and hyperparam != "traversal"
            and hyperparam != "smooth"
            and hyperparam != "minlength"
            and hyperparam != "samples"
            and hyperparam != "nodetype"
            and hyperparam != "template"
            and hyperparam != "signal"

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
        if "tol-" in file_renamed:
            if "smooth" not in hyperparam_dict.keys():
                hyperparam_dict["smooth"] = [str(file_renamed.split(
                    "tol-")[1].split("_")[0].split("fwhm")[0])]
            else:
                hyperparam_dict["smooth"].append(str(file_renamed.split(
                    "tol-")[1].split("_")[0].split("fwhm")[0]))
        else:
            if 'smooth' not in hyperparam_dict.keys():
                hyperparam_dict['smooth'] = [str(0)]
            hyperparam_dict["smooth"].append(str(0))
            metaparams.append("smooth")
        if "hpass-" in file_renamed:
            if "hpass" not in hyperparam_dict.keys():
                hyperparam_dict["hpass"] = [str(file_renamed.split(
                    "hpass-")[1].split("_")[0].split("Hz")[0])]
            else:
                hyperparam_dict["hpass"].append(
                    str(file_renamed.split("hpass-"
                                           )[1].split("_")[0].split("Hz")[0]))
            metaparams.append("hpass")
        if "signal-" in file_renamed:
            if "signal" not in hyperparam_dict.keys():
                hyperparam_dict["signal"] = [
                    str(file_renamed.split("signal-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["signal"].append(
                    str(file_renamed.split("signal-")[1].split("_")[0])
                )
            metaparams.append("signal")

    elif modality == "dwi":
        if "traversal-" in file_renamed:
            if "traversal" not in hyperparam_dict.keys():
                hyperparam_dict["traversal"] = [
                    str(file_renamed.split("traversal-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["traversal"].append(
                    str(file_renamed.split("traversal-")[1].split("_")[0])
                )
            metaparams.append("traversal")
        if "minlength-" in file_renamed:
            if "minlength" not in hyperparam_dict.keys():
                hyperparam_dict["minlength"] = [
                    str(file_renamed.split("minlength-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["minlength"].append(
                    str(file_renamed.split("minlength-")[1].split("_")[0])
                )
            metaparams.append("minlength")
        if "tol-" in file_renamed:
            if "error_margin" not in hyperparam_dict.keys():
                hyperparam_dict["error_margin"] = [
                    str(file_renamed.split("tol-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["error_margin"].append(
                    str(file_renamed.split("tol-")[1].split("_")[0])
                )
            metaparams.append("error_margin")

    for key in hyperparam_dict:
        hyperparam_dict[key] = list(set(hyperparam_dict[key]))

    return hyperparam_dict, metaparams
