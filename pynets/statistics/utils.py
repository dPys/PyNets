"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner
"""
import glob
import itertools
import os
import typing
import warnings
from collections import OrderedDict

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

matplotlib.use("Agg")
warnings.simplefilter("ignore")


def mahalanobis_distances(
    X: np.ndarray, y: typing.Optional[np.ndarray] = None
) -> np.ndarray:
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
            (np.dot(X_test, inv_cov) * X_test[i, :]) ** 2, axis=1
        )
    return np.sqrt(sq_mahal_dist)


def slice_by_corr(X, r_min=0):
    # Create correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    # Find features with correlation greater than r_min
    return X[[column for column in upper.columns if any(upper[column] > r_min)]]


def variance_inflation_factor(X, exog_idx):
    clf = LinearRegression(fit_intercept=True)
    sub_X = np.delete(np.nan_to_num(X), exog_idx, axis=1)
    sub_y = X[:, exog_idx][np.newaxis].T
    sub_clf = clf.fit(sub_X, sub_y)
    return 1 / (1 - r2_score(sub_y, sub_clf.predict(sub_X)))


def preprocess_x_y(
    X: typing.Union[np.ndarray, pd.DataFrame],
    y: typing.Union[np.ndarray, pd.DataFrame],
    nuisance_cols: list,
    nodrop_columns: list = [],
    var_thr: float = 0.95,
    remove_multi: bool = True,
    remove_outliers: bool = True,
    standardize: bool = True,
    standard_method: str = "ss",
    std_dev: int = 3,
    vif_thr: int = 10,
    missingness_thr: float = 0.50,
    zero_thr: float = 0.50,
    oversample: bool = False,
) -> typing.Tuple[
    typing.Union[np.ndarray, pd.DataFrame],
    typing.Union[np.ndarray, pd.DataFrame],
]:
    """

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data matrix.
    y : np.ndarray or pd.DataFrame
        Target vector.
    nuisance_cols : list
        List of nuisance columns to remove.
    nodrop_columns : list, optional
        List of columns in X to retain regardless of other constraints,
        by default [].
    var_thr : float, optional
        A threshold for invariance above which a column in X will be dropped,
        by default 0.95.
    remove_multi : bool, optional
        Whether to drop columns of X on the basis of multicollinearity,
        by default True.
    remove_outliers : bool, optional
        Whether to drop rows of X if they have meet the criteria of the detection method,
        by default True.
    outlier_removal_method : str, optional
        The method to use for outlier removal, by default `IF` (isolation forest).
        Other options are `LOF` (local outlier factor), `NB` (naive bayes), and
        `LR` (linear regression).
    standardize : bool, optional
        Whether to standardize the values for each column of X, by default True.
    standard_method : str, optional
        The method to use for standardization. Options are `ss` for StandardScaler and `mm`
        for MinMaxScaler. Default is `ss`.
    std_dev : int, optional
        Number of standard deviations used to establish what counts as an outlier,
        by default 3.
    vif_thr : int, optional
        Variable Inflation Factor threshold. VIF=10 is considered liberal,
        whereas VIF=5 is considered conservative. Default is 10.
    missingness_thr : float, optional
        Threshold for percentage of rows whose values for a particular column in X can be missing,
        above which the respective column will be dropped, by default 0.50.
    zero_thr : float, optional
        Threshold for percentage of rows whose values for a particular column in X are zero,
        above which the respective column will be dropped, by default 0.50.
    oversample : bool, optional
        Whether to apply an oversampling method to correct for imbalanced classes in `y`,
        by default False.

    Returns
    -------
    X : np.ndarray or pd.DataFrame
        Preprocessed data matrix.
    y : np.ndarray or pd.DataFrame
        Preprocessed target vector.

    """
    from colorama import Fore, Style
    from pynets.statistics.interfaces import ReduceVIF, DeConfounder

    # Replace all near-zero with zeros
    # Drop excessively sparse columns with >zero_thr zeros
    if zero_thr > 0:
        X = X.apply(lambda x: np.where(np.abs(x) < 0.000001, 0, x))
        X_tmp = X.T.loc[(X == 0).sum() < (float(zero_thr)) * X.shape[0]].T

        if len(nodrop_columns) > 0:
            X = pd.concat(
                [
                    X_tmp,
                    X[
                        [
                            i
                            for i in X.columns
                            if i in nodrop_columns and i not in X_tmp.columns
                        ]
                    ],
                ],
                axis=1,
            )
        else:
            X = X_tmp
        del X_tmp

        if X.empty or len(X.columns) < 5:
            print(
                f"\n\n{Fore.RED}Empty feature-space (Zero Columns): "
                f"{X}{Style.RESET_ALL}\n\n"
            )
            return X, y

    # Remove columns with excessive missing values
    X = X.dropna(thresh=len(X) * (1 - missingness_thr), axis=1)
    if X.empty:
        print(
            f"\n\n{Fore.RED}Empty feature-space (missingness): "
            f"{X}{Style.RESET_ALL}\n\n"
        )
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
    X = pd.DataFrame(imp1.fit_transform(X.astype("float32")), columns=X.columns)

    # Deconfound X by any non-connectome columns present, and then remove them
    if len(nuisance_cols) > 0:
        deconfounder = DeConfounder()
        net_cols = [i for i in X.columns if i not in nuisance_cols]
        z_cols = [i for i in X.columns if i in nuisance_cols]
        if len(z_cols) > 0:
            deconfounder.fit(X[net_cols].values, X[z_cols].values)
            X[net_cols] = pd.DataFrame(
                deconfounder.transform(X[net_cols].values, X[z_cols].values),
                columns=X[net_cols].columns,
            )
            print(f"Deconfounding with {z_cols}...")
            X.drop(columns=z_cols, inplace=True)

    # Standardize X
    if standardize is True:
        if standard_method == "ss":
            scaler = StandardScaler()
        elif standard_method == "mm":
            scaler = MinMaxScaler()
        else:
            raise ValueError("standard_method must be either `ss` or `mm`")
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Remove low-variance columns
    sel = VarianceThreshold(threshold=var_thr)
    sel.fit(X)
    if len(nodrop_columns) > 0:
        good_var_cols = X.columns[
            np.concatenate(
                [
                    sel.get_support(indices=True),
                    np.array(
                        [X.columns.get_loc(c) for c in nodrop_columns if c in X]
                    ),
                ]
            )
        ]
    else:
        good_var_cols = X.columns[sel.get_support(indices=True)]

    low_var_cols = [i for i in X.columns if i not in list(good_var_cols)]
    if len(low_var_cols) > 0:
        print(f"Dropping {low_var_cols} for low variance...")
    X = X[good_var_cols]

    if X.empty:
        print(
            f"\n\n{Fore.RED}Empty feature-space (low-variance): "
            f"{X}{Style.RESET_ALL}\n\n"
        )
        return X, y

    # Remove outliers
    if remove_outliers is True:
        X, y = de_outlier(X, y, std_dev, "IF")
        if X.empty and len(y) < 50:
            print(
                f"\n\n{Fore.RED}Empty feature-space (outliers): "
                f"{X}{Style.RESET_ALL}\n\n"
            )
            return X, y

    # Remove missing y
    # imp2 = SimpleImputer()
    # y = imp2.fit_transform(np.array(y).reshape(-1, 1))

    # y_missing_mask = np.invert(np.isnan(y))
    # X = X[y_missing_mask]
    # y = y[y_missing_mask]
    if X.empty or len(y) < 50:
        print(
            f"\n\n{Fore.RED}Empty feature-space (missing y): "
            f"{X}, {y}{Style.RESET_ALL}\n\n"
        )
        return X, y

    # Remove multicollinear columns
    if remove_multi is True:
        try:
            rvif = ReduceVIF(thresh=vif_thr)
            X = rvif.fit_transform(X)[0]
            if X.empty or len(X.columns) < 5:
                print(
                    f"\n\n{Fore.RED}Empty feature-space "
                    f"(multicollinearity): "
                    f"{X}{Style.RESET_ALL}\n\n"
                )
                return X, y
        except BaseException:
            print(
                f"\n\n{Fore.RED}Empty feature-space (multicollinearity): "
                f"{X}{Style.RESET_ALL}\n\n"
            )
            return X, y

    if oversample is True:
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=42, sampling_strategy="auto")
            X, y = sm.fit_resample(X, y)
        except BaseException:
            pass

    print(f"\nX: {X}\ny: {y}\n")
    print(f"Features: {list(X.columns)}\n")
    return X, y


def make_param_grids() -> dict:
    """
    Generate a list of parameter grids for GridSearchCV or RandomizedSearchCV.
    """
    param_space = {}
    param_space["Cs"] = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
    param_space["l1_ratios"] = [0, 0.25, 0.50, 0.75, 1]
    param_space["alphas"] = [1e-8, 1e-4, 1e-2, 1e-1, 0.25, 0.5, 1]
    return param_space


def get_ensembles_embedding(
    modality: str, alg: str, base_dir: str
) -> typing.Optional[list]:
    if alg == "OMNI" or alg == "ASE":
        ensembles_pre = list(
            set(
                [
                    "subnet-"
                    + i.split("subnet-")[1].split("_")[0]
                    + "_granularity-"
                    + i.split("granularity-")[1].split("/")[0]
                    + "_"
                    + os.path.basename(i)
                    .split(modality + "_")[1]
                    .replace(".npy", "")
                    for i in glob.glob(
                        f"{base_dir}/pynets/sub-*/ses-*/{modality}/subnet-*/"
                        f"embeddings/gradient-{alg}*.npy"
                    )
                ]
            )
        )
        ensembles = []
        for i in ensembles_pre:
            if "_thrtype" in i:
                ensembles.append(i.split("_thrtype")[0])
            else:
                ensembles.append(i)
    elif (
        alg == "eigenvector"
        or alg == "betweenness"
        or alg == "degree"
        or alg == "local_efficiency"
        or alg == "local_clustering"
    ):
        ensembles_pre = list(
            set(
                [
                    "subnet-"
                    + i.split("subnet-")[1].split("_")[0]
                    + "_granularity-"
                    + i.split("granularity-")[1].split("/")[0]
                    + "_"
                    + os.path.basename(i)
                    .split(modality + "_")[1]
                    .replace(".csv", "")
                    for i in glob.glob(
                        f"{base_dir}/pynets/sub-*/ses-*/{modality}/subnet-*/"
                        f"embeddings/gradient-{alg}*.csv"
                    )
                ]
            )
        )
        ensembles = []
        for i in ensembles_pre:
            if "_thrtype" in i:
                ensembles.append(i.split("_thrtype")[0])
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
            df_top.drop(df_top.filter(regex="Unnamed: 0"), axis=1, inplace=True)
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
    import gc
    import tempfile

    from joblib import Parallel, delayed

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
                grid_params_mod.append(
                    (
                        signal,
                        hpass,
                        model,
                        granularity,
                        parcellation,
                        str(smooth),
                    )
                )
            except BaseException:
                try:
                    signal, hpass, model, granularity, parcellation = comb
                    smooth = "0"
                    grid_params_mod.append(
                        (
                            signal,
                            hpass,
                            model,
                            granularity,
                            parcellation,
                            str(smooth),
                        )
                    )
                except BaseException:
                    print(f"Failed to parse: {comb}")

    elif target_modality == "dwi":
        for comb in grid_params:
            try:
                (
                    traversal,
                    minlength,
                    model,
                    granularity,
                    parcellation,
                    error_margin,
                ) = comb
                grid_params_mod.append(
                    (
                        traversal,
                        minlength,
                        model,
                        granularity,
                        parcellation,
                        error_margin,
                    )
                )
            except BaseException:
                print(f"Failed to parse: {comb}")

    with Parallel(
        n_jobs=-1, backend="loky", verbose=10, temp_folder=cache_dir
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
                mets,
            )
            for grid_param in grid_params_mod
        )
    for fs, grid_param in outs:
        ml_dfs[target_modality][target_embedding_type][grid_param] = fs
        del fs, grid_param
    gc.collect()
    return ml_dfs


def build_grid(
    modality: str, hyperparam_dict: dict, hyperparams: list, ensembles: list
):
    for ensemble in ensembles:
        try:
            build_mp_dict(ensemble, modality, hyperparam_dict, hyperparams)
        except BaseException:
            print(f"Failed to parse ensemble {ensemble}...")

    if "subnet" in hyperparam_dict.keys():
        hyperparam_dict["subnet"] = [
            i for i in hyperparam_dict["subnet"] if "granularity" not in i
        ]

    hyperparam_dict = OrderedDict(
        sorted(hyperparam_dict.items(), key=lambda x: x[0])
    )
    grid = list(
        itertools.product(
            *(hyperparam_dict[param] for param in hyperparam_dict.keys())
        )
    )

    return hyperparam_dict, grid


def save_netmets(dir_path, est_path, metric_list_names, net_met_val_list_final):
    import os
    from pynets.core import utils

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


def flatten_latent_positions(
    base_dir, subject_dict, ID, ses, modality, grid_param, alg
):
    from pynets.core.nodemaker import parse_closest_ixs

    if grid_param in subject_dict[ID][str(ses)][modality][alg].keys():
        rsn_dict = subject_dict[ID][str(ses)][modality][alg][grid_param]

        if "data" in rsn_dict.keys():
            ixs = [i for i in rsn_dict["index"] if i is not None]

            if not isinstance(rsn_dict["data"], np.ndarray):
                if rsn_dict["data"].endswith(".npy"):
                    rsn_dict["data"] = np.load(
                        rsn_dict["data"], allow_pickle=True
                    )
                elif rsn_dict["data"].endswith(".csv"):
                    rsn_dict["data"] = np.array(
                        pd.read_csv(rsn_dict["data"])
                    ).reshape(-1, 1)

            emb_shape = rsn_dict["data"].shape[0]

            if len(ixs) != emb_shape:
                node_files = glob.glob(
                    f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/subnet-"
                    f"{grid_param[-2]}_granularity-{grid_param[-3]}/nodes/"
                    f"*.json"
                )
                ixs, node_dict = parse_closest_ixs(node_files, emb_shape)

            if len(ixs) > 0:
                if len(ixs) == emb_shape:
                    rsn_arr = rsn_dict["data"].T.reshape(
                        1,
                        rsn_dict["data"].T.shape[0]
                        * rsn_dict["data"].T.shape[1],
                    )
                    if rsn_dict["data"].shape[1] == 1:
                        df_lps = pd.DataFrame(
                            rsn_arr,
                            columns=[
                                f"{i}_subnet-"
                                f"{grid_param[-2]}_"
                                f"granularity-"
                                f"{grid_param[-3]}_"
                                f"dim1"
                                for i in ixs
                            ],
                        )
                    elif rsn_dict["data"].shape[1] == 3:
                        df_lps = pd.DataFrame(
                            rsn_arr,
                            columns=[
                                f"{i}_subnet-{grid_param[-2]}_"
                                f"granularity-{grid_param[-3]}_dim1"
                                for i in ixs
                            ]
                            + [
                                f"{i}_subnet-{grid_param[-2]}_"
                                f"granularity-{grid_param[-3]}_dim2"
                                for i in ixs
                            ]
                            + [
                                f"{i}_subnet-{grid_param[-2]}_"
                                f"granularity-{grid_param[-3]}_dim3"
                                for i in ixs
                            ],
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
                print(
                    UserWarning(
                        f"Missing indices for " f"{grid_param} universe..."
                    )
                )
                df_lps = None
        else:
            print(UserWarning(f"Missing {grid_param} universe..."))
            df_lps = None
    else:
        print(UserWarning(f"Missing {grid_param} universe..."))
        df_lps = None

    return df_lps


def create_feature_space(
    base_dir, df, grid_param, subject_dict, ses, modality, alg, mets=None
):
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
            print(
                f"Modality: {modality} not found for ID {ID}, " f"ses-{ses}..."
            )
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
            print(
                f"{Fore.GREEN}✓{Style.RESET_ALL} Grid Param: {grid_param} "
                f"found for {ID}"
            )
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
            print(
                f"Feature-space null for ID {ID} & ses-{ses}, modality: "
                f"{modality}, embedding: {alg}..."
            )
            continue

    if len(df_tmps) > 0:
        dfs = [
            dff.set_index("participant_id") for dff in df_tmps if not dff.empty
        ]
        df_all = pd.concat(dfs, axis=0)
        # df_all = df_all.replace({0: np.nan})
        # df_all = df_all.apply(lambda x: np.where(x < 0.00001, np.nan, x))
        # print(len(df_all))
        del df_tmps
        return df_all, grid_param
    else:
        return pd.Series(np.nan), grid_param


def graph_theory_prep(
    df: pd.DataFrame, thr_type: str, drop_thr: float = 0.50
) -> typing.Tuple[pd.DataFrame, list]:

    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler

    cols = [
        j
        for j in set(
            [
                i.split("_thrtype-" + thr_type + "_")[0]
                for i in list(set(df.columns))
            ]
        )
        if j != "id"
    ]

    id_col = df["id"]

    df = df.dropna(thresh=len(df) * drop_thr, axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    imp = KNNImputer(n_neighbors=7)
    df = pd.DataFrame(
        imp.fit_transform(
            scaler.fit_transform(df[[i for i in df.columns if i != "id"]])
        ),
        columns=[i for i in df.columns if i != "id"],
    )

    df = pd.concat([id_col, df], axis=1)

    return df, cols


def split_df_to_dfs_by_prefix(df: pd.DataFrame, prefixes: list = []) -> list:
    """
    Split a dataframe into a list of dataframes based on the prefixes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to split.
    prefixes : list
        List of prefixes to split the dataframe by.

    Returns
    -------
    dfs : list
        List of dataframes.
    """
    from pynets.core.utils import flatten

    df_splits = []
    for p in prefixes:
        df_splits.append(
            df[
                list(
                    set(
                        list(
                            flatten([c for c in df.columns if c.startswith(p)])
                        )
                    )
                )
            ]
        )
    # pref_selected = list(set(list(flatten([i.columns for i in df_splits]))))
    # df_other = df[[j for j in df.columns if j not in pref_selected]]
    # return df_splits + [df_other]

    return df_splits


def de_outlier(
    X: np.ndarray, y: np.ndarray, sd: int, deoutlier_type: str = "IF"
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Remove any gross outlier row in X whose linear residual
    when regressing y against X is > sd standard deviations
    away from the mean residual. For classifiers, use a NaiveBayes estimator
    since it does not require tuning, and for regressors, use simple
    linear regression.

    """
    if deoutlier_type == "IF":
        model = IsolationForest(
            random_state=42, bootstrap=True, contamination="auto"
        )
        outlier_mask = model.fit_predict(X)
        outlier_mask[outlier_mask == -1] = 0
        outlier_mask = outlier_mask.astype("bool")
    elif deoutlier_type == "LOF":
        mask = np.zeros(X.shape, dtype=np.bool)
        model = LocalOutlierFactor(n_neighbors=10, metric="mahalanobis")
        model.fit_predict(X)
        X_scores = model.negative_outlier_factor_
        X_scores = (X_scores.max() - X_scores) / (
            X_scores.max() - X_scores.min()
        )
        median_score = np.median(X_scores)
        outlier_mask = np.logical_or(
            [X_scores[i] > median_score for i in range(len(X.shape[0]))], mask
        )
        outlier_mask[outlier_mask == -1] = 0
        outlier_mask = outlier_mask.astype("bool")
    else:
        if deoutlier_type == "NB":
            model = GaussianNB()
        elif deoutlier_type == "LR":
            model = LinearRegression(normalize=True)
        else:
            raise ValueError("predict_type not recognized!")
        model.fit(X, y)
        predicted_y = model.predict(X)

        resids = (y - predicted_y) ** 2

        outlier_mask = (
            np.abs(stats.zscore(np.array(resids).reshape(-1, 1))) < float(sd)
        ).all(axis=1)

    return X[outlier_mask], y[outlier_mask]


def get_scorer_ens(scorer_name: str) -> typing.Callable:
    import importlib
    import sklearn.metrics

    found = False

    try:
        scoring = sklearn.metrics.get_scorer(scorer_name)
        found = True
    except ValueError:
        pass

    if not found:
        i = scorer_name.rfind(".")
        if i < 0:
            raise ValueError(
                "Invalid scorer import path: {}".format(scorer_name)
            )
        module_name, scorer_name_ = scorer_name[:i], scorer_name[i + 1 :]
        mod = importlib.import_module(module_name)
        scoring = getattr(mod, scorer_name_)
        found = True

    return scoring


def _draw_bootstrap_sample(
    rng: np.random.RandomState, X: pd.DataFrame, y: pd.DataFrame
):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return (
        X.iloc[bootstrap_indices.tolist(), :],
        y.iloc[bootstrap_indices.tolist(), :],
    )


def bias_variance_decomp(
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    loss: str = "0-1_loss",
    num_rounds: int = 200,
    random_seed: typing.Union[int, None] = None,
    **fit_params,
):
    """
    Nonparametric Permutation Test for Bias-Variance Decomposition

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

    """
    supported = ["0-1_loss", "mse"]
    if loss not in supported:
        raise NotImplementedError(
            "loss must be one of the following: %s" % supported
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y
    )

    rng = np.random.RandomState(random_seed)

    if loss == "0-1_loss":
        dtype = np.int
    elif loss == "mse":
        dtype = np.float

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=dtype)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(
            rng, pd.DataFrame(X_train), pd.DataFrame(y_train)
        )

        # Keras support
        pred = estimator.fit(X_boot, y_boot, **fit_params).predict(X_test)
        all_pred[i] = pred

    if loss == "0-1_loss":
        main_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred
        )

        avg_expected_loss = np.apply_along_axis(
            lambda x: (x != y_test.values).mean(), axis=1, arr=all_pred
        ).mean()

        avg_bias = (
            np.sum(main_predictions != y_test.values) / y_test.values.size
        )

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int)
        var /= num_rounds

        avg_var = var.sum() / y_test.shape[0]

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x: ((x - y_test.values) ** 2).mean(), axis=1, arr=all_pred
        ).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = (
            np.sum((main_predictions - y_test.values) ** 2) / y_test.values.size
        )
        avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var


def make_x_y(
    input_dict: dict,
    drop_cols: list,
    target_var: str,
    embedding_type: str,
    grid_param: str,
) -> typing.Tuple[typing.Optional[pd.DataFrame], typing.Optional[pd.DataFrame]]:
    import pandas as pd
    import json

    print(target_var)
    print(embedding_type)
    print(grid_param)

    if input_dict is None:
        return None, None

    if not os.path.isfile(input_dict):
        return None, None

    with open(input_dict) as data_file:
        data_loaded = json.load(data_file)
    data_file.close()

    if data_loaded == "{}":
        return None, None

    if str(grid_param) in data_loaded.keys():
        df_all = pd.read_json(data_loaded[str(grid_param)])
        # if df_all[target_var].isin([np.nan, 1]).all():
        #     df_all[target_var] = df_all[target_var].replace({np.nan: 0})
        if df_all is None:
            return None, None
        else:
            df_all = df_all.loc[:, ~df_all.columns.duplicated()]
            df_all.reset_index(level=0, inplace=True)
            df_all.rename(columns={"index": "id"}, inplace=True)
            if (
                all(
                    df_all.drop(
                        columns=[
                            "id",
                            "participant_id",
                        ]
                    )
                    .isnull()
                    .all()
                )
                or len(df_all.columns) == 1
                or (
                    np.abs(
                        np.array(
                            df_all.drop(
                                columns=[
                                    "id",
                                    "participant_id",
                                ]
                            )
                        )
                    )
                    < 0.00001
                ).all()
            ):
                return None, None
            else:
                df_all.drop(columns=["id", "participant_id"], inplace=True)
                if len(df_all.columns) < 5:
                    print(f"Too few columns detected for {grid_param}...")
                    return None, None
    else:
        return None, None

    if len(df_all) < 50:
        print(
            "\nToo few cases in feature-space after preprocessing, "
            "skipping...\n"
        )
        return None, None
    elif len(df_all) > 50:
        drop_cols = [
            i
            for i in drop_cols
            if (i in df_all.columns)
            or (i.replace("Behavioral_", "") in df_all.columns)
            or (f"Behavioral_{i}" in df_all.columns)
        ]

        return df_all.drop(columns=drop_cols), df_all[target_var].values
    else:
        print("\nEmpty/Missing Feature-space...\n")
        return None, None


def concatenate_frames(
    out_dir,
    modality,
    embedding_type,
    target_var,
    files_,
    n_boots,
    dummy_run,
    search_method,
    stack,
    stack_prefix_list,
):
    import pandas as pd
    import os

    if len(files_) > 1:
        dfs = []
        parcellations = []
        for file_ in files_:
            df = pd.read_csv(file_, chunksize=100000).read()
            try:
                df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
            except BaseException:
                pass
            dfs.append(df)
            parcellations.append(
                file_.split("_grid_param_")[1].split("/")[0].split(".")[-2]
            )
        try:
            frame = pd.concat(
                dfs, axis=0, join="outer", sort=True, ignore_index=False
            )

            out_path = (
                f"{out_dir}/final_predictions_modality-{modality}_"
                f"subnet-{str(list(set(parcellations)))}_"
                f"gradient-{embedding_type}_outcome-{target_var}_"
                f"boots-{n_boots}_search-{search_method}"
            )

            if dummy_run is True:
                out_path = out_path + "_dummy"

            if stack is True:
                out_path = out_path + "_stacked-" + str(stack_prefix_list)

            out_path = out_path.replace("['", "").replace("']", "") + ".csv"

            print(f"Saving to {out_path}...")
            if os.path.isfile(out_path):
                os.remove(out_path)
            frame.to_csv(out_path, index=False)
        except ValueError:
            print(
                f"Dataframe concatenation failed for {modality}, "
                f"{embedding_type}, {target_var}..."
            )

        return out_path, embedding_type, target_var, modality
    else:
        return None, embedding_type, target_var, modality


def make_subject_dict(
    modalities,
    base_dir,
    thr_type,
    mets,
    embedding_types,
    template,
    sessions,
    rsns,
    IDS=None,
):
    import gc
    import shutil
    import tempfile

    import psutil
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    from pynets.core.utils import load_runconfig, mergedicts

    hardcoded_params = load_runconfig()
    embedding_methods = hardcoded_params["embed"]
    # hyperparams_func = hardcoded_params["hyperparams_func"]
    # hyperparams_dwi = hardcoded_params["hyperparams_dwi"]

    miss_frames_all = []
    subject_dict_all = {}
    modality_grids = {}
    for modality in modalities:
        print(f"MODALITY: {modality}")
        hyperparams = eval(f"hyperparams_{modality}")
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
                    ensembles = get_ensembles_embedding(modality, alg, base_dir)
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
                    modality,
                    hyperparam_dict,
                    sorted(list(set(hyperparams))),
                    ensembles,
                )[1]

                grid = list(
                    set([i for i in grid if i != () and len(list(i)) > 0])
                )

                grid = [i for i in grid if any(n in i for n in rsns)]

                modality_grids[modality] = grid

                par_dict = subject_dict_all.copy()
                cache_dir = tempfile.mkdtemp()

                max_bytes = int(
                    float(list(psutil.virtual_memory())[4] / len(ids))
                )
                with Parallel(
                    n_jobs=-1,
                    backend="loky",
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
                            df_top,
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
                del (
                    par_dict,
                    outs_tup,
                    outs,
                    df_top,
                    miss_frames,
                    ses_name,
                    grid,
                    hyperparam_dict,
                    parallel,
                )
                gc.collect()
            del alg
        del hyperparams
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
    import gc

    # from colorama import Fore, Style
    # print(id)
    ID = id.split("_")[0].split("sub-")[1]
    ses = id.split("_")[1].split("ses-")[1]

    # completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"

    if ID not in subject_dict.keys():
        subject_dict[ID] = {}

    if ses not in subject_dict[ID].keys():
        subject_dict[ID][str(ses)] = {}

    if modality not in subject_dict[ID][str(ses)].keys():
        subject_dict[ID][str(ses)][modality] = {}

    if alg not in subject_dict[ID][str(ses)][modality].keys():
        subject_dict[ID][str(ses)][modality][alg] = {}

    subject_dict[ID][str(ses)][modality][alg] = dict.fromkeys(grid, np.nan)

    missingness_frame = pd.DataFrame(
        columns=["id", "ses", "modality", "alg", "grid"]
    )

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
            [subject_dict, missingness_frame] = func_grabber(
                comb,
                subject_dict,
                missingness_frame,
                ID,
                ses,
                modality,
                alg,
                mets,
                thr_type,
                base_dir,
                template,
                df_top,
                embedding_methods,
            )
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
            [subject_dict, missingness_frame] = dwi_grabber(
                comb,
                subject_dict,
                missingness_frame,
                ID,
                ses,
                modality,
                alg,
                mets,
                thr_type,
                base_dir,
                template,
                df_top,
                embedding_methods,
            )
            gc.collect()
    del modality, ID, ses, df_top
    gc.collect()
    return subject_dict, missingness_frame


def dwi_grabber(
    comb,
    subject_dict,
    missingness_frame,
    ID,
    ses,
    modality,
    alg,
    mets,
    thr_type,
    base_dir,
    template,
    df_top,
    embedding_methods,
):
    import gc
    from colorama import Fore, Style
    from pynets.core.utils import filter_cols_from_targets
    from pynets.core.nodemaker import get_index_labels

    try:
        (
            traversal,
            minlength,
            model,
            granularity,
            parcellation,
            error_margin,
        ) = comb
    except BaseException:
        print(
            UserWarning(
                f"{Fore.YELLOW}Failed to parse: " f"{comb}{Style.RESET_ALL}"
            )
        )
        return subject_dict, missingness_frame

    # comb_tuple = (parcellation, traversal, minlength, model, granularity, error_margin)
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

        if template == "any":
            embeddings = [
                i
                for i in embeddings
                if (alg in i)
                and (f"granularity-{granularity}" in i)
                and (f"subnet-{parcellation}" in i)
                and (f"model-{model}" in i)
                and (f"traversal-{traversal}" in i)
                and (f"minlength-{minlength}" in i)
                and (f"tol-{error_margin}" in i)
                and ("_NULL" not in i)
            ]
        else:
            embeddings = [
                i
                for i in embeddings
                if (alg in i)
                and (f"granularity-{granularity}" in i)
                and (f"subnet-{parcellation}" in i)
                and (f"template-{template}" in i)
                and (f"model-{model}" in i)
                and (f"traversal-{traversal}" in i)
                and (f"minlength-{minlength}" in i)
                and (f"tol-{error_margin}" in i)
                and ("_NULL" not in i)
            ]

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
            embeddings_raw = [
                i
                for i in embeddings
                if "thrtype" not in i or "thr-1.0" in i or "thr-" not in i
            ]
            if len(embeddings_raw) == 1:
                embedding = embeddings_raw[0]
            else:
                embeddings_raw = [
                    i
                    for i in embeddings_raw
                    if (
                        f"/subnet-{parcellation}_"
                        f"granularity-{granularity}/" in i
                    )
                    and (parcellation in os.path.basename(i))
                    and (granularity in os.path.basename(i))
                ]
                if len(embeddings_raw) > 0:
                    sorted_embeddings = sorted(
                        embeddings_raw,
                        key=lambda x: int(
                            x.partition("samples-")[2].partition("streams")[0]
                        ),
                        reverse=False,
                    )
                    # TODO: Change "reverse" above to True to grab the MOST
                    #  number of samples (ideal).

                    sorted_embeddings = sorted(
                        sorted_embeddings, key=os.path.getmtime
                    )
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
                if embedding.endswith(".npy"):
                    emb_shape = np.load(
                        embedding, allow_pickle=True, mmap_mode=None
                    ).shape[0]
                elif embedding.endswith(".csv"):
                    with open(embedding, "r+") as a:
                        emb_shape = len(pd.read_csv(a).columns)
                    a.close()
                else:
                    raise NotImplementedError(
                        f"Format of {embedding} "
                        f"not recognized! "
                        f"Only .npy and .csv "
                        f"currently supported."
                    )
                gc.collect()
            except BaseException:
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
                ixs = get_index_labels(
                    base_dir,
                    ID,
                    ses,
                    modality,
                    parcellation,
                    granularity,
                    emb_shape,
                )
            except BaseException:
                print(
                    f"{Fore.LIGHTYELLOW_EX}Failed to load indices for "
                    f"{embedding}{Style.RESET_ALL}"
                )
                return subject_dict, missingness_frame

            if not isinstance(
                subject_dict[ID][str(ses)][modality][alg][comb_tuple], dict
            ):
                subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
            subject_dict[ID][str(ses)][modality][alg][comb_tuple]["index"] = ixs
            # subject_dict[ID][str(ses)][modality][alg][comb_tuple]["labels"]
            # = labels
            subject_dict[ID][str(ses)][modality][alg][comb_tuple][
                "data"
            ] = embedding
            # print(data)
            completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}"
            )
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
                print(
                    f"{Fore.YELLOW}Missing metric {met} for ID: {ID}, "
                    f"SESSION: {ses}{Style.RESET_ALL}"
                )
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
                f"UNIVERSE: {comb_tuple}, COMPLETENESS: {completion_status}"
            )
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
            completion_status = f"{Fore.YELLOW}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}"
            )
        subject_dict[ID][str(ses)][modality][alg][comb_tuple] = data
        # save_embed_data_to_sql(data, ixs, ID, str(ses), modality, alg,
        # comb_tuple)
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame


def func_grabber(
    comb,
    subject_dict,
    missingness_frame,
    ID,
    ses,
    modality,
    alg,
    mets,
    thr_type,
    base_dir,
    template,
    df_top,
    embedding_methods,
):
    import gc
    from colorama import Fore, Style
    from pynets.core.nodemaker import get_index_labels
    from pynets.core.utils import filter_cols_from_targets

    try:
        signal, hpass, model, granularity, parcellation, smooth = comb
    except BaseException:
        try:
            signal, hpass, model, granularity, parcellation = comb
            smooth = "0"
        except BaseException:
            print(
                UserWarning(
                    f"{Fore.YELLOW}Failed to parse: " f"{comb}{Style.RESET_ALL}"
                )
            )
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

        if template == "any":
            embeddings = [
                i
                for i in embeddings
                if (
                    (alg in i)
                    and (f"granularity-" f"{granularity}" in i)
                    and (f"subnet-{parcellation}" in i)
                    and (f"model-{model}" in i)
                    and (f"hpass-{hpass}Hz" in i)
                    and (f"signal-{signal}" in i)
                )
                and ("_NULL" not in i)
            ]
        else:
            embeddings = [
                i
                for i in embeddings
                if (
                    (alg in i)
                    and (f"granularity-" f"{granularity}" in i)
                    and (f"subnet-{parcellation}" in i)
                    and (f"template-{template}" in i)
                    and (f"model-{model}" in i)
                    and (f"hpass-{hpass}Hz" in i)
                    and (f"signal-{signal}" in i)
                )
                and ("_NULL" not in i)
            ]

        if smooth == "0":
            embeddings = [i for i in embeddings if "smooth" not in i]
        else:
            embeddings = [i for i in embeddings if f"tol-{smooth}fwhm" in i]

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
            embeddings_raw = [
                i
                for i in embeddings
                if "thrtype" not in i or "thr-1.0" in i or "thr-" not in i
            ]
            if len(embeddings_raw) == 1:
                embedding = embeddings_raw[0]
            else:
                embeddings_raw = [
                    i
                    for i in embeddings_raw
                    if f"/subnet-{parcellation}_"
                    f"granularity-{granularity}/" in i
                    and (parcellation in os.path.basename(i))
                    and (granularity in os.path.basename(i))
                ]
                if len(embeddings_raw) > 0:
                    sorted_embeddings = sorted(
                        embeddings_raw, key=os.path.getmtime
                    )
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
                if embedding.endswith(".npy"):
                    emb_shape = np.load(
                        embedding, allow_pickle=True, mmap_mode=None
                    ).shape[0]
                elif embedding.endswith(".csv"):
                    with open(embedding, "r+") as a:
                        emb_shape = len(pd.read_csv(a).columns)
                    a.close()
                else:
                    raise NotImplementedError(
                        f"Format of {embedding} "
                        f"not recognized! "
                        f"Only .npy and .csv "
                        f"currently supported."
                    )
                gc.collect()
            except BaseException:
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
                ixs = get_index_labels(
                    base_dir,
                    ID,
                    ses,
                    modality,
                    parcellation,
                    granularity,
                    emb_shape,
                )
            except BaseException:
                print(
                    f"{Fore.LIGHTYELLOW_EX}Failed to load indices for "
                    f"{embedding} {Style.RESET_ALL}"
                )
                return subject_dict, missingness_frame
            if not isinstance(
                subject_dict[ID][str(ses)][modality][alg][comb_tuple], dict
            ):
                subject_dict[ID][str(ses)][modality][alg][comb_tuple] = {}
            subject_dict[ID][str(ses)][modality][alg][comb_tuple]["index"] = ixs
            # subject_dict[ID][str(ses)][modality][alg][comb_tuple]["labels"]
            # = labels
            subject_dict[ID][str(ses)][modality][alg][comb_tuple][
                "data"
            ] = embedding
            # print(data)
            completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}"
            )
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
        if smooth == "0":
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
                if comb_tuple[-1] == "0":
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
                print(
                    f"{Fore.YELLOW}Missing metric {met} for ID: {ID}, "
                    f"SESSION: {ses}{Style.RESET_ALL}"
                )
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
                f"COMPLETENESS: {completion_status}"
            )
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
            completion_status = f"{Fore.YELLOW}X{Style.RESET_ALL}"
            print(
                f"ID: {ID}, SESSION: {ses}, EMBEDDING: {alg}, "
                f"UNIVERSE: {comb_tuple}, "
                f"COMPLETENESS: {completion_status}"
            )
        subject_dict[ID][str(ses)][modality][alg][comb_tuple] = data
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame


def cleanNullTerms(d: dict):
    """
    Remove null terms from the dictionary.
    """
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = cleanNullTerms(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif v is not None and v is not np.nan and not isinstance(v, pd.Series):
            clean[k] = v
    return clean


def gen_sub_vec(base_dir, sub_dict_clean, ID, modality, alg, comb_tuple):
    """
    Generate a vector for a given subject, modality, algorithm, and universe.
    """
    vects = []
    for ses in sub_dict_clean[ID].keys():
        # print(ses)
        if comb_tuple in sub_dict_clean[ID][str(ses)][modality][alg].keys():
            if alg == "topology":
                vect = sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple]
            else:
                vect = flatten_latent_positions(
                    base_dir,
                    sub_dict_clean,
                    ID,
                    ses,
                    modality,
                    comb_tuple,
                    alg,
                )
            vects.append(vect)
    vects = [
        i for i in vects if i is not None and not np.isnan(np.array(i)).all()
    ]
    if len(vects) > 0 and alg == "topology":
        out = np.concatenate(vects, axis=1)
    elif len(vects) > 0:
        out = pd.concat(vects, axis=0)
        del vects
    else:
        out = None
    # print(out)
    return out


def tuple_insert(tup: tuple, pos: int, ele: typing.Any) -> tuple:
    """
    Extend tuple with additional element.
    """
    tup = tup[:pos] + (ele,) + tup[pos:]
    return tup


def build_mp_dict(
    file_renamed: str, modality: str, hyperparam_dict: dict, hyperparams: list
):
    """
    A function to build a metaparameter dictionary by parsing a given
    file path.
    """

    for hyperparam in hyperparams:
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
                hyperparam_dict["smooth"] = [
                    str(
                        file_renamed.split("tol-")[1]
                        .split("_")[0]
                        .split("fwhm")[0]
                    )
                ]
            else:
                hyperparam_dict["smooth"].append(
                    str(
                        file_renamed.split("tol-")[1]
                        .split("_")[0]
                        .split("fwhm")[0]
                    )
                )
        else:
            if "smooth" not in hyperparam_dict.keys():
                hyperparam_dict["smooth"] = [str(0)]
            hyperparam_dict["smooth"].append(str(0))
            hyperparams.append("smooth")
        if "hpass-" in file_renamed:
            if "hpass" not in hyperparam_dict.keys():
                hyperparam_dict["hpass"] = [
                    str(
                        file_renamed.split("hpass-")[1]
                        .split("_")[0]
                        .split("Hz")[0]
                    )
                ]
            else:
                hyperparam_dict["hpass"].append(
                    str(
                        file_renamed.split("hpass-")[1]
                        .split("_")[0]
                        .split("Hz")[0]
                    )
                )
            hyperparams.append("hpass")
        if "signal-" in file_renamed:
            if "signal" not in hyperparam_dict.keys():
                hyperparam_dict["signal"] = [
                    str(file_renamed.split("signal-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["signal"].append(
                    str(file_renamed.split("signal-")[1].split("_")[0])
                )
            hyperparams.append("signal")

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
            hyperparams.append("traversal")
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
            if "error_margin" not in hyperparam_dict.keys():
                hyperparam_dict["error_margin"] = [
                    str(file_renamed.split("tol-")[1].split("_")[0])
                ]
            else:
                hyperparam_dict["error_margin"].append(
                    str(file_renamed.split("tol-")[1].split("_")[0])
                )
            hyperparams.append("error_margin")

    for key in hyperparam_dict:
        hyperparam_dict[key] = list(set(hyperparam_dict[key]))

    return hyperparam_dict, hyperparams
