import matplotlib
import os
import typing
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    StratifiedKFold,
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition
from collections import OrderedDict
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from pynets.core.utils import flatten
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from pynets.statistics.utils import (
    bias_variance_decomp,
    split_df_to_dfs_by_prefix,
    make_param_grids,
    preprocess_x_y,
)

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
warnings.simplefilter("ignore")

import_list = [
    "import pandas as pd",
    "import os",
    "import re",
    "import glob",
    "import numpy as np",
    "from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, "
    "GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate",
    "from sklearn.dummy import DummyClassifier, DummyRegressor",
    "from sklearn.feature_selection import VarianceThreshold, SelectKBest, "
    "f_regression, f_classif",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler",
    "from sklearn import linear_model, decomposition",
    "from pynets.statistics.group.benchmarking import build_hp_dict",
    "import seaborn as sns",
    "import matplotlib",
    "matplotlib.use('Agg')",
    "import matplotlib.pyplot as plt",
    "from pynets.statistics.individual.embeddings import "
    "build_asetomes, _omni_embed",
    "from joblib import Parallel, delayed",
    "from pynets.core import utils",
    "from itertools import groupby",
    "import shutil",
    "from pathlib import Path",
    "from collections import OrderedDict",
    "from operator import itemgetter",
    "from statsmodels.stats.outliers_influence import "
    "variance_inflation_factor",
    "from sklearn.impute import SimpleImputer",
    "from pynets.core.utils import flatten",
    "import pickle",
    "import dill",
    "from sklearn.model_selection._split import _BaseKFold",
    "from sklearn.utils import check_random_state",
]


@ignore_warnings(category=ConvergenceWarning)
def nested_fit(
    X: pd.DataFrame,
    y: pd.DataFrame,
    estimators: list,
    boot: int,
    pca_reduce: bool,
    k_folds: int,
    predict_type: str,
    search_method: str = "grid",
    razor: bool = False,
    n_jobs: int = 1,
) -> typing.Tuple[Pipeline, str]:

    # Instantiate an inner-fold
    if predict_type == "regressor":
        inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=boot)
    elif predict_type == "classifier":
        inner_cv = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=boot
        )
    else:
        raise ValueError("Prediction method not recognized")

    if predict_type == "regressor":
        scoring = ["explained_variance", "neg_mean_squared_error"]
        refit_score = "explained_variance"
        feature_selector = f_regression
    elif predict_type == "classifier":
        scoring = ["f1", "neg_mean_squared_error"]
        refit_score = "f1"
        feature_selector = f_classif

    # k Features
    k = [10]

    # Instantiate a working dictionary of performance within a bootstrap
    means_all_exp_var = {}
    means_all_MSE = {}

    # Model + feature selection by iterating grid-search across linear
    # estimators
    for estimator_name, estimator in sorted(estimators.items()):
        if pca_reduce is True and X.shape[0] < X.shape[1]:
            param_grid = {
                "feature_select__n_components": k,
            }

            # Pipeline feature selection (PCA) with model fitting
            pipe = Pipeline(
                [
                    (
                        "feature_select",
                        decomposition.PCA(random_state=boot, whiten=True),
                    ),
                    (estimator_name, estimator),
                ]
            )
            if razor is True:
                from pynets.statistics.interfaces import Razors

                refit = Razors.simplify(
                    param="n_components",
                    scoring=refit_score,
                    rule="se",
                    sigma=1,
                )
            else:
                refit = refit_score
        else:
            # <k Features, don't perform feature selection, but produce a
            # userwarning
            if X.shape[1] <= min(k):
                param_grid = {}
                pipe = Pipeline([(estimator_name, estimator)])

                refit = refit_score
            else:
                param_grid = {
                    "feature_select__k": k,
                }
                pipe = Pipeline(
                    [
                        ("feature_select", SelectKBest(feature_selector)),
                        (estimator_name, estimator),
                    ]
                )
                if razor is True:
                    from pynets.statistics.interfaces import Razors

                    refit = Razors.simplify(
                        param="k", scoring=refit_score, rule="se", sigma=1
                    )
                else:
                    refit = refit_score

        ## Model-specific naming chunk 1
        # Make hyperparameter search-spaces
        param_space = make_param_grids()

        if predict_type == "classifier":
            if (
                "svm" in estimator_name
                or "en" in estimator_name
                or "l1" in estimator_name
                or "l2" in estimator_name
            ):
                param_grid[estimator_name + "__C"] = param_space["Cs"]
        elif predict_type == "regressor":
            param_grid[estimator_name + "__alpha"] = param_space["alphas"]
        else:
            raise ValueError("Prediction method not recognized")

        if "en" in estimator_name:
            param_grid[estimator_name + "__l1_ratio"] = param_space["l1_ratios"]
        ## Model-specific naming chunk 1

        # Establish grid-search feature/model tuning windows,
        # refit the best model using a 1 SE rule of MSE values.

        if search_method == "grid":
            pipe_grid_cv = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=n_jobs,
                cv=inner_cv,
            )
        elif search_method == "random":
            pipe_grid_cv = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=n_jobs,
                cv=inner_cv,
            )
        else:
            raise ValueError(
                f"Search method {search_method} not " f"recognized..."
            )

        # Fit pipeline to data
        pipe_grid_cv.fit(X, y.values.ravel())

        # Grab mean
        means_exp_var = pipe_grid_cv.cv_results_[f"mean_test_{refit_score}"]
        means_MSE = pipe_grid_cv.cv_results_[
            f"mean_test_neg_mean_squared_error"
        ]

        hyperparam_space = ""
        ## Model-specific naming chunk 2
        if predict_type == "classifier":
            if (
                "svm" in estimator_name
                or "en" in estimator_name
                or "l1" in estimator_name
                or "l2" in estimator_name
            ):
                c_best = pipe_grid_cv.best_estimator_.get_params()[
                    f"{estimator_name}__C"
                ]
                hyperparam_space = f"C-{c_best}"
        elif predict_type == "regressor":
            alpha_best = pipe_grid_cv.best_estimator_.get_params()[
                f"{estimator_name}__alpha"
            ]
            hyperparam_space = f"_alpha-{alpha_best}"
        else:
            raise ValueError("Prediction method not recognized")

        if "en" in estimator_name:
            best_l1 = pipe_grid_cv.best_estimator_.get_params()[
                f"{estimator_name}__l1_ratio"
            ]
            hyperparam_space = hyperparam_space + f"_l1ratio-{best_l1}"
        ## Model-specific naming chunk 2

        if pca_reduce is True and X.shape[0] < X.shape[1]:
            best_k = pipe_grid_cv.best_estimator_.named_steps[
                "feature_select"
            ].n_components
            best_estimator_name = (
                f"{predict_type}-{estimator_name}"
                f"_{hyperparam_space}_nfeatures-"
                f"{best_k}"
            )
        else:
            if X.shape[1] <= min(k):
                best_estimator_name = (
                    f"{predict_type}-{estimator_name}_{hyperparam_space}"
                )
            else:
                best_k = pipe_grid_cv.best_estimator_.named_steps[
                    "feature_select"
                ].k
                best_estimator_name = (
                    f"{predict_type}-{estimator_name}_{hyperparam_space}"
                    f"_nfeatures-{best_k}"
                )
        # print(best_estimator_name)
        means_all_exp_var[best_estimator_name] = np.nanmean(means_exp_var)
        means_all_MSE[best_estimator_name] = np.nanmean(means_MSE)

    # Get best estimator across models
    best_estimator = max(means_all_exp_var, key=means_all_exp_var.get)
    est = estimators[best_estimator.split(f"{predict_type}-")[1].split("_")[0]]

    ## Model-specific naming chunk 3
    if "en" in best_estimator:
        est.l1_ratio = float(best_estimator.split("l1ratio-")[1].split("_")[0])

    if predict_type == "classifier":
        if (
            "svm" in best_estimator
            or "en" in best_estimator
            or "l1" in best_estimator
            or "l2" in best_estimator
        ):
            est.C = float(best_estimator.split("C-")[1].split("_")[0])
    elif predict_type == "regressor":
        est.alpha = float(best_estimator.split("alpha-")[1].split("_")[0])
    else:
        raise ValueError("Prediction method not recognized")

    ## Model-specific naming chunk 3
    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = decomposition.PCA(
            n_components=int(
                best_estimator.split("nfeatures-")[1].split("_")[0]
            ),
            whiten=True,
        )
        reg = Pipeline(
            [
                ("feature_select", pca),
                (
                    best_estimator.split(f"{predict_type}-")[1].split("_")[0],
                    est,
                ),
            ]
        )
    else:
        if X.shape[1] <= min(k):
            reg = Pipeline(
                [
                    (
                        best_estimator.split(f"{predict_type}-")[1].split("_")[
                            0
                        ],
                        est,
                    )
                ]
            )
        else:
            kbest = SelectKBest(
                feature_selector,
                k=int(best_estimator.split("nfeatures-")[1].split("_")[0]),
            )
            reg = Pipeline(
                [
                    ("feature_select", kbest),
                    (
                        best_estimator.split(f"{predict_type}-")[1].split("_")[
                            0
                        ],
                        est,
                    ),
                ]
            )

    return reg, best_estimator


def build_stacked_ensemble(
    X: pd.DataFrame,
    y: pd.DataFrame,
    base_estimators: list,
    boot: bool,
    pca_reduce: bool,
    k_folds_inner: int,
    predict_type: str = "classifier",
    search_method: str = "grid",
    prefixes: list = [],
    stacked_folds: int = 10,
    voting: bool = True,
) -> typing.Tuple[object, str]:
    """
    Builds a stacked ensemble of estimators.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
    y : array-like, shape (n_samples,)
        Target values.
    base_estimators : list of estimators
        List of estimators to use in the ensemble.
    boot : bool
        Whether to bootstrap the data.
    pca_reduce : bool
        Whether to reduce the dimensionality of the data using PCA.
    k_folds_inner : int
        Number of folds for inner cross-validation.
    predict_type : str, optional
        Type of prediction to use. Either "classifier" or "regressor".
    search_method : str, optional
        Method to use for hyperparameter search. Either "grid" or "random".
    prefixes : list, optional
        List of prefixes to use for the estimators.
    stacked_folds : int, optional
        Number of folds for the stacked cross-validation.
    voting : bool, optional
        Whether to use voting or not.

    Returns
    -------
    reg : estimator
        Stacked ensemble of estimators.

    """
    from sklearn import ensemble
    from sklearn import metrics
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    if predict_type == "regressor":
        scoring = ["explained_variance", "neg_mean_squared_error"]
        refit_score = "explained_variance"
    elif predict_type == "classifier":
        scoring = ["f1", "neg_mean_squared_error"]
        refit_score = "f1"

    ens_estimators = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
    )

    feature_subspaces = split_df_to_dfs_by_prefix(X_train, prefixes=prefixes)

    if voting is True:
        layer_ests = []
        layer_weights = []
        for X_subspace in feature_subspaces:
            final_est, best_estimator = nested_fit(
                X_subspace,
                y_train,
                base_estimators,
                boot,
                pca_reduce,
                k_folds_inner,
                predict_type,
                search_method=search_method,
            )

            final_est.steps.insert(
                0,
                (
                    "selector",
                    ColumnTransformer(
                        [
                            (
                                "selector",
                                "passthrough",
                                list(X_subspace.columns),
                            )
                        ],
                        remainder="drop",
                    ),
                ),
            )
            layer_ests.append(
                (
                    list(set([i.split("_")[0] for i in X_subspace.columns]))[0],
                    final_est,
                )
            )
            prediction = cross_validate(
                final_est,
                X_train,
                y_train,
                cv=k_folds_inner,
                scoring="f1",
                return_estimator=False,
            )
            layer_weights.append(np.nanmean(prediction["test_score"]))

        norm_weights = [float(i) / sum(layer_weights) for i in layer_weights]
        ec = ensemble.VotingClassifier(
            estimators=layer_ests,
            voting="soft",
            weights=np.array(norm_weights),
        )

        ensemble_fitted = ec.fit(X_train, y_train)
        meta_est_name = "voting"
        ens_estimators[meta_est_name] = {}
        ens_estimators[meta_est_name]["model"] = ensemble_fitted
        ens_estimators[meta_est_name]["error"] = metrics.brier_score_loss(
            ensemble_fitted.predict(X_test), y_test
        )
        ens_estimators[meta_est_name][
            "score"
        ] = metrics.balanced_accuracy_score(
            ensemble_fitted.predict(X_test), y_test
        )
    else:
        # param_grid_rfc = {
        #     'n_estimators': [200, 500],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [4, 5, 6, 7, 8],
        #     'criterion': ['gini', 'entropy']
        # }
        #
        # rfc = ensemble.RandomForestClassifier(random_state=boot)
        # CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc,
        #                       cv=StratifiedKFold(n_splits=k_folds_inner,
        #                                          shuffle=True,
        #                            random_state=boot),
        #                       refit=refit_score,
        #                       scoring=scoring)

        # param_grid_gbc = {
        #     "loss": ["deviance"],
        #     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        #     "min_samples_split": np.linspace(0.1, 0.5, 12),
        #     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        #     "max_depth": [3, 5, 8],
        #     "max_features": ["log2", "sqrt"],
        #     "criterion": ["friedman_mse", "mae"],
        #     "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        #     "n_estimators": [10]
        # }
        #
        # gbc = ensemble.GradientBoostingClassifier(random_state=boot)
        # CV_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc,
        #                       cv=KFold(n_splits=k_folds_inner),
        #                       refit=refit_score,
        #                       scoring=["f1", "neg_mean_squared_error"])

        from sklearn.linear_model import LinearRegression
        from sklearn.naive_bayes import GaussianNB

        if predict_type == "classifier":
            reg = GaussianNB()
            meta_name = "nb"
        elif predict_type == "regressor":
            reg = LinearRegression(normalize=True)
            meta_name = "lr"
        else:
            raise ValueError("predict_type not recognized!")

        meta_estimators = {
            meta_name: reg,
        }

        for meta_est_name, meta_est in meta_estimators.items():
            layer_ests = []
            for X_subspace in feature_subspaces:
                final_est, best_estimator = nested_fit(
                    X_subspace,
                    y_train,
                    base_estimators,
                    boot,
                    pca_reduce,
                    k_folds_inner,
                    predict_type,
                    search_method=search_method,
                )

                final_est.steps.insert(
                    0,
                    (
                        "selector",
                        ColumnTransformer(
                            [
                                (
                                    "selector",
                                    "passthrough",
                                    list(X_subspace.columns),
                                )
                            ],
                            remainder="drop",
                        ),
                    ),
                )
                layer_ests.append(
                    (
                        list(
                            set([i.split("_")[0] for i in X_subspace.columns])
                        )[0],
                        final_est,
                    )
                )

            ec = ensemble.StackingClassifier(
                estimators=layer_ests,
                final_estimator=meta_est,
                passthrough=False,
                cv=stacked_folds,
            )

            ensemble_fitted = ec.fit(X_train, y_train)
            ens_estimators[meta_est_name] = {}
            ens_estimators[meta_est_name]["model"] = ensemble_fitted
            ens_estimators[meta_est_name]["error"] = metrics.brier_score_loss(
                ensemble_fitted.predict(X_test), y_test
            )
            ens_estimators[meta_est_name][
                "score"
            ] = metrics.balanced_accuracy_score(
                ensemble_fitted.predict(X_test), y_test
            )

    # Select best SuperLearner
    best_estimator = min(
        ens_estimators, key=lambda v: ens_estimators[v]["error"]
    )
    final_est = Pipeline(
        [(best_estimator, ens_estimators[best_estimator]["model"])]
    )
    return final_est, best_estimator


def get_feature_imp(
    X: pd.DataFrame,
    pca_reduce: bool,
    fitted: object,
    best_estimator: str,
    predict_type: str,
    dummy_run: bool,
    stack: bool,
):
    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = fitted.named_steps["feature_select"]
        comps_all = pd.DataFrame(pca.components_, columns=X.columns)

        n_pcs = pca.components_.shape[0]

        best_positions = list(
            flatten(
                [np.nanargmax(np.abs(pca.components_[i])) for i in range(n_pcs)]
            )
        )

        if dummy_run is True:
            coefs = list(flatten(np.abs(np.ones(len(best_positions))).tolist()))
        else:
            coefs = list(
                flatten(
                    np.abs(
                        fitted.named_steps[
                            best_estimator.split(f"{predict_type}-")[1].split(
                                "_"
                            )[0]
                        ].coef_
                    ).tolist()
                )
            )
        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(comps_all, coefs)).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )

        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(best_positions, feat_imp_dict.values())).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )
    else:
        if "feature_select" not in list(fitted.named_steps.keys()):
            best_positions = list(X.columns)
        else:
            best_positions = [
                column[0]
                for column in zip(
                    X.columns,
                    fitted.named_steps["feature_select"].get_support(
                        indices=True
                    ),
                )
                if column[1]
            ]

        if dummy_run is True:
            coefs = list(flatten(np.abs(np.ones(len(best_positions))).tolist()))
        elif stack is True:
            coefs = list(
                flatten(np.abs(fitted.named_steps[best_estimator].coef_))
            )
        else:
            coefs = list(
                flatten(
                    np.abs(
                        fitted.named_steps[
                            best_estimator.split(f"{predict_type}-")[1].split(
                                "_"
                            )[0]
                        ].coef_
                    ).tolist()
                )
            )

        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(best_positions, coefs)).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )
    return best_positions, feat_imp_dict


def boot_nested_iteration(
    X: pd.DataFrame,
    y: pd.DataFrame,
    predict_type: str,
    boot: int,
    feature_imp_dicts: list,
    best_positions_list: list,
    grand_mean_best_estimator: dict,
    grand_mean_best_score: dict,
    grand_mean_best_error: dict,
    grand_mean_y_predicted: dict,
    k_folds_inner: int = 10,
    k_folds_outer: int = 10,
    pca_reduce: bool = True,
    dummy_run: bool = False,
    search_method: str = "grid",
    stack: bool = False,
    stack_prefix_list: list = [],
    voting: bool = True,
) -> typing.Tuple[list, list, dict, dict, dict, dict, object]:

    # Grab CV prediction values
    if predict_type == "classifier":
        final_scorer = "roc_auc"
    else:
        final_scorer = "r2"

    if stack is True:
        k_folds_outer = len(stack_prefix_list) ** 2

    # Instantiate a dictionary of estimators
    if predict_type == "regressor":
        estimators = {
            "en": linear_model.ElasticNet(random_state=boot, warm_start=True),
            # "svm": LinearSVR(random_state=boot)
        }
        if dummy_run is True:
            estimators = {
                "dummy": DummyRegressor(strategy="prior"),
            }
    elif predict_type == "classifier":
        estimators = {
            "en": linear_model.LogisticRegression(
                penalty="elasticnet",
                fit_intercept=True,
                solver="saga",
                class_weight="auto",
                random_state=boot,
                warm_start=True,
            ),
            # "svm": LinearSVC(random_state=boot, class_weight='balanced',
            #                  loss='hinge')
        }
        if dummy_run is True:
            estimators = {
                "dummy": DummyClassifier(
                    strategy="stratified", random_state=boot
                ),
            }
    else:
        raise ValueError("Prediction method not recognized")

    # Instantiate an outer-fold
    if predict_type == "regressor":
        outer_cv = KFold(
            n_splits=k_folds_outer, shuffle=True, random_state=boot + 1
        )
    elif predict_type == "classifier":
        outer_cv = StratifiedKFold(
            n_splits=k_folds_outer, shuffle=True, random_state=boot + 1
        )
    else:
        raise ValueError("Prediction method not recognized")

    if stack is True and predict_type == "classifier":
        final_est, best_estimator = build_stacked_ensemble(
            X,
            y,
            estimators,
            boot,
            False,
            k_folds_inner,
            predict_type=predict_type,
            search_method=search_method,
            prefixes=stack_prefix_list,
            stacked_folds=k_folds_outer,
            voting=voting,
        )
    else:
        final_est, best_estimator = nested_fit(
            X,
            y,
            estimators,
            boot,
            pca_reduce,
            k_folds_inner,
            predict_type,
            search_method=search_method,
        )

    prediction = cross_validate(
        final_est,
        X,
        y,
        cv=outer_cv,
        scoring=(final_scorer, "neg_mean_squared_error"),
        return_estimator=True,
    )

    final_est.fit(X, y)

    if stack is True:
        for super_fitted in prediction["estimator"]:
            for sub_fitted in super_fitted.named_steps[
                best_estimator
            ].estimators_:
                X_subspace = pd.DataFrame(
                    sub_fitted.named_steps["selector"].transform(X),
                    columns=sub_fitted.named_steps[
                        "selector"
                    ].get_feature_names(),
                )
                best_sub_estimator = [
                    i
                    for i in sub_fitted.named_steps.keys()
                    if i in list(estimators.keys())[0] and i != "selector"
                ][0]
                best_positions, feat_imp_dict = get_feature_imp(
                    X_subspace,
                    pca_reduce,
                    sub_fitted,
                    best_sub_estimator,
                    predict_type,
                    dummy_run,
                    stack,
                )
                feature_imp_dicts.append(feat_imp_dict)
                best_positions_list.append(best_positions)
        #         print(X_subspace[list(best_positions)].columns)
        #         print(len(X_subspace[list(best_positions)].columns))
        #         X_subspace[list(best_positions)].to_csv(
        #             f"X_{X_subspace.columns[0][:3].upper()}_"
        #             f"{y.columns[0].split('_')[1]}_stack-"
        #             f"{'_'.join(stack_prefix_list)}.csv", index=False)
        # y.to_csv(f"y_{y.columns[0].split('_')[1]}_stack-"
        #          f"{'_'.join(stack_prefix_list)}.csv", index=False)
    else:
        for fitted in prediction["estimator"]:
            best_positions, feat_imp_dict = get_feature_imp(
                X,
                pca_reduce,
                fitted,
                best_estimator,
                predict_type,
                dummy_run,
                stack,
            )
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

    # Evaluate Bias-Variance
    if predict_type == "classifier":
        [avg_expected_loss, avg_bias, avg_var] = bias_variance_decomp(
            final_est, X, y, loss="mse", num_rounds=200, random_seed=42
        )

    # Save the mean CV scores for this bootstrapped iteration
    if dummy_run is False and stack is False:
        params = f"{best_estimator}"
        out_final = final_est.named_steps[
            best_estimator.split(f"{predict_type}-")[1].split("_")[0]
        ]
        if hasattr(out_final, "intercept_"):
            params = f"{params}_intercept={out_final.intercept_[0]:.5f}"

        if hasattr(out_final, "coef_"):
            params = f"{params}_betas={np.array(out_final.coef_).tolist()[0]}"

        if hasattr(out_final, "l1_ratio") and (
            hasattr(out_final, "C") or hasattr(out_final, "alpha")
        ):
            l1_ratio = out_final.l1_ratio

            if hasattr(out_final, "C"):
                C = out_final.C
                l1_val = np.round(l1_ratio * (1 / C), 5)
                l2_val = np.round((1 - l1_ratio) * (1 / C), 5)
            else:
                alpha = out_final.alpha
                l1_val = np.round(l1_ratio * alpha, 5)
                l2_val = np.round((1 - l1_ratio) * 0.5 * alpha, 5)
            params = f"{params}_lambda1={l1_val}"
            params = f"{params}_lambda2={l2_val}"

        if predict_type == "classifier":
            params = (
                f"{params}_AvgExpLoss={avg_expected_loss:.5f}_"
                f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"
            )
    else:
        if predict_type == "classifier":
            params = (
                f"{best_estimator}_AvgExpLoss={avg_expected_loss:.5f}_"
                f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"
            )
        else:
            params = f"{best_estimator}"

    grand_mean_best_estimator[boot] = params

    if predict_type == "regressor":
        grand_mean_best_score[boot] = np.nanmean(
            prediction[f"test_{final_scorer}"][
                prediction[f"test_{final_scorer}"] > 0
            ]
        )
        grand_mean_best_error[boot] = -np.nanmean(
            prediction["test_neg_mean_squared_error"]
        )
    elif predict_type == "classifier":
        grand_mean_best_score[boot] = np.nanmean(
            prediction[f"test_{final_scorer}"][
                prediction[f"test_{final_scorer}"] > 0
            ]
        )
        grand_mean_best_error[boot] = -np.nanmean(
            prediction[f"test_neg_mean_squared_error"]
        )
    else:
        raise ValueError("Prediction method not recognized")
    grand_mean_y_predicted[boot] = final_est.predict(X)

    return (
        feature_imp_dicts,
        best_positions_list,
        grand_mean_best_estimator,
        grand_mean_best_score,
        grand_mean_best_error,
        grand_mean_y_predicted,
        final_est,
    )


def bootstrapped_nested_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    nuisance_cols: list = [],
    predict_type: str = "classifier",
    n_boots: int = 10,
    nodrop_columns: list = [],
    dummy_run: bool = False,
    search_method: str = "grid",
    stack: bool = False,
    stack_prefix_list: list = [],
    remove_multi: bool = True,
    n_jobs: int = 2,
    voting: bool = True,
):
    import gc
    from joblib import Parallel, delayed
    from pynets.core.utils import mergedicts

    # Preprocess data
    [X, y] = preprocess_x_y(
        X,
        y,
        nuisance_cols,
        nodrop_columns,
        remove_multi=remove_multi,
        oversample=False,
    )

    if X.empty or len(X.columns) < 5:
        return (
            {0: "None"},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: "None"},
            None,
        )

    # Standardize Y
    if predict_type == "regressor":
        scaler = MinMaxScaler()
        y = pd.DataFrame(scaler.fit_transform(np.array(y).reshape(-1, 1)))
        # y = pd.DataFrame(np.array(y).reshape(-1, 1))
    elif predict_type == "classifier":
        y = pd.DataFrame(y)
    else:
        raise ValueError("Prediction method not recognized")

    # Instantiate a working dictionary of performance across bootstraps
    final_est = None

    # Bootstrap nested CV's "simulate" the variability of incoming data,
    # particularly when training on smaller datasets.
    # They are intended for model evaluation, rather then final deployment
    # and therefore do not include train-test splits.
    outs = Parallel(n_jobs=n_jobs)(
        delayed(boot_nested_iteration)(
            X,
            y,
            predict_type,
            boot,
            [],
            [],
            {},
            {},
            {},
            {},
            dummy_run=dummy_run,
            search_method=search_method,
            stack=stack,
            stack_prefix_list=stack_prefix_list,
            voting=True,
        )
        for boot in range(0, n_boots)
    )

    feature_imp_dicts = []
    best_positions_list = []
    final_est_list = []
    grand_mean_best_estimator_list = []
    grand_mean_best_score_list = []
    grand_mean_best_error_list = []
    grand_mean_y_predicted_list = []
    for boot in range(0, n_boots):
        [
            feature_imp_dicts_boot,
            best_positions_list_boot,
            grand_mean_best_estimator_boot,
            grand_mean_best_score_boot,
            grand_mean_best_error_boot,
            grand_mean_y_predicted_boot,
            final_est_boot,
        ] = outs[boot]
        feature_imp_dicts.append(feature_imp_dicts_boot)
        best_positions_list.append(best_positions_list_boot)
        grand_mean_best_estimator_list.append(grand_mean_best_estimator_boot)
        grand_mean_best_score_list.append(grand_mean_best_score_boot)
        grand_mean_best_error_list.append(grand_mean_best_error_boot)
        grand_mean_y_predicted_list.append(grand_mean_y_predicted_boot)
        final_est_list.append(final_est_boot)

    grand_mean_best_estimator = {}
    for d in grand_mean_best_estimator_list:
        grand_mean_best_estimator = dict(
            mergedicts(grand_mean_best_estimator, d)
        )

    grand_mean_best_score = {}
    for d in grand_mean_best_score_list:
        grand_mean_best_score = dict(mergedicts(grand_mean_best_score, d))

    grand_mean_best_error = {}
    for d in grand_mean_best_error_list:
        grand_mean_best_error = dict(mergedicts(grand_mean_best_error, d))

    grand_mean_y_predicted = {}
    for d in grand_mean_y_predicted_list:
        grand_mean_y_predicted = dict(mergedicts(grand_mean_y_predicted, d))

    unq_best_positions = list(flatten(list(np.unique(best_positions_list))))
    mega_feat_imp_dict = dict.fromkeys(unq_best_positions)

    feature_imp_dicts = [
        item for sublist in feature_imp_dicts for item in sublist
    ]

    for feat in unq_best_positions:
        running_mean = []
        for ref in feature_imp_dicts:
            if feat in ref.keys():
                running_mean.append(ref[feat])
        mega_feat_imp_dict[feat] = np.nanmean(list(flatten(running_mean)))

    mega_feat_imp_dict = OrderedDict(
        sorted(mega_feat_imp_dict.items(), key=itemgetter(1), reverse=True)
    )

    del X, y
    gc.collect()

    return (
        grand_mean_best_estimator,
        grand_mean_best_score,
        grand_mean_best_error,
        mega_feat_imp_dict,
        grand_mean_y_predicted,
        final_est,
    )


def create_wf(
    grid_params_mod: list,
    basedir: str,
    n_boots: int,
    nuisance_cols: list,
    dummy_run: bool,
    search_method: str,
    stack: bool,
    stack_prefix_list: list,
):
    import uuid
    import os
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.statistics.interfaces import BSNestedCV, MakeDF, MakeXY
    from pynets.statistics.utils import concatenate_frames
    from time import strftime

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_wf = pe.Workflow(name=f"ensemble_connectometry_{run_uuid}")
    os.makedirs(f"{basedir}/{run_uuid}", exist_ok=True)
    ml_wf.base_dir = f"{basedir}/{run_uuid}"

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["modality", "target_var", "embedding_type", "json_dict"]
        ),
        name="inputnode",
    )

    make_x_y_func_node = pe.Node(MakeXY(), name="make_x_y_func_node")

    make_x_y_func_node.iterables = [
        ("grid_param", [str(i) for i in grid_params_mod[1:]])
    ]
    make_x_y_func_node.inputs.grid_param = str(grid_params_mod[0])
    make_x_y_func_node.interface.n_procs = 1
    make_x_y_func_node.interface._mem_gb = 6

    bootstrapped_nested_cv_node = pe.Node(
        BSNestedCV(
            n_boots=int(n_boots),
            nuisance_cols=nuisance_cols,
            dummy_run=dummy_run,
            search_method=search_method,
            stack=stack,
            stack_prefix_list=stack_prefix_list,
        ),
        name="bootstrapped_nested_cv_node",
    )

    bootstrapped_nested_cv_node.interface.n_procs = 8
    bootstrapped_nested_cv_node.interface._mem_gb = 24

    make_df_node = pe.Node(MakeDF(), name="make_df_node")

    make_df_node.interface.n_procs = 1
    make_df_node.interface._mem_gb = 1

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(
            fields=[
                "df_summary",
                "modality",
                "embedding_type",
                "target_var",
                "grid_param",
            ]
        ),
        name="df_join_node",
        joinfield=["df_summary", "grid_param"],
        joinsource=make_x_y_func_node,
    )

    concatenate_frames_node = pe.Node(
        niu.Function(
            input_names=[
                "out_dir",
                "modality",
                "embedding_type",
                "target_var",
                "files_",
                "n_boots",
                "dummy_run",
                "search_method",
                "stack",
                "stack_prefix_list",
            ],
            output_names=[
                "out_path",
                "embedding_type",
                "target_var",
                "modality",
            ],
            function=concatenate_frames,
        ),
        name="concatenate_frames_node",
    )
    concatenate_frames_node.inputs.n_boots = n_boots
    concatenate_frames_node.inputs.dummy_run = dummy_run
    concatenate_frames_node.inputs.search_method = search_method
    concatenate_frames_node.inputs.stack = stack
    concatenate_frames_node.inputs.stack_prefix_list = stack_prefix_list

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["target_var", "df_summary", "embedding_type", "modality"]
        ),
        name="outputnode",
    )

    ml_wf.connect(
        [
            (
                inputnode,
                make_x_y_func_node,
                [
                    ("target_var", "target_var"),
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type"),
                    ("json_dict", "json_dict"),
                ],
            ),
            (
                make_x_y_func_node,
                bootstrapped_nested_cv_node,
                [
                    ("X", "X"),
                    ("Y", "y"),
                    ("target_var", "target_var"),
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type"),
                    ("grid_param", "grid_param"),
                ],
            ),
            (
                bootstrapped_nested_cv_node,
                make_df_node,
                [
                    ("grand_mean_best_estimator", "grand_mean_best_estimator"),
                    ("grand_mean_best_score", "grand_mean_best_score"),
                    ("grand_mean_y_predicted", "grand_mean_y_predicted"),
                    ("grand_mean_best_error", "grand_mean_best_error"),
                    ("mega_feat_imp_dict", "mega_feat_imp_dict"),
                    ("target_var", "target_var"),
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type"),
                    ("grid_param", "grid_param"),
                ],
            ),
            (
                make_df_node,
                df_join_node,
                [("df_summary", "df_summary"), ("grid_param", "grid_param")],
            ),
            (
                inputnode,
                df_join_node,
                [
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type"),
                    ("target_var", "target_var"),
                ],
            ),
            (
                df_join_node,
                concatenate_frames_node,
                [("df_summary", "files_")],
            ),
            (
                inputnode,
                concatenate_frames_node,
                [
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type"),
                    ("target_var", "target_var"),
                ],
            ),
            (
                concatenate_frames_node,
                outputnode,
                [
                    ("out_path", "df_summary"),
                    ("embedding_type", "embedding_type"),
                    ("target_var", "target_var"),
                    ("modality", "modality"),
                ],
            ),
        ]
    )

    print("Running workflow...")
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 0.5
    execution_dict["crashfile_format"] = "txt"
    execution_dict["local_hash_check"] = False
    execution_dict["stop_on_first_crash"] = False
    execution_dict["hash_method"] = "timestamp"
    execution_dict["keep_inputs"] = True
    execution_dict["use_relative_paths"] = False
    execution_dict["remove_unnecessary_outputs"] = False
    execution_dict["remove_node_directories"] = False
    execution_dict["raise_insufficient"] = False
    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            ml_wf.config[key][setting] = value

    return ml_wf


def build_predict_workflow(args: dict, retval: dict, verbose: bool = True):
    import uuid
    import psutil
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.statistics.interfaces import CopyJsonDict
    from time import strftime

    base_dir = args["base_dir"]
    feature_spaces = args["feature_spaces"]
    modality_grids = args["modality_grids"]
    target_vars = args["target_vars"]
    embedding_type = args["embedding_type"]
    modality = args["modality"]
    n_boots = args["n_boots"]
    nuisance_cols = args["nuisance_cols"]
    dummy_run = args["dummy_run"]
    search_method = args["search_method"]
    stack = args["stack"]
    stack_prefix_list = args["stack_prefix_list"]

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_meta_wf = pe.Workflow(name="pynets_multipredict")
    ml_meta_wf.base_dir = f"{base_dir}/pynets_multiperform_{run_uuid}"

    os.makedirs(ml_meta_wf.base_dir, exist_ok=True)

    grid_param_combos = [list(i) for i in modality_grids[modality]]

    grid_params_mod = []
    if modality == "func":
        for comb in grid_param_combos:
            try:
                signal, hpass, model, granularity, parcellation, smooth = comb
            except:
                try:
                    signal, hpass, model, granularity, parcellation = comb
                    smooth = "0"
                except:
                    raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append(
                [signal, hpass, model, granularity, parcellation, smooth]
            )
    elif modality == "dwi":
        for comb in grid_param_combos:
            try:
                (
                    traversal,
                    minlength,
                    model,
                    granularity,
                    parcellation,
                    error_margin,
                ) = comb
            except:
                raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append(
                [
                    traversal,
                    minlength,
                    model,
                    granularity,
                    parcellation,
                    error_margin,
                ]
            )

    meta_inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "feature_spaces",
                "base_dir",
                "modality" "modality_grids",
                "grid_params_mod",
                "embedding_type",
            ]
        ),
        name="meta_inputnode",
    )

    meta_inputnode.inputs.base_dir = base_dir
    meta_inputnode.inputs.feature_spaces = feature_spaces
    meta_inputnode.inputs.modality = modality
    meta_inputnode.inputs.modality_grids = modality_grids
    meta_inputnode.inputs.grid_params_mod = grid_params_mod
    meta_inputnode.inputs.embedding_type = embedding_type

    target_var_iter_info_node = pe.Node(
        niu.IdentityInterface(fields=["target_var"]),
        name="target_var_iter_info_node",
        nested=True,
    )

    copy_json_dict_node = pe.Node(
        CopyJsonDict(),
        name="copy_json_dict_node",
    )

    target_var_iter_info_node.iterables = [("target_var", target_vars)]

    create_wf_node = create_wf(
        grid_params_mod,
        ml_meta_wf.base_dir,
        n_boots,
        nuisance_cols,
        dummy_run,
        search_method,
        stack,
        stack_prefix_list,
    )

    final_join_node = pe.JoinNode(
        niu.IdentityInterface(
            fields=["df_summary", "embedding_type", "target_var", "modality"]
        ),
        name="final_join_node",
        joinfield=["df_summary", "embedding_type", "target_var", "modality"],
        joinsource=target_var_iter_info_node,
    )

    meta_outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["df_summary", "embedding_type", "target_var", "modality"]
        ),
        name="meta_outputnode",
    )

    create_wf_node.get_node("bootstrapped_nested_cv_node").interface.n_procs = 8
    create_wf_node.get_node(
        "bootstrapped_nested_cv_node"
    ).interface._mem_gb = 24
    create_wf_node.get_node("make_x_y_func_node").interface.n_procs = 1
    create_wf_node.get_node("make_x_y_func_node").interface._mem_gb = 6

    ml_meta_wf.connect(
        [
            (
                meta_inputnode,
                copy_json_dict_node,
                [
                    ("modality", "modality"),
                    ("feature_spaces", "feature_spaces"),
                    ("embedding_type", "embedding_type"),
                ],
            ),
            (
                target_var_iter_info_node,
                copy_json_dict_node,
                [
                    ("target_var", "target_var"),
                ],
            ),
            (
                copy_json_dict_node,
                create_wf_node,
                [
                    ("json_dict", "inputnode.json_dict"),
                    ("target_var", "inputnode.target_var"),
                    ("embedding_type", "inputnode.embedding_type"),
                    ("modality", "inputnode.modality"),
                ],
            ),
            (
                meta_inputnode,
                create_wf_node,
                [
                    ("base_dir", "concatenate_frames_node.out_dir"),
                ],
            ),
            (
                create_wf_node,
                final_join_node,
                [
                    ("outputnode.df_summary", "df_summary"),
                    ("outputnode.modality", "modality"),
                    ("outputnode.target_var", "target_var"),
                    ("outputnode.embedding_type", "embedding_type"),
                ],
            ),
            (
                final_join_node,
                meta_outputnode,
                [
                    ("df_summary", "df_summary"),
                    ("modality", "modality"),
                    ("target_var", "target_var"),
                    ("embedding_type", "embedding_type"),
                ],
            ),
        ]
    )
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_meta_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 1
    execution_dict["crashfile_format"] = "txt"
    execution_dict["local_hash_check"] = False
    execution_dict["stop_on_first_crash"] = False
    execution_dict["hash_method"] = "timestamp"
    execution_dict["keep_inputs"] = True
    execution_dict["use_relative_paths"] = False
    execution_dict["remove_unnecessary_outputs"] = False
    execution_dict["remove_node_directories"] = False
    execution_dict["raise_insufficient"] = False
    nthreads = psutil.cpu_count()
    vmem = int(list(psutil.virtual_memory())[4] / 1000000000) - 1
    procmem = [int(nthreads), [vmem if vmem > 8 else int(8)][0]]
    plugin_args = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "topological_sort",
    }
    execution_dict["plugin_args"] = plugin_args
    cfg = dict(execution=execution_dict)

    # if verbose is True:
    #     from nipype import config, logging
    #
    #     cfg_v = dict(
    #         logging={
    #             "workflow_level": "DEBUG",
    #             "utils_level": "DEBUG",
    #             "log_to_file": True,
    #             "interface_level": "DEBUG",
    #             "filemanip_level": "DEBUG",
    #         }
    #     )
    #     logging.update_logging(config)
    #     config.update_config(cfg_v)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            ml_meta_wf.config[key][setting] = value

    out = ml_meta_wf.run(plugin="MultiProc", plugin_args=plugin_args)
    # out = ml_meta_wf.run(plugin='Linear')
    return out
