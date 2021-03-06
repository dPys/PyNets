#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016
@authors: Derek Pisner
"""
import pandas as pd
import os
import numpy as np
import warnings
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
    SimpleInterface,
)
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, \
    cross_validate, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, \
    f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import linear_model, decomposition
from collections import OrderedDict
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pynets.core.utils import flatten
try:
    from sklearn.utils._testing import ignore_warnings
except:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore")

import_list = [
    "import pandas as pd",
    "import os",
    "import re",
    "import glob",
    "import numpy as np",
    "from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate",
    "from sklearn.dummy import Dummyestimator",
    "from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler",
    "from sklearn import linear_model, decomposition",
    "from pynets.stats.benchmarking import build_hp_dict",
    "import seaborn as sns",
    "import matplotlib.pyplot as plt",
    "from sklearn.base import BaseEstimator, TransformerMixin",
    "from pynets.stats.embeddings import build_asetomes, _omni_embed",
    "from joblib import Parallel, delayed",
    "from pynets.core import utils",
    "from itertools import groupby",
    "import shutil",
    "from pathlib import Path",
    "from collections import OrderedDict",
    "from operator import itemgetter",
    "from statsmodels.stats.outliers_influence import variance_inflation_factor",
    "from sklearn.impute import KNNImputer",
    "from pynets.core.utils import flatten",
    "import pickle",
    "import dill",
]


# def stack_ensemble(X, y, estimator_files, meta_estimator):
#     import joblib
#     from mlens.ensemble import SuperLearner
#
#     ensemble = SuperLearner()
#     ests = []
#     for est_file in estimator_files:
#         ests.append(joblib.load(est_file))
#
#     ensemble.add(ests)
#     ensemble.add_meta(meta_estimator)
#     ensemble.fit(X, y)
#     return ensemble


# We create a callable class, called `RazorCV`
class RazorCV(object):
    """
    PR to SKlearn by dPys 2019

    RazorCV is a callable refit option for CV whose aim is to balance model
    complexity and cross-validated score in the spirit of the
    "one standard error" rule of Breiman et al. (1984), which demonstrated that
    the tuning parameter associated with the best performance may be prone to
    overfit. To ensure model parsimony, we can instead pick the simplest model
    within one standard error (or some percentile/alpha-level tolerance) of
    the empirically optimal model. This assumes that the models can be easily
    ordered from simplest to most complex based on some user-defined target
    parameter. Greater values of this parameter are to be defined as
    'more complex', and a target scoring metric (i.e. in the case of
    multi-metric scoring), the `RazorCV` function can be called directly by
    `refit` (e.g. in GridSearchCV).

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.
    param : str
        Parameter with the largest influence on model complexity.
    greater_is_complex : bool
        Whether complexity increases as `param` increases. Default is True.
    scoring : str
        Refit scoring metric.
    method : str
        Method for balancing model complexity with performance.
        Options are 'onese', 'percentile', and 'ranksum'. Default is 'onese'.
    tol : float
        Acceptable percent tolerance in the case that a percentile threshold
        is used. Required if `method`=='percentile'. Default is 0.25.
    alpha : float
        Alpha-level to use for wilcoxon rank sum hypothesis testing. Required
        if `method`=='ranksum'. Default is 0.01.

    References
    ----------
    Breiman, Friedman, Olshen, and Stone. (1984) Classification and Regression
    Trees. Wadsworth.

    Notes
    -----
    Here, 'simplest' is defined by the complexity of the model as influenced by
    some user-defined target parameter (e.g. number of components, number of
    estimators, polynomial degree, cost, scale, number hidden units, weight
    decay, number of nearest neighbors, L1/L2 penalty, etc.).
    See :ref:`sphx_glr_auto_examples_applications_plot_model_complexity_influence.py`
    See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_refit_callable.py`
    """

    def __init__(
        self,
        cv_results,
        param,
        greater_is_complex,
        scoring,
        method,
        tol=0.25,
        alpha=0.01,
    ):
        import sklearn.metrics

        self.cv_results = cv_results
        self.param = param
        self.greater_is_complex = greater_is_complex
        self.scoring = scoring
        self.method = method
        self.scoring_funcs = [
            met
            for met in sklearn.metrics.__all__
            if (met.endswith("_score")) or (met.endswith("_error"))
        ]
        # Set _score metrics to True and _error metrics to False
        self.scoring_dict = dict(
            zip(
                self.scoring_funcs,
                [met.endswith("_score") for met in self.scoring_funcs],
            )
        )
        self.greater_is_better = self._check_scorer()
        self.tol = tol
        self.alpha = alpha

    def _check_scorer(self):
        """
        Check whether the target refit scorer is negated. If so, adjusted
        greater_is_better accordingly.
        """
        if (
            self.scoring not in self.scoring_dict.keys()
            and f"{self.scoring}_score" not in self.scoring_dict.keys()
        ):
            if self.scoring.startswith("neg_"):
                self.greater_is_better = True
            else:
                raise KeyError(f"Scoring metric {self.scoring} not "
                               f"recognized.")
        else:
            self.greater_is_better = [
                value for key, value in self.scoring_dict.items() if
                self.scoring in key][0]
        return self.greater_is_better

    def _best_low_complexity(self):
        """
        Balance model complexity with cross-validated score.
        """
        # Check parameter whose complexity we seek to restrict
        if not any(self.param in x for x in
                   self.cv_results["params"][0].keys()):
            raise KeyError("Parameter not found in cv grid.")
        else:
            self.param = [
                i for i in self.cv_results["params"][0].keys() if
                i.endswith(self.param)][0]

        if self.method == "onese":
            threshold = self.call_standard_error()
        elif self.method == "percentile":
            if self.tol is None:
                raise ValueError(
                    "For percentile method, the tolerance "
                    "(i.e. `tol`) parameter cannot be null."
                )
            threshold = self.call_percentile(tol=self.tol)
        elif self.method == "ranksum":
            if self.alpha is None:
                raise ValueError(
                    "For ranksum method, the alpha-level "
                    "(i.e. `alpha`) parameter cannot be null."
                )
            threshold = self.call_rank_sum_test(alpha=self.alpha)
        else:
            raise ValueError("Method " + self.method + " is not valid.")

        if self.greater_is_complex is True:
            candidate_idx = np.flatnonzero(
                self.cv_results["mean_test_" + self.scoring] >= threshold
            )
        else:
            candidate_idx = np.flatnonzero(
                self.cv_results["mean_test_" + self.scoring] <= threshold
            )

        best_idx = candidate_idx[
            self.cv_results["param_" + self.param][candidate_idx].argmin()
        ]
        return best_idx

    def call_standard_error(self):
        """
        Calculate the upper/lower bound within 1 standard deviation
        of the best `mean_test_scores`.
        """
        best_mean_score = self.cv_results["mean_test_" + self.scoring]
        best_std_score = self.cv_results["std_test_" + self.scoring]
        if self.greater_is_better is True:
            best_score_idx = np.argmax(best_mean_score)
            outstandard_error = (
                best_mean_score[best_score_idx] - best_std_score[best_score_idx]
            )
        else:
            best_score_idx = np.argmin(best_mean_score)
            outstandard_error = (
                best_mean_score[best_score_idx] + best_std_score[best_score_idx]
            )
        return outstandard_error

    def call_rank_sum_test(self, alpha):
        """
        Returns the performance of the simplest model whose performance is not
        significantly different across folds.
        """
        from scipy.stats import wilcoxon
        import itertools

        folds = np.vstack(
            [
                self.cv_results[fold]
                for fold in [
                    i
                    for i in self.cv_results.keys()
                    if ("split" in i) and (self.scoring in i)
                ]
            ]
        )
        tests = np.array(list(itertools.combinations(range(folds.shape[1]), 2)))

        p_dict = {}
        i = 0
        for test in tests:
            p_dict[i] = wilcoxon(folds[:, test[0]], folds[:, test[1]])[1]
            i = i + 1

        p_dict_filt = {key: val for key, val in p_dict.items() if val > alpha}
        unq_cols = np.unique(np.hstack([tests[i] for i in p_dict_filt.keys()]))

        if len(unq_cols) == 0:
            raise ValueError(
                "Models are all significantly different from one" " another"
            )
        best_mean_score = self.cv_results["mean_test_" + self.scoring][unq_cols]
        if self.greater_is_better is True:
            best_score_idx = np.argmax(best_mean_score)
        else:
            best_score_idx = np.argmin(best_mean_score)

        outstandard_error = best_mean_score[best_score_idx]
        return outstandard_error

    def call_percentile(self, tol):
        """
        Returns the simplest model that is within a percent tolerance of the
        empirically optimal model with the best `mean_test_scores`.
        """
        best_mean_score = self.cv_results["mean_test_" + self.scoring]
        if self.greater_is_better is True:
            best_score_idx = np.argmax(best_mean_score)
        else:
            best_score_idx = np.argmin(best_mean_score)

        outstandard_error = (np.abs(best_mean_score[best_score_idx]) -
                             tol) / tol
        return outstandard_error

    def standard_error(param, greater_is_complex, scoring):
        """
        Standard error callable

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        greater_is_complex : bool
            Whether complexity increases as `param` increases. Default is True.
        scoring : str
            Refit scoring metric.
        """
        from functools import partial

        def razor_pass(cv_results, param, greater_is_complex, scoring,
                       method="onese"):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(
            razor_pass,
            param=param,
            greater_is_complex=greater_is_complex,
            scoring=scoring,
        )

    def ranksum(param, greater_is_complex, scoring, alpha):
        """
        Rank sum test (Wilcoxon) callable

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        greater_is_complex : bool
            Whether complexity increases as `param` increases. Default is True.
        scoring : str
            Refit scoring metric.
        alpha : float
            Alpha-level to use for wilcoxon rank sum hypothesis testing.
            Required if `method`=='ranksum'. Default is 0.01.
        """
        from functools import partial

        def razor_pass(
            cv_results, param, greater_is_complex, scoring, alpha,
            method="ranksum"
        ):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method, alpha)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(
            razor_pass,
            param=param,
            greater_is_complex=greater_is_complex,
            scoring=scoring,
            alpha=alpha,
        )

    def percentile(param, greater_is_complex, scoring, tol):
        """
        Percentile callable

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        greater_is_complex : bool
            Whether complexity increases as `param` increases. Default is True.
        scoring : str
            Refit scoring metric.
        tol : float
            Acceptable percent tolerance in the case that a percentile
            threshold is used. Required if `method`=='percentile'.
            Default is 0.25.
        """
        from functools import partial

        def razor_pass(
            cv_results, param, greater_is_complex, scoring, tol,
            method="percentile"
        ):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method, tol)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(
            razor_pass,
            param=param,
            greater_is_complex=greater_is_complex,
            scoring=scoring,
            tol=tol,
        )


@ignore_warnings(category=ConvergenceWarning)
def nested_fit(X, y, estimators, boot, pca_reduce, k_folds,
               predict_type, search_method='grid'):

    # Instantiate an inner-fold
    if predict_type == 'regressor':
        inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=boot)
    elif predict_type == 'classifier':
        inner_cv = StratifiedKFold(n_splits=k_folds, shuffle=True,
                                   random_state=boot)
    else:
        raise ValueError('Prediction method not recognized')

    # Scoring metrics
    scoring = ["explained_variance", "neg_mean_squared_error"]
    refit_score = "explained_variance"

    if predict_type == 'regressor':
        feature_selector = f_regression
        alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5,
                  0.75, 1, 5]
    elif predict_type == 'classifier':
        feature_selector = f_classif
        Cs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # Instantiate grid of model/feature-selection params
    n_comps = [10]
    l1_ratios = [0, 0.25, 0.5, 0.75, 1]

    # Instantiate a working dictionary of performance within a bootstrap
    means_all_exp_var = {}
    means_all_MSE = {}

    # Model + feature selection by iterating grid-search across linear
    # estimators
    for estimator_name, estimator in sorted(estimators.items()):
        if pca_reduce is True and X.shape[0] < X.shape[1]:
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
            if 'en' in estimator_name:
                param_grid = {
                    estimator_name + "__l1_ratio": l1_ratios,
                    "feature_select__n_components": n_comps,
                }
            else:
                param_grid = {
                    "feature_select__n_components": n_comps,
                }

            refit = RazorCV.standard_error("n_components", True, refit_score)
            #refit = refit_score
        else:
            # <25 Features, don't perform feature selection, but produce a
            # userwarning
            if X.shape[1] < 25:
                pipe = Pipeline([(estimator_name, estimator)])
                if 'en' in estimator_name:
                    param_grid = {estimator_name + "__l1_ratio": l1_ratios}

                refit = refit_score
            else:
                pipe = Pipeline(
                    [
                        ("feature_select", SelectKBest(feature_selector)),
                        (estimator_name, estimator),
                    ]
                )
                if 'en' in estimator_name:
                    param_grid = {
                        estimator_name + "__l1_ratio": l1_ratios,
                        "feature_select__k": n_comps,
                    }
                else:
                    param_grid = {
                        "feature_select__k": n_comps,
                    }

                #refit = refit_score
                refit = RazorCV.standard_error("k", True, refit_score)

            if predict_type == 'classifier':
                param_grid[estimator_name + "__C"] = Cs
            elif predict_type == 'regressor':
                param_grid[estimator_name + "__alpha"] = alphas
            else:
                raise ValueError('Prediction method not recognized')

        # Establish grid-search feature/model tuning windows,
        # refit the best model using a 1 SE rule of MSE values.

        if search_method == 'grid':
            pipe_grid_cv = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=1,
                cv=inner_cv,
            )
        elif search_method == 'random':
            pipe_grid_cv = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=1,
                cv=inner_cv,
            )
        else:
            raise ValueError(f"Search method {search_method} not "
                             f"recognized...")

        # Fit pipeline to data
        pipe_grid_cv.fit(X, y.values.ravel())

        # Grab mean
        means_exp_var = pipe_grid_cv.cv_results_[f"mean_test_{refit_score}"]
        means_MSE = pipe_grid_cv.cv_results_[f"mean_test_neg_mean_squared_error"]

        # Apply PCA in the case that the # of features exceeds the number of
        # observations
        if predict_type == 'classifier':
            hyperparam_space = f"C-{pipe_grid_cv.best_estimator_.get_params()[estimator_name + '__C']}"
        elif predict_type == 'regressor':
            hyperparam_space = f"alpha-{pipe_grid_cv.best_estimator_.get_params()[estimator_name + '__alpha']}"
        else:
            raise ValueError('Prediction method not recognized')

        if pca_reduce is True and X.shape[0] < X.shape[1]:
            if 'en' in estimator_name:
                best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}_l1ratio-{pipe_grid_cv.best_estimator_.get_params()[estimator_name + '__l1_ratio']}_nfeatures-{pipe_grid_cv.best_estimator_.named_steps['feature_select'].n_components}"
            else:
                best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}_nfeatures-{pipe_grid_cv.best_estimator_.named_steps['feature_select'].n_components}"
        else:
            if X.shape[1] < 25:
                if 'en' in estimator_name:
                    best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}_l1ratio-{pipe_grid_cv.best_estimator_.get_params()[estimator_name + '__l1_ratio']}"
                else:
                    best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}"
            else:
                if 'en' in estimator_name:
                    best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}_l1ratio-{pipe_grid_cv.best_estimator_.get_params()[estimator_name + '__l1_ratio']}_nfeatures-{pipe_grid_cv.best_estimator_.named_steps['feature_select'].k}"
                else:
                    best_estimator_name = f"{predict_type}-{estimator_name}_{hyperparam_space}_nfeatures-{pipe_grid_cv.best_estimator_.named_steps['feature_select'].k}"

        means_all_exp_var[best_estimator_name] = np.nanmean(means_exp_var)
        means_all_MSE[best_estimator_name] = np.nanmean(means_MSE)

    # Get best estimator across models
    best_estimator = max(means_all_exp_var, key=means_all_exp_var.get)
    est = estimators[best_estimator.split(f"{predict_type}-")[1].split('_')[0]]

    if 'en' in estimator_name:
        est.l1_ratio = float(best_estimator.split("l1ratio-")[1].split('_')[0])

    if predict_type == 'classifier':
        est.C = float(best_estimator.split("C-")[1].split('_')[0])
    elif predict_type == 'regressor':
        est.alpha = float(best_estimator.split("alpha-")[1].split('_')[0])
    else:
        raise ValueError('Prediction method not recognized')

    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = decomposition.PCA(
            n_components=int(best_estimator.split("nfeatures-")[1].split('_')[0]),
            whiten=True
        )
        reg = Pipeline([("feature_select", pca),
                        (best_estimator.split(f"{predict_type}-")[1].split('_')[0], est)])
    else:
        if X.shape[1] < 25:
            reg = Pipeline([(best_estimator.split(f"{predict_type}-")[1].split('_')[0], est)])
        else:
            kbest = SelectKBest(feature_selector,
                                k=int(best_estimator.split("nfeatures-")[1].split('_')[0]))
            reg = Pipeline(
                [("feature_select", kbest),
                 (best_estimator.split(f"{predict_type}-")[1].split('_')[0], est)]
            )

    return reg, best_estimator


def bootstrapped_nested_cv(
    X,
    y,
    predict_type='regressor',
    n_boots=10,
    var_thr=.50,
    k_folds_outer=10,
    k_folds_inner=10,
    pca_reduce=False,
    remove_multi=True,
    std_dev=3,
    alpha=0.95,
    missingness_thr=0.50,
    zero_thr=0.50
):

    # y = df_all[target_var].values
    # X = df_all.drop(columns=drop_cols)
    if predict_type == 'regressor':
        scoring_metrics = ("r2", "neg_mean_squared_error")
    elif predict_type == 'classifier':
        scoring_metrics = ("f1", "neg_mean_squared_error")
    else:
        raise ValueError('Prediction method not recognized')

    # Instantiate a working dictionary of performance across bootstraps
    grand_mean_best_estimator = {}
    grand_mean_best_score = {}
    grand_mean_best_error = {}
    grand_mean_y_predicted = {}

    # Remove columns with excessive missing values
    X = X.dropna(thresh=len(X) * (1 - missingness_thr), axis=1)
    if X.empty:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (missingness): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Apply a simple imputer (note that this assumes extreme cases of
    # missingness have already been addressed). The SimpleImputer is better
    # for smaller datasets, whereas the IterativeImputer performs best on
    # larger sets.
    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer
    # imp = IterativeImputer(random_state=0, sample_posterior=True)
    # X = pd.DataFrame(imp.fit_transform(X, y), columns=X.columns)
    imp1 = SimpleImputer()
    X = pd.DataFrame(imp1.fit_transform(X), columns=X.columns)

    if X.empty:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (imputation): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Standardize X
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Drop columns with identical (or mostly identical) values in each row
    nunique = X.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique < 10].index
    X.drop(cols_to_drop, axis=1, inplace=True)
    if X.empty:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (duplication): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Remove low-variance columns
    sel = VarianceThreshold(threshold=(var_thr * (1 - var_thr)))
    sel.fit(X)
    X = X[X.columns[sel.get_support(indices=True)]]
    if X.empty:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (low-variance): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Remove outliers
    outlier_mask = (np.abs(stats.zscore(X)) < float(std_dev)).all(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]
    if X.empty and len(y) < 50:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (outliers): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Remove missing y
    y_missing_mask = np.invert(np.isnan(y))
    X = X[y_missing_mask]
    y = y[y_missing_mask]
    if X.empty or len(y) < 50:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (missing y): "
              f"{X}, {y}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )
    # imp2 = SimpleImputer()
    # y = imp2.fit_transform(np.array(y).reshape(-1, 1))

    # Remove multicollinear columns
    if remove_multi is True:
        # Create correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )

        # Find index of feature columns with correlation greater than alpha
        to_drop = [column for column in upper.columns if
                   any(upper[column] > alpha)]
        try:
            X = X.drop(X[to_drop], axis=1)
        except:
            pass

    if X.empty or len(X.columns) < 5:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (multicollinearity): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Drop sparse columns with >50% zeros
    if zero_thr > 0:
        X = X.apply(lambda x: np.where(x < 0.000001, 0, x))
        X = X.loc[:, (X == 0).mean() < zero_thr]

    if X.empty or len(X.columns) < 5:
        from colorama import Fore, Style
        print(f"\n\n{Fore.RED}Empty feature-space (Zero Columns): "
              f"{X}{Style.RESET_ALL}\n\n")
        return (
            {0: 'None'},
            {0: np.nan},
            {0: np.nan},
            {0: np.nan},
            {0: 'None'},
            None
        )

    # Standardize Y
    if predict_type == 'regressor':
        scaler = MinMaxScaler()
        y = pd.DataFrame(scaler.fit_transform(np.array(y).reshape(-1, 1)))
        # y = pd.DataFrame(np.array(y).reshape(-1, 1))
    elif predict_type == 'classifier':
        y = pd.DataFrame(y)
    else:
        raise ValueError('Prediction method not recognized')

    print(f"\nX: {X}\ny: {y}\n")

    final_est = None

    # Bootstrap nested CV's "simulates" the variability of incoming data,
    # particularly when training on smaller datasets
    feature_imp_dicts = []
    best_positions_list = []
    for boot in range(0, n_boots):
        # Instantiate a dictionary of estimators
        if predict_type == 'regressor':
            estimators = {
                # "l1": linear_model.Lasso(
                #     random_state=boot, warm_start=True
                # ),
                # "l2": linear_model.Ridge(random_state=boot, warm_start=True),
                "en": linear_model.ElasticNet(random_state=boot,
                                              warm_start=True)
            }
        elif predict_type == 'classifier':
            estimators = {
                "en": linear_model.LogisticRegression(penalty='elasticnet',
                                         solver='saga',
                                         class_weight='balanced',
                                         random_state=boot,
                                         warm_start=True)
            }
        else:
            raise ValueError('Prediction method not recognized')

        # Instantiate an outer-fold
        if predict_type == 'regressor':
            outer_cv = KFold(n_splits=k_folds_outer,
                             shuffle=True, random_state=boot + 1)
        elif predict_type == 'classifier':
            outer_cv = StratifiedKFold(n_splits=k_folds_outer, shuffle=True,
                                       random_state=boot + 1)
        else:
            raise ValueError('Prediction method not recognized')

        final_est, best_estimator = nested_fit(
            X, y, estimators, boot, pca_reduce, k_folds_inner, predict_type
        )

        # Grab CV prediction values
        prediction = cross_validate(
            final_est,
            X,
            y,
            cv=outer_cv,
            scoring=scoring_metrics,
            return_estimator=True,
        )

        for fitted in prediction["estimator"]:
            if pca_reduce is True and X.shape[0] < X.shape[1]:
                pca = fitted.named_steps["feature_select"]
                pca.fit_transform(X)
                comps_all = pd.DataFrame(pca.components_, columns=X.columns)
                coefs = np.abs(fitted.named_steps[best_estimator.split(
                    f"{predict_type}-")[1].split('_')[0]].coef_)
                feat_imp_dict = OrderedDict(
                    sorted(
                        dict(zip(comps_all, coefs)).items(),
                        key=itemgetter(1),
                        reverse=True,
                    )
                )

                n_pcs = pca.components_.shape[0]

                best_positions = [
                    np.abs(pca.components_[i]).argmax() for i in range(n_pcs)
                ]

                feat_imp_dict = OrderedDict(
                    sorted(
                        dict(zip(best_positions,
                                 feat_imp_dict.values())).items(),
                        key=itemgetter(1),
                        reverse=True,
                    )
                )
            else:
                if X.shape[1] < 25:
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

                coefs = np.abs(fitted.named_steps[best_estimator.split(
                    f"{predict_type}-")[1].split('_')[0]].coef_)

                feat_imp_dict = OrderedDict(
                    sorted(
                        dict(zip(best_positions, coefs)).items(),
                        key=itemgetter(1),
                        reverse=True,
                    )
                )
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

        # just_lps = [
        #     i
        #     for i in X.columns
        #     if i == "rum_1" or i == "dep_1" or i == "age" or i == "sex"
        # ]
        #final_est.fit(X.drop(columns=just_lps), y)
        final_est.fit(X, y)
        # Save the mean CV scores for this bootstrapped iteration
        grand_mean_best_estimator[boot] = best_estimator
        if predict_type == 'regressor':
            grand_mean_best_score[boot] = np.nanmean(prediction["test_r2"][prediction["test_r2"]>0])
            grand_mean_best_error[boot] = -np.nanmean(prediction["test_neg_mean_squared_error"])
            # grand_mean_best_score[boot] = np.nanmean(prediction["test_r2"][(np.abs(stats.zscore(prediction["test_r2"])) < float(std_dev)).all(axis=0)])
            # grand_mean_best_error[boot] = -np.nanmean(prediction["test_neg_mean_squared_error"][(np.abs(stats.zscore(prediction["test_neg_mean_squared_error"])) < float(std_dev)).all(axis=0)])
            # grand_mean_y_predicted[boot] = final_est.predict(
            #     X.drop(columns=just_lps))
        elif predict_type == 'classifier':
            grand_mean_best_score[boot] = np.nanmean(prediction["test_f1"][prediction["test_f1"]>0])
            grand_mean_best_error[boot] = -np.nanmean(prediction["test_neg_mean_squared_error"])
        else:
            raise ValueError('Prediction method not recognized')
        grand_mean_y_predicted[boot] = final_est.predict(X)

    unq_best_positions = list(flatten(list(np.unique(best_positions_list))))

    mega_feat_imp_dict = dict.fromkeys(unq_best_positions)

    for feat in unq_best_positions:
        running_mean = []
        for ref in feature_imp_dicts:
            if feat in ref.keys():
                running_mean.append(ref[feat])
        mega_feat_imp_dict[feat] = np.nanmean(list(flatten(running_mean)))

    mega_feat_imp_dict = OrderedDict(
        sorted(mega_feat_imp_dict.items(), key=itemgetter(1), reverse=True)
    )

    del X, y, scaler

    return (
        grand_mean_best_estimator,
        grand_mean_best_score,
        grand_mean_best_error,
        mega_feat_imp_dict,
        grand_mean_y_predicted,
        final_est
    )


def make_x_y(input_dict, drop_cols, target_var, embedding_type, grid_param):
    import pandas as pd
    from time import sleep
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

    if data_loaded == '{}':
        return None, None

    if str(grid_param) in data_loaded.keys():
        df_all = pd.read_json(data_loaded[str(grid_param)])
        if df_all[target_var].isin([np.nan,1]).all():
            df_all[target_var] = df_all[target_var].replace({np.nan: 0})
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
                            "age",
                            "num_visits",
                            "sex",
                            "dataset",
                            "DAY_LAG"
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
                                    "age",
                                    "num_visits",
                                    "sex",
                                    "dataset",
                                    "DAY_LAG"
                                ]
                            )
                        )
                    )
                    < 0.00001
                ).all()
            ):
                return None, None
            else:
                df_all.drop(columns=["id"], inplace=True)
                if len(df_all.columns) < 5:
                    print(f"Too few columns detected for {grid_param}...")
                    return None, None
    else:
        return None, None

    if len(df_all) < 50:
        print("\nToo few cases in feature-space after preprocessing, "
              "skipping...\n")
        return None, None
    elif len(df_all) > 50:
        drop_cols = [i for i in drop_cols if i in df_all.columns]
        return df_all.drop(columns=drop_cols), df_all[target_var].values
    else:
        print("\nEmpty/Missing Feature-space...\n")
        return None, None


def concatenate_frames(out_dir, modality, embedding_type, target_var, files_):
    import pandas as pd
    import os

    if len(files_) > 1:
        dfs = []
        for file_ in files_:
            df = pd.read_csv(file_, chunksize=100000).read()
            try:
                df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
            except BaseException:
                pass
            dfs.append(df)
        try:
            frame = pd.concat(dfs, axis=0, join="outer", sort=True,
                              ignore_index=False)
            out_path = f"{out_dir}/final_df_{modality}_{embedding_type}" \
                       f"_{target_var}.csv"
            print(f"Saving to {out_path}...")
            if os.path.isfile(out_path):
                os.remove(out_path)
            frame.to_csv(out_path, index=False)
        except ValueError:
            print(f"Dataframe concatenation failed for {modality}, "
                  f"{embedding_type}, {target_var}...")

        return out_path, embedding_type, target_var, modality
    else:
        return None, embedding_type, target_var, modality


class _copy_json_dictInputSpec(BaseInterfaceInputSpec):
    feature_spaces = traits.Any(mandatory=True)
    modality = traits.Any(mandatory=True)
    embedding_type = traits.Any(mandatory=True)
    target_var = traits.Any(mandatory=True)


class _copy_json_dictOutputSpec(TraitedSpec):
    json_dict = traits.Any(mandatory=True)
    modality = traits.Any(mandatory=True)
    embedding_type = traits.Any(mandatory=True)
    target_var = traits.Any(mandatory=True)


class copy_json_dict(SimpleInterface):
    """Interface wrapper for copy_json_dict"""

    input_spec = _copy_json_dictInputSpec
    output_spec = _copy_json_dictOutputSpec

    def _run_interface(self, runtime):
        import uuid
        import os
        # import time
        # import random
        from time import strftime
        run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
        from nipype.utils.filemanip import fname_presuffix, copyfile

        #time.sleep(random.randint(1, 30))
        if self.inputs.feature_spaces is not None and self.inputs.modality is \
            not None and self.inputs.embedding_type is not None and \
            self.inputs.target_var is not None:
            input_dict_tmp = self.inputs.feature_spaces[
                f"{self.inputs.modality}_{self.inputs.embedding_type}"
            ]
            json_dict = fname_presuffix(
                input_dict_tmp, suffix=f"_{run_uuid}_{self.inputs.modality}_"
                                       f"{self.inputs.embedding_type}_"
                                       f"{self.inputs.target_var}.json",
                newpath=runtime.cwd
            )
            copyfile(input_dict_tmp, json_dict, use_hardlink=False)
        else:
            json_dict = f"{runtime.cwd}/{run_uuid}_{self.inputs.modality}_" \
                        f"{self.inputs.embedding_type}_" \
                        f"{self.inputs.target_var}.json"
            os.mknod(json_dict)

        # time.sleep(random.randint(1, 30))

        self._results["json_dict"] = json_dict
        self._results["modality"] = self.inputs.modality
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["target_var"] = self.inputs.target_var

        return runtime


class _MakeXYInputSpec(BaseInterfaceInputSpec):
    json_dict = traits.Any(mandatory=True)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class _MakeXYOutputSpec(TraitedSpec):
    X = traits.Any(mandatory=False)
    Y = traits.Any(mandatory=False)
    embedding_type = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)
    target_var = traits.Str(mandatory=True)


class MakeXY(SimpleInterface):
    """Interface wrapper for MakeXY"""

    input_spec = _MakeXYInputSpec
    output_spec = _MakeXYOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import time
        from ast import literal_eval
        from pynets.stats.prediction import make_x_y

        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param
        self._results["modality"] = self.inputs.modality
        self._results["target_var"] = self.inputs.target_var

        if self.inputs.json_dict is not None:
            if os.path.isfile(self.inputs.json_dict) and \
                self.inputs.json_dict.endswith('.json') and \
                os.stat(self.inputs.json_dict).st_size != 0:
                if self.inputs.target_var == "rumination_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "dep_2", "rum_2"]
                elif self.inputs.target_var == "depression_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "rumination_persist_phenotype",
                                 "rum_1", "dep_2", "rum_2"]
                elif self.inputs.target_var == "rum_1":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "rumination_persist_phenotype",
                                 "rum_2", "dep_2", "dep_1"]
                elif self.inputs.target_var == "dep_1":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "rumination_persist_phenotype",
                                 "rum_2", "dep_2", "rum_1"]
                elif self.inputs.target_var == "dep_2":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "rumination_persist_phenotype",
                                 "rum_2", "rum_1"]
                elif self.inputs.target_var == "rum_2":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "rumination_persist_phenotype",
                                 "dep_2"]
                else:
                    drop_cols = [self.inputs.target_var,
                                 "rumination_persist_phenotype",
                                 "depression_persist_phenotype",
                                 "dep_1", "rum_1"]

                drop_cols = drop_cols + ["id", "participant_id"]

                [X, Y] = make_x_y(
                    self.inputs.json_dict,
                    drop_cols,
                    self.inputs.target_var,
                    self.inputs.embedding_type,
                    tuple(literal_eval(self.inputs.grid_param)),
                )
                if X is not None:
                    self._results["X"] = X
                    self._results["Y"] = Y
                else:
                    self._results["X"] = None
                    self._results["Y"] = None
            else:
                self._results["X"] = None
                self._results["Y"] = None
        else:
            self._results["X"] = None
            self._results["Y"] = None

        return runtime


class _BSNestedCVInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for BSNestedCV"""

    X = traits.Any(mandatory=False)
    y = traits.Any(mandatory=False)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class _BSNestedCVOutputSpec(TraitedSpec):
    """Output interface wrapper for BSNestedCV"""

    grand_mean_best_estimator = traits.Dict({0: 'None'}, mandatory=True,
                                            usedefault=True)
    grand_mean_best_score = traits.Dict({0: np.nan}, mandatory=True,
                                        usedefault=True)
    grand_mean_y_predicted = traits.Dict({0: np.nan}, mandatory=True,
                                         usedefault=True)
    grand_mean_best_error = traits.Dict({0: np.nan}, mandatory=True,
                                        usedefault=True)
    mega_feat_imp_dict = traits.Dict({0: 'None'}, mandatory=True,
                                     usedefault=True)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)
    out_path_est = traits.Any()


class BSNestedCV(SimpleInterface):
    """Interface wrapper for BSNestedCV"""

    input_spec = _BSNestedCVInputSpec
    output_spec = _BSNestedCVOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        from colorama import Fore, Style
        from joblib import dump

        self._results["target_var"] = self.inputs.target_var
        self._results["modality"] = self.inputs.modality
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param

        if 'phenotype' in self.inputs.target_var:
            predict_type = 'classifier'
        else:
            predict_type = 'regressor'

        if self.inputs.X is None:
            return runtime

        if not self.inputs.X.empty and not np.isnan(self.inputs.y).all():
            if isinstance(self.inputs.X, pd.DataFrame):
                if self.inputs.modality == 'func':
                    [
                        grand_mean_best_estimator,
                        grand_mean_best_score,
                        grand_mean_best_error,
                        mega_feat_imp_dict,
                        grand_mean_y_predicted,
                        final_est
                    ] = bootstrapped_nested_cv(self.inputs.X, self.inputs.y,
                                               predict_type=predict_type)
                else:
                    [
                        grand_mean_best_estimator,
                        grand_mean_best_score,
                        grand_mean_best_error,
                        mega_feat_imp_dict,
                        grand_mean_y_predicted,
                        final_est
                    ] = bootstrapped_nested_cv(self.inputs.X, self.inputs.y,
                                               predict_type=predict_type,
                                               var_thr=.20, zero_thr=0.75)
                if final_est:
                    out_path_est = f"{runtime.cwd}/estimator_" \
                                   f"{self.inputs.target_var}_" \
                                   f"{self.inputs.modality}_" \
                                   f"{self.inputs.embedding_type}_" \
                                   f"{self.inputs.grid_param.replace(', ', '_')}.joblib"

                    dump(final_est, out_path_est)
                else:
                    out_path_est = None

                if len(mega_feat_imp_dict.keys()) > 1:
                    print(
                        f"\n\n{Fore.BLUE}Target Outcome: "
                        f"{Fore.GREEN}{self.inputs.target_var}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Modality: "
                        f"{Fore.RED}{self.inputs.modality}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Embedding type: "
                        f"{Fore.RED}{self.inputs.embedding_type}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Grid Params: "
                        f"{Fore.RED}{self.inputs.grid_param}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Best Estimator: "
                        f"{Fore.RED}{grand_mean_best_estimator}{Style.RESET_ALL}"
                    )
                    print(
                        f"\n{Fore.BLUE}Variance: "
                        f"{Fore.RED}{grand_mean_best_score}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Error: "
                        f"{Fore.RED}{grand_mean_best_error}{Style.RESET_ALL}\n"
                    )
                    #print(f"y_actual: {self.inputs.y}")
                    #print(f"y_predicted: {grand_mean_y_predicted}")
                    print(
                        f"{Fore.BLUE}Feature Importance: "
                        f"{Fore.RED}{list(mega_feat_imp_dict.keys())}{Style.RESET_ALL} "
                        f"with {Fore.RED}{len(mega_feat_imp_dict.keys())} "
                        f"features...{Style.RESET_ALL}\n\n"
                    )
                else:
                    print(f"{Fore.RED}Empty feature-space for "
                          f"{self.inputs.grid_param}, "
                          f"{self.inputs.target_var}, "
                          f"{self.inputs.embedding_type}, "
                          f"{self.inputs.modality}{Style.RESET_ALL}")
                    grand_mean_best_estimator = {0: 'None'}
                    grand_mean_best_score = {0: np.nan}
                    grand_mean_y_predicted = {0: np.nan}
                    grand_mean_best_error = {0: np.nan}
                    mega_feat_imp_dict = {0: 'None'}
            else:
                print(f"{Fore.RED}{self.inputs.X} is not pd.DataFrame for"
                      f" {self.inputs.grid_param}, {self.inputs.target_var},"
                      f" {self.inputs.embedding_type}, "
                      f"{self.inputs.modality}{Style.RESET_ALL}")
                grand_mean_best_estimator = {0: 'None'}
                grand_mean_best_score = {0: np.nan}
                grand_mean_y_predicted = {0: np.nan}
                grand_mean_best_error = {0: np.nan}
                mega_feat_imp_dict = {0: 'None'}
                out_path_est = None
        else:
            print(
                f"{Fore.RED}Empty feature-space for {self.inputs.grid_param},"
                f" {self.inputs.target_var}, {self.inputs.embedding_type},"
                f" {self.inputs.modality}{Style.RESET_ALL}")
            grand_mean_best_estimator = {0: 'None'}
            grand_mean_best_score = {0: np.nan}
            grand_mean_y_predicted = {0: np.nan}
            grand_mean_best_error = {0: np.nan}
            mega_feat_imp_dict = {0: 'None'}
            out_path_est = None
        gc.collect()

        self._results[
            "grand_mean_best_estimator"] = grand_mean_best_estimator
        self._results[
            "grand_mean_best_score"] = grand_mean_best_score
        self._results[
            "grand_mean_y_predicted"] = grand_mean_y_predicted
        self._results[
            "grand_mean_best_error"] = grand_mean_best_error
        self._results["mega_feat_imp_dict"] = mega_feat_imp_dict
        self._results["out_path_est"] = out_path_est

        return runtime


class _MakeDFInputSpec(BaseInterfaceInputSpec):
    grand_mean_best_estimator = traits.Dict(mandatory=True)
    grand_mean_best_score = traits.Dict(mandatory=True)
    grand_mean_y_predicted = traits.Dict(mandatory=True)
    grand_mean_best_error = traits.Dict(mandatory=True)
    mega_feat_imp_dict = traits.Dict(mandatory=True)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class _MakeDFOutputSpec(TraitedSpec):
    df_summary = traits.Any(mandatory=False)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    target_var = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class MakeDF(SimpleInterface):

    input_spec = _MakeDFInputSpec
    output_spec = _MakeDFOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        from ast import literal_eval
        import pandas as pd
        import numpy as np
        from colorama import Fore, Style

        def get_CI(stats, alpha=0.95):
            p = ((1.0 - alpha) / 2.0) * 100
            lower = max(0.0, np.nanpercentile(stats, p))
            p = (alpha + ((1.0 - alpha) / 2.0)) * 100
            upper = min(1.0, np.nanpercentile(stats, p))
            # print('%.1f confidence interval %.1f%% and %.1f%%' % (
            #     alpha * 100, lower * 100, upper * 100))
            return lower, upper

        df_summary = pd.DataFrame(
            columns=[
                "modality",
                "grid",
                "embedding_type",
                "best_estimator",
                "Score",
                "Error",
                "Score_95CI_upper",
                "Score_95CI_lower",
                "Error_95CI_upper",
                "Error_95CI_lower",
                "Score_90CI_upper",
                "Score_90CI_lower",
                "Error_90CI_upper",
                "Error_90CI_lower",
                "target_variable",
                "lp_importance",
                "Predicted_y",
            ]
        )

        df_summary.at[0, "target_variable"] = self.inputs.target_var
        df_summary.at[0, "modality"] = self.inputs.modality
        df_summary.at[0, "embedding_type"] = self.inputs.embedding_type
        df_summary.at[0, "grid"] = tuple(literal_eval(self.inputs.grid_param))

        if bool(self.inputs.grand_mean_best_score) is True and \
            len(self.inputs.mega_feat_imp_dict.keys()) > 1:
            y_pred_boots = [i for i in
                            list(self.inputs.grand_mean_y_predicted.values())
                            if not np.isnan(i).all()]
            if len(y_pred_boots) > 0:
                max_row_len = max([len(ll) for ll in y_pred_boots])
                y_pred_vals = np.nanmean(
                    [
                        [el for el in row] +
                        [np.NaN] * max(0, max_row_len - len(row))
                        for row in y_pred_boots
                    ],
                    axis=0,
                )
            else:
                y_pred_vals = np.nan
        else:
            y_pred_vals = np.nan

        if bool(self.inputs.grand_mean_best_estimator) is True:
            df_summary.at[0, "best_estimator"] = max(
                set(list(self.inputs.grand_mean_best_estimator.values())),
                key=list(self.inputs.grand_mean_best_estimator.values()).count,
            )
            df_summary.at[0, "Score"] = np.nanmean(
                list(self.inputs.grand_mean_best_score.values())
            )
            df_summary.at[0, "Predicted_y"] = y_pred_vals
            df_summary.at[0, "Error"] = np.mean(
                list(self.inputs.grand_mean_best_error.values())
            )
            df_summary.at[0, "Score_95CI_upper"] = get_CI(list(self.inputs.grand_mean_best_score.values()), alpha=0.95)[1]
            df_summary.at[0, "Score_95CI_lower"] = get_CI(list(self.inputs.grand_mean_best_score.values()), alpha=0.95)[0]
            df_summary.at[0, "Score_90CI_upper"] = get_CI(list(self.inputs.grand_mean_best_score.values()), alpha=0.90)[1]
            df_summary.at[0, "Score_90CI_lower"] = get_CI(list(self.inputs.grand_mean_best_score.values()), alpha=0.90)[0]
            df_summary.at[0, "Error_95CI_upper"] = get_CI(list(self.inputs.grand_mean_best_error.values()), alpha=0.95)[1]
            df_summary.at[0, "Error_95CI_lower"] = get_CI(list(self.inputs.grand_mean_best_error.values()), alpha=0.95)[0]
            df_summary.at[0, "Error_90CI_upper"] = get_CI(list(self.inputs.grand_mean_best_error.values()), alpha=0.90)[1]
            df_summary.at[0, "Error_90CI_lower"] = get_CI(list(self.inputs.grand_mean_best_error.values()), alpha=0.90)[0]
            df_summary.at[0, "lp_importance"] = np.array(
                list(self.inputs.mega_feat_imp_dict.keys())
            )
        else:
            df_summary.at[0, "best_estimator"] = np.nan
            df_summary.at[0, "Score"] = np.nan
            df_summary.at[0, "Predicted_y"] = np.nan
            df_summary.at[0, "Error"] = np.nan
            df_summary.at[0, "Score_95CI_upper"] = np.nan
            df_summary.at[0, "Score_95CI_lower"] = np.nan
            df_summary.at[0, "Score_90CI_upper"] = np.nan
            df_summary.at[0, "Score_90CI_lower"] = np.nan
            df_summary.at[0, "Error_95CI_upper"] = np.nan
            df_summary.at[0, "Error_95CI_lower"] = np.nan
            df_summary.at[0, "Error_90CI_upper"] = np.nan
            df_summary.at[0, "Error_90CI_lower"] = np.nan
            df_summary.at[0, "lp_importance"] = np.nan

        out_df_summary = f"{runtime.cwd}/df_summary_" \
                         f"{self.inputs.target_var}_" \
                         f"{self.inputs.modality}_" \
                         f"{self.inputs.embedding_type}_" \
                         f"{self.inputs.grid_param.replace(', ','_')}.csv"

        df_summary.to_csv(out_df_summary, index=False)
        print(f"Writing dataframe to file {out_df_summary}...")
        self._results["df_summary"] = out_df_summary
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["modality"] = self.inputs.modality
        self._results["grid_param"] = self.inputs.grid_param
        gc.collect()

        return runtime


def create_wf(grid_params_mod, basedir):
    from pynets.stats.prediction import MakeXY, MakeDF, BSNestedCV, \
        concatenate_frames
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    import uuid
    import os
    from time import strftime

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_wf = pe.Workflow(name=f"ensemble_connectometry_{run_uuid}")
    os.makedirs(f"{basedir}/{run_uuid}", exist_ok=True)
    ml_wf.base_dir = f"{basedir}/{run_uuid}"

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "modality",
                "target_var",
                "embedding_type",
                "json_dict",
            ]
        ),
        name="inputnode",
    )

    make_x_y_func_node = pe.Node(MakeXY(), name="make_x_y_func_node")

    make_x_y_func_node.iterables = [("grid_param",
                                     [str(i) for i in grid_params_mod[1:]])]
    make_x_y_func_node.inputs.grid_param = str(grid_params_mod[0])
    make_x_y_func_node.interface.n_procs = 1
    make_x_y_func_node.interface._mem_gb = 2

    bootstrapped_nested_cv_node = pe.Node(
        BSNestedCV(), name="bootstrapped_nested_cv_node")

    bootstrapped_nested_cv_node.interface.n_procs = 1
    bootstrapped_nested_cv_node.interface._mem_gb = 4

    make_df_node = pe.Node(MakeDF(), name="make_df_node")

    make_df_node.interface.n_procs = 1
    make_df_node.interface._mem_gb = 1

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary", "modality", "embedding_type",
                                      "target_var", "grid_param"]),
        name="df_join_node",
        joinfield=["df_summary", "grid_param"],
        joinsource=make_x_y_func_node,
    )

    concatenate_frames_node = pe.Node(
        niu.Function(
            input_names=["out_dir", "modality", "embedding_type",
                         "target_var", "files_"],
            output_names=["out_path", "embedding_type", "target_var",
                          "modality"],
            function=concatenate_frames,
        ),
        name="concatenate_frames_node",
    )
    concatenate_frames_node.inputs.out_dir = ml_wf.base_dir

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["target_var", "df_summary",
                                      "embedding_type", "modality"]),
        name="outputnode"
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
                    ("json_dict", "json_dict")
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
                [
                    ("df_summary", "df_summary"),
                    ("grid_param", "grid_param")
                ],
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
                [("modality", "modality"),
                 ("embedding_type", "embedding_type"),
                 ("target_var", "target_var")],
            ),
            (concatenate_frames_node, outputnode,
             [("out_path", "df_summary"), ("embedding_type", "embedding_type"),
              ("target_var", "target_var"), ("modality", "modality")]),
        ]
    )

    print("Running workflow...")
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 0.5
    execution_dict["crashfile_format"] = "txt"
    execution_dict["local_hash_check"] = False
    execution_dict["stop_on_first_crash"] = False
    execution_dict['hash_method'] = 'timestamp'
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


def build_predict_workflow(args, retval, verbose=True):
    from pynets.stats.prediction import create_wf
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    import uuid
    import psutil
    from time import strftime

    base_dir = args["base_dir"]
    feature_spaces = args["feature_spaces"]
    modality_grids = args["modality_grids"]
    target_vars = args["target_vars"]
    embedding_type = args["embedding_type"]
    modality = args["modality"]

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_meta_wf = pe.Workflow(name="pynets_multipredict")
    ml_meta_wf.base_dir = f"{base_dir}/pynets_multiperform_{run_uuid}"

    os.makedirs(ml_meta_wf.base_dir, exist_ok=True)

    grid_param_combos = [list(i) for i in modality_grids[modality]]

    grid_params_mod = []
    if modality == "func":
        for comb in grid_param_combos:
            try:
                extract, hpass, model, res, atlas, smooth = comb
            except:
                try:
                    extract, hpass, model, res, atlas = comb
                    smooth = "0"
                except:
                    raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append([extract, hpass, model, res, atlas, smooth])
    elif modality == "dwi":
        for comb in grid_param_combos:
            try:
                directget, minlength, model, res, atlas, tol = comb
            except:
                raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append([directget, minlength, model, res, atlas, tol])

    meta_inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "feature_spaces",
                "base_dir",
                "modality"
                "modality_grids",
                "grid_params_mod",
                "embedding_type"
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
        niu.IdentityInterface(
            fields=[
                "target_var"
            ]
        ),
        name="target_var_iter_info_node", nested=True
    )

    copy_json_dict_node = pe.Node(
        copy_json_dict(),
        name="copy_json_dict_node",
    )

    target_var_iter_info_node.iterables = [("target_var", target_vars)]

    create_wf_node = create_wf(grid_params_mod, ml_meta_wf.base_dir)

    final_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary", "embedding_type",
                                      "target_var", "modality"]),
        name="final_join_node",
        joinfield=["df_summary", "embedding_type", "target_var", "modality"],
        joinsource=target_var_iter_info_node,
    )

    meta_outputnode = pe.Node(
        niu.IdentityInterface(fields=["df_summary", "embedding_type",
                                      "target_var", "modality"]),
        name="meta_outputnode"
    )

    ml_meta_wf.connect(
        [
            (
                meta_inputnode,
                copy_json_dict_node,
                [
                    ("modality", "modality"),
                    ("feature_spaces", "feature_spaces"),
                    ("embedding_type", "embedding_type")
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
                    ("modality", "inputnode.modality")
                ],
            ),
            (
                create_wf_node,
                final_join_node,
                [
                    ("outputnode.df_summary", "df_summary"),
                    ("outputnode.modality", "modality"),
                    ("outputnode.target_var", "target_var"),
                    ("outputnode.embedding_type", "embedding_type")
                ],
            ),
            (final_join_node, meta_outputnode,
             [("df_summary", "df_summary"),
              ("modality", "modality"),
              ("target_var", "target_var"),
              ("embedding_type", "embedding_type")
              ]),
        ]
    )
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_meta_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 1
    execution_dict["crashfile_format"] = "txt"
    execution_dict["local_hash_check"] = False
    execution_dict["stop_on_first_crash"] = False
    execution_dict['hash_method'] = 'timestamp'
    execution_dict["keep_inputs"] = True
    execution_dict["use_relative_paths"] = False
    execution_dict["remove_unnecessary_outputs"] = False
    execution_dict["remove_node_directories"] = False
    execution_dict["raise_insufficient"] = False
    nthreads = psutil.cpu_count() * 2
    procmem = [int(nthreads),
               int(list(psutil.virtual_memory())[4] / 1000000000) - 2]
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

    out = ml_meta_wf.run(plugin='MultiProc', plugin_args=plugin_args)
    #out = ml_meta_wf.run(plugin='Linear')
    return out
