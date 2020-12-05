#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016
@authors: Derek Pisner
"""
import pandas as pd
import os
import dill
import re
import glob
import numpy as np
import itertools
import warnings
from nipype.interfaces.base import (
    BaseInterface,
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
from pynets.core.utils import flatten, mergedicts
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


def get_ensembles_embedding(modality, alg, base_dir):
    if alg == "OMNI":
        ensembles = list(
            set(
                [
                    "rsn-"
                    + os.path.basename(i).split(alg + "_")[1].split("_")[1]
                    + "_res-"
                    + os.path.basename(i).split(alg + "_")[1].split("_")[0]
                    + "_"
                    + os.path.basename(i).split(modality + "_")[1].replace(".npy", "")
                    for i in glob.glob(
                        f"{base_dir}/embeddings_all_{modality}/*/*/*/*{alg}*.npy"
                    )
                ]
            )
        )
    elif alg == "ASE" or alg == "vectorize":
        ensembles = list(
            set(
                [
                    "rsn-"
                    + os.path.basename(i).split(alg + "_")[1].split("_rawgraph")[0].split('_')[1]
                    + "_res-"
                    + os.path.basename(i).split(alg + "_")[1].split("_")[0]
                    + "_"
                    + os.path.basename(i)
                        .split(modality + "_")[1]
                        .replace(".npy", "")
                    for i in glob.glob(
                        f"{base_dir}/embeddings_all_{modality}/*/*/*/*{alg}*.npy")
                ]
            )
        )
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
        df_top = df_top.rename(
            columns=lambda x: re.sub("_partcorr", "_model-partcorr", x)
        )
        df_top = df_top.rename(columns=lambda x: re.sub("_corr", "_model-corr", x))
        df_top = df_top.rename(columns=lambda x: re.sub("_cov", "_model-cov", x))
        df_top = df_top.rename(columns=lambda x: re.sub("_sfm", "_model-sfm", x))
        df_top = df_top.rename(columns=lambda x: re.sub("_csa", "_model-csa", x))
        df_top = df_top.rename(columns=lambda x: re.sub("_tensor", "_model-tensor", x))
        df_top = df_top.rename(columns=lambda x: re.sub("_csd", "_model-csd", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("thrtype-PROP", "thrtype-MST", x))
        # df_top = df_top.dropna(how='all')
        # df_top = df_top.dropna(axis='columns',
        #                        thresh=drop_thr * len(df_top)
        #                        )
        if not df_top.empty and len(df_top.columns) > 1:
            [df_top, ensembles] = graph_theory_prep(df_top, thr_type)
            #print(df_top)
            ensembles = [i for i in ensembles if i != "id"]
        else:
            ensembles = None
            df_top = None
    else:
        ensembles = None
        df_top = None
    return ensembles, df_top


def make_feature_space_dict(
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
                extract, hpass, model, res, atlas, smooth = comb
            except:
                try:
                    extract, hpass, model, res, atlas = comb
                    smooth = "0"
                except:
                    raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append((atlas, extract, hpass, model, res,
                                    str(smooth)))
    elif target_modality == "dwi":
        for comb in grid_params:
            try:
                directget, minlength, model, res, atlas, tol = comb
            except:
                raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append((atlas, directget, minlength, model, res,
                                    tol))

    par_dict = subject_dict.copy()

    with Parallel(
        n_jobs=-1, backend='loky', verbose=10, temp_folder=cache_dir
    ) as parallel:
        outs = parallel(
            delayed(create_feature_space)(
                df,
                grid_param,
                par_dict,
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
                iid=False,
            )
        elif search_method == 'random':
            pipe_grid_cv = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=1,
                cv=inner_cv,
                iid=False,
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


def build_grid(modality, hyperparam_dict, hyperparams, ensembles):
    from pynets.stats.benchmarking import build_hp_dict

    for ensemble in ensembles:
        try:
            build_hp_dict(ensemble, modality, hyperparam_dict, hyperparams)
        except:
            print(f"Failed to parse ensemble {ensemble}...")

    if "rsn" in hyperparam_dict.keys():
        hyperparam_dict["rsn"] = [i for i in hyperparam_dict["rsn"] if "res"
                                  not in i]

    hyperparam_dict = OrderedDict(sorted(hyperparam_dict.items(),
                                         key=lambda x: x[0]))
    grid = list(
        itertools.product(*(hyperparam_dict[param] for param in
                            hyperparam_dict.keys()))
    )

    return hyperparam_dict, grid


def get_index_labels(base_dir, ID, ses, modality, grid_param, emb_shape):
    import os
    import json
    import ast

    label_file = glob.glob(
        f"{base_dir}/embeddings_all_dwi/sub-{ID}/ses-{ses}/rsn-{grid_param[0]}_res-{grid_param[-2]}/nodes/*.json")
    if len(label_file) > 0:
        label_file = label_file[0]
        try:
            with open(label_file, 'r+') as f:
                node_dict = json.loads(f.read())
            f.close()
        except:
            print(UserWarning(f"Failed to parse {label_file}..."))
            return [None]
    else:
        label_file = glob.glob(
            f"{base_dir}/pynets/sub-{ID}/ses-{ses}/{modality}/rsn-{grid_param[0]}_res-{grid_param[-2]}/nodes/*.json")
        if len(label_file) > 0:
            label_file = label_file[0]
            with open(label_file, 'r+') as f:
                node_dict = json.loads(f.read())
            f.close()
        else:
            return [None]

    # Correct labels/index if needed
    if isinstance(node_dict, list):
        if all(v is None for v in [i['label'] for i in node_dict]):
            node_dict_revised = {}
            for i in range(len(node_dict)):
                node_dict_revised[i] = {}
                node_dict_revised[i]['label'], node_dict_revised[i][
                    'index'] = ast.literal_eval(
                    node_dict[i]['index'].replace('\n', ','))
            ixs = [i['index'] for i in node_dict_revised.values()]
        else:
            ixs = [i['index'] for i in node_dict]

    else:
        ixs = [i['index'] for i in node_dict.values()]

    if emb_shape == len(ixs):
        return ixs
    else:
        return [None]


def flatten_latent_positions(subject_dict, ID, ses, modality, grid_param,
                             alg):

    if grid_param in subject_dict[ID][str(ses)][modality][alg].keys():
        rsn_dict = subject_dict[ID][str(ses)][modality][alg][grid_param]

        if 'data' in rsn_dict.keys():
            if not isinstance(rsn_dict["data"], np.ndarray):
                rsn_dict["data"] = np.load(rsn_dict["data"])

            ixs = rsn_dict["index"]

            if len(ixs) == rsn_dict["data"].shape[0]:
                rsn_arr = rsn_dict["data"].T.reshape(
                    1, rsn_dict["data"].T.shape[0] * rsn_dict["data"].T.shape[1]
                )
                if rsn_dict["data"].shape[1] == 1:
                    df_lps = pd.DataFrame(rsn_arr, columns=[f"{i}_rsn-{grid_param[0]}_res-{grid_param[-2]}_dim1"
                                                            for i in ixs])
                elif rsn_dict["data"].shape[1] == 3:
                    df_lps = pd.DataFrame(
                        rsn_arr,
                        columns=[f"{i}_rsn-{grid_param[0]}_res-{grid_param[-2]}_dim1" for i in ixs]
                        + [f"{i}_rsn-{grid_param[0]}_res-{grid_param[-2]}_dim2" for i in ixs]
                        + [f"{i}_rsn-{grid_param[0]}_res-{grid_param[-2]}_dim3" for i in ixs],
                    )
                else:
                    raise ValueError(
                        f"Number of dimensions {rsn_dict['data'].shape[1]} "
                        f"not supported. See flatten_latent_positions "
                        f"function..."
                    )
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
            print(UserWarning(f"Missing {grid_param} universe..."))
            df_lps = None
    else:
        print(UserWarning(f"Missing {grid_param} universe..."))
        df_lps = None

    return df_lps


def create_feature_space(df, grid_param, subject_dict, ses, modality, alg,
                         mets=None):
    df_tmps = []

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

        if alg == "OMNI" or alg == "ASE":
            df_lps = flatten_latent_positions(
                subject_dict, ID, ses, modality, grid_param, alg
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
        df_all = df_all.replace({0: np.nan})
        # df_all = df_all.apply(lambda x: np.where(x < 0.00001, np.nan, x))
        #print(len(df_all))
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


def bootstrapped_nested_cv(
    X,
    y,
    predict_type='regressor',
    n_boots=10,
    var_thr=.50,
    k_folds_outer=5,
    k_folds_inner=5,
    pca_reduce=False,
    remove_multi=True,
    std_dev=3,
    alpha=0.95,
    missingness_thr=0.20,
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
        )

    # Drop sparse columns with >50% zeros
    X = X.apply(lambda x: np.where(x < 0.000001, 0, x))
    X = X.loc[:, (X == 0).mean() < .50]

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
        del final_est

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
    )


def make_subject_dict(
    modalities, base_dir, thr_type, mets, embedding_types, template, sessions,
    rsns):
    from joblib import Parallel, delayed
    from pynets.core.utils import mergedicts
    import tempfile
    import psutil
    import shutil
    import gc

    hyperparams_func = ["rsn", "res", "model", "hpass", "extract", "smooth"]
    hyperparams_dwi = ["rsn", "res", "model", "directget", "minlength", "tol"]

    miss_frames_all = []
    subject_dict_all = {}
    modality_grids = {}
    for modality in modalities:
        print(f"MODALITY: {modality}")
        hyperparams = eval(f"hyperparams_{modality}")
        for alg in embedding_types:
            print(f"EMBEDDING TYPE: {alg}")
            for ses_name in sessions:
                if alg == "ASE" or alg == "OMNI" or alg == "vectorize":
                    ids = [
                        f"{os.path.basename(i)}_ses-{ses_name}"
                        for i in
                        glob.glob(f"{base_dir}/embeddings_all_{modality}/*")
                        if os.path.basename(i).startswith("sub")
                    ]
                else:
                    ids = [
                        f"{os.path.basename(i)}_ses-{ses_name}"
                        for i in glob.glob(f"{base_dir}/pynets/*")
                        if os.path.basename(i).startswith("sub")
                    ]

                ids = [i for i in ids if 's030' not in i]
                if alg == "ASE" or alg == "OMNI" or alg == "vectorize":
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
                    modality, hyperparam_dict, sorted(list(set(hyperparams))),
                    ensembles)[1]

                grid = list(set([i for i in grid if i != () and
                                 len(list(i)) > 0]))

                # In the case that we are using all of the 3 RSN connectomes
                # (pDMN, coSN, and fECN) in the feature-space,
                # rather than varying them as hyperparameters (i.e. we assume
                # they each add distinct variance
                # from one another) Create an abridged grid, where
                if modality == "func":
                    modality_grids[modality] = grid
                else:
                    modality_grids[modality] = grid

                par_dict = subject_dict_all.copy()
                cache_dir = tempfile.mkdtemp()

                with Parallel(
                    n_jobs=len(ids),
                    backend='loky',
                    verbose=1,
                    max_nbytes=f"{int(float(list(psutil.virtual_memory())[4]/len(ids)))}M",
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
                            mets,
                            df_top,
                        )
                        for id in ids
                    )
                del par_dict
                gc.collect()
                outs = [i[0] for i in outs_tup]
                miss_frames = [i[1] for i in outs_tup if not i[1].empty]
                del outs_tup
                if len(miss_frames) > 1:
                    miss_frames = pd.concat(miss_frames)
                miss_frames_all.append(miss_frames)
                for d in outs:
                    subject_dict_all = dict(mergedicts(subject_dict_all, d))
                del outs, df_top, miss_frames
                gc.collect()
                shutil.rmtree(cache_dir, ignore_errors=True)
            del ses_name, grid, hyperparam_dict
            gc.collect()
        del alg, hyperparams
        gc.collect()
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
    mets=None,
    df_top=None,
):
    from colorama import Fore, Style
    from joblib import Parallel, delayed
    import gc

    # print(id)
    ID = id.split("_")[0].split("sub-")[1]
    ses = id.split("_")[1].split("ses-")[1]

    completion_status = f"{Fore.GREEN}✓{Style.RESET_ALL}"

    if ID not in subject_dict.keys():
        subject_dict[ID] = {}

    if ses not in subject_dict[ID].keys():
        subject_dict[ID][ses] = {}

    if modality not in subject_dict[ID][ses].keys():
        subject_dict[ID][ses][modality] = {}

    if alg not in subject_dict[ID][ses][modality].keys():
        subject_dict[ID][ses][modality][alg] = {}

    subject_dict[ID][ses][modality][alg] = dict.fromkeys(grid, np.nan)

    missingness_frame = pd.DataFrame(columns=["id", "ses", "modality", "alg",
                                              "grid"])

    # Functional case
    if modality == "func":
        with Parallel(
            n_jobs=4,
            require='sharedmem',
            verbose=1,
        ) as parallel:
            parallel(
                delayed(func_grabber)(comb, subject_dict, missingness_frame,
                                      completion_status,
                                      ID, ses, modality, alg, mets, thr_type,
                                      base_dir,
                                      template,
                                      df_top)
                for comb in grid
            )

    # Structural case
    elif modality == "dwi":
        with Parallel(
            n_jobs=4,
            require='sharedmem',
            verbose=1,
        ) as parallel:
            parallel(
                delayed(dwi_grabber)(comb, subject_dict, missingness_frame,
                                      completion_status,
                                      ID, ses, modality, alg, mets, thr_type,
                                      base_dir,
                                      template,
                                      df_top)
                for comb in grid
            )
    print(f"ID: {ID}, SESSION: {ses}, COMPLETENESS: {completion_status}")
    del modality, ID, ses, df_top
    gc.collect()
    return subject_dict, missingness_frame


def dwi_grabber(comb, subject_dict, missingness_frame, completion_status,
                 ID, ses, modality, alg, mets, thr_type, base_dir, template,
                 df_top):
    from pynets.core.utils import filter_cols_from_targets
    from colorama import Fore, Style
    import gc

    try:
        directget, minlength, model, res, atlas, tol = comb
    except BaseException:
        print(UserWarning(f"Failed to parse recipe: {comb}"))
        return subject_dict, missingness_frame, completion_status

    comb_tuple = (atlas, directget, minlength, model, res, tol)
    # print(comb_tuple)
    subject_dict[ID][ses][modality][alg][comb_tuple] = {}
    if alg == "ASE" or alg == "OMNI" or alg == "vectorize":
        embeddings = glob.glob(
            f"{base_dir}/embeddings_all"
            f"_{modality}/sub-{ID}/ses-{ses}/rsn-{atlas}_"
            f"res-{res}/"
            f"gradient*{alg}*{res}*{atlas}*{ID}"
            f"*modality-{modality}*model-{model}"
            f"*template-{template}*directget-"
            f"{directget}"
            f"*minlength-{minlength}*tol-{tol}*npy"
        )
        if len(embeddings) == 0 and 'RUM' in atlas:
            embeddings = glob.glob(
                f"{base_dir}/embeddings_all"
                f"_{modality}/sub-{ID}/ses-{ses}/rsn-{atlas.replace('RUM', '')}_"
                f"res-{res}/"
                f"gradient*{alg}*{res}*{atlas.replace('RUM', '')}*{ID}"
                f"*modality-{modality}*model-{model}"
                f"*template-{template}*directget-"
                f"{directget}"
                f"*minlength-{minlength}*tol-{tol}*npy"
            )
        if len(embeddings) == 0:
            print(
                f"No structural embeddings found for {ID} and"
                f" recipe {comb_tuple} & {alg}..."
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
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            return subject_dict, missingness_frame, completion_status
        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            embeddings_raw = [i for i in embeddings if "thrtype" not
                              in i]
            if len(embeddings_raw) == 1:
                embedding = embeddings[0]

            elif len(embeddings_raw) > 1:
                sorted_embeddings = sorted(embeddings_raw,
                                           key=os.path.getmtime)
                print(
                    f"Multiple functional embeddings found for {ID} and"
                    f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                    f" recent..."
                )
                embedding = sorted_embeddings[0]
            else:
                sorted_embeddings = sorted(embeddings,
                                           key=os.path.getmtime)
                print(
                    f"Multiple functional embeddings found for {ID} and"
                    f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                    f" recent..."
                )
                embedding = sorted_embeddings[0]

        if os.path.isfile(embedding):
            # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
            ixs = get_index_labels(base_dir, ID, ses, modality,
                                   comb_tuple,
                                   np.load(embedding).shape[0])
            if (
                alg
                not in subject_dict[ID][ses][modality][alg][
                comb_tuple].keys()
            ):
                subject_dict[ID][ses][modality][alg][comb_tuple] = {}
            subject_dict[ID][ses][modality][alg][comb_tuple]["index"] = ixs
            # subject_dict[ID][ses][modality][alg][comb_tuple]["labels"] = labels
            subject_dict[ID][ses][modality][alg][comb_tuple][
                "data"] = embedding
            # print(data)
        else:
            print(
                f"Structural embedding not found for {ID} and"
                f" recipe {comb_tuple} & {alg}..."
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
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            return subject_dict, missingness_frame, completion_status
    elif alg == "topology":
        data = np.empty([len(mets), 1], dtype=np.float32)
        data[:] = np.nan
        targets = [
            f"minlength-{minlength}",
            f"directget-{directget}",
            f"model-{model}",
            f"res-{res}",
            f"rsn-{atlas}",
            f"tol-{tol}",
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
                print(
                    f"Structural topology not found for {ID}, "
                    f"{met}, and recipe {comb_tuple}..."
                )
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
                completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
                return subject_dict, missingness_frame, completion_status

            out = df_top[df_top["id"] == "sub-" + ID + "_ses-" + ses][
                col
            ].values
            if len(out) == 0:
                print(
                    f"Structural topology not found for {ID}, "
                    f"{met}, and recipe {comb_tuple}..."
                )
                data[i] = np.nan
            else:
                data[i] = out

            del col, out
            i += 1
        if (np.abs(data) < 0.0000001).all():
            data[:] = np.nan
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
        subject_dict[ID][ses][modality][alg][comb_tuple] = data
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame, completion_status


def func_grabber(comb, subject_dict, missingness_frame, completion_status,
                 ID, ses, modality, alg, mets, thr_type, base_dir, template,
                 df_top):
    from pynets.core.utils import filter_cols_from_targets
    from colorama import Fore, Style
    import gc

    try:
        extract, hpass, model, res, atlas, smooth = comb
    except:
        try:
            extract, hpass, model, res, atlas = comb
            smooth = "0"
        except BaseException:
            print(UserWarning(f"Failed to parse recipe: {comb}"))
            return subject_dict, missingness_frame, completion_status

    comb_tuple = (atlas, extract, hpass, model, res, str(smooth))
    # print(comb_tuple)
    subject_dict[ID][ses][modality][alg][comb_tuple] = {}
    if alg == "ASE" or alg == "OMNI" or alg == "vectorize":
        if smooth == "0":
            embeddings = [
                i
                for i in glob.glob(
                    f"{base_dir}/embeddings_all_"
                    f"{modality}/sub-{ID}/ses-{ses}/rsn-"
                    f"{atlas}_res-{res}/"
                    f"gradient-{alg}_{res}_*{ID}"
                    f"*modality-{modality}*model-"
                    f"{model}*template-{template}*"
                    f"hpass-{hpass}Hz*extract-"
                    f"{extract}*npy"
                )
                if "smooth" not in i
            ]
        else:
            embeddings = [
                i
                for i in glob.glob(
                    f"{base_dir}/embeddings_all_"
                    f"{modality}/sub-{ID}/ses-{ses}/"
                    f"rsn-{atlas}"
                    f"_res-{res}/"
                    f"gradient*{alg}*{res}*"
                    f"*{ID}*modality-{modality}*model-"
                    f"{model}*template-{template}*"
                    f"hpass-{hpass}Hz*extract-"
                    f"{extract}*npy"
                )
                if f"smooth-{smooth}fwhm" in i
            ]
        if len(embeddings) == 0:
            if smooth == "0":
                embeddings = [
                    i
                    for i in glob.glob(
                        f"{base_dir}/embeddings_all_"
                        f"{modality}/sub-{ID}/ses-{ses}"
                        f"/rsn-"
                        f"{atlas}_res-{res}/"
                        f"gradient*{alg}*{res}*"
                        f"{atlas}*{ID}"
                        f"*modality-{modality}*model-"
                        f"{model}*template-{template}*"
                        f"hpass-{hpass}Hz*extract-"
                        f"{extract}*npy"
                    )
                    if "smooth" not in i
                ]
            else:
                embeddings = [
                    i
                    for i in glob.glob(
                        f"{base_dir}/embeddings_all_"
                        f"{modality}/sub-{ID}/ses-{ses}"
                        f"/rsn-{atlas}"
                        f"_res-{res}/"
                        f"gradient*{alg}*{res}*{atlas}"
                        f"*{ID}*modality-{modality}*model-"
                        f"{model}*template-{template}*"
                        f"hpass-{hpass}Hz*extract-"
                        f"{extract}*npy"
                    )
                    if f"smooth-{smooth}fwhm" in i
                ]
        if len(embeddings) == 0:
            print(
                f"No functional embeddings found for {ID} and"
                f" recipe {comb_tuple} & {alg}..."
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
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            return subject_dict, missingness_frame, completion_status

        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            embeddings_raw = [i for i in embeddings if "thrtype"
                              not in i]
            if len(embeddings_raw) == 1:
                embedding = embeddings[0]

            elif len(embeddings_raw) > 1:
                sorted_embeddings = sorted(embeddings_raw,
                                           key=os.path.getmtime)
                print(
                    f"Multiple functional embeddings found for {ID} and"
                    f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                    f" recent..."
                )
                embedding = sorted_embeddings[0]
            else:
                sorted_embeddings = sorted(embeddings, key=os.path.getmtime)
                print(
                    f"Multiple functional embeddings found for {ID} and"
                    f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                    f" recent..."
                )
                embedding = sorted_embeddings[0]

        if os.path.isfile(embedding):
            # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
            try:
                ixs = get_index_labels(base_dir, ID, ses, modality,
                                       comb_tuple, np.load(embedding).shape[0])
            except BaseException:
                print(f"Failed to load {embedding} for {ID}-{ses}")
                return subject_dict, missingness_frame, completion_status
            if (
                alg
                not in subject_dict[ID][ses][modality][alg][comb_tuple].keys()
            ):
                subject_dict[ID][ses][modality][alg][comb_tuple] = {}
            subject_dict[ID][ses][modality][alg][comb_tuple]["index"] = ixs
            # subject_dict[ID][ses][modality][alg][comb_tuple]["labels"] = labels
            subject_dict[ID][ses][modality][alg][comb_tuple][
                "data"] = embedding
            # print(data)
        else:
            print(
                f"Functional embedding not found for {ID} and"
                f" recipe {comb_tuple} & {alg}..."
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
            completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
            return subject_dict, missingness_frame, completion_status

    elif alg == "topology":
        data = np.empty([len(mets), 1], dtype=np.float32)
        data[:] = np.nan
        if smooth == '0':
            targets = [
                f"extract-{extract}",
                f"hpass-{hpass}Hz",
                f"model-{model}",
                f"res-{res}",
                f"rsn-{atlas}",
                f"thrtype-{thr_type}",
            ]
        else:
            targets = [
                f"extract-{extract}",
                f"hpass-{hpass}Hz",
                f"model-{model}",
                f"res-{res}",
                f"rsn-{atlas}",
                f"smooth-{smooth}fwhm",
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
                completion_status = f"{Fore.RED}X{Style.RESET_ALL}"
                return subject_dict, missingness_frame, completion_status

            out = df_top[df_top["id"] == f"sub-{ID}_ses-{ses}"][col].values
            if len(out) == 0:
                print(
                    f"Functional topology not found for {ID}, {met}, "
                    f"and recipe {comb_tuple}..."
                )
                data[i] = np.nan
            else:
                data[i] = out

            del col, out
            i += 1
        if (np.abs(data) < 0.0000001).all():
            data[:] = np.nan
        elif (np.abs(data) < 0.0000001).any():
            data[data < 0.0000001] = np.nan
        subject_dict[ID][ses][modality][alg][comb_tuple] = data
        # print(data)
    del comb, comb_tuple
    gc.collect()

    return subject_dict, missingness_frame, completion_status


def cleanNullTerms(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = cleanNullTerms(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif v is not None and v is not np.nan:
            clean[k] = v
    return clean


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
            # Remove incomplete cases
            df_all = df_all.loc[
                (df_all["id"] != "s057")
                & (df_all["id"] != "s054")
                & (df_all["id"] != "25667")
                & (df_all["id"] != "A00076381")
                & (df_all["id"] != "25853")
                ]
            if (
                all(
                    df_all.drop(
                        columns=[
                            "id",
                            "age",
                            "num_visits",
                            "sex"
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
                                    "age",
                                    "num_visits",
                                    "sex"
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

        return out_path, embedding_type, target_var
    else:
        return None, embedding_type, target_var


def copy_json_dict(feature_spaces, modality, embedding_type, tempdir):
    import uuid
    import os
    from time import strftime
    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    from nipype.utils.filemanip import fname_presuffix, copyfile
    import tempfile

    if feature_spaces is not None:
        input_dict_tmp = feature_spaces[
            f"{modality}_{embedding_type}"
        ]
        if os.path.isfile(input_dict_tmp) is True:
            json_dict = fname_presuffix(
                input_dict_tmp, suffix=f"_tmp_{run_uuid}",
                newpath=tempdir
            )

            copyfile(input_dict_tmp, json_dict, copy=True)

            return json_dict
        else:
            new_file, filename = tempfile.mkstemp()
            return new_file
    else:
        new_file, filename = tempfile.mkstemp()
        return new_file


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
            if os.path.isfile(self.inputs.json_dict) and self.inputs.json_dict.endswith('.json'):
                if self.inputs.target_var == "rumination_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype", "dep_1",
                                 "rum_1", "rum_2", "dep_2"]
                elif self.inputs.target_var == "depression_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "rumination_persist_phenotype",
                                 "dep_1", "rum_1", "dep_2", "rum_2"]
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
                                 "dep_2", "dep_1"]
                else:
                    drop_cols = [self.inputs.target_var,
                                 "rumination_persist_phenotype",
                                 "depression_persist_phenotype",
                                 "dep_1", "rum_1"]

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


class BSNestedCV(SimpleInterface):
    """Interface wrapper for BSNestedCV"""

    input_spec = _BSNestedCVInputSpec
    output_spec = _BSNestedCVOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        from colorama import Fore, Style

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
                    ] = bootstrapped_nested_cv(self.inputs.X, self.inputs.y,
                                               predict_type=predict_type)
                else:
                    [
                        grand_mean_best_estimator,
                        grand_mean_best_score,
                        grand_mean_best_error,
                        mega_feat_imp_dict,
                        grand_mean_y_predicted,
                    ] = bootstrapped_nested_cv(self.inputs.X, self.inputs.y,
                                               predict_type=predict_type,
                                               missingness_thr=0.20,
                                               std_dev=3)
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
                    self._results[
                        "grand_mean_best_estimator"] = grand_mean_best_estimator
                    self._results[
                        "grand_mean_best_score"] = grand_mean_best_score
                    self._results[
                        "grand_mean_y_predicted"] = grand_mean_y_predicted
                    self._results[
                        "grand_mean_best_error"] = grand_mean_best_error
                    self._results["mega_feat_imp_dict"] = mega_feat_imp_dict
                else:
                    print(f"{Fore.RED}Empty feature-space for "
                          f"{self.inputs.grid_param}, "
                          f"{self.inputs.target_var}, "
                          f"{self.inputs.embedding_type}, "
                          f"{self.inputs.modality}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}{self.inputs.X} is not pd.DataFrame for"
                      f" {self.inputs.grid_param}, {self.inputs.target_var},"
                      f" {self.inputs.embedding_type}, "
                      f"{self.inputs.modality}{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.RED}Empty feature-space for {self.inputs.grid_param},"
                f" {self.inputs.target_var}, {self.inputs.embedding_type},"
                f" {self.inputs.modality}{Style.RESET_ALL}")

        gc.collect()

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
            print('%.1f confidence interval %.1f%% and %.1f%%' % (
                alpha * 100, lower * 100, upper * 100))
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


def create_wf(modality_grids, modality, basedir):
    from pynets.stats.prediction import MakeXY, MakeDF, BSNestedCV, \
        concatenate_frames
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    import uuid
    from time import strftime

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_wf = pe.Workflow(name=f"ensemble_connectometry_{run_uuid}")
    os.makedirs(f"/{basedir}/{run_uuid}", exist_ok=True)
    ml_wf.base_dir = f"/{basedir}/{run_uuid}"
    os.makedirs(f"{ml_wf.base_dir}/json_tmps", exist_ok=True)

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
            grid_params_mod.append([atlas, extract, hpass, model, res,
                                    str(smooth)])
    elif modality == "dwi":
        for comb in grid_param_combos:
            try:
                directget, minlength, model, res, atlas, tol = comb
            except:
                raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append([atlas, directget, minlength, model, res,
                                    tol])

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "feature_spaces",
                "out_dir",
                "modality",
                "target_var",
                "embedding_type",
            ]
        ),
        name="inputnode",
    )

    copy_json_dict_node = pe.Node(
        niu.Function(
            input_names=["feature_spaces", "modality", "embedding_type",
                         "tempdir"],
            output_names=["json_dict"],
            function=copy_json_dict,
        ),
        name="copy_json_dict_node",
    )
    copy_json_dict_node.inputs.tempdir = f"{ml_wf.base_dir}/json_tmps"

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
            output_names=["out_path", "embedding_type", "target_var"],
            function=concatenate_frames,
        ),
        name="concatenate_frames_node",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["target_var", "df_summary",
                                      "embedding_type"]), name="outputnode"
    )

    ml_wf.connect(
        [
            (
                inputnode,
                copy_json_dict_node,
                [
                    ("feature_spaces", "feature_spaces"),
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type")
                ],
            ),
            (
                inputnode,
                make_x_y_func_node,
                [
                    ("target_var", "target_var"),
                    ("modality", "modality"),
                    ("embedding_type", "embedding_type")
                ],
            ),
            (
                copy_json_dict_node,
                make_x_y_func_node,
                [
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
                [("out_dir", "out_dir"),
                 ("modality", "modality"), ("embedding_type", "embedding_type"),
                 ("target_var", "target_var")],
            ),
            (concatenate_frames_node, outputnode,
             [("out_path", "df_summary"), ("embedding_type", "embedding_type"),
              ("target_var", "target_var")]),
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
    embedding_types = args["embedding_types"]
    modality = args["modality"]

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_meta_wf = pe.Workflow(name="pynets_multipredict")
    ml_meta_wf.base_dir = f"{base_dir}/pynets_multiperform_{run_uuid}"

    os.makedirs(ml_meta_wf.base_dir, exist_ok=True)

    meta_inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "feature_spaces",
                "base_dir",
                "modality"
            ]
        ),
        name="meta_inputnode",
    )

    meta_inputnode.inputs.base_dir = base_dir
    meta_inputnode.inputs.feature_spaces = feature_spaces
    meta_inputnode.inputs.modality = modality
    meta_inputnode.inputs.modality_grids = modality_grids

    meta_iter_info_node = pe.Node(
        niu.IdentityInterface(
            fields=[
                "target_var",
                "embedding_type",
            ]
        ),
        name="meta_iter_info_node", nested=True
    )

    # Set up as iterables
    vars_embeddings_iters = list(
        itertools.product(target_vars, embedding_types)
    )
    target_vars_list = [i[0] for i in vars_embeddings_iters]
    embedding_types_list = [i[1] for i in vars_embeddings_iters]

    vars_embeddings_iters_list = []
    vars_embeddings_iters_list.append(("target_var", target_vars_list))
    vars_embeddings_iters_list.append(("embedding_type", embedding_types_list))
    meta_iter_info_node.iterables = vars_embeddings_iters_list

    create_wf_node = create_wf(modality_grids, modality, ml_meta_wf.base_dir)

    final_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary", "embedding_type",
                                      "target_var"]),
        name="final_join_node",
        joinfield=["df_summary", "embedding_type", "target_var"],
        joinsource=meta_iter_info_node,
    )

    meta_outputnode = pe.Node(
        niu.IdentityInterface(fields=["df_summary"]), name="meta_outputnode"
    )

    ml_meta_wf.connect(
        [
            (
                meta_inputnode,
                create_wf_node,
                [
                    ("base_dir", "inputnode.out_dir"),
                    ("feature_spaces", "inputnode.feature_spaces"),
                    ("modality", "inputnode.modality"),
                ],
            ),
            (
                meta_iter_info_node,
                create_wf_node,
                [
                    ("target_var", "inputnode.target_var"),
                    ("embedding_type", "inputnode.embedding_type")
                ],
            ),
            (
                create_wf_node,
                final_join_node,
                [
                    ("outputnode.df_summary", "df_summary"),
                    ("outputnode.target_var", "target_var"),
                    ("outputnode.embedding_type", "embedding_type"),
                ],
            ),
            (final_join_node, meta_outputnode,
             [("df_summary", "df_summary")]),
        ]
    )
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_meta_wf.base_dir)
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


def main():
    import json

    #base_dir = "/working/tuning_set/outputs_shaeffer/func_ml"
    #base_dir = "/working/tuning_set/outputs_clustering"
    base_dir = "/working/tuning_set/outputs_language"
    #base_dir = "/working/tuning_set/outputs_clustering/func_ml"
    df = pd.read_csv(
        "/working/tuning_set/outputs_shaeffer/df_rum_persist_all.csv",
        index_col=False
    )

    # Hard-Coded #
    embedding_types = ['ASE']
    #embedding_types = ['OMNI', 'ASE']
    #modalities = ["dwi", "func"]
    modalities = ["dwi"]
    thr_type = "MST"
    template = "MNI152_T1"
    mets = [
        "global_efficiency",
        "average_shortest_path_length",
        "degree_assortativity_coefficient",
        "average_eigenvector_centrality",
        "average_betweenness_centrality",
        "modularity",
        "smallworldness",
    ]
    hyperparams_func = ["rsn", "res", "model", "hpass", "extract", "smooth"]
    hyperparams_dwi = ["rsn", "res", "model", "directget", "minlength", "tol"]

    # User-Specified #
    target_modality = 'dwi'
    #target_modality = 'func'
    target_vars = ["rumination_persist_phenotype",
                   "depression_persist_phenotype",
                   "dep_2", 'rum_2', 'rum_1', 'dep_1']

    #rsns = ["triple", "kmeans"]
    #rsns = ["tripleRUM", "kmeansRUM", "language"]
    rsns = ["language"]

    sessions = ["1"]

    subject_dict_file_path = (
        f"{base_dir}/pynets_subject_dict_{target_modality}_{'_'.join(embedding_types)}.pkl"
    )
    subject_mod_grids_file_path = (
        f"{base_dir}/pynets_modality_grids_{target_modality}_{'_'.join(embedding_types)}.pkl"
    )
    missingness_summary = (
        f"{base_dir}/pynets_missingness_summary_{target_modality}_{'_'.join(embedding_types)}.csv"
    )

    if not os.path.isfile(subject_dict_file_path) or not os.path.isfile(
        subject_mod_grids_file_path
    ):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            modalities, base_dir, thr_type, mets, embedding_types, template,
            sessions, rsns
        )
        sub_dict_clean = cleanNullTerms(subject_dict)
        missingness_frames = [i for i in missingness_frames if
                              isinstance(i, pd.DataFrame)]
        if len(missingness_frames) != 0:
            if len(missingness_frames) > 1:
                final_missingness_summary = pd.concat(missingness_frames)
            elif len(missingness_frames) == 1:
                final_missingness_summary = missingness_frames[0]
            final_missingness_summary.to_csv(missingness_summary, index=False)
        with open(subject_dict_file_path, "wb") as f:
            dill.dump(sub_dict_clean, f)
        f.close()
        with open(subject_mod_grids_file_path, "wb") as f:
            dill.dump(modality_grids, f)
        f.close()
    else:
        with open(subject_dict_file_path, "rb") as f:
            sub_dict_clean = dill.load(f)
        f.close()
        with open(subject_mod_grids_file_path, "rb") as f:
            modality_grids = dill.load(f)
        f.close()

    # Subset only those participants which have usable data
    for ID in df["participant_id"]:
        if len(ID) == 1:
            df.loc[df.participant_id == ID, "participant_id"] = "s00" + str(ID)
        if len(ID) == 2:
            df.loc[df.participant_id == ID, "participant_id"] = "s0" + str(ID)

    df = df[df["participant_id"].isin(list(sub_dict_clean.keys()))]
    df['sex'] = df['sex'].map({1:0, 2:1})
    df = df[
        ["participant_id", "age", "num_visits", "sex"] + target_vars
    ]

    ml_dfs_dict = {}
    for modality in modalities:
        ml_dfs_dict[modality] = {}
        for alg in embedding_types:
            dict_file_path = f"{base_dir}/pynets_ml_dict_{modality}_{alg}.pkl"
            if not os.path.isfile(dict_file_path) or not \
                os.path.isfile(dict_file_path):
                ml_dfs = {}
                ml_dfs = make_feature_space_dict(
                    ml_dfs,
                    df,
                    modality,
                    sub_dict_clean,
                    sessions[0],
                    modality_grids,
                    alg,
                    mets
                )

                with open(dict_file_path, "wb") as f:
                    dill.dump(ml_dfs, f)
                f.close()
                ml_dfs_dict[modality][alg] = dict_file_path
                del ml_dfs
            else:
                ml_dfs_dict[modality][alg] = dict_file_path

    outs = []
    for modality in modalities:
        for alg in embedding_types:
            with open(ml_dfs_dict[modality][alg], "rb") as f:
                outs.append(dill.load(f))
            f.close()

    ml_dfs = outs[0]
    for d in outs:
        ml_dfs = dict(mergedicts(ml_dfs, d))

    # with open('pynets_ml_dict_func_topology.pkl', "rb") as f:
    #     ml_dfs = dill.load(f)
    # f.close()

    tables = list(itertools.product(modalities, embedding_types))

    feature_spaces = {}
    for comb in tables:
        modality = comb[0]
        alg = comb[1]
        iter = f"{modality}_{alg}"
        out_dict = {}
        for recipe in ml_dfs[modality][alg].keys():
            try:
                out_dict[str(recipe)] = ml_dfs[modality][alg][recipe].to_json()
            except:
                print(f"{recipe} recipe not found...")
                continue
        out_json_path = f"{base_dir}/{iter}.json"
        if os.path.isfile(out_json_path):
            os.remove(out_json_path)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)
        f.close()
        feature_spaces[iter] = out_json_path

    del ml_dfs

    args = {}
    args["base_dir"] = base_dir
    args["feature_spaces"] = feature_spaces
    args["modality_grids"] = modality_grids
    args["target_vars"] = target_vars
    args["embedding_types"] = embedding_types
    args["modality"] = target_modality

    return args


if __name__ == "__main__":
    import warnings
    import sys
    import gc
    import json
    from multiprocessing import set_start_method, Process, Manager
    from pynets.stats.prediction import build_predict_workflow

    try:
        set_start_method("forkserver")
    except:
        pass
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = main()

    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_predict_workflow, args=(args, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        gc.collect()
    mgr.shutdown()
