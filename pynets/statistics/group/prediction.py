#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017
"""
import matplotlib
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
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, \
    cross_validate, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition
from collections import OrderedDict
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from pynets.core.utils import flatten
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from pynets.statistics.utils import bias_variance_decomp, \
    split_df_to_dfs_by_prefix, make_param_grids, preprocess_x_y

try:
    from sklearn.utils._testing import ignore_warnings
except:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

matplotlib.use('Agg')
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
    "from sklearn.base import BaseEstimator, TransformerMixin",
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
    "from sklearn.utils import check_random_state"
]


class Razors(object):
    """
    Razors is a callable refit option for `GridSearchCV` whose aim is to
    balance model complexity and cross-validated score in the spirit of the
    "one standard error" rule of Breiman et al. (1984), which showed that
    the tuning hyperparameter associated with the best performing model may be
    prone to overfit. To help mitigate this risk, we can instead instruct
    gridsearch to refit the highest performing 'parsimonious' model, as defined
    using simple statistical rules (e.g. standard error (`sigma`),
    percentile (`eta`), or significance level (`alpha`)) to compare
    distributions of model performance across folds. Importantly, this
    strategy assumes that the grid of multiple cross-validated models
    can be principly ordered from simplest to most complex with respect to some
    target hyperparameter of interest. To use the razors suite, supply
    the `simplify` function partial of the `Razors` class as a callable
    directly to the `refit` argument of `GridSearchCV`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.
    scoring : str
        Refit scoring metric.
    param : str
        Parameter whose complexity will be optimized.
    rule : str
        Rule for balancing model complexity with performance.
        Options are 'se', 'percentile', and 'ranksum'. Default is 'se'.
    sigma : int
        Number of standard errors tolerance in the case that a standard error
        threshold is used to filter outlying scores across folds. Required if
        `rule`=='se'. Default is 1.
    eta : float
        Percentile tolerance in the case that a percentile threshold
        is used to filter outlier scores across folds. Required if
        `rule`=='percentile'. Default is 0.68.
    alpha : float
        An alpha significance level in the case that wilcoxon rank sum
        hypothesis testing is used to filter outlying scores across folds.
        Required if `rule`=='ranksum'. Default is 0.05.

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

    The callable API accordingly assumes that the `params` attribute of
    `cv_results_` 1) contains the indicated hyperparameter (`param`) of
    interest, and 2) contains a sequence of values (numeric, boolean, or
    categorical) that are ordered from least to most complex.
    """
    __slots__ = ('cv_results', 'param', 'param_complexity', 'scoring',
                 'rule', 'greater_is_better',
                 '_scoring_funcs', '_scoring_dict',
                 '_n_folds', '_splits', '_score_grid',
                 '_cv_means', '_sigma', '_eta', '_alpha')

    def __init__(
            self,
            cv_results_,
            param,
            scoring,
            rule,
            sigma=1,
            eta=0.95,
            alpha=0.01,
    ):
        import sklearn.metrics

        self.cv_results = cv_results_
        self.param = param
        self.scoring = scoring
        self.rule = rule
        self._scoring_funcs = [
            met
            for met in sklearn.metrics.__all__
            if (met.endswith("_score")) or (met.endswith("_error"))
        ]
        # Set _score metrics to True and _error metrics to False
        self._scoring_dict = dict(
            zip(
                self._scoring_funcs,
                [met.endswith("_score") for met in self._scoring_funcs],
            )
        )
        self.greater_is_better = self._check_scorer()
        self._n_folds = len(list(set([i.split('_')[0] for i in
                                     list(self.cv_results.keys()) if
                                     i.startswith('split')])))
        # Extract subgrid corresponding to the scoring metric of interest
        self._splits = [i for i in list(self.cv_results.keys()) if
                        i.endswith(f"test_{self.scoring}") and
                        i.startswith('split')]
        self._score_grid = np.vstack([self.cv_results[cv] for cv in
                                      self._splits]).T
        self._cv_means = np.array(np.nanmean(self._score_grid, axis=1))
        self._sigma = sigma
        self._eta = eta
        self._alpha = alpha

    def _check_scorer(self):
        """
        Check whether the target refit scorer is negated. If so, adjust
        greater_is_better accordingly.
        """

        if (
                self.scoring not in self._scoring_dict.keys()
                and f"{self.scoring}_score" not in self._scoring_dict.keys()
        ):
            if self.scoring.startswith("neg_"):
                self.greater_is_better = True
            else:
                raise NotImplementedError(f"Scoring metric {self.scoring} not "
                                          f"recognized.")
        else:
            self.greater_is_better = [
                value for key, value in self._scoring_dict.items() if
                self.scoring in key][0]
        return self.greater_is_better

    def _best_low_complexity(self):
        """
        Balance model complexity with cross-validated score.

        Return
        ------
        int
            Index of a model that has the lowest complexity but its test score
            is the highest on average across folds as compared to other models
            that are equally likely to occur.
        """

        # Check parameter(s) whose complexity we seek to restrict
        if not any(self.param in x for x in
                   self.cv_results["params"][0].keys()):
            raise KeyError(f"Parameter {self.param} not found in cv grid.")
        else:
            hyperparam = [
                i for i in self.cv_results["params"][0].keys() if
                i.endswith(self.param)][0]

        # Select low complexity threshold based on specified evaluation rule
        if self.rule == "se":
            if not self._sigma:
                raise ValueError(
                    "For `se` rule, the tolerance "
                    "(i.e. `_sigma`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_standard_error()
        elif self.rule == "percentile":
            if not self._eta:
                raise ValueError(
                    "For `percentile` rule, the tolerance "
                    "(i.e. `_eta`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_percentile()
        elif self.rule == "ranksum":
            if not self._alpha:
                raise ValueError(
                    "For `ranksum` rule, the alpha-level "
                    "(i.e. `_alpha`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_rank_sum_test()
        else:
            raise NotImplementedError(f"{self.rule} is not a valid "
                                      f"rule of RazorCV.")

        self.cv_results[f"param_{hyperparam}"].mask = np.where(
            (self._cv_means >= float(l_cutoff)) &
            (self._cv_means <= float(h_cutoff)),
            True, False)

        if np.sum(self.cv_results[f"param_{hyperparam}"].mask) == 0:
            print(f"\nLow: {l_cutoff}")
            print(f"High: {h_cutoff}")
            print(f"{self._cv_means}")
            print(f"hyperparam: {hyperparam}\n")
            raise ValueError("No valid grid columns remain within the "
                             "boundaries of the specified razor")

        highest_surviving_rank = np.nanmin(
            self.cv_results[f"rank_test_{self.scoring}"][
                self.cv_results[f"param_{hyperparam}"].mask])

        # print(f"Highest surviving rank: {highest_surviving_rank}\n")

        return np.flatnonzero(np.isin(
            self.cv_results[f"rank_test_{self.scoring}"],
            highest_surviving_rank))[0]

    def call_standard_error(self):
        """
        Returns the simplest model whose performance is within `sigma`
        standard errors of the average highest performing model.
        """

        # Estimate the standard error across folds for each column of the grid
        cv_se = np.array(np.nanstd(self._score_grid, axis=1) /
                         np.sqrt(self._n_folds))

        # Determine confidence interval
        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
            h_cutoff = self._cv_means[best_score_idx] + cv_se[best_score_idx]
            l_cutoff = self._cv_means[best_score_idx] - cv_se[best_score_idx]
        else:
            best_score_idx = np.nanargmin(self._cv_means)
            h_cutoff = self._cv_means[best_score_idx] - cv_se[best_score_idx]
            l_cutoff = self._cv_means[best_score_idx] + cv_se[best_score_idx]

        return l_cutoff, h_cutoff

    def call_rank_sum_test(self):
        """
        Returns the simplest model whose paired performance across folds is
        insignificantly different from the average highest performing,
        at a predefined `alpha` level of significance.
        """

        from scipy.stats import wilcoxon
        import itertools

        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
        else:
            best_score_idx = np.nanargmin(self._cv_means)

        # Perform signed Wilcoxon rank sum test for each pair combination of
        # columns against the best average score column
        tests = [pair for pair in list(itertools.combinations(range(
            self._score_grid.shape[0]), 2)) if best_score_idx in pair]

        p_dict = {}
        for i, test in enumerate(tests):
            p_dict[i] = wilcoxon(self._score_grid[test[0], :],
                                 self._score_grid[test[1], :])[1]

        # Sort and prune away significant tests
        p_dict = {k: v for k, v in sorted(p_dict.items(),
                                          key=lambda item: item[1]) if
                  v > self._alpha}

        # Flatten list of tuples, remove best score index, and take the
        # lowest and highest remaining bounds
        tests = [j for j in list(set(list(sum([tests[i] for i in
                                               list(p_dict.keys())],
                                              ())))) if j != best_score_idx]
        if self.greater_is_better:
            h_cutoff = self._cv_means[
                np.nanargmin(self.cv_results[
                                 f"rank_test_{self.scoring}"][tests])]
            l_cutoff = self._cv_means[
                np.nanargmax(self.cv_results[
                                 f"rank_test_{self.scoring}"][tests])]
        else:
            h_cutoff = self._cv_means[
                np.nanargmax(self.cv_results[
                                 f"rank_test_{self.scoring}"][tests])]
            l_cutoff = self._cv_means[
                np.nanargmin(self.cv_results[
                                 f"rank_test_{self.scoring}"][tests])]

        return l_cutoff, h_cutoff


    def call_percentile(self):
        """
        Returns the simplest model whose performance is within the `eta`
        percentile of the average highest performing model.
        """

        # Estimate the indicated percentile, and its inverse, across folds for
        # each column of the grid
        perc_cutoff = np.nanpercentile(self._score_grid,
                                       [100 * self._eta,
                                        100 - 100 * self._eta], axis=1)

        # Determine bounds of the percentile interval
        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
            h_cutoff = perc_cutoff[0, best_score_idx]
            l_cutoff = perc_cutoff[1, best_score_idx]
        else:
            best_score_idx = np.nanargmin(self._cv_means)
            h_cutoff = perc_cutoff[0, best_score_idx]
            l_cutoff = perc_cutoff[1, best_score_idx]

        return l_cutoff, h_cutoff

    @staticmethod
    def simplify(param, scoring, rule='se', sigma=1, eta=0.68, alpha=0.01):
        """
        Callable to be run as `refit` argument of `GridsearchCV`.

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        scoring : str
            Refit scoring metric.
        sigma : int
            Number of standard errors tolerance in the case that a standard
            error threshold is used to filter outlying scores across folds.
            Only applicable if `rule`=='se'. Default is 1.
        eta : float
            Acceptable percent tolerance in the case that a percentile
            threshold is used. Only applicable if `rule`=='percentile'.
            Default is 0.68.
        alpha : float
            Alpha-level to use for signed wilcoxon rank sum testing.
            Only applicable if `rule`=='ranksum'. Default is 0.01.
        """
        from functools import partial

        def razor_pass(
                cv_results_, param, scoring, rule, sigma, alpha, eta
        ):
            rcv = Razors(cv_results_, param, scoring, rule=rule,
                         sigma=sigma, alpha=alpha, eta=eta)
            return rcv._best_low_complexity()

        return partial(
            razor_pass,
            param=param,
            scoring=scoring,
            rule=rule,
            sigma=sigma,
            alpha=alpha,
            eta=eta,
        )


@ignore_warnings(category=ConvergenceWarning)
def nested_fit(X, y, estimators, boot, pca_reduce, k_folds,
               predict_type, search_method='grid', razor=False, n_jobs=1):

    # Instantiate an inner-fold
    if predict_type == 'regressor':
        inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=boot)
    elif predict_type == 'classifier':
        inner_cv = StratifiedKFold(n_splits=k_folds, shuffle=True,
                                   random_state=boot)
    else:
        raise ValueError('Prediction method not recognized')

    if predict_type == 'regressor':
        scoring = ["explained_variance", "neg_mean_squared_error"]
        refit_score = "explained_variance"
        feature_selector = f_regression
    elif predict_type == 'classifier':
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
                refit = Razors.simplify(param='n_components',
                                        scoring=refit_score,
                                        rule='se', sigma=1)
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
                    refit = Razors.simplify(param='k',
                                            scoring=refit_score,
                                            rule='se', sigma=1)
                else:
                    refit = refit_score

        ## Model-specific naming chunk 1
        # Make hyperparameter search-spaces
        param_space = make_param_grids()

        if predict_type == 'classifier':
            if 'svm' in estimator_name or 'en' in estimator_name or 'l1' in \
                estimator_name or 'l2' in estimator_name:
                param_grid[estimator_name + "__C"] = param_space['Cs']
        elif predict_type == 'regressor':
                param_grid[estimator_name + "__alpha"] = param_space['alphas']
        else:
            raise ValueError('Prediction method not recognized')

        if 'en' in estimator_name:
            param_grid[estimator_name + "__l1_ratio"] = \
                param_space['l1_ratios']
        ## Model-specific naming chunk 1

        # Establish grid-search feature/model tuning windows,
        # refit the best model using a 1 SE rule of MSE values.

        if search_method == 'grid':
            pipe_grid_cv = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=n_jobs,
                cv=inner_cv
            )
        elif search_method == 'random':
            pipe_grid_cv = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=n_jobs,
                cv=inner_cv
            )
        else:
            raise ValueError(f"Search method {search_method} not "
                             f"recognized...")

        # Fit pipeline to data
        pipe_grid_cv.fit(X, y.values.ravel())

        # Grab mean
        means_exp_var = pipe_grid_cv.cv_results_[f"mean_test_{refit_score}"]
        means_MSE = pipe_grid_cv.cv_results_[
            f"mean_test_neg_mean_squared_error"]

        hyperparam_space = ''
        ## Model-specific naming chunk 2
        if predict_type == 'classifier':
            if 'svm' in estimator_name or 'en' in estimator_name or 'l1' in \
                estimator_name or 'l2' in estimator_name:
                c_best = pipe_grid_cv.best_estimator_.get_params()[
                    f"{estimator_name}__C"]
                hyperparam_space = \
                    f"C-{c_best}"
        elif predict_type == 'regressor':
            alpha_best = pipe_grid_cv.best_estimator_.get_params()[
                 f"{estimator_name}__alpha"]
            hyperparam_space = f"_alpha-{alpha_best}"
        else:
            raise ValueError('Prediction method not recognized')

        if 'en' in estimator_name:
            best_l1 = pipe_grid_cv.best_estimator_.get_params()[
                 f"{estimator_name}__l1_ratio"]
            hyperparam_space = hyperparam_space + f"_l1ratio-{best_l1}"
        ## Model-specific naming chunk 2

        if pca_reduce is True and X.shape[0] < X.shape[1]:
            best_k = pipe_grid_cv.best_estimator_.named_steps[
                'feature_select'].n_components
            best_estimator_name = f"{predict_type}-{estimator_name}" \
                                  f"_{hyperparam_space}_nfeatures-" \
                                  f"{best_k}"
        else:
            if X.shape[1] <= min(k):
                best_estimator_name = \
                    f"{predict_type}-{estimator_name}_{hyperparam_space}"
            else:
                best_k = pipe_grid_cv.best_estimator_.named_steps[
                    'feature_select'].k
                best_estimator_name = \
                    f"{predict_type}-{estimator_name}_{hyperparam_space}" \
                    f"_nfeatures-{best_k}"
        # print(best_estimator_name)
        means_all_exp_var[best_estimator_name] = np.nanmean(means_exp_var)
        means_all_MSE[best_estimator_name] = np.nanmean(means_MSE)

    # Get best estimator across models
    best_estimator = max(means_all_exp_var, key=means_all_exp_var.get)
    est = estimators[best_estimator.split(f"{predict_type}-")[1].split('_')[0]]

    ## Model-specific naming chunk 3
    if 'en' in best_estimator:
        est.l1_ratio = float(best_estimator.split("l1ratio-")[1].split('_')[0])

    if predict_type == 'classifier':
        if 'svm' in best_estimator or 'en' in best_estimator or 'l1' in \
            best_estimator or 'l2' in best_estimator:
            est.C = float(best_estimator.split("C-")[1].split('_')[0])
    elif predict_type == 'regressor':
        est.alpha = float(best_estimator.split("alpha-")[1].split('_')[0])
    else:
        raise ValueError('Prediction method not recognized')

    ## Model-specific naming chunk 3
    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = decomposition.PCA(
            n_components=int(best_estimator.split("nfeatures-")[1].split(
                '_')[0]),
            whiten=True
        )
        reg = Pipeline([("feature_select", pca),
                        (best_estimator.split(f"{predict_type}-")[1].split(
                            '_')[0], est)])
    else:
        if X.shape[1] <= min(k):
            reg = Pipeline([(best_estimator.split(
                f"{predict_type}-")[1].split('_')[0], est)])
        else:
            kbest = SelectKBest(feature_selector,
                                k=int(best_estimator.split(
                                    "nfeatures-")[1].split('_')[0]))
            reg = Pipeline(
                [("feature_select", kbest),
                 (best_estimator.split(f"{predict_type}-")[1].split(
                     '_')[0], est)]
            )

    return reg, best_estimator


class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def fit(self, X, y=None):
        from sklearn.utils.validation import check_X_y
        from sklearn.utils.multiclass import unique_labels
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        X_out, y = check_X_y(X, y)

        X = X if hasattr(X, 'columns') else X_out

        if hasattr(X, 'columns'):
            self.expected_ = list(X.columns)
            self.expected_n_ = X.shape[1]
        else:
            self.expected_ = None
            self.expected_n_ = X.shape[1]
            warnings.warn('Input does not have a columns attribute, '
                          'only number of columns will be validated')

        self.classes_ = unique_labels(y)

        self.ensemble_fitted = self.ensemble.fit(X, y)

        return self

    def predict(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = check_array(X)

        return self.ensemble_fitted.predict(X)


def build_stacked_ensemble(X, y, base_estimators, boot, pca_reduce,
                           k_folds_inner, predict_type='classifier',
                           search_method='grid', prefixes=[],
                           stacked_folds=10, voting=True):
    from sklearn import ensemble
    from sklearn import metrics
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    if predict_type == 'regressor':
        scoring = ["explained_variance", "neg_mean_squared_error"]
        refit_score = "explained_variance"
    elif predict_type == 'classifier':
        scoring = ["f1", "neg_mean_squared_error"]
        refit_score = "f1"

    ens_estimators = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=y)

    feature_subspaces = split_df_to_dfs_by_prefix(X_train, prefixes=prefixes)

    if voting is True:
        layer_ests = []
        layer_weights = []
        for X_subspace in feature_subspaces:
            final_est, best_estimator = nested_fit(
                X_subspace, y_train, base_estimators, boot, pca_reduce,
                k_folds_inner, predict_type,
                search_method=search_method
            )

            final_est.steps.insert(0, ('selector', ColumnTransformer([
                    ("selector", "passthrough", list(X_subspace.columns))
                ], remainder="drop")))
            layer_ests.append((list(set([i.split('_')[0] for i in
                                         X_subspace.columns]))[0], final_est))
            prediction = cross_validate(
                final_est,
                X_train,
                y_train,
                cv=k_folds_inner,
                scoring='f1',
                return_estimator=False,
            )
            layer_weights.append(np.nanmean(prediction['test_score']))

        norm_weights = [float(i) / sum(layer_weights) for i in layer_weights]
        ec = ensemble.VotingClassifier(estimators=layer_ests,
                                       voting='soft',
                                       weights=np.array(norm_weights))

        ensemble_fitted = ec.fit(X_train, y_train)
        meta_est_name = 'voting'
        ens_estimators[meta_est_name] = {}
        ens_estimators[meta_est_name]['model'] = ensemble_fitted
        ens_estimators[meta_est_name]['error'] = metrics.brier_score_loss(
            ensemble_fitted.predict(X_test), y_test)
        ens_estimators[meta_est_name]['score'] = \
            metrics.balanced_accuracy_score(
            ensemble_fitted.predict(X_test), y_test)
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

        if predict_type == 'classifier':
            reg = GaussianNB()
            meta_name = 'nb'
        elif predict_type == 'regressor':
            reg = LinearRegression(normalize=True)
            meta_name = 'lr'
        else:
            raise ValueError('predict_type not recognized!')

        meta_estimators = {
            meta_name: reg,
        }

        for meta_est_name, meta_est in meta_estimators.items():
            layer_ests = []
            for X_subspace in feature_subspaces:
                final_est, best_estimator = nested_fit(
                    X_subspace, y_train, base_estimators, boot, pca_reduce,
                    k_folds_inner, predict_type, search_method=search_method
                )

                final_est.steps.insert(0, ('selector', ColumnTransformer([
                        ("selector", "passthrough", list(X_subspace.columns))
                    ], remainder="drop")))
                layer_ests.append((list(set([i.split('_')[0] for i in
                                             X_subspace.columns]))[0],
                                   final_est))

            ec = ensemble.StackingClassifier(estimators=layer_ests,
                                             final_estimator=meta_est,
                                             passthrough=False,
                                             cv=stacked_folds)

            ensemble_fitted = ec.fit(X_train, y_train)
            ens_estimators[meta_est_name] = {}
            ens_estimators[meta_est_name]['model'] = ensemble_fitted
            ens_estimators[meta_est_name]['error'] = metrics.brier_score_loss(
                ensemble_fitted.predict(X_test), y_test)
            ens_estimators[meta_est_name]['score'] = \
                metrics.balanced_accuracy_score(
                ensemble_fitted.predict(X_test), y_test)

    # Select best SuperLearner
    best_estimator = min(ens_estimators,
                         key=lambda v: ens_estimators[v]['error'])
    final_est = Pipeline([(best_estimator,
                           ens_estimators[best_estimator]['model'])])
    return final_est, best_estimator


def get_feature_imp(X, pca_reduce, fitted, best_estimator, predict_type,
                    dummy_run, stack):
    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = fitted.named_steps["feature_select"]
        comps_all = pd.DataFrame(pca.components_, columns=X.columns)

        n_pcs = pca.components_.shape[0]

        best_positions = list(flatten([
            np.nanargmax(np.abs(pca.components_[i])) for i in
            range(n_pcs)
        ]))

        if dummy_run is True:
            coefs = list(flatten(np.abs(
                np.ones(len(best_positions))).tolist()))
        else:
            coefs = list(flatten(np.abs(fitted.named_steps[
                                            best_estimator.split(
                                                f"{predict_type}-")[1
                                            ].split('_')[
                                                0]].coef_).tolist()))
        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(comps_all, coefs)).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )

        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(best_positions,
                         feat_imp_dict.values())).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )
    else:
        if 'feature_select' not in list(fitted.named_steps.keys()):
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
            coefs = list(flatten(np.abs(
                np.ones(len(best_positions))).tolist()))
        elif stack is True:
            coefs = list(flatten(np.abs(
                fitted.named_steps[best_estimator].coef_)))
        else:
            coefs = list(flatten(np.abs(fitted.named_steps[
                                            best_estimator.split(
                                                f"{predict_type}-")[1].split(
                                                '_')[0]].coef_).tolist()))

        feat_imp_dict = OrderedDict(
            sorted(
                dict(zip(best_positions, coefs)).items(),
                key=itemgetter(1),
                reverse=True,
            )
        )
    return best_positions, feat_imp_dict


def boot_nested_iteration(X, y, predict_type, boot,
                          feature_imp_dicts, best_positions_list,
                          grand_mean_best_estimator, grand_mean_best_score,
                          grand_mean_best_error, grand_mean_y_predicted,
                          k_folds_inner=10, k_folds_outer=10,
                          pca_reduce=True, dummy_run=False,
                          search_method='grid', stack=False,
                          stack_prefix_list=[], voting=True):

    # Grab CV prediction values
    if predict_type == 'classifier':
        final_scorer = 'roc_auc'
    else:
        final_scorer = 'r2'

    if stack is True:
        k_folds_outer = len(stack_prefix_list)**2

    # Instantiate a dictionary of estimators
    if predict_type == 'regressor':
        estimators = {
            "en": linear_model.ElasticNet(random_state=boot, warm_start=True),
            # "svm": LinearSVR(random_state=boot)
        }
        if dummy_run is True:
            estimators = {
                "dummy": DummyRegressor(strategy="prior"),
            }
    elif predict_type == 'classifier':
        estimators = {
            "en": linear_model.LogisticRegression(penalty='elasticnet',
                                                  fit_intercept=True,
                                                  solver='saga',
                                                  class_weight='auto',
                                                  random_state=boot,
                                                  warm_start=True),
            # "svm": LinearSVC(random_state=boot, class_weight='balanced',
            #                  loss='hinge')
        }
        if dummy_run is True:
            estimators = {
                "dummy": DummyClassifier(strategy="stratified",
                                         random_state=boot),
            }
    else:
        raise ValueError('Prediction method not recognized')

    # Instantiate an outer-fold
    if predict_type == 'regressor':
        outer_cv = KFold(n_splits=k_folds_outer,
                         shuffle=True, random_state=boot + 1)
    elif predict_type == 'classifier':
        outer_cv = StratifiedKFold(n_splits=k_folds_outer,
                                   shuffle=True,
                                   random_state=boot + 1)
    else:
        raise ValueError('Prediction method not recognized')

    if stack is True and predict_type == 'classifier':
        final_est, best_estimator = build_stacked_ensemble(
            X, y, estimators, boot, False,
            k_folds_inner, predict_type=predict_type,
            search_method=search_method, prefixes=stack_prefix_list,
            stacked_folds=k_folds_outer, voting=voting)
    else:
        final_est, best_estimator = nested_fit(
            X, y, estimators, boot, pca_reduce, k_folds_inner, predict_type,
            search_method=search_method
        )

    prediction = cross_validate(
        final_est,
        X,
        y,
        cv=outer_cv,
        scoring=(final_scorer, 'neg_mean_squared_error'),
        return_estimator=True,
    )

    final_est.fit(X, y)

    if stack is True:
        for super_fitted in prediction["estimator"]:
            for sub_fitted in \
                super_fitted.named_steps[best_estimator].estimators_:
                X_subspace = pd.DataFrame(
                    sub_fitted.named_steps['selector'].transform(X),
                    columns=sub_fitted.named_steps['selector'
                    ].get_feature_names())
                best_sub_estimator = [i for i in
                                      sub_fitted.named_steps.keys() if i in
                                      list(estimators.keys())[0] and i is not
                                      'selector'][0]
                best_positions, feat_imp_dict = get_feature_imp(
                    X_subspace, pca_reduce, sub_fitted, best_sub_estimator,
                    predict_type,
                    dummy_run, stack)
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
                X, pca_reduce, fitted, best_estimator, predict_type,
                dummy_run, stack)
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

    # Evaluate Bias-Variance
    if predict_type == 'classifier':
        [avg_expected_loss, avg_bias, avg_var] = bias_variance_decomp(
            final_est, X, y, loss='mse', num_rounds=200, random_seed=42)

    # Save the mean CV scores for this bootstrapped iteration
    if dummy_run is False and stack is False:
        params = f"{best_estimator}"
        out_final = final_est.named_steps[
            best_estimator.split(f"{predict_type}-")[1].split('_')[0]]
        if hasattr(out_final, 'intercept_'):
            params = f"{params}_intercept={out_final.intercept_[0]:.5f}"

        if hasattr(out_final, 'coef_'):
            params = f"{params}_betas={np.array(out_final.coef_).tolist()[0]}"

        if hasattr(out_final, 'l1_ratio') and (hasattr(out_final, 'C') or
                                               hasattr(out_final, 'alpha')):
            l1_ratio = out_final.l1_ratio

            if hasattr(out_final, 'C'):
                C = out_final.C
                l1_val = np.round(l1_ratio * (1/C), 5)
                l2_val = np.round((1 - l1_ratio) * (1/C), 5)
            else:
                alpha = out_final.alpha
                l1_val = np.round(l1_ratio * alpha, 5)
                l2_val = np.round((1 - l1_ratio) * 0.5 * alpha, 5)
            params = f"{params}_lambda1={l1_val}"
            params = f"{params}_lambda2={l2_val}"

        if predict_type == 'classifier':
            params = f"{params}_AvgExpLoss={avg_expected_loss:.5f}_" \
                     f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"
    else:
        if predict_type == 'classifier':
            params = f"{best_estimator}_AvgExpLoss={avg_expected_loss:.5f}_" \
                     f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"
        else:
            params = f"{best_estimator}"

    grand_mean_best_estimator[boot] = params

    if predict_type == 'regressor':
        grand_mean_best_score[boot] = np.nanmean(
            prediction[f"test_{final_scorer}"][
                prediction[f"test_{final_scorer}"] > 0])
        grand_mean_best_error[boot] = -np.nanmean(
            prediction["test_neg_mean_squared_error"])
    elif predict_type == 'classifier':
        grand_mean_best_score[boot] = np.nanmean(
            prediction[f"test_{final_scorer}"][
                prediction[f"test_{final_scorer}"] > 0])
        grand_mean_best_error[boot] = -np.nanmean(
            prediction[f"test_neg_mean_squared_error"])
    else:
        raise ValueError('Prediction method not recognized')
    grand_mean_y_predicted[boot] = final_est.predict(X)

    return feature_imp_dicts, best_positions_list, grand_mean_best_estimator, \
           grand_mean_best_score, grand_mean_best_error, \
           grand_mean_y_predicted, final_est


def bootstrapped_nested_cv(
    X,
    y,
    nuisance_cols=[],
    predict_type='classifier',
    n_boots=10,
    nodrop_columns=[],
    dummy_run=False,
    search_method='grid',
    stack=False,
    stack_prefix_list=[],
    remove_multi=True,
    n_jobs=2,
    voting=True
):
    from joblib import Parallel, delayed
    from pynets.core.utils import mergedicts

    # Preprocess data
    [X, y] = preprocess_x_y(X, y, nuisance_cols, nodrop_columns,
                            remove_multi=remove_multi, oversample=False)

    if X.empty or len(X.columns) < 5:
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

    # Instantiate a working dictionary of performance across bootstraps
    final_est = None

    # Bootstrap nested CV's "simulate" the variability of incoming data,
    # particularly when training on smaller datasets.
    # They are intended for model evaluation, rather then final deployment
    # and therefore do not include train-test splits.
    outs = Parallel(n_jobs=n_jobs)(
        delayed(boot_nested_iteration)(X, y, predict_type, boot,
                              [], [], {}, {}, {}, {}, dummy_run=dummy_run,
                              search_method=search_method, stack=stack,
                              stack_prefix_list=stack_prefix_list, voting=True)
        for boot in range(0, n_boots))

    feature_imp_dicts = []
    best_positions_list = []
    final_est_list = []
    grand_mean_best_estimator_list = []
    grand_mean_best_score_list = []
    grand_mean_best_error_list = []
    grand_mean_y_predicted_list = []
    for boot in range(0, n_boots):
        [feature_imp_dicts_boot, best_positions_list_boot,
         grand_mean_best_estimator_boot,
         grand_mean_best_score_boot, grand_mean_best_error_boot,
         grand_mean_y_predicted_boot, final_est_boot] = outs[boot]
        feature_imp_dicts.append(feature_imp_dicts_boot)
        best_positions_list.append(best_positions_list_boot)
        grand_mean_best_estimator_list.append(grand_mean_best_estimator_boot)
        grand_mean_best_score_list.append(grand_mean_best_score_boot)
        grand_mean_best_error_list.append(grand_mean_best_error_boot)
        grand_mean_y_predicted_list.append(grand_mean_y_predicted_boot)
        final_est_list.append(final_est_boot)

    grand_mean_best_estimator = {}
    for d in grand_mean_best_estimator_list:
        grand_mean_best_estimator = dict(mergedicts(grand_mean_best_estimator,
                                                    d))

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

    feature_imp_dicts = [item for sublist in feature_imp_dicts for item in
                         sublist]

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
                df_all.drop(columns=["id","participant_id"], inplace=True)
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
        drop_cols = [i for i in drop_cols if (i in df_all.columns) or
                     (i.replace('Behavioral_', '') in df_all.columns) or
                     (f"Behavioral_{i}" in df_all.columns)]

        return df_all.drop(columns=drop_cols), df_all[target_var].values
    else:
        print("\nEmpty/Missing Feature-space...\n")
        return None, None


def concatenate_frames(out_dir, modality, embedding_type, target_var, files_,
                       n_boots, dummy_run, search_method, stack,
                       stack_prefix_list):
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
                file_.split('_grid_param_')[1].split('/')[0].split('.')[-2])
        try:
            frame = pd.concat(dfs, axis=0, join="outer", sort=True,
                              ignore_index=False)

            out_path = f"{out_dir}/final_predictions_modality-{modality}_" \
                       f"subnet-{str(list(set(parcellations)))}_" \
                       f"gradient-{embedding_type}_outcome-{target_var}_" \
                       f"boots-{n_boots}_search-{search_method}"

            if dummy_run is True:
                out_path = out_path + '_dummy'

            if stack is True:
                out_path = out_path + '_stacked-' + str(stack_prefix_list)

            out_path = out_path.replace('[\'', '').replace('\']', '') + ".csv"

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
        import os
        from ast import literal_eval

        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param
        self._results["modality"] = self.inputs.modality
        self._results["target_var"] = self.inputs.target_var

        def prefix_df_columns(df, cols, prefix):
            new_names = [(i, f"{prefix}_{i}") for i in
                         df[cols].columns.values]
            df.rename(columns=dict(new_names), inplace=True)
            return df

        if self.inputs.json_dict is not None:
            if os.path.isfile(self.inputs.json_dict) and \
                    self.inputs.json_dict.endswith('.json') and \
                    os.stat(self.inputs.json_dict).st_size != 0:
                if self.inputs.target_var == 'MDE_conversion':
                    drop_cols = [self.inputs.target_var,
                                 'MDE_chronic']
                elif self.inputs.target_var == 'MDE_chronic':
                    drop_cols = [self.inputs.target_var,
                                 'MDE_conversion']
                elif self.inputs.target_var == "dep_persistence":
                    drop_cols = [self.inputs.target_var,
                                 "rum_persistence",
                                 "rum_2", "dep_2", "rum_1"]
                elif self.inputs.target_var == "rum_persistence":
                    drop_cols = [self.inputs.target_var,
                                 "dep_persistence",
                                 "rum_2", "dep_2"]
                elif self.inputs.target_var == "rum_1":
                    drop_cols = [self.inputs.target_var,
                                 "dep_persistence",
                                 "rum_persistence",
                                 "rum_2", "dep_2", "dep_1"]
                elif self.inputs.target_var == "dep_1":
                    drop_cols = [self.inputs.target_var,
                                 "dep_persistence",
                                 "rum_persistence",
                                 "rum_2", "dep_2", "rum_1"]
                else:
                    drop_cols = [self.inputs.target_var,
                                 "rum_persistence",
                                 "dep_persistence",
                                 "dep_1", "rum_1", "dep_2", "rum_2",
                                 'MDE_conversion', 'MDE_chronic']

                drop_cols = drop_cols + ['Behavioral_brooding_severity',
                                         'Behavioral_emotion_utilization',
                                         'Behavioral_social_ability_sum',
                                         'Behavioral_disability',
                                         'Behavioral_emotional_appraisal',
                                         'Behavioral_emotional_control',
                                         'Behavioral_Trait_anxiety',
                                         'Behavioral_State_anxiety',
                                         'Behavioral_perceptual_IQ']

                drop_cols = drop_cols + ["id", "participant_id"]

                [X, Y] = make_x_y(
                    self.inputs.json_dict,
                    drop_cols,
                    self.inputs.target_var,
                    self.inputs.embedding_type,
                    tuple(literal_eval(self.inputs.grid_param)),
                )

                numeric_cols = [col for col in X if col[0].isdigit()]
                if len(numeric_cols) > 0:
                    X = prefix_df_columns(X,
                                          numeric_cols,
                                          self.inputs.modality)

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
    n_boots = traits.Int()
    dummy_run = traits.Bool()
    search_method = traits.Str(mandatory=True)
    stack = traits.Bool()
    stack_prefix_list = traits.Any([], mandatory=True, usedefault=True)
    nuisance_cols = traits.Any()


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
        from colorama import Fore, Style
        from joblib import dump

        self._results["target_var"] = self.inputs.target_var
        self._results["modality"] = self.inputs.modality
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param

        if 'phenotype' in self.inputs.target_var or \
            'MDE_chronic' in self.inputs.target_var or \
            'MDE_conversion' in self.inputs.target_var:
            predict_type = 'classifier'
        else:
            predict_type = 'regressor'

        if self.inputs.X is None:
            return runtime

        if not self.inputs.X.empty and not np.isnan(self.inputs.y).all():
            if isinstance(self.inputs.X, pd.DataFrame):
                [
                    grand_mean_best_estimator,
                    grand_mean_best_score,
                    grand_mean_best_error,
                    mega_feat_imp_dict,
                    grand_mean_y_predicted,
                    final_est
                ] = bootstrapped_nested_cv(
                    self.inputs.X, self.inputs.y,
                    nuisance_cols=self.inputs.nuisance_cols,
                    predict_type=predict_type,
                    n_boots=self.inputs.n_boots,
                    dummy_run=self.inputs.dummy_run,
                    search_method=self.inputs.search_method,
                    stack=self.inputs.stack,
                    stack_prefix_list=self.inputs.stack_prefix_list)
                if final_est:
                    grid_param_name = self.inputs.grid_param.replace(', ', '_')
                    out_path_est = f"{runtime.cwd}/estimator_" \
                                   f"{self.inputs.target_var}_" \
                                   f"{self.inputs.modality}_" \
                                   f"{self.inputs.embedding_type}_" \
                                   f"{grid_param_name}.joblib"

                    dump(final_est, out_path_est)
                else:
                    out_path_est = None

                if len(grand_mean_best_estimator.keys()) > 1:
                    print(
                        f"\n\n{Fore.BLUE}Target Outcome: "
                        f"{Fore.GREEN}{self.inputs.target_var}"
                        f"{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Embedding type: "
                        f"{Fore.RED}{self.inputs.embedding_type}"
                        f"{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Grid Params: "
                        f"{Fore.RED}{self.inputs.grid_param}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Best Estimator: "
                        f"{Fore.RED}{grand_mean_best_estimator}"
                        f"{Style.RESET_ALL}"
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
                    if self.inputs.stack is False:
                        print(
                            f"{Fore.BLUE}Feature Importance: "
                            f"{Fore.RED}{list(mega_feat_imp_dict.keys())}"
                            f"{Style.RESET_ALL} "
                            f"with {Fore.RED}{len(mega_feat_imp_dict.keys())} "
                            f"features...{Style.RESET_ALL}\n\n"
                        )
                        print(
                            f"{Fore.BLUE}Modality: "
                            f"{Fore.RED}{self.inputs.modality}"
                            f"{Style.RESET_ALL}"
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
        from ast import literal_eval
        import pandas as pd
        import numpy as np

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
            df_summary.at[0, "best_estimator"] = list(
                self.inputs.grand_mean_best_estimator.values())
            df_summary.at[0, "Score"] = list(
                self.inputs.grand_mean_best_score.values())
            df_summary.at[0, "Predicted_y"] = y_pred_vals
            df_summary.at[0, "Error"] = list(
                self.inputs.grand_mean_best_error.values())
            df_summary.at[0, "Score_95CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()),
                alpha=0.95)[1]
            df_summary.at[0, "Score_95CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()),
                alpha=0.95)[0]
            df_summary.at[0, "Score_90CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()),
                alpha=0.90)[1]
            df_summary.at[0, "Score_90CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()),
                alpha=0.90)[0]
            df_summary.at[0, "Error_95CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()),
                alpha=0.95)[1]
            df_summary.at[0, "Error_95CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()),
                alpha=0.95)[0]
            df_summary.at[0, "Error_90CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()),
                alpha=0.90)[1]
            df_summary.at[0, "Error_90CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()),
                alpha=0.90)[0]
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


def create_wf(grid_params_mod, basedir, n_boots, nuisance_cols, dummy_run,
              search_method, stack, stack_prefix_list):
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
                "json_dict"
            ]
        ),
        name="inputnode",
    )

    make_x_y_func_node = pe.Node(MakeXY(), name="make_x_y_func_node")

    make_x_y_func_node.iterables = [("grid_param",
                                     [str(i) for i in grid_params_mod[1:]])]
    make_x_y_func_node.inputs.grid_param = str(grid_params_mod[0])
    make_x_y_func_node.interface.n_procs = 1
    make_x_y_func_node.interface._mem_gb = 6

    bootstrapped_nested_cv_node = pe.Node(
        BSNestedCV(n_boots=int(n_boots), nuisance_cols=nuisance_cols,
                   dummy_run=dummy_run, search_method=search_method,
                   stack=stack, stack_prefix_list=stack_prefix_list),
        name="bootstrapped_nested_cv_node")

    bootstrapped_nested_cv_node.interface.n_procs = 8
    bootstrapped_nested_cv_node.interface._mem_gb = 24

    make_df_node = pe.Node(MakeDF(), name="make_df_node")

    make_df_node.interface.n_procs = 1
    make_df_node.interface._mem_gb = 1

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary", "modality",
                                      "embedding_type",
                                      "target_var", "grid_param"]),
        name="df_join_node",
        joinfield=["df_summary", "grid_param"],
        joinsource=make_x_y_func_node,
    )

    concatenate_frames_node = pe.Node(
        niu.Function(
            input_names=["out_dir", "modality", "embedding_type",
                         "target_var", "files_", "n_boots",
                         "dummy_run", "search_method", "stack",
                         "stack_prefix_list"],
            output_names=["out_path", "embedding_type", "target_var",
                          "modality"],
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
            grid_params_mod.append([signal, hpass, model, granularity,
                                    parcellation, smooth])
    elif modality == "dwi":
        for comb in grid_param_combos:
            try:
                traversal, minlength, model, granularity, parcellation, \
                error_margin = comb
            except:
                raise ValueError(f"Failed to parse recipe: {comb}")
            grid_params_mod.append([traversal, minlength, model, granularity,
                                    parcellation, error_margin])

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

    create_wf_node = create_wf(grid_params_mod, ml_meta_wf.base_dir,
                               n_boots, nuisance_cols, dummy_run,
              search_method, stack, stack_prefix_list)

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

    create_wf_node.get_node('bootstrapped_nested_cv_node'
                            ).interface.n_procs = 8
    create_wf_node.get_node('bootstrapped_nested_cv_node'
                            ).interface._mem_gb = 24
    create_wf_node.get_node('make_x_y_func_node').interface.n_procs = 1
    create_wf_node.get_node('make_x_y_func_node').interface._mem_gb = 6

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
    nthreads = psutil.cpu_count()
    vmem = int(list(psutil.virtual_memory())[4] / 1000000000) - 1
    procmem = [int(nthreads),
               [vmem if vmem > 8 else int(8)][0]]
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
