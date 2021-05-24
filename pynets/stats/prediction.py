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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, \
    cross_validate, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, \
    f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import linear_model, decomposition
from collections import OrderedDict
from operator import itemgetter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pynets.core.utils import flatten
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin, clone, \
    ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split

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
    "from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, "
    "GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate",
    "from sklearn.dummy import DummyClassifier, DummyRegressor",
    "from sklearn.feature_selection import VarianceThreshold, SelectKBest, "
    "f_regression, f_classif",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler",
    "from sklearn import linear_model, decomposition",
    "from pynets.stats.benchmarking import build_hp_dict",
    "import seaborn as sns",
    "import matplotlib",
    "matplotlib.use('Agg')",
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
    "from statsmodels.stats.outliers_influence import "
    "variance_inflation_factor",
    "from sklearn.impute import SimpleImputer",
    "from pynets.core.utils import flatten",
    "import pickle",
    "import dill",
    "from sklearn.model_selection._split import _BaseKFold",
    "from sklearn.utils import check_random_state"
]


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


class RazorCV(object):
    """
    PR to SKlearn by @dPys 2019

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
            np.nanargmin(self.cv_results["param_" + self.param][candidate_idx])
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
            best_score_idx = np.nanargmax(best_mean_score)
            outstandard_error = (
                best_mean_score[best_score_idx] -
                best_std_score[best_score_idx]
            )
        else:
            best_score_idx = np.nanargmin(best_mean_score)
            outstandard_error = (
                best_mean_score[best_score_idx] +
                best_std_score[best_score_idx]
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
        tests = np.array(list(itertools.combinations(range(folds.shape[1]),
                                                     2)))

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
        best_mean_score = self.cv_results["mean_test_" +
                                          self.scoring][unq_cols]
        if self.greater_is_better is True:
            best_score_idx = np.nanargmax(best_mean_score)
        else:
            best_score_idx = np.nanargmin(best_mean_score)

        outstandard_error = best_mean_score[best_score_idx]
        return outstandard_error

    def call_percentile(self, tol):
        """
        Returns the simplest model that is within a percent tolerance of the
        empirically optimal model with the best `mean_test_scores`.
        """
        best_mean_score = self.cv_results["mean_test_" + self.scoring]
        if self.greater_is_better is True:
            best_score_idx = np.nanargmax(best_mean_score)
        else:
            best_score_idx = np.nanargmin(best_mean_score)

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


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X.iloc[bootstrap_indices.tolist(), :], y.iloc[bootstrap_indices.tolist(), :]


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

        avg_bias = np.sum(main_predictions != y_test.values) / y_test.values.size

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

        avg_bias = np.sum((main_predictions - y_test.values)**2) / y_test.values.size
        avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var


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


def de_outlier(X, y, sd, predict_type):
    """
    Remove any gross outlier row in X whose linear residual
    when regressing y against X is > sd standard deviations
    away from the mean residual. For classifiers, use a NaiveBayes estimator
    since it does not require tuning, and for regressors, use simple
    linear regression.

    """
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.naive_bayes import GaussianNB

    if predict_type == 'classifier':
        reg = GaussianNB()
    elif predict_type == 'regressor':
        reg = LinearRegression(normalize=True)
    else:
        raise ValueError('predict_type not recognized!')
    reg.fit(X, y)
    predicted_y = reg.predict(X)

    resids = (y - predicted_y)**2

    outlier_mask = (np.abs(stats.zscore(np.array(resids).reshape(-1, 1))) <
                    float(sd)).all(axis=1)

    return X[outlier_mask], y[outlier_mask]


def make_param_grids():
    param_space = {}
    param_space['Cs'] = [1e-16, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
    param_space['l1_ratios'] = [0, 0.25, 0.50, 0.75, 1]
    param_space['alphas'] = [1e-8, 1e-4, 1e-2, 1e-1, 0.25, 0.5, 1]
    return param_space


@ignore_warnings(category=ConvergenceWarning)
def nested_fit(X, y, estimators, boot, pca_reduce, k_folds,
               predict_type, search_method='grid', n_jobs=1):

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

    # N Features
    n_comps = [10]

    # Instantiate a working dictionary of performance within a bootstrap
    means_all_exp_var = {}
    means_all_MSE = {}

    # Model + feature selection by iterating grid-search across linear
    # estimators
    for estimator_name, estimator in sorted(estimators.items()):
        if pca_reduce is True and X.shape[0] < X.shape[1]:
            param_grid = {
                "feature_select__n_components": n_comps,
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

            refit = RazorCV.standard_error("n_components", True, refit_score)
            # refit = refit_score
        else:
            # <25 Features, don't perform feature selection, but produce a
            # userwarning
            if X.shape[1] < 25:
                param_grid = {}
                pipe = Pipeline([(estimator_name, estimator)])

                refit = refit_score
            else:
                param_grid = {
                    "feature_select__k": n_comps,
                }
                pipe = Pipeline(
                    [
                        ("feature_select", SelectKBest(feature_selector)),
                        (estimator_name, estimator),
                    ]
                )

                # refit = refit_score
                refit = RazorCV.standard_error("k", True, refit_score)

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
                    estimator_name + '__C']
                hyperparam_space = \
                    f"C-{c_best}"
        elif predict_type == 'regressor':
            alpha_best = pipe_grid_cv.best_estimator_.get_params()[
                estimator_name + '__alpha']
            hyperparam_space = f"_alpha-{alpha_best}"
        else:
            raise ValueError('Prediction method not recognized')

        if 'en' in estimator_name:
            best_l1 = pipe_grid_cv.best_estimator_.get_params()[
                estimator_name + '__l1_ratio']
            hyperparam_space = hyperparam_space + f"_l1ratio-{best_l1}"
        ## Model-specific naming chunk 2

        if pca_reduce is True and X.shape[0] < X.shape[1]:
            best_n_comps = pipe_grid_cv.best_estimator_.named_steps[
                'feature_select'].n_components
            best_estimator_name = f"{predict_type}-{estimator_name}" \
                                  f"_{hyperparam_space}_nfeatures-" \
                                  f"{best_n_comps}"
        else:
            if X.shape[1] < 25:
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
        if X.shape[1] < 25:
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


def preprocess_x_y(X, y, predict_type, nuisance_cols, nodrop_columns=[],
                   var_thr=.85, remove_multi=True,
                   remove_outliers=True, standardize=True,
                   std_dev=3, vif_thr=20, missingness_thr=0.50,
                   zero_thr=0.50, oversample=True):
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
        good_var_cols = X.columns[np.concatenate([sel.get_support(indices=True),
                                                 np.array([X.columns.get_loc(c)
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
        X, y = de_outlier(X, y, std_dev, predict_type)
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
                print(f"\n\n{Fore.RED}Empty feature-space (multicollinearity): "
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


def build_stacked_ensemble(X, y, base_estimators, boot,
                           k_folds_inner, predict_type='classifier',
                           search_method='grid', prefixes=[],
                           stacked_folds=10):
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.metrics import brier_score_loss
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    meta_estimators= {
        # "lr": LogisticRegression(random_state=boot),
        # "svm": LinearSVC(random_state=boot),
        "rfc": RandomForestClassifier(random_state=boot)
    }

    ens_estimators = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=0,
                                                        shuffle=True,
                                                        stratify=y)

    feature_subspaces = split_df_to_dfs_by_prefix(X_train, prefixes=prefixes)

    for meta_est_name, meta_est in meta_estimators.items():
        layer_ests = []
        for X_subspace in feature_subspaces:
            final_est, best_estimator = nested_fit(
                X_subspace, y_train, base_estimators, boot, False,
                k_folds_inner, predict_type, search_method=search_method
            )

            final_est.steps.insert(0, ('selector', ColumnTransformer([
                    ("selector", "passthrough", list(X_subspace.columns))
                ], remainder="drop")))
            layer_ests.append((list(set([i.split('_')[0] for i in
                                         X_subspace.columns]))[0], final_est))

        ec = StackingClassifier(estimators=layer_ests,
                                final_estimator=meta_est, passthrough=True,
                                cv=stacked_folds)

        ensemble_fitted = ec.fit(X_train, y_train)
        ens_estimators[meta_est_name] = {}
        ens_estimators[meta_est_name]['model'] = ensemble_fitted
        ens_estimators[meta_est_name]['score'] = brier_score_loss(
            ensemble_fitted.predict(X_test), y_test)

    # Select best SuperLearner
    best_estimator = min(ens_estimators,
                         key=lambda v: ens_estimators[v]['score'])
    final_est = Pipeline([(best_estimator,
                           ens_estimators[best_estimator]['model'])])
    return final_est, best_estimator


def split_df_to_dfs_by_prefix(df, prefixes=[]):
    from pynets.core.utils import flatten

    df_splits = []
    for p in prefixes:
        df_splits.append(df[list(set(list(flatten([c for c in df.columns if
                                                   c.startswith(p)]))))])
    pref_selected = list(set(list(flatten([i.columns for i in df_splits]))))
    # df_other = df[[j for j in df.columns if j not in pref_selected]]
    #return df_splits + [df_other]

    return df_splits


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

        if dummy_run is True:
            coefs = list(flatten(np.abs(
                np.ones(len(best_positions))).tolist()))
        elif stack is True:
            coefs = list(flatten(np.abs(fitted.named_steps[best_estimator].coef_)))
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
                          stack_prefix_list=[]):

    # Grab CV prediction values
    if predict_type == 'classifier':
        final_scorer = 'roc_auc'
    else:
        final_scorer = 'r2'

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
            X, y, estimators, boot,
            k_folds_inner, predict_type=predict_type,
            search_method=search_method, prefixes=stack_prefix_list)
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
            for sub_fitted in super_fitted.named_steps[best_estimator].estimators_:
                X_subspace = pd.DataFrame(sub_fitted.named_steps['selector'].transform(X))
                best_sub_estimator = [i for i in sub_fitted.named_steps.keys() if i in list(estimators.keys())[0] and i is not 'selector'][0]
                best_positions, feat_imp_dict = get_feature_imp(
                    X_subspace, pca_reduce, sub_fitted, best_sub_estimator, predict_type,
                    dummy_run, stack)
                feature_imp_dicts.append(feat_imp_dict)
                best_positions_list.append(best_positions)
    else:
        for fitted in prediction["estimator"]:
            best_positions, feat_imp_dict = get_feature_imp(
                X, pca_reduce, fitted, best_estimator, predict_type,
                dummy_run, stack)
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

    # Evaluate Bias-Variance
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

        if hasattr(out_final, 'l1_ratio'):
            l1_ratio = out_final.l1_ratio
            C = out_final.C
            l1_val = np.round(l1_ratio * (1/C), 5)
            l2_val = np.round((1 - l1_ratio) * (1/C), 5)
            params = f"{params}_lambda1={l1_val}"
            params = f"{params}_lambda2={l2_val}"

        params = f"{params}_AvgExpLoss={avg_expected_loss:.5f}_" \
                 f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"
    else:
        params = f"{best_estimator}_AvgExpLoss={avg_expected_loss:.5f}_" \
                 f"AvgBias={avg_bias:.5f}_AvgVar={avg_var:.5f}"

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
    n_jobs=2
):
    from joblib import Parallel, delayed
    from pynets.core.utils import mergedicts

    # Preprocess data
    [X, y] = preprocess_x_y(X, y, predict_type, nuisance_cols, nodrop_columns,
                            remove_multi=remove_multi)

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
                              stack_prefix_list=stack_prefix_list)
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
        rsns = []
        for file_ in files_:
            df = pd.read_csv(file_, chunksize=100000).read()
            try:
                df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
            except BaseException:
                pass
            dfs.append(df)
            rsns.append(file_.split('_grid_param_')[1].split('/')[0].split('.')[-2])
        try:
            frame = pd.concat(dfs, axis=0, join="outer", sort=True,
                              ignore_index=False)

            out_path = f"{out_dir}/final_predictions_modality-{modality}_" \
                       f"rsn-{str(list(set(rsns)))}_" \
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
        import gc
        import os
        import time
        from ast import literal_eval
        from pynets.stats.prediction import make_x_y

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
                elif self.inputs.target_var == "depression_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "rumination_persist_phenotype",
                                 "rum_2", "dep_2"]
                elif self.inputs.target_var == "rumination_persist_phenotype":
                    drop_cols = [self.inputs.target_var,
                                 "depression_persist_phenotype",
                                 "rum_2", "dep_2"]
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

                drop_cols = drop_cols + ['Behavioral_brooding_severity', 'Behavioral_emotion_utilization', 'Behavioral_social_ability_sum', 'Behavioral_disability', 'Behavioral_emotional_appraisal', 'Behavioral_emotional_control', 'Behavioral_Trait_anxiety', 'Behavioral_State_anxiety', 'Behavioral_perceptual_IQ']

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
        import os
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
                ] = bootstrapped_nested_cv(self.inputs.X, self.inputs.y,
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
                            f"{Fore.RED}{self.inputs.modality}{Style.RESET_ALL}"
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
            df_summary.at[0, "best_estimator"] = list(self.inputs.grand_mean_best_estimator.values())
            df_summary.at[0, "Score"] = list(self.inputs.grand_mean_best_score.values())
            df_summary.at[0, "Predicted_y"] = y_pred_vals
            df_summary.at[0, "Error"] = list(self.inputs.grand_mean_best_error.values())
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
            grid_params_mod.append([directget, minlength, model, res, atlas,
                                    tol])

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

    create_wf_node.get_node('bootstrapped_nested_cv_node').interface.n_procs = 8
    create_wf_node.get_node('bootstrapped_nested_cv_node').interface._mem_gb = 24
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
