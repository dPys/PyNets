#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@authors: Derek Pisner
"""
import pandas as pd
import numpy as np
import itertools
import re
from sklearn.metrics.pairwise import (
    cosine_distances,
    haversine_distances,
    manhattan_distances,
    euclidean_distances,
)
from sklearn.utils import check_X_y
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
import pandas as pd
import re
import numpy as np
import itertools
import warnings
warnings.simplefilter("ignore")
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, decomposition
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import OrderedDict
from operator import itemgetter
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
            and hyperparam != "minlength"
            and hyperparam != "samples"
            and hyperparam != "nodetype"
            and hyperparam != "template"

        ):
            if hyperparam not in hyperparam_dict.keys():
                hyperparam_dict[hyperparam] = [
                    file_renamed.split(hyperparam + "-")[1].split("_")[0]
                ]
            else:
                hyperparam_dict[hyperparam].append(
                    file_renamed.split(hyperparam + "-")[1].split("_")[0]
                )

    if modality == "func":
        if "smooth-" in file_renamed:
            if "smooth" not in hyperparam_dict.keys():
                hyperparam_dict["smooth"] = [file_renamed.split(
                    "smooth-")[1].split("_")[0].split("fwhm")[0]]
            else:
                hyperparam_dict["smooth"].append(
                    file_renamed.split("smooth-"
                                       )[1].split("_")[0].split("fwhm")[0])
            hyperparams.append("smooth")
        if "hpass-" in file_renamed:
            if "hpass" not in hyperparam_dict.keys():
                hyperparam_dict["hpass"] = [file_renamed.split(
                    "hpass-")[1].split("_")[0].split("Hz")[0]]
            else:
                hyperparam_dict["hpass"].append(
                    file_renamed.split("hpass-"
                                       )[1].split("_")[0].split("Hz")[0])
            hyperparams.append("hpass")
        if "extract-" in file_renamed:
            if "extract" not in hyperparam_dict.keys():
                hyperparam_dict["extract"] = [
                    file_renamed.split("extract-")[1].split("_")[0]
                ]
            else:
                hyperparam_dict["extract"].append(
                    file_renamed.split("extract-")[1].split("_")[0]
                )
            hyperparams.append("extract")

    elif modality == "dwi":
        if "directget-" in file_renamed:
            if "directget" not in hyperparam_dict.keys():
                hyperparam_dict["directget"] = [
                    file_renamed.split("directget-")[1].split("_")[0]
                ]
            else:
                hyperparam_dict["directget"].append(
                    file_renamed.split("directget-")[1].split("_")[0]
                )
            hyperparams.append("directget")
        if "minlength-" in file_renamed:
            if "minlength" not in hyperparam_dict.keys():
                hyperparam_dict["minlength"] = [
                    file_renamed.split("minlength-")[1].split("_")[0]
                ]
            else:
                hyperparam_dict["minlength"].append(
                    file_renamed.split("minlength-")[1].split("_")[0]
                )
            hyperparams.append("minlength")

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
    if (counts != 1).sum() <= 1:
        msg = "You have passed a vector containing only a single unique" \
              " sample id."
        raise ValueError(msg)
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
    A function for computing the reliability density function of a dataset.

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


def reshape_graphs(graphs):
    n, v1, v2 = np.shape(graphs)
    return np.reshape(graphs, (n, v1 * v2))


def CronbachAlpha(itemscores):
    itemscores = np.asarray([i for i in itemscores if np.nan not in i])
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    calpha = (nitems / float(nitems - 1) *
              (1 - itemvars.sum() / float(tscores.var(ddof=1))))

    return calpha


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        # print('Dropping columns by excessive VIF...')
        return self

    def transform(self, X, y=None):
        columns = X.columns.tolist()
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        dropped = True
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
                # print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X


# We create a callable class, called `RazorCV`,
class RazorCV(object):
    '''
    PR to SKlearn by dPys 2019

    RazorCV is a callable refit option for CV whose aim is to balance model
    complexity and cross-validated score in the spirit of the
    "one standard error" rule of Breiman et al. (1984), which demonstrated that
    the tuning parameter associated with the best performance may be prone to
    overfit. To ensure model parsimony, we can instead pick the simplest model
    within one standard error (or some percentile/alpha-level tolerance) of
    the empirically optimal model. This assumes that the models can be easily
    ordered from simplest to most complex based on some user-defined target
    parameter. By enabling the user to specify this parameter of interest,
    whether greater values of this parameter are to be defined as
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
    Here, simplest is defined by the complexity of the model as influenced by
    some user-defined target parameter (e.g. number of components, number of
    estimators, polynomial degree, cost, scale, number hidden units, weight
    decay, number of nearest neighbors, L1/L2 penalty, etc.).
    See :ref:`sphx_glr_auto_examples_applications_plot_model_complexity_influence.py`
    See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_refit_callable.py`
    '''

    def __init__(self, cv_results, param, greater_is_complex, scoring, method,
                 tol=0.25, alpha=0.01):
        import sklearn.metrics
        self.cv_results = cv_results
        self.param = param
        self.greater_is_complex = greater_is_complex
        self.scoring = scoring
        self.method = method
        self.scoring_funcs = [met for met in sklearn.metrics.__all__ if
                              (met.endswith('_score')) or
                              (met.endswith('_error'))]
        # Set _score metrics to True and _error metrics to False
        self.scoring_dict = dict(zip(self.scoring_funcs,
                                     [met.endswith('_score') for met in
                                      self.scoring_funcs]))
        self.greater_is_better = self._check_scorer()
        self.tol = tol
        self.alpha = alpha

    def _check_scorer(self):
        """
        Check whether the target refit scorer is negated. If so, adjusted
        greater_is_better accordingly.
        """
        if self.scoring not in self.scoring_dict.keys() and f"{self.scoring}_score" not in self.scoring_dict.keys():
            if self.scoring.startswith('neg_'):
                self.greater_is_better = True
            else:
                raise KeyError(f"Scoring metric {self.scoring} not "
                               f"recognized.")
        else:
            self.greater_is_better = [value for key, value in
                                      self.scoring_dict.items() if
                                      self.scoring in key][0]
        return self.greater_is_better

    def _best_low_complexity(self):
        """
        Balance model complexity with cross-validated score.
        """
        # Check parameter whose complexity we seek to restrict
        if not any(
            self.param in x for x in self.cv_results['params'][0].keys()):
            raise KeyError('Parameter not found in cv grid.')
        else:
            self.param = [i for i in self.cv_results['params'][0].keys() if
                          i.endswith(self.param)][0]

        if self.method == 'onese':
            threshold = self.call_standard_error()
        elif self.method == 'percentile':
            if self.tol is None:
                raise ValueError('For percentile method, the tolerance '
                                 '(i.e. `tol`) parameter cannot be null.')
            threshold = self.call_percentile(tol=self.tol)
        elif self.method == 'ranksum':
            if self.alpha is None:
                raise ValueError('For ranksum method, the alpha-level '
                                 '(i.e. `alpha`) parameter cannot be null.')
            threshold = self.call_rank_sum_test(alpha=self.alpha)
        else:
            raise ValueError('Method ' + self.method + ' is not valid.')

        if self.greater_is_complex is True:
            candidate_idx = np.flatnonzero(self.cv_results['mean_test_' +
                                                           self.scoring] >=
                                           threshold)
        else:
            candidate_idx = np.flatnonzero(self.cv_results['mean_test_' +
                                                           self.scoring] <=
                                           threshold)

        best_idx = candidate_idx[self.cv_results['param_' + self.param]
        [candidate_idx].argmin()]
        return best_idx

    def call_standard_error(self):
        """
        Calculate the upper/lower bound within 1 standard deviation
        of the best `mean_test_scores`.
        """
        best_mean_score = self.cv_results['mean_test_' + self.scoring]
        best_std_score = self.cv_results['std_test_' + self.scoring]
        if self.greater_is_better is True:
            best_score_idx = np.argmax(best_mean_score)
            outstandard_error = (best_mean_score[best_score_idx] -
                                 best_std_score[best_score_idx])
        else:
            best_score_idx = np.argmin(best_mean_score)
            outstandard_error = (best_mean_score[best_score_idx] +
                                 best_std_score[best_score_idx])
        return outstandard_error

    def call_rank_sum_test(self, alpha):
        """
        Returns the performance of the simplest model whose performance is not
        significantly different across folds.
        """
        from scipy.stats import wilcoxon
        import itertools
        folds = np.vstack([self.cv_results[fold] for fold in
                           [i for i in self.cv_results.keys() if
                            ('split' in i) and (self.scoring in i)]])
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
            raise ValueError('Models are all significantly different from one'
                             ' another')
        best_mean_score = self.cv_results['mean_test_' +
                                          self.scoring][unq_cols]
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
        best_mean_score = self.cv_results['mean_test_' + self.scoring]
        if self.greater_is_better is True:
            best_score_idx = np.argmax(best_mean_score)
        else:
            best_score_idx = np.argmin(best_mean_score)

        outstandard_error = \
            (np.abs(best_mean_score[best_score_idx]) - tol) / tol
        return outstandard_error

    def standard_error(param, greater_is_complex, scoring):
        '''
        Standard error callable

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        greater_is_complex : bool
            Whether complexity increases as `param` increases. Default is True.
        scoring : str
            Refit scoring metric.
        '''
        from functools import partial

        def razor_pass(cv_results, param, greater_is_complex, scoring,
                       method='onese'):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(razor_pass, param=param,
                       greater_is_complex=greater_is_complex, scoring=scoring)

    def ranksum(param, greater_is_complex, scoring, alpha):
        '''
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
        '''
        from functools import partial

        def razor_pass(cv_results, param, greater_is_complex, scoring, alpha,
                       method='ranksum'):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method, alpha)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(razor_pass, param=param,
                       greater_is_complex=greater_is_complex, scoring=scoring,
                       alpha=alpha)

    def percentile(param, greater_is_complex, scoring, tol):
        '''
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
        '''
        from functools import partial

        def razor_pass(cv_results, param, greater_is_complex, scoring, tol,
                       method='percentile'):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method, tol)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(razor_pass, param=param,
                       greater_is_complex=greater_is_complex, scoring=scoring,
                       tol=tol)


def nested_fit(X, y, regressors, boot, pca_reduce, k_folds):
    # Instantiate an inner-fold
    inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=boot)

    # Scoring metrics
    scoring = ['explained_variance', 'neg_root_mean_squared_error']

    refit_score = 'explained_variance'

    # Instantiate grid of model/feature-selection params
    alphas = [0.000001, 0.00001, 0.0001, 0.001]
    n_comps = [5, 10, 15]

    # Instantiate a working dictionary of performance within a bootstrap
    means_all_exp_var = {}
    means_all_MSE = {}

    # Model + feature selection by iterating grid-search across linear regressors
    for regressor_name, regressor in sorted(regressors.items()):
        if pca_reduce is True and X.shape[0] < X.shape[1]:
            # Pipeline feature selection (PCA) with model fitting
            pipe = Pipeline([
                ('feature_select',
                 decomposition.PCA(random_state=boot, whiten=True)),
                (regressor_name, regressor),
            ])
            param_grid = {regressor_name + '__alpha': alphas,
                          'feature_select__n_components': n_comps}
            param = 'n_components'
        else:
            pipe = Pipeline([
                ('feature_select', SelectKBest(f_regression)),
                (regressor_name, regressor),
            ])
            param_grid = {regressor_name + '__alpha': alphas,
                          'feature_select__k': n_comps}
            param = 'k'

        # Establish grid-search feature/model tuning windows,
        # refit the best model using a 1 SE rule of MSE values.
        pipe_grid_cv = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit=RazorCV.standard_error(param, True, refit_score),
            #refit=refit_score,
            n_jobs=1,
            cv=inner_cv,
            iid=False
        )

        # Fit pipeline to data
        pipe_grid_cv.fit(X, y)

        # Grab mean
        means_exp_var = pipe_grid_cv.cv_results_[
            f"mean_test_explained_variance"]
        means_MSE = pipe_grid_cv.cv_results_[
            f"mean_test_neg_root_mean_squared_error"]

        # Apply PCA in the case that the # of features exceeds the number of observations
        if pca_reduce is True and X.shape[0] < X.shape[1]:
            best_estimator_name = f"{regressor_name}_{pipe_grid_cv.best_estimator_.get_params()[regressor_name + '__alpha']}_{pipe_grid_cv.best_estimator_.named_steps['feature_select'].n_components}"
        else:
            best_estimator_name = f"{regressor_name}_{pipe_grid_cv.best_estimator_.get_params()[regressor_name + '__alpha']}_{pipe_grid_cv.best_estimator_.named_steps['feature_select'].k}"

        means_all_exp_var[best_estimator_name] = np.nanmean(means_exp_var)
        means_all_MSE[best_estimator_name] = np.nanmean(means_MSE)

    # Get best regressor across models
    best_regressor = max(means_all_exp_var, key=means_all_exp_var.get)
    est = regressors[best_regressor.split('_')[0]]
    est.alpha = float(best_regressor.split('_')[-2])

    if pca_reduce is True and X.shape[0] < X.shape[1]:
        pca = decomposition.PCA(
            n_components=int(best_regressor.split('_')[-1]),
            whiten=True)
        reg = Pipeline(
            [('feature_select', pca), (best_regressor.split('_')[0], est)])
    else:
        kbest = SelectKBest(f_regression, k=int(best_regressor.split('_')[-1]))
        reg = Pipeline(
            [('feature_select', kbest), (best_regressor.split('_')[0], est)])

    return reg, best_regressor


def build_grid(modality, hyperparam_dict, hyperparams, ensembles):
    for ensemble in ensembles:
        build_hp_dict(
            ensemble,
            modality,
            hyperparam_dict,
            hyperparams)

    hyperparam_dict = OrderedDict(
        sorted(hyperparam_dict.items(), key=lambda x: x[0]))

    grid = list(
        itertools.product(
            *(hyperparam_dict[param] for param in hyperparam_dict.keys())
        )
    )
    return hyperparam_dict, grid


def get_coords_labels(embedding):
    import os
    import pickle

    coords_file = f"{os.path.dirname(embedding)}/nodes/all_mni_coords.pkl"
    if os.path.isfile(coords_file):
        with open(coords_file, "rb") as file_:
            coords = pickle.load(file_)
        file_.close()
    labels_file = f"{os.path.dirname(embedding)}/nodes/all_mni_labels.pkl"
    if os.path.isfile(labels_file):
        with open(labels_file, "rb") as file_:
            labels = pickle.load(file_)
        file_.close()
    return coords, labels


def flatten_latent_positions(rsn, subject_dict, ID, ses, modality, grid_param, alg):
    if ((rsn,) + grid_param) in subject_dict[ID][ses][modality]:
        if alg in subject_dict[ID][ses][modality][((rsn,) + grid_param)].keys():
            rsn_dict = subject_dict[ID][ses][modality][((rsn,) + grid_param)][alg]
            ixs = [i[1] for i in rsn_dict['labels']]
            if len(ixs) == rsn_dict['data'].shape[0]:
                rsn_arr = rsn_dict['data'].T.reshape(1, rsn_dict['data'].T.shape[0] * rsn_dict['data'].T.shape[1])
                df_lps = pd.DataFrame(rsn_arr, columns=[f"{rsn}_{i}_dim1" for i in ixs] + [f"{rsn}_{i}_dim2" for i in ixs] + [f"{rsn}_{i}_dim3" for i in ixs] + [f"{rsn}_{i}_dim4" for i in ixs])
                print(df_lps)
            else:
                df_lps = None
        else:
            df_lps = None
    else:
        df_lps = None

    return df_lps


def create_feature_space(df, grid_param, subject_dict, ses, modality, alg):
    df_tmps = []
    rsns = ['SalVentAttnA', 'DefaultA', 'ContB']

    for ID in df['participant_id']:
        if ID not in subject_dict.keys():
            continue

        if ses not in subject_dict[ID].keys():
            continue

        if modality not in subject_dict[ID][ses].keys():
            continue

        rsn_frames = []
        for rsn in rsns:
            df_lps = flatten_latent_positions(rsn, subject_dict, ID, ses,
                                              modality, grid_param, alg)
            rsn_frames.append(df_lps)

        rsn_frames = [i for i in rsn_frames if i is not None]
        if len(rsn_frames) == 3:
            rsn_big_df = pd.concat(rsn_frames, axis=1)
            df_tmp = df[df["participant_id"] == ID].reset_index().drop(columns='index').join(rsn_big_df, how='right')
            df_tmps.append(df_tmp)
            del df_tmp

    if len(df_tmps) > 0:
        dfs = [dff.set_index('participant_id') for dff in df_tmps]
        df_all = pd.concat(dfs, axis=0)
        del df_tmps
        return df_all
    else:
        return None


def graph_theory_prep(df, thr_type):
    cols = [
        j
        for j in set(
            [i.split("_thrtype-" + thr_type + "_")[0] for i in
             list(set(df.columns))]
        )
        if j != "id"
    ]
    for col in [i for i in df.columns if i != "id" and i != "participant_id"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df[col] = df[col][df[col] > 0]
        df[col] = df[col][df[col] < 1]
    return df, cols


def bootstrapped_nested_cv(X, y, n_boots=10, var_thr=.8, k_folds=10,
                           pca_reduce=True, remove_multi=True, std_dev = 3):

    # Remove columns with > 10% missinng values
    X = X.dropna(thresh=len(X) * .80, axis=1)

    # Apply a simple imputer (note that this assumes extreme cases of
    # missingness have already been addressed)
    imp = SimpleImputer()
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Remove low-variance columns
    sel = VarianceThreshold(threshold=(var_thr * (1 - var_thr)))
    sel.fit(X)
    X = X[X.columns[sel.get_support(indices=True)]]

    # Remove multicollinear columns
    if remove_multi is True:
        transformer = ReduceVIF()
        X = transformer.fit_transform(X, y)

    outlier_mask = (np.abs(stats.zscore(X)) < float(std_dev)).all(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]

    # Standardize X
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Standardize Y
    # scaler = StandardScaler()
    # y = pd.DataFrame(scaler.fit_transform(y.reshape(-1,1)))

    # Instantiate a working dictionary of performance across bootstraps
    grand_mean_best_estimator = {}
    grand_mean_best_Rsquared = {}
    grand_mean_best_MSE = {}

    # Bootstrap train-test split
    # Repeated test-train splits "simulates" the variability of incoming data,
    # particularly when training on smaller datasets
    feature_imp_dicts = []
    best_positions_list = []
    for boot in range(0, n_boots):
        # Instantiate a dictionary of regressors
        regressors = {'l1': linear_model.Lasso(random_state=boot, fit_intercept=True, warm_start=True),
                      'l2': linear_model.Ridge(random_state=boot, fit_intercept=True)}

        # Instantiate an outer-fold
        outer_cv = KFold(n_splits=k_folds, shuffle=True, random_state=boot + 1)

        final_est, best_regressor = nested_fit(X, y, regressors, boot,
                                               pca_reduce, k_folds)

        # Grab CV prediction values on test-set
        prediction = cross_validate(final_est, X, y,
                                    cv=outer_cv,
                                    scoring=('r2',
                                             'neg_root_mean_squared_error'),
                                    return_estimator=True)

        for fitted in prediction['estimator']:
            if pca_reduce is True and X.shape[0] < X.shape[1]:
                pca = fitted.named_steps['feature_select']
                pca.fit_transform(X)
                comps_all = pd.DataFrame(pca.components_, columns=X.columns)
                coefs = np.abs(
                    fitted.named_steps[best_regressor.split('_')[0]].coef_)
                feat_imp_dict = OrderedDict(sorted(dict(zip(comps_all,
                                                            coefs)).items(),
                                                   key=itemgetter(1),
                                                   reverse=True))

                n_pcs = pca.components_.shape[0]

                best_positions = [np.abs(pca.components_[i]).argmax() for i
                                  in range(n_pcs)]

                feat_imp_dict = OrderedDict(sorted(dict(zip(best_positions,
                                            feat_imp_dict.values())).items(),
                                   key=itemgetter(1),
                                   reverse=True))


            else:
                best_positions = [column[0] for
                                    column in zip(X.columns,
                                                  fitted.named_steps['feature_select'].get_support(indices=True)) if column[1]]

                coefs = np.abs(fitted.named_steps[best_regressor.split('_')[0]].coef_)

                feat_imp_dict = OrderedDict(sorted(dict(zip(best_positions,
                                                            coefs)).items(),
                                                   key=itemgetter(1),
                                                   reverse=True))
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

        # Save the mean CV scores for this bootstrapped iteration
        grand_mean_best_estimator[boot] = best_regressor
        grand_mean_best_Rsquared[boot] = np.nanmean(
            prediction['test_r2'])
        grand_mean_best_MSE[boot] = -np.nanmean(
            prediction['test_neg_root_mean_squared_error'])

    unq_best_positions = list(flatten(list(np.unique(best_positions_list))))

    mega_feat_imp_dict = dict.fromkeys(unq_best_positions)

    for feat in unq_best_positions:
        running_mean = []
        for ref in feature_imp_dicts:
            if feat in ref.keys():
                running_mean.append(ref[feat])
        mega_feat_imp_dict[feat] = np.mean(running_mean)

    mega_feat_imp_dict = OrderedDict(sorted(mega_feat_imp_dict.items(),
                                            key=itemgetter(1), reverse=True))

    del X, y, scaler

    return grand_mean_best_estimator, grand_mean_best_Rsquared, grand_mean_best_MSE, mega_feat_imp_dict


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"

    working_dir = '/Users/derekpisner/Downloads'
    thr_type = "MST"
    icc = True
    disc = True
    modality = 'func'

    mets = [
        "global_efficiency",
        "transitivity",
        "average_clustering",
        "average_shortest_path_length",
        "average_betweenness_centrality",
        "average_eigenvector_centrality",
        "average_degree_centrality",
        "average_diversity_coefficient",
        "average_participation_coefficient"
    ]

    df = pd.read_csv(working_dir + f"/all_subs_neat_{modality}.csv")
    df = df.dropna(subset=["id"])
    df['id'] = df['id'].str.replace('topology_auc_sub-', '')
    df['id'] = df['id'].str.replace("_ses-ses-", "_ses-")
    df['id'] = df['id'].str.replace(".csv", "")

    df = df.rename(columns=lambda x: re.sub("partcorr", "model-partcorr", x))
    df = df.rename(columns=lambda x: re.sub("corr", "model-corr", x))
    df = df.rename(columns=lambda x: re.sub("cov", "model-cov", x))

    cols = [
        j
        for j in set(
            [i.split("_thrtype-" + thr_type + "_")[0] for i in
             list(set(df.columns))]
        )
        if j != "id"
    ]

    for col in [i for i in df.columns if i != "id"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        df[col] = df[col][df[col] > 0]
        df[col] = df[col][df[col] < 1]
        # df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3)]

    # df = df.drop(
    #     df.loc[:, list((100 * (df.isnull().sum() /
    #                            len(df.index)) > 20))].columns, 1
    # )

    hyperparam_dict = {}

    if icc is True and disc is False:
        df_summary = pd.DataFrame(columns=["grid", "icc"])
    elif icc is False and disc is True:
        df_summary = pd.DataFrame(columns=["grid", "discriminability"])
    elif icc is True and disc is True:
        df_summary = pd.DataFrame(columns=["grid", "discriminability", "icc"])

    if modality == "func":
        #gen_hyperparams = ["model", "clust", "_k"]
        hyperparams = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']

        for col in cols:
            build_hp_dict(
                col,
                "func",
                hyperparam_dict,
                hyperparams)

        grid = list(
            itertools.product(
                *(hyperparam_dict[param] for param in hyperparam_dict.keys())
            )
        )

        subject_dict = {}
        columns = df.columns
        for id in df["id"]:
            print(id)
            ID = id.split("_")[0].split("sub-")[1]
            ses = id.split("_")[1].split("ses-")[1]
            if ID not in subject_dict.keys():
                subject_dict[ID] = {}
            subject_dict[ID][ses] = dict.fromkeys(grid, np.nan)

            for atlas, res, model, extract, hpass, smooth in \
                subject_dict[ID][ses]:
                subject_dict[ID][ses][(
                    atlas, res, model, extract, hpass, smooth)] = {}

            # for atlas, model, clust, _k, smooth, hpass in \
            #     subject_dict[ID][ses]:
            #     subject_dict[ID][ses][(
            #         atlas, model, clust, _k, smooth, hpass)] = {}
                met_vals = np.empty([len(mets), 1], dtype=np.float32)
                met_vals[:] = np.nan
                i = 0
                for met in mets:
                    # col = (
                    #     atlas
                    #     + "_clust-"
                    #     + clust
                    #     + "_k-"
                    #     + str(_k)
                    #     + "_model-"
                    #     + model
                    #     + "_nodetype-parc_"
                    #     + "smooth-"
                    #     + str(smooth)
                    #     + "fwhm_hpass-"
                    #     + str(hpass)
                    #     + "Hz_"
                    #     + "thrtype-"
                    #     + thr_type
                    #     + "_topology_"
                    #     + met
                    #     + "_auc"
                    # )
                    col = (
                        'rsn-'
                        + atlas
                        + "_res-"
                        + res
                        + "_model-"
                        + model
                        + '_template-MNI152_T1_nodetype-parc_'
                        + "smooth-"
                        + str(smooth)
                        + "fwhm_hpass-"
                        + str(hpass)
                        + "Hz_extract-"
                        + extract
                        + "_thrtype-"
                        + thr_type
                        + "_auc_"
                        + met
                        + "_auc"
                    )
                    if col in columns:
                        out = df[df["id"] == "sub-" + ID + "_ses-" +
                                 ses][col].values[0]
                        print(out)
                    else:
                        out = None
                        # print(
                        #     "No values found for: " +
                        #     met +
                        #     " in column: " +
                        #     col +
                        #     "\n")
                        met_vals[i] = np.nan
                    if str(out) != 'nan':
                        #print(col)
                        met_vals[i] = out
                    else:
                        # print(
                        #     "No values found for: " +
                        #     met +
                        #     " in column: " +
                        #     col +
                        #     "\n")
                        met_vals[i] = np.nan
                    del col
                    i += 1

                if np.sum(np.isnan(met_vals)) != len(met_vals):
                    subject_dict[ID][ses][(
                        atlas, res, model, extract, hpass, smooth)
                    ]["topology"] = met_vals
                del i, atlas, model, hpass, smooth, extract
            del ID, ses


        if icc is True:
            i = 0
            for atlas, res, model, extract, hpass, smooth in grid:
                df_summary.at[i, "grid"] = (
                    atlas, res, model, extract, hpass, smooth)
                print(atlas, res, model, extract, hpass, smooth)
                id_list = []
                icc_list = []
                for ID in subject_dict.keys():
                    ses_list = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        ses_list.append(
                            subject_dict[ID][ses][
                                (atlas, res, model, extract, hpass, smooth)
                            ]["topology"]
                        )
                    meas = np.hstack(ses_list)
                    try:
                        icc_out = CronbachAlpha(meas)
                        icc_list.append(icc_out)
                        df_summary.at[i, "icc"] = np.nanmean(icc_list)
                        del icc_out, ses_list
                    except BaseException:
                        continue
                del icc_list
                i += 1

        if disc is True:
            i = 0
            for atlas, res, model, extract, hpass, smooth in grid:
                print(atlas, res, model, extract, hpass, smooth)
                id_list = []
                vect_all = []
                for ID in subject_dict.keys():
                    vects = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        vects.append(
                            subject_dict[ID][ses][
                                (atlas, res, model, extract, hpass, smooth)
                            ]["topology"]
                        )
                    vect_all.append(np.concatenate(vects, axis=1))
                    del vects
                X_top = np.swapaxes(np.hstack(vect_all), 0, 1)

                Y = np.array(id_list)
                try:
                    df_summary.at[i, "grid"] = (
                        atlas, res, model, extract, hpass, smooth)
                    bad_ixs = [i[1] for i in np.argwhere(np.isnan(X_top))]
                    for m in set(bad_ixs):
                        if (X_top.shape[0] - bad_ixs.count(m)
                                ) / X_top.shape[0] < 0.50:
                            X_top = np.delete(X_top, m, axis=1)
                    imp = IterativeImputer(max_iter=50, random_state=42)
                    X_top = imp.fit_transform(X_top)
                    scaler = StandardScaler()
                    X_top = scaler.fit_transform(X_top)
                    discr_stat_val, rdf = discr_stat(X_top, Y)
                    df_summary.at[i, "discriminability"] = discr_stat_val
                    print(discr_stat_val)
                    # print(rdf)
                    del discr_stat_val
                    i += 1
                except BaseException:
                    i += 1
                    continue
    elif modality == "dwi":
        gen_hyperparams = ["model", "clust", "_k"]
        for col in cols:
            build_hp_dict(
                col,
                col.split("_clust")[0],
                "dwi",
                hyperparam_dict,
                gen_hyperparams)

        for key in hyperparam_dict:
            hyperparam_dict[key] = list(set(hyperparam_dict[key]))

        grid = list(
            itertools.product(
                *(hyperparam_dict[param] for param in hyperparam_dict.keys())
            )
        )

        subject_dict = {}
        for id in df["id"]:
            print(id)
            ID = id.split("_")[0].split("sub-")[1]
            ses = id.split("_")[1].split("ses-")[1]
            if ID not in subject_dict.keys():
                subject_dict[ID] = {}
            subject_dict[ID][ses] = dict.fromkeys(grid, np.nan)
            for (
                atlas,
                model,
                clust,
                _k,
                track_type,
                directget,
                min_length,
            ) in subject_dict[ID][ses]:
                subject_dict[ID][ses][
                    (atlas, model, clust, _k, track_type, directget,
                     min_length)
                ] = {}
                met_vals = np.empty([len(mets), 1], dtype=np.float32)
                met_vals[:] = np.nan
                i = 0
                for met in mets:
                    col = (
                        atlas
                        + "_clust-"
                        + clust
                        + "_k-"
                        + str(_k)
                        + "_model-"
                        + model
                        + "_nodetype-parc_samples-20000streams_tracktype-"
                        + track_type
                        + "_directget-"
                        + directget
                        + "_minlength-"
                        + min_length
                        + "_thrtype-"
                        + thr_type
                        + "_topology_"
                        + met
                        + "_auc"
                    )
                    try:
                        met_vals[i] = df[df["id"] == "sub-" +
                                         ID + "_ses-" + ses][col].values[0]
                    except BaseException:
                        print(
                            "No values found for: " +
                            met +
                            " in column: " +
                            col +
                            "\n")
                        met_vals[i] = np.nan
                    del col
                    i += 1
                subject_dict[ID][ses][
                    (atlas, model, clust, _k, track_type, directget,
                     min_length)]["topology"] = met_vals
                del i, atlas, model, clust, _k, track_type, directget, \
                    min_length
            del ID, ses

        if icc is True:
            i = 0
            for atlas, model, clust, _k, track_type, directget, min_length in \
                grid:
                df_summary.at[i, "grid"] = (
                    atlas,
                    model,
                    clust,
                    _k,
                    track_type,
                    directget,
                    min_length,
                )
                print(atlas, model, clust, _k, track_type, directget,
                      min_length)
                id_list = []
                icc_list = []
                for ID in subject_dict.keys():
                    ses_list = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        ses_list.append(
                            subject_dict[ID][ses][
                                (
                                    atlas,
                                    model,
                                    clust,
                                    _k,
                                    track_type,
                                    directget,
                                    min_length,
                                )
                            ]["topology"]
                        )
                    meas = np.hstack(ses_list)
                    try:
                        icc_out = CronbachAlpha(meas)
                        icc_list.append(icc_out)
                        df_summary.at[i, "icc"] = np.nanmean(icc_list)
                        del icc_out, ses_list
                    except BaseException:
                        continue
                del icc_list
                i += 1

        if disc is True:
            i = 0
            for atlas, model, clust, _k, track_type, directget, min_length in\
                grid:
                print(atlas, model, clust, _k, track_type, directget,
                      min_length)
                id_list = []
                vect_all = []
                for ID in subject_dict.keys():
                    vects = []
                    for ses in subject_dict[ID].keys():
                        id_list.append(ID)
                        vects.append(
                            subject_dict[ID][ses][
                                (
                                    atlas,
                                    model,
                                    clust,
                                    _k,
                                    track_type,
                                    directget,
                                    min_length,
                                )
                            ]["topology"]
                        )
                    vect_all.append(np.concatenate(vects, axis=1))
                    del vects
                X_top = np.swapaxes(np.hstack(vect_all), 0, 1)

                Y = np.array(id_list)
                try:
                    df_summary.at[i, "grid"] = (
                        atlas,
                        model,
                        clust,
                        _k,
                        track_type,
                        directget,
                        min_length,
                    )
                    bad_ixs = [i[1] for i in np.argwhere(np.isnan(X_top))]
                    for m in set(bad_ixs):
                        if (X_top.shape[0] - bad_ixs.count(m)
                                ) / X_top.shape[0] < 0.50:
                            X_top = np.delete(X_top, m, axis=1)
                    imp = IterativeImputer(max_iter=50, random_state=42)
                    X_top = imp.fit_transform(X_top)
                    scaler = StandardScaler()
                    X_top = scaler.fit_transform(X_top)
                    discr_stat_val, rdf = discr_stat(X_top, Y)
                    df_summary.at[i, "discriminability"] = discr_stat_val
                    print(discr_stat_val)
                    # print(rdf)
                    del discr_stat_val
                    i += 1
                except BaseException:
                    i += 1
                    continue

    if icc is True and disc is False:
        df_summary = df_summary.sort_values("icc", ascending=False)
        # df_summary = df_summary[df_summary.topological_icc >
        #                         df_summary.icc.quantile(.50)]
    elif icc is False and disc is True:
        df_summary = df_summary.sort_values(
            "discriminability", ascending=False)
        # df_summary = df_summary[df_summary.discriminability >
        #                         df_summary.discriminability.quantile(.50)]
    elif icc is True and disc is True:
        df_summary = df_summary.sort_values(
            by=["discriminability", "icc"], ascending=False
        )

    df_summary.to_csv(working_dir + "/grid_clean_" + modality + ".csv")
