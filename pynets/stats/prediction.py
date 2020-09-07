#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017
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
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
    SimpleInterface,
)
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, \
    f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import linear_model, decomposition
from collections import OrderedDict
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from pynets.core.utils import flatten

warnings.simplefilter("ignore")

import_list = ["import pandas as pd",
               "import os",
               "import re",
               "import glob",
               "import numpy as np",
               "from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate",
               "from sklearn.dummy import DummyRegressor",
               "from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression",
               "from sklearn.pipeline import Pipeline",
               "from sklearn.impute import SimpleImputer",
               "from sklearn.preprocessing import StandardScaler",
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
               "from sklearn.impute import KNNImputer", "from pynets.core.utils import flatten", "import pickle", "import dill"]


def get_ensembles_embedding(modality, alg, base_dir):
    if alg == 'OMNI':
        ensembles = list(set(['rsn-' +
                              os.path.basename(i).split(alg + '_')[
                                  1].split('_')[1] + '_res-' +
                              os.path.basename(i).split(alg + '_')[
                                  1].split('_')[0] + '_' +
                              os.path.basename(i).split(modality + '_')[
                                  1].replace('.npy', '') for i in
                              glob.glob(
                                  f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
        if len(ensembles) == 0:
            ensembles = list(set(['rsn-' +
                                  os.path.basename(i).split(alg + '_')[
                                      1].split('_')[1] + '_res-' +
                                  os.path.basename(i).split(alg + '_')[
                                      1].split('_')[0] + '_' +
                                  os.path.basename(i).split(modality + '_')[
                                      1].replace('.npy', '') for i in
                                  glob.glob(
                                      f"{base_dir}/embeddings_all_{modality}/*/*/*/*{alg}*.npy")]))
    elif alg == 'ASE':
        ensembles = list(set([os.path.basename(i).split(alg + '_')[
                                  1].split('_rawgraph')[
                                  0] + '_' +
                              os.path.basename(i).split(modality + '_')[
                                  1].replace('.npy', '') for i in
                              glob.glob(
                                  f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
        if len(ensembles) == 0:
            ensembles = list(set([os.path.basename(i).split(alg + '_')[
                                      1].split('_rawgraph')[
                                      0] + '_' +
                                  os.path.basename(i).split(modality + '_')[
                                      1].replace('.npy', '') for i in
                                  glob.glob(
                                      f"{base_dir}/embeddings_all_{modality}/*/*/*/*{alg}*.npy")]))
        ensembles = ['rsn-triple_res-' + i.replace('triple_', '') for i in ensembles if 'rsn-' not in i]
    else:
        ensembles = None
    return ensembles


def get_ensembles_top(modality, thr_type, base_dir, drop_thr=0.50):
    topology_file = f"{base_dir}/all_subs_neat_{modality}.csv"
    if os.path.isfile(topology_file):
        df_top = pd.read_csv(topology_file)
        df_top = df_top.dropna(subset=["id"])
        df_top = df_top.rename(
            columns=lambda x: re.sub("_partcorr", "_model-partcorr", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_corr", "_model-corr", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_cov", "_model-cov", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_sfm", "_model-sfm", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_csa", "_model-csa", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_tensor", "_model-tensor", x))
        df_top = df_top.rename(
            columns=lambda x: re.sub("_csd", "_model-csd", x))
        # df_top = df_top.dropna(how='all')
        # df_top = df_top.dropna(axis='columns',
        #                        thresh=drop_thr * len(df_top)
        #                        )
        if not df_top.empty and len(df_top.columns) > 1:
            [df_top, ensembles] = graph_theory_prep(df_top, thr_type)
            print(df_top)
            ensembles = [i for i in ensembles if i != 'id']
        else:
            ensembles = None
            df_top = None
    else:
        ensembles = None
        df_top = None
    return ensembles, df_top


def make_feature_space_dict(ml_dfs, df, target_modality, subject_dict, ses,
                            modality_grids, target_embedding_type, mets=None):
    from joblib import Parallel, delayed
    import tempfile
    import gc
    cache_dir = tempfile.mkdtemp()

    if target_modality not in ml_dfs.keys():
        ml_dfs[target_modality] = {}
    if target_embedding_type not in ml_dfs[target_modality].keys():
        ml_dfs[target_modality][target_embedding_type] = {}
    grid_params = modality_grids[target_modality]
    par_dict = subject_dict.copy()
    with Parallel(n_jobs=112, backend='loky',
                  verbose=10,
                  temp_folder=cache_dir) as parallel:
        outs = parallel(delayed(create_feature_space)(df, grid_param, par_dict,
                                                      ses, target_modality,
                                                      target_embedding_type,
                                                      mets)
                        for grid_param in grid_params)
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
        if self.scoring not in self.scoring_dict.keys() and \
            f"{self.scoring}_score" not in self.scoring_dict.keys():
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
                       method='onese'):
            # from sklearn.model_selection._search import RazorCV

            rcv = RazorCV(cv_results, param, greater_is_complex, scoring,
                          method)
            best_idx = rcv._best_low_complexity()
            return best_idx

        return partial(razor_pass, param=param,
                       greater_is_complex=greater_is_complex, scoring=scoring)

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
    alphas = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001]
    n_comps = [5, 10, 15, 20, 25]

    # def kernel(X, Y):
    #     """
    #     3-Dimensional Kernel:
    #
    #                  (2  0)
    #     k(X, Y) = X  (    ) Y.T
    #                  (0  1)
    #     """
    #     M = np.array([[2, 0], [0, 1.0]])
    #
    #     np.array([[1, 9], [0, 1.0]])
    #     return np.dot(np.dot(X, M), Y.T)

    # Instantiate a working dictionary of performance within a bootstrap
    means_all_exp_var = {}
    means_all_MSE = {}

    # Model + feature selection by iterating grid-search across linear
    # regressors
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
            refit = RazorCV.standard_error('n_components', True, refit_score)
        else:
            # <25 Features, don't perform feature selection, but produce a
            # userwarning
            if X.shape[1] < 25:
                pipe = Pipeline([
                    (regressor_name, regressor),
                ])
                param_grid = {regressor_name + '__alpha': alphas}
                refit = refit_score
            else:
                pipe = Pipeline([
                    ('feature_select', SelectKBest(f_regression)),
                    (regressor_name, regressor),
                ])
                param_grid = {regressor_name + '__alpha': alphas,
                              'feature_select__k': n_comps}
                refit = RazorCV.standard_error('k', True, refit_score)

        # Establish grid-search feature/model tuning windows,
        # refit the best model using a 1 SE rule of MSE values.
        pipe_grid_cv = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
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

        # Apply PCA in the case that the # of features exceeds the number of
        # observations
        if pca_reduce is True and X.shape[0] < X.shape[1]:
            best_estimator_name = f"{regressor_name}_{pipe_grid_cv.best_estimator_.get_params()[regressor_name + '__alpha']}_{pipe_grid_cv.best_estimator_.named_steps['feature_select'].n_components}"
        else:
            if X.shape[1] < 25:
                best_estimator_name = f"{regressor_name}_{pipe_grid_cv.best_estimator_.get_params()[regressor_name + '__alpha']}"
            else:
                best_estimator_name = f"{regressor_name}_{pipe_grid_cv.best_estimator_.get_params()[regressor_name + '__alpha']}_{pipe_grid_cv.best_estimator_.named_steps['feature_select'].k}"

        means_all_exp_var[best_estimator_name] = np.nanmean(means_exp_var)
        means_all_MSE[best_estimator_name] = np.nanmean(means_MSE)

    # Get best regressor across models
    best_regressor = max(means_all_exp_var, key=means_all_exp_var.get)
    est = regressors[best_regressor.split('_')[0]]

    if pca_reduce is True and X.shape[0] < X.shape[1]:
        est.alpha = float(best_regressor.split('_')[-2])
        pca = decomposition.PCA(
            n_components=int(best_regressor.split('_')[-1]),
            whiten=True)
        reg = Pipeline(
            [('feature_select', pca), (best_regressor.split('_')[0], est)])
    else:
        if X.shape[1] < 25:
            est.alpha = float(best_regressor.split('_')[-1])
            reg = Pipeline(
                [(best_regressor.split('_')[0], est)])
        else:
            est.alpha = float(best_regressor.split('_')[-2])
            kbest = SelectKBest(f_regression,
                                k=int(best_regressor.split('_')[-1]))
            reg = Pipeline(
                [('feature_select', kbest),
                 (best_regressor.split('_')[0], est)])

    return reg, best_regressor


def build_grid(modality, hyperparam_dict, hyperparams, ensembles):
    from pynets.stats.benchmarking import build_hp_dict

    for ensemble in ensembles:
        try:
            build_hp_dict(
                ensemble,
                modality,
                hyperparam_dict,
                hyperparams)
        except:
            print(f"Failed to parse ensemble {ensemble}...")

    if 'rsn' in hyperparam_dict.keys():
        hyperparam_dict['rsn'] = [i for i in hyperparam_dict['rsn'] if
                                  'res' not in i]

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
    coords_file = f"{os.path.dirname(embedding)}/nodes/all_mni_coords.pkl"
    labels_file = f"{os.path.dirname(embedding)}/nodes/all_mni_labels.pkl"
    return coords_file, labels_file


def flatten_latent_positions(rsn, subject_dict, ID, ses, modality, grid_param, alg):
    import pickle
    if ((rsn,) + grid_param) in subject_dict[ID][str(ses)][modality]:
        if alg in subject_dict[ID][str(ses)][modality][((rsn,) + grid_param)].keys():
            rsn_dict = subject_dict[ID][str(ses)][modality][((rsn,) + grid_param)][alg]
            if not isinstance(rsn_dict['coords'], list):
                if os.path.isfile(rsn_dict['coords']):
                    with open(rsn_dict['coords'], "rb") as file_:
                        rsn_dict['coords'] = pickle.load(file_)
                    file_.close()
            if not isinstance(rsn_dict['labels'], list):
                if os.path.isfile(rsn_dict['labels']):
                    with open(rsn_dict['labels'], "rb") as file_:
                        rsn_dict['labels'] = pickle.load(file_)
                    file_.close()
            if not isinstance(rsn_dict['data'], np.ndarray):
                rsn_dict['data'] = np.load(rsn_dict['data'])
            ixs = [i[1] for i in rsn_dict['labels']]
            if len(ixs) == rsn_dict['data'].shape[0]:
                rsn_arr = rsn_dict['data'].T.reshape(1, rsn_dict['data'].T.shape[0] * rsn_dict['data'].T.shape[1])
                if rsn_dict['data'].shape[1] == 1:
                    df_lps = pd.DataFrame(rsn_arr, columns=[f"{rsn}_{i}_dim1" for i in ixs])
                elif rsn_dict['data'].shape[1] == 3:
                    df_lps = pd.DataFrame(rsn_arr, columns=[f"{rsn}_{i}_dim1" for i in ixs] + [f"{rsn}_{i}_dim2" for i in ixs] + [f"{rsn}_{i}_dim3" for i in ixs])
                else:
                    raise ValueError(f"Number of dimensions {rsn_dict['data'].shape[1]} not supported. See flatten_latent_positions function...")
                print(df_lps)
            else:
                print(f"Length of indices {len(ixs)} does not equal the "
                      f"number of rows {rsn_dict['data'].shape[0]} in the "
                      f"embedding-space for {ID} {ses} {modality} "
                      f"{((rsn,) + grid_param)}. This means that at point a"
                      f" node index was dropped from the parcellation, but "
                      f"not from the final graph...")
                df_lps = None
        else:
            df_lps = None
    else:
        df_lps = None

    return df_lps


def create_feature_space(df, grid_param, subject_dict, ses, modality, alg,
                         mets=None):
    df_tmps = []
    #rsns = ['SalVentAttnA', 'DefaultA', 'ContB']
    rsns = ['triple']
    grid_param = tuple(x for x in grid_param if x not in rsns)

    for ID in df['participant_id']:
        if len(ID) == 2:
            ID = 's0' + str(ID)
        if ID not in subject_dict.keys():
            print(f"ID: {ID} not found...")
            continue

        if str(ses) not in subject_dict[ID].keys():
            print(f"Session: {ses} not found for ID {ID}...")
            continue

        if modality not in subject_dict[ID][str(ses)].keys():
            print(f"Modality: {modality} not found for ID {ID}, ses-{ses}...")
            continue

        if alg not in subject_dict[ID][str(ses)][modality].keys():
            print(f"Modality: {modality} not found for ID {ID}, ses-{ses}, "
                  f"{alg}...")
            continue

        # rsn_frames = []
        # for rsn in rsns:
        #     if alg == 'OMNI' or alg == 'ASE':
        #         df_lps = flatten_latent_positions(rsn, subject_dict, ID, ses,
        #                                           modality, grid_param, alg)
        #     else:
        #         if ((rsn,) + grid_param) in subject_dict[ID][str(ses)][modality].keys():
        #             if alg in subject_dict[ID][str(ses)][modality][((rsn,) + grid_param)].keys():
        #                 df_lps = pd.DataFrame(subject_dict[ID][str(ses)][modality][((rsn,) + grid_param)][alg].T, columns=mets)
        #             else:
        #                 df_lps = None
        #         else:
        #             df_lps = None
        #     rsn_frames.append(df_lps)
        #     del df_lps
        #
        # rsn_frames = [i for i in rsn_frames if i is not None]
        # rsn_big_df = pd.concat(rsn_frames, axis=1)
        # del rsn_frames
        # df_tmp = df[df["participant_id"] ==
        #             ID].reset_index().drop(columns='index').join(
        #     rsn_big_df, how='right')
        # df_tmps.append(df_tmp)
        # del df_tmp, rsn_big_df

        if alg == 'OMNI' or alg == 'ASE':
            df_lps = flatten_latent_positions('triple', subject_dict, ID,
                                              ses, modality, grid_param, alg)
        else:
            if (('triple',) + grid_param) in subject_dict[ID][str(ses)][modality][alg].keys():
                df_lps = pd.DataFrame(subject_dict[ID][str(ses)][modality][alg][(('triple',) + grid_param)].T, columns=mets)
            else:
                df_lps = None

        if df_lps is not None:
            df_tmp = df[df["participant_id"] ==
                        ID].reset_index().drop(columns='index').join(
                df_lps, how='right')
            df_tmps.append(df_tmp)
            del df_tmp
        else:
            continue

    if len(df_tmps) > 0:
        dfs = [dff.set_index('participant_id') for dff in df_tmps]
        df_all = pd.concat(dfs, axis=0)
        df_all = df_all.replace({0: np.nan})
        del df_tmps
        return df_all, grid_param
    else:
        return None, grid_param


def graph_theory_prep(df, thr_type):
    #from sklearn.impute import KNNImputer

    cols = [
        j
        for j in set(
            [i.split("_thrtype-" + thr_type + "_")[0] for i in
             list(set(df.columns))]
        )
        if j != "id"
    ]

    id_col = df['id']

    #scaler = StandardScaler()
    #imp = KNNImputer(n_neighbors=5)
    imp = SimpleImputer()
    df = pd.DataFrame(imp.fit_transform(
        df[[i for i in df.columns if i != "id"]]),
                      columns=[i for i in df.columns if i != "id"])

    # df = pd.DataFrame(scaler.fit_transform(df[[i for
    #                                            i in df.columns if
    #                                            i != "id"]]),
    #                   columns=[i for i in df.columns if i != "id"])

    df = pd.concat([id_col, df], axis=1)

    return df, cols


def bootstrapped_nested_cv(X, y, n_boots=10, var_thr=.8, k_folds=5,
                           pca_reduce=True, remove_multi=False, std_dev=3):

    # Instantiate a working dictionary of performance across bootstraps
    grand_mean_best_estimator = {}
    grand_mean_best_Rsquared = {}
    grand_mean_best_MSE = {}
    grand_mean_y_predicted = {}

    # Remove columns with > 20% missing values
    X = X.dropna(thresh=len(X) * .80, axis=1)
    if X.empty:
        return grand_mean_best_estimator, grand_mean_best_Rsquared, \
               grand_mean_best_MSE, {}, grand_mean_y_predicted

    # Remove low-variance columns
    sel = VarianceThreshold(threshold=(var_thr * (1 - var_thr)))
    sel.fit(X)
    X = X[X.columns[sel.get_support(indices=True)]]
    if X.empty:
        return grand_mean_best_estimator, grand_mean_best_Rsquared, \
               grand_mean_best_MSE, {}, grand_mean_y_predicted

    # Apply a simple imputer (note that this assumes extreme cases of
    # missingness have already been addressed)
    imp = SimpleImputer()
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Remove outliers
    outlier_mask = (np.abs(stats.zscore(X)) < float(std_dev)).all(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]
    if X.empty:
        return grand_mean_best_estimator, grand_mean_best_Rsquared, \
               grand_mean_best_MSE, {}, grand_mean_y_predicted

    # Standardize X
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Remove multicollinear columns
    if remove_multi is True:
        ## X = X.loc[:, X.columns.str.endswith('dim3')]

        # Create correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        if X.shape[1] < 25:
            alpha = 0.99
        else:
            alpha = 0.95

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if
                   any(upper[column] > alpha)]
        try:
            X = X.drop(X[to_drop], axis=1)
        except:
            pass

    if X.shape[1] < 25:
        print(f"Low feature count: {X.shape[1]}")
        print(f"\nX: {X}\ny: {y}\n")

    # Standardize Y
    scaler = StandardScaler()
    y = pd.DataFrame(scaler.fit_transform(y.reshape(-1, 1)))

    # Bootstrap nested CV's "simulates" the variability of incoming data,
    # particularly when training on smaller datasets
    feature_imp_dicts = []
    best_positions_list = []
    for boot in range(0, n_boots):
        # Instantiate a dictionary of regressors
        regressors = {'l1': linear_model.Lasso(random_state=boot,
                                               fit_intercept=True,
                                               warm_start=True),
                      'l2': linear_model.Ridge(random_state=boot,
                                               fit_intercept=True)}

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
                if X.shape[1] < 25:
                    best_positions = list(X.columns)
                else:
                    best_positions = [column[0] for
                                        column in
                                      zip(X.columns, fitted.named_steps[
                                          'feature_select'].get_support(
                                          indices=True)) if column[1]]

                coefs = np.abs(fitted.named_steps[
                                   best_regressor.split('_')[0]].coef_)

                feat_imp_dict = OrderedDict(sorted(dict(zip(best_positions,
                                                            coefs)).items(),
                                                   key=itemgetter(1),
                                                   reverse=True))
            feature_imp_dicts.append(feat_imp_dict)
            best_positions_list.append(best_positions)

        drop_cols = [i for i in X.columns if i=='rum_1' or i=='dep_1' or
                     i=='age' or i=='sex']
        final_est.fit(X.drop(columns=drop_cols), y)
        # Save the mean CV scores for this bootstrapped iteration
        grand_mean_best_estimator[boot] = best_regressor
        grand_mean_best_Rsquared[boot] = np.nanmean(
            prediction['test_r2'])
        grand_mean_best_MSE[boot] = -np.nanmean(
            prediction['test_neg_root_mean_squared_error'])
        grand_mean_y_predicted[boot] = \
            final_est.predict(X.drop(columns=drop_cols))
        del final_est

    unq_best_positions = list(flatten(list(np.unique(best_positions_list))))

    mega_feat_imp_dict = dict.fromkeys(unq_best_positions)

    for feat in unq_best_positions:
        running_mean = []
        for ref in feature_imp_dicts:
            if feat in ref.keys():
                running_mean.append(ref[feat])
        mega_feat_imp_dict[feat] = np.nanmean(list(flatten(running_mean)))

    mega_feat_imp_dict = OrderedDict(sorted(mega_feat_imp_dict.items(),
                                            key=itemgetter(1), reverse=True))

    del X, y, scaler

    return grand_mean_best_estimator, grand_mean_best_Rsquared, \
           grand_mean_best_MSE, mega_feat_imp_dict, grand_mean_y_predicted


def make_subject_dict(modalities, base_dir, thr_type, mets, embedding_types,
                      template, sessions):
    from joblib import Parallel, delayed
    import tempfile
    import gc

    #rsns = ['SalVentAttnA', 'DefaultA', 'ContB']
    rsns = ['triple']
    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength', 'tol']

    def mergedicts(dict1, dict2):
        for k in set(dict1.keys()).union(dict2.keys()):
            if k in dict1 and k in dict2:
                if isinstance(dict1[k], dict) and \
                    isinstance(dict2[k], dict):
                    yield (k, dict(mergedicts(dict1[k],
                                              dict2[k])))
                else:
                    yield (k, dict2[k])
            elif k in dict1:
                yield (k, dict1[k])
            else:
                yield (k, dict2[k])

    subject_dict_all = {}
    modality_grids = {}
    for modality in modalities:
        print(f"MODALITY: {modality}")
        hyperparams = eval(f"hyperparams_{modality}")
        for alg in embedding_types:
            print(f"EMBEDDING TYPE: {alg}")
            for ses_name in sessions:
                if alg == 'ASE' or alg == 'OMNI':
                    ids = [f"{os.path.basename(i)}_ses-{ses_name}" for i in
                           glob.glob(
                        f"{base_dir}/embeddings_all_{modality}/*") if
                           os.path.basename(i).startswith('sub')]
                else:
                    ids = [f"{os.path.basename(i)}_ses-{ses_name}" for i in
                           glob.glob(
                        f"{base_dir}/pynets/*") if
                           os.path.basename(i).startswith('sub')]

                if alg == 'ASE' or alg == 'OMNI':
                    ensembles = get_ensembles_embedding(modality, alg,
                                                        base_dir)
                    df_top = None
                    if ensembles is None:
                        print('No ensembles found.')
                        continue
                elif alg == 'topology':
                    ensembles, df_top = get_ensembles_top(modality, thr_type,
                                                          f"{base_dir}/pynets")
                    if 'missing' in df_top.columns:
                        df_top.drop(columns='missing', inplace=True)

                    if ensembles is None or df_top is None:
                        print('Missing topology outputs.')
                        continue
                else:
                    ensembles = None
                    print('No ensembles specified.')
                    continue

                hyperparam_dict = {}

                grid = build_grid(modality, hyperparam_dict,
                                  sorted(list(set(hyperparams))),
                                  ensembles)[1]

                grid = list(set([i for i in grid if i != () and
                                 len(list(i)) > 0]))

                grid_mod = list(set([tuple(x for x in i if x not in rsns)
                                     for i in grid]))

                # Since we are using all of the 3 RSN connectomes (pDMN, coSN,
                # and fECN) in the feature-space,
                # rather than varying them as hyperparameters (i.e. we assume
                # they each add distinct variance
                # from one another) Create an abridged grid, where
                if modality == 'func':
                    modality_grids[modality] = grid_mod
                else:
                    modality_grids[modality] = grid_mod

                par_dict = subject_dict_all.copy()
                cache_dir = tempfile.mkdtemp()

                with Parallel(n_jobs=112, backend='multiprocessing',
                              verbose=10, max_nbytes=None,
                              temp_folder=cache_dir) as parallel:
                    outs = parallel(delayed(populate_subject_dict)(id, modality, grid,
                                                           par_dict, alg,
                                                           base_dir, template,
                                                           thr_type, mets,
                                                           df_top) for
                            id in ids)
                # for id in ids:
                #     subject_dict = populate_subject_dict(id, modality, grid,
                #                                    subject_dict, alg, base_dir,
                #                                    template, thr_type, mets,
                #                                    df_top)

                for d in outs:
                    subject_dict_all = dict(mergedicts(subject_dict_all, d))
                del outs, df_top
                gc.collect()
            del ses_name, grid, grid_mod, hyperparam_dict
            gc.collect()
        del alg, hyperparams
        gc.collect()
    del modality
    gc.collect()

    return subject_dict_all, modality_grids


def populate_subject_dict(id, modality, grid, subject_dict, alg, base_dir,
                          template, thr_type, mets=None, df_top=None):
    from pynets.core.utils import filter_cols_from_targets

    # print(id)
    ID = id.split("_")[0].split("sub-")[1]
    ses = id.split("_")[1].split("ses-")[1]
    print(f"ID: {ID}, SESSION: {ses}")

    if ID not in subject_dict.keys():
        subject_dict[ID] = {}

    if ses not in subject_dict[ID].keys():
        subject_dict[ID][ses] = {}

    if modality not in subject_dict[ID][ses].keys():
        subject_dict[ID][ses][modality] = {}

    if alg not in subject_dict[ID][ses][modality].keys():
        subject_dict[ID][ses][modality][alg] = {}

    subject_dict[ID][ses][modality][alg] = dict.fromkeys(grid, np.nan)

    # Functional case
    if modality == 'func':
        for comb in grid:
            try:
                extract, hpass, model, res, atlas, smooth = comb
            except:
                try:
                    extract, hpass, model, res, atlas = comb
                    smooth = 0
                except:
                    raise ValueError(f"Failed to parse recipe: {comb}")
            comb_tuple = (atlas, extract, hpass, model, res, smooth)
            print(comb_tuple)
            subject_dict[ID][ses][modality][alg][comb_tuple] = {}
            if alg == 'ASE' or alg == 'OMNI':
                if smooth == 0 or smooth is None:
                    embeddings = [i for i in
                                  glob.glob(f"{base_dir}/embeddings_all_"
                                            f"{modality}/sub-{ID}/rsn-"
                                            f"{atlas}_res-{res}/"
                                            f"gradient*{alg}*{res}*"
                                            f"{atlas}*{ID}"
                                            f"*modality-{modality}*model-"
                                            f"{model}*template-{template}*"
                                            f"hpass-{hpass}Hz*extract-"
                                            f"{extract}.npy")
                                  if 'smooth' not in i]
                else:
                    embeddings = [i for i in
                                  glob.glob(f"{base_dir}/embeddings_all_"
                                            f"{modality}/sub-{ID}/rsn-{atlas}"
                                            f"_res-{res}/"
                                            f"gradient*{alg}*{res}*{atlas}"
                                            f"*{ID}*modality-{modality}*model-"
                                            f"{model}*template-{template}*"
                                            f"hpass-{hpass}Hz*extract-"
                                            f"{extract}.npy")
                                  if f"smooth-{smooth}fwhm" in i]
                if len(embeddings) == 0:
                    if smooth == 0 or smooth is None:
                        embeddings = [i for i in
                                      glob.glob(f"{base_dir}/embeddings_all_"
                                                f"{modality}/sub-{ID}/rsn-"
                                                f"{atlas}_res-{res}/"
                                                f"gradient*{alg}*{res}*"
                                                f"{atlas}*{ID}"
                                                f"*modality-{modality}*model-"
                                                f"{model}*template-{template}*"
                                                f"hpass-{hpass}Hz*extract-"
                                                f"{extract}.npy")
                                      if 'smooth' not in i]
                    else:
                        embeddings = [i for i in
                                      glob.glob(f"{base_dir}/embeddings_all_"
                                                f"{modality}/sub-{ID}/rsn-{atlas}"
                                                f"_res-{res}/"
                                                f"gradient*{alg}*{res}*{atlas}"
                                                f"*{ID}*modality-{modality}*model-"
                                                f"{model}*template-{template}*"
                                                f"hpass-{hpass}Hz*extract-"
                                                f"{extract}.npy")
                                      if f"smooth-{smooth}fwhm" in i]
                if len(embeddings) == 0:
                    print(
                        f"\nNo functional embeddings found for {id} and"
                        f" recipe {comb_tuple}...")
                    continue
                elif len(embeddings) == 1:
                    embedding = embeddings[0]
                else:
                    sorted_embeddings = sorted(embeddings,
                                               key=os.path.getmtime)
                    if comb_tuple[-1] == 0:
                        col = [i for i in sorted_embeddings if 'fwhm' not
                               in i][0]
                    else:
                        print(
                            f"Multiple functional embeddings found for {id} and"
                            f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                            f" recent...")
                        embedding = sorted_embeddings[0]

                if os.path.isfile(embedding):
                    # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                    #data = np.load(embedding)
                    coords, labels = get_coords_labels(embedding)
                    if alg not in subject_dict[ID][ses][modality][alg][
                        comb_tuple].keys():
                        subject_dict[ID][ses][modality][alg][comb_tuple] = {}
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'coords'] = coords
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'labels'] = labels
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'data'] = embedding
                    # print(data)
                else:
                    print(
                        f"\nFunctional embedding not found for {id} and"
                        f" recipe {comb_tuple}...")
                    continue
            elif alg == 'topology':
                data = np.empty([len(mets), 1], dtype=np.float32)
                data[:] = np.nan
                if smooth == 0:
                    targets = [f"extract-{extract}",
                               f"hpass-{hpass}Hz",
                               f"model-{model}", f"res-{res}",
                               f"rsn-{atlas}", f"thrtype-{thr_type}"]
                else:
                    targets = [f"extract-{extract}",
                               f"hpass-{hpass}Hz",
                               f"model-{model}", f"res-{res}",
                               f"rsn-{atlas}",
                               f"smooth-{smooth}fwhm", f"thrtype-{thr_type}"]

                cols = filter_cols_from_targets(df_top, targets)

                i = 0
                for met in mets:
                    col_met = [j for j in cols if met in j]
                    if len(col_met) == 1:
                        col = col_met[0]
                    elif len(col_met) > 1:
                        if comb_tuple[-1] == 0:
                            col = [i for i in col_met if 'fwhm' not in i][0]
                        else:
                            print(f"Multiple columns detected: {col_met}")
                            col = col_met[0]
                    else:
                        data[i] = np.nan
                        i += 1
                        continue

                    out = df_top[df_top[
                                         "id"]
                                     == "sub-" + ID + "_ses-"
                                     + ses][
                        col].values
                    if len(out) == 0:
                        print(
                            f"\nFunctional topology not found for {id}, {met}, "
                            f"and recipe {comb_tuple}...")
                        data[i] = np.nan
                    else:
                        data[i] = out

                    del col, out
                    i += 1
                if (np.abs(data) < 0.0001).all():
                    data[:] = np.nan
                subject_dict[ID][ses][modality][alg][comb_tuple] = data
                print(data)
            del comb, comb_tuple
    # Structural case
    elif modality == 'dwi':
        for comb in grid:
            try:
                directget, minlength, model, res, atlas, tol = comb
            except:
                print(f"Failed to parse recipe: {comb}")
                continue
            comb_tuple = (atlas, directget, minlength, model, res, tol)
            print(comb_tuple)
            subject_dict[ID][ses][modality][alg][comb_tuple] = {}
            if alg == 'ASE' or alg == 'OMNI':
                embeddings = glob.glob(f"{base_dir}/embeddings_all"
                                       f"_{modality}/sub-{ID}/rsn-{atlas}_"
                                       f"res-{res}/"
                                       f"gradient*{alg}*{res}*{atlas}*{ID}"
                                       f"*modality-{modality}*model-{model}"
                                       f"*template-{template}*directget-"
                                       f"{directget}"
                                       f"*minlength-{minlength}*tol-{tol}.npy")
                if len(embeddings) == 0:
                    embeddings = glob.glob(f"{base_dir}/embeddings_all"
                                           f"_{modality}/sub-{ID}/ses-{ses}/rsn-{atlas}_"
                                           f"res-{res}/"
                                           f"gradient*{alg}*{res}*{atlas}*{ID}"
                                           f"*modality-{modality}*model-{model}"
                                           f"*template-{template}*directget-"
                                           f"{directget}"
                                           f"*minlength-{minlength}*tol-{tol}.npy")
                if len(embeddings) == 0:
                    print(
                        f"\nNo structural embeddings found for {id} and"
                        f" recipe {comb_tuple}...")
                    continue
                elif len(embeddings) == 1:
                    embedding = embeddings[0]
                else:
                    print(
                        f"\nMultiple structural embeddings found for {id} and"
                        f" recipe {comb_tuple}:\n{embeddings}\nTaking the most"
                        f" recent...")
                    embedding = \
                        sorted(embeddings, key=os.path.getmtime)[0]
                if os.path.isfile(embedding):
                    # print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                    #data = np.load(embedding)
                    coords, labels = get_coords_labels(embedding)
                    if alg not in subject_dict[ID][ses][modality][alg][
                        comb_tuple].keys():
                        subject_dict[ID][ses][modality][alg][comb_tuple] = {}
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'coords'] = coords
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'labels'] = labels
                    subject_dict[ID][ses][modality][alg][comb_tuple][
                        'data'] = embedding
                    # print(data)
                else:
                    print(
                        f"\nStructural embedding not found for {id} and"
                        f" recipe {comb_tuple}...")
                    continue
            elif alg == 'topology':
                data = np.empty([len(mets), 1], dtype=np.float32)
                data[:] = np.nan
                targets = [f"minlength-{minlength}",
                           f"directget-{directget}",
                           f"model-{model}", f"res-{res}",
                           f"rsn-{atlas}", f"tol-{tol}",
                           f"thrtype-{thr_type}"]

                cols = filter_cols_from_targets(df_top,
                                                targets)
                i = 0
                for met in mets:
                    col_met = [j for j in cols if met in j]
                    if len(col_met) == 1:
                        col = col_met[0]
                    elif len(col_met) > 1:
                        print(f"\nMultiple columns detected: {col_met}")
                        col = col_met[0]
                    else:
                        print(
                            f"\nStructural topology not found for {id}, "
                            f"{met}, and recipe {comb_tuple}...")
                        data[i] = np.nan
                        i += 1
                        continue

                    out = df_top[df_top[
                                         "id"]
                                     == "sub-" + ID + "_ses-"
                                     + ses][
                        col].values
                    if len(out) == 0:
                        print(
                            f"\nStructural topology not found for {id}, "
                            f"{met}, and recipe {comb_tuple}...")
                        data[i] = np.nan
                    else:
                        data[i] = out

                    del col, out
                    i += 1
                if (np.abs(data) < 0.0001).all():
                    data[:] = np.nan
                subject_dict[ID][ses][modality][alg][comb_tuple] = data
                print(data)
            del comb, comb_tuple
    del modality, ID, ses

    return subject_dict


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


def make_x_y(input_dict, drop_cols, target_var, alg, grid_param):
    import pandas as pd
    from time import sleep
    import json

    print(target_var)
    print(alg)
    print(grid_param)
    while not os.path.isfile(input_dict):
        sleep(1)

    with open(input_dict) as data_file:
        data_loaded = json.load(data_file)
    data_file.close()

    if str(grid_param) in data_loaded.keys():
        df_all = pd.read_json(data_loaded[str(grid_param)])

        if df_all is None:
            df_all = pd.Series()
        else:
            try:
                df_all = df_all.loc[:, ~df_all.columns.duplicated()]
                df_all.reset_index(level=0, inplace=True)
                df_all.rename(columns={'index': 'id'}, inplace=True)
                if all(df_all.drop(columns=['id', 'age', 'rum_1', 'dep_1', 'rum_persist']).isnull().all()) or len(df_all.columns) == 1 or (np.abs(np.array(df_all.drop(columns=['id', 'age', 'rum_1', 'dep_1', 'rum_persist']))) < 0.0001).all():
                    df_all = pd.Series()
                else:
                    df_all.drop(columns=['id'], inplace=True)
            except:
                df_all = pd.Series()
    else:
        df_all = pd.Series()

    if len(df_all) < 30:
        X = None
        Y = None
        print('\nFeature-space NA\n')
    else:
        Y = df_all[target_var].values
        X = df_all.drop(columns=drop_cols)
    return X, Y


def concatenate_frames(out_dir, modality, alg, target_var, files_):
    import pandas as pd
    import os
    dfs = []
    for file_ in files_:
        df = pd.read_csv(file_, chunksize=100000).read()
        try:
            df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
        except BaseException:
            pass
        dfs.append(df)
    frame = pd.concat(dfs, axis=0, join="outer", sort=True,
                      ignore_index=False)
    out_path = f"{out_dir}/final_df_{modality[0]}_{alg[0]}_{target_var}.csv"
    print(f"Saving to {out_path}...")
    if os.path.isfile(out_path):
        os.remove(out_path)
    frame.to_csv(out_path, index=False)

    return out_path


class _MakeXYInputSpec(BaseInterfaceInputSpec):
    feature_spaces = traits.Any()
    drop_cols = traits.List()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _MakeXYOutputSpec(TraitedSpec):
    X = traits.Any()
    Y = traits.Any()
    alg = traits.Str()
    modality = traits.Str()
    grid_param = traits.List()
    target_var = traits.Str()


class MakeXY(SimpleInterface):
    """Interface wrapper for MakeXY"""

    input_spec = _MakeXYInputSpec
    output_spec = _MakeXYOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import pandas as pd
        from nipype.utils.filemanip import fname_presuffix, copyfile
        import uuid
        from time import strftime
        from pynets.stats.prediction import make_x_y

        run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"

        json_dict = self.inputs.feature_spaces[
            f"{self.inputs.modality}_{self.inputs.alg}"]

        input_dict_tmp = fname_presuffix(
            json_dict, suffix=f"_tmp_{run_uuid}",
            newpath=runtime.cwd
        )
        copyfile(
            json_dict,
            input_dict_tmp,
            copy=True,
            use_hardlink=False)
        print(f"Loading {input_dict_tmp}...")

        [X, Y] = make_x_y(input_dict_tmp, self.inputs.drop_cols,
                          self.inputs.target_var, self.inputs.alg,
                          tuple(self.inputs.grid_param))

        if isinstance(X, pd.DataFrame):
            out_X = f"{runtime.cwd}/X_{self.inputs.target_var}_" \
                    f"{self.inputs.modality}_{self.inputs.alg}_" \
                    f"{'_'.join(str(v) for v in self.inputs.grid_param)}.csv"

            if os.path.isfile(out_X):
                os.remove(out_X)
            X.to_csv(out_X, index=False)
        else:
            out_X = None
            Y = None

        self._results["X"] = out_X
        self._results["Y"] = Y
        self._results["alg"] = self.inputs.alg
        self._results["grid_param"] = list(self.inputs.grid_param)
        self._results["modality"] = self.inputs.modality
        self._results["target_var"] = self.inputs.target_var

        gc.collect()

        return runtime


class _BSNestedCVInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for BSNestedCV"""
    X = traits.Any()
    y = traits.Any()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _BSNestedCVOutputSpec(TraitedSpec):
    """Output interface wrapper for BSNestedCV"""

    grand_mean_best_estimator = traits.Dict()
    grand_mean_best_Rsquared = traits.Dict()
    grand_mean_y_predicted = traits.Dict()
    grand_mean_best_MSE = traits.Dict()
    mega_feat_imp_dict = traits.Dict()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class BSNestedCV(SimpleInterface):
    """Interface wrapper for BSNestedCV"""

    input_spec = _BSNestedCVInputSpec
    output_spec = _BSNestedCVOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        from colorama import Fore, Style

        if self.inputs.X is not None:
            if os.path.isfile(self.inputs.X):
                X = pd.read_csv(
                    self.inputs.X, chunksize=100000).read()
                [grand_mean_best_estimator, grand_mean_best_Rsquared,
                 grand_mean_best_MSE, mega_feat_imp_dict,
                 grand_mean_y_predicted] = \
                    bootstrapped_nested_cv(X, self.inputs.y)
                if len(mega_feat_imp_dict) > 1:
                    print(f"\n\n{Fore.BLUE}Target Outcome: {Fore.GREEN}{self.inputs.target_var}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}Modality: {Fore.RED}{self.inputs.modality}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}Embedding type: {Fore.RED}{self.inputs.alg}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}Grid Params: {Fore.RED}{self.inputs.grid_param}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}Best Estimator: {Fore.RED}{grand_mean_best_estimator}{Style.RESET_ALL}")
                    print(f"\n{Fore.BLUE}R2: {Fore.RED}{grand_mean_best_Rsquared}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}MSE: {Fore.RED}{grand_mean_best_MSE}{Style.RESET_ALL}\n")
                    print(f"y_actual: {self.inputs.y}")
                    print(f"y_predicted: {grand_mean_y_predicted}")
                    print(f"{Fore.BLUE}Feature Importance: {Fore.RED}{list(mega_feat_imp_dict.keys())}{Style.RESET_ALL}\n\n")
                else:
                    print('Empty feature-space!')
                    mega_feat_imp_dict = OrderedDict()
            else:
                print('Feature-space .csv file not found!')
                grand_mean_best_estimator = dict()
                grand_mean_best_Rsquared = dict()
                grand_mean_y_predicted = dict()
                grand_mean_best_MSE = dict()
                mega_feat_imp_dict = OrderedDict()
        else:
            print(f"{Fore.RED}Empty feature-space!{Style.RESET_ALL}")
            grand_mean_best_estimator = dict()
            grand_mean_best_Rsquared = dict()
            grand_mean_y_predicted = dict()
            grand_mean_best_MSE = dict()
            mega_feat_imp_dict = OrderedDict()

        self._results["grand_mean_best_estimator"] = grand_mean_best_estimator
        self._results["grand_mean_best_Rsquared"] = grand_mean_best_Rsquared
        self._results["grand_mean_y_predicted"] = grand_mean_y_predicted
        self._results["grand_mean_best_MSE"] = grand_mean_best_MSE
        self._results["mega_feat_imp_dict"] = mega_feat_imp_dict
        self._results["target_var"] = self.inputs.target_var
        self._results["modality"] = self.inputs.modality
        self._results["alg"] = self.inputs.alg
        self._results["grid_param"] = list(self.inputs.grid_param)

        gc.collect()

        return runtime


class _MakeDFInputSpec(BaseInterfaceInputSpec):
    grand_mean_best_estimator = traits.Dict()
    grand_mean_best_Rsquared = traits.Dict()
    grand_mean_y_predicted = traits.Dict()
    grand_mean_best_MSE = traits.Dict()
    mega_feat_imp_dict = traits.Dict()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _MakeDFOutputSpec(TraitedSpec):
    df_summary = traits.Str()
    modality = traits.Str()
    alg = traits.Str()


class MakeDF(SimpleInterface):

    input_spec = _MakeDFInputSpec
    output_spec = _MakeDFOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import pandas as pd
        import numpy as np

        df_summary = pd.DataFrame(
            columns=["modality", "grid", "alg", "best_estimator", "Rsquared",
                     "MSE", "target_variable", "lp_importance", "Predicted_y"])

        df_summary.at[0, "target_variable"] = self.inputs.target_var
        df_summary.at[0, "modality"] = self.inputs.modality
        df_summary.at[0, "alg"] = self.inputs.alg
        df_summary.at[0, "grid"] = tuple(self.inputs.grid_param)

        y_pred_boots = self.inputs.grand_mean_y_predicted.values()
        if len(y_pred_boots) > 0:
            max_row_len = max([len(ll) for ll in y_pred_boots])
            y_pred_vals = np.nanmean(
                [[el for el in row] + [np.NaN] * max(0, max_row_len - len(row))
                 for row in y_pred_boots], axis=0)
        else:
            max_row_len = np.nan
            y_pred_boots = np.nan

        if self.inputs.grand_mean_best_estimator:
            df_summary.at[0, "best_estimator"] = max(
                set(list(self.inputs.grand_mean_best_estimator.values())),
                key=list(self.inputs.grand_mean_best_estimator.values()).count)
            df_summary.at[0, "Rsquared"] = np.mean(
                list(self.inputs.grand_mean_best_Rsquared.values()))
            df_summary.at[0, "Predicted_y"] = y_pred_vals
            df_summary.at[0, "MSE"] = \
                np.mean(list(self.inputs.grand_mean_best_MSE.values()))
            df_summary.at[0, "lp_importance"] = \
                np.array(list(self.inputs.mega_feat_imp_dict.keys()))

        else:
            df_summary.at[0, "best_estimator"] = np.nan
            df_summary.at[0, "Rsquared"] = np.nan
            df_summary.at[0, "Predicted_y"] = np.nan
            df_summary.at[0, "MSE"] = np.nan
            df_summary.at[0, "lp_importance"] = np.nan

        out_df_summary = f"{runtime.cwd}/df_summary_" \
                         f"{self.inputs.target_var}_" \
                f"{self.inputs.modality}_{self.inputs.alg}_" \
                f"{'_'.join(str(v) for v in self.inputs.grid_param)}.csv"
        if os.path.isfile(out_df_summary):
            os.remove(out_df_summary)
        df_summary.to_csv(out_df_summary, index=False)

        self._results["df_summary"] = out_df_summary
        self._results["alg"] = self.inputs.alg
        self._results["modality"] = self.inputs.modality

        gc.collect()

        return runtime


def create_wf(base_dir, feature_spaces, modality_grids, drop_cols,
              target_var, embedding_type, modality):
    import uuid
    import itertools
    from time import strftime
    from pynets.stats.prediction import MakeXY, MakeDF, BSNestedCV, \
        concatenate_frames
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    ml_wf = pe.Workflow(name="ensemble_connectometry")
    ml_wf.base_dir = f"{base_dir}/pynets_ml_{run_uuid}"

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "feature_spaces",
                "drop_cols",
                "out_dir",
                "modality",
                "target_var",
                "embedding_type"
            ]
        ),
        name="inputnode",
    )

    os.makedirs(ml_wf.base_dir, exist_ok=True)
    inputnode.inputs.out_dir = base_dir
    inputnode.inputs.feature_spaces = feature_spaces
    inputnode.inputs.drop_cols = drop_cols
    inputnode.inputs.modality = modality
    inputnode.inputs.target_var = target_var
    inputnode.inputs.embedding_type = embedding_type

    make_x_y_func_node = pe.Node(
        MakeXY(),
        name="make_x_y_func_node",
        nested=True
    )

    x_y_iters = []
    x_y_iters.append(("grid_param", [list(i) for i in
                                     modality_grids[modality]]))

    make_x_y_func_node.iterables = x_y_iters
    make_x_y_func_node.interface.n_procs = 2
    make_x_y_func_node._mem_gb = 4

    bootstrapped_nested_cv_node = pe.Node(
        BSNestedCV(),
        name="bootstrapped_nested_cv_node",
        nested=True
    )

    bootstrapped_nested_cv_node.interface.n_procs = 1
    bootstrapped_nested_cv_node.interface._mem_gb = 2

    make_df_node = pe.Node(
        MakeDF(),
        name="make_df_node"
    )

    make_df_node.interface.n_procs = 1
    make_df_node.interface._mem_gb = 1

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary", "modality", "alg"]),
        name="df_join_node",
        joinfield=["df_summary", "modality", "alg"],
        joinsource=make_x_y_func_node,
    )

    concatenate_frames_node = pe.Node(
        niu.Function(
            input_names=["out_dir", "files_", "modality", "alg", "target_var"],
            output_names=["out_path"],
            function=concatenate_frames,
        ),
        name="concatenate_frames_node",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["df_summary"]),
        name="outputnode")

    ml_wf.connect(
        [
            (
                inputnode, make_x_y_func_node,
                [("feature_spaces", "feature_spaces"),
                 ("drop_cols", "drop_cols"),
                 ("target_var", "target_var"),
                 ("modality", "modality"), ("embedding_type", "alg")]
            ),
            (
                make_x_y_func_node, bootstrapped_nested_cv_node,
                [("X", "X"), ("Y", "y"), ("target_var", "target_var"),
                 ("modality", "modality"),
                 ("alg", "alg"), ("grid_param", "grid_param")]
            ),
            (
                bootstrapped_nested_cv_node, make_df_node,
                [("grand_mean_best_estimator", "grand_mean_best_estimator"),
                 ("grand_mean_best_Rsquared", "grand_mean_best_Rsquared"),
                 ("grand_mean_y_predicted", "grand_mean_y_predicted"),
                 ("grand_mean_best_MSE", "grand_mean_best_MSE"),
                 ("mega_feat_imp_dict", "mega_feat_imp_dict"),
                 ("target_var", "target_var"), ("modality", "modality"),
                 ("alg", "alg"), ("grid_param", "grid_param")]
            ),
            (
                make_df_node, df_join_node,
                [("df_summary", "df_summary"),
                 ("modality", "modality"),
                 ("alg", "alg")]
            ),
            (
                df_join_node, concatenate_frames_node,
                [("df_summary", "files_"),
                 ("modality", "modality"),
                 ("alg", "alg")]
            ),
            (
                inputnode, concatenate_frames_node,
                [("out_dir", "out_dir"), ('target_var', 'target_var')]
            ),
            (
                concatenate_frames_node, outputnode,
                [("out_path", "df_summary")]
            )
        ]
    )
    return ml_wf


def build_predict_workflow(args, retval):
    from pynets.stats.prediction import create_wf
    import psutil

    base_dir = args['base_dir']
    feature_spaces = args['feature_spaces']
    modality_grids = args['modality_grids']
    drop_cols = args['drop_cols']
    target_var = args['target_var']
    embedding_type = args['embedding_type']
    target_modality = args['target_modality']

    ml_wf = create_wf(base_dir, feature_spaces, modality_grids, drop_cols,
                      target_var, embedding_type, target_modality)

    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 1
    execution_dict["crashfile_format"] = 'txt'
    execution_dict['local_hash_check'] = False
    execution_dict['stop_on_first_crash'] = False
    execution_dict['keep_inputs'] = True
    execution_dict['remove_unnecessary_outputs'] = False
    execution_dict['remove_node_directories'] = False
    execution_dict['raise_insufficient'] = False

    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            ml_wf.config[key][setting] = value

    nthreads = psutil.cpu_count()
    procmem = [int(nthreads),
               int(list(psutil.virtual_memory())[4]/1000000000) - 2]
    plugin_args = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "topological_sort",
    }
    print('Running workflow...')
    out = ml_wf.run(plugin='MultiProc', plugin_args=plugin_args)
    #out = ml_wf.run(plugin='Linear', plugin_args=plugin_args)
    return


def main():
    import json

    base_dir = '/working/tuning_set/outputs_shaeffer'
    df = pd.read_csv(
        '/working/tuning_set/outputs_shaeffer/df_rum_persist_all.csv',
        index_col=False)

    #embedding_types = ['topology', 'OMNI', 'ASE']
    embedding_types = ['topology']
    #embedding_types = ['OMNI']
    modalities = ['func', 'dwi']
    thr_type = 'MST'

    ###
    target_embedding_type = 'topology'
    target_modality = 'dwi'
    target_var = 'dep_1'
    ###

    if target_var == 'rum_persist' or target_var == 'dep_1':
        drop_cols = [target_var]
    else:
        drop_cols = [target_var, 'rum_persist', 'dep_1', 'rum_1']

    template = 'MNI152_T1'
    mets = ["global_efficiency",
            "average_clustering",
            "average_shortest_path_length",
            "average_local_efficiency_nodewise",
            "average_eigenvector_centrality",
            "modularity"]

    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength', 'tol']

    sessions = ['1']

    subject_dict_file_path = f"{base_dir}/pynets_subject_dict.pkl"
    subject_mod_grids_file_path = f"{base_dir}/pynets_modality_grids.pkl"

    if not os.path.isfile(subject_dict_file_path) or not os.path.isfile(
        subject_mod_grids_file_path):
        subject_dict, modality_grids = make_subject_dict(modalities, base_dir,
                                                         thr_type, mets,
                                                         embedding_types,
                                                         template, sessions)
        sub_dict_clean = cleanNullTerms(subject_dict)

        with open(subject_dict_file_path, 'wb') as f:
            dill.dump(sub_dict_clean, f)
        f.close()
        with open(subject_mod_grids_file_path, 'wb') as f:
            dill.dump(modality_grids, f)
        f.close()
    else:
        with open(subject_dict_file_path, 'rb') as f:
            sub_dict_clean = dill.load(f)
        f.close()
        with open(subject_mod_grids_file_path, 'rb') as f:
            modality_grids = dill.load(f)
        f.close()

    # Subset only those participants which have usable data
    df = df[df['participant_id'].isin(list(sub_dict_clean.keys()))]
    df = df[['participant_id', 'rum_persist', 'rum_1', 'dep_1', 'age']]

    # Remove 4 outliers subjects with excessive missing behavioral data
    df = df.loc[(df['participant_id'] != '33') &
                (df['participant_id'] != '21') &
                (df['participant_id'] != '54') &
                (df['participant_id'] != '14')]

    dict_file_path = f"{base_dir}/pynets_ml_dict_{target_modality}_" \
                     f"{target_var}_{target_embedding_type}.pkl"

    if not os.path.isfile(dict_file_path) or not os.path.isfile(
        dict_file_path):
        ml_dfs = {}
        for modality in modalities:
            for alg in embedding_types:
                ml_dfs = make_feature_space_dict(ml_dfs, df, modality,
                                                 sub_dict_clean,
                                                 sessions[0], modality_grids,
                                                 alg, mets)

        with open(dict_file_path, 'wb') as f:
            dill.dump(ml_dfs, f)
        f.close()
    else:
        with open(dict_file_path, 'rb') as f:
            ml_dfs = dill.load(f)
        f.close()

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
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=4)
        f.close()
        feature_spaces[iter] = out_json_path

    del ml_dfs

    args = {}
    args['base_dir'] = base_dir
    args['feature_spaces'] = feature_spaces
    args['modality_grids'] = modality_grids
    args['drop_cols'] = drop_cols
    args['target_var'] = target_var
    args['embedding_type'] = target_embedding_type
    args['target_modality'] = target_modality

    return args


if __name__ == "__main__":
    import warnings
    import sys
    import gc
    import json
    from multiprocessing import set_start_method, Process, Manager
    from pynets.stats.prediction import build_predict_workflow

    try:
        set_start_method('forkserver')
    except:
        pass
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
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
