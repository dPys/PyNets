import pandas as pd
import os
import re
import glob
import numpy as np
import pickle
import itertools
import warnings
import psutil
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
    SimpleInterface,
)
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from scipy import stats
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, \
    f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import linear_model, decomposition
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import OrderedDict
from operator import itemgetter
from pynets.core.utils import flatten
from sklearn.preprocessing import StandardScaler

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
               "from statsmodels.stats.outliers_influence import variance_inflation_factor"]


def get_ensembles_ase(modality, alg, base_dir):
    ensembles = list(set([os.path.basename(i).split(alg + '_')[
                              1].split('_all_nodes_rawgraph')[
                              0] + '_' +
                          os.path.basename(i).split(modality + '_')[
                              1].replace('.npy', '') for i in
                          glob.glob(
                              f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
    return ensembles


def get_ensembles_omni(modality, alg, base_dir):
    ensembles = list(set(['rsn-' +
                          os.path.basename(i).split(alg + '_')[
                              1].split('_')[1] + '_res-' +
                          os.path.basename(i).split(alg + '_')[
                              1].split('_')[0] + '_' +
                          os.path.basename(i).split(modality + '_')[
                              1].replace('.npy', '') for i in
                          glob.glob(
                              f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
    return ensembles


def get_ensembles_top(modality, thr_type):
    df_top = pd.read_csv(
        f"{base_dir}/all_subs_neat_{modality}.csv")
    df_top = df_top.dropna(subset=["id"])
    df_top['id'] = df_top['id'].str.replace('topology_auc_sub-', '')
    df_top = df_top.rename(
        columns=lambda x: re.sub("partcorr", "model-partcorr", x))
    df_top = df_top.rename(
        columns=lambda x: re.sub("_corr", "_model-corr", x))
    df_top = df_top.rename(
        columns=lambda x: re.sub("_cov", "_model-cov", x))
    df_top['participant_id'] = df_top['id'].str.replace("_ses-ses-",
                                                        "_ses-")
    df_top['participant_id'] = df_top['participant_id'].str.replace(
        ".csv", "")
    [df_top, ensembles] = graph_theory_prep(df_top, thr_type)
    ensembles = [i for i in ensembles if
                 i != 'id' and i != 'participant_id']
    return ensembles, df_top


def make_feature_space_dict(df, subject_dict, ses, base_dir):
    ml_dfs = {}
    for modality in modalities:
        print(modality)
        if modality not in ml_dfs.keys():
            ml_dfs[modality] = {}
        for alg in embedding_types:
            if alg not in ml_dfs[modality].keys():
                ml_dfs[modality][alg] = {}
            for grid_param in modality_grids[modality]:
                print(grid_param)
                # Skip any invalid combinations (e.g. the covariance of the
                # variance is singular...)
                if 'cov' in grid_param and 'variance' in grid_param:
                    continue

                # save feature space to dict
                ml_dfs[modality][alg][grid_param] = create_feature_space(df, grid_param, subject_dict, ses, modality, alg)

    dict_file_path = f"{base_dir}/pynets_ml_dict.pkl"
    with open(dict_file_path, 'wb') as f:
        pickle.dump(ml_dfs, f, protocol=2)
    f.close()

    return dict_file_path


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
    from pynets.stats.benchmarking import build_hp_dict
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

    # Remove outliers
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


def make_subject_dict(modalities, base_dir, thr_type):

    subject_dict = {}
    modality_grids = {}
    for modality in modalities:
        hyperparams = eval(f"hyperparams_{modality}")
        ids = [os.path.basename(i) + '_ses-1' for i in glob.glob(
            f"{base_dir}/embeddings_all_{modality}/*") if
               os.path.basename(i).startswith('sub')]

        for alg in embedding_types:
            if alg == 'ASE':
                ensembles = get_ensembles_ase(modality, alg, base_dir)
            elif alg == 'OMNI':
                ensembles = get_ensembles_omni(modality, alg, base_dir)
            elif alg == 'topology':
                ensembles, df_top = get_ensembles_top(modality, thr_type)

            hyperparam_dict = {}

            [hyperparam_dict, grid] = build_grid(modality, hyperparam_dict,
                                                 sorted(list(set(hyperparams))),
                                                 ensembles)

            # Since we are using all of the 3 RSN connectomes (pDMN, coSN, and fECN) in the feature-space,
            # rather than varying them as hyperparameters (i.e. we assume they each add distinct variance
            # from one another) Create an abridged grid, where
            if modality == 'func':
                modality_grids[modality] = list(
                    set([i[:-2] + tuple(i[-1]) for i in grid]))
            else:
                modality_grids[modality] = list(set([i[:-1] for i in grid]))

            for id in ids:
                ID = id.split("_")[0].split("sub-")[1]
                ses = id.split("_")[1].split("ses-")[1]

                if ID not in subject_dict.keys():
                    subject_dict[ID] = {}

                if ses not in subject_dict[ID].keys():
                    subject_dict[ID][ses] = {}

                if modality not in subject_dict[ID][ses].keys():
                    subject_dict[ID][ses][modality] = {}

                subject_dict[ID][ses][modality] = dict.fromkeys(grid, np.nan)

                # Functional case
                if modality == 'func':
                    for comb in grid:
                        extract, hpass, model, res, atlas, smooth = comb
                        comb_tuple = (atlas, extract, hpass, model, res, smooth)
                        subject_dict[ID][ses][modality][comb_tuple] = {}
                        if alg == 'ASE' or alg == 'OMNI':
                            if smooth == 0:
                                embeddings = [i for i in glob.glob(f"{base_dir}/embeddings_all_"
                                          f"{modality}/sub-{ID}/rsn-{atlas}_res-{res}/"
                                          f"gradient*{alg}*{res}*{atlas}*{ID}"
                                          f"*modality-{modality}*model-{model}*template-{template}*hpass-{hpass}Hz*extract-{extract}.npy") if 'smooth' not in i]
                            else:
                                embeddings = [i for i in glob.glob(f"{base_dir}/embeddings_all_"
                                          f"{modality}/sub-{ID}/rsn-{atlas}_res-{res}/"
                                          f"gradient*{alg}*{res}*{atlas}*{ID}"
                                          f"*modality-{modality}*model-{model}*template-{template}*hpass-{hpass}Hz*extract-{extract}.npy") if f"smooth-{smooth}fwhm" in i]
                            if len(embeddings) == 0:
                                print(
                                    f"No functional embeddings found for {id} and"
                                    f" recipe {comb_tuple}...")
                                continue
                            elif len(embeddings) == 1:
                                embedding = embeddings[0]
                            else:
                                print(
                                    f"Too many structural embeddings found for {id} and"
                                    f" recipe {comb_tuple}:\n{embeddings}")
                                embedding = \
                                    sorted(embeddings, key=os.path.getmtime)[0]

                            if os.path.isfile(embedding):
                                #print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                                data = np.load(embedding)
                                coords, labels = get_coords_labels(embedding)
                                if alg not in subject_dict[ID][ses][modality][comb_tuple].keys():
                                    subject_dict[ID][ses][modality][comb_tuple][alg] = {}
                                subject_dict[ID][ses][modality][comb_tuple][alg]['coords'] = coords
                                subject_dict[ID][ses][modality][comb_tuple][alg]['labels'] = labels
                                subject_dict[ID][ses][modality][comb_tuple][alg]['data'] = data
                            else:
                                print(
                                    f"Functional embedding not found for {id} and"
                                    f" recipe {comb_tuple}...")
                                continue
                        elif alg == 'topology':
                            data = np.empty([len(mets), 1], dtype=np.float32)
                            data[:] = np.nan
                            i = 0
                            for met in mets:
                                col = (
                                    'rsn-'
                                    + atlas
                                    + "_res-"
                                    + res
                                    + "_model-"
                                    + model
                                    + f"_template-{template}_nodetype-parc_"
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
                                if col in df_top.columns:
                                    try:
                                        data[i] = df_top[df_top[
                                                             "participant_id"]
                                                         == "sub-" + ID + "_ses-"
                                                         + ses][
                                            col].values[0]
                                    except BaseException:
                                        data[i] = np.nan
                                else:
                                    data[i] = np.nan
                                del col
                                i += 1
                            subject_dict[ID][ses][modality][comb_tuple][alg] = data

                # Structural case
                elif modality == 'dwi':
                    for comb in grid:
                        directget, minlength, model, res, atlas = comb
                        comb_tuple = (atlas, directget, minlength, model, res)
                        subject_dict[ID][ses][modality][comb_tuple] = {}
                        if alg == 'ASE' or alg == 'OMNI':
                            embeddings = glob.glob(f"{base_dir}/embeddings_all"
                                        f"_{modality}/sub-{ID}/rsn-{atlas}_res-{res}/"
                                        f"gradient*{alg}*{res}*{atlas}*{ID}"
                                        f"*modality-{modality}*model-{model}*template-{template}*directget-{directget}"
                                        f"*minlength-{minlength}*.npy")
                            if len(embeddings) == 0:
                                print(
                                    f"No functional embeddings found for {id} and"
                                    f" recipe {comb_tuple}...")
                                continue
                            elif len(embeddings) == 1:
                                embedding = embeddings[0]
                            else:
                                print(
                                    f"Too many structural embeddings found for {id} and"
                                    f" recipe {comb_tuple}:\n{embeddings}")
                                embedding = \
                                sorted(embeddings, key=os.path.getmtime)[0]
                            if os.path.isfile(embedding):
                                #print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                                data = np.load(embedding)
                                coords, labels = get_coords_labels(embedding)
                                if alg not in subject_dict[ID][ses][modality][comb_tuple].keys():
                                    subject_dict[ID][ses][modality][comb_tuple][alg] = {}
                                subject_dict[ID][ses][modality][comb_tuple][alg]['coords'] = coords
                                subject_dict[ID][ses][modality][comb_tuple][alg]['labels'] = labels
                                subject_dict[ID][ses][modality][comb_tuple][alg]['data'] = data
                            else:
                                print(
                                    f"Structural embedding not found for {id} and"
                                    f" recipe {comb_tuple}...")
                                continue
                        elif alg == 'topology':
                            data = np.empty([len(mets), 1], dtype=np.float32)
                            data[:] = np.nan
                            i = 0
                            for met in mets:
                                col = (
                                    'rsn-'
                                    + atlas
                                    + "_res-"
                                    + res
                                    + "_model-"
                                    + model
                                    + f"_template-{template}_"
                                    + "_nodetype-parc_samples-20000streams"
                                      "_tracktype-"
                                    + "local"
                                    + "_directget-"
                                    + directget
                                    + "_minlength-"
                                    + minlength
                                    + "_thrtype-"
                                    + thr_type
                                    + "_topology_"
                                    + met
                                    + "_auc"
                                )
                                if col in df_top.columns:
                                    try:
                                        data[i] = df_top[df_top[
                                                             "participant_id"] ==
                                                         "sub-" + ID + "_ses-" +
                                                         ses][
                                            col].values[0]
                                    except BaseException:
                                        data[i] = np.nan
                                else:
                                    data[i] = np.nan
                                del col
                                i += 1
                            subject_dict[ID][ses][modality][comb_tuple][alg] = data
    return subject_dict, modality_grids


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


def make_x_y(input_dict, drop_cols, target_var, alg, grid_param, modality):
    import pandas as pd
    import pickle

    print(target_var)
    print(alg)
    print(grid_param)
    with open(input_dict, 'rb') as f:
        ml_dfs = pickle.load(f)
    f.close()

    if grid_param in ml_dfs[modality][alg].keys():
        df_all = ml_dfs[modality][alg][grid_param]
        if df_all is None:
            df_all = pd.Series()
    else:
        df_all = pd.Series()

    if len(df_all) < 30:
        X = None
        Y = None
        print('Feature-space NA')
    else:
        Y = df_all[target_var].values
        X = df_all.drop(columns=drop_cols)
        print(X)
    return X, Y


def concatenate_frames(out_dir, files_):
    import pandas as pd

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
    out_path = f"{out_dir}/final_df.csv"
    frame.to_csv(out_path, index=False)

    return out_path


class _MakeXYInputSpec(BaseInterfaceInputSpec):
    input_dict = traits.Str()
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
        import pandas as pd
        from nipype.utils.filemanip import fname_presuffix, copyfile

        input_dict_tmp = fname_presuffix(
            self.inputs.input_dict, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.input_dict,
            input_dict_tmp,
            copy=True,
            use_hardlink=False)

        [X, Y] = \
            make_x_y(input_dict_tmp, self.inputs.drop_cols,
                     self.inputs.target_var, self.inputs.alg,
                     tuple(self.inputs.grid_param), self.inputs.modality)

        if isinstance(X, pd.DataFrame):
            out_X = f"{runtime.cwd}/X_{self.inputs.target_var}_" \
                    f"{self.inputs.modality}_{self.inputs.alg}_" \
                    f"{'_'.join(self.inputs.grid_param)}.csv"

            X.to_csv(out_X, index=False)
        else:
            out_X = None

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

        if self.inputs.X:
            X = pd.read_csv(self.inputs.X, index_col=False)
            [grand_mean_best_estimator, grand_mean_best_Rsquared,
             grand_mean_best_MSE, mega_feat_imp_dict] = bootstrapped_nested_cv(
                X, self.inputs.y)
            print(f"Target Outcome: {self.inputs.target_var}")
            print(f"Modality: {self.inputs.modality}")
            print(f"Embedding type: {self.inputs.alg}")
            print(f"Grid Params: {self.inputs.grid_param}")
            print(f"Best Estimator: {grand_mean_best_estimator}")
            print(f"R2: {grand_mean_best_Rsquared}")
            print(f"MSE: {grand_mean_best_MSE}")
            print(
                f"Most important latent positions: "
                f"{list(mega_feat_imp_dict.keys())}")
        else:
            print('Empty feature-space!')
            grand_mean_best_estimator = dict()
            grand_mean_best_Rsquared = dict()
            grand_mean_best_MSE = dict()
            mega_feat_imp_dict = OrderedDict()

        self._results["grand_mean_best_estimator"] = grand_mean_best_estimator
        self._results["grand_mean_best_Rsquared"] = grand_mean_best_Rsquared
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
    grand_mean_best_MSE = traits.Dict()
    mega_feat_imp_dict = traits.Dict()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _MakeDFOutputSpec(TraitedSpec):

    df_summary = traits.Str()


class MakeDF(SimpleInterface):

    input_spec = _MakeDFInputSpec
    output_spec = _MakeDFOutputSpec

    def _run_interface(self, runtime):
        import gc
        import pandas as pd
        import numpy as np

        df_summary = pd.DataFrame(
            columns=["modality", "grid", "alg", "best_estimator", "Rsquared",
                     "MSE", "target_variable", "lp_importance"])

        df_summary.at[0, "target_variable"] = self.inputs.target_var
        df_summary.at[0, "modality"] = self.inputs.modality
        df_summary.at[0, "alg"] = self.inputs.alg
        df_summary.at[0, "grid"] = tuple(self.inputs.grid_param)

        if self.inputs.grand_mean_best_estimator:
            df_summary.at[0, "best_estimator"] = max(
                set(list(self.inputs.grand_mean_best_estimator.values())),
                key=list(self.inputs.grand_mean_best_estimator.values()).count)
            df_summary.at[0, "Rsquared"] = np.mean(
                list(self.inputs.grand_mean_best_Rsquared.values()))
            df_summary.at[0, "MSE"] = \
                np.mean(list(self.inputs.grand_mean_best_MSE.values()))
            df_summary.at[0, "lp_importance"] = \
                np.array(list(self.inputs.mega_feat_imp_dict.keys()))

        else:
            df_summary.at[0, "best_estimator"] = np.nan
            df_summary.at[0, "Rsquared"] = np.nan
            df_summary.at[0, "MSE"] = np.nan
            df_summary.at[0, "lp_importance"] = np.nan

        out_df_summary = f"{runtime.cwd}/df_summary_" \
                         f"{self.inputs.target_var}_" \
                f"{self.inputs.modality}_{self.inputs.alg}_" \
                f"{'_'.join(self.inputs.grid_param)}.csv"
        df_summary.to_csv(out_df_summary, index=False)

        self._results["df_summary"] = out_df_summary

        gc.collect()

        return runtime


def create_wf(base_dir, modality_grids):
    ml_wf = pe.Workflow(name="ensemble_connectometry")
    ml_wf.base_dir = f"{base_dir}/pynets_ml"

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "input_dict",
                "drop_cols",
                "out_dir"
            ]
        ),
        name="inputnode",
    )


    os.makedirs(f"{base_dir}/pynets_ml", exist_ok=True)
    inputnode.inputs.out_dir = f"{base_dir}/pynets_ml"
    inputnode.inputs.input_dict = dict_file_path
    inputnode.inputs.drop_cols = drop_cols

    make_x_y_func_node = pe.Node(
        MakeXY(),
        name="make_x_y_func_node",
        nested=True
    )

    combos = list(itertools.product(target_vars,
                                    embedding_types,
                                    [list(i) for i in
                                     modality_grids['func'][:10]]))

    x_y_iters = []
    target_vars_list = [i[0] for i in combos]
    embedding_types_list = [i[1] for i in combos]
    grid_param_list = [i[2] for i in combos]
    x_y_iters.append(("target_var", target_vars_list[1:]))
    x_y_iters.append(("alg", embedding_types_list[1:]))
    x_y_iters.append(("grid_param", grid_param_list[1:]))

    make_x_y_func_node.iterables = x_y_iters
    make_x_y_func_node.interface.n_procs = 1
    make_x_y_func_node._mem_gb = 4
    make_x_y_func_node.inputs.modality = 'func'
    make_x_y_func_node.inputs.target_var = target_vars_list[0]
    make_x_y_func_node.inputs.alg = embedding_types_list[0]
    make_x_y_func_node.inputs.grid_param = grid_param_list[0]

    bootstrapped_nested_cv_node = pe.Node(
        BSNestedCV(),
        name="bootstrapped_nested_cv_node",
        nested=True
    )

    bootstrapped_nested_cv_node.interface.n_procs = 1
    bootstrapped_nested_cv_node.interface._mem_gb = 1


    make_df_node = pe.Node(
        MakeDF(),
        name="make_df_node"
    )

    make_df_node.interface.n_procs = 1
    make_df_node.interface._mem_gb = 1

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["df_summary"]),
        name="df_join_node",
        joinfield=["df_summary"],
        joinsource=make_x_y_func_node,
    )

    concatenate_frames_node = pe.Node(
        niu.Function(
            input_names=["out_dir", "files_"],
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
                [("input_dict", "input_dict"), ("drop_cols", "drop_cols")]
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
                 ("grand_mean_best_MSE", "grand_mean_best_MSE"),
                 ("mega_feat_imp_dict", "mega_feat_imp_dict"),
                 ("target_var", "target_var"), ("modality", "modality"),
                 ("alg", "alg"), ("grid_param", "grid_param")]
            ),
            (
                make_df_node, df_join_node,
                [("df_summary", "df_summary")]
            ),
            (
                df_join_node, concatenate_frames_node,
                [("df_summary", "files_")]
            ),
            (
                inputnode, concatenate_frames_node,
                [("out_dir", "out_dir")]
            ),
            (
                concatenate_frames_node, outputnode,
                [("out_path", "df_summary")]
            )
        ]
    )
    return ml_wf


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"

    base_dir = '/working/tuning_set/outputs_shaeffer'
    df = pd.read_csv(
        '/working/tuning_set/outputs_shaeffer/df_rum_persist_all.csv',
        index_col=False)

    # target_vars = ['rum_persist', 'dep_1', 'age']
    target_vars = ['rum_persist']
    thr_type = 'MST'
    drop_cols = ['rum_persist', 'dep_1', 'age', 'sex']
    # embedding_types = ['OMNI', 'ASE']
    embedding_types = ['OMNI']
    modalities = ['func', 'dwi']
    template = 'MNI152_T1'
    mets = ["global_efficiency", "average_clustering",
            "average_shortest_path_length", "average_betweenness_centrality",
            "average_eigenvector_centrality", "average_degree_centrality",
            "average_diversity_coefficient",
            "average_participation_coefficient"]
    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength']

    ses = 1

    subject_dict, modality_grids = make_subject_dict(modalities, base_dir, thr_type)
    sub_dict_clean = cleanNullTerms(subject_dict)

    # Subset only those participants which have usable data
    df = df[df['participant_id'].isin(list(subject_dict.keys()))]
    df = df[['participant_id', 'rum_persist', 'dep_1', 'age', 'sex']]

    dict_file_path = make_feature_space_dict(df, sub_dict_clean, ses, base_dir)

    ml_wf = create_wf(base_dir, modality_grids)

    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 0.1
    execution_dict["crashfile_format"] = 'txt'
    execution_dict['local_hash_check'] = False
    execution_dict['hash_method'] = 'timestamp'

    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            ml_wf.config[key][setting] = value

    nthreads = psutil.cpu_count(logical=False)
    procmem = [int(nthreads),
               int(list(psutil.virtual_memory())[4]/1000000000) - 2]
    plugin_args = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "mem_thread",
    }
    out = ml_wf.run(plugin='MultiProc', plugin_args=plugin_args)

