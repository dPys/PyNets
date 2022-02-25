#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import warnings
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def get_optimal_cov_estimator(time_series, cv=5, max_iter=200):
    from sklearn.covariance import GraphicalLassoCV

    estimator = GraphicalLassoCV(cv=cv, assume_centered=True)
    print("\nSearching for best Lasso...\n")
    try:
        estimator.fit(time_series)
        return estimator
    except BaseException:
        ix = 0
        print("\nModel did not converge on first attempt. "
              "Varying tolerance...\n")
        while not hasattr(estimator, 'covariance_') and \
                not hasattr(estimator, 'precision_') and ix < 3:
            for t in [0.1, 0.01, 0.001, 0.0001]:
                print(f"Tolerance={t}")
                estimator = GraphicalLassoCV(cv=cv, max_iter=max_iter, tol=t,
                                             assume_centered=True)
                try:
                    estimator.fit(time_series)
                    return estimator
                except BaseException:
                    ix += 1
                    continue

    if not hasattr(estimator, 'covariance_') and not hasattr(estimator,
                                                             'precision_'):
        print(
            "Unstable Lasso estimation. Applying shrinkage to empirical "
            "covariance..."
        )
        from sklearn.covariance import (
            GraphicalLasso,
            empirical_covariance,
            shrunk_covariance,
        )
        try:
            emp_cov = empirical_covariance(time_series, assume_centered=True)
            for i in np.arange(0.8, 0.99, 0.01):
                print(f"Shrinkage={i}:")
                shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                alphaRange = 10.0 ** np.arange(-8, 0)
                for alpha in alphaRange:
                    print(f"Auto-tuning alpha={alpha}...")
                    estimator_shrunk = GraphicalLasso(alpha,
                                                      assume_centered=True)
                    try:
                        estimator_shrunk.fit(shrunk_cov)
                        return estimator_shrunk
                    except BaseException:
                        continue
        except BaseException:
            return None
    else:
        return estimator


def get_conn_matrix(
    time_series,
    conn_model,
    dir_path,
    node_radius,
    smooth,
    dens_thresh,
    subnet,
    ID,
    roi,
    min_span_tree,
    disp_filt,
    parc,
    prune,
    atlas,
    parcellation,
    labels,
    coords,
    norm,
    binary,
    hpass,
    signal,
):
    """
    Computes a functional connectivity matrix based on a node-extracted
    time-series array. Includes a library of routines across Nilearn,
    scikit-learn, and skggm packages, among others.

    Parameters
    ----------
    time_series : array
        2D m x n array consisting of the time-series signal for each ROI node
        where m = number of scans and n = number of ROI's.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for partial
       correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    signal : str
        The name of a valid function used to reduce the time-series region
        extraction.

    Returns
    -------
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for
       covariance, sps for precision covariance, partcorr for partial
       correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting
        signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming
        (e.g. 'Default') used to filter nodes in the study of brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    signal : str
        The name of a valid function used to reduce the time-series region
        extraction.

    References
    ----------
    .. [1] Varoquaux, G., & Craddock, R. C. (2013). Learning and comparing
      functional connectomes across subjects. NeuroImage.
      https://doi.org/10.1016/j.neuroimage.2013.04.007
    .. [2] Jason Laska, Manjari Narayan, 2017. skggm 0.2.7:
      A scikit-learn compatible package for Gaussian and related Graphical
      Models. doi:10.5281/zenodo.830033

    """
    import sys
    from pynets.core import utils
    from pynets.fmri.estimation import get_optimal_cov_estimator
    from nilearn.connectome import ConnectivityMeasure

    nilearn_kinds = ["cov", "covariance", "covar", "corr", "cor",
                     "correlation", "partcorr", "parcorr",
                     "partialcorrelation", "cov", "covariance", "covar",
                     "sps", "sparse", "precision"]

    conn_matrix = None
    estimator = get_optimal_cov_estimator(time_series)

    def _fallback_covariance(time_series):
        from sklearn.ensemble import IsolationForest
        from sklearn import covariance

        # Remove gross outliers
        model = IsolationForest(contamination=0.02)
        model.fit(time_series)
        outlier_mask = model.predict(time_series)
        outlier_mask[outlier_mask == -1] = 0
        time_series = time_series[outlier_mask.astype('bool')]

        # Fall back to LedoitWolf
        print('Matrix estimation failed with Lasso and shrinkage due to '
              'ill conditions. Removing potential anomalies from the '
              'time-series using IsolationForest...')
        try:
            print("Attempting with Ledoit-Wolf...")
            conn_measure = ConnectivityMeasure(
                cov_estimator=covariance.LedoitWolf(store_precision=True,
                                                    assume_centered=True),
                kind=kind)
            conn_matrix = conn_measure.fit_transform([time_series])[0]
        except (np.linalg.linalg.LinAlgError, FloatingPointError):
            print("Attempting Oracle Approximating Shrinkage Estimator...")
            conn_measure = ConnectivityMeasure(
                cov_estimator=covariance.OAS(assume_centered=True),
                kind=kind)
            try:
                conn_matrix = conn_measure.fit_transform([time_series])[0]
            except (np.linalg.linalg.LinAlgError, FloatingPointError):
                raise ValueError('All covariance estimators failed to '
                                 'converge...')

        return conn_matrix

    if conn_model in nilearn_kinds:
        if conn_model == "corr" or conn_model == "cor" or \
                conn_model == "correlation":
            print("\nComputing correlation matrix...\n")
            kind = "correlation"
        elif conn_model == "partcorr" or conn_model == "parcorr" or \
                conn_model == "partialcorrelation":
            print("\nComputing partial correlation matrix...\n")
            kind = "partial correlation"
        elif conn_model == "sps" or conn_model == "sparse" or \
                conn_model == "precision":
            print("\nComputing precision matrix...\n")
            kind = "precision"
        elif conn_model == "cov" or conn_model == "covariance" or \
                conn_model == "covar":
            print("\nComputing covariance matrix...\n")
            kind = "covariance"
        else:
            raise ValueError(
                "\nERROR! No connectivity model specified at runtime. "
                "Select a valid estimator using the -mod flag.")

        # Try with the best-fitting Lasso estimator
        if estimator:
            conn_measure = ConnectivityMeasure(cov_estimator=estimator,
                                               kind=kind)
            try:
                conn_matrix = conn_measure.fit_transform([time_series])[0]
            except (np.linalg.linalg.LinAlgError, FloatingPointError):
                conn_matrix = _fallback_covariance(time_series)
        else:
            conn_matrix = _fallback_covariance(time_series)
    else:
        if conn_model == "QuicGraphicalLasso":
            try:
                from inverse_covariance import QuicGraphicalLasso
            except ImportError as e:
                print(e, "Cannot run QuicGraphLasso. "
                         "Skggm not installed!")

            # Compute the sparse inverse covariance via QuicGraphLasso
            # credit: skggm
            model = QuicGraphicalLasso(
                init_method="cov", lam=0.5, mode="default", verbose=1
            )
            print("\nCalculating QuicGraphLasso precision matrix using "
                  "skggm...\n")
            model.fit(time_series)
            conn_matrix = model.precision_
        elif conn_model == "QuicGraphicalLassoCV":
            try:
                from inverse_covariance import QuicGraphicalLassoCV
            except ImportError as e:
                print(e, "Cannot run QuicGraphLassoCV. "
                         "Skggm not installed!")

            # Compute the sparse inverse covariance via QuicGraphLassoCV
            # credit: skggm
            model = QuicGraphicalLassoCV(init_method="cov", verbose=1)
            print("\nCalculating QuicGraphLassoCV precision "
                  "matrix using skggm...\n")
            model.fit(time_series)
            conn_matrix = model.precision_
        elif conn_model == "QuicGraphicalLassoEBIC":
            try:
                from inverse_covariance import QuicGraphicalLassoEBIC
            except ImportError as e:
                print(e, "Cannot run QuicGraphLassoEBIC. "
                         "Skggm not installed!")

            # Compute the sparse inverse covariance via QuicGraphLassoEBIC
            # credit: skggm
            model = QuicGraphicalLassoEBIC(init_method="cov", verbose=1)
            print("\nCalculating QuicGraphLassoEBIC "
                  "precision matrix using skggm...\n")
            model.fit(time_series)
            conn_matrix = model.precision_
        elif conn_model == "AdaptiveQuicGraphicalLasso":
            try:
                from inverse_covariance import (
                    AdaptiveQuicGraphicalLasso,
                    QuicGraphicalLassoEBIC,
                )
            except ImportError as e:
                print(e, "Cannot run AdaptiveGraphLasso. "
                         "Skggm not installed!")

            # Compute the sparse inverse covariance via
            # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
            # credit: skggm
            model = AdaptiveQuicGraphicalLasso(
                estimator=QuicGraphicalLassoEBIC(
                    init_method="cov",), method="binary", )
            print("\nCalculating AdaptiveQuicGraphLasso precision matrix using"
                  " skggm...\n")
            model.fit(time_series)
            conn_matrix = model.estimator_.precision_
        else:
            raise ValueError(
                "\nNo connectivity model specified at runtime. "
                "Select a valid estimator using the -mod flag.")

    # Enforce symmetry
    conn_matrix = np.nan_to_num(np.maximum(conn_matrix, conn_matrix.T))

    if parc is True:
        node_radius = "parc"

    # Save unthresholded
    utils.save_mat(
        conn_matrix,
        utils.create_raw_path_func(
            ID,
            subnet,
            conn_model,
            roi,
            dir_path,
            node_radius,
            smooth,
            hpass,
            parc,
            signal,
        ),
    )

    if conn_matrix.shape < (2, 2):
        raise RuntimeError(
            "\nMatrix estimation selection yielded an "
            "empty or 1-dimensional graph. "
            "Check time-series for errors or try using a "
            "different atlas")

    if subnet is not None:
        atlas_name = f"{atlas}_{subnet}_stage-rawgraph"
    else:
        atlas_name = f"{atlas}_stage-rawgraph"

    utils.save_coords_and_labels_to_json(coords, labels, dir_path,
                                         atlas_name, indices=None)

    coords = np.array(coords)
    labels = np.array(labels)

    # assert coords.shape[0] == labels.shape[0] == conn_matrix.shape[0]

    del time_series

    return (
        conn_matrix,
        conn_model,
        dir_path,
        node_radius,
        smooth,
        dens_thresh,
        subnet,
        ID,
        roi,
        min_span_tree,
        disp_filt,
        parc,
        prune,
        atlas,
        parcellation,
        labels,
        coords,
        norm,
        binary,
        hpass,
        signal,
    )


def timeseries_bootstrap(tseries, block_size):
    """
    Generates a bootstrap sample derived from the input time-series.
    Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries

    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
      changes in brain connectivity for functional MRI. Statistica Sinica,
      special issue on Statistical Challenges and Advances in Brain Science,
      2008, 18: 1253-1268.

    """
    # calculate number of blocks
    k = int(np.ceil(float(tseries.shape[0]) / block_size))

    # generate random indices of blocks
    r_ind = np.floor(np.random.rand(1, k) * tseries.shape[0])
    blocks = np.dot(np.arange(0, block_size)[:, np.newaxis], np.ones([1, k]))

    block_offsets = np.dot(np.ones([block_size, 1]), r_ind)
    block_mask = (blocks + block_offsets).flatten("F")[: tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])

    return tseries[block_mask.astype("uint8"), :], block_mask.astype("uint8")


def fill_confound_nans(confounds, dir_path, drop_thr=0.50):
    """Fill the NaN values of a confounds dataframe with mean values"""
    import uuid
    from time import strftime
    import os

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    confounds_nonan = confounds.apply(lambda x: x.fillna(x.mean()), axis=0)
    confounds_nonan = confounds_nonan.dropna(thresh=len(
        confounds_nonan)*float(drop_thr), axis=1)
    os.makedirs(f"{dir_path}{'/confounds_tmp'}", exist_ok=True)
    conf_corr = (
        f"{dir_path}/confounds_tmp/confounds_mean_corrected_{run_uuid}.tsv"
    )
    confounds_nonan.to_csv(conf_corr, sep="\t", index=False)
    return conf_corr

