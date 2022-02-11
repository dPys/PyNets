#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
# import time
import nibabel as nib
import os
import tempfile
import pandas as pd
import logging
# from inspect import getargspec
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(50)
from pynets.fmri.estimation import (get_conn_matrix, timeseries_bootstrap,
                                    fill_confound_nans)
from pynets.fmri.interfaces import TimeseriesExtraction
from nilearn.tests.test_signal import generate_signals
from nilearn._utils.extmath import is_spd
from numpy.testing import assert_array_almost_equal

# fMRI
@pytest.mark.parametrize("conn_model_in",
    [
        # Standard models
        'corr', 'partcorr', 'cov', 'sps',

        # These models require skggm and will fail if not installed
        pytest.param('QuicGraphicalLasso', marks=pytest.mark.xfail),
        pytest.param('QuicGraphicalLassoCV', marks=pytest.mark.xfail),
        pytest.param('QuicGraphicalLassoEBIC', marks=pytest.mark.xfail),
        pytest.param('AdaptiveQuicGraphicalLasso', marks=pytest.mark.xfail),
        pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))
    ]
)
@pytest.mark.parametrize("n_features",
    [
        50,
        pytest.param(2, marks=pytest.mark.xfail)
    ]
)
def test_get_conn_matrix(conn_model_in, n_features):
    """ Test computing a functional connectivity matrix."""

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    subnet = 'Default'
    ID = '002'
    smooth = 2
    coords = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    node_radius = 8
    signal = 'zscore'
    roi = None
    atlas = None
    parcellation = None
    labels = [1, 2, 3]

    time_series = generate_signals(n_features=n_features, n_confounds=5,
                                   length=n_features, same_variance=False)[0]

    outs = get_conn_matrix(
        time_series,
        conn_model_in,
        dir_path,
        node_radius,
        smooth,
        False,
        subnet,
        ID,
        roi,
        False,
        False,
        True,
        True,
        atlas,
        parcellation,
        labels,
        coords,
        3,
        False,
        0,
        signal,
    )

    conn_matrix_out, conn_model_out = outs[0], outs[1]

    assert isinstance(conn_matrix_out, np.ndarray)
    assert np.shape(conn_matrix_out) == np.shape(time_series)
    assert conn_model_in == conn_model_out

    if "corr" in conn_model_in:
        assert (is_spd(conn_matrix_out, decimal=7))
        d = np.sqrt(np.diag(np.diag(conn_matrix_out)))
        assert_array_almost_equal(np.diag(conn_matrix_out),
                                  np.ones(n_features))
    elif "partcorr" in conn_model_in:
        prec = np.linalg.inv(conn_matrix_out)
        d = np.sqrt(np.diag(np.diag(prec)))
        assert_array_almost_equal(d.dot(conn_matrix_out).dot(d), -prec +
                                  2 * np.diag(np.diag(prec)))
    tmp.cleanup()


def test_timeseries_bootstrap():
    """Test bootstrapping a sample of time series."""

    tseries = np.random.rand(100, 10)
    bseries = timeseries_bootstrap(tseries, 5)

    assert np.shape(bseries[0]) == np.shape(tseries)
    assert len(bseries[1]) == len(tseries)


def test_fill_confound_nans():
    """ Testing filling pd dataframe np.nan values with mean."""

    confounds = np.ones((5, 5))
    confounds[0][0] = np.nan
    confounds = pd.DataFrame({'Column1': [np.nan, 2, 4]})
    with tempfile.TemporaryDirectory() as dir_path:
        conf_corr = fill_confound_nans(confounds, dir_path)
        conf_corr = np.genfromtxt(conf_corr, delimiter='\t', skip_header=True)

    assert not np.isnan(conf_corr).any()
    assert conf_corr[0] == np.mean(conf_corr[1:])


@pytest.mark.parametrize("conf",
                         [True, pytest.param(False, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("hpass", [0, 0.028, 0.080])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("func",
                         [True, pytest.param(False, marks=pytest.mark.xfail)])
def test_timseries_extraction(fmri_estimation_data, parcellation_data, conf,
                              hpass, mask, func):
    """ Test preparing inputs method of the TimeseriesExtraction class."""

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    if func:
        func_file = fmri_estimation_data['func_file']
    else:
        func_file = None

    func_img = nib.load(func_file)

    if mask:
        mask = fmri_estimation_data['mask_file']
    else:
        mask = None

    if conf:
        conf = fmri_estimation_data['conf_file']
    else:
        conf = None

    subnet = 'Default'
    ID = '002'
    smooth = 2
    node_radius = 8
    signal = 'median'
    roi = None
    net_parcels_map_nifti_file = \
        parcellation_data['net_parcels_map_nifti_file']
    te = TimeseriesExtraction(
        net_parcels_nii_path=net_parcels_map_nifti_file,
        node_radius=node_radius, conf=conf,
        func_file=func_file, roi=roi,
        dir_path=dir_path, ID=ID, subnet=subnet, smooth=smooth, hpass=hpass,
        mask=mask, signal=signal)
    te.prepare_inputs()
    te.extract_ts_parc()

    parcels = parcellation_data['parcels']
    assert np.shape(te._func_img) == np.shape(func_img)
    assert np.shape(te.ts_within_nodes) == \
           (np.shape(func_img)[-1] - 5,
            len(np.unique(parcels)) - 1)

    if hpass and hpass > 0:
        assert te.hpass == hpass
        assert te._detrending is False
    else:
        assert te.hpass is None
        assert te._detrending is True

    if mask:
        assert np.shape(te._mask_img) == np.shape(func_img)[:-1]
        # Test save and clean up
        te._mask_path = te._mask_img
        te.save_and_cleanup()

    tmp.cleanup()
