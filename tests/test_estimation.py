#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
import numpy as np
import time
import nibabel as nib
import os
import tempfile
import pandas as pd
import logging
from inspect import getargspec
from dipy.io import save_pickle
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(50)
from pynets.fmri.estimation import (get_conn_matrix, timeseries_bootstrap,
                                    fill_confound_nans, TimeseriesExtraction)
from pynets.dmri.estimation import (create_anisopowermap, tens_mod_fa_est,
                                    tens_mod_est, csa_mod_est, csd_mod_est,
                                    streams2graph, sfm_mod_est)
from nilearn._utils import as_ndarray
from nilearn.tests.test_signal import generate_signals
from nilearn._utils.extmath import is_spd
from numpy.testing import assert_array_almost_equal


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nib.Nifti1Image(data, affine), nib.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)

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

    network = 'Default'
    ID = '002'
    smooth = 2
    coords = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    node_size = 8
    extract_strategy = 'zscore'
    roi = None
    atlas = None
    uatlas = None
    labels = [1, 2, 3]

    time_series = generate_signals(n_features=n_features,
                                            n_confounds=5,
                                            length=n_features,
                                            same_variance=False)[0]

    outs = get_conn_matrix(
        time_series,
        conn_model_in,
        tempfile.TemporaryDirectory(),
        node_size,
        smooth,
        False,
        network,
        ID,
        roi,
        False,
        False,
        True,
        True,
        atlas,
        uatlas,
        labels,
        coords,
        3,
        False,
        0,
        extract_strategy,
    )

    conn_matrix_out, conn_model_out = outs[0], outs[1]

    assert isinstance(conn_matrix_out, np.ndarray)
    assert np.shape(conn_matrix_out) == np.shape(time_series)
    assert conn_model_in == conn_model_out

    if "corr" in conn_model_in:
        assert (is_spd(conn_matrix_out, decimal=7))
        d = np.sqrt(np.diag(np.diag(conn_matrix_out)))
        assert_array_almost_equal(np.diag(conn_matrix_out),
                                  np.ones((n_features)))
    elif "partcorr" in conn_model_in:
        prec = np.linalg.inv(conn_matrix_out)
        d = np.sqrt(np.diag(np.diag(prec)))
        assert_array_almost_equal(d.dot(conn_matrix_out).dot(d), -prec +
                                  2 * np.diag(np.diag(prec)))


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


@pytest.mark.parametrize("conf", [True, pytest.param(False, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("hpass", [None, 0.028, 0.080])
@pytest.mark.parametrize("mask", [True, None])
@pytest.mark.parametrize("func_file", [True, pytest.param(None, marks=pytest.mark.xfail)])
def test_timseries_extraction_prepare_inputs(conf, hpass, mask, func_file):
    """ Test preparing inputs method of the TimeseriesExtraction class."""

    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/BIDS/sub-25659/ses-1/func"

    shape1 = (13, 11, 12)
    affine1 = np.eye(4)
    length = 30

    func_img, mask_img = generate_random_img(shape1, affine=affine1,
                                             length=length)

    func_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    func_img.to_filename(func_file.name)

    # Create a temp parcel file
    parcels_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    parcels = np.zeros((50, 50, 50))
    parcels[10:20, 0, 0], parcels[0, 10:20, 0], parcels[0, 0, 10:20] = 1, 2, 3
    nib.Nifti1Image(parcels, np.eye(4)).to_filename(parcels_tmp.name)
    net_parcels_map_nifti_file = parcels_tmp.name

    # Create empty mask file
    mask_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask_img.to_filename(mask_tmp.name)
    mask = mask_tmp.name

    if conf:
        conf_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv')
        conf_mat = np.random.rand(length)
        conf_df = pd.DataFrame({'Conf1': conf_mat, "Conf2": [np.nan]*len(conf_mat)})
        conf_df.to_csv(conf_file.name, sep='\t', index=False)
        conf = conf_file.name

    smooth = 1
    network = 'Default'
    ID = '002'
    smooth = 2
    coords = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    node_size = 8
    extract_strategy = 'median'
    roi = None
    atlas = None
    uatlas = None
    labels = [1, 2, 3]

    te = TimeseriesExtraction(net_parcels_nii_path=net_parcels_map_nifti_file, node_size=node_size,
                              conf=conf, func_file=func_file.name, roi=roi,
                              dir_path=dir_path, ID=ID, network=network, smooth=smooth,
                              hpass=hpass, mask=mask,
                              extract_strategy=extract_strategy)
    te.prepare_inputs()

    assert np.shape(te._func_img) == np.shape(func_img)

    if hpass and hpass > 0:
        assert te.hpass == hpass
        assert te._detrending is False
    else:
        assert te.hpass is None
        assert te._detrending is True

    if mask:
        assert np.shape(te._mask_img) == np.shape(func_img)[:-1]

    if func_file:
        func_file.close()
    if conf:
        func_file.close()
    if mask:
        mask_tmp.close()

@pytest.mark.parametrize("conf", [True, None])
def test_timseries_extraction_extract(conf):
    """Test timeseries extraction and save methods of the TimeseriesExtraction class."""

    dir_path_tmp = tempfile.TemporaryDirectory()
    dir_path = dir_path_tmp.name

    shape1 = (13, 11, 12)
    affine1 = np.eye(4)
    length = 30

    func_img, mask_img = generate_random_img(shape1, affine=affine1,
                                             length=length)

    func_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    func_img.to_filename(func_file.name)

    # Create a temp parcel file
    parcels_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    parcels = np.zeros((50, 50, 50))
    parcels[10:20, 0, 0], parcels[0, 10:20, 0], parcels[0, 0, 10:20] = 1, 2, 3
    nib.Nifti1Image(parcels, np.eye(4)).to_filename(parcels_tmp.name)
    net_parcels_map_nifti_file = parcels_tmp.name

    # Create empty mask file
    mask_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask_img.to_filename(mask_tmp.name)
    mask = mask_tmp.name

    if conf:
        conf_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv')
        conf_mat = np.random.rand(length)
        conf_df = pd.DataFrame({'Conf1': conf_mat, "Conf2": [np.nan]*len(conf_mat)})
        conf_df.to_csv(conf_file.name, sep='\t', index=False)
        conf = conf_file.name

    smooth = 1
    network = 'Default'
    ID = '002'
    smooth = 2
    hpass = 100
    extract_strategy = 'mean'
    coords = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    node_size = 2
    roi = None
    atlas = None
    uatlas = None
    labels = [1, 2, 3]

    te = TimeseriesExtraction(net_parcels_nii_path=net_parcels_map_nifti_file, node_size=node_size,
                              conf=conf, func_file=func_file.name,
                              roi=roi, dir_path=dir_path, ID=ID, network=network, smooth=smooth,
                              hpass=hpass, mask=mask,
                              extract_strategy=extract_strategy)
    te.prepare_inputs()

    # Test parc extraction
    te.extract_ts_parc()
    assert np.shape(te.ts_within_nodes) == (np.shape(func_img)[-1] - 5, len(np.unique(parcels)) - 1)

    # Test save and clean up
    te._mask_path = te._mask_img
    te.save_and_cleanup()

    assert '_parcel_masker' not in te.__dict__.keys()

    func_file.close()
    parcels_tmp.close()
    mask_tmp.close()
    if conf:
        conf_file.close()


# dMRI
def test_create_anisopowermap(dmri_estimation_data):
    """ Test creating an anisotropic power map."""

    gtab = dmri_estimation_data['gtab']
    dwi_img = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    gtab_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.pkl')
    dwi_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')

    save_pickle(gtab_file.name, gtab)
    nib.save(dwi_img, dwi_file.name)
    nib.save(B0_mask_img, B0_mask_file.name)

    [anisopwr_path, _, _, _] = \
        create_anisopowermap(gtab_file.name, dwi_file.name, B0_mask_file.name)

    assert os.path.isfile(anisopwr_path)

    gtab_file.close()
    dwi_file.close()
    B0_mask_file.close()


def test_tens_mod_fa_est(dmri_estimation_data):
    """Test tensor FA image estimation."""

    gtab = dmri_estimation_data['gtab']
    dwi_img = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    gtab_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.pkl')
    dwi_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')

    save_pickle(gtab_file.name, gtab)
    nib.save(dwi_img, dwi_file.name)
    nib.save(B0_mask_img, B0_mask_file.name)

    [fa_path, _, _, _] = \
        tens_mod_fa_est(gtab_file.name, dwi_file.name, B0_mask_file.name)

    assert os.path.isfile(fa_path)

    gtab_file.close()
    dwi_file.close()
    B0_mask_file.close()


def test_tens_mod_est(dmri_estimation_data):
    """Test tensor ODF model estimation."""
    gtab = dmri_estimation_data['gtab']
    dwi_img = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')

    nib.save(B0_mask_img, B0_mask_file.name)
    dwi_data = dwi_img.get_fdata()

    [mod_odf, model] = \
        tens_mod_est(gtab, dwi_data, B0_mask_file.name)

    assert mod_odf is not None
    assert model is not None

    B0_mask_file.close()


def test_csa_mod_est(dmri_estimation_data):
    """Test CSA model estmation."""

    gtab = dmri_estimation_data['gtab']
    dwi_img = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    dwi_data = dwi_img.get_fdata()
    nib.save(B0_mask_img, B0_mask_file.name)

    [csa_mod, model] = \
        csa_mod_est(gtab, dwi_data, B0_mask_file.name, sh_order=0)

    assert csa_mod is not None
    assert model is not None

    B0_mask_file.close()


def test_csd_mod_est(dmri_estimation_data):
    """Test CSD model estimation."""

    gtab = dmri_estimation_data['gtab']
    dwi_img = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    nib.save(B0_mask_img, B0_mask_file.name)
    dwi_data = dwi_img.get_fdata()

    [csd_mod, model] = csd_mod_est(gtab, dwi_data, B0_mask_file.name, sh_order=0)

    assert csd_mod is not None
    assert model is not None

    B0_mask_file.close()


def test_sfm_mod_est(dmri_estimation_data):
    """Test SFM model estimation."""

    gtab = dmri_estimation_data['gtab']
    dwi_data = dmri_estimation_data['dwi_img']
    B0_mask_img = dmri_estimation_data['B0_mask_img']

    B0_mask_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    nib.save(B0_mask_img, B0_mask_file.name)
    dwi_data = dwi_data.get_fdata()

    [sf_odf, model] = sfm_mod_est(gtab, dwi_data, B0_mask_file.name)

    assert sf_odf is not None
    assert model is not None

    B0_mask_file.close()


@pytest.mark.parametrize("dsn", [True, False])
@pytest.mark.parametrize("fa_wei", [True, False])
def test_streams2graph(fa_wei, dsn):
    from dipy.core.gradients import gradient_table
    from pynets.registration import register
    from pynets.core import nodemaker
    from dipy.io import save_pickle
    import random
    import os

    base_dir = str(Path(__file__).parent/"examples")
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    conn_model = 'csd'
    min_length = 10
    error_margin = 2
    directget = 'prob'
    track_type = 'particle'
    target_samples = 500
    min_span_tree = True
    prune = 3
    norm = 6
    binary = False
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    network = 'Default'
    ID = '003'
    parc = True
    disp_filt = False
    node_size = None
    dens_thresh = False
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"
    t1_aligned_mni = f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"
    atlas_dwi = f"{base_dir}/003/dmri/whole_brain_cluster_labels_PCA200_dwi_track.nii.gz"
    streams = f"{base_dir}/miscellaneous/streamlines_model-csd_nodetype-parc_samples-1000streams_tracktype-particle_directget-prob_minlength-10.trk"
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab_file = f"{base_dir}/gtab.pkl"
    gtab = gradient_table(bvals, bvecs)
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0
    save_pickle(gtab_file, gtab)
    fa_path = tens_mod_fa_est(gtab_file, dwi_file, B0_mask)[0]

    coords = [(random.random()*2.0, random.random()*2.0, random.random()*2.0) for _ in range(200)]
    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    # if dsn is True:
    #     os.makedirs(f"{dir_path}/dmri_reg/DSN", exist_ok=True)
    #     (streams_mni, dir_path, track_type, target_samples, conn_model, network, node_size, dens_thresh, ID, roi,
    #      min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, directget,
    #      warped_fa, min_length, error_margin) = register.direct_streamline_norm(streams, fa_path, fa_path, dir_path,
    #                                                                             track_type, target_samples, conn_model,
    #                                                                             network, node_size, dens_thresh, ID,
    #                                                                             roi, min_span_tree, disp_filt, parc,
    #                                                                             prune, atlas, atlas_dwi, uatlas,
    #                                                                             labels, coords, norm, binary, uatlas,
    #                                                                             dir_path, [0.1, 0.2], [40, 30],
    #                                                                             directget, min_length, t1_aligned_mni,
    #                                                                             error_margin)
    #
    #     conn_matrix = streams2graph(atlas_mni, streams_mni, dir_path, track_type, target_samples,
    #                                 conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree,
    #                                 disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary,
    #                                 directget, warped_fa, error_margin, min_length)[2]
    # else:
    #     conn_matrix = streams2graph(atlas_dwi, streams, dir_path, track_type, target_samples,
    #                                 conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree,
    #                                 disp_filt, parc, prune, atlas, atlas_dwi, labels, coords, norm, binary,
    #                                 directget, fa_path, error_margin, min_length)[2]

    conn_matrix = streams2graph(atlas_dwi, streams, dir_path, track_type, target_samples,
                                conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree,
                                disp_filt, parc, prune, atlas, atlas_dwi, labels, coords, norm, binary,
                                directget, fa_path, min_length, error_margin)[2]

    assert conn_matrix is not None
