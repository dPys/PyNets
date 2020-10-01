#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.core import thresholding
import networkx as nx
import pytest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("cp", [True, False])
@pytest.mark.parametrize("thr", [pytest.param(-0.2, marks=pytest.mark.xfail), 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
@pytest.mark.parametrize("mat_size", [(10, 10), (100, 100)])
def test_conn_mat_operations(cp, thr, mat_size):
    """ Includes original tests using .npy and new tests from randomly generate arrays, as
        well as additional assert statements.
    """
    def test_binarize(x, thr, cp):
        y = thresholding.threshold_proportional(x, thr, copy=cp)
        s = thresholding.binarize(y)
        assert np.sum(s) == np.count_nonzero(y)

    def test_normalize(x, thr, cp):
        x = thresholding.threshold_proportional(x, 1, copy=True) # remove diagonal
        s = thresholding.normalize(x)
        assert np.max(s) <= 1 and np.min(s) >= 0
        assert np.max(s) == 1 and np.min(s) == round(min(x.flatten())/max(x.flatten()), 1)

    def test_threshold_absolute(x, thr, cp):
        s = thresholding.threshold_absolute(x, thr, copy=cp)
        s_test = [val for arr in s for val in arr if val >= thr] # search for value > thr
        assert round(np.sum(s), 10) == round(np.sum(s_test), 10)

    def test_invert(x, thr, cp):
        x_cp = x.copy() # invert modifies array in place and need orig to assert.
        x_cp = thresholding.threshold_proportional(x_cp, thr, copy=cp)
        s = thresholding.invert(x_cp)
        x = x.flatten() # flatten arrays to more easily check s > x.
        s = s.flatten()
        s_gt_x = [inv_val > x[idx] for idx, inv_val in enumerate(s) if inv_val > 0]
        assert False not in s_gt_x

    def test_autofix(x, thr, cp):
        x[1][1] = np.inf
        x[2][1] = np.nan
        s = thresholding.autofix(x)
        assert (np.nan not in s) and (np.inf not in s)

    def test_density(x, thr):
        d_known = thresholding.est_density(thresholding.threshold_absolute(x, thr, copy=True))
        x = thresholding.density_thresholding(x, d_known)
        d_test = thresholding.est_density(x)
        assert np.equal(np.round(d_known, 1), np.round(d_test, 1))

    def test_thr2prob(x, thr):
        s = thresholding.threshold_absolute(thresholding.normalize(x), thr)
        s[0][0] = 0.0000001
        t = thresholding.thr2prob(s)
        assert float(len(t[np.logical_and(t < 0.001, t > 0)])) == float(0.0)

    def test_local_thresholding_prop(x, thr):
        coords = []
        labels = []
        for idx, val in enumerate(x):
            coords.append(idx)
            labels.append('ROI_' + str(idx))

        # Disconnect graph for edge case.
        x_undir = nx.from_numpy_matrix(x).to_undirected()
        for i in range(1, 10):
            x_undir.remove_edge(0, i)
        x_undir = nx.to_numpy_matrix(x_undir)

        conn_matrix_thr = thresholding.local_thresholding_prop(x, thr)
        assert conn_matrix_thr is not None
        conn_matrix_thr_undir = thresholding.local_thresholding_prop(x_undir, thr)
        assert conn_matrix_thr_undir is not None

    def test_knn(x, thr):
        k = int(thr * 10)
        gra = thresholding.knn(x, k)

    def test_disparity_filter(x, thr):
        G_undir = nx.from_numpy_matrix(x)
        G_dir = G_undir.to_directed()

        # Test edge case where {in,out}_degree are 0.
        for e in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
            G_dir.remove_edge(0, e)
        for e in range(1, 10):
            G_dir.remove_edge(e, 1)
        for e in range(1, 10):
            G_undir.remove_edge(0, e)

        N = thresholding.disparity_filter(G_dir, weight='weight')
        assert N is not None
        N = thresholding.disparity_filter(G_undir, weight='weight')
        assert N is not None

    def test_disparity_filter_alpha_cut(x):
        G_undir = nx.from_numpy_matrix(x)
        G_dir = G_undir.to_directed()
        G_dir.add_edge(0, 1, alpha_in=0.1, alpha_out=0.1)
        G_undir.add_edge(0, 1, alpha=0.1, weight=0.5)

        for mode in ['or', 'and']:
            N = thresholding.disparity_filter_alpha_cut(G_dir, weight='weight', cut_mode = mode)
            assert N is not None
            N = thresholding.disparity_filter_alpha_cut(G_undir, weight='weight', cut_mode = mode)
            assert N is not None

    def test_weight_conversion(x, cp):
        # Cross test all wcm and copy combinations
        for wcm in ['binarize', 'lengths']:
            w = thresholding.weight_conversion(x, wcm, cp)
            assert w is not None

    def test_weight_to_distance(x):
        G = nx.from_numpy_matrix(x)
        w = thresholding.weight_to_distance(G)
        assert w is not None

    def test_standardize(x):
        w = thresholding.standardize(x)
        assert w is not None

    base_dir = str(Path(__file__).parent/"examples")
    # base_dir = '/Users/derekpisner/Applications/PyNets/tests/examples'
    W = np.load(f"{base_dir}/miscellaneous/002_rsn-Default_model-cov_raw_mat.npy")

    x_orig = W.copy()
    x_rand = x = np.random.rand(mat_size[0], mat_size[1])
    test_binarize(x_orig, thr, cp)
    test_binarize(x_rand, thr, cp)

    x_orig = W.copy()
    x_rand = np.random.rand(mat_size[0], mat_size[1])
    test_normalize(x_orig, thr, cp)
    test_normalize(x_rand, thr, cp)

    x_orig = W.copy()
    x_rand = np.random.rand(mat_size[0], mat_size[1])
    test_threshold_absolute(x_orig, thr, cp)
    test_threshold_absolute(x_rand, thr, cp)

    x_orig = W.copy()
    x_rand = np.random.rand(mat_size[0], mat_size[1])
    test_invert(x_orig, thr, cp)
    test_invert(x, thr, cp)

    x_orig = W.copy()
    x_rand = np.random.rand(mat_size[0], mat_size[1])
    test_autofix(x_orig, thr, cp)
    test_autofix(x, thr, cp)

    # Prevent redundant testing.
    if cp == True:
        x_orig = W.copy()
        x_rand = np.random.rand(mat_size[0], mat_size[1])
        test_density(x_orig, thr)
        test_density(x_orig, thr)

        x_orig = W.copy()
        x_rand = np.random.rand(mat_size[0], mat_size[1])
        test_thr2prob(x_orig, thr)
        test_thr2prob(x_rand, thr)

        x_orig = W.copy()
        x_rand = np.random.rand(mat_size[0], mat_size[1])
        test_local_thresholding_prop(x_rand, thr)
        test_knn(x_rand, thr)
        test_disparity_filter(x_rand, thr)

    if thr == 0.2 and cp == True:
        test_disparity_filter_alpha_cut(x_rand)
        test_weight_to_distance(x_rand)
        test_standardize(x_rand)

    if thr == 0.2:
        test_weight_conversion(x_rand, cp)


@pytest.mark.parametrize("thr", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_edge_cases(thr):
    # local_thresholding_prop: nng.number_of_edges() == 0 and number_before >= maximum_edges
    x = np.zeros((10,10))
    x = nx.to_numpy_array(nx.from_numpy_matrix(x).to_directed())
    coords = [idx for idx, val in enumerate(x)]
    labels = ['ROI_' + str(idx) for idx, val in enumerate(x)]

    for idx, i in enumerate(range(0, 10)):
        if idx < 9:
            x[i][idx+1] = 1
        if idx < 10 and idx > 0:
            x[i][idx-1] = 1

    conn_mat_edge_one = thresholding.local_thresholding_prop(x, thr)
    assert conn_mat_edge_one is not None


@pytest.mark.parametrize("type,parc,all_zero,frag_g",
    [
        pytest.param('func', True, True, True, marks=pytest.mark.xfail),
        pytest.param('struct', True, True, True, marks=pytest.mark.xfail),
        ('either', True, False, True),
        ('either', False, False, False)
    ]
)
@pytest.mark.parametrize("min_span_tree", [True, False])
@pytest.mark.parametrize("disp_filt", [True, False])
@pytest.mark.parametrize("dens_thresh", [True, False])
def test_thresh_func(type, parc, all_zero, min_span_tree, disp_filt, dens_thresh, frag_g):
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/miscellaneous"

    if all_zero == True and type == 'func':
        conn_matrix = np.zeros((10, 10))
    elif frag_g == True:
        conn_matrix = np.random.rand(10, 10)
        for i in range(1, 10):
            conn_matrix[0][i] = 0
            conn_matrix[i][0] = 0
    else:
        conn_matrix = np.random.rand(10, 10)

    ID = '002'
    network = 'Default'
    conn_model = 'sps'
    thr = 0.5
    node_size = 6
    smooth = 2
    roi = f"{dir_path}/002_parcels_resampled2roimask_pDMN_3_bin.nii.gz"
    coord_file_path = f"{dir_path}/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{dir_path}/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    # The arguments below arr never used in the thresholding.tresh_func, but are returned.
    prune = True
    atlas = 'whole_brain_cluster_labels_PCA200'
    uatlas = None
    norm = 1
    binary = False
    hpass = False
    extract_strategy = 'mean'

    conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, smooth, \
        prune, ID, dir_path, atlas, uatlas, labels, coords, norm, binary, hpass, extract_strategy = \
        thresholding.thresh_func(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path,
                                 roi, node_size, min_span_tree, smooth, disp_filt, parc, prune,
                                 atlas, uatlas, labels, coords, norm, binary, hpass, extract_strategy,
                                 check_consistency=False)

    assert conn_matrix_thr is not None
    if min_span_tree is False and disp_filt is False and dens_thresh is True:
        assert edge_threshold is None # edge_threshold will be none in one case
    else:
        assert edge_threshold is not None
    assert os.path.isfile(est_path) is True
    assert thr is not None
    if node_size != 'parc':
        assert isinstance(node_size, int)
    else:
        assert isinstance(node_size, str)
    assert network is not None
    assert conn_model is not None
    assert roi is not None
    assert smooth is not None
    assert prune is not None
    assert ID is not None
    assert dir_path is not None
    assert atlas is not None
    assert uatlas is None # Set to none above
    assert labels is not None
    assert coords is not None
    assert norm is not None
    assert binary is not None
    assert hpass is not None

    # Additional arguments for thresh_struc
    if all_zero == True and type == 'struct':
        conn_matrix = np.zeros((10, 10))

    target_samples = 2
    track_type = 'local'
    atlas_mni = f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"
    streams = f"{base_dir}/miscellaneous/streamlines_model-csd_nodetype-parc_samples-10000streams_tracktype-local_directget-prob_minlength-0.trk"
    directget = 'prob'
    min_length = 20
    error_margin = 6

    conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, prune, \
        ID, dir_path, atlas, uatlas, labels, coords, norm, binary, target_samples, track_type, \
        atlas_mni, streams, directget, min_length, error_margin = thresholding.thresh_struct(dens_thresh, thr, conn_matrix,
                                                                               conn_model, network, ID,
                                                                               dir_path, roi, node_size,
                                                                               min_span_tree, disp_filt, parc,
                                                                               prune, atlas, uatlas, labels,
                                                                               coords, norm, binary,
                                                                               target_samples, track_type,
                                                                               atlas_mni, streams, directget,
                                                                               min_length, error_margin, check_consistency=False)

    assert dens_thresh is not None
    assert thr is not None
    assert conn_matrix is not None
    assert conn_model is not None
    assert network is not None
    assert ID is not None
    assert dir_path is not None
    assert roi is not None
    assert node_size is not None
    assert min_span_tree is not None
    assert disp_filt is not None
    assert parc is not None
    assert prune is not None
    assert atlas is not None
    assert uatlas is None
    assert labels is not None
    assert coords is not None
    assert norm is not None
    assert binary is not None
    assert target_samples is not None
    assert track_type is not None
    assert atlas_mni is not None
    assert streams is not None
    assert directget is not None


def test_thresh_raw_graph():
    from pynets.core import thresholding

    thr = 0.5
    conn_matrix = np.random.rand(10, 10)
    min_span_tree = True
    dens_thresh = True
    disp_filt = True
    # base_dir = str(Path(__file__).parent/"examples")
    base_dir = '/Users/derekpisner/Applications/PyNets/tests/examples'
    est_path = f"{base_dir}/miscellaneous/sub-0021001_rsn-Default_" \
               f"nodetype-parc_model-sps_template-MNI152_T1_thrtype-" \
               f"DENS_thr-0.19.npy"

    [thr_type, edge_threshold, conn_matrix_thr, thr, est_path] = \
        thresholding.thresh_raw_graph(
        conn_matrix,
        thr,
        min_span_tree,
        dens_thresh,
        disp_filt,
        est_path)

    assert thr_type is not None
    assert edge_threshold is not None
    assert conn_matrix_thr is not None
    assert thr is not None
    assert est_path is None
