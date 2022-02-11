#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
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
import tempfile
import pkg_resources
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("cp", [True, False])
@pytest.mark.parametrize("thr",
                         [pytest.param(-0.2, marks=pytest.mark.xfail), 0.0,
                          0.2, 0.4, 0.6, 0.8, 1.0])
@pytest.mark.parametrize("mat_size", [(10, 10), (100, 100)])
def test_conn_mat_operations(gen_mat_data, cp, thr, mat_size):
    """ Includes original tests using .npy and new tests from randomly
    generate arrays, as well as additional assert statements.
    """

    def test_binarize(x, thr, cp):
        y = thresholding.threshold_proportional(x, thr, copy=cp)
        s = thresholding.binarize(y)
        assert np.sum(s) == np.count_nonzero(y)

    def test_normalize(x, thr, cp):
        x = thresholding.threshold_proportional(x, 1,
                                                copy=True)  # remove diagonal
        s = thresholding.normalize(x)
        assert np.max(s) <= 1 and np.min(s) >= 0
        assert np.max(s) == 1 and np.min(s) == round(
            min(x.flatten()) / max(x.flatten()), 1)

    def test_threshold_absolute(x, thr, cp):
        s = thresholding.threshold_absolute(x, thr, copy=cp)
        s_test = [val for arr in s for val in arr if
                  val >= thr]  # search for value > thr
        assert np.round(np.sum(s), 4) == np.round(np.sum(s_test), 4)

    def test_invert(x, thr, cp):
        x_cp = x.copy()
        x_cp = thresholding.threshold_proportional(x_cp, thr, copy=cp)
        s = thresholding.invert(x_cp)
        x = x.flatten()  # flatten arrays to more easily check s > x.
        s = s.flatten()
        s_gt_x = [inv_val > x[idx] for idx, inv_val in enumerate(s) if
                  inv_val > 0]
        assert False not in s_gt_x

    def test_autofix(x, thr, cp):
        x[1][1] = np.inf
        x[2][1] = np.nan
        s = thresholding.autofix(x)
        assert (np.nan not in s) and (np.inf not in s)

    def test_density(x, thr):
        d_known = thresholding.est_density(
            thresholding.threshold_absolute(x, thr, copy=True))
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
        assert isinstance(conn_matrix_thr, np.ndarray)
        conn_matrix_thr_undir = thresholding.local_thresholding_prop(x_undir,
                                                                     thr)
        assert isinstance(conn_matrix_thr_undir, np.ndarray)

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

        B = thresholding.disparity_filter(G_dir, weight='weight')
        assert isinstance(B, nx.Graph)
        B = thresholding.disparity_filter(G_undir, weight='weight')
        assert isinstance(B, nx.Graph)

    def test_disparity_filter_alpha_cut(x):
        G_undir = nx.from_numpy_matrix(x)
        G_dir = G_undir.to_directed()
        G_dir.add_edge(0, 1, alpha_in=0.1, alpha_out=0.1)
        G_undir.add_edge(0, 1, alpha=0.1, weight=0.5)

        for mode in ['or', 'and']:
            B = thresholding.disparity_filter_alpha_cut(G_dir, weight='weight',
                                                        cut_mode=mode)
            assert isinstance(B, nx.Graph)
            B = thresholding.disparity_filter_alpha_cut(G_undir,
                                                        weight='weight',
                                                        cut_mode=mode)
            assert isinstance(B, nx.Graph)

    def test_weight_conversion(x, cp):
        # Cross test all wcm and copy combinations
        for wcm in ['binarize', 'lengths']:
            w = thresholding.weight_conversion(x, wcm, cp)
            assert isinstance(w, np.ndarray)

    def test_weight_to_distance(x):
        G = nx.from_numpy_matrix(x)
        w = thresholding.weight_to_distance(G)
        assert isinstance(w, nx.Graph)

    def test_standardize(x):
        w = thresholding.standardize(x)
        assert isinstance(w, np.ndarray)

    W = gen_mat_data(asfile=False)['mat_list'][0]

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
    # local_thresholding_prop: nng.number_of_edges() == 0 and
    # number_before >= maximum_edges
    x = np.zeros((10, 10))
    x = nx.to_numpy_array(nx.from_numpy_matrix(x).to_directed())

    for idx, i in enumerate(range(0, 10)):
        if idx < 9:
            x[i][idx + 1] = 1
        if idx < 10 and idx > 0:
            x[i][idx - 1] = 1

    conn_mat_edge_one = thresholding.local_thresholding_prop(x, thr)
    assert isinstance(conn_mat_edge_one, np.ndarray)


@pytest.mark.parametrize("type,parc,all_zero,frag_g",
                         [
                             pytest.param('func', True, True, True,
                                          marks=pytest.mark.xfail),
                             pytest.param('struct', True, True, True,
                                          marks=pytest.mark.xfail),
                             ('either', True, False, True),
                             ('either', False, False, False)
                         ]
                         )
@pytest.mark.parametrize("min_span_tree", [True, False])
@pytest.mark.parametrize("disp_filt", [True, False])
@pytest.mark.parametrize("dens_thresh", [True, False])
def test_thresh_func(type, parc, all_zero, min_span_tree, disp_filt,
                     dens_thresh, frag_g):
    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

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
    subnet = 'Default'
    conn_model = 'corr'
    thr = 0.5
    node_radius = 6
    smooth = 2
    roi = pkg_resources.resource_filename(
        "pynets", "templates/rois/pDMN_3_bin.nii.gz")
    coords = ['Coord_' + str(idx) for idx, val in
              enumerate(range(conn_matrix.shape[0]))]
    labels = ['ROI_' + str(idx) for idx, val in
              enumerate(range(conn_matrix.shape[0]))]

    # The arguments below arr never used in the thresholding.tresh_func,
    # but are returned.
    prune = 0
    atlas = 'whole_brain_cluster_labels_PCA200'
    parcellation = None
    norm = 1
    binary = False
    hpass = False
    signal = 'mean'

    edge_threshold, est_path, thr, node_radius, subnet, conn_model, roi, \
    smooth, prune, ID, dir_path, atlas, parcellation, labels, coords, norm, \
    binary, hpass, signal = thresholding.thresh_func(
        dens_thresh, thr, conn_matrix, conn_model, subnet, ID, dir_path,
        roi, node_radius, min_span_tree, smooth, disp_filt, parc, prune,
        atlas, parcellation, labels, coords, norm, binary, hpass, signal,
        check_consistency=False)

    if min_span_tree is False and disp_filt is False and dens_thresh is True:
        assert edge_threshold is None
    else:
        assert isinstance(edge_threshold, str)
    assert os.path.isfile(est_path) is True
    assert (0 <= thr <= 1)
    if isinstance(node_radius, str):
        assert node_radius is 'parc'
    else:
        assert isinstance(node_radius, int)
    assert isinstance(subnet, str)
    assert isinstance(conn_model, str)
    if roi is not None:
        assert isinstance(roi, str)
        assert os.path.isfile(roi) is True
    assert isinstance(smooth, int)
    assert isinstance(ID, str)
    assert isinstance(dir_path, str)
    assert isinstance(atlas, str)
    if parcellation is not None:
        assert isinstance(parcellation, str)
        assert os.path.isfile(parcellation) is True
    assert isinstance(labels, list)
    assert isinstance(coords, list)
    assert isinstance(norm, int)
    assert isinstance(binary, bool)
    assert isinstance(hpass, bool)

    # Additional arguments for thresh_struc
    if all_zero == True and type == 'struct':
        conn_matrix = np.zeros((10, 10))

    track_type = 'local'
    atlas_for_streams = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    streams = f"{base_dir}/miscellaneous/streamlines_model-csd_" \
              f"nodetype-parc_tracktype-particle_" \
              f"traversal-prob_minlength-10.trk"
    traversal = 'prob'
    min_length = 20
    error_margin = 6

    edge_threshold, est_path, thr, node_radius, subnet, conn_model, roi, prune, \
    ID, dir_path, atlas, parcellation, labels, coords, norm, binary, track_type, \
    atlas_mni, streams, traversal, min_length, error_margin = thresholding.thresh_struct(
        dens_thresh, thr, conn_matrix,
        conn_model, subnet, ID,
        dir_path, roi, node_radius,
        min_span_tree, disp_filt, parc,
        prune, atlas, parcellation, labels,
        coords, norm, binary,
        track_type,
        atlas_for_streams, streams, traversal,
        min_length, error_margin, check_consistency=False)

    assert isinstance(dens_thresh, bool)
    assert (0 <= thr <= 1)
    assert conn_matrix.size is 100
    assert isinstance(conn_model, str)
    assert isinstance(subnet, str)
    assert isinstance(ID, str)
    assert isinstance(dir_path, str)
    assert isinstance(roi, str)
    if isinstance(node_radius, str):
        assert node_radius is 'parc'
    else:
        assert isinstance(node_radius, int)
    assert isinstance(min_span_tree, bool)
    assert isinstance(disp_filt, bool)
    assert isinstance(parc, bool)
    assert isinstance(atlas, str)
    assert parcellation is None
    assert isinstance(labels, list)
    assert len(coords) is len(labels)
    assert isinstance(coords, list)
    assert isinstance(norm, int)
    assert isinstance(binary, bool)
    assert isinstance(track_type, str)
    assert isinstance(atlas_mni, str)
    assert isinstance(streams, str)
    assert isinstance(traversal, str)

    tmp.cleanup()


def test_thresh_raw_graph(gen_mat_data):
    from pynets.core import thresholding

    thr = 0.5
    conn_matrix = np.random.rand(10, 10)
    min_span_tree = True
    dens_thresh = True
    disp_filt = True

    est_path = gen_mat_data()['mat_file_list'][0]

    [thr_type, edge_threshold, conn_matrix_thr, thr, est_path] = \
        thresholding.thresh_raw_graph(
            conn_matrix,
            thr,
            min_span_tree,
            dens_thresh,
            disp_filt,
            est_path)
    assert isinstance(thr_type, str)
    assert isinstance(edge_threshold, str)
    assert conn_matrix_thr.size is 100
    assert (0 <= thr <= 1)
    assert os.path.isfile(est_path) is True
