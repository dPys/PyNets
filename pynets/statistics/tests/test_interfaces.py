#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
@authors: Derek Pisner & Ryan Hammonds
"""
import pytest
import os
import numpy as np
import networkx as nx
import time
from pathlib import Path
from pynets.statistics.individual import algorithms
import logging
from tempfile import NamedTemporaryFile
from ...conftest import gen_mat_data

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("conn_model",
                         ['corr', 'partcorr', 'cov', 'sps'])
@pytest.mark.parametrize("prune",
                         [pytest.param(0,
                                       marks=pytest.mark.xfail(
                                           raises=UnboundLocalError)), 1, 2,
                          3])
@pytest.mark.parametrize("norm", [i for i in range(1, 7)])
def test_clean_graphs(gen_mat_data, conn_model, prune, norm):
    # test_CleanGraphs
    """
    Test all combination of parameters for the CleanGraphs class
    """

    in_mat = gen_mat_data()['mat_list'][0]
    est_path = gen_mat_data()['mat_file_list'][0]

    clean = algorithms.CleanGraphs(0.5, conn_model, est_path, prune, norm)
    clean.normalize_graph()
    clean.print_summary()
    clean.create_length_matrix()
    clean.binarize_graph()

    clean.prune_graph()

    G = nx.from_numpy_array(in_mat)
    assert len(clean.G) >= 0
    assert len(clean.G) <= len(G)


@pytest.mark.parametrize("binary", ['True', 'False'])
@pytest.mark.parametrize("prune", ['0', '1', '3'])
@pytest.mark.parametrize("norm", ['0', '1', '2', '3', '4', '5', '6'])
@pytest.mark.parametrize("conn_model", ['corr', 'cov'])
def test_extractnetstats(gen_mat_data, binary, prune, norm, conn_model):
    """
    Test extractnetstats functionality
    """
    base_dir = str(Path(__file__).parent / "examples")
    ID = '002'
    subnet = 'Default'
    thr = 0.95

    start_time = time.time()

    f_temp = NamedTemporaryFile(mode='w+', suffix='.npy')

    in_mat = gen_mat_data(asfile=False)['mat_list'][0]

    np.save(f_temp.name, in_mat)
    est_path = f_temp.name

    roi = None
    try:
        out_path = algorithms.extractnetstats(ID, subnet, thr, conn_model,
                                              est_path, roi, prune,
                                              norm, binary)
        print("%s%s%s" % (
        'finished: ',
        str(np.round(time.time() - start_time, 1)), 's'))
        assert out_path is not None

    except PermissionError:
        pass


@pytest.mark.parametrize("plot_switch", [True, False])
@pytest.mark.parametrize("embed", [True, False])
@pytest.mark.parametrize("create_summary", [True, False])
@pytest.mark.parametrize("graph_num", [
    pytest.param(-1, marks=pytest.mark.xfail(raises=UserWarning)),
    pytest.param(0, marks=pytest.mark.xfail(raises=IndexError)),
    1,
    2])
def test_collect_pandas_df_make(plot_switch, embed, create_summary, graph_num):
    """
    Test for collect_pandas_df_make() functionality
    """
    base_dir = str(Path(__file__).parent / "examples")
    subnet = None
    ID = '002'

    if graph_num == -1:
        net_mets_csv_list = [
            f"{base_dir}/miscellaneous/002_parcels_Default.nii.gz"]
    elif graph_num == 0:
        net_mets_csv_list = []
    elif graph_num == 1:
        net_mets_csv_list = [
            f"{base_dir}/topology/metrics_sub-0021001_modality-dwi_"
            f"nodetype-parc_model-csa_thrtype-PROP_thr-0.2.csv"]
    else:
        net_mets_csv_list = [
            f"{base_dir}/topology/metrics_sub-0021001_modality-dwi_"
            f"nodetype-parc_model-csa_thrtype-PROP_thr-0.2.csv",
            f"{base_dir}/topology/metrics_sub-0021001_modality-dwi_"
            f"nodetype-parc_model-csa_thrtype-PROP_thr-0.3.csv"]

    combination_complete = algorithms.collect_pandas_df_make(
        net_mets_csv_list, ID, subnet, plot_switch=plot_switch, embed=embed,
        create_summary=create_summary)

    assert combination_complete is True
