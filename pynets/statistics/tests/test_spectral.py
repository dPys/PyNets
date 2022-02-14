#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import itertools
import pytest
from pathlib import Path
import numpy as np
import logging
from pynets.statistics.individual import spectral

logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize("granularity", [
    10,
    100,
    pytest.param(1, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("n_graphs", [
    pytest.param(1, marks=pytest.mark.xfail),
    4,
    pytest.param(0, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("modality", [
    'func',
    'dwi',
])
def test_omni(gen_mat_data, granularity, n_graphs, modality):
    atlas = "tmp"
    mat_dat = gen_mat_data(m=granularity, n=granularity, n_graphs=n_graphs,
                           asfile=True, mat_type='sb', modality=modality)
    pop_array = mat_dat['mat_list']
    pop_file_list = mat_dat['mat_file_list']

    output_file = spectral._omni_embed(pop_array, atlas, pop_file_list)

    emb_mat = np.load(output_file)

    assert Path(output_file).is_file() and output_file.endswith(".npy") and \
           emb_mat.shape[0] == granularity

    output_paths = spectral.build_omnetome(mat_dat['mat_file_list'])

    if modality == 'func':
        output_path = output_paths[1][0]
    elif modality == 'dwi':
        output_path = output_paths[0][0]

    emb_mat = np.load(output_path)
    assert Path(output_path).is_file() and output_path.endswith(".npy") \
           and emb_mat.shape[0] == granularity and emb_mat.shape[1] == 1


@pytest.mark.parametrize("granularity", [
    100,
    pytest.param(1, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("n_graphs", [
    pytest.param(1, marks=pytest.mark.xfail),
    2,
    pytest.param(0, marks=pytest.mark.xfail),
])
def test_MASE(gen_mat_data, granularity, n_graphs):
    mat_dat_func = gen_mat_data(m=granularity, n=granularity,
                                n_graphs=n_graphs, asfile=True,
                                mat_type='sb', modality='func')
    pop_array_func = mat_dat_func['mat_list']
    pop_file_list_func = mat_dat_func['mat_file_list']

    mat_dat_dwi = gen_mat_data(m=granularity, n=granularity,
                               n_graphs=n_graphs, asfile=True,
                               mat_type='sb', modality='dwi')
    pop_array_dwi = mat_dat_dwi['mat_list']
    pop_file_list_dwi = mat_dat_dwi['mat_file_list']

    atlas = 'tmp'
    output_file = spectral._mase_embed([pop_array_func[0], pop_array_dwi[0]],
                                       atlas,
                                       pop_file_list_func[0])

    emb_mat = np.load(output_file)

    assert Path(output_file).is_file() and output_file.endswith(".npy") and \
           emb_mat.shape[0] == granularity and emb_mat.shape[1] == 2

    output_path = spectral.build_masetome([list(itertools.product([
    pop_file_list_func, pop_file_list_dwi]))[0][0]])[0]
    emb_mat = np.load(output_path)
    assert Path(output_path).is_file() and output_path.endswith(".npy") \
           and emb_mat.shape[0] == granularity and emb_mat.shape[1] == 1


@pytest.mark.parametrize("granularity", [
    100,
    pytest.param(1, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("n_graphs", [
    pytest.param(4, marks=pytest.mark.xfail),
    1,
    pytest.param(0, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("modality", [
    'func',
    'dwi',
])
def test_ASE(gen_mat_data, granularity, n_graphs, modality):
    atlas = "tmp"
    mat_dat = gen_mat_data(m=granularity, n=granularity, n_graphs=n_graphs,
                           asfile=True, mat_type='sb', modality=modality)

    output_file = spectral._ase_embed(mat_dat['mat_list'][0], atlas,
                                      mat_dat['mat_file_list'][0])

    emb_mat = np.load(output_file)

    assert Path(output_file).is_file() and output_file.endswith(".npy") and \
           emb_mat.shape[0] == granularity


@pytest.mark.parametrize("granularity", [
    100,
    pytest.param(1, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("n_graphs", [
    2,
    1,
    pytest.param(0, marks=pytest.mark.xfail),
])
@pytest.mark.parametrize("modality", [
    'func',
    'dwi',
])
def test_build_asetomes(gen_mat_data, granularity, n_graphs, modality):
    mat_dat = gen_mat_data(m=granularity, n=granularity, n_graphs=n_graphs,
                           asfile=True, mat_type='sb', modality=modality)

    output_paths = spectral.build_asetomes(mat_dat['mat_file_list'])
    for output_file in output_paths:
        emb_mat = np.load(output_file)
        assert Path(output_file).is_file() and output_file.endswith(".npy") \
               and emb_mat.shape[0] == granularity
