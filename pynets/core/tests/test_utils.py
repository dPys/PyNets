#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017
"""
import numpy as np
import os
import pkg_resources
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.core import utils
import nibabel as nib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import pytest
import logging
import tempfile
from nilearn._utils import data_gen

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_save_coords_and_labels_to_json(connectivity_data):
    """
    Test save_RSN_coords_and_labels_to_json functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    coords = connectivity_data['coords']
    labels = connectivity_data['labels']
    subnet = 'Default'
    indices = np.arange(len(coords) + 1)[np.arange(len(coords) + 1)
                                         != 0].tolist()

    nodes_path = utils.save_coords_and_labels_to_json(coords, labels,
                                                      dir_path, subnet,
                                                      indices)

    assert os.path.isfile(nodes_path) is True
    tmp.cleanup()


def test_save_nifti_parcels_map():
    """
    Test save_nifti_parcels_map functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    ID = '002'
    vox_size = '2mm'
    subnet = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(ID, dir_path,
                                                        subnet,
                                                        net_parcels_map_nifti,
                                                        vox_size)
    assert os.path.isfile(net_parcels_nii_path) is True
    tmp.cleanup()


def test_save_ts_to_file():
    """
    Test save_ts_to_file functionality
    """

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    roi = None
    smooth = None
    hpass = None
    subnet = None
    node_size = 'parc'
    signal = 'mean'
    ID = '002'
    ts_within_nodes = data_gen.generate_timeseries(10, 10)

    out_path_ts = utils.save_ts_to_file(roi, subnet, ID, dir_path,
                                        ts_within_nodes, smooth, hpass,
                                        node_size, signal)
    assert os.path.isfile(out_path_ts) is True
    tmp.cleanup()


def test_check_est_path_existence(gen_mat_data):
    """
    Test check_est_path_existence functionality
    """

    est_path_iterlist = gen_mat_data(n_graphs=10)['mat_file_list']
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_iterlist)
    assert est_path_list_ex is not None


@pytest.mark.parametrize("embed", [False, True])
@pytest.mark.parametrize("plot_switch", [False, True])
def test_collect_pandas_df(plot_switch, embed):
    """
    Test collect_pandas_df_make functionality
    """
    import glob
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    multi_nets = None
    multimodal = False
    subnet = None
    ID = '002'
    net_mets_csv_list = [i for i in glob.glob(f"{base_dir}/topology/*.csv")
                         if '_neat.csv' not in i]
    out = utils.collect_pandas_df(subnet, ID, net_mets_csv_list,
                                  plot_switch, multi_nets, multimodal, embed)
    assert out is True
    assert isinstance(net_mets_csv_list, list)
    assert len(net_mets_csv_list) == 9


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("hpass", [100, None])
@pytest.mark.parametrize("smooth", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_create_est_path_func(node_size, hpass, smooth, parc):
    """
    Test create_est_path_diff functionality
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    subnet = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    thr_type = 'prop'
    thr = 0.75
    signal = 'mean'

    est_path = utils.create_est_path_func(ID, subnet, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc,
                                          signal)
    assert est_path is not None
    tmp.cleanup()


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_create_est_path_diff(node_size, parc):
    """
    Test create_est_path_func functionality
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    subnet = 'Default'
    ID = '002'
    roi = None
    traversal = 'prob'
    min_length = 20
    conn_model = 'corr'
    thr_type = 'prop'
    track_type = 'local'
    thr = 0.75
    error_margin = 6

    est_path = utils.create_est_path_diff(ID, subnet, conn_model, thr, roi,
                                          dir_path, node_size,
                                          track_type, thr_type, parc,
                                          traversal, min_length, error_margin)
    assert est_path is not None
    tmp.cleanup()


def test_create_csv_path():
    """
    Test create_csv_path functionality
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    # fmri case
    subnet = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    node_size = 6
    smooth = 6
    hpass = 100
    parc = True
    thr = 0.75
    thr_type = 'prop'
    signal = 'mean'

    est_path = utils.create_est_path_func(ID, subnet, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc,
                                          signal)
    out_path = utils.create_csv_path(dir_path, est_path)
    assert out_path is not None
    tmp.cleanup()

@pytest.mark.parametrize("fmt", ['edgelist_csv', 'gpickle', 'graphml', 'txt',
                                 'npy', 'edgelist_ssv',
                                 pytest.param(None, marks=pytest.mark.xfail(
                                     raises=ValueError))])
def test_save_mat(fmt):
    import glob as glob
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    est_path = f"{dir_path}/G_out"
    conn_matrix = np.random.rand(10, 10)

    utils.save_mat(conn_matrix, est_path, fmt)

    save_mat_path = glob.glob(est_path + '*')[0]
    assert os.path.isfile(save_mat_path)


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("hpass", [100, 0])
@pytest.mark.parametrize("smooth", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_create_unthr_path(node_size, hpass, smooth, parc):
    """
    Test create_unthr_path functionality
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    subnet = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    signal = 'mean'

    unthr_path_func = utils.create_raw_path_func(ID, subnet, conn_model, roi,
                                                 dir_path, node_size, smooth,
                                                 hpass, parc, signal)
    assert unthr_path_func is not None

    subnet = 'Default'
    track_type = 'local'
    conn_model = 'csd'
    roi = None
    traversal = 'prob'
    min_length = 20
    error_margin = 6

    unthr_path_diff = utils.create_raw_path_diff(ID, subnet, conn_model, roi,
                                                 dir_path, node_size,
                                                 track_type,
                                                 parc, traversal, min_length,
                                                 error_margin)
    assert unthr_path_diff is not None
    tmp.cleanup()


@pytest.mark.parametrize("atlas", ['Power', 'Shirer', 'Shen', 'Smith',
                                    pytest.param(None,
                                                 marks=pytest.mark.xfail(
                                                     raises=ValueError))])
@pytest.mark.parametrize("input", ['fmri', 'dmri'])
def test_do_dir_path(atlas, input):
    """
    Test do_dir_path functionality
    """

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

    if input == 'fmri':
        in_file = f"{base_dir}/003/func/sub-003_ses-01_task-rest_bold.nii.gz"
    elif input == 'dmri':
        in_file = f"{base_dir}/003/dmri/sub-003_dwi.nii.gz"

    # Delete existing atlas dirs in in_file parent
    dir_path = utils.do_dir_path(
        atlas, f"{os.path.dirname(os.path.realpath(in_file))}")
    assert dir_path is not None


def test_flatten():
    """
    Test list flatten functionality
    """
    # Slow, but successfully flattens a large array
    l = np.random.rand(3, 3, 3).tolist()
    l = utils.flatten(l)

    i = 0
    for item in l:
        i += 1
    assert i == (3*3*3)


def test_get_file():
    """
    Test get_file functionality
    """
    base_path = utils.get_file()
    assert base_path is not None


def test_merge_dicts():
    """
    Test merge_dicts functionality
    """
    x = {
        'a': 1,
        'b': 2,
        'c': 3
    }
    y = {
        'd': 4,
        'e': 5,
        'f': 6
    }
    z = utils.merge_dicts(x, y)

    dic_len = len(x)+len(y)
    assert len(z) == dic_len


def test_pass_meta_ins(gen_mat_data, random_mni_roi_data):
    """
    Test pass_meta_ins functionality
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    conn_model = 'corr'
    est_path = gen_mat_data()['mat_file_list'][0]
    subnet = 'Default'
    thr = 0.09
    prune = True
    ID = 'sub-0021001'
    roi = random_mni_roi_data['roi_file']
    norm = 10
    binary = True

    [conn_model_iterlist, est_path_iterlist, subnet_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist,
        norm_iterlist, binary_iterlist] = utils.pass_meta_ins(
        conn_model, est_path, subnet, thr, prune, ID, roi, norm, binary)

    assert conn_model_iterlist is not None
    assert est_path_iterlist is not None
    assert subnet_iterlist is not None
    assert thr_iterlist is not None
    assert prune_iterlist is not None
    assert ID_iterlist is not None
    assert roi_iterlist is not None
    assert norm_iterlist is not None
    assert binary_iterlist is not None
    tmp.cleanup()

def test_pass_meta_ins_multi(gen_mat_data, random_mni_roi_data):
    """
    Test pass_meta_ins_multi functionality
    """
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = gen_mat_data(
        binary=True, mat_type='er')['mat_file_list'][0]
    est_path_struct = gen_mat_data()['mat_file_list'][0]
    subnet_func = 'Default'
    subnet_struct = 'Default'
    thr_func = 0.95
    thr_struct = 1.00
    prune_func = True
    prune_struct = False
    ID_func = '002'
    ID_struct = '25659'
    roi_func = random_mni_roi_data['roi_file']
    roi_struct = random_mni_roi_data['roi_file']
    norm_func = 1
    norm_struct = 2
    binary_func = False
    binary_struct = True

    [conn_model_iterlist, est_path_iterlist, subnet_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist,
     norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
        conn_model_func, est_path_func, subnet_func, thr_func, prune_func,
        ID_func, roi_func, norm_func, binary_func,
        conn_model_struct, est_path_struct, subnet_struct, thr_struct,
        prune_struct, ID_struct, roi_struct,
        norm_struct, binary_struct)

    assert len(conn_model_iterlist) == 2
    assert len(est_path_iterlist) == 2
    assert len(subnet_iterlist) == 2
    assert len(thr_iterlist) == 2
    assert len(prune_iterlist) == 2
    assert len(ID_iterlist) == 2
    assert len(roi_iterlist) == 2
    assert len(norm_iterlist) == 2
    assert len(binary_iterlist) == 2


def test_collectpandasjoin():
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    net_mets_csv = f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_" \
                   f"model-cov_template-MNI152_T1_thrtype-PROP_thr-0.95_" \
                   f"net_metrics.csv"
    net_mets_csv_out = utils.collectpandasjoin(net_mets_csv)

    assert net_mets_csv == net_mets_csv_out


@pytest.mark.parametrize("x", [[1, 2, 3], 1])
def test_as_list(x):
    y = utils.as_list(x)
    assert isinstance(y, list)


@pytest.mark.parametrize("s", [0, 2])
def test_timeout(s):
    import time
    @utils.timeout(1)
    def t_sleep(sec):
        try:
            time.sleep(sec)
        except:
            pass

    t_sleep(s)


@pytest.mark.parametrize("modality", ['func', 'dwi'])
def test_build_mp_dict(gen_mat_data, modality):
    import tempfile
    from pynets.statistics.utils import build_mp_dict

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    if modality == 'func':
        file_orig = gen_mat_data(
        binary=True, mat_type='er')['mat_file_list'][0]
        file_renamed = f"{os.path.dirname(file_orig)}/graph_sub-002_modality-" \
                       f"func_rsn-Default_model-cov_template-MNI152_T1_" \
                       f"nodetype-spheres-2mm_tol-2fwhm_hpass-0.1Hz_signal-" \
                       f"mean_template-MNI152_T1_thrtype-PROP_thr-0.95.npy"
    elif modality == 'dwi':
        file_orig = gen_mat_data()['mat_file_list'][0]
        file_renamed = f"{os.path.dirname(file_orig)}/0025427_" \
                       f"modality-dwi_model-csd_nodetype-parc_tt-particle_" \
                       f"traversal-prob_ml-10_tol-10_template-MNI152_T1_" \
                       f"thrtype-PROP_thr-1.0.npy"

    os.rename(file_orig, file_renamed)

    gen_metaparams = ['modality', 'model', 'nodetype', 'template']

    metaparam_dict = {}
    metaparam_dict, metaparams = build_mp_dict(file_renamed,
                                                 modality,
                                                 metaparam_dict,
                                                 gen_metaparams)
    assert metaparam_dict is not None
    assert metaparams is not None
    tmp.cleanup()
