#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import os
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

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_save_coords_and_labels_to_json():
    """
    Test save_RSN_coords_and_labels_to_json functionality
    """
    import tempfile

    base_dir = str(Path(__file__).parent/"examples")
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    network = 'Default'
    indices = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    nodes_path = utils.save_coords_and_labels_to_json(coords, labels,
                                                      dir_path, network,
                                                      indices)

    assert os.path.isfile(nodes_path) is True


def test_save_nifti_parcels_map():
    """
    Test save_nifti_parcels_map functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    ID = '002'
    vox_size = '2mm'
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(ID, dir_path,
                                                        network,
                                                        net_parcels_map_nifti,
                                                        vox_size)
    assert os.path.isfile(net_parcels_nii_path) is True


def test_save_ts_to_file():
    """
    Test save_ts_to_file functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    base_dir = str(Path(__file__).parent/"examples")
    roi = None
    smooth = None
    hpass = None
    network = None
    node_size = 'parc'
    extract_strategy = 'mean'
    ID = '002'
    ts_within_nodes = f"{base_dir}/miscellaneous/002_Default_rsn_net_ts.npy"

    out_path_ts = utils.save_ts_to_file(roi, network, ID, dir_path,
                                        ts_within_nodes, smooth, hpass,
                                        node_size, extract_strategy)
    assert os.path.isfile(out_path_ts) is True


def test_check_est_path_existence():
    """
    Test check_est_path_existence functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    est_path_iterlist = [f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.09.npy",
                         f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.08.npy",
                         f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.07.npy",
                         f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.06.npy",
                         f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.1.npy",
                         f"{base_dir}/miscellaneous/bad_path.npy"]
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_iterlist)
    assert est_path_list_ex is not None


@pytest.mark.parametrize("embed", [False, True])
@pytest.mark.parametrize("plot_switch", [False, True])
def test_collect_pandas_df(plot_switch, embed):
    """
    Test collect_pandas_df_make functionality
    """
    import glob
    base_dir = str(Path(__file__).parent/"examples")
    multi_nets = None
    multimodal = False
    network = None
    ID = '002'
    net_mets_csv_list = [i for i in glob.glob(f"{base_dir}/topology/*.csv")
                         if '_neat.csv' not in i]
    out = utils.collect_pandas_df(network, ID, net_mets_csv_list,
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

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    thr_type = 'prop'
    thr = 0.75
    extract_strategy = 'mean'

    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc,
                                          extract_strategy)
    assert est_path is not None


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_create_est_path_diff(node_size, parc):
    """
    Test create_est_path_func functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    network = 'Default'
    ID = '002'
    roi = None
    directget = 'prob'
    min_length = 20
    conn_model = 'corr'
    thr_type = 'prop'
    target_samples = 10
    track_type = 'local'
    thr = 0.75
    error_margin = 6

    est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, target_samples,
                                          track_type, thr_type, parc,
                                          directget, min_length, error_margin)
    assert est_path is not None


def test_create_csv_path():
    """
    Test create_csv_path functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)

    # fmri case
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    node_size = 6
    smooth = 6
    hpass = 100
    parc = True
    thr = 0.75
    thr_type = 'prop'
    extract_strategy = 'mean'

    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc,
                                          extract_strategy)
    out_path = utils.create_csv_path(dir_path, est_path)
    assert out_path is not None


@pytest.mark.parametrize("fmt", ['edgelist_csv', 'gpickle', 'graphml', 'txt',
                                 'npy', 'edgelist_ssv',
                                 pytest.param(None, marks=pytest.mark.xfail(
                                     raises=ValueError))])
def test_save_mat(fmt):
    import glob as glob
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)

    est_path = f"{dir_path}/G_out"
    conn_matrix = np.random.rand(10, 10)

    utils.save_mat(conn_matrix, est_path, fmt)

    save_mat_path = glob.glob(est_path + '*')[0]
    assert os.path.isfile(save_mat_path)


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("hpass", [100, None])
@pytest.mark.parametrize("smooth", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_create_unthr_path(node_size, hpass, smooth, parc):
    """
    Test create_unthr_path functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    extract_strategy = 'mean'

    unthr_path_func = utils.create_raw_path_func(ID, network, conn_model, roi,
                                                 dir_path, node_size, smooth,
                                                 hpass, parc, extract_strategy)
    assert unthr_path_func is not None

    network = 'Default'
    target_samples = 1000
    track_type = 'local'
    conn_model = 'csd'
    roi = None
    directget = 'prob'
    min_length = 20
    error_margin = 6

    unthr_path_diff = utils.create_raw_path_diff(ID, network, conn_model, roi,
                                                 dir_path, node_size,
                                                 target_samples, track_type,
                                                 parc, directget, min_length,
                                                 error_margin)
    assert unthr_path_diff is not None



@pytest.mark.parametrize("atlas", ['Power', 'Shirer', 'Shen', 'Smith',
                                    pytest.param(None,
                                                 marks=pytest.mark.xfail(raises=ValueError))])
@pytest.mark.parametrize("input", ['fmri', 'dmri'])
def test_do_dir_path(atlas, input):
    """
    Test do_dir_path functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    base_dir = str(Path(__file__).parent/"examples")

    if input == 'fmri':
        in_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    elif input == 'dmri':
        in_file = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz"

    # Delete existing atlas dirs in in_file parent
    atlas_dir = os.path.dirname(os.path.realpath(in_file)) + '/' + str(atlas)

    dir_path = utils.do_dir_path(atlas, atlas_dir)
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


def test_pass_meta_ins():
    """
    Test pass_meta_ins functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    base_dir = str(Path(__file__).parent/"examples")
    conn_model = 'corr'
    est_path = f"{base_dir}/miscellaneous/sub-0021001_modality-dwi_rsn-Default_model-tensor_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-DENS_thr-0.09.npy"
    network = 'Default'
    thr = 0.09
    prune = True
    ID = 'sub-0021001'
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    norm = 10
    binary = True

    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist,
        norm_iterlist, binary_iterlist] = utils.pass_meta_ins(conn_model, est_path, network, thr, prune, ID, roi, norm,
                                                              binary)

    assert conn_model_iterlist is not None
    assert est_path_iterlist is not None
    assert network_iterlist is not None
    assert thr_iterlist is not None
    assert prune_iterlist is not None
    assert ID_iterlist is not None
    assert roi_iterlist is not None
    assert norm_iterlist is not None
    assert binary_iterlist is not None


def test_pass_meta_ins_multi():
    """
    Test pass_meta_ins_multi functionality
    """
    base_dir = str(Path(__file__).parent/"examples")

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = f"{base_dir}/miscellaneous/002_modality-func_rsn-Default_model-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_template-MNI152_T1_thrtype-PROP_thr-0.95.npy"
    est_path_struct = f"{base_dir}/miscellaneous/0025427_modality-dwi_model-csd_nodetype-parc_samples-10000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-PROP_thr-1.0.npy"
    network_func = 'Default'
    network_struct = 'Default'
    thr_func = 0.95
    thr_struct = 1.00
    prune_func = True
    prune_struct = False
    ID_func = '002'
    ID_struct = '25659'
    roi_func = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    roi_struct = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    norm_func = 1
    norm_struct = 2
    binary_func = False
    binary_struct = True

    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist,
     norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
        conn_model_func, est_path_func, network_func, thr_func, prune_func, ID_func, roi_func, norm_func, binary_func,
        conn_model_struct, est_path_struct, network_struct, thr_struct, prune_struct, ID_struct, roi_struct,
        norm_struct, binary_struct)

    assert len(conn_model_iterlist) == 2
    assert len(est_path_iterlist) == 2
    assert len(network_iterlist) == 2
    assert len(thr_iterlist) == 2
    assert len(prune_iterlist) == 2
    assert len(ID_iterlist) == 2
    assert len(roi_iterlist) == 2
    assert len(norm_iterlist) == 2
    assert len(binary_iterlist) == 2


def test_collectpandasjoin():
    base_dir = str(Path(__file__).parent/"examples")
    net_mets_csv = f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_model-cov_template-MNI152_T1_thrtype-PROP_thr-0.95_net_metrics.csv"
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
def test_build_mp_dict(modality):
    import tempfile
    from pynets.stats.utils import build_mp_dict
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    base_dir = str(Path(__file__).parent / "examples")

    if modality == 'func':
        file_renamed = f"{base_dir}/miscellaneous/graphs/graph_sub-002_modality-func_rsn-Default_model-cov_template-MNI152_T1_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_template-MNI152_T1_thrtype-PROP_thr-0.95.npy"
    elif modality == 'dwi':
        file_renamed = f"{base_dir}/miscellaneous/graphs/0025427_modality-dwi_model-csd_nodetype-parc_samples-10000streams_tt-particle_dg-prob_ml-10_template-MNI152_T1_thrtype-PROP_thr-1.0.npy"
    gen_metaparams = ['modality', 'model', 'nodetype', 'template']

    metaparam_dict = {}
    file_renamed = file_renamed.split('graphs/')[1]
    metaparam_dict, metaparams = build_mp_dict(file_renamed,
                                                 modality,
                                                 metaparam_dict,
                                                 gen_metaparams)

    # test_build_sql_db
    if modality == 'func':
        import pandas as pd
        ID = '002'
        metaparams.append('atlas')
        metaparams.append('AUC')
        df_summary_auc = {'AUC': 0.8}
        db = utils.build_sql_db(dir_path, ID)
        db.create_modality_table('func')
        db.add_hp_columns(metaparams)
        db.add_row_from_df(pd.DataFrame([{'AUC': 0.8}], index=[0]),
                           metaparam_dict)
