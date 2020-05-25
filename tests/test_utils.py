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
import indexed_gzip
import pytest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_save_RSN_coords_and_labels_to_pickle():
    """
    Test save_RSN_coords_and_labels_to_pickle functionality
    """
    import tempfile

    base_dir = str(Path(__file__).parent/"examples")
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)
    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    network = None

    [coord_path, labels_path] = utils.save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network)

    assert os.path.isfile(coord_path) is True
    assert os.path.isfile(labels_path) is True


def test_save_nifti_parcels_map():
    """
    Test save_nifti_parcels_map functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)
    ID = '002'
    roi = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti)
    assert os.path.isfile(net_parcels_nii_path) is True


def test_save_ts_to_file():
    """
    Test save_ts_to_file functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)
    base_dir = str(Path(__file__).parent/"examples")
    roi = None
    smooth = None
    hpass = None
    network = None
    node_size = 'parc'
    ID = '002'
    ts_within_nodes = f"{base_dir}/miscellaneous/002_Default_rsn_net_ts.npy"

    out_path_ts = utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, smooth, hpass, node_size)
    assert os.path.isfile(out_path_ts) is True


# def test_build_embedded_connectome():
#     """
#     Test build_embedded_connectome functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     ID = '002'
#     multimodal = False
#     types = ['omni', 'mase']
#     est_path_iterlist = [f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.09.npy",
#                          f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.08.npy",
#                          f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.07.npy",
#                          f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.06.npy",
#                          f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.09.npy"]
#     for type in types:
#         out_path = utils.build_embedded_connectome(est_path_iterlist, ID, multimodal, type)
#         assert out_path is not None


def test_check_est_path_existence():
    """
    Test check_est_path_existence functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    est_path_iterlist = [f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.09.npy",
                         f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.08.npy",
                         f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.07.npy",
                         f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.06.npy",
                         f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-csd_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.1.npy",
                         f"{base_dir}/miscellaneous/bad_path.npy"]
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_iterlist)
    assert est_path_list_ex is not None


# def test_collect_pandas_df():
#     """
#     Test collect_pandas_df functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     multi_nets = ['Default', 'SalVentAttn']
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     multimodal = False
#     net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.1_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.1_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.1_net_mets.csv"]
#     utils.collect_pandas_df(network, ID, net_mets_csv_list, plot_switch, multi_nets, multimodal)
#
#
# def test_collect_pandas_df_make():
#     """
#     Test collect_pandas_df_make functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     net_pickle_mt_list = [f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csa_thrtype-PROP_thr-0.1_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-csd_thrtype-PROP_thr-0.1_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.05_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.06_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.07_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.08_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.09_net_mets.csv",
#                           f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-tensor_thrtype-PROP_thr-0.1_net_mets.csv"]
#     utils.collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)

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
    os.chdir(dir_path)
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    thr_type = 'prop'
    thr = 0.75

    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc)
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
    os.chdir(dir_path)
    network = 'Default'
    ID = '002'
    roi = None
    directget = 'prob'
    max_length = 200
    conn_model = 'corr'
    thr_type = 'prop'
    target_samples = 10
    track_type = 'local'
    thr = 0.75

    est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, target_samples,
                                          track_type, thr_type, parc,
                                          directget, max_length)
    assert est_path is not None


def test_create_csv_path():
    """
    Test create_csv_path functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)

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

    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi,
                                          dir_path, node_size, smooth,
                                          thr_type, hpass, parc)
    out_path = utils.create_csv_path(dir_path, est_path)
    assert out_path is not None


@pytest.mark.parametrize("fmt", ['edgelist_csv', 'gpickle', 'graphml', 'txt',
                                 'npy', 'edgelist_ssv',
                                 pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))])
def test_save_mat(fmt):
    import glob as glob
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)

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
    os.chdir(dir_path)
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None

    unthr_path_func = utils.create_raw_path_func(ID, network, conn_model, roi,
                                                 dir_path, node_size, smooth,
                                                 hpass, parc)
    assert unthr_path_func is not None

    network = 'Default'
    target_samples = 1000
    track_type = 'local'
    conn_model = 'csd'
    roi = None
    directget = 'prob'
    max_length = 200

    unthr_path_diff = utils.create_raw_path_diff(ID, network, conn_model, roi,
                                                 dir_path, node_size,
                                                 target_samples, track_type,
                                                 parc, directget, max_length)
    assert unthr_path_diff is not None



@pytest.mark.parametrize("atlas", ['Power', 'Shirer', 'Shen', 'Smith',
                                    pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))])
@pytest.mark.parametrize("input", ['fmri', 'dmri'])
def test_do_dir_path(atlas, input):
    """
    Test do_dir_path functionality
    """
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)
    base_dir = str(Path(__file__).parent/"examples")

    if input == 'fmri':
        in_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    elif input == 'dmri':
        in_file = f"{base_dir}/BIDS/sub-0025427/ses-1/dwi/final_preprocessed_dwi.nii.gz"

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
    os.chdir(dir_path)
    base_dir = str(Path(__file__).parent/"examples")
    conn_model = 'corr'
    est_path = f"{base_dir}/miscellaneous/0021001_modality-dwi_rsn-Default_est-tensor_nodetype-parc_samples-100000streams_tt-particle_dg-prob_ml-10_thrtype-DENS_thr-0.09.npy"
    network = 'Default'
    thr = 0.09
    prune = True
    ID = '0021001'
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
    est_path_func = f"{base_dir}/miscellaneous/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy"
    est_path_struct = f"{base_dir}/miscellaneous/0025427_modality-dwi_est-csd_nodetype-parc_samples-10000streams_tt-particle_dg-prob_ml-10_thrtype-PROP_thr-1.0.npy"
    network_func = 'Default'
    network_struct = 'Default'
    thr_func = 0.95
    thr_struct = 1.00
    prune_func = True
    prune_struct = False
    ID_func = '002'
    ID_struct = '0025427'
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


# @pytest.mark.parametrize("embed_multimodal_multiplex",
#                          [[None, False, 0], pytest.param(['omni', True, 1],
#                                                          marks=pytest.mark.xfail(raises=IndexError))])
# def test_pass_meta_outs(embed_multimodal_multiplex):
#     """
#     Test pass_meta_outs functionality
#
#     Note: omni argument may be failing due to functions in netmotifs or due to
#     an unexpected input. Marked for xfail and should be updated after tests for
#     netmotifs are created.
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     conn_model_func = 'corr'
#     conn_model_struct = 'csa'
#     network_func = 'Default'
#     network_struct = 'Default'
#     thr_func = 0.6
#     thr_struct = 0.8
#     prune_func = True
#     prune_struct = False
#     ID_func = '002'
#     ID_struct = '002'
#     roi_func = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
#     roi_struct = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
#     norm_func = 1
#     norm_struct = 2
#     binary_func = False
#     binary_struct = True
#     embed = embed_multimodal_multiplex[0]
#     multimodal = embed_multimodal_multiplex[1]
#     multiplex = embed_multimodal_multiplex[2]
#
#     node_size = 6
#     smooth = 6
#     thr_type = 'prop'
#     hpass = 100
#     parc = True
#     directget = 'prob'
#     max_length = 200
#     thr_type = 'prop'
#     target_samples = 10
#     track_type = 'local'
#     conn_matrix_diff = np.random.rand(10, 10)
#     conn_matrix_func = np.random.rand(10, 10)
#
#     est_path_func = utils.create_est_path_func(ID_func, network_func,
#                                                conn_model_func, thr_func,
#                                                roi_func, func_path, node_size,
#                                                smooth, thr_type, hpass,
#                                                parc)
#
#     est_path_diff = utils.create_est_path_diff(ID_struct, network_struct,
#                                                conn_model_struct, thr_struct,
#                                                roi_struct, dmri_path,
#                                                node_size, target_samples,
#                                                track_type, thr_type, parc,
#                                                directget, max_length)
#
#     utils.save_mat(conn_matrix_diff, est_path_diff)
#     utils.save_mat(conn_matrix_func, est_path_func)
#
#     [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist,
#      norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
#         conn_model_func, est_path_func, network_func, thr_func, prune_func, ID_func, roi_func, norm_func, binary_func,
#         conn_model_struct, est_path_diff, network_struct, thr_struct, prune_struct, ID_struct, roi_struct,
#         norm_struct, binary_struct)
#
#     [conn_model_iterlist_out, est_path_iterlist_out, network_iterlist_out, thr_iterlist_out, prune_iterlist_out,
#      ID_iterlist_out, roi_iterlist_out, norm_iterlist_out, binary_iterlist_out] = utils.pass_meta_outs(
#         conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist,
#         roi_iterlist, norm_iterlist, binary_iterlist, embed, multimodal, multiplex)
#
#     assert conn_model_iterlist_out is not None
#     assert est_path_iterlist_out is not None
#     assert network_iterlist_out is not None
#     assert thr_iterlist_out is not None
#     assert prune_iterlist_out is not None
#     assert ID_iterlist_out is not None
#     assert roi_iterlist_out is not None
#     assert norm_iterlist_out is not None
#     assert binary_iterlist_out is not None


def test_collectpandasjoin():
    base_dir = str(Path(__file__).parent/"examples")
    net_mets_csv = f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_est-cov_thrtype-PROP_thr-0.95_net_metrics.csv"
    net_mets_csv_out = utils.collectpandasjoin(net_mets_csv)

    assert net_mets_csv == net_mets_csv_out


def test_collect_pandas_df():
    base_dir = str(Path(__file__).parent/"examples")
    network = None
    ID = '002'
    plot_switch = False
    multi_nets = None
    multimodal = False
    if multi_nets is not None and multimodal is False:
        net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.3_net_mets.csv",
                             f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.2_net_mets.csv"]
    elif multi_nets is not None and multimodal is True:
        net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.3_net_mets.csv",
                             f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.2_net_mets.csv"]
    elif multi_nets is None and multimodal is False:
        net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.3_net_mets.csv",
                             f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.2_net_mets.csv"]
    elif multi_nets is None and multimodal is True:
        net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.3_net_mets.csv",
                             f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.2_net_mets.csv"]
    else:
        return

    combination_complete = utils.collect_pandas_df(network, ID,
                                                   net_mets_csv_list,
                                                   plot_switch, multi_nets,
                                                   multimodal)

    assert combination_complete is not None


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
def test_build_hp_dict(modality):
    import tempfile

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path)
    os.chdir(dir_path)
    base_dir = str(Path(__file__).parent / "examples")
    atlas = 'Power'

    if modality == 'func':
        file_renamed = f"{base_dir}/miscellaneous/graphs/002_modality-func_rsn-Default_est-cov_nodetype-spheres-2mm_smooth-2fwhm_hpass-0.1Hz_thrtype-PROP_thr-0.95.npy"
        hyperparams = ['modality', 'rsn', 'est', 'nodetype', 'smooth', 'hpass', 'thrtype', 'thr']
    elif modality == 'dwi':
        file_renamed = f"{base_dir}/miscellaneous/graphs/0025427_modality-dwi_est-csd_nodetype-parc_samples-10000streams_tt-particle_dg-prob_ml-10_thrtype-PROP_thr-1.0.npy"
        hyperparams = ['modality', 'est', 'nodetype', 'samples', 'tt', 'dg', 'ml', 'thrtype', 'thr']

    hyperparam_dict = dict.fromkeys(hyperparams)
    file_renamed = file_renamed.split('graphs/')[1]
    hyperparam_dict, hyperparams = utils.build_hp_dict(file_renamed, atlas,
                                                       modality,
                                                       hyperparam_dict,
                                                       hyperparams)

    # test_build_sql_db
    if modality == 'func':
        import pandas as pd
        ID = '002'
        hyperparams.append('atlas')
        hyperparams.append('AUC')
        df_summary_auc = {'AUC': 0.8}
        db = utils.build_sql_db(dir_path, ID)
        db.create_modality_table('func')
        db.add_hp_columns(hyperparams)
        db.add_row_from_df(pd.DataFrame([{'AUC': 0.8}], index=[0]),
                           hyperparam_dict)
