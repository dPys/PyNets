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
import indexed_gzip
import nibabel as nib


def test_save_RSN_coords_and_labels_to_pickle():
    """
    Test save_RSN_coords_and_labels_to_pickle functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    coord_file_path = dir_path + '/DesikanKlein2012/Default_coords_rsn.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = dir_path + '/DesikanKlein2012/Default_labels_rsn.pkl'
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
    base_dir = str(Path(__file__).parent/"examples")
    ID = '002'
    dir_path = base_dir + '/002/fmri'
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
    base_dir = str(Path(__file__).parent/"examples")
    roi = None
    c_boot = 3
    smooth=2
    hpass=None
    network = None
    node_size = 'parc'
    ID = '002'
    dir_path = base_dir + '/002/fmri'
    ts_within_nodes = '/tmp/'
    out_path_ts = utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot, smooth, hpass, node_size)
    assert os.path.isfile(out_path_ts) is True


# def test_build_embedded_connectome():
#     """
#     Test build_embedded_connectome functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     ID = '002'
#     multimodal = False
#     types = ['omni', 'mase']
#     est_path_iterlist = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
#     for type in types:
#         out_path = utils.build_embedded_connectome(est_path_iterlist, ID, multimodal, type)
#         assert out_path is not None


def test_check_est_path_existence():
    """
    Test check_est_path_existence functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    est_path_list = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_list)
    assert est_path_list_ex is not None


# def test_collect_pandas_df():
#     """
#     Test collect_pandas_df functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     multi_nets = ['Default', 'SalVentAttn']
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     multimodal = False
#     net_mets_csv_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.1_parc.csv']
#     utils.collect_pandas_df(network, ID, net_mets_csv_list, plot_switch, multi_nets, multimodal)
#
#
# def test_collect_pandas_df_make():
#     """
#     Test collect_pandas_df_make functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     net_pickle_mt_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.1_parc.csv']
#     utils.collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)


def test_create_est_path_func():
    """
    Test create_est_path_diff functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    node_size = 6
    smooth = 6
    c_boot = 1000
    hpass = 100
    parc = True

    # Cross test various connectivity models, thresholds, and parc true/false.
    for conn_model in models:
        for val in range(1, 10):
            thr = round(val*0.1, 1)
            for thr_type in ['prop', 'abs', 'dens', 'mst', 'disp']:
                for parc in [True, False]:
                    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size,
                                                          smooth, c_boot,
                                               thr_type, hpass, parc)
                    assert est_path is not None


def test_create_est_path_diff():
    """
    Test create_est_path_func functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    node_size = 6

    for conn_model in models:
        for val in range(1, 10):
            thr = round(val*0.1, 1)
            for thr_type in ['prop', 'abs', 'dens', 'mst', 'disp']:
                for target_samples in range(0, 100, 1000):
                    for track_type in ['local', 'particle']:
                        for parc in [True, False]:
                            est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi,
                                                                  dir_path, node_size, target_samples,
                                                                  track_type, thr_type, parc)
                            assert est_path is not None


def test_create_csv_path():
    """
    Test create_csv_path functionality
    """
    base_dir = str(Path(__file__).parent/"examples")

    # fmri case
    dir_path = base_dir + '/002/fmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    node_size = 6
    smooth = 6
    c_boot = 100
    hpass = 100
    parc = True

    # Cross test various connectivity models, thresholds, and parc true/false.
    for conn_model in models:
        for val in range(1, 10):
            thr = round(val*0.1, 1)
            for thr_type in ['prop', 'abs', 'dens', 'mst', 'disp']:
                for parc in [True, False]:
                    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size,
                                                          smooth, c_boot,
                                               thr_type, hpass, parc)
                    out_path = utils.create_csv_path(dir_path, est_path)
                    assert out_path is not None

    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    node_size = 6

    for conn_model in models:
        for val in range(1, 10):
            thr = round(val*0.1, 1)
            for thr_type in ['prop', 'abs', 'dens', 'mst', 'disp']:
                for target_samples in range(0, 100, 1000):
                    for track_type in ['local', 'particle']:
                        for parc in [True, False]:
                            est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi,
                                                                  dir_path, node_size, target_samples,
                                                                  track_type, thr_type, parc)
                            out_path = utils.create_csv_path(dir_path, est_path)
                            assert out_path is not None


def test_create_unthr_path():
    """
    Test create_unthr_path functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = 'Default'
    ID = '002'
    node_size = 2
    smooth = 2
    c_boot = 10
    hpass = 2
    parc_types = [True, False]
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    for conn_model in models:
        for parc in parc_types:
            unthr_path_func = utils.create_raw_path_func(ID, network, conn_model, roi, dir_path, node_size, smooth,
                                                         c_boot, hpass, parc)
            assert unthr_path_func is not None

    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    node_size = 2
    target_samples = 1000
    track_type = 'local'
    parc_types = [True, False]
    models = ['csd', 'csa']
    roi = None
    for conn_model in models:
        for parc in parc_types:
            unthr_path_diff = utils.create_raw_path_diff(ID, network, conn_model, roi, dir_path, node_size,
                                                         target_samples, track_type, parc)
            assert unthr_path_diff is not None


def test_do_dir_path():
    """
    Test do_dir_path functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    func_path = base_dir + '/002/fmri'
    dwi_path = base_dir + '/002/dmri'
    in_func = func_path + '/002.nii.gz'
    in_dwi = dwi_path + '/std_dmri/iso_eddy_corrected_data_denoised_pre_reor.nii.gz'
    in_files = [in_func, in_dwi]

    atlases = ['Power', 'Shirer', 'Shen', 'Smith']
    for inputs in in_files:
        for atlas in atlases:
            dir_path = utils.do_dir_path(atlas, inputs)
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
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    conn_model = 'corr'
    est_path = dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy'
    network = 'Default'
    thr = 0.5
    prune = True
    ID = '002'
    roi = dir_path + 'pDMN_3_bin_mask.nii.gz'
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
    dmri_path = base_dir + '/002/dmri'
    func_path = base_dir + '/002/fmri'

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = func_path + '/002_Default_est_cov_0.95propTEST_mm3_nb2_fwhm0.1_Hz.npy'
    est_path_struct = dmri_path + '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    network_func = 'Default'
    network_struct = 'Default'
    thr_func = 0.6
    thr_struct = 0.8
    prune_func = True
    prune_struct = False
    ID_func = '002'
    ID_struct = '002'
    roi_func = func_path + '/pDMN_3_bin_mask.nii.gz'
    roi_struct = func_path + '/pDMN_3_bin_mask.nii.gz'
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


def test_pass_meta_outs():
    """
    Test pass_meta_outs functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dmri_path = base_dir + '/002/dmri'
    func_path = base_dir + '/002/fmri'

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = dmri_path + '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    est_path_struct = dmri_path + '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    network_func = 'Default'
    network_struct = 'Default'
    thr_func = 0.6
    thr_struct = 0.8
    prune_func = True
    prune_struct = False
    ID_func = '002'
    ID_struct = '002'
    roi_func = func_path + 'pDMN_3_bin_mask.nii.gz'
    roi_struct = func_path + 'pDMN_3_bin_mask.nii.gz'
    norm_func = 1
    norm_struct = 2
    binary_func = False
    binary_struct = True

    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist,
     norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
        conn_model_func, est_path_func, network_func, thr_func, prune_func, ID_func, roi_func, norm_func, binary_func,
        conn_model_struct, est_path_struct, network_struct, thr_struct, prune_struct, ID_struct, roi_struct,
        norm_struct, binary_struct)

    [conn_model_iterlist_out, est_path_iterlist_out, network_iterlist_out, thr_iterlist_out, prune_iterlist_out,
     ID_iterlist_out, roi_iterlist_out, norm_iterlist_out, binary_iterlist_out] = utils.pass_meta_outs(
        conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist, prune_iterlist, ID_iterlist,
        roi_iterlist, norm_iterlist, binary_iterlist, embed=None, multimodal=False, multiplex=False)

    assert conn_model_iterlist_out is not None
    assert est_path_iterlist_out is not None
    assert network_iterlist_out is not None
    assert thr_iterlist_out is not None
    assert prune_iterlist_out is not None
    assert ID_iterlist_out is not None
    assert roi_iterlist_out is not None
    assert norm_iterlist_out is not None
    assert binary_iterlist_out is not None
