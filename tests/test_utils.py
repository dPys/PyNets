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

# Fails


def test_export_to_pandas():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    csv_loc = dir_path + '/whole_brain_cluster_labels_PCA200/002_net_metrics_sps_0.9_pDMN_3_bin.csv'
    network = None
    roi = None
    ID = '002'

    outfile = utils.export_to_pandas(csv_loc, ID, network, roi)
    assert outfile is not None

# Fails


def test_save_RSN_coords_and_labels_to_pickle():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    network = None

    [coord_path, labels_path] = utils.save_RSN_coords_and_labels_to_pickle(
        coords, labels, dir_path, network)
    assert os.path.isfile(coord_path) is True
    assert os.path.isfile(labels_path) is True


def test_save_nifti_parcels_map():
    base_dir = str(Path(__file__).parent/"examples")
    ID = '002'
    dir_path = base_dir + '/002/fmri'
    roi = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(
        ID, dir_path, roi, network, net_parcels_map_nifti)
    assert os.path.isfile(net_parcels_nii_path) is True


def test_save_ts_to_file():
    base_dir = str(Path(__file__).parent/"examples")
    roi = None
    c_boot = 3
    network = None
    ID = '002'
    dir_path = base_dir + '/002/fmri'
    ts_within_nodes = '/tmp/'
    out_path_ts = utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    assert os.path.isfile(out_path_ts) is True


def test_build_omnetome():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    ID = '002'
    multimodal = False
    est_path_iterlist = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
                         dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
                         dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
                         dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
                         dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
    out_path = utils.build_omnetome(est_path_iterlist, ID, multimodal)
    assert out_path is not None


def test_check_est_path_existence():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    est_path_list = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_list)
    assert est_path_list_ex is not None


def test_collect_pandas_df():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    multi_nets = ['Default', 'SalVentAttn']
    network = 'Default'
    ID = '002'
    plot_switch = True
    multimodal = False
    net_pickle_mt_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.1_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.1_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.1_parc']
    utils.collect_pandas_df(network, ID, net_pickle_mt_list, plot_switch, multi_nets, multimodal)


def test_collect_pandas_df_make():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    plot_switch = True
    net_pickle_mt_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_0.1_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_0.1_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.05_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.06_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.07_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.08_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.09_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_0.1_parc']
    utils.collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)


def test_create_csv_path():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    roi = None
    models = ['corr', 'cov', 'sps', 'partcorr']
    node_size = 6

    # Cross test all models and thresh 0 to 1 by 0.1
    for conn_model in models:
        for val in range(1, 11):
            thr = round(val*0.1, 1)
            out_path = utils.create_csv_path(ID, network, conn_model, thr, roi, dir_path, node_size)
            assert out_path is not None


def test_create_est_path_diff():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
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
                    utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path,
                                               node_size, smooth, c_boot, thr_type, hpass, parc)


def test_create_est_path_func():
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


def test_create_unthr_path():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    for conn_model in models:
        unthr_path = utils.create_unthr_path(ID, network, conn_model, roi, dir_path)
        assert unthr_path is not None

# No function utils.cuberoot


def test_do_dir_path():
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
    # Slow, but successfully flattens a large array
    l = np.random.rand(3, 3, 3).tolist()
    l = utils.flatten(l)

    i = 0
    for item in l:
        i += 1
    assert i == (3*3*3)


def test_get_file():
    base_path = utils.get_file()
    assert base_path is not None


def test_list_first_mems():
    est_path_in = ['002_Default__rsn_net_ts.npy', '002_Default_est_cov_0.95propTEST_mm3_nb2_fwhm0.1_Hz.npy',
                   '002_Default_est_cov_0.95prop_TESTmm_3nb_2fwhm_0.1Hz.npy', '002_Default_est_cov_raw_mat.npy',
                   '002_Default_rsn_net_ts.npy']
    network_in = ['Default', 'Salience', 'ECN', 'LeftParietal']
    thr_in = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    node_size_in = [0, 10, 100]
    base_dir = str(Path(__file__).parent/"examples")
    path1 = base_dir + '/002/dmri'
    path2 = base_dir + '/002/fmri'
    dir_path_in = [path1, path2]
    smooth_in = [3, 6, 8, 10]
    c_boot_in = [100, 1000, 10000]
    hpass_in = [50, 100, 150, 200]

    [est_path, network, thr, dir_path, node_size, smooth, c_boot, hpass] = utils.list_first_mems(
        est_path_in, network_in, thr_in, dir_path_in, node_size_in, smooth_in, c_boot_in, hpass_in)

    assert est_path == est_path_in[0] == '002_Default__rsn_net_ts.npy'
    assert network == network_in[0] == 'Default'
    assert thr == thr_in[0] == 0.1
    assert node_size == node_size_in[0] == 0
    assert dir_path == dir_path_in[0] == path1
    assert c_boot == c_boot_in[0] == 100
    assert hpass == hpass_in[0] == 50


def test_make_gtab_and_bmask():
    base_dir = str(Path(__file__).parent/"examples")
    dwi_path = base_dir + '/002/dmri'
    fbval = dwi_path + '/bval.bval'
    fbvec = dwi_path + '/bvec.bvec'
    dwi_file = dwi_path + '/iso_eddy_corrected_data_denoised.nii.gz'
    network = 'Default'
    node_size = 6
    atlases = ['Power', 'Shirer', 'Shen', 'Smith']

    for atlas in atlases:
        [gtab_file, B0_bet, B0_mask, dwi_file] = utils.make_gtab_and_bmask(
            fbval, fbvec, dwi_file, network, node_size, atlas)

    assert gtab_file is not None
    assert B0_bet is not None
    assert B0_mask is not None
    assert dwi_file is not None


def test_merge_dicts():
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
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    conn_model = 'corr'
    est_path = dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy'
    network = 'Default'
    node_size = 6
    thr = 0.5
    prune = True
    ID = '002'
    roi = dir_path + 'pDMN_3_bin_mask.nii.gz'
    norm = 10
    binary = True

    [conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist,
        norm_iterlist, binary_iterlist] = utils.pass_meta_ins(conn_model, est_path, network, node_size, thr, prune, ID, roi, norm, binary)

    assert conn_model_iterlist is not None
    assert est_path_iterlist is not None
    assert network_iterlist is not None
    assert node_size_iterlist is not None
    assert thr_iterlist is not None
    assert prune_iterlist is not None
    assert ID_iterlist is not None
    assert roi_iterlist is not None
    assert norm_iterlist is not None
    assert binary_iterlist is not None


def test_pass_meta_ins_multi():
    base_dir = str(Path(__file__).parent/"examples")
    dmri_path = base_dir + '/002/dmri'
    func_path = base_dir + '/002/fmri'

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = func_path + '/002_Default_est_cov_0.95propTEST_mm3_nb2_fwhm0.1_Hz.npy'
    est_path_struct = dmri_path + \
        '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    network_func = 'Default'
    network_struct = 'Default'
    node_size_func = 6
    node_size_struct = 8
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

    [conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
        conn_model_func, est_path_func, network_func, node_size_func, thr_func, prune_func, ID_func, roi_func, norm_func, binary_func, conn_model_struct, est_path_struct, network_struct, node_size_struct, thr_struct, prune_struct, ID_struct, roi_struct, norm_struct, binary_struct)

    assert len(conn_model_iterlist) == 2
    assert len(est_path_iterlist) == 2
    assert len(network_iterlist) == 2
    assert len(node_size_iterlist) == 2
    assert len(thr_iterlist) == 2
    assert len(prune_iterlist) == 2
    assert len(ID_iterlist) == 2
    assert len(roi_iterlist) == 2
    assert len(norm_iterlist) == 2
    assert len(binary_iterlist) == 2


def test_pass_meta_outs():
    # Fails, this one was messy
    base_dir = str(Path(__file__).parent/"examples")
    dmri_path = base_dir + '/002/dmri'
    func_path = base_dir + '/002/fmri'

    conn_model_func = 'cor'
    conn_model_struct = 'cov'
    est_path_func = dmri_path + '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    est_path_struct = dmri_path + \
        '/DesikanKlein2012/0021001_Default_est_tensor_0.05dens_100000samples_particle_track.npy'
    network_func = 'Default'
    network_struct = 'Default'
    node_size_func = 6
    node_size_struct = 8
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

    [conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist] = utils.pass_meta_ins_multi(
        conn_model_func, est_path_func, network_func, node_size_func, thr_func, prune_func, ID_func, roi_func, norm_func, binary_func, conn_model_struct, est_path_struct, network_struct, node_size_struct, thr_struct, prune_struct, ID_struct, roi_struct, norm_struct, binary_struct)

    [conn_model_iterlist_out, est_path_iterlist_out, network_iterlist_out, node_size_iterlist_out, thr_iterlist_out, prune_iterlist_out, ID_iterlist_out, roi_iterlist_out, norm_iterlist_out, binary_iterlist_out] = utils.pass_meta_outs(
        conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist, embed=True, multimodal=False)

    assert conn_model_iterlist_out is not None
    assert est_path_iterlist_out is not None
    assert network_iterlist_out is not None
    assert node_size_iterlist_out is not None
    assert thr_iterlist_out is not None
    assert prune_iterlist_out is not None
    assert ID_iterlist_out is not None
    assert roi_iterlist_out is not None
    assert norm_iterlist_out is not None
    assert binary_iterlist_out is not None
