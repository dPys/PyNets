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
from pynets import utils


def test_export_to_pandas():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    csv_loc = dir_path + '/whole_brain_cluster_labels_PCA200/002_net_metrics_sps_0.9_pDMN_3_bin.csv'
    network = None
    roi = None
    ID = '002'

    outfile = utils.export_to_pandas(csv_loc, ID, network, roi)
    assert outfile is not None


def test_save_RSN_coords_and_labels_to_pickle():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    coord_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_coords_wb.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = dir_path + '/whole_brain_cluster_labels_PCA200/Default_func_labelnames_wb.pkl'
    labels_file = open(labels_file_path,'rb')
    labels = pickle.load(labels_file)
    network = None

    [coord_path, labels_path] = utils.save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network)
    assert os.path.isfile(coord_path) is True
    assert os.path.isfile(labels_path) is True


def test_save_nifti_parcels_map():
    import nibabel as nib
    base_dir = str(Path(__file__).parent/"examples")
    ID='002'
    dir_path = base_dir + '/002/fmri'
    roi = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti)
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
    net_pickle_mt_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.15_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.16_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.17_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.18_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.19_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_partcorr_0.2_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.15_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.16_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.17_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.18_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.19_parc',
                          dir_path + '/DesikanKlein2012/0021001_net_mets_Default_sps_0.2_parc']
    utils.collect_pandas_df(network, ID, net_pickle_mt_list, plot_switch, multi_nets, multimodal)
