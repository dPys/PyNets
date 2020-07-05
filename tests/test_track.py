#!/usr/bin/env python
"""
Created on Thur July 18 20:19:14 2019

@authors: Derek Pisner & Ryan Hammonds

"""
import pytest
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
import nibabel as nib
import indexed_gzip
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_create_density_map():
    """
    Test for create_density_map functionality
    """
    from pynets.dmri import track
    from dipy.tracking._utils import _mapping_to_voxel

    base_dir = str(Path(__file__).parent/"examples")
    dir_path = f"{base_dir}/BIDS/sub-25659/ses-1/dwi"
    dwi_file = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz"
    dwi_img = nib.load(dwi_file)

    # Load output from test_filter_streamlines: dictionary of streamline info
    streamlines_trk = f"{base_dir}/miscellaneous/streamlines_est-csd_nodetype-parc_samples-10000streams_tt-local_dg-prob_ml-0.trk"
    streamlines = nib.streamlines.load(streamlines_trk).streamlines

    # Remove streamlines with negative voxel indices
    lin_T, offset = _mapping_to_voxel(np.eye(4))
    streams_final_filt_final = []
    for sl in streamlines:
        inds = np.dot(sl, lin_T)
        inds += offset
        if not inds.min().round(decimals=6) < 0:
            streams_final_filt_final.append(sl)

    conn_model = 'csd'
    target_samples = 10000
    node_size = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    network = None
    roi = None
    directget = 'prob'
    max_length = 0

    [dir_path, dm_path] = track.create_density_map(dwi_img, dir_path, streams_final_filt_final, conn_model,
                                                   target_samples, node_size, curv_thr_list, step_list,
                                                   network, roi, directget, max_length, '/tmp')

    assert dir_path is not None
    assert dm_path is not None


@pytest.mark.parametrize("tiss_class", ['act', 'wm', 'cmc', 'wb'])
def test_prep_tissues(tiss_class):
    """
    Test for prep_tissues functionality
    """
    from pynets.dmri import track
    base_dir = str(Path(__file__).parent/"examples")
    t1w_mask = f"{base_dir}/003/dmri/gm_mask_dmri.nii.gz"
    B0_mask = f"{base_dir}/003/dmri/sub-003_b0_brain_mask.nii.gz"
    gm_in_dwi = f"{base_dir}/003/dmri/gm_mask_dmri.nii.gz"
    vent_csf_in_dwi = f"{base_dir}/003/dmri/csf_mask_dmri.nii.gz"
    wm_in_dwi = f"{base_dir}/003/dmri/wm_mask_dmri.nii.gz"

    tiss_classifier = track.prep_tissues(t1w_mask, gm_in_dwi, vent_csf_in_dwi,
                                         wm_in_dwi, tiss_class, B0_mask,
                                         cmc_step_size=0.2)
    assert tiss_classifier is not None


@pytest.mark.parametrize("conn_model", ['csa', 'csd', 'ten'])
def test_reconstruction(conn_model):
    """
    Test for reconstruction functionality
    """
    from pynets.dmri import track
    from dipy.core.gradients import gradient_table
    base_dir = str(Path(__file__).parent/"examples")

    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{dir_path}/sub-003_dwi.bvec"
    gtab = gradient_table(bvals, bvecs)
    dwi_file = f"{dir_path}/sub-003_dwi.nii.gz"
    wm_in_dwi = f"{dir_path}/wm_mask_dmri.nii.gz"

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    model, mod = track.reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)
    assert model is not None
    assert mod is not None


@pytest.mark.parametrize("directget", ['det', 'prob'])
@pytest.mark.parametrize("target_samples",
                         [1000, pytest.param(0, marks=pytest.mark.xfail)])
def test_track_ensemble(directget, target_samples):
    """
    Test for ensemble tractography functionality
    """
    from pynets.dmri import track
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    gm_in_dwi = f"{base_dir}/003/anat/t1w_gm_in_dwi.nii.gz"
    vent_csf_in_dwi = f"{base_dir}/003/anat/t1w_vent_csf_in_dwi.nii.gz"
    wm_in_dwi = f"{base_dir}/003/anat/t1w_wm_in_dwi.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-" \
               f"2mm.nii.gz"
    atlas_data_wm_gm_int = f"{dir_path}/whole_brain_cluster_labels_PCA200_" \
                           f"dwi_track_wmgm_int.nii.gz"
    labels_im_file = f"{dir_path}/whole_brain_cluster_labels_PCA200_dwi_" \
                     f"track.nii.gz"
    conn_model = 'csa'
    tiss_class = 'wm'
    min_length = 10
    maxcrossing = 2
    roi_neighborhood_tol = 6
    waymask = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    sphere = get_sphere('repulsion724')
    track_type = 'local'

    # Load atlas parcellation (and its wm-gm interface reduced version for
    # seeding)
    atlas_data = np.array(nib.load(labels_im_file).dataobj).astype('uint16')
    atlas_data_wm_gm_int = np.asarray(nib.load(atlas_data_wm_gm_int).dataobj
                                      ).astype('uint16')

    # Build mask vector from atlas for later roi filtering
    parcels = []
    i = 0
    for roi_val in np.unique(atlas_data)[1:]:
        parcels.append(atlas_data == roi_val)
        i = i + 1

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    model, _ = track.reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)

    tiss_classifier = track.prep_tissues(gm_in_dwi, gm_in_dwi, vent_csf_in_dwi,
                                         wm_in_dwi, tiss_class, B0_mask,
                                         cmc_step_size=0.2)

    track.track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, model,
                         tiss_classifier, sphere, directget,
                         curv_thr_list, step_list, track_type, maxcrossing,
                         roi_neighborhood_tol, min_length, waymask,
                         B0_mask, max_length=1000, n_seeds_per_iter=500,
                         pft_back_tracking_dist=2,
                         pft_front_tracking_dist=1, particle_count=15,
                         min_separation_angle=20)


def test_track_ensemble_particle():
    """
    Test for ensemble tractography functionality
    """
    from pynets.dmri import track
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
    from dipy.io.streamline import save_tractogram

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    gm_in_dwi = f"{base_dir}/003/anat/t1w_gm_in_dwi.nii.gz"
    vent_csf_in_dwi = f"{base_dir}/003/anat/t1w_vent_csf_in_dwi.nii.gz"
    wm_in_dwi = f"{base_dir}/003/anat/t1w_wm_in_dwi.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"
    atlas_data_wm_gm_int = f"{dir_path}/whole_brain_cluster_labels_PCA200_dwi_track_wmgm_int.nii.gz"
    labels_im_file = f"{dir_path}/whole_brain_cluster_labels_PCA200_dwi_track.nii.gz"
    conn_model = 'csd'
    tiss_class = 'cmc'
    min_length = 10
    maxcrossing = 2
    roi_neighborhood_tol = 6
    waymask = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    sphere = get_sphere('repulsion724')
    directget = 'prob'
    track_type = 'particle'
    target_samples = 1000

    # Load atlas parcellation (and its wm-gm interface reduced version for seeding)
    atlas_data = np.array(nib.load(labels_im_file).dataobj).astype('uint16')
    atlas_data_wm_gm_int = np.asarray(nib.load(atlas_data_wm_gm_int).dataobj).astype('uint16')

    # Build mask vector from atlas for later roi filtering
    parcels = []
    i = 0
    for roi_val in np.unique(atlas_data)[1:]:
        parcels.append(atlas_data == roi_val)
        i = i + 1

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    model, _ = track.reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)

    tiss_classifier = track.prep_tissues(gm_in_dwi, gm_in_dwi, vent_csf_in_dwi,
                                         wm_in_dwi, tiss_class, B0_mask,
                                         cmc_step_size=0.2)

    streamlines = track.track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, model, tiss_classifier, sphere,
                                       directget, curv_thr_list, step_list, track_type, maxcrossing,
                                       roi_neighborhood_tol, min_length, waymask, B0_mask, max_length=1000,
                                       n_seeds_per_iter=500, pft_back_tracking_dist=2, pft_front_tracking_dist=1,
                                       particle_count=15, min_separation_angle=20)
    streams = f"{base_dir}/miscellaneous/003_streamlines_est-csd_nodetype-parc_samples-1000streams_tt-particle_dg-prob_ml-10.trk"
    save_tractogram(StatefulTractogram(streamlines, reference=dwi_img, space=Space.RASMM, origin=Origin.TRACKVIS),
                    streams, bbox_valid_check=False)
