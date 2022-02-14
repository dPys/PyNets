#!/usr/bin/env python
"""
Created on Thur July 18 20:19:14 2019
"""
import pytest
import os
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
import pkg_resources
import nibabel as nib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import numpy as np
import logging
import h5py

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_create_density_map():
    """
    Test for create_density_map functionality
    """
    from pynets.dmri import track
    from dipy.tracking._utils import _mapping_to_voxel

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    dir_path = f"{base_dir}/BIDS/sub-25659/ses-1/dwi"
    dwi_file = f"{base_dir}/BIDS/sub-25659/ses-1/dwi/final_preprocessed_" \
               f"dwi.nii.gz"
    dwi_img = nib.load(dwi_file)

    # Load output from test_filter_streamlines: dictionary of streamline info
    streamlines_trk = f"{base_dir}/miscellaneous/streamlines_model-csd_" \
                      f"nodetype-parc_tracktype-local_traversal-prob_" \
                      f"minlength-0.trk"
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
    node_radius = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    subnet = None
    roi = None
    traversal = 'prob'
    max_length = 0

    [dir_path, dm_path] = track.create_density_map(dwi_img, dir_path,
                                                   streams_final_filt_final,
                                                   conn_model, node_radius,
                                                   curv_thr_list, step_list,
                                                   subnet, roi, traversal,
                                                   max_length, '/tmp')

    assert dir_path is not None
    assert dm_path is not None


@pytest.mark.parametrize("tiss_class", ['act', 'wm', 'cmc', 'wb'])
def test_prep_tissues(tiss_class):
    """
    Test for prep_tissues functionality
    """
    from pynets.dmri import track
    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    t1w_mask = f"{base_dir}/003/dmri/gm_mask_dmri.nii.gz"
    B0_mask = f"{base_dir}/003/dmri/sub-003_b0_brain_mask.nii.gz"
    gm_in_dwi = f"{base_dir}/003/dmri/gm_mask_dmri.nii.gz"
    vent_csf_in_dwi = f"{base_dir}/003/dmri/csf_mask_dmri.nii.gz"
    wm_in_dwi = f"{base_dir}/003/dmri/wm_mask_dmri.nii.gz"

    tiss_classifier = track.prep_tissues(nib.load(t1w_mask),
                                         nib.load(gm_in_dwi),
                                         nib.load(vent_csf_in_dwi),
                                         nib.load(wm_in_dwi), tiss_class,
                                         nib.load(B0_mask),
                                         cmc_step_size=0.2)
    assert tiss_classifier is not None


@pytest.mark.parametrize("traversal", ['det', 'prob'])
@pytest.mark.parametrize("target_samples",
                         [300, pytest.param(0, marks=pytest.mark.xfail)])
def test_track_ensemble(traversal, target_samples):
    """
    Test for ensemble tractography functionality
    """
    import tempfile
    from pynets.dmri import track, estimation
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from nibabel.streamlines.array_sequence import ArraySequence

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
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
    tiss_class = 'wb'
    min_length = 2
    maxcrossing = 3
    roi_neighborhood_tol = 6
    waymask = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    sphere = get_sphere('repulsion724')
    track_type = 'local'

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    tmp = tempfile.TemporaryDirectory()
    temp_dir = str(tmp.name)
    os.makedirs(temp_dir, exist_ok=True)
    recon_path = f"{temp_dir}/model_file.hdf5"
    model = estimation.reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)[0]

    with h5py.File(recon_path, 'w') as hf:
        hf.create_dataset("reconstruction",
                          data=model.astype('float32'))
    hf.close()

    streamlines = track.track_ensemble(target_samples, atlas_data_wm_gm_int,
                                       labels_im_file, recon_path, sphere,
                                       traversal, curv_thr_list, step_list,
                                       track_type, maxcrossing,
                                       roi_neighborhood_tol, min_length,
                                       waymask, B0_mask, gm_in_dwi, gm_in_dwi,
                                       vent_csf_in_dwi, wm_in_dwi, tiss_class)

    assert isinstance(streamlines, ArraySequence)
    tmp.cleanup()


def test_track_ensemble_particle():
    """
    Test for ensemble tractography functionality
    """
    import tempfile
    from pynets.dmri import track, estimation
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    # from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
    # from dipy.io.streamline import save_tractogram
    from nibabel.streamlines.array_sequence import ArraySequence

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"
    gm_in_dwi = f"{base_dir}/003/anat/t1w_gm_in_dwi.nii.gz"
    vent_csf_in_dwi = f"{base_dir}/003/anat/t1w_vent_csf_in_dwi.nii.gz"
    wm_in_dwi = f"{base_dir}/003/anat/t1w_wm_in_dwi.nii.gz"
    dir_path = f"{base_dir}/003/dmri"
    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"
    gtab = gradient_table(bvals, bvecs)
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS" \
               f"_res-2mm.nii.gz"
    atlas_data_wm_gm_int = f"{dir_path}/whole_brain_cluster_labels_PCA200" \
                           f"_dwi_track_wmgm_int.nii.gz"
    labels_im_file = f"{dir_path}/whole_brain_cluster_labels_PCA200_dwi" \
                     f"_track.nii.gz"
    conn_model = 'csd'
    tiss_class = 'cmc'
    min_length = 2
    maxcrossing = 3
    roi_neighborhood_tol = 6
    waymask = None
    curv_thr_list = [40, 30]
    step_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    sphere = get_sphere('repulsion724')
    traversal = 'prob'
    track_type = 'particle'
    target_samples = 1000

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    model = estimation.reconstruction(conn_model, gtab, dwi_data, wm_in_dwi)[0]
    tmp = tempfile.TemporaryDirectory()
    temp_dir = str(tmp.name)
    recon_path = f"{str(temp_dir)}/model_file.hdf5"

    with h5py.File(recon_path, 'w') as hf:
        hf.create_dataset("reconstruction",
                          data=model.astype('float32'))
    hf.close()

    streamlines = track.track_ensemble(target_samples, atlas_data_wm_gm_int,
                                       labels_im_file, recon_path, sphere,
                                       traversal, curv_thr_list, step_list,
                   track_type, maxcrossing, roi_neighborhood_tol, min_length,
                   waymask, B0_mask, gm_in_dwi, gm_in_dwi, vent_csf_in_dwi,
                   wm_in_dwi, tiss_class)

    assert isinstance(streamlines, ArraySequence)
    tmp.cleanup()
