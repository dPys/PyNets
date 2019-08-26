#!/usr/bin/env python
"""
Created on Thur July 18 20:19:14 2019

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.dmri import track

import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines
from dipy.io.streamline import load_trk


def test_filter_streamlines():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/001/dmri'
    dwi_file = dir_path + '/HARDI150.nii.gz'
    fbval = dir_path + '/HARDI150.bval'
    fbvec = dir_path + '/HARDI150.bvec'
    labels = dir_path + '/aparc-reduced.nii.gz'
    t1 = dir_path + '/t1.nii.gz'

    data = nib.load(dwi_file)
    data = data.get_data()

    labels = nib.load(labels)
    labels = labels.get_data()

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    # Generate streamlines
    t1_data = nib.load(t1)
    t1_data = t1_data.get_data()

    # Taken from dipy's example
    white_matter = (labels == 1) | (labels == 2)
    csamodel = shm.CsaOdfModel(gtab, 6)
    csapeaks = peaks.peaks_from_model(model=csamodel, data=data, sphere=peaks.default_sphere,
                                      relative_peak_threshold=.8, min_separation_angle=45, mask=white_matter)
    seeds = utils.seeds_from_mask(white_matter, density=2)
    streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
                                odf_vertices=peaks.default_sphere.vertices, a_low=.05, step_sz=.5, seeds=seeds)
    affine = streamline_generator.affine
    streamlines = Streamlines(streamline_generator, buffer_size=512)

    life_run = False
    min_length = 20
    conn_model = 'csa'
    target_samples = 10
    node_size = 5
    curv_thr_list = [2, 4, 6]
    step_list = [0.1, 0.2, 0.5]
    network = 'Default'
    roi = None

    [streams, dir_path, dm_path] = track.filter_streamlines(dwi_file, dir_path, gtab, streamlines, life_run, min_length, conn_model, target_samples,
                                                            node_size, curv_thr_list, step_list, network, roi)

    assert streams is not None
    assert dir_path is not None
    assert dm_path is not None


def test_prep_tissues():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/003/dmri'
    B0_mask = dir_path + '/sub-003_b0_brain_mask.nii.gz'
    gm_in_dwi = dir_path + '/gm_mask_dmri.nii.gz'
    vent_csf_in_dwi = dir_path + '/csf_mask_dmri.nii.gz'
    wm_in_dwi = dir_path + '/wm_mask_dmri.nii.gz'
    
    for tiss_class in ['act', 'bin', 'cmc', 'other']:
        tiss_classifier = track.prep_tissues(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, cmc_step_size=0.2)
        assert tiss_classifier is not None
        
        
def test_reconstruction():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/003/dmri'
    
    bvals = dir_path + '/sub-003_dwi.bval'
    bvecs = dir_path + '/sub-003_dwi.bvec'
    gtab = gradient_table(bvals, bvecs)
    dwi_file = dir_path + '/sub-003_dwi.nii'
    wm_in_dwi = dir_path + '/wm_mask_dmri.nii.gz'
    for conn_model in ['csa', 'tensor', 'csd']:
        mod = track.reconstruction(conn_model, gtab, dwi_file, wm_in_dwi)
        assert mod is not None
        
        
def test_run_LIFE_all():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/001/dmri'
    dwi_file = dir_path + '/HARDI150.nii.gz'
    fbval = dir_path + '/HARDI150.bval'
    fbvec = dir_path + '/HARDI150.bvec'
    labels = dir_path + '/aparc-reduced.nii.gz'
    t1 = dir_path + '/t1.nii.gz'

    data = nib.load(dwi_file)
    data = data.get_data()

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    
    fbval = dir_path + '/HARDI150.bval'
    fbvec = dir_path + '/HARDI150.bvec'
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    
    ''' This doesn't work, not sure why. Have to regen streams...
    streamlines_trk = dir_path + '/tractography/streamlines_Default_csa_10_5mm_curv[2_4_6]_step[0.1_0.2_0.5].trk'
    # Load output from test_filter_streamlines: dictionary of streamline info
    streamlines = load_trk(streamlines_trk)
    # First entry is nibabel.streamlines.array_sequence... type
    streamlines = streamlines[0]
    '''
    
    # Taken from dipy's example
    # Generate streamlines
    labels = nib.load(labels)
    labels = labels.get_data()
    t1_data = nib.load(t1)
    t1_data = t1_data.get_data()
    
    white_matter = (labels == 1) | (labels == 2)
    csamodel = shm.CsaOdfModel(gtab, 6)
    csapeaks = peaks.peaks_from_model(model=csamodel, data=data, sphere=peaks.default_sphere,
                                      relative_peak_threshold=.8, min_separation_angle=45, mask=white_matter)
    seeds = utils.seeds_from_mask(white_matter, density=2)
    streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
                                odf_vertices=peaks.default_sphere.vertices, a_low=.05, step_sz=.5, seeds=seeds)
    affine = streamline_generator.affine
    streamlines = Streamlines(streamline_generator, buffer_size=512)
    
    # Shape of streamline array is (560583,) should test something smaller.
    streamlines_filt, mean_rmse = track.run_LIFE_all(data, gtab, streamlines)
    
    assert streamlines_filt is not None
    assert mean_rmse is not None
    
def test_save_streams():
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/001/dmri'
    dwi_file = dir_path + '/HARDI150.nii.gz'
    dwi_img = nib.load(dwi_file)
    
    # Load output from test_filter_streamlines: dictionary of streamline info
    streamlines_trk = dir_path + '/tractography/streamlines_Default_csa_10_5mm_curv[2_4_6]_step[0.1_0.2_0.5].trk'
    streamlines = load_trk(streamlines_trk)
    # First entry is nibabel.streamlines.array_sequence... type
    streamlines = streamlines[0]
    streams = dir_path + '/tractography/test_save.trk'
    streams_out = track.save_streams(dwi_img, streamlines, streams)
    assert streams_out is not None