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
