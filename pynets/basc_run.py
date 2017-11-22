#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:58:43 2017

@author: aki.nikolaidis
"""
#BASC_RUN
def basc_run(subjects_list, basc_config):
    import os
    import numpy as np
    import nibabel as nib
    import yaml
    from basc.basc_workflow_runner import run_basc_workflow
    from pathlib import Path

    subject_file_list= subjects_list

    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')

    ##Determine the voxel size from sample subject's func image affine to pull correct MNI_152 image
    bna_img = nib.load(subject_file_list[0])

    x_vox = np.diagonal(bna_img.affine[:3,0:3])[0]
    y_vox = np.diagonal(bna_img.affine[:3,0:3])[1]
    z_vox = np.diagonal(bna_img.affine[:3,0:3])[2]

    if x_vox <= 1 and y_vox <= 1 and z_vox <=1:
        roi_mask_file = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    else:
        roi_mask_file = FSLDIR + '/data/standard/MNI152_T1_2mm_brain.nii.gz'

    basc_config=Path(__file__).parent/'basc_config.yaml'
    f = open(basc_config)
    basc_dict_yaml=yaml.load(f)
    basc_dict =basc_dict_yaml['instance']
    proc_mem=basc_dict['proc_mem']
    dataset_bootstraps= basc_dict['dataset_bootstraps']
    timeseries_bootstraps= basc_dict['timeseries_bootstraps']
    n_clusters= basc_dict['n_clusters']
    output_size= basc_dict['output_size']
    bootstrap_list= eval(basc_dict['bootstrap_list'])
    cross_cluster= basc_dict['cross_cluster']
    affinity_threshold= basc_dict['affinity_threshold']
    out_dir= Path(__file__).parent/'rsnrefs'
    run= basc_dict['run']
    similarity_metric= basc_dict['similarity_metric']

    run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, similarity_metric, cross_cluster=cross_cluster, roi2_mask_file=None, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
