#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:58:43 2017

@author: aki.nikolaidis
"""
import basc
#BASC_RUN
def basc_run(subjects_list, basc_config):
    import utils
    import os
    import numpy as np
    import scipy.stats
    from os.path import expanduser
    import yaml
    from basc_workflow_runner import run_basc_workflow

    
    
    subject_file_list= subjects_list
    
    #basc_config = Path(__file__).parent/"basc_config.yaml"
    
    basc_config='/Users/aki.nikolaidis/git_repo/ALL_PYNETS/PyNets/basc_config.yaml'
    f = open(basc_config)
    basc_dict_yaml=yaml.load(f)
    basc_dict =basc_dict_yaml['instance']
    
    proc_mem=basc_dict['proc_mem']
    roi_mask_file=basc_dict['roi_mask_file']
    dataset_bootstraps= basc_dict['dataset_bootstraps']
    timeseries_bootstraps= basc_dict['timeseries_bootstraps']
    n_clusters= basc_dict['n_clusters']
    output_size= basc_dict['output_size']
    bootstrap_list= eval(basc_dict['bootstrap_list'])
    cross_cluster= basc_dict['cross_cluster']
    affinity_threshold= basc_dict['affinity_threshold']
    out_dir= basc_dict['out_dir']
    run= basc_dict['run']

    
    
    
    basc_test= run_basc_workflow(subject_file_list, roi_mask_file, dataset_bootstraps, timeseries_bootstraps, n_clusters, output_size, bootstrap_list, proc_mem, cross_cluster=cross_cluster, roi2_mask_file=None, affinity_threshold=affinity_threshold, out_dir=out_dir, run=run)
            
#basc_run('/Users/aki.nikolaidis/git_repo/ALL_PYNETS/PyNets_NHW1/Sublist.txt', '/Users/aki.nikolaidis/git_repo/ALL_PYNETS/PyNets/pynets/basc_config.yaml')
#runfile('/Users/aki.nikolaidis/git_repo/ALL_PYNETS/PyNets/pynets/pynets_run.py',args="-i '/Users/aki.nikolaidis/git_repo/ALL_PYNETS/PyNets_NHW1/Sublist.txt' -basc 'True' -dt '0.3' -ns '4' -model 'sps' -mt", wdir='/Users/aki.nikolaidis/git_repo/ALL_PYNETS')