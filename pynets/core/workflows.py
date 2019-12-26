#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def workflow_selector(func_file, ID, atlas, network, node_size, roi, thr, uatlas, multi_nets,
                      conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_file, anat_file, parc,
                      ref_txt, procmem, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k,
                      clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                      node_size_list, num_total_samples, conn_model_list, min_span_tree, verbose, plugin_type,
                      use_AAL_naming, smooth, smooth_list, disp_filt, clust_type, clust_type_list, c_boot, block_size,
                      mask, norm, binary, fbval, fbvec, target_samples, curv_thr_list, step_list, overlap_thr,
                      overlap_thr_list, track_type, max_length, maxcrossing, min_length, directget,
                      tiss_class, runtime_dict, execution_dict, embed, multi_directget, multimodal, hpass, hpass_list,
                      template, template_mask, vox_size, multiplex, waymask, local_corr, clean=True):
    """A meta-interface for selecting nested workflows to link into a given single-subject workflow"""
    import yaml
    from pathlib import Path
    import pkg_resources
    from pynets.core import workflows
    from nipype import Workflow
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.core.utils import pass_meta_ins, pass_meta_outs, pass_meta_ins_multi

    # Available functional and structural connectivity models
    with open("%s%s" % (str(Path(__file__).parent.parent), '/runconfig.yaml'), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        try:
            func_models = hardcoded_params['available_models']['func_models']
        except KeyError:
            print('ERROR: available functional models not successfully extracted from runconfig.yaml')
        try:
            struct_models = hardcoded_params['available_models']['struct_models']
        except KeyError:
            print('ERROR: available structural models not successfully extracted from runconfig.yaml')

    # Handle modality logic
    if (func_file is not None) and (dwi_file is not None):
        print('Parsing multimodal models...')
        func_model_list = []
        dwi_model_list = []
        if conn_model_list is not None:
            for conn_model in conn_model_list:
                if conn_model in func_models:
                    func_model_list.append(conn_model)
                    conn_model_func = None
                if conn_model in struct_models:
                    dwi_model_list.append(conn_model)
                    conn_model_dwi = None
            if len(func_model_list) == 1:
                conn_model_func = func_model_list[0]
                func_model_list = None
            if len(dwi_model_list) == 1:
                conn_model_dwi = dwi_model_list[0]
                dwi_model_list = None
        else:
            raise RuntimeError('ERROR: Multimodal fMRI-dMRI pipeline specified, but only one connectivity model '
                               'specified.')
    elif (dwi_file is not None) and (func_file is None):
        print('Parsing diffusion models...')
        conn_model_dwi = conn_model
        dwi_model_list = conn_model_list
        conn_model_func = None
        func_model_list = None
    elif (func_file is not None) and (dwi_file is None):
        print('Parsing functional models...')
        conn_model_func = conn_model
        func_model_list = conn_model_list
        conn_model_dwi = None
        dwi_model_list = None

    # Set paths to templates
    if template is None:
        template = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size + "_brain.nii.gz")

    if template_mask is None:
        template_mask = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                        "_brain_mask.nii.gz")

    # for each file input, delete corresponding t1w anatomical copies.
    if clean is True:
        import os.path as op
        import shutil
        file_list = [dwi_file, func_file, anat_file]
        for _file in file_list:
            if _file is not None:
                outdir = op.dirname(_file)
                if op.isdir("%s%s" % (outdir, '/anat_tmp')):
                    shutil.rmtree("%s%s" % (outdir, '/anat_tmp'))

    # Workflow 1: Structural connectome
    if dwi_file is not None:
        sub_struct_wf = workflows.dmri_connectometry(ID, atlas, network, node_size, roi,
                                                     uatlas, plot_switch, parc, ref_txt, procmem,
                                                     dwi_file, fbval, fbvec, anat_file, thr, dens_thresh,
                                                     conn_model_dwi, user_atlas_list, multi_thr, multi_atlas,
                                                     max_thr, min_thr, step_thr, node_size_list,
                                                     dwi_model_list, min_span_tree, use_AAL_naming, disp_filt,
                                                     plugin_type, multi_nets, prune, mask, norm, binary,
                                                     target_samples, curv_thr_list, step_list, overlap_thr,
                                                     overlap_thr_list, track_type, max_length, maxcrossing,
                                                     min_length, directget, tiss_class, runtime_dict, execution_dict,
                                                     multi_directget, template, template_mask, vox_size, waymask)
        if func_file is None:
            sub_func_wf = None
        sub_struct_wf._n_procs = procmem[0]
        sub_struct_wf._mem_gb = procmem[1]
        sub_struct_wf.n_procs = procmem[0]
        sub_struct_wf.mem_gb = procmem[1]

    # Workflow 2: Functional connectome
    if func_file is not None:
        sub_func_wf = workflows.fmri_connectometry(func_file, ID, atlas, network, node_size,
                                                   roi, thr, uatlas, conn_model_func, dens_thresh, conf,
                                                   plot_switch, parc, ref_txt, procmem,
                                                   multi_thr, multi_atlas, max_thr, min_thr, step_thr,
                                                   k, clust_mask, k_min, k_max, k_step, k_clustering,
                                                   user_atlas_list, clust_mask_list, node_size_list,
                                                   func_model_list, min_span_tree, use_AAL_naming, smooth,
                                                   smooth_list, disp_filt, prune, multi_nets, clust_type,
                                                   clust_type_list, plugin_type, c_boot, block_size, mask,
                                                   norm, binary, anat_file, runtime_dict, execution_dict, hpass,
                                                   hpass_list, template, template_mask, vox_size, local_corr)
        if dwi_file is None:
            sub_struct_wf = None
        sub_func_wf._n_procs = procmem[0]
        sub_func_wf._mem_gb = procmem[1]
        sub_func_wf.n_procs = procmem[0]
        sub_func_wf.mem_gb = procmem[1]

    # Create meta-workflow to organize graph simulation sets in prep for analysis
    base_dirname = "%s%s" % ('meta_wf_', ID)
    meta_wf = Workflow(name=base_dirname)

    if verbose is True:
        from nipype import config, logging
        cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'log_to_file': True,
                              'interface_level': 'DEBUG'},
                     monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True})
        logging.update_logging(config)
        config.update_config(cfg_v)
        config.enable_debug_mode()
        config.enable_resource_monitor()
    execution_dict['plugin_args'] = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                                     'scheduler': 'mem_thread'}
    execution_dict['plugin'] = str(plugin_type)
    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            meta_wf.config[key][setting] = value

    meta_inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 'atlas', 'network', 'thr',
                                                           'node_size', 'roi', 'uatlas', 'multi_nets',
                                                           'conn_model_func', 'conn_model_dwi', 'dens_thresh',
                                                           'conf', 'adapt_thresh', 'plot_switch', 'dwi_file',
                                                           'anat_file', 'parc', 'ref_txt', 'procmem', 'multi_thr',
                                                           'multi_atlas', 'max_thr', 'min_thr', 'step_thr', 'k',
                                                           'clust_mask', 'k_min', 'k_max', 'k_step', 'k_clustering',
                                                           'user_atlas_list', 'clust_mask_list', 'prune',
                                                           'node_size_list', 'num_total_samples',
                                                           'func_model_list', 'dwi_model_list', 'min_span_tree',
                                                           'verbose', 'plugin_type', 'use_AAL_naming', 'smooth',
                                                           'smooth_list', 'disp_filt', 'clust_type',
                                                           'clust_type_list', 'c_boot', 'block_size', 'mask',
                                                           'norm', 'binary', 'fbval', 'fbvec', 'target_samples',
                                                           'curv_thr_list', 'step_list', 'overlap_thr',
                                                           'overlap_thr_list', 'track_type', 'max_length',
                                                           'maxcrossing', 'min_length', 'directget',
                                                           'tiss_class', 'embed', 'multi_directget', 'multimodal',
                                                           'hpass', 'hpass_list', 'template', 'template_mask',
                                                           'vox_size', 'multiplex', 'waymask', 'local_corr']),
                             name='meta_inputnode')
    meta_inputnode.inputs.func_file = func_file
    meta_inputnode.inputs.ID = ID
    meta_inputnode.inputs.atlas = atlas
    meta_inputnode.inputs.network = network
    meta_inputnode.inputs.thr = thr
    meta_inputnode.inputs.node_size = node_size
    meta_inputnode.inputs.roi = roi
    meta_inputnode.inputs.uatlas = uatlas
    meta_inputnode.inputs.multi_nets = multi_nets
    meta_inputnode.inputs.conn_model_func = conn_model_func
    meta_inputnode.inputs.conn_model_dwi = conn_model_dwi
    meta_inputnode.inputs.dens_thresh = dens_thresh
    meta_inputnode.inputs.conf = conf
    meta_inputnode.inputs.adapt_thresh = adapt_thresh
    meta_inputnode.inputs.plot_switch = plot_switch
    meta_inputnode.inputs.dwi_file = dwi_file
    meta_inputnode.inputs.fbval = fbval
    meta_inputnode.inputs.fbvec = fbvec
    meta_inputnode.inputs.anat_file = anat_file
    meta_inputnode.inputs.parc = parc
    meta_inputnode.inputs.ref_txt = ref_txt
    meta_inputnode.inputs.procmem = procmem
    meta_inputnode.inputs.multi_thr = multi_thr
    meta_inputnode.inputs.multi_atlas = multi_atlas
    meta_inputnode.inputs.max_thr = max_thr
    meta_inputnode.inputs.min_thr = min_thr
    meta_inputnode.inputs.step_thr = step_thr
    meta_inputnode.inputs.k = k
    meta_inputnode.inputs.clust_mask = clust_mask
    meta_inputnode.inputs.k_min = k_min
    meta_inputnode.inputs.k_max = k_max
    meta_inputnode.inputs.k_step = k_step
    meta_inputnode.inputs.k_clustering = k_clustering
    meta_inputnode.inputs.user_atlas_list = user_atlas_list
    meta_inputnode.inputs.clust_mask_list = clust_mask_list
    meta_inputnode.inputs.prune = prune
    meta_inputnode.inputs.node_size_list = node_size_list
    meta_inputnode.inputs.num_total_samples = num_total_samples
    meta_inputnode.inputs.func_model_list = func_model_list
    meta_inputnode.inputs.dwi_model_list = dwi_model_list
    meta_inputnode.inputs.min_span_tree = min_span_tree
    meta_inputnode.inputs.verbose = verbose
    meta_inputnode.inputs.plugin_type = plugin_type
    meta_inputnode.inputs.use_AAL_naming = use_AAL_naming
    meta_inputnode.inputs.smooth = smooth
    meta_inputnode.inputs.smooth_list = smooth_list
    meta_inputnode.inputs.hpass = hpass
    meta_inputnode.inputs.hpass_list = hpass_list
    meta_inputnode.inputs.disp_filt = disp_filt
    meta_inputnode.inputs.clust_type = clust_type
    meta_inputnode.inputs.clust_type_list = clust_type_list
    meta_inputnode.inputs.c_boot = c_boot
    meta_inputnode.inputs.block_size = block_size
    meta_inputnode.inputs.mask = mask
    meta_inputnode.inputs.norm = norm
    meta_inputnode.inputs.binary = binary
    meta_inputnode.inputs.target_samples = target_samples
    meta_inputnode.inputs.curv_thr_list = curv_thr_list
    meta_inputnode.inputs.step_list = step_list
    meta_inputnode.inputs.overlap_thr = overlap_thr
    meta_inputnode.inputs.overlap_thr_list = overlap_thr_list
    meta_inputnode.inputs.track_type = track_type
    meta_inputnode.inputs.max_length = max_length
    meta_inputnode.inputs.maxcrossing = maxcrossing
    meta_inputnode.inputs.min_length = min_length
    meta_inputnode.inputs.directget = directget
    meta_inputnode.inputs.tiss_class = tiss_class
    meta_inputnode.inputs.embed = embed
    meta_inputnode.inputs.multimodal = multimodal
    meta_inputnode.inputs.multi_directget = multi_directget
    meta_inputnode.inputs.template = template
    meta_inputnode.inputs.template_mask = template_mask
    meta_inputnode.inputs.vox_size = vox_size
    meta_inputnode.inputs.multiplex = multiplex
    meta_inputnode.inputs.waymask = waymask
    meta_inputnode.inputs.local_corr = local_corr

    if multimodal is True:
        # Create input/output nodes
        print('Running Multimodal Meta-Workflow...')
        pass_meta_ins_multi_node = pe.Node(niu.Function(input_names=['conn_model_func', 'est_path_func', 'network_func',
                                                                     'thr_func', 'prune_func', 'ID_func', 'roi_func',
                                                                     'norm_func', 'binary_func', 'conn_model_struct',
                                                                     'est_path_struct', 'network_struct', 'thr_struct',
                                                                     'prune_struct', 'ID_struct', 'roi_struct',
                                                                     'norm_struct', 'binary_struct'],
                                                        output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                                      'network_iterlist', 'thr_iterlist',
                                                                      'prune_iterlist', 'ID_iterlist',
                                                                      'roi_iterlist', 'norm_iterlist',
                                                                      'binary_iterlist'],
                                                        function=pass_meta_ins_multi),
                                           name='pass_meta_ins_multi_node')

        meta_wf.add_nodes([sub_struct_wf])
        meta_wf.get_node(sub_struct_wf.name)._n_procs = procmem[0]
        meta_wf.get_node(sub_struct_wf.name)._mem_gb = procmem[1]
        meta_wf.get_node(sub_struct_wf.name).n_procs = procmem[0]
        meta_wf.get_node(sub_struct_wf.name).mem_gb = procmem[1]
        meta_wf.add_nodes([sub_func_wf])
        meta_wf.get_node(sub_func_wf.name)._n_procs = procmem[0]
        meta_wf.get_node(sub_func_wf.name)._mem_gb = procmem[1]
        meta_wf.get_node(sub_func_wf.name).n_procs = procmem[0]
        meta_wf.get_node(sub_func_wf.name).mem_gb = procmem[1]
        meta_wf.connect([(meta_inputnode, sub_struct_wf, [('ID', 'inputnode.ID'),
                                                          ('dwi_file', 'inputnode.dwi_file'),
                                                          ('fbval', 'inputnode.fbval'),
                                                          ('fbvec', 'inputnode.fbvec'),
                                                          ('anat_file', 'inputnode.anat_file'),
                                                          ('atlas', 'inputnode.atlas'),
                                                          ('network', 'inputnode.network'),
                                                          ('thr', 'inputnode.thr'),
                                                          ('node_size', 'inputnode.node_size'),
                                                          ('roi', 'inputnode.roi'),
                                                          ('uatlas', 'inputnode.uatlas'),
                                                          ('multi_nets', 'inputnode.multi_nets'),
                                                          ('conn_model_dwi', 'inputnode.conn_model'),
                                                          ('dens_thresh', 'inputnode.dens_thresh'),
                                                          ('plot_switch', 'inputnode.plot_switch'),
                                                          ('parc', 'inputnode.parc'),
                                                          ('ref_txt', 'inputnode.ref_txt'),
                                                          ('procmem', 'inputnode.procmem'),
                                                          ('multi_thr', 'inputnode.multi_thr'),
                                                          ('multi_atlas', 'inputnode.multi_atlas'),
                                                          ('max_thr', 'inputnode.max_thr'),
                                                          ('min_thr', 'inputnode.min_thr'),
                                                          ('step_thr', 'inputnode.step_thr'),
                                                          ('user_atlas_list', 'inputnode.user_atlas_list'),
                                                          ('prune', 'inputnode.prune'),
                                                          ('dwi_model_list', 'inputnode.conn_model_list'),
                                                          ('min_span_tree', 'inputnode.min_span_tree'),
                                                          ('use_AAL_naming', 'inputnode.use_AAL_naming'),
                                                          ('disp_filt', 'inputnode.disp_filt'),
                                                          ('mask', 'inputnode.mask'),
                                                          ('norm', 'inputnode.norm'),
                                                          ('binary', 'inputnode.binary'),
                                                          ('target_samples', 'inputnode.target_samples'),
                                                          ('curv_thr_list', 'inputnode.curv_thr_list'),
                                                          ('step_list', 'inputnode.step_list'),
                                                          ('overlap_thr', 'inputnode.overlap_thr'),
                                                          ('overlap_thr_list', 'inputnode.overlap_thr_list'),
                                                          ('track_type', 'inputnode.track_type'),
                                                          ('max_length', 'inputnode.max_length'),
                                                          ('maxcrossing', 'inputnode.maxcrossing'),
                                                          ('min_length', 'inputnode.min_length'),
                                                          ('directget', 'inputnode.directget'),
                                                          ('tiss_class', 'inputnode.tiss_class'),
                                                          ('multi_directget', 'inputnode.multi_directget'),
                                                          ('template', 'inputnode.template'),
                                                          ('template_mask', 'inputnode.template_mask'),
                                                          ('vox_size', 'inputnode.vox_size'),
                                                          ('waymask', 'inputnode.waymask')
                                                          ])
                         ])
        meta_wf.connect([(meta_inputnode, sub_func_wf, [('func_file', 'inputnode.func_file'),
                                                        ('ID', 'inputnode.ID'),
                                                        ('anat_file', 'inputnode.anat_file'),
                                                        ('atlas', 'inputnode.atlas'),
                                                        ('network', 'inputnode.network'),
                                                        ('thr', 'inputnode.thr'),
                                                        ('node_size', 'inputnode.node_size'),
                                                        ('roi', 'inputnode.roi'),
                                                        ('uatlas', 'inputnode.uatlas'),
                                                        ('multi_nets', 'inputnode.multi_nets'),
                                                        ('conn_model_func', 'inputnode.conn_model'),
                                                        ('dens_thresh', 'inputnode.dens_thresh'),
                                                        ('conf', 'inputnode.conf'),
                                                        ('plot_switch', 'inputnode.plot_switch'),
                                                        ('parc', 'inputnode.parc'),
                                                        ('ref_txt', 'inputnode.ref_txt'),
                                                        ('procmem', 'inputnode.procmem'),
                                                        ('multi_thr', 'inputnode.multi_thr'),
                                                        ('multi_atlas', 'inputnode.multi_atlas'),
                                                        ('max_thr', 'inputnode.max_thr'),
                                                        ('min_thr', 'inputnode.min_thr'),
                                                        ('step_thr', 'inputnode.step_thr'),
                                                        ('k', 'inputnode.k'),
                                                        ('clust_mask', 'inputnode.clust_mask'),
                                                        ('k_min', 'inputnode.k_min'),
                                                        ('k_max', 'inputnode.k_max'),
                                                        ('k_step', 'inputnode.k_step'),
                                                        ('k_clustering', 'inputnode.k_clustering'),
                                                        ('user_atlas_list', 'inputnode.user_atlas_list'),
                                                        ('clust_mask_list', 'inputnode.clust_mask_list'),
                                                        ('prune', 'inputnode.prune'),
                                                        ('func_model_list', 'inputnode.conn_model_list'),
                                                        ('min_span_tree', 'inputnode.min_span_tree'),
                                                        ('use_AAL_naming', 'inputnode.use_AAL_naming'),
                                                        ('smooth', 'inputnode.smooth'),
                                                        ('hpass', 'inputnode.hpass'),
                                                        ('hpass_list', 'inputnode.hpass_list'),
                                                        ('disp_filt', 'inputnode.disp_filt'),
                                                        ('clust_type', 'inputnode.clust_type'),
                                                        ('clust_type_list', 'inputnode.clust_type_list'),
                                                        ('c_boot', 'inputnode.c_boot'),
                                                        ('block_size', 'inputnode.block_size'),
                                                        ('mask', 'inputnode.mask'),
                                                        ('norm', 'inputnode.norm'),
                                                        ('binary', 'inputnode.binary'),
                                                        ('template', 'inputnode.template'),
                                                        ('template_mask', 'inputnode.template_mask'),
                                                        ('vox_size', 'inputnode.vox_size'),
                                                        ('local_corr', 'inputnode.local_corr')])
                         ])

        # Connect outputs of nested workflow to parent wf
        meta_wf.connect([
            (sub_func_wf.get_node('outputnode'), pass_meta_ins_multi_node, [('conn_model', 'conn_model_func'),
                                                                            ('est_path', 'est_path_func'),
                                                                            ('network', 'network_func'),
                                                                            ('thr', 'thr_func'),
                                                                            ('prune', 'prune_func'),
                                                                            ('ID', 'ID_func'),
                                                                            ('roi', 'roi_func'),
                                                                            ('norm', 'norm_func'),
                                                                            ('binary', 'binary_func')]),
            (sub_struct_wf.get_node('outputnode'), pass_meta_ins_multi_node, [('conn_model', 'conn_model_struct'),
                                                                              ('est_path', 'est_path_struct'),
                                                                              ('network', 'network_struct'),
                                                                              ('thr', 'thr_struct'),
                                                                              ('prune', 'prune_struct'),
                                                                              ('ID', 'ID_struct'),
                                                                              ('roi', 'roi_struct'),
                                                                              ('norm', 'norm_struct'),
                                                                              ('binary', 'binary_struct')])
        ])

    else:
        print('Running Unimodal Meta-Workflow...')

        if dwi_file:
            pass_meta_ins_struct_node = pe.Node(niu.Function(input_names=['conn_model', 'est_path', 'network',
                                                                          'thr', 'prune', 'ID', 'roi', 'norm',
                                                                          'binary'],
                                                             output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                                           'network_iterlist', 'thr_iterlist',
                                                                           'prune_iterlist', 'ID_iterlist',
                                                                           'roi_iterlist', 'norm_iterlist',
                                                                           'binary_iterlist'], function=pass_meta_ins),
                                                name='pass_meta_ins_struct_node')

            meta_wf.add_nodes([sub_struct_wf])
            meta_wf.get_node(sub_struct_wf.name)._n_procs = procmem[0]
            meta_wf.get_node(sub_struct_wf.name)._mem_gb = procmem[1]
            meta_wf.get_node(sub_struct_wf.name).n_procs = procmem[0]
            meta_wf.get_node(sub_struct_wf.name).mem_gb = procmem[1]

            meta_wf.connect([(meta_inputnode, sub_struct_wf, [('ID', 'inputnode.ID'),
                                                              ('dwi_file', 'inputnode.dwi_file'),
                                                              ('fbval', 'inputnode.fbval'),
                                                              ('fbvec', 'inputnode.fbvec'),
                                                              ('anat_file', 'inputnode.anat_file'),
                                                              ('atlas', 'inputnode.atlas'),
                                                              ('network', 'inputnode.network'),
                                                              ('thr', 'inputnode.thr'),
                                                              ('node_size', 'inputnode.node_size'),
                                                              ('roi', 'inputnode.roi'),
                                                              ('uatlas', 'inputnode.uatlas'),
                                                              ('multi_nets', 'inputnode.multi_nets'),
                                                              ('conn_model_dwi', 'inputnode.conn_model'),
                                                              ('dens_thresh', 'inputnode.dens_thresh'),
                                                              ('plot_switch', 'inputnode.plot_switch'),
                                                              ('parc', 'inputnode.parc'),
                                                              ('ref_txt', 'inputnode.ref_txt'),
                                                              ('procmem', 'inputnode.procmem'),
                                                              ('multi_thr', 'inputnode.multi_thr'),
                                                              ('multi_atlas', 'inputnode.multi_atlas'),
                                                              ('max_thr', 'inputnode.max_thr'),
                                                              ('min_thr', 'inputnode.min_thr'),
                                                              ('step_thr', 'inputnode.step_thr'),
                                                              ('user_atlas_list', 'inputnode.user_atlas_list'),
                                                              ('prune', 'inputnode.prune'),
                                                              ('dwi_model_list', 'inputnode.conn_model_list'),
                                                              ('min_span_tree', 'inputnode.min_span_tree'),
                                                              ('use_AAL_naming', 'inputnode.use_AAL_naming'),
                                                              ('disp_filt', 'inputnode.disp_filt'),
                                                              ('mask', 'inputnode.mask'),
                                                              ('norm', 'inputnode.norm'),
                                                              ('binary', 'inputnode.binary'),
                                                              ('target_samples', 'inputnode.target_samples'),
                                                              ('curv_thr_list', 'inputnode.curv_thr_list'),
                                                              ('step_list', 'inputnode.step_list'),
                                                              ('overlap_thr', 'inputnode.overlap_thr'),
                                                              ('overlap_thr_list', 'inputnode.overlap_thr_list'),
                                                              ('track_type', 'inputnode.track_type'),
                                                              ('max_length', 'inputnode.max_length'),
                                                              ('maxcrossing', 'inputnode.maxcrossing'),
                                                              ('min_length', 'inputnode.min_length'),
                                                              ('directget', 'inputnode.directget'),
                                                              ('tiss_class', 'inputnode.tiss_class'),
                                                              ('multi_directget', 'inputnode.multi_directget'),
                                                              ('template', 'inputnode.template'),
                                                              ('template_mask', 'inputnode.template_mask'),
                                                              ('vox_size', 'inputnode.vox_size'),
                                                              ('waymask', 'inputnode.waymask')
                                                              ])
                             ])

            # Connect outputs of nested workflow to parent wf
            meta_wf.connect([(sub_struct_wf.get_node('outputnode'),
                              pass_meta_ins_struct_node, [('conn_model', 'conn_model'), ('est_path', 'est_path'),
                                                          ('network', 'network'), ('thr', 'thr'), ('prune', 'prune'),
                                                          ('ID', 'ID'), ('roi', 'roi'), ('norm', 'norm'),
                                                          ('binary', 'binary')])
                             ])

        if func_file:
            pass_meta_ins_func_node = pe.Node(niu.Function(input_names=['conn_model', 'est_path', 'network',
                                                                        'thr', 'prune', 'ID', 'roi',
                                                                        'norm', 'binary'],
                                                           output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                                         'network_iterlist', 'thr_iterlist',
                                                                         'prune_iterlist', 'ID_iterlist',
                                                                         'roi_iterlist', 'norm_iterlist',
                                                                         'binary_iterlist'],
                                                           function=pass_meta_ins), name='pass_meta_ins_func_node')

            meta_wf.add_nodes([sub_func_wf])
            meta_wf.get_node(sub_func_wf.name)._n_procs = procmem[0]
            meta_wf.get_node(sub_func_wf.name)._mem_gb = procmem[1]
            meta_wf.get_node(sub_func_wf.name).n_procs = procmem[0]
            meta_wf.get_node(sub_func_wf.name).mem_gb = procmem[1]
            meta_wf.connect([(meta_inputnode, sub_func_wf, [('func_file', 'inputnode.func_file'),
                                                            ('ID', 'inputnode.ID'),
                                                            ('anat_file', 'inputnode.anat_file'),
                                                            ('atlas', 'inputnode.atlas'),
                                                            ('network', 'inputnode.network'),
                                                            ('thr', 'inputnode.thr'),
                                                            ('node_size', 'inputnode.node_size'),
                                                            ('roi', 'inputnode.roi'),
                                                            ('uatlas', 'inputnode.uatlas'),
                                                            ('multi_nets', 'inputnode.multi_nets'),
                                                            ('conn_model_func', 'inputnode.conn_model'),
                                                            ('dens_thresh', 'inputnode.dens_thresh'),
                                                            ('conf', 'inputnode.conf'),
                                                            ('plot_switch', 'inputnode.plot_switch'),
                                                            ('parc', 'inputnode.parc'),
                                                            ('ref_txt', 'inputnode.ref_txt'),
                                                            ('procmem', 'inputnode.procmem'),
                                                            ('multi_thr', 'inputnode.multi_thr'),
                                                            ('multi_atlas', 'inputnode.multi_atlas'),
                                                            ('max_thr', 'inputnode.max_thr'),
                                                            ('min_thr', 'inputnode.min_thr'),
                                                            ('step_thr', 'inputnode.step_thr'),
                                                            ('k', 'inputnode.k'),
                                                            ('clust_mask', 'inputnode.clust_mask'),
                                                            ('k_min', 'inputnode.k_min'),
                                                            ('k_max', 'inputnode.k_max'),
                                                            ('k_step', 'inputnode.k_step'),
                                                            ('k_clustering', 'inputnode.k_clustering'),
                                                            ('user_atlas_list', 'inputnode.user_atlas_list'),
                                                            ('clust_mask_list', 'inputnode.clust_mask_list'),
                                                            ('prune', 'inputnode.prune'),
                                                            ('func_model_list', 'inputnode.conn_model_list'),
                                                            ('min_span_tree', 'inputnode.min_span_tree'),
                                                            ('use_AAL_naming', 'inputnode.use_AAL_naming'),
                                                            ('smooth', 'inputnode.smooth'),
                                                            ('hpass', 'inputnode.hpass'),
                                                            ('hpass_list', 'inputnode.hpass_list'),
                                                            ('disp_filt', 'inputnode.disp_filt'),
                                                            ('clust_type', 'inputnode.clust_type'),
                                                            ('clust_type_list', 'inputnode.clust_type_list'),
                                                            ('c_boot', 'inputnode.c_boot'),
                                                            ('block_size', 'inputnode.block_size'),
                                                            ('mask', 'inputnode.mask'),
                                                            ('norm', 'inputnode.norm'),
                                                            ('binary', 'inputnode.binary'),
                                                            ('template', 'inputnode.template'),
                                                            ('template_mask', 'inputnode.template_mask'),
                                                            ('vox_size', 'inputnode.vox_size'),
                                                            ('local_corr', 'inputnode.local_corr')])
                             ])

            # Connect outputs of nested workflow to parent wf
            meta_wf.connect([(sub_func_wf.get_node('outputnode'), pass_meta_ins_func_node,
                              [('conn_model', 'conn_model'), ('est_path', 'est_path'), ('network', 'network'),
                               ('thr', 'thr'), ('prune', 'prune'), ('ID', 'ID'),
                               ('roi', 'roi'), ('norm', 'norm'), ('binary', 'binary')])
                             ])

    pass_meta_outs_node = pe.Node(niu.Function(input_names=['conn_model_iterlist', 'est_path_iterlist',
                                                            'network_iterlist', 'thr_iterlist', 'prune_iterlist',
                                                            'ID_iterlist', 'roi_iterlist', 'norm_iterlist',
                                                            'binary_iterlist', 'embed', 'multimodal', 'multiplex'],
                                               output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                             'network_iterlist', 'thr_iterlist', 'prune_iterlist',
                                                             'ID_iterlist', 'roi_iterlist', 'norm_iterlist',
                                                             'binary_iterlist'],
                                               function=pass_meta_outs), name='pass_meta_outs_node')

    meta_wf.connect([(meta_inputnode, pass_meta_outs_node, [('embed', 'embed'),
                                                            ('multimodal', 'multimodal'),
                                                            ('multiplex', 'multiplex')])
                     ])

    if (func_file and not dwi_file) or (dwi_file and not func_file):
        if func_file and not dwi_file:
            meta_wf.connect([(pass_meta_ins_func_node, pass_meta_outs_node,
                              [('conn_model_iterlist', 'conn_model_iterlist'),
                               ('est_path_iterlist', 'est_path_iterlist'),
                               ('network_iterlist', 'network_iterlist'),
                               ('thr_iterlist', 'thr_iterlist'),
                               ('prune_iterlist', 'prune_iterlist'),
                               ('ID_iterlist', 'ID_iterlist'),
                               ('roi_iterlist', 'roi_iterlist'),
                               ('norm_iterlist', 'norm_iterlist'),
                               ('binary_iterlist', 'binary_iterlist')])
                             ])
        elif dwi_file and not func_file:
            meta_wf.connect([(pass_meta_ins_struct_node, pass_meta_outs_node,
                              [('conn_model_iterlist', 'conn_model_iterlist'),
                               ('est_path_iterlist', 'est_path_iterlist'),
                               ('network_iterlist', 'network_iterlist'),
                               ('thr_iterlist', 'thr_iterlist'),
                               ('prune_iterlist', 'prune_iterlist'),
                               ('ID_iterlist', 'ID_iterlist'),
                               ('roi_iterlist', 'roi_iterlist'),
                               ('norm_iterlist', 'norm_iterlist'),
                               ('binary_iterlist', 'binary_iterlist')])
                             ])
    elif func_file and dwi_file:
        meta_wf.connect([(pass_meta_ins_multi_node, pass_meta_outs_node,
                          [('conn_model_iterlist', 'conn_model_iterlist'),
                           ('est_path_iterlist', 'est_path_iterlist'),
                           ('network_iterlist', 'network_iterlist'),
                           ('thr_iterlist', 'thr_iterlist'),
                           ('prune_iterlist', 'prune_iterlist'),
                           ('ID_iterlist', 'ID_iterlist'),
                           ('roi_iterlist', 'roi_iterlist'),
                           ('norm_iterlist', 'norm_iterlist'),
                           ('binary_iterlist', 'binary_iterlist')])
                         ])
    else:
        raise ValueError('ERROR: meta-workflow options not defined.')

    # Set resource restrictions at level of the meta wf
    if func_file:
        wf_selected = "%s%s" % ('fmri_connectometry_', ID)
        for node_name in sub_func_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._mem_gb = runtime_dict[node_name][1]
                try:
                    meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name)).interface.n_procs = runtime_dict[node_name][0]
                    meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name)).interface.mem_gb = runtime_dict[node_name][1]
                except:
                    continue

    if dwi_file:
        wf_selected = "%s%s" % ('dmri_connectometry_', ID)
        for node_name in sub_struct_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._mem_gb = runtime_dict[node_name][1]
                try:
                    meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name)).interface.n_procs = runtime_dict[node_name][0]
                    meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name)).interface.mem_gb = runtime_dict[node_name][1]
                except:
                    continue

    return meta_wf


def dmri_connectometry(ID, atlas, network, node_size, roi, uatlas, plot_switch, parc, ref_txt,
                       procmem, dwi_file, fbval, fbvec, anat_file, thr, dens_thresh, conn_model, user_atlas_list,
                       multi_thr, multi_atlas, max_thr, min_thr, step_thr, node_size_list, conn_model_list,
                       min_span_tree, use_AAL_naming, disp_filt, plugin_type, multi_nets, prune, mask, norm,
                       binary, target_samples, curv_thr_list, step_list, overlap_thr, overlap_thr_list,
                       track_type, max_length, maxcrossing, min_length, directget, tiss_class,
                       runtime_dict, execution_dict, multi_directget, template, template_mask, vox_size, waymask):
    """A function interface for generating a dMRI nested workflow"""
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.core import nodemaker, thresholding, utils
    from pynets.registration import register
    from pynets.registration import reg_utils as regutils
    from pynets.dmri import estimation, track
    from pynets.dmri import dmri_utils as dmriutils
    from pynets.plotting import plot_gen
    import os.path as op

    import_list = ["import warnings", "warnings.filterwarnings(\"ignore\")", "import sys", "import os",
                   "import numpy as np", "import networkx as nx", "import indexed_gzip", "import nibabel as nib"]
    base_dirname = "%s%s" % ('dmri_connectometry_', ID)
    dmri_connectometry_wf = pe.Workflow(name=base_dirname)

    # Create basedir_path
    if parc is True:
        basedir_path = utils.do_dir_path('reg_dmri', dwi_file)
    else:
        basedir_path = None

    # Create input/output nodes
    inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'atlas', 'network', 'node_size', 'roi',
                                                      'uatlas', 'plot_switch', 'parc', 'ref_txt', 'procmem',
                                                      'dwi_file', 'fbval', 'fbvec', 'anat_file', 'thr', 'dens_thresh',
                                                      'conn_model', 'user_atlas_list', 'multi_thr', 'multi_atlas',
                                                      'max_thr', 'min_thr', 'step_thr', 'min_span_tree',
                                                      'use_AAL_naming', 'disp_filt', 'multi_nets', 'prune', 'mask',
                                                      'norm', 'binary', 'template', 'template_mask', 'target_samples',
                                                      'curv_thr_list', 'step_list', 'overlap_thr', 'overlap_thr_list',
                                                      'track_type', 'max_length', 'maxcrossing',
                                                      'min_length', 'directget', 'tiss_class', 'vox_size',
                                                      'basedir_path', 'multi_directget', 'waymask']),
                        name='inputnode')

    inputnode.inputs.ID = ID
    inputnode.inputs.atlas = atlas
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.roi = roi
    inputnode.inputs.uatlas = uatlas
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dwi_file = dwi_file
    inputnode.inputs.fbval = fbval
    inputnode.inputs.fbvec = fbvec
    inputnode.inputs.anat_file = anat_file
    inputnode.inputs.thr = thr
    inputnode.inputs.dens_thresh = dens_thresh
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.user_atlas_list = user_atlas_list
    inputnode.inputs.multi_thr = multi_thr
    inputnode.inputs.multi_atlas = multi_atlas
    inputnode.inputs.max_thr = max_thr
    inputnode.inputs.min_thr = min_thr
    inputnode.inputs.step_thr = step_thr
    inputnode.inputs.node_size_list = node_size_list
    inputnode.inputs.conn_model_list = conn_model_list
    inputnode.inputs.min_span_tree = min_span_tree
    inputnode.inputs.use_AAL_naming = use_AAL_naming
    inputnode.inputs.disp_filt = disp_filt
    inputnode.inputs.multi_nets = multi_nets
    inputnode.inputs.prune = prune
    inputnode.inputs.mask = mask
    inputnode.inputs.norm = norm
    inputnode.inputs.binary = binary
    inputnode.inputs.template = template
    inputnode.inputs.template_mask = template_mask
    inputnode.inputs.target_samples = target_samples
    inputnode.inputs.curv_thr_list = curv_thr_list
    inputnode.inputs.step_list = step_list
    inputnode.inputs.overlap_thr = overlap_thr
    inputnode.inputs.overlap_thr_list = overlap_thr_list
    inputnode.inputs.track_type = track_type
    inputnode.inputs.max_length = max_length
    inputnode.inputs.maxcrossing = maxcrossing
    inputnode.inputs.min_length = min_length
    inputnode.inputs.directget = directget
    inputnode.inputs.tiss_class = tiss_class
    inputnode.inputs.plugin_type = plugin_type
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.basedir_path = basedir_path
    inputnode.inputs.multi_directget = multi_directget
    inputnode.inputs.waymask = waymask

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('dwi_file: ', dwi_file))
    # print("%s%s" % ('fbval: ', fbval))
    # print("%s%s" % ('fbvec: ', fbvec))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('atlas: ', atlas))
    # print("%s%s" % ('network: ', network))
    # print("%s%s" % ('node_size: ', node_size))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('uatlas: ', uatlas))
    # print("%s%s" % ('conn_model: ', conn_model))
    # print("%s%s" % ('conn_model_list: ', conn_model_list))
    # print("%s%s" % ('dens_thresh: ', dens_thresh))
    # print("%s%s" % ('plot_switch: ', plot_switch))
    # print("%s%s" % ('multi_thr: ', multi_thr))
    # print("%s%s" % ('multi_atlas: ', multi_atlas))
    # print("%s%s" % ('min_thr: ', min_thr))
    # print("%s%s" % ('max_thr: ', max_thr))
    # print("%s%s" % ('step_thr: ', step_thr))
    # print("%s%s" % ('parc: ', parc))
    # print("%s%s" % ('ref_txt: ', ref_txt))
    # print("%s%s" % ('procmem: ', procmem))
    # print("%s%s" % ('user_atlas_list: ', user_atlas_list))
    # print("%s%s" % ('prune: ', prune))
    # print("%s%s" % ('node_size_list: ', node_size_list))
    # print("%s%s" % ('mask: ', mask))
    # print("%s%s" % ('multi_nets: ', multi_nets))
    # print("%s%s" % ('norm: ', norm))
    # print("%s%s" % ('binary: ', binary))
    # print("%s%s" % ('template: ', template))
    # print("%s%s" % ('template_mask: ', template_mask))
    # print("%s%s" % ('basedir_path: ', basedir_path))
    # print("%s%s" % ('multi_directget: ', multi_directget))
    # print('\n\n\n\n\n')

    # Create function nodes
    check_orient_and_dims_dwi_node = pe.Node(niu.Function(input_names=['infile', 'vox_size', 'bvecs'],
                                                          output_names=['outfile', 'bvecs'],
                                                          function=regutils.check_orient_and_dims,
                                                          imports=import_list), name="check_orient_and_dims_dwi_node")

    check_orient_and_dims_anat_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=regutils.check_orient_and_dims,
                                                           imports=import_list),
                                              name="check_orient_and_dims_anat_node")

    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas', 'uatlas', 'ref_txt', 'parc',
                                                                    'in_file', 'use_AAL_naming'],
                                                       output_names=['labels', 'coords', 'atlas',
                                                                     'networks_list', 'parcel_list', 'par_max',
                                                                     'uatlas', 'dir_path'],
                                                       function=nodemaker.fetch_nodes_and_labels,
                                                       imports=import_list), name="fetch_nodes_and_labels_node")

    fetch_nodes_and_labels_node.synchronize = True

    if parc is False:
        prep_spherical_nodes_node = pe.Node(niu.Function(input_names=['coords', 'node_size', 'template_mask'],
                                                         output_names=['parcel_list', 'par_max', 'node_size', 'parc'],
                                                         function=nodemaker.create_spherical_roi_volumes,
                                                         imports=import_list),
                                            name="prep_spherical_nodes_node")

        if node_size_list:
            prep_spherical_nodes_node.iterables = [("node_size", node_size_list)]

    prep_spherical_nodes_node.synchronize = True

    save_nifti_parcels_node = pe.Node(niu.Function(input_names=['ID', 'dir_path', 'roi', 'network',
                                                                'net_parcels_map_nifti'],
                                                   output_names=['net_parcels_nii_path'],
                                                   function=utils.save_nifti_parcels_map, imports=import_list),
                                      name="save_nifti_parcels_node")

    # Generate nodes
    if roi:
        # Masking case
        node_gen_node = pe.Node(niu.Function(input_names=['roi', 'coords', 'parcel_list', 'labels', 'dir_path',
                                                          'ID', 'parc', 'atlas', 'uatlas'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'labels',
                                                           'atlas', 'uatlas', 'dir_path'],
                                             function=nodemaker.node_gen_masking, imports=import_list),
                                name="node_gen_node")
    else:
        # Non-masking case
        node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'labels', 'dir_path',
                                                          'ID', 'parc', 'atlas', 'uatlas'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'labels',
                                                           'atlas', 'uatlas', 'dir_path'],
                                             function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    # RSN case
    if network or multi_nets:
        get_node_membership_node = pe.Node(niu.Function(input_names=['network', 'infile', 'coords', 'labels',
                                                                     'parc', 'parcel_list', 'perc_overlap', 'error'],
                                                        output_names=['net_coords', 'net_parcel_list', 'net_labels',
                                                                      'network'],
                                                        function=nodemaker.get_node_membership, imports=import_list),
                                           name="get_node_membership_node")

        save_coords_and_labels_node = pe.Node(niu.Function(input_names=['coords', 'labels', 'dir_path', 'network'],
                                                           function=utils.save_RSN_coords_and_labels_to_pickle,
                                                           imports=import_list), name="save_coords_and_labels_node")

        if multi_nets:
            get_node_membership_node_iterables = []
            network_iterables = ("network", multi_nets)
            get_node_membership_node_iterables.append(network_iterables)
            get_node_membership_node.iterables = get_node_membership_node_iterables

    get_node_membership_node.synchronize = True

    gtab_node = pe.Node(niu.Function(input_names=['fbval', 'fbvec', 'dwi_file', 'network', 'node_size', 'atlas'],
                                     output_names=['gtab_file', 'B0_bet', 'B0_mask', 'dwi_file'],
                                     function=dmriutils.make_gtab_and_bmask, imports=import_list), name="gtab_node")

    get_fa_node = pe.Node(niu.Function(input_names=['gtab_file', 'dwi_file', 'B0_mask'],
                                       output_names=['fa_path', 'B0_mask', 'gtab_file', 'dwi_file'],
                                       function=estimation.tens_mod_fa_est, imports=import_list), name="get_fa_node")

    register_node = pe.Node(niu.Function(input_names=['basedir_path', 'fa_path', 'B0_mask', 'anat_file',
                                                      'gtab_file', 'dwi_file', 'vox_size', 'waymask'],
                                         output_names=['wm_in_dwi', 'gm_in_dwi', 'vent_csf_in_dwi',
                                                       'csf_mask_dwi', 'anat_file', 'B0_mask', 'fa_path',
                                                       'gtab_file', 'dwi_file', 'waymask_dwi'],
                                         function=register.register_all_dwi, imports=import_list),
                            name="register_node")

    # Check orientation and resolution
    check_orient_and_dims_uatlas_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'],
                                                             output_names=['outfile'],
                                                             function=regutils.check_orient_and_dims,
                                                             imports=import_list),
                                                name="check_orient_and_dims_uatlas_node")

    register_atlas_node = pe.Node(niu.Function(input_names=['uatlas', 'uatlas_parcels', 'atlas', 'node_size',
                                                            'basedir_path', 'fa_path', 'B0_mask', 'anat_file',
                                                            'coords', 'labels', 'gm_in_dwi', 'vent_csf_in_dwi',
                                                            'wm_in_dwi', 'gtab_file', 'dwi_file', 'vox_size'],
                                               output_names=['dwi_aligned_atlas_wmgm_int', 'dwi_aligned_atlas',
                                                             'aligned_atlas_t1mni', 'uatlas', 'atlas',
                                                             'coords', 'labels', 'node_size', 'gm_in_dwi',
                                                             'vent_csf_in_dwi', 'wm_in_dwi', 'fa_path', 'gtab_file',
                                                             'B0_mask', 'dwi_file'],
                                               function=register.register_atlas_dwi, imports=import_list),
                                  name="register_atlas_node")

    run_tracking_node = pe.Node(niu.Function(input_names=['B0_mask', 'gm_in_dwi', 'vent_csf_in_dwi', 'wm_in_dwi',
                                                          'tiss_class', 'labels_im_file_wm_gm_int',
                                                          'labels_im_file', 'target_samples', 'curv_thr_list',
                                                          'step_list', 'track_type', 'max_length', 'maxcrossing',
                                                          'directget', 'conn_model', 'gtab_file', 'dwi_file', 'network',
                                                          'node_size', 'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                          'disp_filt', 'parc', 'prune', 'atlas',
                                                          'uatlas', 'labels', 'coords', 'norm', 'binary',
                                                          'atlas_mni', 'min_length', 'fa_path', 'waymask'],
                                             output_names=['streams', 'track_type', 'target_samples',
                                                           'conn_model', 'dir_path', 'network', 'node_size',
                                                           'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                           'disp_filt', 'parc', 'prune', 'atlas',
                                                           'uatlas', 'labels', 'coords', 'norm', 'binary',
                                                           'atlas_mni', 'curv_thr_list', 'step_list', 'fa_path',
                                                           'dm_path', 'directget', 'labels_im_file',
                                                           'roi_neighborhood_tol'],
                                             function=track.run_track,
                                             imports=import_list),
                                name="run_tracking_node")
    run_tracking_node.synchronize = True

    # Set reconstruction model iterables
    if conn_model_list or multi_directget:
        run_tracking_node_iterables = []
        if conn_model_list:
            run_tracking_node_iterables.append(("conn_model", conn_model_list))
        else:
            dmri_connectometry_wf.connect([(inputnode, run_tracking_node, [('conn_model', 'conn_model')])])
        if multi_directget:
            run_tracking_node_iterables.append(("directget", multi_directget))
        else:
            dmri_connectometry_wf.connect([(inputnode, run_tracking_node, [('directget', 'directget')])])
        run_tracking_node.iterables = run_tracking_node_iterables
    else:
        dmri_connectometry_wf.connect([(inputnode, run_tracking_node, [('conn_model', 'conn_model'),
                                                                       ('directget', 'directget')])])

    dsn_node = pe.Node(niu.Function(input_names=['streams', 'fa_path', 'dir_path', 'track_type', 'target_samples',
                                                 'conn_model', 'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                 'min_span_tree', 'disp_filt', 'parc', 'prune', 'atlas',
                                                 'labels_im_file', 'uatlas', 'labels', 'coords', 'norm', 'binary',
                                                 'atlas_mni', 'basedir_path', 'curv_thr_list', 'step_list',
                                                 'directget'],
                                    output_names=['streams_mni', 'dir_path', 'track_type', 'target_samples',
                                                  'conn_model', 'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                  'min_span_tree', 'disp_filt', 'parc', 'prune', 'atlas',
                                                  'uatlas', 'labels', 'coords', 'norm', 'binary',
                                                  'atlas_mni', 'directget', 'warped_fa'],
                                    function=register.direct_streamline_norm,
                                    imports=import_list), name="dsn_node")
    dsn_node.synchronize = True

    streams2graph_node = pe.Node(niu.Function(input_names=['atlas_mni', 'streams', 'overlap_thr', 'dir_path',
                                                           'track_type', 'target_samples', 'conn_model',
                                                           'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                           'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                           'atlas', 'uatlas', 'labels', 'coords',
                                                           'norm', 'binary', 'directget', 'warped_fa', 'error_margin'],
                                              output_names=['atlas_mni', 'streams', 'conn_matrix', 'track_type',
                                                            'target_samples', 'dir_path', 'conn_model', 'network',
                                                            'node_size', 'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                            'disp_filt', 'parc', 'prune', 'atlas', 'uatlas', 'labels',
                                                            'coords', 'norm', 'binary', 'directget'],
                                              function=estimation.streams2graph,
                                              imports=import_list), name="streams2graph_node")
    streams2graph_node.synchronize = True

    # Set streams2graph_node iterables
    streams2graph_node_iterables = []
    if overlap_thr_list:
        streams2graph_node_iterables.append(("overlap_thr", overlap_thr_list))
        streams2graph_node.iterables = streams2graph_node_iterables

    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                       'conn_model', 'norm', 'binary']),
                         name='outputnode')

    if (multi_atlas is not None and user_atlas_list is None and uatlas is None) or (multi_atlas is None and
                                                                                    atlas is None and
                                                                                    user_atlas_list is not None):
        flexi_atlas = False
        if multi_atlas is not None and user_atlas_list is None:
            fetch_nodes_and_labels_node.iterables = [("atlas", multi_atlas)]
        elif multi_atlas is None and user_atlas_list is not None:
            fetch_nodes_and_labels_node.iterables = [("uatlas", user_atlas_list)]
    elif ((atlas is not None and uatlas is None) or (atlas is None and uatlas is not None)) and (multi_atlas is None
                                                                                                 and user_atlas_list is
                                                                                                 None):
        flexi_atlas = False
        pass
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(niu.IdentityInterface(fields=['atlas', 'uatlas']),
                                     name='flexi_atlas_source')
        flexi_atlas_source.synchronize = True
        if multi_atlas is not None and user_atlas_list is not None:
            flexi_atlas_source_iterables = [("atlas", len(user_atlas_list) * [None] + multi_atlas),
                                            ("uatlas", user_atlas_list + len(multi_atlas) * [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif multi_atlas is not None and uatlas is not None and user_atlas_list is None:
            flexi_atlas_source_iterables = [("atlas", multi_atlas + [None]),
                                            ("uatlas", len(multi_atlas) * [None] + [uatlas])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and user_atlas_list is not None and multi_atlas is None:
            flexi_atlas_source_iterables = [("atlas", len(user_atlas_list) * [None] + [atlas]),
                                            ("uatlas", user_atlas_list + [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and uatlas is not None and user_atlas_list is None and multi_atlas is None:
            flexi_atlas_source_iterables = [("atlas", [atlas, None]),
                                            ("uatlas", [None, uatlas])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables

    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout joins
    map_fields = ['conn_model', 'dir_path', 'conn_matrix', 'node_size', 'dens_thresh', 'network', 'ID',
                  'roi', 'min_span_tree', 'disp_filt', 'parc', 'prune', 'thr', 'atlas', 'uatlas',
                  'labels', 'coords', 'norm', 'binary', 'target_samples', 'track_type', 'atlas_mni', 'streams',
                  'directget']

    map_connects = [('conn_model', 'conn_model'), ('dir_path', 'dir_path'), ('conn_matrix', 'conn_matrix'),
                    ('node_size', 'node_size'), ('dens_thresh', 'dens_thresh'), ('ID', 'ID'),
                    ('roi', 'roi'), ('min_span_tree', 'min_span_tree'), ('disp_filt', 'disp_filt'), ('parc', 'parc'),
                    ('prune', 'prune'), ('network', 'network'), ('thr', 'thr'), ('atlas', 'atlas'),
                    ('uatlas', 'uatlas'), ('labels', 'labels'), ('coords', 'coords'),
                    ('norm', 'norm'), ('binary', 'binary'), ('target_samples', 'target_samples'),
                    ('track_type', 'track_type'), ('atlas_mni', 'atlas_mni'), ('streams', 'streams'),
                    ('directget', 'directget')]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='thr_info_node')
    # Joinsource logic for atlas varieties
    if user_atlas_list or multi_atlas or flexi_atlas is True:
        if flexi_atlas is True:
            atlas_join_source = flexi_atlas_source
        else:
            atlas_join_source = fetch_nodes_and_labels_node
    else:
        atlas_join_source = None

    # Connect all streams2graph_node outputs to the "thr_info" node
    dmri_connectometry_wf.connect([(streams2graph_node, thr_info_node,
                                    [x for x in map_connects if x != ('thr', 'thr')])])
    # Begin joinnode chaining logic
    if conn_model_list or multi_directget or node_size_list or user_atlas_list or multi_atlas or flexi_atlas is True or multi_thr is True:
        join_iters_node_thr = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_node_thr',
                                          joinsource=thr_info_node,
                                          joinfield=map_fields)
        join_iters_node_atlas = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                            name='join_iters_node_atlas',
                                            joinsource=atlas_join_source,
                                            joinfield=map_fields)
        if not conn_model_list and not multi_directget and (node_size_list and parc is False):
            # print('Node extraction iterables...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_prep_spheres_node',
                                          joinsource=prep_spherical_nodes_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif (conn_model_list or multi_directget) and not node_size_list:
            # print('Multiple connectivity models...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_run_track_node',
                                          joinsource=run_tracking_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif not conn_model_list and not multi_directget and not node_size_list:
            # print('No connectivity model or node extraction iterables...')
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                  name='join_iters_iteratlas_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=thr_info_node, joinfield=map_fields)
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif (conn_model_list or multi_directget) or (node_size_list and parc is False):
            print('Connectivity model and node extraction iterables...')
            join_iters_node_prep_spheres = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                       name='join_iters_node_prep_spheres',
                                                       joinsource=prep_spherical_nodes_node, joinfield=map_fields)
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=run_tracking_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_prep_spheres, map_connects),
                                                   (join_iters_node_prep_spheres, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_prep_spheres, map_connects),
                                                   (join_iters_node_prep_spheres, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_prep_spheres, map_connects),
                                                   (join_iters_node_prep_spheres, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    dmri_connectometry_wf.connect([(thr_info_node, join_iters_node_prep_spheres, map_connects),
                                                   (join_iters_node_prep_spheres, join_iters_node, map_connects)])
        else:
            raise RuntimeError('\nERROR: Unknown join context.')

        no_iters = False
    else:
        # Minimal case of no iterables
        print('\nNo iterables...\n')
        join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
        dmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        no_iters = True

    # Create final thresh_diff node that performs the thresholding
    if no_iters is True:
        thresh_diff_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                             'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                             'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                             'atlas', 'uatlas', 'labels', 'coords',
                                                             'norm', 'binary', 'target_samples', 'track_type',
                                                             'atlas_mni', 'streams', 'directget'],
                                                output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                              'node_size', 'network', 'conn_model', 'roi',
                                                              'prune', 'ID', 'dir_path', 'atlas',
                                                              'uatlas', 'labels', 'coords',
                                                              'norm', 'binary', 'target_samples',
                                                              'track_type', 'atlas_mni', 'streams', 'directget'],
                                                function=thresholding.thresh_struct, imports=import_list),
                                   name="thresh_diff_node")
    else:
        thresh_diff_node = pe.MapNode(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                                'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                                'atlas', 'uatlas', 'labels',
                                                                'coords', 'norm', 'binary', 'target_samples',
                                                                'track_type', 'atlas_mni', 'streams', 'directget'],
                                                   output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                                 'node_size', 'network', 'conn_model', 'roi',
                                                                 'prune', 'ID', 'dir_path', 'atlas',
                                                                 'uatlas', 'labels', 'coords',
                                                                 'norm', 'binary', 'target_samples', 'track_type',
                                                                 'atlas_mni', 'streams', 'directget'],
                                                   function=thresholding.thresh_struct, imports=import_list),
                                      name="thresh_diff_node", iterfield=['dens_thresh', 'thr', 'conn_matrix',
                                                                          'conn_model', 'network', 'ID', 'dir_path',
                                                                          'roi', 'node_size', 'min_span_tree',
                                                                          'disp_filt', 'parc', 'prune', 'atlas',
                                                                          'uatlas', 'labels', 'coords', 'norm',
                                                                          'binary', 'target_samples', 'track_type',
                                                                          'atlas_mni', 'streams', 'directget'],
                                      nested=True)
        thresh_diff_node.synchronize = True

    # Set iterables for thr on thresh_diff, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr), float(max_thr),
                                                                          float(step_thr)), decimals=2).tolist()] +
                                      [str(float(max_thr))])))
        thr_info_node.iterables = ("thr", iter_thresh)
    else:
        thr_info_node.iterables = ("thr", [thr])

    thr_info_node.synchronize = True

    # Plotting
    if plot_switch is True:
        plot_fields = ['conn_matrix', 'conn_model', 'atlas', 'dir_path', 'ID', 'network', 'labels', 'roi',
                       'coords', 'thr', 'node_size', 'edge_threshold', 'prune', 'uatlas', 'target_samples', 'norm',
                       'binary', 'track_type', 'directget']

        # # Plotting iterable graph solutions
        if conn_model_list or node_size_list or multi_directget or multi_thr or user_atlas_list or multi_atlas or flexi_atlas is True:
            plot_all_node = pe.MapNode(niu.Function(input_names=plot_fields, output_names='None',
                                                    function=plot_gen.plot_all_struct, imports=import_list),
                                       itersource=thr_info_node,
                                       iterfield=plot_fields,
                                       name="plot_all_node", nested=True)
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(niu.Function(input_names=plot_fields, output_names='None',
                                                 function=plot_gen.plot_all_struct, imports=import_list),
                                    name="plot_all_node")

        # Connect thresh_diff_node outputs to plotting node
        dmri_connectometry_wf.connect([(thresh_diff_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                                                          ('conn_model', 'conn_model'),
                                                                          ('atlas', 'atlas'),
                                                                          ('dir_path', 'dir_path'),
                                                                          ('ID', 'ID'),
                                                                          ('network', 'network'),
                                                                          ('labels', 'labels'),
                                                                          ('roi', 'roi'),
                                                                          ('coords', 'coords'),
                                                                          ('thr', 'thr'),
                                                                          ('node_size', 'node_size'),
                                                                          ('edge_threshold', 'edge_threshold'),
                                                                          ('prune', 'prune'),
                                                                          ('atlas_mni', 'uatlas'),
                                                                          ('target_samples', 'target_samples'),
                                                                          ('norm', 'norm'),
                                                                          ('binary', 'binary'),
                                                                          ('track_type', 'track_type'),
                                                                          ('directget', 'directget')])
                                       ])

    # Connect nodes of workflow
    dmri_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('atlas', 'atlas'),
                                                  ('uatlas', 'uatlas'),
                                                  ('parc', 'parc'),
                                                  ('ref_txt', 'ref_txt'),
                                                  ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, node_gen_node, [('ID', 'ID')]),
        (inputnode, check_orient_and_dims_dwi_node, [('dwi_file', 'infile'),
                                                     ('fbvec', 'bvecs'),
                                                     ('vox_size', 'vox_size')]),
        (check_orient_and_dims_dwi_node, fetch_nodes_and_labels_node, [('outfile', 'in_file')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path'),
                                                      ('par_max', 'par_max'),
                                                      ('networks_list', 'networks_list')]),
        (check_orient_and_dims_dwi_node, gtab_node, [('bvecs', 'fbvec'),
                                                     ('outfile', 'dwi_file')]),
        (inputnode, gtab_node, [('fbval', 'fbval')]),
        (fetch_nodes_and_labels_node, gtab_node, [('atlas', 'atlas')]),
        (inputnode, register_node, [('basedir_path', 'basedir_path'), ('vox_size', 'vox_size')]),
        (inputnode, check_orient_and_dims_anat_node, [('anat_file', 'infile'), ('vox_size', 'vox_size')]),
        (inputnode, check_orient_and_dims_uatlas_node, [('vox_size', 'vox_size')]),
        (fetch_nodes_and_labels_node, check_orient_and_dims_uatlas_node, [('uatlas', 'infile')]),
        (check_orient_and_dims_uatlas_node, register_atlas_node, [('outfile', 'uatlas')]),
        (check_orient_and_dims_anat_node, register_node, [('outfile', 'anat_file')]),
        (inputnode, register_atlas_node, [('basedir_path', 'basedir_path'), ('vox_size', 'vox_size')]),
        (register_node, register_atlas_node, [('anat_file', 'anat_file'),
                                              ('gm_in_dwi', 'gm_in_dwi'),
                                              ('vent_csf_in_dwi', 'vent_csf_in_dwi'),
                                              ('wm_in_dwi', 'wm_in_dwi'),
                                              ('fa_path', 'fa_path'),
                                              ('B0_mask', 'B0_mask'),
                                              ('gtab_file', 'gtab_file'),
                                              ('dwi_file', 'dwi_file')]),
        (gtab_node, get_fa_node, [('B0_mask', 'B0_mask'),
                                  ('gtab_file', 'gtab_file'),
                                  ('dwi_file', 'dwi_file')]),
        (get_fa_node, register_node, [('fa_path', 'fa_path'),
                                      ('B0_mask', 'B0_mask'),
                                      ('gtab_file', 'gtab_file'),
                                      ('dwi_file', 'dwi_file')]),
        (register_atlas_node, run_tracking_node, [('dwi_aligned_atlas_wmgm_int', 'labels_im_file_wm_gm_int'),
                                                  ('dwi_aligned_atlas', 'labels_im_file'),
                                                  ('fa_path', 'fa_path'),
                                                  ('aligned_atlas_t1mni', 'atlas_mni'),
                                                  ('atlas', 'atlas'),
                                                  ('uatlas', 'uatlas'),
                                                  ('coords', 'coords'),
                                                  ('labels', 'labels'),
                                                  ('gm_in_dwi', 'gm_in_dwi'),
                                                  ('vent_csf_in_dwi', 'vent_csf_in_dwi'),
                                                  ('wm_in_dwi', 'wm_in_dwi'),
                                                  ('gtab_file', 'gtab_file'),
                                                  ('B0_mask', 'B0_mask'),
                                                  ('dwi_file', 'dwi_file')]),
        (inputnode, run_tracking_node, [('tiss_class', 'tiss_class'),
                                        ('dens_thresh', 'dens_thresh'),
                                        ('ID', 'ID'),
                                        ('roi', 'roi'),
                                        ('min_span_tree', 'min_span_tree'),
                                        ('disp_filt', 'disp_filt'),
                                        ('parc', 'parc'),
                                        ('prune', 'prune'),
                                        ('norm', 'norm'),
                                        ('binary', 'binary'),
                                        ('target_samples', 'target_samples'),
                                        ('curv_thr_list', 'curv_thr_list'),
                                        ('step_list', 'step_list'),
                                        ('track_type', 'track_type'),
                                        ('max_length', 'max_length'),
                                        ('maxcrossing', 'maxcrossing'),
                                        ('min_length', 'min_length')]),
        (inputnode, streams2graph_node, [('overlap_thr', 'overlap_thr')]),
        (inputnode, dsn_node, [('basedir_path', 'basedir_path')]),
        (run_tracking_node, dsn_node, [('dir_path', 'dir_path'),
                                       ('streams', 'streams'),
                                       ('curv_thr_list', 'curv_thr_list'),
                                       ('step_list', 'step_list'),
                                       ('track_type', 'track_type'),
                                       ('target_samples', 'target_samples'),
                                       ('conn_model', 'conn_model'),
                                       ('node_size', 'node_size'),
                                       ('dens_thresh', 'dens_thresh'),
                                       ('ID', 'ID'),
                                       ('roi', 'roi'),
                                       ('min_span_tree', 'min_span_tree'),
                                       ('disp_filt', 'disp_filt'),
                                       ('parc', 'parc'),
                                       ('prune', 'prune'),
                                       ('atlas', 'atlas'),
                                       ('labels_im_file', 'labels_im_file'),
                                       ('labels', 'labels'),
                                       ('coords', 'coords'),
                                       ('norm', 'norm'),
                                       ('binary', 'binary'),
                                       ('atlas_mni', 'atlas_mni'),
                                       ('fa_path', 'fa_path'),
                                       ('directget', 'directget')]),
        (run_tracking_node, streams2graph_node, [('roi_neighborhood_tol', 'error_margin')]),
        (dsn_node, streams2graph_node, [('streams_mni', 'streams'),
                                        ('dir_path', 'dir_path'),
                                        ('track_type', 'track_type'),
                                        ('target_samples', 'target_samples'),
                                        ('conn_model', 'conn_model'),
                                        ('node_size', 'node_size'),
                                        ('dens_thresh', 'dens_thresh'),
                                        ('ID', 'ID'),
                                        ('roi', 'roi'),
                                        ('min_span_tree', 'min_span_tree'),
                                        ('disp_filt', 'disp_filt'),
                                        ('parc', 'parc'),
                                        ('prune', 'prune'),
                                        ('atlas', 'atlas'),
                                        ('uatlas', 'uatlas'),
                                        ('labels', 'labels'),
                                        ('coords', 'coords'),
                                        ('norm', 'norm'),
                                        ('binary', 'binary'),
                                        ('atlas_mni', 'atlas_mni'),
                                        ('directget', 'directget'),
                                        ('warped_fa', 'warped_fa')]),
        (join_iters_node, thresh_diff_node, map_connects)
    ])

    if flexi_atlas is True:
        dmri_connectometry_wf.disconnect([(inputnode, fetch_nodes_and_labels_node,
                                           [('uatlas', 'uatlas'),
                                            ('atlas', 'atlas')])
                                          ])
        dmri_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                        [('uatlas', 'uatlas'),
                                         ('atlas', 'atlas')])
                                       ])

    if waymask is not None:
        check_orient_and_dims_waymask_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'],
                                                                  output_names=['outfile'],
                                                                  function=regutils.check_orient_and_dims,
                                                                  imports=import_list),
                                                     name="check_orient_and_dims_waymask_node")
        dmri_connectometry_wf.connect([
            (inputnode, check_orient_and_dims_waymask_node, [('waymask', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_waymask_node, register_node, [('outfile', 'waymask')]),
            (register_node, run_tracking_node, [('waymask_dwi', 'waymask')]),
        ])
    else:
        dmri_connectometry_wf.connect([
            (inputnode, register_node, [('waymask', 'waymask')]),
            (inputnode, run_tracking_node, [('waymask', 'waymask')]),
        ])

    # Handle masking scenarios (brain mask and/or roi)
    if (mask is not None) and (roi is None):
        check_orient_and_dims_mask_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'],
                                                               output_names=['outfile'],
                                                               function=regutils.check_orient_and_dims,
                                                               imports=import_list),
                                                  name="check_orient_and_dims_mask_node")
        dmri_connectometry_wf.connect([
            (inputnode, check_orient_and_dims_mask_node, [('mask', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_mask_node, node_gen_node, [('outfile', 'roi')]),
        ])
    elif (op.isfile(template_mask) is True) and (roi is None):
        dmri_connectometry_wf.connect([
            (inputnode, node_gen_node, [('template_mask', 'roi')]),
        ])
    else:
        dmri_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
        ])

    if parc is False:
        if network or multi_nets:
            dmri_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                  ('template', 'infile'),
                                                                                  ('parc', 'parc')]),
                                           (get_node_membership_node, prep_spherical_nodes_node,
                                            [('net_coords', 'coords')]),
                                           (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                            [('dir_path', 'dir_path')]),
                                           (fetch_nodes_and_labels_node, get_node_membership_node,
                                            [('coords', 'coords'), ('labels', 'labels'),
                                             ('networks_list', 'networks_list'), ('parcel_list', 'parcel_list')]),
                                           (get_node_membership_node, save_coords_and_labels_node,
                                            [('net_coords', 'coords'), ('net_labels', 'labels'),
                                             ('network', 'network')]),
                                           (get_node_membership_node, run_tracking_node,
                                            [('network', 'network')]),
                                           (prep_spherical_nodes_node, node_gen_node,
                                            [('parc', 'parc'),
                                             ('parcel_list', 'parcel_list')]),
                                           (get_node_membership_node, node_gen_node,
                                            [('net_coords', 'coords'), ('net_labels', 'labels')]),
                                           (get_node_membership_node, gtab_node, [('network', 'network')]),
                                           (get_node_membership_node, save_nifti_parcels_node,
                                            [('network', 'network')]),
                                           (prep_spherical_nodes_node, gtab_node, [('node_size', 'node_size')]),
                                           (save_nifti_parcels_node, register_atlas_node, [('net_parcels_nii_path',
                                                                                            'uatlas_parcels')]),
                                           (save_nifti_parcels_node, dsn_node, [('net_parcels_nii_path', 'uatlas')])
                                           ])
        else:
            dmri_connectometry_wf.connect([(prep_spherical_nodes_node, node_gen_node,
                                            [('parcel_list', 'parcel_list'),
                                             ('par_max', 'par_max'),
                                             ('parc', 'parc')]),
                                           (fetch_nodes_and_labels_node, prep_spherical_nodes_node,
                                            [('coords', 'coords')]),
                                           (fetch_nodes_and_labels_node, node_gen_node,
                                            [('coords', 'coords'),
                                             ('labels', 'labels')]),
                                           (inputnode, save_nifti_parcels_node,
                                            [('network', 'network')]),
                                           (inputnode, gtab_node, [('network', 'network')]),
                                           (prep_spherical_nodes_node, gtab_node, [('node_size', 'node_size')]),
                                           (node_gen_node, register_atlas_node, [('uatlas', 'uatlas_parcels')]),
                                           (run_tracking_node, dsn_node, [('uatlas', 'uatlas')]),
                                           (inputnode, run_tracking_node, [('network', 'network')])
                                           ])

        dmri_connectometry_wf.connect([(inputnode, prep_spherical_nodes_node,
                                        [('node_size', 'node_size'),
                                         ('template_mask', 'template_mask')]),
                                       (fetch_nodes_and_labels_node, prep_spherical_nodes_node,
                                        [('dir_path', 'dir_path')]),
                                       (inputnode, save_nifti_parcels_node,
                                        [('ID', 'ID'),
                                         ('roi', 'roi')]),
                                       (fetch_nodes_and_labels_node, node_gen_node,
                                        [('atlas', 'atlas'), ('uatlas', 'uatlas')]),
                                       (node_gen_node, register_atlas_node, [('atlas', 'atlas'),
                                                                             ('coords', 'coords'),
                                                                             ('labels', 'labels')]),
                                       (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                        [('dir_path', 'dir_path')]),
                                       (node_gen_node, save_nifti_parcels_node,
                                        [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
                                       (prep_spherical_nodes_node, register_atlas_node,
                                        [('node_size', 'node_size')]),
                                       (register_atlas_node, run_tracking_node,
                                        [('node_size', 'node_size')]),
                                       (run_tracking_node, dsn_node,
                                        [('network', 'network')]),
                                       (dsn_node, streams2graph_node,
                                        [('network', 'network')])
                                       ])
    else:
        if network or multi_nets:
            dmri_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                  ('template', 'infile'),
                                                                                  ('parc', 'parc')]),
                                           (fetch_nodes_and_labels_node, get_node_membership_node,
                                            [('coords', 'coords'), ('labels', 'labels'),
                                             ('parcel_list', 'parcel_list'), ('par_max', 'par_max'),
                                             ('networks_list', 'networks_list')]),
                                           (get_node_membership_node, node_gen_node,
                                            [('net_coords', 'coords'), ('net_labels', 'labels'),
                                             ('net_parcel_list', 'parcel_list')]),
                                           (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                            [('dir_path', 'dir_path')]),
                                           (get_node_membership_node, run_tracking_node, [('network', 'network')]),
                                           (get_node_membership_node, gtab_node, [('network', 'network')]),
                                           (get_node_membership_node, save_nifti_parcels_node,
                                            [('network', 'network')]),
                                           (inputnode, gtab_node, [('node_size', 'node_size')]),
                                           (get_node_membership_node, save_coords_and_labels_node,
                                            [('net_coords', 'coords'), ('net_labels', 'labels'),
                                             ('network', 'network')]),
                                           (save_nifti_parcels_node, dsn_node, [('net_parcels_nii_path', 'uatlas')])
                                           ])
        else:
            dmri_connectometry_wf.connect([(fetch_nodes_and_labels_node, node_gen_node,
                                            [('coords', 'coords'),
                                             ('labels', 'labels'),
                                             ('parcel_list', 'parcel_list')]),
                                           (inputnode, gtab_node, [('network', 'network'),
                                                                   ('node_size', 'node_size')]),
                                           (inputnode, save_nifti_parcels_node,
                                            [('network', 'network')]),
                                           (inputnode, run_tracking_node, [('network', 'network')]),
                                           (run_tracking_node, dsn_node, [('uatlas', 'uatlas')])
                                           ])

        dmri_connectometry_wf.connect([(inputnode, run_tracking_node,
                                        [('node_size', 'node_size')]),
                                       (inputnode, node_gen_node,
                                        [('parc', 'parc')]),
                                       (fetch_nodes_and_labels_node, node_gen_node,
                                        [('atlas', 'atlas'), ('uatlas', 'uatlas')]),
                                       (inputnode, save_nifti_parcels_node,
                                        [('ID', 'ID'),
                                         ('roi', 'roi')]),
                                       (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                        [('dir_path', 'dir_path')]),
                                       (node_gen_node, save_nifti_parcels_node,
                                        [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
                                       (save_nifti_parcels_node, register_atlas_node, [('net_parcels_nii_path',
                                                                                        'uatlas_parcels')]),
                                       (node_gen_node, register_atlas_node, [('atlas', 'atlas'),
                                                                             ('coords', 'coords'),
                                                                             ('labels', 'labels')]),
                                       (run_tracking_node, dsn_node,
                                        [('network', 'network')]),
                                       (inputnode, register_atlas_node,
                                        [('node_size', 'node_size')]),
                                       (dsn_node, streams2graph_node,
                                        [('network', 'network')])
                                       ])

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune',
                                                                         'ID', 'roi', 'conn_model', 'node_size',
                                                                         'target_samples', 'track_type', 'norm',
                                                                         'binary', 'atlas_mni', 'streams',
                                                                         'directget']),
                                           name='join_iters_node_nets', joinsource=get_node_membership_node,
                                           joinfield=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                      'conn_model', 'node_size', 'target_samples', 'track_type', 'norm',
                                                      'binary', 'atlas_mni', 'streams', 'directget'])
        dmri_connectometry_wf.connect([
            (thresh_diff_node, join_iters_node_nets, [('thr', 'thr'), ('network', 'network'),
                                                      ('est_path', 'est_path'), ('node_size', 'node_size'),
                                                      ('track_type', 'track_type'), ('roi', 'roi'),
                                                      ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                      ('prune', 'prune'), ('target_samples', 'target_samples'),
                                                      ('norm', 'norm'), ('binary', 'binary'),
                                                      ('atlas_mni', 'atlas_mni'), ('streams', 'streams'),
                                                      ('directget', 'directget')]),
            (join_iters_node_nets, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                                ('roi', 'roi'), ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')])
        ])
    else:
        dmri_connectometry_wf.connect([
            (thresh_diff_node, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                            ('roi', 'roi'), ('conn_model', 'conn_model'), ('ID', 'ID'),
                                            ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')]),
        ])

    for node_name in dmri_connectometry_wf.list_node_names():
        if node_name in runtime_dict:
            dmri_connectometry_wf.get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
            dmri_connectometry_wf.get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
            dmri_connectometry_wf.get_node(node_name).n_procs = runtime_dict[node_name][0]
            dmri_connectometry_wf.get_node(node_name)._mem_gb = runtime_dict[node_name][1]

    execution_dict['plugin_args'] = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                                     'scheduler': 'mem_thread'}
    execution_dict['plugin'] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            dmri_connectometry_wf.config[key][setting] = value

    return dmri_connectometry_wf


def fmri_connectometry(func_file, ID, atlas, network, node_size, roi, thr, uatlas, conn_model,
                       dens_thresh, conf, plot_switch, parc, ref_txt, procmem, multi_thr,
                       multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step,
                       k_clustering, user_atlas_list, clust_mask_list, node_size_list, conn_model_list,
                       min_span_tree, use_AAL_naming, smooth, smooth_list, disp_filt, prune, multi_nets,
                       clust_type, clust_type_list, plugin_type, c_boot, block_size, mask, norm, binary,
                       anat_file, runtime_dict, execution_dict, hpass, hpass_list, template, template_mask, vox_size,
                       local_corr):
    """A function interface for generating an fMRI nested workflow"""
    import os.path as op
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.core import nodemaker, utils, thresholding
    from pynets.plotting import plot_gen
    from pynets.fmri import estimation
    from pynets.registration import register
    from pynets.registration import reg_utils as regutils
    from pynets.core.interfaces import ExtractTimeseries

    import_list = ["import warnings", "warnings.filterwarnings(\"ignore\")", "import sys", "import os",
                   "import numpy as np", "import networkx as nx", "import indexed_gzip", "import nibabel as nib"]
    base_dirname = "%s%s" % ('fmri_connectometry_', ID)
    fmri_connectometry_wf = pe.Workflow(name=base_dirname)

    # Create basedir_path
    if parc is True:
        basedir_path = utils.do_dir_path('reg_fmri', func_file)
    else:
        basedir_path = None

    # Create input/output nodes
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 'atlas', 'network',
                                                      'node_size', 'roi', 'thr',
                                                      'uatlas', 'multi_nets',
                                                      'conn_model', 'dens_thresh',
                                                      'conf', 'plot_switch', 'parc', 'ref_txt',
                                                      'procmem', 'k', 'clust_mask', 'k_min', 'k_max',
                                                      'k_step', 'k_clustering', 'user_atlas_list',
                                                      'min_span_tree', 'use_AAL_naming', 'smooth',
                                                      'disp_filt', 'prune', 'clust_type',
                                                      'c_boot', 'block_size', 'mask', 'norm', 'binary', 'template',
                                                      'template_mask', 'vox_size', 'anat_file', 'basedir_path',
                                                      'hpass', 'hpass_list', 'local_corr']),
                        name='inputnode')

    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas = atlas
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.roi = roi
    inputnode.inputs.thr = thr
    inputnode.inputs.uatlas = uatlas
    inputnode.inputs.multi_nets = multi_nets
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.dens_thresh = dens_thresh
    inputnode.inputs.conf = conf
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.k = k
    inputnode.inputs.clust_mask = clust_mask
    inputnode.inputs.k_min = k_min
    inputnode.inputs.k_max = k_max
    inputnode.inputs.k_step = k_step
    inputnode.inputs.k_clustering = k_clustering
    inputnode.inputs.user_atlas_list = user_atlas_list
    inputnode.inputs.multi_thr = multi_thr
    inputnode.inputs.multi_atlas = multi_atlas
    inputnode.inputs.max_thr = max_thr
    inputnode.inputs.min_thr = min_thr
    inputnode.inputs.step_thr = step_thr
    inputnode.inputs.clust_mask_list = clust_mask_list
    inputnode.inputs.conn_model_list = conn_model_list
    inputnode.inputs.min_span_tree = min_span_tree
    inputnode.inputs.use_AAL_naming = use_AAL_naming
    inputnode.inputs.smooth = smooth
    inputnode.inputs.disp_filt = disp_filt
    inputnode.inputs.prune = prune
    inputnode.inputs.clust_type = clust_type
    inputnode.inputs.clust_type_list = clust_type_list
    inputnode.inputs.c_boot = c_boot
    inputnode.inputs.block_size = block_size
    inputnode.inputs.mask = mask
    inputnode.inputs.norm = norm
    inputnode.inputs.binary = binary
    inputnode.inputs.template = template
    inputnode.inputs.template_mask = template_mask
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.anat_file = anat_file
    inputnode.inputs.basedir_path = basedir_path
    inputnode.inputs.hpass = hpass
    inputnode.inputs.hpass_list = hpass_list
    inputnode.inputs.local_corr = local_corr

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('atlas: ', atlas))
    # print("%s%s" % ('network: ', network))
    # print("%s%s" % ('node_size: ', node_size))
    # print("%s%s" % ('smooth: ', smooth))
    # print("%s%s" % ('hpass: ', hpass))
    # print("%s%s" % ('hpass_list: ', hpass_list))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('uatlas: ', uatlas))
    # print("%s%s" % ('conn_model: ', conn_model))
    # print("%s%s" % ('conn_model_list: ', conn_model_list))
    # print("%s%s" % ('dens_thresh: ', dens_thresh))
    # print("%s%s" % ('conf: ', conf))
    # print("%s%s" % ('plot_switch: ', plot_switch))
    # print("%s%s" % ('multi_thr): ', multi_thr))
    # print("%s%s" % ('multi_atlas: ', multi_atlas))
    # print("%s%s" % ('min_thr: ', min_thr))
    # print("%s%s" % ('max_thr: ', max_thr))
    # print("%s%s" % ('step_thr: ', step_thr))
    # print("%s%s" % ('parc: ', parc))
    # print("%s%s" % ('ref_txt: ', ref_txt))
    # print("%s%s" % ('procmem: ', procmem))
    # print("%s%s" % ('k: ', k))
    # print("%s%s" % ('clust_mask: ', clust_mask))
    # print("%s%s" % ('k_min: ', k_min))
    # print("%s%s" % ('k_max: ', k_max))
    # print("%s%s" % ('k_step): ', k_step))
    # print("%s%s" % ('k_clustering: ', k_clustering))
    # print("%s%s" % ('user_atlas_list: ', user_atlas_list))
    # print("%s%s" % ('clust_mask_list: ', clust_mask_list))
    # print("%s%s" % ('prune: ', prune))
    # print("%s%s" % ('node_size_list: ', node_size_list))
    # print("%s%s" % ('smooth_list: ', smooth_list))
    # print("%s%s" % ('clust_type: ', clust_type))
    # print("%s%s" % ('clust_type_list: ', clust_type_list))
    # print("%s%s" % ('c_boot: ', c_boot))
    # print("%s%s" % ('block_size: ', block_size))
    # print("%s%s" % ('mask: ', mask))
    # print("%s%s" % ('norm: ', norm))
    # print("%s%s" % ('binary: ', binary))
    # print("%s%s" % ('template: ', template))
    # print("%s%s" % ('template_mask: ', template_mask))
    # print("%s%s" % ('vox_size: ', vox_size))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('basedir_path: ', basedir_path))
    # print("%s%s" % ('local_corr: ', local_corr))
    # print('\n\n\n\n\n')

    # Create function nodes
    check_orient_and_dims_func_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=regutils.check_orient_and_dims,
                                                           imports=import_list),
                                              name="check_orient_and_dims_func_node")

    check_orient_and_dims_func_node._n_procs = runtime_dict['check_orient_and_dims_func_node'][0]
    check_orient_and_dims_func_node._mem_gb = runtime_dict['check_orient_and_dims_func_node'][1]

    check_orient_and_dims_anat_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=regutils.check_orient_and_dims,
                                                           imports=import_list),
                                              name="check_orient_and_dims_anat_node")

    register_node = pe.Node(niu.Function(input_names=['basedir_path', 'anat_file', 'vox_size'],
                                         output_names=['reg_fmri_complete'],
                                         function=register.register_all_fmri, imports=import_list),
                            name="register_node")

    register_node._n_procs = runtime_dict['register_node'][0]
    register_node._mem_gb = runtime_dict['register_node'][1]

    register_atlas_node = pe.Node(niu.Function(input_names=['uatlas', 'uatlas_parcels', 'atlas',
                                                            'basedir_path', 'anat_file', 'vox_size',
                                                            'reg_fmri_complete'],
                                               output_names=['aligned_atlas_t1mni_gm'],
                                               function=register.register_atlas_fmri, imports=import_list),
                                  name="register_atlas_node")

    register_atlas_node._n_procs = runtime_dict['register_atlas_node'][0]
    register_atlas_node._mem_gb = runtime_dict['register_atlas_node'][1]

    # Clustering
    if float(k_clustering) > 0:
        from pynets.core.interfaces import IndividualClustering
        clustering_info_node = pe.Node(niu.IdentityInterface(fields=['clust_mask', 'clust_type', 'k']),
                                       name="clustering_info_node")

        clustering_node = pe.Node(IndividualClustering(),
                                  input_names=['func_file', 'conf', 'clust_mask', 'ID', 'k', 'clust_type', 'vox_size',
                                               'local_corr', 'mask'],
                                  output_names=['uatlas', 'atlas', 'clustering', 'clust_mask', 'k', 'clust_type'],
                                  imports=import_list, name="clustering_node")

        clustering_node.interface.n_procs = runtime_dict['clustering_node'][0]
        clustering_node.interface.mem_gb = runtime_dict['clustering_node'][1]
        clustering_node._n_procs = runtime_dict['clustering_node'][0]
        clustering_node._mem_gb = runtime_dict['clustering_node'][1]

        # Don't forget that this setting exists
        clustering_node.synchronize = True

        # clustering_node iterables and names
        if k_clustering == 1:
            mask_name = op.basename(clust_mask).split('.nii')[0]
            cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
            cluster_atlas_file = "%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/',
                                                       mask_name, '_', clust_type, '_k', str(k), '.nii.gz')
            if user_atlas_list:
                user_atlas_list.append(cluster_atlas_file)
            elif uatlas and ((uatlas == cluster_atlas_file) is False):
                user_atlas_list = [uatlas, cluster_atlas_file]
            else:
                uatlas = cluster_atlas_file
        elif k_clustering == 2:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for k in k_list:
                mask_name = op.basename(clust_mask).split('.nii')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 3:
            clustering_info_node.iterables = [("clust_mask", clust_mask_list)]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                mask_name = op.basename(clust_mask).split('.nii')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 4:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split('.nii')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                           func_file), '/', mask_name,
                                                                         '_', clust_type, '_k', str(k), '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 5:
            clustering_info_node.iterables = [("clust_type", clust_type_list)]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                mask_name = op.basename(clust_mask).split('.nii')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 6:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split('.nii')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                           func_file), '/', mask_name,
                                                                         '_', clust_type, '_k', str(k), '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 7:
            clustering_info_node.iterables = [("clust_type", clust_type_list), ("clust_mask", clust_mask_list)]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    mask_name = op.basename(clust_mask).split('.nii')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                           func_file), '/', mask_name,
                                                                         '_', clust_type, '_k', str(k),
                                                                         '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 8:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        mask_name = op.basename(clust_mask).split('.nii')[0]
                        cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                        cluster_atlas_name_list.append(cluster_atlas_name)
                        cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                               func_file), '/',
                                                                             mask_name, '_', clust_type, '_k', str(k),
                                                                             '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas:
                user_atlas_list = cluster_atlas_file_list + [uatlas]
            else:
                user_atlas_list = cluster_atlas_file_list

    # Define nodes
    # Create node definitions Node
    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas', 'uatlas', 'ref_txt',
                                                                    'parc', 'in_file', 'use_AAL_naming', 'clustering'],
                                                       output_names=['labels', 'coords', 'atlas',
                                                                     'networks_list', 'parcel_list', 'par_max',
                                                                     'uatlas', 'dir_path'],
                                                       function=nodemaker.fetch_nodes_and_labels,
                                                       imports=import_list), name="fetch_nodes_and_labels_node")

    fetch_nodes_and_labels_node.synchronize = True

    # Connect clustering solutions to node definition Node
    if float(k_clustering) > 0:
        fmri_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'), ('conf', 'conf'),
                                                                     ('local_corr', 'local_corr')]),
                                       (check_orient_and_dims_func_node, clustering_node,
                                        [('outfile', 'func_file')]),
                                       (inputnode, clustering_node,
                                        [('vox_size', 'vox_size')]),
                                       (clustering_node, fetch_nodes_and_labels_node,
                                        [('uatlas', 'uatlas'),
                                         ('atlas', 'atlas'),
                                         ('clustering', 'clustering')]),
                                       (inputnode, clustering_info_node,
                                        [('clust_mask', 'clust_mask'), ('clust_type', 'clust_type'), ('k', 'k')]),
                                       (clustering_info_node, clustering_node,
                                        [('clust_mask', 'clust_mask'), ('clust_type', 'clust_type'), ('k', 'k')])
                                       ])
    else:
        # Connect atlas input vars to node definition Node
        fmri_connectometry_wf.connect([(inputnode, fetch_nodes_and_labels_node,
                                        [('atlas', 'atlas'),
                                         ('uatlas', 'uatlas')])
                                       ])

    # Set atlas iterables and logic for multiple atlas useage
    if ((multi_atlas is not None and user_atlas_list is None and
         uatlas is None) or (multi_atlas is None and atlas is
                             None and user_atlas_list is not None)) and k_clustering == 0:
        # print('\n\n\n\n')
        # print('No flexi-atlas1')
        # print('\n\n\n\n')
        flexi_atlas = False
        if multi_atlas:
            fetch_nodes_and_labels_node.iterables = [("atlas", multi_atlas)]
        elif user_atlas_list:
            fetch_nodes_and_labels_node.iterables = [("uatlas", user_atlas_list)]

    elif (atlas is not None and
          uatlas is None and k_clustering == 0) or (atlas is None and uatlas is not None and
                                                    k_clustering == 0) or (k_clustering > 0 and atlas is
                                                                           None and multi_atlas is None):
        # print('\n\n\n\n')
        # print('No flexi-atlas2')
        # print('\n\n\n\n')
        flexi_atlas = False
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(niu.IdentityInterface(fields=['atlas', 'uatlas', 'clustering']),
                                     name='flexi_atlas_source')
        flexi_atlas_source.synchronize = True
        if multi_atlas is not None and user_atlas_list is not None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: multiple nilearn atlases + multiple user atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas", len(user_atlas_list) * [None] + multi_atlas),
                                            ("uatlas", user_atlas_list + len(multi_atlas) * [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif multi_atlas is not None and uatlas is not None and user_atlas_list is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single user atlas + multiple nilearn atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas", multi_atlas + [None]),
                                            ("uatlas", len(multi_atlas) * [None] + [uatlas])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and user_atlas_list is not None and multi_atlas is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + multiple user atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas", len(user_atlas_list) * [None] + [atlas]),
                                            ("uatlas", user_atlas_list + [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and uatlas is not None and user_atlas_list is None and multi_atlas is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + single user atlas')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas", [atlas, None]),
                                            ("uatlas", [None, uatlas])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables

    # Connect flexi_atlas inputs to definition node
    if flexi_atlas is True:
        fmri_connectometry_wf.add_nodes([flexi_atlas_source])
        if float(k_clustering) > 0:
            fmri_connectometry_wf.disconnect([(clustering_node, fetch_nodes_and_labels_node,
                                               [('uatlas', 'uatlas'),
                                                ('atlas', 'atlas'),
                                                ('clustering', 'clustering')])
                                              ])
            if float(k_clustering == 1):
                fmri_connectometry_wf.connect([(clustering_node, flexi_atlas_source,
                                                [('clustering', 'clustering')])
                                               ])
            else:
                clust_join_node = pe.JoinNode(niu.IdentityInterface(fields=['clustering', 'k', 'clust_mask',
                                                                            'clust_type']),
                                              name='clust_join_node',
                                              joinsource=clustering_info_node,
                                              joinfield=['k', 'clust_mask', 'clust_type'])
                fmri_connectometry_wf.connect([(clustering_node, clust_join_node,
                                                [('clustering', 'clustering'),
                                                 ('k', 'k'),
                                                 ('clust_mask', 'clust_mask'),
                                                 ('clust_type', 'clust_type')])
                                               ])
                fmri_connectometry_wf.connect([(clust_join_node, flexi_atlas_source,
                                                [('clustering', 'clustering')])
                                               ])
            fmri_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                            [('uatlas', 'uatlas'),
                                             ('atlas', 'atlas'),
                                             ('clustering', 'clustering')])
                                           ])
        else:
            fmri_connectometry_wf.disconnect([(inputnode, fetch_nodes_and_labels_node,
                                               [('uatlas', 'uatlas'),
                                                ('atlas', 'atlas'),
                                                ('k_clustering', 'clustering')])
                                              ])
            fmri_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                            [('uatlas', 'uatlas'),
                                             ('atlas', 'atlas')])
                                           ])

    # Generate nodes
    if roi is not None:
        # Masking case
        node_gen_node = pe.Node(niu.Function(input_names=['roi', 'coords', 'parcel_list', 'labels', 'dir_path',
                                                          'ID', 'parc', 'atlas', 'uatlas'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'labels',
                                                           'atlas', 'uatlas', 'dir_path'],
                                             function=nodemaker.node_gen_masking, imports=import_list),
                                name="node_gen_node")

    else:
        # Non-masking case
        node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'labels', 'dir_path',
                                                          'ID', 'parc', 'atlas', 'uatlas'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'labels',
                                                           'atlas', 'uatlas', 'dir_path'],
                                             function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    # Extract time-series from nodes
    extract_ts_iterables = []
    extract_ts_node = pe.Node(ExtractTimeseries(),
                              input_names=['conf', 'func_file', 'coords', 'dir_path', 'ID', 'roi',
                                           'network', 'smooth', 'atlas', 'uatlas', 'labels', 'c_boot',
                                           'block_size', 'hpass', 'mask', 'parc', 'node_size',
                                           'net_parcels_nii_path'],
                              output_names=['ts_within_nodes', 'node_size', 'smooth', 'dir_path', 'atlas', 'uatlas',
                                            'labels', 'coords', 'c_boot', 'hpass'], imports=import_list,
                              name="extract_ts_node")

    extract_ts_node.interface.n_procs = runtime_dict['extract_ts_node'][0]
    extract_ts_node.interface.mem_gb = runtime_dict['extract_ts_node'][1]
    extract_ts_node._n_procs = runtime_dict['extract_ts_node'][0]
    extract_ts_node._mem_gb = runtime_dict['extract_ts_node'][1]

    if parc is True:
        # Parcels case
        extract_ts_node.inputs.parc = True
        extract_ts_node.inputs.node_size = None
        save_nifti_parcels_node = pe.Node(niu.Function(input_names=['ID', 'dir_path', 'roi', 'network',
                                                                    'net_parcels_map_nifti'],
                                                       output_names=['net_parcels_nii_path'],
                                                       function=utils.save_nifti_parcels_map, imports=import_list),
                                          name="save_nifti_parcels_node")
        fmri_connectometry_wf.add_nodes([save_nifti_parcels_node])
        fmri_connectometry_wf.connect([(inputnode, save_nifti_parcels_node,
                                        [('roi', 'roi')]),
                                       # network supposed to be here?
                                       (inputnode, save_nifti_parcels_node, [('ID', 'ID'),
                                                                             ('network', 'network')]),
                                       (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                        [('dir_path', 'dir_path')]),
                                       (node_gen_node, save_nifti_parcels_node,
                                        [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
                                       (save_nifti_parcels_node, extract_ts_node,
                                        [('net_parcels_nii_path', 'net_parcels_nii_path')])
                                       ])
    else:
        # Coordinate case
        extract_ts_node.inputs.parc = False
        extract_ts_node.inputs.net_parcels_nii_path = None

        # Set extract_ts iterables
        if node_size_list:
            extract_ts_iterables.append(("node_size", node_size_list))
            extract_ts_node.iterables = extract_ts_iterables
        else:
            fmri_connectometry_wf.connect([(inputnode, extract_ts_node, [('node_size', 'node_size')])])

    if smooth_list:
        extract_ts_iterables.append(("smooth", smooth_list))
        extract_ts_node.iterables = extract_ts_iterables
    else:
        fmri_connectometry_wf.connect([(inputnode, extract_ts_node, [('smooth', 'smooth')])])

    if hpass_list:
        extract_ts_iterables.append(("hpass", hpass_list))
        extract_ts_node.iterables = extract_ts_iterables
    else:
        fmri_connectometry_wf.connect([(inputnode, extract_ts_node, [('hpass', 'hpass')])])

    extract_ts_node.synchronize = True

    # Connectivity matrix model fit
    get_conn_matrix_node = pe.Node(niu.Function(input_names=['time_series', 'conn_model', 'dir_path', 'node_size',
                                                             'smooth', 'dens_thresh', 'network', 'ID', 'roi',
                                                             'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                             'atlas', 'uatlas', 'labels', 'coords',
                                                             'c_boot', 'norm', 'binary', 'hpass'],
                                                output_names=['conn_matrix', 'conn_model', 'dir_path', 'node_size',
                                                              'smooth', 'dens_thresh', 'network', 'ID', 'roi',
                                                              'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                              'atlas', 'uatlas', 'labels', 'coords',
                                                              'c_boot', 'norm', 'binary', 'hpass'],
                                                function=estimation.get_conn_matrix, imports=import_list),
                                   name="get_conn_matrix_node")

    # Set get_conn_matrix_node iterables
    if conn_model_list:
        get_conn_matrix_node.iterables = [("conn_model", conn_model_list)]
    else:
        fmri_connectometry_wf.connect([(inputnode, get_conn_matrix_node, [('conn_model', 'conn_model')])])

    get_conn_matrix_node.synchronize = True

    # RSN case
    if network or multi_nets:
        get_node_membership_node = pe.Node(niu.Function(input_names=['network', 'infile', 'coords', 'labels',
                                                                     'parc', 'parcel_list', 'perc_overlap', 'error'],
                                                        output_names=['net_coords', 'net_parcel_list', 'net_labels',
                                                                      'network'],
                                                        function=nodemaker.get_node_membership, imports=import_list),
                                           name="get_node_membership_node")
        save_coords_and_labels_node = pe.Node(niu.Function(input_names=['coords', 'labels', 'dir_path', 'network'],
                                                           function=utils.save_RSN_coords_and_labels_to_pickle,
                                                           imports=import_list), name="save_coords_and_labels_node")
        if multi_nets:
            get_node_membership_node.iterables = [("network", multi_nets)]

        fmri_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                              ('template', 'infile'),
                                                                              ('parc', 'parc')]),
                                       (fetch_nodes_and_labels_node, get_node_membership_node,
                                        [('coords', 'coords'), ('labels', 'labels'),
                                         ('parcel_list', 'parcel_list'), ('par_max', 'par_max'),
                                         ('networks_list', 'networks_list')]),
                                       (get_node_membership_node, node_gen_node,
                                        [('net_coords', 'coords'), ('net_labels', 'labels'),
                                         ('net_parcel_list', 'parcel_list')]),
                                       (get_node_membership_node, save_coords_and_labels_node,
                                        [('net_coords', 'coords'), ('net_labels', 'labels'),
                                         ('network', 'network')]),
                                       (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                        [('dir_path', 'dir_path')]),
                                       (get_node_membership_node, extract_ts_node,
                                        [('network', 'network')]),
                                       (get_node_membership_node, get_conn_matrix_node,
                                        [('network', 'network')])
                                       ])
    else:
        fmri_connectometry_wf.connect([(fetch_nodes_and_labels_node, node_gen_node,
                                        [('coords', 'coords'), ('labels', 'labels'),
                                         ('parcel_list', 'parcel_list')]),
                                       (inputnode, extract_ts_node,
                                        [('network', 'network')]),
                                       (inputnode, get_conn_matrix_node,
                                        [('network', 'network')])
                                       ])

    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout joins
    map_fields = ['conn_model', 'dir_path', 'conn_matrix', 'node_size', 'smooth', 'dens_thresh', 'network', 'ID',
                  'roi', 'min_span_tree', 'disp_filt', 'parc', 'prune', 'thr', 'atlas', 'uatlas',
                  'labels', 'coords', 'c_boot', 'norm', 'binary', 'hpass']

    map_connects = [('conn_model', 'conn_model'), ('dir_path', 'dir_path'), ('conn_matrix', 'conn_matrix'),
                    ('node_size', 'node_size'), ('smooth', 'smooth'), ('dens_thresh', 'dens_thresh'), ('ID', 'ID'),
                    ('roi', 'roi'), ('min_span_tree', 'min_span_tree'), ('disp_filt', 'disp_filt'), ('parc', 'parc'),
                    ('prune', 'prune'), ('network', 'network'), ('thr', 'thr'), ('atlas', 'atlas'),
                    ('uatlas', 'uatlas'), ('labels', 'labels'), ('coords', 'coords'),
                    ('c_boot', 'c_boot'), ('norm', 'norm'), ('binary', 'binary'), ('hpass', 'hpass')]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='thr_info_node')
    # Joinsource logic for atlas varieties
    if user_atlas_list or multi_atlas or float(k_clustering) > 0 or flexi_atlas is True:
        if flexi_atlas is True:
            atlas_join_source = flexi_atlas_source
        elif float(k_clustering) > 1 and flexi_atlas is False:
            atlas_join_source = clustering_info_node
        else:
            atlas_join_source = fetch_nodes_and_labels_node
    else:
        atlas_join_source = None

    # Connect all get_conn_matrix_node outputs to the "thr_info" node
    fmri_connectometry_wf.connect([(get_conn_matrix_node, thr_info_node,
                                    [x for x in map_connects if x != ('thr', 'thr')])])
    # Begin joinnode chaining logic
    if conn_model_list or node_size_list or smooth_list or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True or multi_thr is True or hpass is not None:
        join_iters_node_thr = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_node_thr',
                                          joinsource=thr_info_node,
                                          joinfield=map_fields)
        join_iters_node_atlas = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                            name='join_iters_node_atlas',
                                            joinsource=atlas_join_source,
                                            joinfield=map_fields)
        if not conn_model_list and (node_size_list or smooth_list or hpass_list):
            # print('Time-series node extraction iterables...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_extract_ts_node',
                                          joinsource=extract_ts_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list and (not node_size_list and not smooth_list and not hpass_list):
            # print('Multiple connectivity models...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_get_conn_matrix_node',
                                          joinsource=get_conn_matrix_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif not conn_model_list and not node_size_list and not smooth_list and not hpass_list:
            # print('No connectivity model or time-series node extraction iterables...')
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                  name='join_iters_iteratlas_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                  name='join_iters_thr_info_node',
                                                  joinsource=thr_info_node, joinfield=map_fields)
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                  name='join_iters_iteratlas_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list and (node_size_list or smooth_list or hpass_list):
            # print('Connectivity model and time-series node extraction iterables...')
            join_iters_node_ext_ts = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                 name='join_iters_node_ext_ts', joinsource=extract_ts_node,
                                                 joinfield=map_fields)
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_get_conn_matrix_node',
                                          joinsource=get_conn_matrix_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_ext_ts, map_connects),
                                                   (join_iters_node_ext_ts, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects),
                                                   (join_iters_node_thr, join_iters_node_ext_ts, map_connects),
                                                   (join_iters_node_ext_ts, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_ext_ts, map_connects),
                                                   (join_iters_node_ext_ts, join_iters_node_atlas, map_connects),
                                                   (join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    fmri_connectometry_wf.connect([(thr_info_node, join_iters_node_ext_ts, map_connects),
                                                   (join_iters_node_ext_ts, join_iters_node, map_connects)])
        else:
            raise RuntimeError('\nERROR: Unknown join context.')

        no_iters = False
    else:
        # Minimal case of no iterables
        print('\nNo iterables...\n')
        join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
        fmri_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        no_iters = True

    # Create final thresh_func node that performs the thresholding
    if no_iters is True:
        thresh_func_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                             'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                             'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                             'atlas', 'uatlas', 'labels', 'coords',
                                                             'c_boot', 'norm', 'binary', 'hpass'],
                                                output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                              'node_size', 'network', 'conn_model', 'roi', 'smooth',
                                                              'prune', 'ID', 'dir_path', 'atlas',
                                                              'uatlas', 'labels', 'coords', 'c_boot',
                                                              'norm', 'binary', 'hpass'],
                                                function=thresholding.thresh_func, imports=import_list),
                                   name="thresh_func_node")
    else:
        thresh_func_node = pe.MapNode(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                                'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                                'atlas', 'uatlas', 'labels',
                                                                'coords', 'c_boot', 'norm', 'binary', 'hpass'],
                                                   output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                                 'node_size', 'network', 'conn_model', 'roi', 'smooth',
                                                                 'prune', 'ID', 'dir_path', 'atlas',
                                                                 'uatlas', 'labels', 'coords', 'c_boot',
                                                                 'norm', 'binary', 'hpass'],
                                                   function=thresholding.thresh_func, imports=import_list),
                                      name="thresh_func_node", iterfield=['dens_thresh', 'thr', 'conn_matrix',
                                                                          'conn_model', 'network', 'ID', 'dir_path',
                                                                          'roi', 'node_size', 'min_span_tree', 'smooth',
                                                                          'disp_filt', 'parc', 'prune', 'atlas',
                                                                          'uatlas', 'labels', 'coords',
                                                                          'c_boot', 'norm', 'binary', 'hpass'],
                                      nested=True)

        thresh_func_node.synchronize = True

    # Set iterables for thr on thresh_func, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr), float(max_thr),
                                                                          float(step_thr)), decimals=2).tolist()] +
                                      [str(float(max_thr))])))
        thr_info_node.iterables = ("thr", iter_thresh)
    else:
        thr_info_node.iterables = ("thr", [thr])

    thr_info_node.synchronize = True

    # Plotting
    if plot_switch is True:
        plot_fields = ['conn_matrix', 'conn_model', 'atlas', 'dir_path', 'ID', 'network', 'labels', 'roi',
                       'coords', 'thr', 'node_size', 'edge_threshold', 'smooth', 'prune', 'uatlas', 'c_boot',
                       'norm', 'binary', 'hpass']
        # Plotting iterable graph solutions
        if conn_model_list or node_size_list or smooth_list or multi_thr or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True or hpass_list:
            plot_all_node = pe.MapNode(niu.Function(input_names=plot_fields, output_names='None',
                                                    function=plot_gen.plot_all_func, imports=import_list),
                                       itersource=thr_info_node,
                                       iterfield=plot_fields,
                                       name="plot_all_node", nested=True)
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(niu.Function(input_names=plot_fields, output_names='None',
                                                 function=plot_gen.plot_all_func, imports=import_list),
                                    name="plot_all_node")

        # Connect thresh_func_node outputs to plotting node
        fmri_connectometry_wf.connect([(thresh_func_node, plot_all_node, [('ID', 'ID'),
                                                                          ('roi', 'roi'),
                                                                          ('network', 'network'),
                                                                          ('prune', 'prune'),
                                                                          ('node_size', 'node_size'),
                                                                          ('smooth', 'smooth'),
                                                                          ('dir_path', 'dir_path'),
                                                                          ('conn_matrix_thr', 'conn_matrix'),
                                                                          ('edge_threshold', 'edge_threshold'),
                                                                          ('thr', 'thr'),
                                                                          ('conn_model', 'conn_model'),
                                                                          ('atlas', 'atlas'),
                                                                          ('uatlas', 'uatlas'),
                                                                          ('labels', 'labels'),
                                                                          ('coords', 'coords'),
                                                                          ('c_boot', 'c_boot'),
                                                                          ('norm', 'norm'),
                                                                          ('binary', 'binary'),
                                                                          ('hpass', 'hpass')])
                                       ])
    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                       'conn_model', 'norm', 'binary']),
                         name='outputnode')

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune',
                                                                         'ID', 'roi', 'conn_model', 'node_size',
                                                                         'smooth', 'c_boot', 'norm', 'binary',
                                                                         'hpass']),
                                           name='join_iters_node_nets', joinsource=get_node_membership_node,
                                           joinfield=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                      'conn_model', 'node_size', 'smooth', 'c_boot', 'norm',
                                                      'binary', 'hpass'])
        fmri_connectometry_wf.connect([
            (thresh_func_node, join_iters_node_nets, [('thr', 'thr'), ('network', 'network'),
                                                      ('est_path', 'est_path'), ('node_size', 'node_size'),
                                                      ('smooth', 'smooth'), ('roi', 'roi'),
                                                      ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                      ('prune', 'prune'), ('c_boot', 'c_boot'),
                                                      ('norm', 'norm'), ('binary', 'binary'), ('hpass', 'hpass')]),
            (join_iters_node_nets, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                                ('roi', 'roi'), ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')])
        ])
    else:
        fmri_connectometry_wf.connect([
            (thresh_func_node, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                            ('roi', 'roi'), ('conn_model', 'conn_model'), ('ID', 'ID'),
                                            ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')])
        ])

    # Handle masking scenarios (brain mask and/or roi)
    if (mask is not None) and (roi is None):
        check_orient_and_dims_mask_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'],
                                                               output_names=['outfile'],
                                                               function=regutils.check_orient_and_dims,
                                                               imports=import_list),
                                                  name="check_orient_and_dims_mask_node")
        fmri_connectometry_wf.connect([
            (inputnode, extract_ts_node, [('roi', 'roi')]),
            (inputnode, check_orient_and_dims_mask_node, [('mask', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_mask_node, extract_ts_node, [('outfile', 'mask')]),
            (check_orient_and_dims_mask_node, node_gen_node, [('outfile', 'roi')]),
            (check_orient_and_dims_mask_node, get_conn_matrix_node, [('outfile', 'roi')]),
        ])
        if k_clustering > 0:
            fmri_connectometry_wf.connect([
                (check_orient_and_dims_mask_node, clustering_node, [('outfile', 'mask')]),
            ])
        else:
            fmri_connectometry_wf.connect([
                (inputnode, clustering_node, [('mask', 'mask')]),
            ])
    elif (op.isfile(template_mask) is True) and (roi is None):
        fmri_connectometry_wf.connect([
            (inputnode, node_gen_node, [('template_mask', 'roi')]),
            (inputnode, get_conn_matrix_node, [('template_mask', 'roi')]),
            (inputnode, extract_ts_node, [('mask', 'mask'), ('roi', 'roi')])
        ])
    else:
        fmri_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
            (inputnode, get_conn_matrix_node, [('roi', 'roi')]),
            (inputnode, extract_ts_node, [('mask', 'mask'), ('roi', 'roi')])
        ])

    # Connect remaining nodes of workflow
    fmri_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('parc', 'parc'), ('ref_txt', 'ref_txt'),
                                                  ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, check_orient_and_dims_func_node, [('func_file', 'infile'),
                                                      ('vox_size', 'vox_size')]),
        (check_orient_and_dims_func_node, extract_ts_node, [('outfile', 'func_file')]),
        (check_orient_and_dims_func_node, fetch_nodes_and_labels_node, [('outfile', 'in_file')]),
        (inputnode, node_gen_node, [('ID', 'ID'), ('parc', 'parc')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('atlas', 'atlas'), ('uatlas', 'uatlas'),
                                                      ('dir_path', 'dir_path'), ('par_max', 'par_max')]),
        (inputnode, extract_ts_node, [('conf', 'conf'), ('ID', 'ID'), ('block_size', 'block_size'),
                                      ('c_boot', 'c_boot')]),
        (inputnode, get_conn_matrix_node, [('dens_thresh', 'dens_thresh'),
                                           ('ID', 'ID'),
                                           ('min_span_tree', 'min_span_tree'),
                                           ('disp_filt', 'disp_filt'),
                                           ('parc', 'parc'),
                                           ('prune', 'prune'),
                                           ('norm', 'norm'),
                                           ('binary', 'binary')]),
        (node_gen_node, extract_ts_node, [('coords', 'coords'), ('labels', 'labels'),
                                          ('atlas', 'atlas'), ('uatlas', 'uatlas'),
                                          ('dir_path', 'dir_path')]),
        (extract_ts_node, get_conn_matrix_node, [('ts_within_nodes', 'time_series'), ('dir_path', 'dir_path'),
                                                 ('node_size', 'node_size'), ('smooth', 'smooth'),
                                                 ('coords', 'coords'), ('labels', 'labels'),
                                                 ('atlas', 'atlas'), ('uatlas', 'uatlas'),
                                                 ('c_boot', 'c_boot'), ('hpass', 'hpass')]),
        (join_iters_node, thresh_func_node, map_connects)
    ])

    # Handle case that t1w image is available to refine parcellation
    if anat_file and parc is True:
        fmri_connectometry_wf.disconnect([
            (node_gen_node, extract_ts_node, [('uatlas', 'uatlas')]),
        ])
        # Check orientation and resolution
        check_orient_and_dims_uatlas_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'],
                                                                 output_names=['outfile'],
                                                                 function=regutils.check_orient_and_dims,
                                                                 imports=import_list),
                                                    name="check_orient_and_dims_uatlas_node")
        fmri_connectometry_wf.add_nodes([register_atlas_node])
        fmri_connectometry_wf.connect([
            (inputnode, check_orient_and_dims_anat_node, [('anat_file', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_anat_node, register_node, [('outfile', 'anat_file')]),
            (check_orient_and_dims_anat_node, register_atlas_node, [('outfile', 'anat_file')]),
            (inputnode, register_node, [('basedir_path', 'basedir_path'), ('vox_size', 'vox_size')]),
            (inputnode, register_atlas_node, [('basedir_path', 'basedir_path'),
                                              ('vox_size', 'vox_size')]),
            (register_node, register_atlas_node, [('reg_fmri_complete', 'reg_fmri_complete')]),
            (inputnode, check_orient_and_dims_uatlas_node, [('vox_size', 'vox_size')]),
            (fetch_nodes_and_labels_node, check_orient_and_dims_uatlas_node, [('uatlas', 'infile')]),
            (check_orient_and_dims_uatlas_node, register_atlas_node, [('outfile', 'uatlas')]),
            (node_gen_node, register_atlas_node, [('atlas', 'atlas'), ('uatlas', 'uatlas_parcels')]),
            (register_atlas_node, extract_ts_node, [('aligned_atlas_t1mni_gm', 'uatlas')]),
        ])

    # Set cpu/memory reqs
    for node_name in fmri_connectometry_wf.list_node_names():
        if node_name in runtime_dict:
            fmri_connectometry_wf.get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
            fmri_connectometry_wf.get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
            fmri_connectometry_wf.get_node(node_name).n_procs = runtime_dict[node_name][0]
            fmri_connectometry_wf.get_node(node_name)._mem_gb = runtime_dict[node_name][1]

    # Set runtime/logging configurations
    execution_dict['plugin_args'] = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                                     'scheduler': 'mem_thread'}
    execution_dict['plugin'] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            fmri_connectometry_wf.config[key][setting] = value

    return fmri_connectometry_wf
