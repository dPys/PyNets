# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def workflow_selector(func_file, ID, atlas_select, network, node_size, roi, thr, uatlas_select, multi_nets,
                      conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_file, anat_file, parc,
                      ref_txt, procmem, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k,
                      clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                      node_size_list, num_total_samples, conn_model_list, min_span_tree, verbose, plugin_type,
                      use_AAL_naming, smooth, smooth_list, disp_filt, clust_type, clust_type_list, c_boot, block_size,
                      mask, norm, binary, fbval, fbvec, target_samples, curv_thr_list, step_list, overlap_thr,
                      overlap_thr_list, track_type, max_length, maxcrossing, life_run, min_length, directget,
                      tiss_class, runtime_dict):
    from pynets import workflows
    from nipype import Workflow
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.utils import pass_meta_ins, pass_meta_outs


    # Workflow 1: Functional connectome
    if func_file is not None:
        sub_func_wf = workflows.functional_connectometry(func_file, ID, atlas_select, network, node_size,
                                                         roi, thr, uatlas_select, conn_model, dens_thresh, conf,
                                                         plot_switch, parc, ref_txt, procmem,
                                                         multi_thr, multi_atlas, max_thr, min_thr, step_thr,
                                                         k, clust_mask, k_min, k_max, k_step, k_clustering,
                                                         user_atlas_list, clust_mask_list, node_size_list,
                                                         conn_model_list, min_span_tree, use_AAL_naming, smooth,
                                                         smooth_list, disp_filt, prune, multi_nets, clust_type,
                                                         clust_type_list, plugin_type, c_boot, block_size, mask,
                                                         norm, binary, runtime_dict, anat_file)
        if dwi_file is None:
            sub_struct_wf = None
    # Workflow 2: Structural connectome
    if dwi_file is not None:
        sub_struct_wf = workflows.structural_connectometry(ID, atlas_select, network, node_size, roi,
                                                           uatlas_select, plot_switch, parc, ref_txt, procmem,
                                                           dwi_file, fbval, fbvec, anat_file, thr, dens_thresh,
                                                           conn_model, user_atlas_list, multi_thr, multi_atlas,
                                                           max_thr, min_thr, step_thr, node_size_list,
                                                           conn_model_list, min_span_tree, use_AAL_naming, disp_filt,
                                                           plugin_type, multi_nets, prune, mask, norm, binary,
                                                           target_samples, curv_thr_list, step_list, overlap_thr,
                                                           overlap_thr_list, track_type, max_length, maxcrossing,
                                                           life_run, min_length, directget, tiss_class, runtime_dict)
        if func_file is None:
            sub_func_wf = None

    # Create meta-workflow to organize graph simulation sets in prep for analysis
    base_dirname = "%s%s" % ('Meta_wf_', ID)
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
    cfg = dict(execution={'stop_on_first_crash': False, 'crashfile_format': 'txt', 'parameterize_dirs': True,
                          'display_variable': ':0', 'job_finished_timeout': 120, 'matplotlib_backend': 'Agg',
                          'plugin': str(plugin_type), 'use_relative_paths': True, 'remove_unnecessary_outputs': False,
                          'remove_node_directories': False})
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            meta_wf.config[key][setting] = value
    # Create input/output nodes
    meta_inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'network', 'thr',
                                                           'node_size', 'roi', 'uatlas_select', 'multi_nets',
                                                           'conn_model', 'dens_thresh', 'conf', 'adapt_thresh',
                                                           'plot_switch', 'dwi_file', 'anat_file', 'parc', 'ref_txt',
                                                           'procmem', 'multi_thr', 'multi_atlas', 'max_thr',
                                                           'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max',
                                                           'k_step', 'k_clustering', 'user_atlas_list',
                                                           'clust_mask_list', 'prune', 'node_size_list',
                                                           'num_total_samples', 'conn_model_list',
                                                           'min_span_tree', 'verbose', 'plugin_type', 'use_AAL_naming',
                                                           'smooth', 'smooth_list', 'disp_filt', 'clust_type',
                                                           'clust_type_list', 'c_boot', 'block_size', 'mask', 'norm',
                                                           'binary', 'fbval', 'fbvec', 'target_samples',
                                                           'curv_thr_list', 'step_list', 'overlap_thr',
                                                           'overlap_thr_list', 'track_type', 'max_length',
                                                           'maxcrossing', 'life_run', 'min_length', 'directget',
                                                           'tiss_class']), name='meta_inputnode')

    meta_inputnode.inputs.in_file = func_file
    meta_inputnode.inputs.ID = ID
    meta_inputnode.inputs.atlas_select = atlas_select
    meta_inputnode.inputs.network = network
    meta_inputnode.inputs.thr = thr
    meta_inputnode.inputs.node_size = node_size
    meta_inputnode.inputs.roi = roi
    meta_inputnode.inputs.uatlas_select = uatlas_select
    meta_inputnode.inputs.multi_nets = multi_nets
    meta_inputnode.inputs.conn_model = conn_model
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
    meta_inputnode.inputs.conn_model_list = conn_model_list
    meta_inputnode.inputs.min_span_tree = min_span_tree
    meta_inputnode.inputs.verbose = verbose
    meta_inputnode.inputs.plugin_type = plugin_type
    meta_inputnode.inputs.use_AAL_naming = use_AAL_naming
    meta_inputnode.inputs.smooth = smooth
    meta_inputnode.inputs.smooth_list = smooth_list
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
    meta_inputnode.inputs.life_run = life_run
    meta_inputnode.inputs.min_length = min_length
    meta_inputnode.inputs.directget = directget
    meta_inputnode.inputs.tiss_class = tiss_class

    if func_file:
        pass_meta_ins_func_node = pe.Node(niu.Function(input_names=['conn_model', 'est_path', 'network', 'node_size',
                                                                    'thr', 'prune', 'ID', 'roi', 'norm', 'binary'],
                                                       output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                                     'network_iterlist', 'node_size_iterlist',
                                                                     'thr_iterlist', 'prune_iterlist', 'ID_iterlist',
                                                                     'roi_iterlist', 'norm_iterlist', 'binary_iterlist'],
                                                       function=pass_meta_ins), name='pass_meta_ins_func_node')

        meta_wf.add_nodes([sub_func_wf])
        meta_wf.connect([(meta_inputnode, sub_func_wf, [('ID', 'inputnode.ID'),
                                                        ('anat_file', 'inputnode.anat_file'),
                                                        ('atlas_select', 'inputnode.atlas_select'),
                                                        ('network', 'inputnode.network'),
                                                        ('thr', 'inputnode.thr'),
                                                        ('node_size', 'inputnode.node_size'),
                                                        ('roi', 'inputnode.roi'),
                                                        ('uatlas_select', 'inputnode.uatlas_select'),
                                                        ('multi_nets', 'inputnode.multi_nets'),
                                                        ('conn_model', 'inputnode.conn_model'),
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
                                                        ('conn_model_list', 'inputnode.conn_model_list'),
                                                        ('min_span_tree', 'inputnode.min_span_tree'),
                                                        ('use_AAL_naming', 'inputnode.use_AAL_naming'),
                                                        ('smooth', 'inputnode.smooth'),
                                                        ('disp_filt', 'inputnode.disp_filt'),
                                                        ('clust_type', 'inputnode.clust_type'),
                                                        ('clust_type_list', 'inputnode.clust_type_list'),
                                                        ('c_boot', 'inputnode.c_boot'),
                                                        ('block_size', 'inputnode.block_size'),
                                                        ('mask', 'inputnode.mask'),
                                                        ('norm', 'inputnode.norm'),
                                                        ('binary', 'inputnode.binary')])
                         ])

        # Connect outputs of nested workflow to parent wf
        meta_wf.connect([(sub_func_wf.get_node('outputnode'), pass_meta_ins_func_node, [('conn_model', 'conn_model'),
                                                                                        ('est_path', 'est_path'),
                                                                                        ('network', 'network'),
                                                                                        ('node_size', 'node_size'),
                                                                                        ('thr', 'thr'),
                                                                                        ('prune', 'prune'),
                                                                                        ('ID', 'ID'),
                                                                                        ('roi', 'roi'),
                                                                                        ('norm', 'norm'),
                                                                                        ('binary', 'binary')])
                     ])

    if dwi_file:
        pass_meta_ins_struct_node = pe.Node(niu.Function(input_names=['conn_model', 'est_path', 'network', 'node_size',
                                                                      'thr', 'prune', 'ID', 'roi', 'norm', 'binary'],
                                                         output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                                       'network_iterlist', 'node_size_iterlist',
                                                                       'thr_iterlist', 'prune_iterlist', 'ID_iterlist',
                                                                       'roi_iterlist', 'norm_iterlist',
                                                                       'binary_iterlist'], function=pass_meta_ins),
                                            name='pass_meta_ins_struct_node')

        meta_wf.add_nodes([sub_struct_wf])
        meta_wf.connect([(meta_inputnode, sub_struct_wf, [('ID', 'inputnode.ID'),
                                                          ('dwi_file', 'inputnode.dwi_file'),
                                                          ('fbval', 'inputnode.fbval'),
                                                          ('fbvec', 'inputnode.fbvec'),
                                                          ('anat_file', 'inputnode.anat_file'),
                                                          ('atlas_select', 'inputnode.atlas_select'),
                                                          ('network', 'inputnode.network'),
                                                          ('thr', 'inputnode.thr'),
                                                          ('node_size', 'inputnode.node_size'),
                                                          ('roi', 'inputnode.roi'),
                                                          ('uatlas_select', 'inputnode.uatlas_select'),
                                                          ('multi_nets', 'inputnode.multi_nets'),
                                                          ('conn_model', 'inputnode.conn_model'),
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
                                                          ('conn_model_list', 'inputnode.conn_model_list'),
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
                                                          ('life_run', 'inputnode.life_run'),
                                                          ('min_length', 'inputnode.min_length'),
                                                          ('directget', 'inputnode.directget'),
                                                          ('tiss_class', 'inputnode.tiss_class')
                                                          ])
                         ])

        # Connect outputs of nested workflow to parent wf
        meta_wf.connect([(sub_struct_wf.get_node('outputnode'), pass_meta_ins_struct_node, [('conn_model', 'conn_model'),
                                                                                            ('est_path', 'est_path'),
                                                                                            ('network', 'network'),
                                                                                            ('node_size', 'node_size'),
                                                                                            ('thr', 'thr'),
                                                                                            ('prune', 'prune'),
                                                                                            ('ID', 'ID'),
                                                                                            ('roi', 'roi'),
                                                                                            ('norm', 'norm'),
                                                                                            ('binary', 'binary')])
                         ])

    pass_meta_outs_node = pe.Node(niu.Function(input_names=['conn_model_iterlist', 'est_path_iterlist',
                                                            'network_iterlist', 'node_size_iterlist',
                                                            'thr_iterlist', 'prune_iterlist', 'ID_iterlist',
                                                            'roi_iterlist', 'norm_iterlist', 'binary_iterlist'],
                                               output_names=['conn_model_iterlist', 'est_path_iterlist',
                                                             'network_iterlist', 'node_size_iterlist',
                                                             'thr_iterlist', 'prune_iterlist', 'ID_iterlist',
                                                             'roi_iterlist', 'norm_iterlist', 'binary_iterlist'],
                                               function=pass_meta_outs), name='pass_meta_outs_node')

    if (func_file and not dwi_file) or (dwi_file and not func_file):
        if func_file and not dwi_file:
            meta_wf.connect([(pass_meta_ins_func_node, pass_meta_outs_node, [('conn_model_iterlist', 'conn_model_iterlist'),
                                                                            ('est_path_iterlist', 'est_path_iterlist'),
                                                                            ('network_iterlist', 'network_iterlist'),
                                                                            ('node_size_iterlist', 'node_size_iterlist'),
                                                                            ('thr_iterlist', 'thr_iterlist'),
                                                                            ('prune_iterlist', 'prune_iterlist'),
                                                                            ('ID_iterlist', 'ID_iterlist'),
                                                                            ('roi_iterlist', 'roi_iterlist'),
                                                                            ('norm_iterlist', 'norm_iterlist'),
                                                                            ('binary_iterlist', 'binary_iterlist')])
                             ])
        elif dwi_file and not func_file:
            meta_wf.connect([(pass_meta_ins_struct_node, pass_meta_outs_node, [('conn_model_iterlist', 'conn_model_iterlist'),
                                                                            ('est_path_iterlist', 'est_path_iterlist'),
                                                                            ('network_iterlist', 'network_iterlist'),
                                                                            ('node_size_iterlist', 'node_size_iterlist'),
                                                                            ('thr_iterlist', 'thr_iterlist'),
                                                                            ('prune_iterlist', 'prune_iterlist'),
                                                                            ('ID_iterlist', 'ID_iterlist'),
                                                                            ('roi_iterlist', 'roi_iterlist'),
                                                                            ('norm_iterlist', 'norm_iterlist'),
                                                                            ('binary_iterlist', 'binary_iterlist')])
                             ])
    elif func_file and dwi_file:
        meta_wf.connect([(pass_meta_ins_struct_node, pass_meta_outs_node, [('conn_model_iterlist', 'conn_model_iterlist'),
                                                                           ('est_path_iterlist', 'est_path_iterlist'),
                                                                           ('network_iterlist', 'network_iterlist'),
                                                                           ('node_size_iterlist', 'node_size_iterlist'),
                                                                           ('thr_iterlist', 'thr_iterlist'),
                                                                           ('prune_iterlist', 'prune_iterlist'),
                                                                           ('ID_iterlist', 'ID_iterlist'),
                                                                           ('roi_iterlist', 'roi_iterlist'),
                                                                           ('norm_iterlist', 'norm_iterlist'),
                                                                           ('binary_iterlist', 'binary_iterlist')]),
                         (pass_meta_ins_func_node, pass_meta_outs_node, [('conn_model_iterlist', 'conn_model_iterlist'),
                                                                         ('est_path_iterlist', 'est_path_iterlist'),
                                                                         ('network_iterlist', 'network_iterlist'),
                                                                         ('node_size_iterlist', 'node_size_iterlist'),
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
        wf_selected = "%s%s" % ('functional_connectometry_', ID)
        for node_name in sub_func_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._mem_gb = runtime_dict[node_name][1]
        if k_clustering > 0:
            meta_wf.get_node("%s%s" % (wf_selected, '.clustering_node'))._n_procs = 1
            meta_wf.get_node("%s%s" % (wf_selected, '.clustering_node'))._mem_gb = 4

    if dwi_file:
        wf_selected = "%s%s" % ('structural_connectometry_', ID)
        for node_name in sub_struct_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node("%s%s%s" % (wf_selected, '.', node_name))._mem_gb = runtime_dict[node_name][1]

    return meta_wf


def functional_connectometry(func_file, ID, atlas_select, network, node_size, roi, thr, uatlas_select, conn_model,
                             dens_thresh, conf, plot_switch, parc, ref_txt, procmem, multi_thr,
                             multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step,
                             k_clustering, user_atlas_list, clust_mask_list, node_size_list, conn_model_list,
                             min_span_tree, use_AAL_naming, smooth, smooth_list, disp_filt, prune, multi_nets,
                             clust_type, clust_type_list, plugin_type, c_boot, block_size, mask, norm, binary,
                             runtime_dict, anat_file, vox_size='2mm'):
    import os
    import os.path as op
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, utils, thresholding
    from pynets.plotting import plot_gen
    from pynets.fmri import estimation, clustools
    from pynets.registration import register
    try:
        FSLDIR = os.environ['FSLDIR']
    except KeyError:
        print('FSLDIR environment variable not set!')

    import_list = ["import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib",
                   "import warnings", "warnings.filterwarnings(\"ignore\")"]
    functional_connectometry_wf = pe.Workflow(name="%s%s" % ('functional_connectometry_', ID))
    base_dirname = "%s%s" % ('functional_connectometry_', ID)
    if not os.path.isdir("%s%s" % ('/tmp/', base_dirname)):
        os.mkdir("%s%s" % ('/tmp/', base_dirname))
    functional_connectometry_wf.base_directory = "%s%s" % ('/tmp/', base_dirname)

    # Set paths to templates
    template = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain.nii.gz')
    template_mask = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain_mask.nii.gz')

    # Create basedir_path
    basedir_path = utils.do_dir_path('registration', func_file)

    # Create input/output nodes
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 'atlas_select', 'network',
                                                      'node_size', 'roi', 'thr',
                                                      'uatlas_select', 'multi_nets',
                                                      'conn_model', 'dens_thresh',
                                                      'conf', 'plot_switch', 'parc', 'ref_txt',
                                                      'procmem', 'k', 'clust_mask', 'k_min', 'k_max',
                                                      'k_step', 'k_clustering', 'user_atlas_list',
                                                      'min_span_tree', 'use_AAL_naming', 'smooth',
                                                      'disp_filt', 'prune', 'multi_nets', 'clust_type',
                                                      'c_boot', 'block_size', 'mask', 'norm', 'binary', 'template',
                                                      'template_mask', 'vox_size', 'anat_file', 'basedir_path']),
                        name='inputnode')

    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.roi = roi
    inputnode.inputs.thr = thr
    inputnode.inputs.uatlas_select = uatlas_select
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
    inputnode.inputs.multi_nets = multi_nets
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

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('atlas_select: ', atlas_select))
    # print("%s%s" % ('network: ', network))
    # print("%s%s" % ('node_size: ', node_size))
    # print("%s%s" % ('smooth: ', smooth))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('uatlas_select: ', uatlas_select))
    # print("%s%s" % ('conn_model: ', conn_model))
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
    # print('\n\n\n\n\n')

    # Create function nodes
    check_orient_and_dims_func_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=utils.check_orient_and_dims, imports=import_list),
                                              name="check_orient_and_dims_func_node")

    check_orient_and_dims_anat_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=utils.check_orient_and_dims, imports=import_list),
                                              name="check_orient_and_dims_anat_node")

    check_orient_and_dims_roi_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                          function=utils.check_orient_and_dims, imports=import_list),
                                             name="check_orient_and_dims_roi_node")

    check_orient_and_dims_mask_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=utils.check_orient_and_dims, imports=import_list),
                                              name="check_orient_and_dims_mask_node")

    register_node = pe.Node(niu.Function(input_names=['basedir_path', 'anat_file'],
                                         function=register.register_all_fmri, imports=import_list),
                            name="register_node")

    register_atlas_node = pe.Node(niu.Function(input_names=['uatlas_select', 'atlas_select', 'node_size',
                                                            'basedir_path', 'anat_file'],
                                               output_names=['aligned_atlas_t1mni_gm'],
                                               function=register.register_atlas_fmri, imports=import_list),
                                  name="register_atlas_node")

    # Clustering
    if float(k_clustering) > 0:
        clustering_node = pe.Node(niu.Function(input_names=['func_file', 'clust_mask', 'ID', 'k', 'clust_type'],
                                               output_names=['uatlas_select', 'atlas_select', 'clustering',
                                                             'clust_mask', 'k', 'clust_type'],
                                               function=clustools.individual_tcorr_clustering,
                                               imports=import_list), name="clustering_node")

        # Don't forget that this setting exists
        clustering_node.synchronize = True
        # clustering_node iterables and names
        if k_clustering == 1:
            mask_name = op.basename(clust_mask).split('.nii.gz')[0]
            cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
            cluster_atlas_file = "%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/',
                                                       mask_name, '_', clust_type, '_k', str(k), '.nii.gz')
            if user_atlas_list:
                user_atlas_list.append(cluster_atlas_file)
            elif uatlas_select and ((uatlas_select == cluster_atlas_file) is False):
                user_atlas_list = [uatlas_select, cluster_atlas_file]
            else:
                uatlas_select = cluster_atlas_file
        elif k_clustering == 2:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for k in k_list:
                mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 3:
            k_cluster_iterables = []
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 4:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                           func_file), '/', mask_name,
                                                                         '_', clust_type, '_k', str(k), '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 5:
            k_cluster_iterables = []
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                     '/', mask_name, '_', clust_type, '_k', str(k),
                                                                     '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 6:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                           func_file), '/', mask_name,
                                                                         '_', clust_type, '_k', str(k), '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 7:
            k_cluster_iterables = []
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file),
                                                                         '/', mask_name, '_', clust_type, '_k', str(k),
                                                                         '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 8:
            k_cluster_iterables = []
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        mask_name = op.basename(clust_mask).split('.nii.gz')[0]
                        cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                        cluster_atlas_name_list.append(cluster_atlas_name)
                        cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name,
                                                                                               func_file), '/',
                                                                             mask_name, '_', clust_type, '_k', str(k),
                                                                             '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list

    # Define nodes
    # Create node definitions Node
    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas_select', 'uatlas_select', 'ref_txt',
                                                                    'parc', 'in_file', 'use_AAL_naming', 'clustering'],
                                                       output_names=['label_names', 'coords', 'atlas_select',
                                                                     'networks_list', 'parcel_list', 'par_max',
                                                                     'uatlas_select', 'dir_path'],
                                                       function=nodemaker.fetch_nodes_and_labels,
                                                       imports=import_list), name="fetch_nodes_and_labels_node")
    # Connect clustering solutions to node definition Node
    if float(k_clustering) > 0:
        functional_connectometry_wf.add_nodes([clustering_node])
        functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID')])
                                             ])
        functional_connectometry_wf.connect([(check_orient_and_dims_func_node, clustering_node,
                                              [('outfile', 'func_file')])
                                             ])
        if k_clustering == 1:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('k', 'k'), ('clust_type', 'clust_type'),
                                                                               ('clust_mask', 'clust_mask')])
                                                 ])
        elif k_clustering == 2:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('clust_mask', 'clust_mask'),
                                                                               ('clust_type', 'clust_type')])
                                                 ])
        elif k_clustering == 3:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('k', 'k'),
                                                                               ('clust_type', 'clust_type')])
                                                 ])
        elif k_clustering == 4:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('clust_type', 'clust_type')])
                                                 ])
        elif k_clustering == 5:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('k', 'k'),
                                                                               ('clust_mask', 'clust_mask')])
                                                 ])
        elif k_clustering == 6:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('clust_mask', 'clust_mask')])
                                                 ])
        elif k_clustering == 7:
            functional_connectometry_wf.connect([(inputnode, clustering_node, [('k', 'k')])
                                                 ])

        functional_connectometry_wf.connect([(clustering_node, fetch_nodes_and_labels_node,
                                              [('uatlas_select', 'uatlas_select'),
                                               ('atlas_select', 'atlas_select'),
                                               ('clustering', 'clustering')])
                                             ])
    else:
        # Connect atlas input vars to node definition Node
        functional_connectometry_wf.connect([(inputnode, fetch_nodes_and_labels_node,
                                              [('atlas_select', 'atlas_select'),
                                               ('uatlas_select', 'uatlas_select')])
                                             ])

    # Set atlas iterables and logic for multiple atlas useage
    if ((multi_atlas is not None and user_atlas_list is None and
         uatlas_select is None) or (multi_atlas is None and atlas_select is
                                    None and user_atlas_list is not None)) and k_clustering == 0:
        # print('\n\n\n\n')
        # print('No flexi-atlas1')
        # print('\n\n\n\n')
        flexi_atlas = False
        if multi_atlas:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("atlas_select", multi_atlas))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
        elif user_atlas_list:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("uatlas_select", user_atlas_list))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
    elif (atlas_select is not None and
          uatlas_select is None and k_clustering == 0) or (atlas_select is None and uatlas_select is not None and
                                                           k_clustering == 0) or (k_clustering > 0 and atlas_select is
                                                                                  None and multi_atlas is None):
        # print('\n\n\n\n')
        # print('No flexi-atlas2')
        # print('\n\n\n\n')
        flexi_atlas = False
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(niu.IdentityInterface(fields=['atlas_select', 'uatlas_select', 'clustering']),
                                     name='flexi_atlas_source')
        flexi_atlas_source.synchronize = True
        if multi_atlas is not None and user_atlas_list is not None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: multiple nilearn atlases + multiple user atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas_select", len(user_atlas_list) * [None] + multi_atlas),
                                            ("uatlas_select", user_atlas_list + len(multi_atlas) * [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif multi_atlas is not None and uatlas_select is not None and user_atlas_list is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single user atlas + multiple nilearn atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas_select", multi_atlas + [None]),
                                            ("uatlas_select", len(multi_atlas) * [None] + [uatlas_select])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas_select is not None and user_atlas_list is not None and multi_atlas is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + multiple user atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas_select", len(user_atlas_list) * [None] + [atlas_select]),
                                            ("uatlas_select", user_atlas_list + [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas_select is not None and uatlas_select is not None and user_atlas_list is None and multi_atlas is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + single user atlas')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [("atlas_select", [atlas_select, None]),
                                            ("uatlas_select", [None, uatlas_select])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables

    # Connect flexi_atlas inputs to definition node
    if flexi_atlas is True:
        functional_connectometry_wf.add_nodes([flexi_atlas_source])
        if float(k_clustering) > 0:
            functional_connectometry_wf.disconnect([(clustering_node, fetch_nodes_and_labels_node,
                                                     [('uatlas_select', 'uatlas_select'),
                                                      ('atlas_select', 'atlas_select'),
                                                      ('clustering', 'clustering')])
                                                    ])
            if float(k_clustering == 1):
                functional_connectometry_wf.connect([(clustering_node, flexi_atlas_source,
                                                      [('clustering', 'clustering')])
                                                     ])
            else:
                clust_join_node = pe.JoinNode(niu.IdentityInterface(fields=['clustering', 'k', 'clust_mask',
                                                                            'clust_type']),
                                              name='clust_join_node',
                                              joinsource=clustering_node,
                                              joinfield=['clustering', 'k', 'clust_mask', 'clust_type'])
                functional_connectometry_wf.connect([(clustering_node, clust_join_node,
                                                      [('clustering', 'clustering'),
                                                       ('k', 'k'),
                                                       ('clust_mask', 'clust_mask'),
                                                       ('clust_type', 'clust_type')])
                                                     ])
                functional_connectometry_wf.connect([(clust_join_node, flexi_atlas_source,
                                                      [('clustering', 'clustering')])
                                                     ])
            functional_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                                  [('uatlas_select', 'uatlas_select'),
                                                   ('atlas_select', 'atlas_select'),
                                                   ('clustering', 'clustering')])
                                                 ])
        else:
            functional_connectometry_wf.disconnect([(inputnode, fetch_nodes_and_labels_node,
                                                     [('uatlas_select', 'uatlas_select'),
                                                      ('atlas_select', 'atlas_select')])
                                                    ])
            functional_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                                  [('uatlas_select', 'uatlas_select'),
                                                   ('atlas_select', 'atlas_select')])
                                                 ])
    # RSN case
    if network or multi_nets:
        get_node_membership_node = pe.Node(niu.Function(input_names=['network', 'infile', 'coords', 'label_names',
                                                                     'parc', 'parcel_list'],
                                                        output_names=['net_coords', 'net_parcel_list',
                                                                      'net_label_names',
                                                                      'network'],
                                                        function=nodemaker.get_node_membership, imports=import_list),
                                           name="get_node_membership_node")
        save_coords_and_labels_node = pe.Node(niu.Function(input_names=['coords', 'label_names', 'dir_path', 'network'],
                                                           function=utils.save_RSN_coords_and_labels_to_pickle,
                                                           imports=import_list), name="save_coords_and_labels_node")
        if multi_nets:
            print('Multiple resting-state networks (RSN\'s)...')
            get_node_membership_node_iterables = []
            network_iterables = ("network", multi_nets)
            get_node_membership_node_iterables.append(network_iterables)
            get_node_membership_node.iterables = get_node_membership_node_iterables

    # Generate nodes
    if roi is not None:
        # Masking case
        node_gen_node = pe.Node(niu.Function(input_names=['roi', 'coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select', 'mask'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select'],
                                             function=nodemaker.node_gen_masking, imports=import_list),
                                name="node_gen_node")

    else:
        # Non-masking case
        node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select'],
                                             function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    # Extract time-series from nodes
    extract_ts_iterables = []
    if parc is True:
        # Parcels case
        save_nifti_parcels_node = pe.Node(niu.Function(input_names=['ID', 'dir_path', 'roi', 'network',
                                                                    'net_parcels_map_nifti'],
                                                       output_names=['net_parcels_nii_path'],
                                                       function=utils.save_nifti_parcels_map, imports=import_list),
                                          name="save_nifti_parcels_node")
        # extract time series from whole brain parcellaions:
        extract_ts_node = pe.Node(niu.Function(input_names=['net_parcels_map_nifti', 'conf', 'func_file', 'coords',
                                                            'roi', 'dir_path', 'ID', 'network', 'smooth',
                                                            'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                            'c_boot', 'block_size'],
                                               output_names=['ts_within_nodes', 'node_size', 'smooth', 'dir_path',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'c_boot'],
                                               function=estimation.extract_ts_parc, imports=import_list),
                                  name="extract_ts_node")
        functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        functional_connectometry_wf.connect([(check_orient_and_dims_roi_node, save_nifti_parcels_node,
                                              [('outfile', 'roi')]),
                                             (inputnode, save_nifti_parcels_node, [('ID', 'ID'),
                                                                                   ('network', 'network')]), # network supposed to be here?
                                             (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                              [('dir_path', 'dir_path')]),
                                             (node_gen_node, save_nifti_parcels_node,
                                              [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                             ])
    else:
        # Coordinate case
        extract_ts_node = pe.Node(niu.Function(input_names=['node_size', 'conf', 'func_file', 'coords', 'dir_path',
                                                            'ID', 'roi', 'network', 'smooth', 'atlas_select',
                                                            'uatlas_select', 'label_names', 'c_boot', 'block_size'],
                                               output_names=['ts_within_nodes', 'node_size', 'smooth', 'dir_path',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'c_boot'],
                                               function=estimation.extract_ts_coords, imports=import_list),
                                  name="extract_ts_node")
        functional_connectometry_wf.disconnect([(node_gen_node, extract_ts_node,
                                                 [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                                ])
        # Set extract_ts iterables
        if node_size_list:
            extract_ts_iterables.append(("node_size", node_size_list))
            extract_ts_node.iterables = extract_ts_iterables

    if smooth_list:
        extract_ts_iterables.append(("smooth", smooth_list))
        extract_ts_node.iterables = extract_ts_iterables

    if c_boot > 0:
        extract_ts_iterables.append(("c_boot", np.arange(1, int(c_boot) + 1)))
        extract_ts_node.iterables = extract_ts_iterables

    # Connectivity matrix model fit
    get_conn_matrix_node = pe.Node(niu.Function(input_names=['time_series', 'conn_model', 'dir_path', 'node_size',
                                                             'smooth', 'dens_thresh', 'network', 'ID', 'roi',
                                                             'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'c_boot', 'norm', 'binary'],
                                                output_names=['conn_matrix', 'conn_model', 'dir_path', 'node_size',
                                                              'smooth', 'dens_thresh', 'network', 'ID', 'roi',
                                                              'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                              'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                              'c_boot', 'norm', 'binary'],
                                                function=estimation.get_conn_matrix, imports=import_list),
                                   name="get_conn_matrix_node")
    # Set get_conn_matrix_node iterables
    get_conn_matrix_node_iterables = []
    if conn_model_list:
        get_conn_matrix_node_iterables.append(("conn_model", conn_model_list))
        get_conn_matrix_node.iterables = get_conn_matrix_node_iterables

    # Connect nodes for RSN case
    if network or multi_nets:
        functional_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                    ('template', 'infile'),
                                                                                    ('parc', 'parc')]),
                                             (fetch_nodes_and_labels_node, get_node_membership_node,
                                              [('coords', 'coords'), ('label_names', 'label_names'),
                                               ('parcel_list', 'parcel_list'), ('par_max', 'par_max'),
                                               ('networks_list', 'networks_list')]),
                                             (get_node_membership_node, node_gen_node,
                                              [('net_coords', 'coords'), ('net_label_names', 'label_names'),
                                               ('net_parcel_list', 'parcel_list')]),
                                             (get_node_membership_node, save_coords_and_labels_node,
                                              [('net_coords', 'coords'), ('net_label_names', 'label_names'),
                                               ('network', 'network')]),
                                             (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                              [('dir_path', 'dir_path')]),
                                             (get_node_membership_node, extract_ts_node,
                                              [('network', 'network')]),
                                             (get_node_membership_node, get_conn_matrix_node,
                                              [('network', 'network')])
                                             ])
    else:
        functional_connectometry_wf.connect([(fetch_nodes_and_labels_node, node_gen_node,
                                              [('coords', 'coords'), ('label_names', 'label_names'),
                                               ('parcel_list', 'parcel_list')]),
                                             (inputnode, extract_ts_node,
                                              [('network', 'network')]),
                                             (inputnode, get_conn_matrix_node,
                                              [('network', 'network')])
                                             ])

    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout joins
    map_fields = ['conn_model', 'dir_path', 'conn_matrix', 'node_size', 'smooth', 'dens_thresh', 'network', 'ID',
                  'roi', 'min_span_tree', 'disp_filt', 'parc', 'prune', 'thr', 'atlas_select', 'uatlas_select',
                  'label_names', 'coords', 'c_boot', 'norm', 'binary']

    map_connects = [('conn_model', 'conn_model'), ('dir_path', 'dir_path'), ('conn_matrix', 'conn_matrix'),
                    ('node_size', 'node_size'), ('smooth', 'smooth'), ('dens_thresh', 'dens_thresh'), ('ID', 'ID'),
                    ('roi', 'roi'), ('min_span_tree', 'min_span_tree'), ('disp_filt', 'disp_filt'), ('parc', 'parc'),
                    ('prune', 'prune'), ('network', 'network'), ('thr', 'thr'), ('atlas_select', 'atlas_select'),
                    ('uatlas_select', 'uatlas_select'), ('label_names', 'label_names'), ('coords', 'coords'),
                    ('c_boot', 'c_boot'), ('norm', 'norm'), ('binary', 'binary')]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='thr_info_node')
    # Joinsource logic for atlas varieties
    if user_atlas_list or multi_atlas or float(k_clustering) > 0 or flexi_atlas is True:
        if flexi_atlas is True:
            atlas_join_source = flexi_atlas_source
        elif float(k_clustering) > 1 and flexi_atlas is False:
            atlas_join_source = clustering_node
        else:
            atlas_join_source = fetch_nodes_and_labels_node
    else:
        atlas_join_source = None

    # Connect all get_conn_matrix_node outputs to the "thr_info" node
    functional_connectometry_wf.connect([(get_conn_matrix_node, thr_info_node,
                                          [x for x in map_connects if x != ('thr', 'thr')])])
    # Begin joinnode chaining logic
    if conn_model_list or node_size_list or smooth_list or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True or multi_thr is True or float(c_boot) > 0:
        join_iters_node_thr = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_node_thr',
                                          joinsource=thr_info_node,
                                          joinfield=map_fields)
        join_iters_node_atlas = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                            name='join_iters_node_atlas',
                                            joinsource=atlas_join_source,
                                            joinfield=map_fields)
        if not conn_model_list and (node_size_list or smooth_list or float(c_boot) > 0):
            # print('Time-series node extraction iterables only...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=extract_ts_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list and not node_size_list and not smooth_list and not float(c_boot) > 0:
            # print('Connectivity model iterables only...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=get_conn_matrix_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif not conn_model_list and not node_size_list and not smooth_list and not float(c_boot) > 0:
            # print('No connectivity model or time-series node extraction iterables...')
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=thr_info_node, joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list and (node_size_list and smooth_list and float(c_boot) > 0) or (node_size_list or
                                                                                            smooth_list or
                                                                                            float(c_boot) > 0):
            # print('Connectivity model and time-series node extraction iterables...')
            join_iters_node_ext_ts = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                 name='join_iters_node_ext_ts', joinsource=extract_ts_node,
                                                 joinfield=map_fields)
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=get_conn_matrix_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_ext_ts, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_ext_ts, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_ext_ts, join_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_iters_node_ext_ts, join_iters_node, map_connects)])
        else:
            raise RuntimeError('\nERROR: Unknown join context.')

        no_iters = False
    else:
        # Minimal case of no iterables
        print('\nNo iterables...\n')
        join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
        functional_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        no_iters = True

    # Create final thresh_func node that performs the thresholding
    if no_iters is True:
        thresh_func_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                             'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                             'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'c_boot', 'norm', 'binary'],
                                                output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                              'node_size', 'network', 'conn_model', 'roi', 'smooth',
                                                              'prune', 'ID', 'dir_path', 'atlas_select', 'uatlas_select',
                                                              'label_names', 'coords', 'c_boot', 'norm', 'binary'],
                                                function=thresholding.thresh_func, imports=import_list),
                                   name="thresh_func_node")
    else:
        thresh_func_node = pe.MapNode(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                                'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                                'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                                'c_boot', 'norm', 'binary'],
                                                   output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                                 'node_size', 'network', 'conn_model', 'roi', 'smooth',
                                                                 'prune', 'ID', 'dir_path', 'atlas_select',
                                                                 'uatlas_select', 'label_names', 'coords', 'c_boot',
                                                                 'norm', 'binary'],
                                                   function=thresholding.thresh_func, imports=import_list),
                                      name="thresh_func_node", iterfield=['dens_thresh', 'thr', 'conn_matrix',
                                                                          'conn_model', 'network', 'ID', 'dir_path',
                                                                          'roi', 'node_size', 'min_span_tree', 'smooth',
                                                                          'disp_filt', 'parc', 'prune', 'atlas_select',
                                                                          'uatlas_select', 'label_names', 'coords',
                                                                          'c_boot', 'norm', 'binary'], nested=True)

    # Set iterables for thr on thresh_func, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr), float(max_thr),
                                                                          float(step_thr)), decimals=2).tolist()] +
                                      [str(float(max_thr))])))
        thr_info_node.iterables = ("thr", iter_thresh)
    else:
        thr_info_node.iterables = ("thr", [thr])

    # Plotting
    if plot_switch is True:
        plot_fields = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'roi',
                       'coords', 'thr', 'node_size', 'edge_threshold', 'smooth', 'prune', 'uatlas_select', 'c_boot',
                       'norm', 'binary']
        # Plotting iterable graph solutions
        if conn_model_list or node_size_list or smooth_list or float(c_boot) > 0 or multi_thr or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True:
            plot_all_node = pe.MapNode(niu.Function(input_names=plot_fields, output_names='None',
                                                    function=plot_gen.plot_all, imports=import_list), nested=True,
                                       itersource=thr_info_node,
                                       iterfield=plot_fields,
                                       name="plot_all_node")
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(niu.Function(input_names=plot_fields, output_names='None',
                                                 function=plot_gen.plot_all, imports=import_list), name="plot_all_node")

        # Connect thresh_func_node outputs to plotting node
        functional_connectometry_wf.connect([(thresh_func_node, plot_all_node, [('ID', 'ID'), ('roi', 'roi'),
                                                                                ('network', 'network'),
                                                                                ('prune', 'prune'),
                                                                                ('node_size', 'node_size'),
                                                                                ('smooth', 'smooth'),
                                                                                ('dir_path', 'dir_path'),
                                                                                ('conn_matrix_thr', 'conn_matrix'),
                                                                                ('edge_threshold', 'edge_threshold'),
                                                                                ('thr', 'thr'),
                                                                                ('conn_model', 'conn_model'),
                                                                                ('atlas_select', 'atlas_select'),
                                                                                ('uatlas_select', 'uatlas_select'),
                                                                                ('label_names', 'label_names'),
                                                                                ('coords', 'coords'),
                                                                                ('c_boot', 'c_boot'),
                                                                                ('norm', 'norm'),
                                                                                ('binary', 'binary')])
                                             ])
    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                       'conn_model', 'node_size', 'norm', 'binary']), name='outputnode')

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune',
                                                                         'ID', 'roi', 'conn_model', 'node_size',
                                                                         'smooth', 'c_boot', 'norm', 'binary']),
                                           name='join_iters_node_nets', joinsource=get_node_membership_node,
                                           joinfield=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                      'conn_model', 'node_size', 'smooth', 'c_boot', 'norm',
                                                      'binary'])
        functional_connectometry_wf.connect([
            (thresh_func_node, join_iters_node_nets, [('thr', 'thr'), ('network', 'network'),
                                                      ('est_path', 'est_path'), ('node_size', 'node_size'),
                                                      ('smooth', 'smooth'), ('roi', 'roi'),
                                                      ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                      ('prune', 'prune'), ('c_boot', 'c_boot'),
                                                      ('norm', 'norm'), ('binary', 'binary')]),
            (join_iters_node_nets, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                                ('node_size', 'node_size'), ('roi', 'roi'),
                                                ('conn_model', 'conn_model'), ('ID', 'ID'), ('prune', 'prune'),
                                                ('norm', 'norm'), ('binary', 'binary')])
        ])
    else:
        functional_connectometry_wf.connect([
            (thresh_func_node, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                            ('node_size', 'node_size'), ('roi', 'roi'), ('conn_model', 'conn_model'),
                                            ('ID', 'ID'), ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')])
        ])

    # Handle masking scenarios (brain mask and/or roi)
    if roi and not mask:
        functional_connectometry_wf.connect([
            (inputnode, check_orient_and_dims_roi_node, [('roi', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_roi_node, node_gen_node, [('outfile', 'roi')]),
            (check_orient_and_dims_roi_node, extract_ts_node, [('outfile', 'roi')]),
            (check_orient_and_dims_roi_node, get_conn_matrix_node, [('outfile', 'roi')]),
            (inputnode, check_orient_and_dims_mask_node, [('mask', 'mask')]),
            (inputnode, node_gen_node, [('mask', 'mask')])
        ])
    elif not roi and not mask:
        functional_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
            (inputnode, extract_ts_node, [('roi', 'roi')]),
            (inputnode, get_conn_matrix_node, [('roi', 'roi')]),
            (inputnode, check_orient_and_dims_mask_node, [('mask', 'mask')]),
            (inputnode, node_gen_node, [('mask', 'mask'), ('vox_size', 'vox_size')])
        ])
    elif mask and not roi:
        functional_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
            (inputnode, extract_ts_node, [('roi', 'roi')]),
            (inputnode, get_conn_matrix_node, [('roi', 'roi')]),
            (inputnode, check_orient_and_dims_mask_node, [('mask', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_mask_node, node_gen_node, [('outfile', 'mask')])
        ])
    elif not mask and op.isfile(template_mask) and not roi:
        functional_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
            (inputnode, extract_ts_node, [('roi', 'roi')]),
            (inputnode, get_conn_matrix_node, [('roi', 'roi')]),
            (inputnode, node_gen_node, [('template_mask', 'infile'), ('vox_size', 'vox_size')]),
        ])
    else:
        functional_connectometry_wf.connect([
            (inputnode, node_gen_node, [('roi', 'roi')]),
            (inputnode, extract_ts_node, [('roi', 'roi')]),
            (inputnode, get_conn_matrix_node, [('roi', 'roi')]),
            (inputnode, node_gen_node, [('mask', 'mask'), ('vox_size', 'vox_size')]),
        ])

    if roi and (mask or op.isfile(template_mask)):
        mask_roi_node = pe.Node(niu.Function(input_names=['dir_path', 'roi', 'mask', 'img_file'],
                                             output_names=['roi'],
                                             function=nodemaker.mask_roi, imports=import_list), name="mask_roi_node")
        if roi:
            functional_connectometry_wf.disconnect([
                (check_orient_and_dims_roi_node, node_gen_node, [('outfile', 'roi')])
            ])
        else:
            functional_connectometry_wf.disconnect([
                (inputnode, node_gen_node, [('roi', 'roi')])
            ])
        if mask:
            functional_connectometry_wf.disconnect([
                (check_orient_and_dims_mask_node, node_gen_node, [('outfile', 'mask')])
            ])
        else:
            if op.isfile(template_mask):
                functional_connectometry_wf.disconnect([
                    (inputnode, node_gen_node, [('template_mask', 'mask')])
                ])
            else:
                functional_connectometry_wf.disconnect([
                    (inputnode, node_gen_node, [('mask', 'mask')])
                ])
        functional_connectometry_wf.connect([
            (check_orient_and_dims_roi_node, mask_roi_node, [('outfile', 'roi')]),
            (mask_roi_node, extract_ts_node, [('roi', 'roi')]),
            (mask_roi_node, node_gen_node, [('roi', 'roi')]),
            (fetch_nodes_and_labels_node, mask_roi_node, [('dir_path', 'dir_path')]),
            (check_orient_and_dims_func_node, mask_roi_node, [('outfile', 'img_file')]),
        ])
        if mask:
            functional_connectometry_wf.connect([
                (check_orient_and_dims_mask_node, mask_roi_node, [('outfile', 'mask')]),
            ])
        else:
            functional_connectometry_wf.connect([
                (inputnode, mask_roi_node, [('template_mask', 'mask')]),
            ])

    # Connect remaining nodes of workflow
    functional_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('parc', 'parc'), ('ref_txt', 'ref_txt'),
                                                  ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, check_orient_and_dims_func_node, [('func_file', 'infile'),
                                                      ('vox_size', 'vox_size')]),
        (check_orient_and_dims_func_node, extract_ts_node, [('outfile', 'func_file')]),
        (check_orient_and_dims_func_node, fetch_nodes_and_labels_node, [('outfile', 'in_file')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('parc', 'parc')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('atlas_select', 'atlas_select'),
                                                      ('uatlas_select', 'uatlas_select'),
                                                      ('dir_path', 'dir_path'), ('par_max', 'par_max')]),
        (inputnode, extract_ts_node, [('conf', 'conf'), ('node_size', 'node_size'),
                                      ('ID', 'ID'), ('smooth', 'smooth'),
                                      ('c_boot', 'c_boot'), ('block_size', 'block_size')]),
        (inputnode, get_conn_matrix_node, [('conn_model', 'conn_model'),
                                           ('dens_thresh', 'dens_thresh'),
                                           ('ID', 'ID'),
                                           ('min_span_tree', 'min_span_tree'),
                                           ('disp_filt', 'disp_filt'),
                                           ('parc', 'parc'),
                                           ('prune', 'prune'),
                                           ('norm', 'norm'),
                                           ('binary', 'binary')]),
        (fetch_nodes_and_labels_node, extract_ts_node, [('dir_path', 'dir_path')]),
        (node_gen_node, extract_ts_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                          ('coords', 'coords'), ('label_names', 'label_names'),
                                          ('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')]),
        (extract_ts_node, get_conn_matrix_node, [('ts_within_nodes', 'time_series'), ('dir_path', 'dir_path'),
                                                 ('node_size', 'node_size'), ('smooth', 'smooth'),
                                                 ('coords', 'coords'), ('label_names', 'label_names'),
                                                 ('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select'),
                                                 ('c_boot', 'c_boot')]),
        (join_iters_node, thresh_func_node, map_connects)
        ])

    # Handle case that t1w image is available to refine parcellation
    if anat_file and (parc is True or float(k_clustering) > 0):
        functional_connectometry_wf.disconnect([
            (node_gen_node, extract_ts_node, [('uatlas_select', 'uatlas_select')]),
        ])
        functional_connectometry_wf.connect([
            (inputnode, check_orient_and_dims_anat_node, [('anat_file', 'infile'), ('vox_size', 'vox_size')]),
            (check_orient_and_dims_anat_node, register_node, [('outfile', 'anat_file')]),
            (check_orient_and_dims_anat_node, register_atlas_node, [('outfile', 'anat_file')]),
            (inputnode, register_node, [('basedir_path', 'basedir_path')]),
            (inputnode, register_atlas_node, [('basedir_path', 'basedir_path'), ('node_size', 'node_size')]),
            (node_gen_node, register_atlas_node, [('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')]),
            (register_atlas_node, extract_ts_node, [('aligned_atlas_t1mni_gm', 'uatlas_select')]),
        ])

    # Set cpu/memory reqs
    for node_name in functional_connectometry_wf.list_node_names():
        if node_name in runtime_dict:
            functional_connectometry_wf.get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
            functional_connectometry_wf.get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
            functional_connectometry_wf.get_node(node_name).n_procs = runtime_dict[node_name][0]
            functional_connectometry_wf.get_node(node_name)._mem_gb = runtime_dict[node_name][1]

    if k_clustering > 0:
        clustering_node._mem_gb = 4
        clustering_node.n_procs = 1
        clustering_node.interface.mem_gb = 4
        clustering_node.interface.n_procs = 1

    # Set runtime/logging configurations
    cfg = dict(execution={'stop_on_first_crash': True, 'hash_method': 'content', 'crashfile_format': 'txt',
                          'display_variable': ':0', 'job_finished_timeout': 65, 'matplotlib_backend': 'Agg',
                          'plugin': str(plugin_type), 'use_relative_paths': True, 'parameterize_dirs': True,
                          'remove_unnecessary_outputs': False, 'remove_node_directories': False,
                          'raise_insufficient': True, 'poll_sleep_duration': 0.01})
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            functional_connectometry_wf.config[key][setting] = value

    return functional_connectometry_wf


def structural_connectometry(ID, atlas_select, network, node_size, roi, uatlas_select, plot_switch, parc, ref_txt,
                             procmem, dwi_file, fbval, fbvec, anat_file, thr, dens_thresh, conn_model, user_atlas_list,
                             multi_thr, multi_atlas, max_thr, min_thr, step_thr, node_size_list, conn_model_list,
                             min_span_tree, use_AAL_naming, disp_filt, plugin_type, multi_nets, prune, mask, norm,
                             binary, target_samples, curv_thr_list, step_list, overlap_thr, overlap_thr_list, track_type,
                             max_length, maxcrossing, life_run, min_length, directget, tiss_class, runtime_dict,
                             vox_size='2mm'):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, thresholding, utils
    from pynets.registration import register
    from pynets.dmri import estimation, track
    from pynets.plotting import plot_gen
    import os
    try:
        FSLDIR = os.environ['FSLDIR']
    except KeyError:
        print('FSLDIR environment variable not set!')

    import_list = ["import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib",
                   "import warnings", "warnings.filterwarnings(\"ignore\")"]
    base_dirname = "%s%s" % ('structural_connectometry_', ID)
    structural_connectometry_wf = pe.Workflow(name=base_dirname)
    if not os.path.isdir("%s%s" % ('/tmp/', base_dirname)):
        os.mkdir("%s%s" % ('/tmp/', base_dirname))
    structural_connectometry_wf.base_directory = "%s%s" % ('/tmp/', base_dirname)

    # Set paths to templates
    template = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain.nii.gz')
    template_mask = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain_mask.nii.gz')

    # Create basedir_path
    basedir_path = utils.do_dir_path('registration', dwi_file)

    # Create input/output nodes
    inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'atlas_select', 'network', 'node_size', 'roi',
                                                      'uatlas_select', 'plot_switch', 'parc', 'ref_txt', 'procmem',
                                                      'dwi_file', 'fbval', 'fbvec', 'anat_file', 'thr', 'dens_thresh',
                                                      'conn_model', 'user_atlas_list', 'multi_thr', 'multi_atlas',
                                                      'max_thr', 'min_thr', 'step_thr', 'min_span_tree',
                                                      'use_AAL_naming', 'disp_filt', 'multi_nets', 'prune', 'mask',
                                                      'norm', 'binary', 'template', 'template_mask', 'target_samples',
                                                      'curv_thr_list', 'step_list', 'overlap_thr', 'overlap_thr_list',
                                                      'track_type', 'max_length', 'maxcrossing', 'life_run', 'min_length',
                                                      'directget', 'tiss_class', 'vox_size', 'basedir_path']),
                        name='inputnode')

    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.roi = roi
    inputnode.inputs.uatlas_select = uatlas_select
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
    inputnode.inputs.life_run = life_run
    inputnode.inputs.min_length = min_length
    inputnode.inputs.directget = directget
    inputnode.inputs.tiss_class = tiss_class
    inputnode.inputs.plugin_type = plugin_type
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.basedir_path = basedir_path

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('dwi_file: ', dwi_file))
    # print("%s%s" % ('fbval: ', fbval))
    # print("%s%s" % ('fbvec: ', fbvec))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('atlas_select: ', atlas_select))
    # print("%s%s" % ('network: ', network))
    # print("%s%s" % ('node_size: ', node_size))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('uatlas_select: ', uatlas_select))
    # print("%s%s" % ('conn_model: ', conn_model))
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
    # print('\n\n\n\n\n')

    # Create function nodes
    check_orient_and_dims_dwi_node = pe.Node(niu.Function(input_names=['infile', 'vox_size', 'bvecs'],
                                                          output_names=['outfile', 'bvecs'],
                                                          function=utils.check_orient_and_dims,
                                                          imports=import_list), name="check_orient_and_dims_dwi_node")

    check_orient_and_dims_anat_node = pe.Node(niu.Function(input_names=['infile', 'vox_size'], output_names=['outfile'],
                                                           function=utils.check_orient_and_dims, imports=import_list),
                                              name="check_orient_and_dims_anat_node")

    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas_select', 'uatlas_select', 'ref_txt', 'parc',
                                                                    'in_file', 'roi', 'use_AAL_naming'],
                                                       output_names=['label_names', 'coords', 'atlas_select',
                                                                     'networks_list', 'parcel_list', 'par_max',
                                                                     'uatlas_select', 'dir_path'],
                                                       function=nodemaker.fetch_nodes_and_labels,
                                                       imports=import_list), name="fetch_nodes_and_labels_node")

    if parc is False:
        prep_spherical_nodes_node = pe.Node(niu.Function(input_names=['coords', 'node_size', 'template_mask'],
                                                         output_names=['parcel_list', 'par_max', 'node_size', 'parc'],
                                                         function=nodemaker.create_spherical_roi_volumes,
                                                         imports=import_list),
                                            name="prep_spherical_nodes_node")

        if node_size_list:
            prep_spherical_nodes_node_iterables = []
            prep_spherical_nodes_node_iterables.append(("node_size", node_size_list))
            prep_spherical_nodes_node.iterables = prep_spherical_nodes_node_iterables

        save_nifti_parcels_node = pe.Node(niu.Function(input_names=['ID', 'dir_path', 'roi', 'network',
                                                                    'net_parcels_map_nifti'],
                                                       output_names=['net_parcels_nii_path'],
                                                       function=utils.save_nifti_parcels_map, imports=import_list),
                                          name="save_nifti_parcels_node")

    # Generate nodes
    if roi:
        # Masking case
        node_gen_node = pe.Node(niu.Function(input_names=['roi', 'coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select', 'mask'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select'],
                                             function=nodemaker.node_gen_masking, imports=import_list),
                                name="node_gen_node")
    else:
        # Non-masking case
        node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select'],
                                             function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    # RSN case
    if network or multi_nets:
        get_node_membership_node = pe.Node(niu.Function(input_names=['network', 'infile', 'coords', 'label_names',
                                                                     'parc', 'parcel_list'],
                                                        output_names=['net_coords', 'net_parcel_list',
                                                                      'net_label_names',
                                                                      'network'],
                                                        function=nodemaker.get_node_membership, imports=import_list),
                                           name="get_node_membership_node")
        save_coords_and_labels_node = pe.Node(niu.Function(input_names=['coords', 'label_names', 'dir_path', 'network'],
                                                           function=utils.save_RSN_coords_and_labels_to_pickle,
                                                           imports=import_list), name="save_coords_and_labels_node")
        if multi_nets:
            print('Multiple resting-state networks (RSN\'s)...')
            get_node_membership_node_iterables = []
            network_iterables = ("network", multi_nets)
            get_node_membership_node_iterables.append(network_iterables)
            get_node_membership_node.iterables = get_node_membership_node_iterables

    gtab_node = pe.Node(niu.Function(input_names=['fbval', 'fbvec', 'dwi_file'],
                                     output_names=['gtab_file', 'nodif_B0', 'nodif_B0_mask', 'dwi_file'],
                                     function=utils.make_gtab_and_bmask, imports=import_list), name="gtab_node")

    get_fa_node = pe.Node(niu.Function(input_names=['gtab_file', 'dwi_file', 'nodif_B0_mask'],
                                       output_names=['fa_path', 'nodif_B0_mask', 'gtab_file', 'dwi_file'],
                                       function=estimation.tens_mod_fa_est, imports=import_list), name="get_fa_node")

    register_node = pe.Node(niu.Function(input_names=['basedir_path', 'fa_path', 'nodif_B0_mask', 'anat_file',
                                                      'gtab_file', 'dwi_file'],
                                         output_names=['wm_gm_int_in_dwi', 'wm_in_dwi', 'gm_in_dwi', 'vent_csf_in_dwi',
                                                       'csf_mask_dwi', 'anat_file', 'nodif_B0_mask', 'fa_path',
                                                       'gtab_file', 'dwi_file'],
                                         function=register.register_all_dwi, imports=import_list),
                            name="register_node")

    register_atlas_node = pe.Node(niu.Function(input_names=['uatlas_select', 'atlas_select', 'node_size',
                                                            'basedir_path', 'fa_path', 'nodif_B0_mask', 'anat_file',
                                                            'wm_gm_int_in_dwi', 'coords', 'label_names',
                                                            'gm_in_dwi', 'vent_csf_in_dwi', 'wm_in_dwi', 'gtab_file',
                                                            'dwi_file'],
                                               output_names=['dwi_aligned_atlas_wmgm_int', 'dwi_aligned_atlas',
                                                             'aligned_atlas_t1mni', 'uatlas_select', 'atlas_select',
                                                             'coords', 'label_names', 'node_size', 'gm_in_dwi',
                                                             'vent_csf_in_dwi', 'wm_in_dwi', 'fa_path', 'gtab_file',
                                                             'nodif_B0_mask', 'dwi_file'],
                                               function=register.register_atlas_dwi, imports=import_list),
                                  name="register_atlas_node")

    run_tracking_node = pe.Node(niu.Function(input_names=['nodif_B0_mask', 'gm_in_dwi', 'vent_csf_in_dwi', 'wm_in_dwi',
                                                          'tiss_class', 'labels_im_file_wm_gm_int',
                                                          'labels_im_file', 'target_samples', 'curv_thr_list',
                                                          'step_list', 'track_type', 'max_length', 'maxcrossing',
                                                          'directget', 'conn_model', 'gtab_file', 'dwi_file', 'network',
                                                          'node_size', 'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                          'disp_filt', 'parc', 'prune', 'atlas_select',
                                                          'uatlas_select', 'label_names', 'coords', 'norm', 'binary',
                                                          'atlas_mni', 'life_run', 'min_length', 'fa_path'],
                                             output_names=['streams', 'track_type', 'target_samples',
                                                           'conn_model', 'dir_path', 'network',  'node_size',
                                                           'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                           'disp_filt', 'parc', 'prune', 'atlas_select',
                                                           'uatlas_select', 'label_names', 'coords', 'norm', 'binary',
                                                           'atlas_mni', 'curv_thr_list', 'step_list', 'fa_path'],
                                             function=track.run_track,
                                             imports=import_list),
                                name="run_tracking_node")

    # Set reconstruction model iterables
    run_tracking_node_iterables = []
    if conn_model_list:
        run_tracking_node_iterables.append(("conn_model", conn_model_list))
        run_tracking_node.iterables = run_tracking_node_iterables

    dsn_node = pe.Node(niu.Function(input_names=['streams', 'fa_path', 'dir_path', 'track_type', 'target_samples',
                                                 'conn_model', 'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                 'min_span_tree', 'disp_filt', 'parc', 'prune', 'atlas_select',
                                                 'uatlas_select', 'label_names', 'coords', 'norm', 'binary',
                                                 'atlas_mni', 'basedir_path', 'curv_thr_list', 'step_list'],
                                    output_names=['streams_warp', 'dir_path', 'track_type', 'target_samples',
                                                  'conn_model', 'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                  'min_span_tree', 'disp_filt', 'parc', 'prune', 'atlas_select',
                                                  'uatlas_select', 'label_names', 'coords', 'norm', 'binary',
                                                  'atlas_mni'],
                                    function=register.direct_streamline_norm,
                                    imports=import_list), name="dsn_node")

    streams2graph_node = pe.Node(niu.Function(input_names=['atlas_mni', 'streams', 'overlap_thr', 'dir_path',
                                                           'track_type', 'target_samples', 'conn_model',
                                                           'network', 'node_size', 'dens_thresh', 'ID', 'roi',
                                                           'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                           'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                           'norm', 'binary', 'curv_thr_list', 'step_list'],
                                              output_names=['atlas_mni', 'streams', 'conn_matrix', 'track_type',
                                                            'target_samples', 'dir_path', 'conn_model', 'network',
                                                            'node_size', 'dens_thresh', 'ID', 'roi', 'min_span_tree',
                                                            'disp_filt', 'parc', 'prune', 'atlas_select',
                                                            'uatlas_select', 'label_names', 'coords', 'norm', 'binary'],
                                              function=estimation.streams2graph,
                                              imports=import_list), name="streams2graph_node")

    # Set streams2graph_node iterables
    streams2graph_node_iterables = []
    if overlap_thr_list:
        streams2graph_node_iterables.append(("overlap_thr", overlap_thr_list))
        streams2graph_node.iterables = streams2graph_node_iterables

    if plot_switch is True:
        structural_plotting_node = pe.Node(niu.Function(input_names=['conn_matrix', 'label_names', 'atlas_select',
                                                                     'ID', 'dwi_file', 'network', 'parc', 'coords',
                                                                     'roi', 'dir_path', 'conn_model', 'thr',
                                                                     'node_size'],
                                                        function=plot_gen.structural_plotting,
                                                        imports=import_list),
                                           name="structural_plotting_node")

    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                       'conn_model', 'node_size', 'norm', 'binary']),
                         name='outputnode')

    if (multi_atlas is not None and user_atlas_list is None and uatlas_select is None) or (multi_atlas is None and
                                                                                           atlas_select is None and
                                                                                           user_atlas_list is not None):
        flexi_atlas = False
        if multi_atlas is not None and user_atlas_list is None:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("atlas_select", multi_atlas))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
        elif multi_atlas is None and user_atlas_list is not None:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("uatlas_select", user_atlas_list))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
    elif ((atlas_select is not None and uatlas_select is None) or
          (atlas_select is None and uatlas_select is not None)) and (multi_atlas is None and user_atlas_list is None):
        flexi_atlas = False
        pass
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(niu.IdentityInterface(fields=['atlas_select', 'uatlas_select']),
                                     name='flexi_atlas_source')
        if multi_atlas is not None and user_atlas_list is not None:
            flexi_atlas_source_iterables = [("atlas_select", len(user_atlas_list) * [None] + multi_atlas),
                                            ("uatlas_select", user_atlas_list + len(multi_atlas) * [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
            flexi_atlas_source.synchronize = True
        elif multi_atlas is not None and uatlas_select is not None and user_atlas_list is None:
            flexi_atlas_source_iterables = [("atlas_select", multi_atlas + [None]),
                                            ("uatlas_select", len(multi_atlas) * [None] + [uatlas_select])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
            flexi_atlas_source.synchronize = True
        elif atlas_select is not None and user_atlas_list is not None and multi_atlas is None:
            flexi_atlas_source_iterables = [("atlas_select", len(user_atlas_list) * [None] + [atlas_select]),
                                            ("uatlas_select", user_atlas_list + [None])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
            flexi_atlas_source.synchronize = True
        elif atlas_select is not None and uatlas_select is not None and user_atlas_list is None and multi_atlas is None:
            flexi_atlas_source_iterables = [("atlas_select", [atlas_select, None]),
                                            ("uatlas_select", [None, uatlas_select])]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
            flexi_atlas_source.synchronize = True

    # if plot_switch is True:
    #     structural_connectometry_wf.add_nodes([structural_plotting_node])
    #     structural_connectometry_wf.connect([(collect_struct_mapping_outputs_node, structural_plotting_node,
    #                                              [('conn_matrix_symm', 'conn_matrix_symm')]),
    #                                             (inputnode, structural_plotting_node, [('ID', 'ID'),
    #                                                                                    ('dwi_file', 'dwi_file'),
    #                                                                                    ('network', 'network'),
    #                                                                                    ('parc', 'parc'),
    #                                                                                    ('roi', 'roi'),
    #                                                                                    ('plot_switch', 'plot_switch')]),
    #                                             (thresh_diff_node, structural_plotting_node,
    #                                              [('thr', 'thr'),
    #                                               ('node_size', 'node_size'), ('conn_model', 'conn_model')]),
    #                                             (node_gen_node, structural_plotting_node,
    #                                              [('label_names', 'label_names'),
    #                                               ('coords', 'coords')]),
    #                                             (fetch_nodes_and_labels_node, structural_plotting_node,
    #                                              [('dir_path', 'dir_path'),
    #                                               ('atlas_select', 'atlas_select')])
    #                                             ])
    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout joins
    map_fields = ['conn_model', 'dir_path', 'conn_matrix', 'node_size', 'dens_thresh', 'network', 'ID',
                  'roi', 'min_span_tree', 'disp_filt', 'parc', 'prune', 'thr', 'atlas_select', 'uatlas_select',
                  'label_names', 'coords', 'norm', 'binary', 'target_samples', 'track_type', 'atlas_mni', 'streams']

    map_connects = [('conn_model', 'conn_model'), ('dir_path', 'dir_path'), ('conn_matrix', 'conn_matrix'),
                    ('node_size', 'node_size'), ('dens_thresh', 'dens_thresh'), ('ID', 'ID'),
                    ('roi', 'roi'), ('min_span_tree', 'min_span_tree'), ('disp_filt', 'disp_filt'), ('parc', 'parc'),
                    ('prune', 'prune'), ('network', 'network'), ('thr', 'thr'), ('atlas_select', 'atlas_select'),
                    ('uatlas_select', 'uatlas_select'), ('label_names', 'label_names'), ('coords', 'coords'),
                    ('norm', 'norm'), ('binary', 'binary'), ('target_samples', 'target_samples'),
                    ('track_type', 'track_type'), ('atlas_mni', 'atlas_mni'), ('streams', 'streams')]

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
    structural_connectometry_wf.connect([(streams2graph_node, thr_info_node,
                                          [x for x in map_connects if x != ('thr', 'thr')])])
    # Begin joinnode chaining logic
    if conn_model_list or node_size_list or user_atlas_list or multi_atlas or flexi_atlas is True or multi_thr is True:
        join_iters_node_thr = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                          name='join_iters_node_thr',
                                          joinsource=thr_info_node,
                                          joinfield=map_fields)
        join_iters_node_atlas = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                            name='join_iters_node_atlas',
                                            joinsource=atlas_join_source,
                                            joinfield=map_fields)
        if not conn_model_list and (node_size_list and parc is False):
            # print('Node extraction iterables only...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=prep_spherical_nodes_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list and not node_size_list:
            # print('Connectivity model iterables only...')
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=run_tracking_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif not conn_model_list and not node_size_list:
            # print('No connectivity model or node extraction iterables...')
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=thr_info_node, joinfield=map_fields)
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                                  joinsource=atlas_join_source, joinfield=map_fields)
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        elif conn_model_list or (node_size_list and parc is False):
            # print('Connectivity model and node extraction iterables...')
            join_iters_node_prep_spheres = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                       name='join_iters_node_prep_spheres',
                                                       joinsource=prep_spherical_nodes_node, joinfield=map_fields)
            join_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields), name='join_iters_node',
                                          joinsource=run_tracking_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_prep_spheres, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_prep_spheres, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_thr, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_thr, join_iters_node_prep_spheres, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_prep_spheres, join_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True:
                    # print('Multiple atlases...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_prep_spheres, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_prep_spheres, join_iters_node_atlas, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_atlas, join_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    structural_connectometry_wf.connect([(thr_info_node, join_iters_node_prep_spheres, map_connects)])
                    structural_connectometry_wf.connect([(join_iters_node_prep_spheres, join_iters_node, map_connects)])
        else:
            raise RuntimeError('\nERROR: Unknown join context.')

        no_iters = False
    else:
        # Minimal case of no iterables
        print('\nNo iterables...\n')
        join_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields), name='join_iters_node')
        structural_connectometry_wf.connect([(thr_info_node, join_iters_node, map_connects)])
        no_iters = True

    # Create final thresh_diff node that performs the thresholding
    if no_iters is True:
        thresh_diff_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                             'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                             'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'norm', 'binary', 'target_samples', 'track_type',
                                                             'atlas_mni', 'streams'],
                                                output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                              'node_size', 'network', 'conn_model', 'roi',
                                                              'prune', 'ID', 'dir_path', 'atlas_select',
                                                              'uatlas_select', 'label_names', 'coords',
                                                              'norm', 'binary', 'target_samples',
                                                              'track_type', 'atlas_mni', 'streams'],
                                                function=thresholding.thresh_diff, imports=import_list),
                                   name="thresh_diff_node")
    else:
        thresh_diff_node = pe.MapNode(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                'network', 'ID', 'dir_path', 'roi', 'node_size',
                                                                'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                                'atlas_select', 'uatlas_select', 'label_names',
                                                                'coords', 'norm', 'binary', 'target_samples',
                                                                'track_type', 'atlas_mni', 'streams'],
                                                   output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                                 'node_size', 'network', 'conn_model', 'roi',
                                                                 'prune', 'ID', 'dir_path', 'atlas_select',
                                                                 'uatlas_select', 'label_names', 'coords',
                                                                 'norm', 'binary', 'target_samples', 'track_type',
                                                                 'atlas_mni', 'streams'],
                                                   function=thresholding.thresh_diff, imports=import_list),
                                      name="thresh_diff_node", iterfield=['dens_thresh', 'thr', 'conn_matrix',
                                                                          'conn_model', 'network', 'ID', 'dir_path',
                                                                          'roi', 'node_size', 'min_span_tree',
                                                                          'disp_filt', 'parc', 'prune', 'atlas_select',
                                                                          'uatlas_select', 'label_names', 'coords',
                                                                          'norm', 'binary', 'target_samples',
                                                                          'track_type', 'atlas_mni', 'streams'],
                                      nested=True)

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune',
                                                                         'ID', 'roi', 'conn_model', 'node_size',
                                                                         'target_samples', 'track_type', 'norm',
                                                                         'binary', 'atlas_mni', 'streams']),
                                           name='join_iters_node_nets', joinsource=get_node_membership_node,
                                           joinfield=['est_path', 'thr', 'network', 'prune', 'ID', 'roi',
                                                      'conn_model', 'node_size', 'target_samples', 'track_type', 'norm',
                                                      'binary', 'atlas_mni', 'streams'])
        structural_connectometry_wf.connect([
            (thresh_diff_node, join_iters_node_nets, [('thr', 'thr'), ('network', 'network'),
                                                      ('est_path', 'est_path'), ('node_size', 'node_size'),
                                                      ('track_type', 'track_type'), ('roi', 'roi'),
                                                      ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                      ('prune', 'prune'), ('target_samples', 'target_samples'),
                                                      ('norm', 'norm'), ('binary', 'binary'),
                                                      ('atlas_mni', 'atlas_mni'), ('streams', 'streams')]),
            (join_iters_node_nets, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                                ('node_size', 'node_size'), ('roi', 'roi'),
                                                ('conn_model', 'conn_model'), ('ID', 'ID'), ('prune', 'prune'),
                                                ('norm', 'norm'), ('binary', 'binary')])
        ])
    else:
        structural_connectometry_wf.connect([
            (thresh_diff_node, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                            ('node_size', 'node_size'), ('roi', 'roi'), ('conn_model', 'conn_model'),
                                            ('ID', 'ID'), ('prune', 'prune'), ('norm', 'norm'), ('binary', 'binary')]),
        ])

    # Set iterables for thr on thresh_diff, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr), float(max_thr),
                                                                          float(step_thr)), decimals=2).tolist()] +
                                      [str(float(max_thr))])))
        thr_info_node.iterables = ("thr", iter_thresh)
    else:
        thr_info_node.iterables = ("thr", [thr])

    # Connect nodes of workflow
    structural_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                  ('uatlas_select', 'uatlas_select'),
                                                  ('parc', 'parc'),
                                                  ('ref_txt', 'ref_txt'),
                                                  ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('roi', 'roi')]),
        (inputnode, check_orient_and_dims_dwi_node, [('dwi_file', 'infile'),
                                                     ('fbvec', 'bvecs'),
                                                     ('vox_size', 'vox_size')]),
        (check_orient_and_dims_dwi_node, fetch_nodes_and_labels_node, [('outfile', 'in_file')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path'),
                                                      ('par_max', 'par_max'),
                                                      ('networks_list', 'networks_list'),
                                                      ('atlas_select', 'atlas_select'),
                                                      ('uatlas_select', 'uatlas_select')]),
        (check_orient_and_dims_dwi_node, gtab_node, [('bvecs', 'fbvec'),
                                                     ('outfile', 'dwi_file')]),
        (inputnode, gtab_node, [('fbval', 'fbval')]),
        (inputnode, register_node, [('basedir_path', 'basedir_path')]),
        (inputnode, check_orient_and_dims_anat_node, [('anat_file', 'infile'), ('vox_size', 'vox_size')]),
        (check_orient_and_dims_anat_node, register_node, [('outfile', 'anat_file')]),
        (inputnode, register_atlas_node, [('basedir_path', 'basedir_path')]),
        (register_node, register_atlas_node, [('anat_file', 'anat_file'),
                                              ('gm_in_dwi', 'gm_in_dwi'),
                                              ('vent_csf_in_dwi', 'vent_csf_in_dwi'),
                                              ('wm_in_dwi', 'wm_in_dwi'),
                                              ('wm_gm_int_in_dwi', 'wm_gm_int_in_dwi'),
                                              ('fa_path', 'fa_path'),
                                              ('nodif_B0_mask', 'nodif_B0_mask'),
                                              ('gtab_file', 'gtab_file'),
                                              ('dwi_file', 'dwi_file')]),
        (gtab_node, get_fa_node, [('nodif_B0_mask', 'nodif_B0_mask'),
                                  ('gtab_file', 'gtab_file'),
                                  ('dwi_file', 'dwi_file')]),
        (get_fa_node, register_node, [('fa_path', 'fa_path'),
                                      ('nodif_B0_mask', 'nodif_B0_mask'),
                                      ('gtab_file', 'gtab_file'),
                                      ('dwi_file', 'dwi_file')]),
        (node_gen_node, register_atlas_node, [('atlas_select', 'atlas_select'),
                                              ('uatlas_select', 'uatlas_select'),
                                              ('coords', 'coords'),
                                              ('label_names', 'label_names')]),
        (register_atlas_node, run_tracking_node, [('dwi_aligned_atlas_wmgm_int', 'labels_im_file_wm_gm_int'),
                                                  ('dwi_aligned_atlas', 'labels_im_file'),
                                                  ('fa_path', 'fa_path'),
                                                  ('aligned_atlas_t1mni', 'atlas_mni'),
                                                  ('atlas_select', 'atlas_select'),
                                                  ('uatlas_select', 'uatlas_select'),
                                                  ('coords', 'coords'),
                                                  ('label_names', 'label_names'),
                                                  ('gm_in_dwi', 'gm_in_dwi'),
                                                  ('vent_csf_in_dwi', 'vent_csf_in_dwi'),
                                                  ('wm_in_dwi', 'wm_in_dwi'),
                                                  ('gtab_file', 'gtab_file'),
                                                  ('nodif_B0_mask', 'nodif_B0_mask'),
                                                  ('dwi_file', 'dwi_file')]),
        (inputnode, run_tracking_node, [('conn_model', 'conn_model'),
                                        ('tiss_class', 'tiss_class'),
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
                                        ('directget', 'directget'),
                                        ('life_run', 'life_run'),
                                        ('min_length', 'min_length')]),
        (inputnode, streams2graph_node, [('overlap_thr', 'overlap_thr'),
                                         ('curv_thr_list', 'curv_thr_list'),
                                         ('step_list', 'step_list')]),
        (inputnode, dsn_node, [('basedir_path', 'basedir_path')]),
        (run_tracking_node, dsn_node, [('dir_path', 'dir_path'),
                                       ('streams', 'streams'),
                                       ('curv_thr_list', 'curv_thr_list'),
                                       ('step_list', 'step_list')]),
        (run_tracking_node, dsn_node, [('track_type', 'track_type'),
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
                                       ('atlas_select', 'atlas_select'),
                                       ('uatlas_select', 'uatlas_select'),
                                       ('label_names', 'label_names'),
                                       ('coords', 'coords'),
                                       ('norm', 'norm'),
                                       ('binary', 'binary'),
                                       ('atlas_mni', 'atlas_mni'),
                                       ('fa_path', 'fa_path')]),
        (dsn_node, streams2graph_node, [('streams_warp', 'streams'),
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
                                        ('atlas_select', 'atlas_select'),
                                        ('uatlas_select', 'uatlas_select'),
                                        ('label_names', 'label_names'),
                                        ('coords', 'coords'),
                                        ('norm', 'norm'),
                                        ('binary', 'binary'),
                                        ('atlas_mni', 'atlas_mni')]),
        (join_iters_node, thresh_diff_node, map_connects)
        ])

    if flexi_atlas is True:
        structural_connectometry_wf.disconnect([(inputnode, fetch_nodes_and_labels_node,
                                                 [('uatlas_select', 'uatlas_select'),
                                                  ('atlas_select', 'atlas_select')])
                                                ])
        structural_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                              [('uatlas_select', 'uatlas_select'),
                                               ('atlas_select', 'atlas_select')])
                                             ])

    # Connect nodes for RSN case
    if parc is False:
        if network or multi_nets:
            structural_connectometry_wf.disconnect([(fetch_nodes_and_labels_node, get_node_membership_node,
                                                     [('parcel_list', 'parcel_list'),
                                                      ('par_max', 'par_max')]),
                                                    (inputnode, node_gen_node, [('parc', 'parc')])
                                                    ])
            structural_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                        ('template', 'infile'),
                                                                                        ('parc', 'parc')]),
                                                 (prep_spherical_nodes_node, get_node_membership_node,
                                                  [('parcel_list', 'parcel_list'),
                                                   ('par_max', 'par_max')]),
                                                 (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                                  [('dir_path', 'dir_path')]),
                                                 (fetch_nodes_and_labels_node, get_node_membership_node,
                                                  [('coords', 'coords'), ('label_names', 'label_names'),
                                                   ('networks_list', 'networks_list')]),
                                                 (get_node_membership_node, save_coords_and_labels_node,
                                                  [('net_coords', 'coords'), ('net_label_names', 'label_names'),
                                                   ('network', 'network')]),
                                                 (get_node_membership_node, run_tracking_node,
                                                  [('network', 'network')]),
                                                 (prep_spherical_nodes_node, node_gen_node,
                                                  [('parc', 'parc'),
                                                   ('parcel_list', 'parcel_list')]),
                                                 (get_node_membership_node, node_gen_node,
                                                  [('net_coords', 'coords'), ('net_label_names', 'label_names')]),
                                                 ])
        else:
            structural_connectometry_wf.disconnect([(fetch_nodes_and_labels_node, node_gen_node,
                                                     [('parcel_list', 'parcel_list'),
                                                      ('par_max', 'par_max')]),
                                                    (inputnode, node_gen_node, [('parc', 'parc')])
                                                    ])
            structural_connectometry_wf.connect([(prep_spherical_nodes_node, node_gen_node,
                                                  [('parcel_list', 'parcel_list'),
                                                   ('par_max', 'par_max'),
                                                   ('parc', 'parc')]),
                                                 (fetch_nodes_and_labels_node, node_gen_node,
                                                  [('coords', 'coords'),
                                                   ('label_names', 'label_names')]),
                                                 (inputnode, run_tracking_node,
                                                  [('network', 'network')])
                                                 ])

        structural_connectometry_wf.disconnect([(node_gen_node, register_atlas_node,
                                                 [('uatlas_select', 'uatlas_select')])
                                                ])
        structural_connectometry_wf.connect([(inputnode, prep_spherical_nodes_node,
                                              [('node_size', 'node_size'),
                                               ('template_mask', 'template_mask')]),
                                             (fetch_nodes_and_labels_node, prep_spherical_nodes_node,
                                              [('dir_path', 'dir_path'),
                                               ('coords', 'coords')]),
                                             (inputnode, save_nifti_parcels_node,
                                              [('ID', 'ID'),
                                               ('roi', 'roi'),
                                               ('network', 'network')]),
                                             (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                              [('dir_path', 'dir_path')]),
                                             (node_gen_node, save_nifti_parcels_node,
                                              [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
                                             (prep_spherical_nodes_node, register_atlas_node,
                                              [('node_size', 'node_size')]),
                                             (save_nifti_parcels_node, register_atlas_node,
                                              [('net_parcels_nii_path', 'uatlas_select')]),
                                             (register_atlas_node, run_tracking_node,
                                              [('node_size', 'node_size')]),
                                             (run_tracking_node, dsn_node,
                                              [('network', 'network')]),
                                             (dsn_node, streams2graph_node,
                                              [('network', 'network')])
                                             ])
    else:
        if network or multi_nets:
            structural_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                        ('template', 'infile'),
                                                                                        ('parc', 'parc')]),
                                                 (fetch_nodes_and_labels_node, get_node_membership_node,
                                                  [('coords', 'coords'), ('label_names', 'label_names'),
                                                   ('parcel_list', 'parcel_list'), ('par_max', 'par_max'),
                                                   ('networks_list', 'networks_list')]),
                                                 (get_node_membership_node, node_gen_node,
                                                  [('net_coords', 'coords'), ('net_label_names', 'label_names'),
                                                   ('net_parcel_list', 'parcel_list')]),
                                                 (fetch_nodes_and_labels_node, save_coords_and_labels_node,
                                                  [('dir_path', 'dir_path')]),
                                                 (get_node_membership_node, save_coords_and_labels_node,
                                                  [('net_coords', 'coords'), ('net_label_names', 'label_names'),
                                                   ('network', 'network')]),
                                                 ])
        else:
            structural_connectometry_wf.connect([(fetch_nodes_and_labels_node, node_gen_node,
                                                  [('coords', 'coords'), ('label_names', 'label_names'),
                                                   ('parcel_list', 'parcel_list')])
                                                 ])

        structural_connectometry_wf.connect([(inputnode, run_tracking_node,
                                              [('node_size', 'node_size'),
                                               ('network', 'network')]),
                                             (inputnode, node_gen_node,
                                               [('parc', 'parc')]),
                                             (run_tracking_node, dsn_node,
                                              [('network', 'network')]),
                                             (inputnode, register_atlas_node,
                                              [('node_size', 'node_size')]),
                                             (dsn_node, streams2graph_node,
                                              [('network', 'network')])
                                             ])

    for node_name in structural_connectometry_wf.list_node_names():
        if node_name in runtime_dict:
            structural_connectometry_wf.get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
            structural_connectometry_wf.get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
            structural_connectometry_wf.get_node(node_name).n_procs = runtime_dict[node_name][0]
            structural_connectometry_wf.get_node(node_name)._mem_gb = runtime_dict[node_name][1]

    cfg = dict(execution={'stop_on_first_crash': True, 'hash_method': 'content', 'crashfile_format': 'txt',
                          'display_variable': ':0', 'job_finished_timeout': 65, 'matplotlib_backend': 'Agg',
                          'plugin': str(plugin_type), 'use_relative_paths': True, 'parameterize_dirs': True,
                          'remove_unnecessary_outputs': False, 'remove_node_directories': False,
                          'raise_insufficient': True})
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            structural_connectometry_wf.config[key][setting] = value

    return structural_connectometry_wf
