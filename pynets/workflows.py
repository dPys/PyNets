# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np


def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, uatlas_select, multi_nets,
                      conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir, anat_loc, parc,
                      ref_txt, procmem, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k,
                      clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                      node_size_list, num_total_samples, conn_model_list, min_span_tree, verbose, plugin_type,
                      use_AAL_naming, smooth, smooth_list, disp_filt, clust_type, clust_type_list):
    from pynets import workflows
    from nipype import Workflow
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.utils import pass_meta_outs

    # Workflow 1: Functional connectome
    if dwi_dir is None:
        sub_func_wf = workflows.functional_connectometry(input_file, ID, atlas_select, network, node_size,
                                                         mask, thr, uatlas_select, conn_model, dens_thresh, conf,
                                                         plot_switch, parc, ref_txt, procmem,
                                                         multi_thr, multi_atlas, max_thr, min_thr, step_thr,
                                                         k, clust_mask, k_min, k_max, k_step, k_clustering,
                                                         user_atlas_list, clust_mask_list, node_size_list,
                                                         conn_model_list, min_span_tree, use_AAL_naming, smooth,
                                                         smooth_list, disp_filt, prune, multi_nets, clust_type,
                                                         clust_type_list)
        sub_struct_wf = None
    # Workflow 2: Structural connectome
    elif dwi_dir is not None and network is None:
        sub_struct_wf = workflows.structural_connectometry(ID, atlas_select, network, node_size, mask,
                                                           uatlas_select, plot_switch, parc, ref_txt, procmem,
                                                           dwi_dir, anat_loc, thr, dens_thresh,
                                                           conn_model, user_atlas_list, multi_thr, multi_atlas,
                                                           max_thr, min_thr, step_thr, node_size_list,
                                                           num_total_samples, conn_model_list, min_span_tree,
                                                           use_AAL_naming, disp_filt)
        sub_func_wf = None

    base_wf = sub_func_wf if sub_func_wf else sub_struct_wf

    # Create meta-workflow to organize graph simulation sets in prep for analysis
    # Credit: @Mathias Goncalves
    base_dirname = "%s%s" % ('Meta_wf_', ID)
    meta_wf = Workflow(name=base_dirname)
    # Create input/output nodes
    meta_inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'network', 'thr',
                                                           'node_size', 'mask', 'uatlas_select', 'multi_nets',
                                                           'conn_model', 'dens_thresh', 'conf', 'adapt_thresh',
                                                           'plot_switch', 'dwi_dir', 'anat_loc', 'parc', 'ref_txt',
                                                           'procmem', 'multi_thr', 'multi_atlas', 'max_thr',
                                                           'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max',
                                                           'k_step', 'k_clustering', 'user_atlas_list',
                                                           'clust_mask_list', 'prune', 'node_size_list',
                                                           'num_total_samples', 'conn_model_list',
                                                           'min_span_tree', 'verbose', 'plugin_type', 'use_AAL_naming',
                                                           'smooth', 'smooth_list', 'disp_filt', 'clust_type',
                                                           'clust_type_list']),
                             name='meta_inputnode')
    meta_inputnode.inputs.in_file = input_file
    meta_inputnode.inputs.ID = ID
    meta_inputnode.inputs.atlas_select = atlas_select
    meta_inputnode.inputs.network = network
    meta_inputnode.inputs.thr = thr
    meta_inputnode.inputs.node_size = node_size
    meta_inputnode.inputs.mask = mask
    meta_inputnode.inputs.uatlas_select = uatlas_select
    meta_inputnode.inputs.multi_nets = multi_nets
    meta_inputnode.inputs.conn_model = conn_model
    meta_inputnode.inputs.dens_thresh = dens_thresh
    meta_inputnode.inputs.conf = conf
    meta_inputnode.inputs.adapt_thresh = adapt_thresh
    meta_inputnode.inputs.plot_switch = plot_switch
    meta_inputnode.inputs.dwi_dir = dwi_dir
    meta_inputnode.inputs.anat_loc = anat_loc
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

    meta_wf.add_nodes([base_wf])
    meta_wf.connect([(meta_inputnode, base_wf, [('ID', 'inputnode.ID'),
                                                ('atlas_select', 'inputnode.atlas_select'),
                                                ('network', 'inputnode.network'),
                                                ('node_size', 'inputnode.node_size'),
                                                ('mask', 'inputnode.mask'),
                                                ('thr', 'inputnode.thr'),
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
                                                ('clust_type_list', 'inputnode.clust_type_list')])
                     ])

    pass_meta_outs_node = pe.Node(niu.Function(input_names=['conn_model', 'est_path', 'network', 'node_size',
                                  'smooth', 'thr', 'prune', 'ID', 'mask'], output_names=['conn_model_iterlist',
                                                                                         'est_path_iterlist',
                                                                                         'network_iterlist',
                                                                                         'node_size_iterlist',
                                                                                         'smooth_iterlist',
                                                                                         'thr_iterlist',
                                                                                         'prune_iterlist',
                                                                                         'ID_iterlist',
                                                                                         'mask_iterlist'],
                                               function=pass_meta_outs), name='pass_meta_outs_node')

    # Set resource restrictions at level of the meta wf
    if input_file:
        wf_selected = "%s%s" % ('functional_connectometry_', ID)
        meta_wf.get_node("%s%s" % (wf_selected, '.fetch_nodes_and_labels_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.fetch_nodes_and_labels_node'))._mem_gb = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.extract_ts_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.extract_ts_node'))._mem_gb = 4
        meta_wf.get_node("%s%s" % (wf_selected, '.node_gen_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.node_gen_node'))._mem_gb = 1
        if k_clustering > 0:
            meta_wf.get_node("%s%s" % (wf_selected, '.clustering_node'))._n_procs = 1
            meta_wf.get_node("%s%s" % (wf_selected, '.clustering_node'))._mem_gb = 8
        meta_wf.get_node("%s%s" % (wf_selected, '.get_conn_matrix_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.get_conn_matrix_node'))._mem_gb = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.thresh_func_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.thresh_func_node'))._mem_gb = 1

    if dwi_dir:
        wf_selected = "%s%s" % ('structural_connectometry_', ID)
        meta_wf.get_node("%s%s" % (wf_selected, '.fetch_nodes_and_labels_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.fetch_nodes_and_labels_node'))._mem_gb = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.thresh_diff_node'))._n_procs = 1
        meta_wf.get_node("%s%s" % (wf_selected, '.thresh_diff_node'))._mem_gb = 1

    # Connect outputs of nested workflow to parent wf
    meta_wf.connect([(base_wf.get_node('outputnode'), pass_meta_outs_node, [('conn_model', 'conn_model'),
                                                                            ('est_path', 'est_path'),
                                                                            ('network', 'network'),
                                                                            ('node_size', 'node_size'),
                                                                            ('smooth', 'smooth'),
                                                                            ('thr', 'thr'),
                                                                            ('prune', 'prune'),
                                                                            ('ID', 'ID'),
                                                                            ('mask', 'mask')])
                     ])

    return meta_wf


def functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, uatlas_select, conn_model,
                             dens_thresh, conf, plot_switch, parc, ref_txt, procmem, multi_thr,
                             multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step,
                             k_clustering, user_atlas_list, clust_mask_list, node_size_list, conn_model_list,
                             min_span_tree, use_AAL_naming, smooth, smooth_list, disp_filt, prune, multi_nets,
                             clust_type, clust_type_list):
    import os
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, utils, graphestimation, plotting, thresholding
    import_list = ["import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]
    functional_connectometry_wf = pe.Workflow(name='functional_connectometry_' + str(ID))
    base_dirname = "%s%s%s%s" % ('functional_connectometry_', str(ID), '/Meta_wf_imp_est_', str(ID))
    functional_connectometry_wf.base_directory = os.path.dirname(func_file) + base_dirname

    # Create input/output nodes
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID',
                                                      'atlas_select', 'network',
                                                      'node_size', 'mask', 'thr',
                                                      'uatlas_select', 'multi_nets',
                                                      'conn_model', 'dens_thresh',
                                                      'conf', 'plot_switch', 'parc', 'ref_txt',
                                                      'procmem', 'k',
                                                      'clust_mask', 'k_min', 'k_max',
                                                      'k_step', 'k_clustering', 'user_atlas_list',
                                                      'min_span_tree', 'use_AAL_naming', 'smooth',
                                                      'disp_filt', 'prune', 'multi_nets', 'clust_type']),
                        name='inputnode')

    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
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

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('atlas_select: ', atlas_select))
    # print("%s%s" % ('network: ', network))
    # print("%s%s" % ('node_size: ', node_size))
    # print("%s%s" % ('smooth: ', smooth))
    # print("%s%s" % ('mask: ', mask))
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
    # print('\n\n\n\n\n')

    # Create function nodes
    # Clustering
    if float(k_clustering) > 0:
        clustering_node = pe.Node(niu.Function(input_names=['func_file', 'clust_mask', 'ID', 'k',
                                                            'clust_type'],
                                               output_names=['uatlas_select', 'atlas_select', 'clustering',
                                                             'clust_mask', 'k', 'clust_type'],
                                               function=utils.individual_tcorr_clustering,
                                               imports=import_list), name="clustering_node")
        clustering_node.synchronize = True

        # clustering_node iterables and names
        if k_clustering == 1:
            mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
            cluster_atlas_file = "%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz')
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
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
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
                        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        cluster_atlas_name = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', k)
                        cluster_atlas_name_list.append(cluster_atlas_name)
                        cluster_atlas_file_list.append("%s%s%s%s%s%s%s%s" % (utils.do_dir_path(cluster_atlas_name, func_file), '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz'))
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif uatlas_select:
                user_atlas_list = cluster_atlas_file_list + [uatlas_select]
            else:
                user_atlas_list = cluster_atlas_file_list

    # Define nodes
    # Create node definitions Node
    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas_select', 'uatlas_select', 'ref_txt',
                                                                    'parc', 'func_file', 'use_AAL_naming', 'clustering'],
                                                       output_names=['label_names', 'coords', 'atlas_select',
                                                                     'networks_list', 'parcel_list', 'par_max',
                                                                     'uatlas_select', 'dir_path'],
                                                       function=nodemaker.fetch_nodes_and_labels,
                                                       imports=import_list), name="fetch_nodes_and_labels_node")

    # Connect clustering solutions to node definition Node
    if float(k_clustering) > 0:
        functional_connectometry_wf.add_nodes([clustering_node])
        functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'),
                                                                           ('func_file', 'func_file')])
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
    if ((multi_atlas is not None and user_atlas_list is None and uatlas_select is None) or (multi_atlas is None and atlas_select is None and user_atlas_list is not None)) and k_clustering == 0:
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
    elif (atlas_select is not None and uatlas_select is None and k_clustering == 0) or (atlas_select is None and uatlas_select is not None and k_clustering == 0) or (k_clustering > 0 and atlas_select is None and multi_atlas is None):
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
        get_node_membership_node = pe.Node(niu.Function(input_names=['network', 'func_file', 'coords', 'label_names',
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
    if mask is not None:
        # Masking case
        node_gen_node = pe.Node(niu.Function(input_names=['mask', 'coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select', 'vox_array'],
                                             function=nodemaker.node_gen_masking, imports=import_list),
                                name="node_gen_masking_node")
    else:
        # Non-masking case
        node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'label_names', 'dir_path',
                                                          'ID', 'parc', 'atlas_select', 'uatlas_select'],
                                             output_names=['net_parcels_map_nifti', 'coords', 'label_names',
                                                           'atlas_select', 'uatlas_select', 'vox_array'],
                                             function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    # Extract time-series from nodes
    extract_ts_iterables = []
    if parc is True:
        # Parcels case
        save_nifti_parcels_node = pe.Node(niu.Function(input_names=['ID', 'dir_path', 'mask', 'network',
                                                                    'net_parcels_map_nifti'],
                                                       function=utils.save_nifti_parcels_map, imports=import_list),
                                          name="save_nifti_parcels_node")
        # extract time series from whole brain parcellaions:
        extract_ts_node = pe.Node(niu.Function(input_names=['net_parcels_map_nifti', 'conf', 'func_file', 'coords',
                                                            'mask', 'dir_path', 'ID', 'network', 'smooth',
                                                            'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                               output_names=['ts_within_nodes', 'node_size', 'smooth', 'dir_path',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                               function=graphestimation.extract_ts_parc, imports=import_list),
                                  name="extract_ts_node")
        functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        functional_connectometry_wf.connect([(inputnode, save_nifti_parcels_node, [('ID', 'ID'), ('mask', 'mask')]),
                                             (inputnode, save_nifti_parcels_node, [('network', 'network')]),
                                             (fetch_nodes_and_labels_node, save_nifti_parcels_node,
                                              [('dir_path', 'dir_path')]),
                                             (node_gen_node, save_nifti_parcels_node,
                                              [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                             ])
    else:
        # Coordinate case
        extract_ts_node = pe.Node(niu.Function(input_names=['node_size', 'conf', 'func_file', 'coords', 'dir_path',
                                                            'ID', 'mask', 'network', 'smooth', 'atlas_select',
                                                            'uatlas_select', 'label_names'],
                                               output_names=['ts_within_nodes', 'node_size', 'smooth', 'dir_path',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                               function=graphestimation.extract_ts_coords, imports=import_list),
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

    # Connectivity matrix model fit
    get_conn_matrix_node = pe.Node(niu.Function(input_names=['time_series', 'conn_model', 'dir_path', 'node_size',
                                                             'smooth', 'dens_thresh', 'network', 'ID', 'mask',
                                                             'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords',
                                                             'vox_array'],
                                                output_names=['conn_matrix', 'conn_model', 'dir_path', 'node_size',
                                                              'smooth', 'dens_thresh', 'network', 'ID', 'mask',
                                                              'min_span_tree', 'disp_filt', 'parc', 'prune',
                                                              'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                                function=graphestimation.get_conn_matrix, imports=import_list),
                                   name="get_conn_matrix_node")

    # Set get_conn_matrix_node iterables
    get_conn_matrix_node_iterables = []
    if conn_model_list:
        get_conn_matrix_node_iterables.append(("conn_model", conn_model_list))
        get_conn_matrix_node.iterables = get_conn_matrix_node_iterables

    # Connect nodes for RSN case
    if network or multi_nets:
        functional_connectometry_wf.connect([(inputnode, get_node_membership_node, [('network', 'network'),
                                                                                    ('func_file', 'func_file'),
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
                  'mask', 'min_span_tree', 'disp_filt', 'parc', 'prune', 'thr', 'atlas_select', 'uatlas_select',
                  'label_names', 'coords']

    map_connects = [('conn_model', 'conn_model'), ('dir_path', 'dir_path'), ('conn_matrix', 'conn_matrix'),
                    ('node_size', 'node_size'), ('smooth', 'smooth'), ('dens_thresh', 'dens_thresh'), ('ID', 'ID'),
                    ('mask', 'mask'), ('min_span_tree', 'min_span_tree'), ('disp_filt', 'disp_filt'), ('parc', 'parc'),
                    ('prune', 'prune'), ('network', 'network'), ('thr', 'thr'), ('atlas_select', 'atlas_select'),
                    ('uatlas_select', 'uatlas_select'), ('label_names', 'label_names'), ('coords', 'coords')]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(niu.IdentityInterface(fields=map_fields),
                            name='thr_info_node')

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
    if conn_model_list or node_size_list or smooth_list or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True or multi_thr is True:
        join_info_iters_node_thr = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                               name='join_info_iters_node_thr',
                                               joinsource=thr_info_node,
                                               joinfield=map_fields)
        join_info_iters_node_atlas = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                 name='join_info_iters_node_atlas',
                                                 joinsource=atlas_join_source,
                                                 joinfield=map_fields)
        if not conn_model_list and (node_size_list or smooth_list):
            # print('Time-series node extraction iterables only...')
            join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                               name='join_info_iters_node',
                                               joinsource=extract_ts_node, joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node_atlas,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node,
                                                          map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_atlas,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node,
                                                          map_connects)])
        elif conn_model_list and not node_size_list and not smooth_list:
            # print('Connectivity model iterables only...')
            join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                               name='join_info_iters_node',
                                               joinsource=get_conn_matrix_node,
                                               joinfield=map_fields)
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node_atlas,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    join_info_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields),
                                                   name='join_info_iters_node')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node,
                                                          map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_atlas, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node, map_connects)])
        elif not conn_model_list and not node_size_list and not smooth_list:
            # print('No connectivity model or time-series node extraction iterables...')
            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                       name='join_info_iters_node',
                                                       joinsource=atlas_join_source,
                                                       joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                       name='join_info_iters_node',
                                                       joinsource=thr_info_node,
                                                       joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node, map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                       name='join_info_iters_node',
                                                       joinsource=atlas_join_source,
                                                       joinfield=map_fields)
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node, map_connects)])
                else:
                    # print('Single atlas...')
                    join_info_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields),
                                                   name='join_info_iters_node')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node, map_connects)])
        elif conn_model_list and (node_size_list and smooth_list) or (node_size_list or smooth_list):
            # print('Connectivity model and time-series node extraction iterables...')
            join_info_iters_node_ext_ts = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                                      name='join_info_iters_node_ext_ts',
                                                      joinsource=extract_ts_node, joinfield=map_fields)
            join_info_iters_node = pe.JoinNode(niu.IdentityInterface(fields=map_fields),
                                               name='join_info_iters_node',
                                               joinsource=get_conn_matrix_node,
                                               joinfield=map_fields)

            if multi_thr:
                # print('Multiple thresholds...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node_ext_ts,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_ext_ts, join_info_iters_node_atlas,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_thr, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_thr, join_info_iters_node_ext_ts,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_ext_ts, join_info_iters_node,
                                                          map_connects)])
            else:
                # print('Single threshold...')
                if user_atlas_list or multi_atlas or flexi_atlas is True or float(k_clustering) > 1:
                    # print('Multiple atlases...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_ext_ts, join_info_iters_node_atlas,
                                                          map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_atlas, join_info_iters_node,
                                                          map_connects)])
                else:
                    # print('Single atlas...')
                    functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node_ext_ts, map_connects)])
                    functional_connectometry_wf.connect([(join_info_iters_node_ext_ts, join_info_iters_node,
                                                          map_connects)])
        else:
            raise RuntimeError('\nERROR: Unknown join context.')

        no_iters = False
    else:
        # Minimal case of no iterables
        print('\nNo iterables...\n')
        join_info_iters_node = pe.Node(niu.IdentityInterface(fields=map_fields),
                                       name='join_info_iters_node')
        functional_connectometry_wf.connect([(thr_info_node, join_info_iters_node, map_connects)])
        no_iters = True

    # Create final thresh_func node that performs the thresholding
    if no_iters is True:
        thresh_func_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                             'network', 'ID', 'dir_path', 'mask', 'node_size',
                                                             'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                             'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                                output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                              'node_size', 'network', 'conn_model', 'mask', 'smooth',
                                                              'prune', 'ID', 'dir_path', 'atlas_select', 'uatlas_select',
                                                              'label_names', 'coords'],
                                                function=thresholding.thresh_func, imports=import_list),
                                   name="thresh_func_node")
    else:
        thresh_func_node = pe.MapNode(niu.Function(input_names=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                'network', 'ID', 'dir_path', 'mask', 'node_size',
                                                                'min_span_tree', 'smooth', 'disp_filt', 'parc', 'prune',
                                                                'atlas_select', 'uatlas_select', 'label_names', 'coords'],
                                                   output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                                 'node_size', 'network', 'conn_model', 'mask', 'smooth',
                                                                 'prune', 'ID', 'dir_path', 'atlas_select', 'uatlas_select',
                                                                 'label_names', 'coords'],
                                                   function=thresholding.thresh_func, imports=import_list),
                                      name="thresh_func_node", iterfield=['dens_thresh', 'thr', 'conn_matrix', 'conn_model',
                                                                          'network', 'ID', 'dir_path', 'mask', 'node_size',
                                                                          'min_span_tree', 'smooth', 'disp_filt', 'parc',
                                                                          'prune', 'atlas_select', 'uatlas_select',
                                                                          'label_names', 'coords'], nested=True)

    # Set iterables for thr on thresh_func, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr),
                                                                          float(max_thr), float(step_thr)),
                                                                decimals=2).tolist()] + [str(float(max_thr))])))
        thr_info_node.iterables = ("thr", iter_thresh)
    else:
        thr_info_node.iterables = ("thr", [thr])

    # Plotting
    if plot_switch is True:
        plot_fields = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask',
                       'coords', 'thr', 'node_size', 'edge_threshold', 'smooth', 'prune', 'uatlas_select']
        # Plotting iterable graph solutions
        if conn_model_list or node_size_list or smooth_list or multi_thr or user_atlas_list or multi_atlas or float(k_clustering) > 1 or flexi_atlas is True:
            plot_all_node = pe.MapNode(niu.Function(input_names=plot_fields, output_names='None',
                                                    function=plotting.plot_all, imports=import_list), nested=True,
                                       itersource=thr_info_node,
                                       iterfield=plot_fields,
                                       name="plot_all_node")
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(niu.Function(input_names=plot_fields,
                                                 output_names='None',
                                                 function=plotting.plot_all, imports=import_list),
                                    name="plot_all_node")
        # Connect thresh_func_node outputs to plotting node
        functional_connectometry_wf.connect([(thresh_func_node, plot_all_node, [('ID', 'ID'), ('mask', 'mask'),
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
                                                                                ('coords', 'coords')])
                                             ])
    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune', 'ID', 'mask',
                                                       'conn_model', 'node_size', 'smooth']), name='outputnode')

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_info_iters_node_nets = pe.JoinNode(niu.IdentityInterface(fields=['est_path', 'thr', 'network', 'prune',
                                                                              'ID', 'mask', 'conn_model', 'node_size',
                                                                              'smooth']),
                                                name='join_info_iters_node_nets',
                                                joinsource=get_node_membership_node,
                                                joinfield=['est_path', 'thr', 'network', 'prune', 'ID', 'mask',
                                                           'conn_model', 'node_size', 'smooth'])
        functional_connectometry_wf.connect([
            (thresh_func_node, join_info_iters_node_nets, [('thr', 'thr'), ('network', 'network'),
                                                           ('est_path', 'est_path'), ('node_size', 'node_size'),
                                                           ('smooth', 'smooth'), ('mask', 'mask'),
                                                           ('conn_model', 'conn_model'), ('ID', 'ID'),
                                                           ('prune', 'prune')]),
            (join_info_iters_node_nets, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                                     ('node_size', 'node_size'), ('smooth', 'smooth'), ('mask', 'mask'),
                                                     ('conn_model', 'conn_model'), ('ID', 'ID'), ('prune', 'prune')])
        ])
    else:
        functional_connectometry_wf.connect([
            (thresh_func_node, outputnode, [('thr', 'thr'), ('network', 'network'), ('est_path', 'est_path'),
                                            ('node_size', 'node_size'), ('smooth', 'smooth'), ('mask', 'mask'),
                                            ('conn_model', 'conn_model'), ('ID', 'ID'), ('prune', 'prune')])
        ])

    # Connect remaining nodes of workflow
    functional_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('func_file', 'func_file'),
                                                  ('parc', 'parc'), ('ref_txt', 'ref_txt'),
                                                  ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('atlas_select', 'atlas_select'),
                                                      ('uatlas_select', 'uatlas_select'),
                                                      ('dir_path', 'dir_path'), ('par_max', 'par_max')]),
        (inputnode, extract_ts_node, [('conf', 'conf'), ('func_file', 'func_file'), ('node_size', 'node_size'),
                                      ('mask', 'mask'), ('ID', 'ID'), ('smooth', 'smooth')]),
        (inputnode, get_conn_matrix_node, [('conn_model', 'conn_model'),
                                           ('dens_thresh', 'dens_thresh'),
                                           ('ID', 'ID'),
                                           ('mask', 'mask'),
                                           ('min_span_tree', 'min_span_tree'),
                                           ('disp_filt', 'disp_filt'),
                                           ('parc', 'parc'),
                                           ('prune', 'prune')]),
        (fetch_nodes_and_labels_node, extract_ts_node, [('dir_path', 'dir_path')]),
        (node_gen_node, extract_ts_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                          ('coords', 'coords'), ('label_names', 'label_names'),
                                          ('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')]),
        (node_gen_node, get_conn_matrix_node, [('vox_array', 'vox_array')]),
        (extract_ts_node, get_conn_matrix_node, [('ts_within_nodes', 'time_series'), ('dir_path', 'dir_path'),
                                                 ('node_size', 'node_size'), ('smooth', 'smooth'),
                                                 ('coords', 'coords'), ('label_names', 'label_names'),
                                                 ('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')]),
        (join_info_iters_node, thresh_func_node, map_connects)
        ])

    # Set cpu/memory reqs
    if k_clustering > 0:
        clustering_node._mem_gb = 8
        clustering_node.n_procs = 1
        clustering_node.interface.mem_gb = 8
        clustering_node.interface.n_procs = 1
    fetch_nodes_and_labels_node.interface.mem_gb = 1
    fetch_nodes_and_labels_node.interface.n_procs = 1
    fetch_nodes_and_labels_node._mem_gb = 1
    fetch_nodes_and_labels_node.n_procs = 1
    node_gen_node.interface.mem_gb = 1
    node_gen_node.interface.n_procs = 1
    node_gen_node._mem_gb = 1
    node_gen_node.n_procs = 1
    extract_ts_node.interface.mem_gb = 4
    extract_ts_node.interface.n_procs = 1
    extract_ts_node._mem_gb = 4
    extract_ts_node.n_procs = 1
    get_conn_matrix_node.interface.mem_gb = 1
    get_conn_matrix_node.interface.n_procs = 1
    get_conn_matrix_node._mem_gb = 1
    get_conn_matrix_node.n_procs = 1
    thresh_func_node._mem_gb = 1
    thresh_func_node.n_procs = 1

    # Set runtime/logging configurations
    functional_connectometry_wf.config['execution']['crashdump_dir'] = functional_connectometry_wf.base_directory
    functional_connectometry_wf.config['execution']['crashfile_format'] = 'txt'
    functional_connectometry_wf.config['execution']['keep_inputs'] = True
    functional_connectometry_wf.config['execution']['remove_unnecessary_outputs'] = False
    functional_connectometry_wf.config['execution']['remove_node_directories'] = False
    functional_connectometry_wf.config['logging']['log_directory'] = functional_connectometry_wf.base_directory
    functional_connectometry_wf.config['logging']['workflow_level'] = 'DEBUG'
    functional_connectometry_wf.config['logging']['utils_level'] = 'DEBUG'
    functional_connectometry_wf.config['logging']['interface_level'] = 'DEBUG'
    functional_connectometry_wf.config['execution']['display_variable'] = ':0'

    return functional_connectometry_wf


def structural_connectometry(ID, atlas_select, network, node_size, mask, uatlas_select, plot_switch, parc, ref_txt,
                             procmem, dwi_dir, anat_loc, thr, dens_thresh, conn_model,
                             user_atlas_list, multi_thr, multi_atlas, max_thr, min_thr, step_thr, node_size_list,
                             num_total_samples, conn_model_list, min_span_tree, use_AAL_naming, disp_filt):
    import os.path
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, diffconnectometry, plotting, thresholding

    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')

    import_list = ["import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]
    structural_connectometry_wf = pe.Workflow(name='structural_connectometry_' + str(ID))
    base_dirname = "%s%s%s%s" % ('structural_connectometry_', str(ID), '/Meta_wf_imp_est_', str(ID))
    structural_connectometry_wf.base_directory = dwi_dir + base_dirname

    # Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'atlas_select', 'network', 'node_size', 'mask',
                                                      'uatlas_select', 'plot_switch', 'parc', 'ref_txt', 'procmem',
                                                      'dir_path', 'dwi_dir', 'anat_loc', 'thr', 'dens_thresh',
                                                      'conn_model', 'user_atlas_list', 'multi_thr', 'multi_atlas',
                                                      'max_thr', 'min_thr', 'step_thr', 'num_total_samples',
                                                      'min_span_tree', 'use_AAL_naming', 'disp_filt']),
                        name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.uatlas_select = uatlas_select
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dir_path = dir_path
    inputnode.inputs.dwi_dir = dwi_dir
    inputnode.inputs.anat_loc = anat_loc
    inputnode.inputs.nodif_brain_mask_path = nodif_brain_mask_path
    inputnode.inputs.thr = thr
    inputnode.inputs.dens_thresh = dens_thresh
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.user_atlas_list = user_atlas_list
    inputnode.inputs.multi_thr = multi_thr
    inputnode.inputs.multi_atlas = multi_atlas
    inputnode.inputs.max_thr = max_thr
    inputnode.inputs.min_thr = min_thr
    inputnode.inputs.step_thr = step_thr
    inputnode.inputs.num_total_samples = num_total_samples
    inputnode.inputs.conn_model_list = conn_model_list
    inputnode.inputs.min_span_tree = min_span_tree
    inputnode.inputs.use_AAL_naming = use_AAL_naming
    inputnode.inputs.disp_filt = disp_filt

    #3) Add variable to function nodes
    # Create function nodes
    fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names=['atlas_select', 'uatlas_select', 'ref_txt',
                                                                       'parc', 'func_file', 'mask', 'use_AAL_naming'],
                                                          output_names=['label_names', 'coords', 'atlas_select',
                                                                        'networks_list', 'parcel_list', 'par_max',
                                                                        'uatlas_select', 'dir_path'],
                                                          function=nodemaker.fetch_nodes_and_labels,
                                                          imports=import_list), name="fetch_nodes_and_labels_node")
    # Node generation
    # if mask is not None:
    #     node_gen_node = pe.Node(niu.Function(input_names=['mask', 'coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'],
    #                                                  output_names=['net_parcels_map_nifti', 'coords', 'label_names'],
    #                                                  function=nodemaker.node_gen_masking, imports=import_list), name="node_gen_masking_node")
    # else:
    #     node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'],
    #                                                  output_names=['net_parcels_map_nifti', 'coords', 'label_names'],
    #                                                  function=nodemaker.node_gen, imports=import_list), name="node_gen_node")

    node_gen_node = pe.Node(niu.Function(input_names=['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'],
                                         output_names=['net_parcels_map_nifti', 'coords', 'label_names'],
                                         function=nodemaker.node_gen, imports=import_list), name="node_gen_node")
    create_mni2diff_transforms_node = pe.Node(niu.Function(input_names=['dwi_dir'], output_names=['out_aff'],
                                                           function=diffconnectometry.create_mni2diff_transforms,
                                                           imports=import_list), name="create_mni2diff_transforms_node")
    CSF_file = "%s%s" % (anat_loc, '/CSF.nii.gz')
    WM_file = "%s%s" % (anat_loc, '/WM.nii.gz')
    if anat_loc and not os.path.isfile(CSF_file) and not os.path.isfile(WM_file):
        gen_anat_segs_node = pe.Node(niu.Function(input_names=['anat_loc', 'out_aff'],
                                                  output_names=['new_file_csf', 'mni_csf_loc', 'new_file_wm'],
                                                  function=diffconnectometry.gen_anat_segs, imports=import_list),
                                     name="gen_anat_segs_node")
        no_segs = False
    else:
        no_segs = True
        print('\nRunning tractography without tissue maps. This is not recommended. Consider including a T1/T2 anatomical image with the -anat flag instead.\n')

    prepare_masks_node = pe.Node(niu.Function(input_names=['dwi_dir', 'csf_loc', 'mni_csf_loc', 'wm_mask_loc',
                                                           'mask'],
                                              output_names=['vent_CSF_diff_mask_path', 'way_mask'],
                                              function=diffconnectometry.prepare_masks, imports=import_list),
                                 name="prepare_masks_node")
    prep_nodes_node = pe.Node(niu.Function(input_names=['dwi_dir', 'node_size', 'parc', 'parcel_list',
                                                        'net_parcels_map_nifti', 'network', 'dir_path', 'mask',
                                                        'atlas_select'],
                                           output_names=['parcel_list', 'seeds_dir', 'node_size'],
                                           function=diffconnectometry.prep_nodes, imports=import_list),
                              name="prep_nodes_node")
    if parc is True:
        reg_parcels2diff_node = pe.Node(niu.Function(input_names=['dwi_dir', 'seeds_dir'],
                                                     output_names=['seeds_list'],
                                                     function=diffconnectometry.reg_parcels2diff, imports=import_list),
                                        name="reg_parcels2diff_node")
    else:
        build_coord_list_node = pe.Node(niu.Function(input_names=['dwi_dir', 'coords'],
                                                     output_names=['coords'],
                                                     function=diffconnectometry.build_coord_list, imports=import_list),
                                        name="build_coord_list_node")
        reg_coords2diff_node = pe.Node(niu.Function(input_names=['coords', 'dwi_dir', 'node_size', 'seeds_dir'],
                                                    output_names=['done_nodes'],
                                                    function=diffconnectometry.reg_coords2diff, imports=import_list),
                                       name="reg_coords2diff_node")
        cleanup_tmp_nodes_node = pe.Node(niu.Function(input_names=['done_nodes', 'coords', 'dir_path', 'seeds_dir'],
                                                      output_names=['seeds_list'],
                                                      function=diffconnectometry.cleanup_tmp_nodes, imports=import_list),
                                         name="cleanup_tmp_nodes_node")
    create_seed_mask_file_node = pe.Node(niu.Function(input_names=['node_size', 'network', 'dir_path', 'parc',
                                                                   'seeds_list', 'atlas_select'],
                                                      output_names=['seeds_text', 'probtrackx_output_dir_path'],
                                                      function=diffconnectometry.create_seed_mask_file,
                                                      imports=import_list),
                                         name="create_seed_mask_file_node")
    run_probtrackx2_node = pe.Node(niu.Function(input_names=['i', 'seeds_text', 'dwi_dir',
                                                             'probtrackx_output_dir_path', 'vent_CSF_diff_mask_path',
                                                             'way_mask', 'procmem', 'num_total_samples'],
                                                function=diffconnectometry.run_probtrackx2, imports=import_list),
                                   name="run_probtrackx2_node")
    run_dipy_tracking_node = pe.Node(niu.Function(input_names=['dwi_dir', 'node_size', 'dir_path',
                                                               'conn_model', 'parc', 'atlas_select',
                                                               'network', 'wm_mask'],
                                                  function=diffconnectometry.dwi_dipy_run, imports=import_list),
                                     name="run_dipy_tracking_node")
    collect_struct_mapping_outputs_node = pe.Node(niu.Function(input_names=['parc', 'dwi_dir', 'network', 'ID',
                                                                            'probtrackx_output_dir_path', 'dir_path',
                                                                            'procmem', 'seeds_dir'],
                                                               output_names=['conn_matrix_symm'],
                                                               function=diffconnectometry.collect_struct_mapping_outputs,
                                                               imports=import_list),
                                                  name="collect_struct_mapping_outputs_node")
    thresh_diff_node = pe.Node(niu.Function(input_names=['dens_thresh', 'thr', 'conn_model', 'network', 'ID',
                                                         'dir_path', 'mask', 'node_size', 'conn_matrix', 'parc',
                                                         'min_span_tree', 'disp_filt'],
                                            output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr',
                                                          'node_size', 'network', 'conn_model', 'mask'],
                                            function=thresholding.thresh_diff,
                                            imports=import_list), name="thresh_diff_node")
    if plot_switch is True:
        structural_plotting_node = pe.Node(niu.Function(input_names=['conn_matrix_symm', 'label_names', 'atlas_select',
                                                                     'ID', 'dwi_dir', 'network', 'parc', 'coords',
                                                                     'mask', 'dir_path', 'conn_model', 'thr',
                                                                     'node_size'],
                                                        function=plotting.structural_plotting,
                                                        imports=import_list),
                                           name="structural_plotting_node")
    outputnode = pe.JoinNode(interface=niu.IdentityInterface(fields=['est_path', 'thr', 'node_size', 'network',
                                                                     'conn_model']),
                             name='outputnode',
                             joinfield=['est_path', 'thr', 'node_size', 'network', 'conn_model'],
                             joinsource='thresh_diff_node')

    run_probtrackx2_node.interface.n_procs = 1
    run_probtrackx2_node.interface.mem_gb = 2
    run_probtrackx2_iterables = []
    iter_i = range(int(procmem[0]))
    run_probtrackx2_iterables.append(("i", iter_i))
    run_probtrackx2_node.iterables = run_probtrackx2_iterables
    if (multi_atlas is not None and user_atlas_list is None and uatlas_select is None) or (multi_atlas is None and atlas_select is None and user_atlas_list is not None):
        flexi_atlas = False
        if multi_atlas is not None and user_atlas_list is None:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("atlas_select", multi_atlas))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
        elif multi_atlas is None and user_atlas_list is not None:
            fetch_nodes_and_labels_node_iterables = []
            fetch_nodes_and_labels_node_iterables.append(("uatlas_select", user_atlas_list))
            fetch_nodes_and_labels_node.iterables = fetch_nodes_and_labels_node_iterables
    elif ((atlas_select is not None and uatlas_select is None) or (atlas_select is None and uatlas_select is not None)) and (multi_atlas is None and user_atlas_list is None):
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

    thresh_diff_node_iterables = []
    if multi_thr is True:
        iter_thresh = sorted(list(set([str(i) for i in np.round(np.arange(float(min_thr),
                                                                          float(max_thr), float(step_thr)),
                                                                decimals=2).tolist()] + [str(float(max_thr))])))
        thresh_diff_node_iterables.append(("thr", iter_thresh))
        if conn_model_list:
            thresh_diff_node_iterables.append(("conn_model", conn_model_list))
        else:
            thresh_diff_node_iterables.append(("conn_model", [conn_model]))
    else:
        if conn_model_list:
            thresh_diff_node_iterables.append(("conn_model", conn_model_list))
            thresh_diff_node_iterables.append(("thr", [thr]))
        else:
            thresh_diff_node_iterables.append(("conn_model", [conn_model]))
            thresh_diff_node_iterables.append(("thr", [thr]))
    thresh_diff_node.iterables = thresh_diff_node_iterables

    if node_size_list and parc is False:
        prep_nodes_node_iterables = []
        prep_nodes_node_iterables.append(("node_size", node_size_list))
        prep_nodes_node.iterables = prep_nodes_node_iterables
    # Connect nodes of workflow
    structural_connectometry_wf.connect([
        (inputnode, fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                     ('uatlas_select', 'uatlas_select'),
                                                     ('parc', 'parc'),
                                                     ('ref_txt', 'ref_txt'),
                                                     ('use_AAL_naming', 'use_AAL_naming')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc')]),
        (inputnode, fetch_nodes_and_labels_node, [('nodif_brain_mask_path', 'func_file')]),
        (fetch_nodes_and_labels_node, node_gen_node, [('coords', 'coords'),
                                                         ('label_names', 'label_names'),
                                                         ('dir_path', 'dir_path'),
                                                         ('parcel_list', 'parcel_list'),
                                                         ('par_max', 'par_max'),
                                                         ('networks_list', 'networks_list')]),
        (fetch_nodes_and_labels_node, prep_nodes_node, [('parcel_list', 'parcel_list')]),
        (node_gen_node, prep_nodes_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                          ('coords', 'coords'),
                                          ('label_names', 'label_names')
                                          ]),
        (inputnode, create_mni2diff_transforms_node, [('dwi_dir', 'dwi_dir')]),
        (fetch_nodes_and_labels_node, prep_nodes_node, [('dir_path', 'dir_path'),
                                                           ('atlas_select', 'atlas_select')]),
        (inputnode, prep_nodes_node, [('dwi_dir', 'dwi_dir'),
                                      ('node_size', 'node_size'),
                                      ('parc', 'parc'),
                                      ('mask', 'mask'),
                                      ('network', 'network')]),
        (inputnode, run_probtrackx2_node, [('dwi_dir', 'dwi_dir'),
                                           ('procmem', 'procmem'),
                                           ('num_total_samples', 'num_total_samples')]),
        (inputnode, create_seed_mask_file_node, [('node_size', 'node_size'), ('parc', 'parc'), ('network', 'network')]),
        (fetch_nodes_and_labels_node, create_seed_mask_file_node, [('dir_path', 'dir_path'),
                                                                      ('atlas_select', 'atlas_select')]),
        (create_seed_mask_file_node, run_probtrackx2_node, [('seeds_text', 'seeds_text'),
                                                            ('probtrackx_output_dir_path','probtrackx_output_dir_path')
                                                            ]),
        (create_seed_mask_file_node, collect_struct_mapping_outputs_node, [('probtrackx_output_dir_path',
                                                                            'probtrackx_output_dir_path')]),
        (fetch_nodes_and_labels_node, collect_struct_mapping_outputs_node, [('dir_path', 'dir_path')]),
        (fetch_nodes_and_labels_node, thresh_diff_node, [('dir_path', 'dir_path')]),
        (inputnode, collect_struct_mapping_outputs_node, [('dwi_dir', 'dwi_dir'),
                                                          ('parc', 'parc'),
                                                          ('network', 'network'),
                                                          ('procmem', 'procmem'),
                                                          ('ID', 'ID')]),
        (prep_nodes_node, collect_struct_mapping_outputs_node, [('node_size', 'node_size'),
                                                                ('seeds_dir', 'seeds_dir')]),
        (inputnode, thresh_diff_node, [('dens_thresh', 'dens_thresh'),
                                       ('thr', 'thr'),
                                       ('network', 'network'),
                                       ('conn_model', 'conn_model'),
                                       ('ID', 'ID'),
                                       ('mask', 'mask'),
                                       ('parc', 'parc'),
                                       ('min_span_tree', 'min_span_tree'),
                                       ('disp_filt', 'disp_filt')]),
        (prep_nodes_node, thresh_diff_node, [('node_size', 'node_size')]),
        (collect_struct_mapping_outputs_node, thresh_diff_node, [('conn_matrix_symm', 'conn_matrix')]),
        (thresh_diff_node, outputnode, [('est_path', 'est_path'),
                                        ('thr', 'thr'),
                                        ('node_size', 'node_size'),
                                        ('network', 'network'),
                                        ('conn_model', 'conn_model')])
        ])
    if no_segs is not True:
        structural_connectometry_wf.add_nodes([gen_anat_segs_node, prepare_masks_node])
        structural_connectometry_wf.connect([(create_mni2diff_transforms_node, gen_anat_segs_node, [('out_aff',
                                                                                                        'out_aff')]),
                                                (inputnode, gen_anat_segs_node, [('anat_loc', 'anat_loc')]),
                                                (inputnode, prepare_masks_node, [('dwi_dir', 'dwi_dir'),
                                                                                 ('mask', 'mask')]),
                                                (gen_anat_segs_node, prepare_masks_node, [('new_file_csf', 'csf_loc'),
                                                                                          ('mni_csf_loc', 'mni_csf_loc'),
                                                                                          ('new_file_wm', 'wm_mask_loc')]),
                                                (prepare_masks_node, run_probtrackx2_node, [('vent_CSF_diff_mask_path',
                                                                                             'vent_CSF_diff_mask_path'),
                                                                                            ('way_mask', 'way_mask')])
                                                ])
    if parc is False:
        structural_connectometry_wf.add_nodes([build_coord_list_node, reg_coords2diff_node, cleanup_tmp_nodes_node])
        structural_connectometry_wf.connect([(inputnode, build_coord_list_node, [('dwi_dir', 'dwi_dir')]),
                                                (fetch_nodes_and_labels_node, build_coord_list_node, [('coords',
                                                                                                          'coords')]),
                                                (prep_nodes_node, reg_coords2diff_node, [('seeds_dir', 'seeds_dir'),
                                                                                         ('node_size', 'node_size')]),
                                                (inputnode, reg_coords2diff_node, [('dwi_dir', 'dwi_dir')]),
                                                (build_coord_list_node, reg_coords2diff_node, [('coords', 'coords')]),
                                                (fetch_nodes_and_labels_node, cleanup_tmp_nodes_node, [('dir_path',
                                                                                                           'dir_path')]),
                                                (reg_coords2diff_node, cleanup_tmp_nodes_node, [('done_nodes',
                                                                                                 'done_nodes')]),
                                                (build_coord_list_node, cleanup_tmp_nodes_node, [('coords', 'coords')]),
                                                (prep_nodes_node, cleanup_tmp_nodes_node, [('seeds_dir', 'seeds_dir')]),
                                                (cleanup_tmp_nodes_node, create_seed_mask_file_node, [('seeds_list',
                                                                                                       'seeds_list')])
                                                ])
    else:
        structural_connectometry_wf.add_nodes([reg_parcels2diff_node])
        structural_connectometry_wf.connect([(inputnode, reg_parcels2diff_node, [('dwi_dir', 'dwi_dir')]),
                                                (prep_nodes_node, reg_parcels2diff_node, [('seeds_dir', 'seeds_dir')]),
                                                (reg_parcels2diff_node, create_seed_mask_file_node, [('seeds_list',
                                                                                                      'seeds_list')])
                                                ])
    if plot_switch is True:
        structural_connectometry_wf.add_nodes([structural_plotting_node])
        structural_connectometry_wf.connect([(collect_struct_mapping_outputs_node, structural_plotting_node,
                                                 [('conn_matrix_symm', 'conn_matrix_symm')]),
                                                (inputnode, structural_plotting_node, [('ID', 'ID'),
                                                                                       ('dwi_dir', 'dwi_dir'),
                                                                                       ('network', 'network'),
                                                                                       ('parc', 'parc'),
                                                                                       ('mask', 'mask'),
                                                                                       ('plot_switch', 'plot_switch')]),
                                                (thresh_diff_node, structural_plotting_node,
                                                 [('thr', 'thr'),
                                                  ('node_size', 'node_size'), ('conn_model', 'conn_model')]),
                                                (node_gen_node, structural_plotting_node,
                                                 [('label_names', 'label_names'),
                                                  ('coords', 'coords')]),
                                                (fetch_nodes_and_labels_node, structural_plotting_node,
                                                 [('dir_path', 'dir_path'),
                                                  ('atlas_select', 'atlas_select')])
                                                ])
    dwi_img = "%s%s" % (dwi_dir, '/dwi.nii.gz')
    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
    bvals = "%s%s" % (dwi_dir, '/bval')
    bvecs = "%s%s" % (dwi_dir, '/bvec')
    if '.bedpostX' not in dir_path and os.path.exists(dwi_img) and os.path.exists(bvals) and os.path.exists(bvecs) and os.path.exists(nodif_brain_mask_path):
        structural_connectometry_wf.disconnect(
            (inputnode, run_probtrackx2_node, [('dwi_dir', 'dwi_dir'),
                                               ('procmem', 'procmem'),
                                               ('num_total_samples', 'num_total_samples')]),
            (create_seed_mask_file_node, run_probtrackx2_node, [('seeds_text', 'seeds_text'),
                                                                ('probtrackx_output_dir_path',
                                                                 'probtrackx_output_dir_path')]),
            (prepare_masks_node, run_probtrackx2_node, [('vent_CSF_diff_mask_path', 'vent_CSF_diff_mask_path'),
                                                        ('way_mask', 'way_mask')]),
            (create_seed_mask_file_node, collect_struct_mapping_outputs_node, [('probtrackx_output_dir_path',
                                                                                'probtrackx_output_dir_path')]),
            (fetch_nodes_and_labels_node, collect_struct_mapping_outputs_node, [('dir_path', 'dir_path')]),
            (inputnode, collect_struct_mapping_outputs_node, [('dwi_dir', 'dwi_dir'),
                                                              ('parc', 'parc'),
                                                              ('network', 'network'),
                                                              ('procmem', 'procmem'),
                                                              ('ID', 'ID')]),
            (prep_nodes_node, collect_struct_mapping_outputs_node, [('node_size', 'node_size'),
                                                                    ('seeds_dir', 'seeds_dir')]),
            (collect_struct_mapping_outputs_node, thresh_diff_node, [('conn_matrix_symm', 'conn_matrix')]),
            (collect_struct_mapping_outputs_node, structural_plotting_node, [('conn_matrix_symm',
                                                                              'conn_matrix_symm')]))
        structural_connectometry_wf.connect(
            (inputnode, run_dipy_tracking_node, [('dwi_dir', 'dwi_dir'),
                                                 ('conn_model', 'conn_model'),
                                                 ('network', 'network'),
                                                 ('parc', 'parc')]),
            (create_seed_mask_file_node, run_dipy_tracking_node, [('seeds_text', 'seeds_text'),
                                                                  ('probtrackx_output_dir_path',
                                                                 'probtrackx_output_dir_path')]),
            (prepare_masks_node, run_dipy_tracking_node, [('way_mask', 'wm_mask')]),
            (prep_nodes_node, run_dipy_tracking_node, [('node_size', 'node_size')]),
            (fetch_nodes_and_labels_node, run_dipy_tracking_node, [('atlas_select', 'atlas_select'),
                                                                      ('dir_path', 'dir_path')]),
            (run_dipy_tracking_node, thresh_diff_node, [('conn_matrix', 'conn_matrix')]),
            (run_dipy_tracking_node, structural_plotting_node, [('conn_matrix', 'conn_matrix')]))

    if flexi_atlas is True:
        structural_connectometry_wf.disconnect([(inputnode, fetch_nodes_and_labels_node,
                                                    [('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')])
                                                   ])
        structural_connectometry_wf.connect([(flexi_atlas_source, fetch_nodes_and_labels_node,
                                                 [('atlas_select', 'atlas_select'), ('uatlas_select', 'uatlas_select')])
                                                ])
    structural_connectometry_wf.config['execution']['crashdump_dir'] = structural_connectometry_wf.base_directory
    structural_connectometry_wf.config['execution']['crashfile_format'] = 'txt'
    structural_connectometry_wf.config['logging']['log_directory'] = structural_connectometry_wf.base_directory
    structural_connectometry_wf.config['logging']['workflow_level'] = 'DEBUG'
    structural_connectometry_wf.config['logging']['utils_level'] = 'DEBUG'
    structural_connectometry_wf.config['logging']['interface_level'] = 'DEBUG'
    structural_connectometry_wf.config['execution']['display_variable'] = ':0'

    return structural_connectometry_wf
