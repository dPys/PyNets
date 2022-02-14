#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017
"""
import os
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import pytest
import logging
import tempfile
import pkg_resources
import networkx as nx
from pynets.core import workflows
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
logger.setLevel(50)

base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

# Test that each possible combination of inputs creates a workflow.
@pytest.mark.parametrize("hpass", [0.08, None])
@pytest.mark.parametrize("smooth", [2, [0, 4], None])
@pytest.mark.parametrize("conn_model", [['partcorr', 'sps'], 'partcorr'])
@pytest.mark.parametrize("subnet", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("thr,max_thr,min_thr,step_thr,multi_thr,thr_type",
    [
        pytest.param(1.0, None, None, None, False, 'prop'),
        pytest.param(None, 0.80, 0.20, 0.10, True, 'prop'),
    ]
)
@pytest.mark.parametrize("plot_switch", [True, False])
@pytest.mark.parametrize("parc,node_radius,node_size_list,atlas,multi_atlas,"
                         "parcellation,user_atlas_list",
    [
        pytest.param(False, None, [4, 8], None, None, None, None,
                     marks=pytest.mark.xfail),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None,
                     None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010',
                                                 'coords_power_2011'], None,
                     None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
                     None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010',
                                            'coords_power_2011'], None, None),
        pytest.param(False, 4, None, None, None, None, None,
                     marks=pytest.mark.xfail),
        pytest.param(True, None, None, None, None, None, None),
        pytest.param(False, None, [4, 8], None, None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz",
                     None),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010',
                                                 'coords_power_2011'],
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(False, 4, None, None,
                     ['coords_dosenbach_2010', 'coords_power_2011'],
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(False, 4, None, None, None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(True, None, None, None, None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                     f"bin.nii.gz", None),
        pytest.param(False, None, [4, 8], None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010',
                                                 'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010',
                                            'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(False, 4, None, None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"]),
        pytest.param(True, None, None, None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_"
                      f"bin.nii.gz"])
    ]
)
def test_func_all(hpass, smooth, parc, conn_model, parcellation,
                  user_atlas_list, atlas, multi_atlas, subnet, thr, max_thr,
                  min_thr, step_thr, multi_thr, thr_type, node_radius,
                  node_size_list, plot_switch):
    """
    Test functional connectometry
    """

    conf = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_desc-confounds_regressors.tsv"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    anat_file = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-preproc_T1w.nii.gz"
    roi = pkg_resources.resource_filename(
        "pynets", "templates/rois/pDMN_3_bin.nii.gz")
    mask = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_" \
           f"mask.nii.gz"

    ID = '25659_1'
    ref_txt = None
    nthreads = cpu_count()
    procmem = [int(nthreads), int(float(nthreads) * 2)]
    prune = 3
    use_AAL_naming = True
    plugin_type = 'MultiProc'
    norm = 6
    binary = False
    local_corr = 'tcorr'
    outdir = f"{base_dir}/outputs"
    vox_size = '2mm'
    template_name = 'MNI152_T1'
    k = None
    k_list = None
    k_clustering = 0
    clust_mask = None
    clust_mask_list = None
    clust_type = None
    clust_type_list = None
    extract_strategy = 'mean'
    extract_strategy_list = None

    execution_dict = {'stop_on_first_crash': True,
                     'crashfile_format': 'txt',
                     'parameterize_dirs': False,
                     'display_variable': ':0',
                     'job_finished_timeout': 120,
                     'matplotlib_backend': 'Agg',
                     'use_relative_paths': True,
                     'keep_inputs': True,
                     'remove_unnecessary_outputs': False,
                     'remove_node_directories': False,
                     'raise_insufficient': False,
                     'poll_sleep_duration': 0,
                     'hash_method': 'timestamp',
                     'local_hash_check': False}

    runtime_dict = {'pass_meta_ins_node': (1, 1),
                 'pass_meta_outs_node': (1, 2),
                 'pass_meta_ins_multi_node': (1, 1),
                 'pass_meta_outs_multi_node': (1, 2),
                 'pass_meta_ins_func_node': (1, 1),
                 'pass_meta_outs_func_node': (1, 2),
                 'pass_meta_ins_struct_node': (1, 1),
                 'pass_meta_outs_struct_node': (1, 2),
                 'fetch_nodes_and_labels_node': (4, 8),
                 'save_nifti_parcels_node': (4, 4),
                 'gtab_node': (1, 1),
                 'save_coords_and_labels_node': (1, 1),
                 'orient_reslice_func_node': (2, 4),
                 'orient_reslice_mask_node': (1, 1),
                 'orient_reslice_uatlas_node': (1, 1),
                 'orient_reslice_anat_node': (1, 1),
                 'node_gen_node': (4, 8),
                 'prep_spherical_nodes_node': (4, 8),
                 'get_node_membership_node': (2, 6),
                 'orient_reslice_dwi_node': (2, 4),
                 'get_anisopwr_node': (2, 2),
                 'extract_ts_node': (4, 8),
                 'extract_ts_info_node': (1, 2),
                 'clustering_node': (4, 8),
                 'get_conn_matrix_node': (2, 6),
                 'thresh_func_node': (1, 2),
                 'thresh_diff_node': (1, 2),
                 'thresh_info_node': (1, 1),
                 'register_node': (4, 8),
                 'reg_nodes_node': (2, 4),
                 'RegisterParcellation2MNIFunc_node': (2, 4),
                 'get_fa_node': (2, 2),
                 'run_tracking_node': (4, 8),
                 'dsn_node': (1, 2),
                 'plot_all_node': (1, 2),
                 'streams2graph_node': (4, 6),
                 'build_multigraphs_node': (2, 8),
                 'plot_all_struct_func_node': (1, 2),
                 'mase_embedding_node': (2, 6),
                 'omni_embedding_node': (1, 2),
                 'omni_embedding_node_func': (1, 2),
                 'omni_embedding_node_struct': (1, 2),
                 'ase_embedding_node_func': (1, 2),
                 'ase_embedding_node_struct': (1, 2),
                 'join_iters_node_thr': (1, 4),
                 'join_iters_node_nets': (1, 1),
                 'join_iters_node_atlas': (1, 1),
                 'join_iters_node_ext_ts': (1, 1),
                 'join_iters_extract_ts_node': (1, 1),
                 'join_iters_node': (1, 4),
                 'join_iters_node_g': (1, 4),
                 'join_iters_prep_spheres_node': (1, 1),
                 'join_iters_get_conn_matrix_node': (1, 1),
                 'join_iters_run_track_node': (1, 1),
                 'clust_join_node': (1, 1),
                 'NetworkAnalysis': (1, 4),
                 'AggregateOutputs': (1, 3),
                 'load_mat_node': (1, 1),
                 'load_mat_ext_node': (1, 1),
                 'save_mat_thresholded_node': (1, 1),
                 'CombineOutputs': (1, 1)}

    if thr_type == 'dens_thresh':
        dens_thresh = True
        min_span_tree = False
        disp_filt = False
    elif thr_type == 'min_span_tree':
        dens_thresh = False
        min_span_tree = True
        disp_filt = False
    elif thr_type == 'disp_filt':
        dens_thresh = False
        min_span_tree = False
        disp_filt = True
    else:
        dens_thresh = False
        min_span_tree = False
        disp_filt = False

    if isinstance(subnet, list) and len(subnet) > 1:
        multi_nets = subnet
        subnet = None
    else:
        multi_nets = None

    if isinstance(conn_model, list) and len(conn_model) > 1:
        conn_model_list = conn_model
        conn_model = None
    else:
        conn_model_list = None

    if isinstance(smooth, list) and len(smooth) > 1:
        smooth_list = smooth
        smooth = None
    else:
        smooth_list = None

    if isinstance(hpass, list) and len(hpass) > 1:
        hpass_list = hpass
        hpass = None
    else:
        hpass_list = None

    # start_time = time.time()
    fmri_connectometry_wf = workflows.fmri_connectometry(
        func_file,
        ID, atlas, subnet, node_radius, roi, thr, parcellation,
        conn_model, dens_thresh, conf,
        plot_switch, parc, ref_txt, procmem,
        multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask,
        k_list, k_clustering, user_atlas_list, clust_mask_list, node_size_list,
        conn_model_list, min_span_tree, use_AAL_naming, smooth, smooth_list,
        disp_filt, prune, multi_nets, clust_type, clust_type_list,
        plugin_type, mask, norm, binary, anat_file, runtime_dict,
        execution_dict, hpass, hpass_list, template_name, vox_size,
        local_corr, extract_strategy, extract_strategy_list, outdir)
    # print("%s%s%s" % ('fmri_connectometry: ',
    #                   str(np.round(time.time() - start_time, 1)), 's'))

#    fmri_connectometry_wf.write_graph(graph2use='hierarchical', simple_form=False)
    assert nx.is_directed_acyclic_graph(fmri_connectometry_wf._graph) is True
    # plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    # out = fmri_connectometry_wf.run(plugin=plugin_type, plugin_args=plugin_args)


@pytest.mark.parametrize("k,k_list,k_clustering,clust_mask,clust_mask_list,clust_type,clust_type_list",
    [
        pytest.param(None, None, 0, None, None, None, None),
        pytest.param(100, None, 1, f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None, 'ward', None),
        pytest.param(100, None, 5, f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None, None, ['ward', 'rena']),
        pytest.param(100, None, 3, None, [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz",
                                          f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"],
                     'ward', None),
        pytest.param(100, None, 7, None, [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz",
                                          f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"],
                     None, ['ward', 'rena']),
        pytest.param(None, [100, 200], 2, f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None, 'ward', None),
        pytest.param(None, [100, 200], 6, f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None, None, ['ward', 'rena']),
        pytest.param(None, [100, 200], 8, None, [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz",
                                                 f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"],
                     None,
                     ['ward', 'rena']),
        pytest.param(None, [100, 200], 4, None, [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz",
                                                 f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"],
                     'ward', None),
    ]
)
@pytest.mark.parametrize("mask", [f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_mask.nii.gz",
                                  None])
@pytest.mark.parametrize("roi", [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None])
@pytest.mark.parametrize("subnet", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("parc,node_radius,node_size_list,atlas,multi_atlas,parcellation,user_atlas_list",
    [
        # pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None, None),
        pytest.param(True, None, None, None, None, None, None),
        # pytest.param(False, 4, None, 'coords_dosenbach_2010', None,
        #              f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz", None),
        pytest.param(True, None, None, None, None,
                     f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz", None),
        # pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
        #              [f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz",
        #               f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"]),
        pytest.param(True, None, None, None, None, None,
                     [f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz",
                      f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"])
    ]
)
@pytest.mark.parametrize("plot_switch", [True, False])
def test_func_clust(parc, parcellation,
                    user_atlas_list, k, k_list, k_clustering, clust_mask,
                    clust_mask_list, clust_type, clust_type_list, plot_switch,
                    roi, mask, subnet, node_radius, node_size_list, atlas,
                    multi_atlas):
    """
    Test functional connectometry with clustering
    """

    conf = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_desc-confounds_regressors.tsv"
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    anat_file = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-preproc_T1w.nii.gz"
    mask = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_" \
           f"mask.nii.gz"

    ID = '25659_1'
    parc = parc
    ref_txt = None
    nthreads = cpu_count()
    procmem = [int(nthreads), int(float(nthreads) * 2)]
    prune = 3
    use_AAL_naming = True
    plugin_type = 'MultiProc'
    norm = 6
    binary = False
    local_corr = 'tcorr'
    outdir = f"{base_dir}/outputs"
    vox_size = '2mm'
    template_name = 'MNI152_T1'
    hpass = None
    smooth = None
    conn_model = 'partcorr'
    thr = 1.0
    multi_thr = False
    max_thr = None
    min_thr = None
    step_thr = None
    extract_strategy = 'mean'
    extract_strategy_list = None

    execution_dict = {'stop_on_first_crash': True,
                     'crashfile_format': 'txt',
                     'parameterize_dirs': False,
                     'display_variable': ':0',
                     'job_finished_timeout': 120,
                     'matplotlib_backend': 'Agg',
                     'use_relative_paths': True,
                     'keep_inputs': True,
                     'remove_unnecessary_outputs': False,
                     'remove_node_directories': False,
                     'raise_insufficient': False,
                     'poll_sleep_duration': 0,
                     'hash_method': 'timestamp',
                     'local_hash_check': False}

    runtime_dict = {'pass_meta_ins_node': (1, 1),
                    'pass_meta_outs_node': (1, 2),
                    'pass_meta_ins_multi_node': (1, 1),
                    'pass_meta_outs_multi_node': (1, 2),
                    'pass_meta_ins_func_node': (1, 1),
                    'pass_meta_outs_func_node': (1, 2),
                    'pass_meta_ins_struct_node': (1, 1),
                    'pass_meta_outs_struct_node': (1, 2),
                    'fetch_nodes_and_labels_node': (4, 8),
                    'save_nifti_parcels_node': (4, 4),
                    'gtab_node': (1, 1),
                    'save_coords_and_labels_node': (1, 1),
                    'orient_reslice_func_node': (2, 4),
                    'orient_reslice_mask_node': (1, 1),
                    'orient_reslice_uatlas_node': (1, 1),
                    'orient_reslice_anat_node': (1, 1),
                    'node_gen_node': (4, 8),
                    'prep_spherical_nodes_node': (4, 8),
                    'get_node_membership_node': (2, 6),
                    'orient_reslice_dwi_node': (2, 4),
                    'get_anisopwr_node': (2, 2),
                    'extract_ts_node': (4, 8),
                    'extract_ts_info_node': (1, 2),
                    'clustering_node': (4, 8),
                    'get_conn_matrix_node': (2, 6),
                    'thresh_func_node': (1, 2),
                    'thresh_diff_node': (1, 2),
                    'thresh_info_node': (1, 1),
                    'register_node': (4, 8),
                    'reg_nodes_node': (2, 4),
                    'RegisterParcellation2MNIFunc_node': (2, 4),
                    'get_fa_node': (2, 2),
                    'run_tracking_node': (4, 8),
                    'dsn_node': (1, 2),
                    'plot_all_node': (1, 2),
                    'streams2graph_node': (4, 6),
                    'build_multigraphs_node': (2, 8),
                    'plot_all_struct_func_node': (1, 2),
                    'mase_embedding_node': (2, 6),
                    'omni_embedding_node': (1, 2),
                    'omni_embedding_node_func': (1, 2),
                    'omni_embedding_node_struct': (1, 2),
                    'ase_embedding_node_func': (1, 2),
                    'ase_embedding_node_struct': (1, 2),
                    'join_iters_node_thr': (1, 4),
                    'join_iters_node_nets': (1, 1),
                    'join_iters_node_atlas': (1, 1),
                    'join_iters_node_ext_ts': (1, 1),
                    'join_iters_extract_ts_node': (1, 1),
                    'join_iters_node': (1, 4),
                    'join_iters_node_g': (1, 4),
                    'join_iters_prep_spheres_node': (1, 1),
                    'join_iters_get_conn_matrix_node': (1, 1),
                    'join_iters_run_track_node': (1, 1),
                    'clust_join_node': (1, 1),
                    'NetworkAnalysis': (1, 4),
                    'AggregateOutputs': (1, 3),
                    'load_mat_node': (1, 1),
                    'load_mat_ext_node': (1, 1),
                    'save_mat_thresholded_node': (1, 1),
                    'CombineOutputs': (1, 1)}

    dens_thresh = False
    min_span_tree = False
    disp_filt = False

    if isinstance(subnet, list) and len(subnet) > 1:
        multi_nets = subnet
        subnet = None
    else:
        multi_nets = None

    if isinstance(conn_model, list) and len(conn_model) > 1:
        conn_model_list = conn_model
        conn_model = None
    else:
        conn_model_list = None

    if isinstance(smooth, list) and len(smooth) > 1:
        smooth_list = smooth
        smooth = None
    else:
        smooth_list = None

    if isinstance(hpass, list) and len(hpass) > 1:
        hpass_list = hpass
        hpass = None
    else:
        hpass_list = None

    # start_time = time.time()
    fmri_connectometry_wf = workflows.fmri_connectometry(
        func_file, ID, atlas, subnet, node_radius, roi, thr, parcellation,
        conn_model, dens_thresh, conf, plot_switch, parc, ref_txt, procmem,
        multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask,
        k_list, k_clustering, user_atlas_list, clust_mask_list,
        node_size_list, conn_model_list, min_span_tree, use_AAL_naming,
        smooth, smooth_list, disp_filt, prune, multi_nets, clust_type,
        clust_type_list, plugin_type, mask, norm, binary, anat_file,
        runtime_dict, execution_dict, hpass, hpass_list, template_name,
        vox_size, local_corr, extract_strategy, extract_strategy_list, outdir)
    # print("%s%s%s" % ('fmri_connectometry (clust): ',
    #                   str(np.round(time.time() - start_time, 1)), 's'))

#    fmri_connectometry_wf.write_graph(graph2use='hierarchical', simple_form=False)
    assert nx.is_directed_acyclic_graph(fmri_connectometry_wf._graph) is True
    # plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    # out = fmri_connectometry_wf.run(plugin=plugin_type, plugin_args=plugin_args)
