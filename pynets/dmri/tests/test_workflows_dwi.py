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
import pkg_resources
import tempfile
import networkx as nx
from pynets.core import workflows
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
logger.setLevel(50)

base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))

# Test that each possible combination of inputs creates a workflow.

@pytest.mark.parametrize("subnet", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("thr,max_thr,min_thr,step_thr,multi_thr,thr_type",
    [
        pytest.param(1.0, None, None, None, False, 'MST'),
        pytest.param(None, 0.80, 0.20, 0.10, True, 'prop'),
    ]
)
@pytest.mark.parametrize("traversal", ['prob', ['det', 'prob']])
@pytest.mark.parametrize("min_length", [0, 5, [0, 5]])
@pytest.mark.parametrize("plot_switch", [True, False])
@pytest.mark.parametrize("mask", [None, f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_mask.nii.gz"])
@pytest.mark.parametrize("track_type,tiss_class,conn_model,conn_model_list",
    [
        pytest.param('local', 'wb', 'csd', None),
        pytest.param('local', 'wb', None, ['csa', 'sfm']),
    ]
)
@pytest.mark.parametrize("parc,node_radius,node_size_list,atlas,multi_atlas,parcellation,user_atlas_list",
    [
        pytest.param(False, None, [4, 8], None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(True, None, None, None, None, None, None),
        pytest.param(False, None, [4, 8], None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                     None),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, f"{base_dir}/miscellaneous/whole_brain_cluster_labels_"
        f"PCA200.nii.gz", None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], f"{base_dir}/miscellaneous/whole_"
        f"brain_cluster_labels_PCA200.nii.gz", None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200."
        f"nii.gz", None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], f"{base_dir}/miscellaneous/whole_brain_"
        f"cluster_labels_PCA200.nii.gz", None),
        pytest.param(False, 4, None, None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                     None),
        pytest.param(True, None, None, None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                     None),
        pytest.param(False, None, [4, 8], None, None, None, [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                                                             f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010',
                     None, None, [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                                  f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, None, [4, 8], None,
                     ['coords_dosenbach_2010', 'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, None, None, None, [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                                                        f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(True, None, None, None, None, None, [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                                                          f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"])
    ]
)
def test_struct_all(node_radius, parc, conn_model, conn_model_list, thr, max_thr, min_thr, step_thr,
                    multi_thr, thr_type, tiss_class, traversal, min_length,
                    track_type, node_size_list, atlas, multi_atlas,
                    parcellation, user_atlas_list, subnet, plot_switch, mask):
    """
    Test structural connectometry
    """

    anat_file = f"{base_dir}/003/anat/sub-003_T1w.nii.gz"
    fbval = f"{base_dir}/003/dmri/sub-003_dwi.bval"
    fbvec = f"{base_dir}/003/dmri/sub-003_dwi.bvec"
    dwi_file = f"{base_dir}/003/dmri/sub-003_dwi.nii.gz"

    roi = None
    ID = '25659_1'
    ref_txt = None
    nthreads = cpu_count()
    procmem = [int(nthreads), int(float(nthreads) * 2)]
    prune = 3
    use_AAL_naming = True
    plugin_type = 'MultiProc'
    norm = 6
    binary = False
    waymask = None
    outdir = f"{base_dir}/outputs"
    vox_size = '2mm'
    template_name = 'MNI152_T1'
    error_margin = 6
    maxcrossing = 3
    step_list = [0.1, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.8]
    curv_thr_list = [80, 50, 40, 40, 30, 10]

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

    if isinstance(min_length, list) and len(min_length) > 1:
        min_length_list = min_length
        min_length = None
    else:
        min_length_list = None

    if isinstance(traversal, list) and len(traversal) > 1:
        multi_directget = traversal
        traversal = None
    else:
        multi_directget = None

    if isinstance(error_margin, list) and len(error_margin) > 1:
        error_margin_list = error_margin
        error_margin = None
    else:
        error_margin_list = None

    # start_time = time.time()
    dmri_connectometry_wf = workflows.dmri_connectometry(ID, atlas, subnet, node_radius, roi, parcellation, plot_switch, parc, ref_txt,
                                               procmem, dwi_file, fbval, fbvec, anat_file, thr, dens_thresh,
                                               conn_model, user_atlas_list, multi_thr, multi_atlas, max_thr, min_thr,
                                               step_thr, node_size_list, conn_model_list, min_span_tree,
                                               use_AAL_naming, disp_filt, plugin_type, multi_nets, prune, mask, norm,
                                               binary, curv_thr_list, step_list,
                                               track_type, min_length, error_margin, maxcrossing, traversal, tiss_class,
                                               runtime_dict, execution_dict, multi_directget, template_name,
                                               vox_size, waymask, min_length_list, error_margin_list, outdir)

    # print("%s%s%s" % ('dmri_connectometry: ',
    #                   str(np.round(time.time() - start_time, 1)), 's'))

#    dmri_connectometry_wf.write_graph(graph2use='hierarchical', simple_form=False)
    assert nx.is_directed_acyclic_graph(dmri_connectometry_wf._graph) is True
    # plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    # out = dmri_connectometry_wf.run(plugin=plugin_type, plugin_args=plugin_args)
