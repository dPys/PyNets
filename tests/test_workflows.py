#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds
"""
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import pytest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)

base_dir = str(Path(__file__).parent/"examples")

# Test that each possible combination of inputs creates a workflow.
@pytest.mark.parametrize("hpass", [0.08, None])
@pytest.mark.parametrize("smooth", [2, [0, 4], None])
@pytest.mark.parametrize("conn_model", [['partcorr', 'sps'], 'partcorr'])
@pytest.mark.parametrize("network", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("thr,max_thr,min_thr,step_thr,multi_thr,thr_type",
    [
        pytest.param(1.0, None, None, None, False, 'prop'),
        pytest.param(None, 0.80, 0.20, 0.10, True, 'prop'),
    ]
)
@pytest.mark.parametrize("parc,node_size,node_size_list,atlas,multi_atlas,uatlas,user_atlas_list",
    [
        pytest.param(False, None, [4, 8], None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(True, None, None, None, None, None, None),
        pytest.param(False, None, [4, 8], None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'],
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, 4, None, None,
                     ['coords_dosenbach_2010', 'coords_power_2011'],
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, 4, None, None, None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(True, None, None, None, None,
                     f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, None, [4, 8], None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(False, 4, None, None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"]),
        pytest.param(True, None, None, None, None, None,
                     [f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz",
                      f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz"])
    ]
)
def test_func_all(hpass, smooth, parc, conn_model, uatlas, user_atlas_list, atlas, multi_atlas, network,
                  thr, max_thr, min_thr, step_thr, multi_thr, thr_type, node_size, node_size_list):
    """
    Test functional connectometry
    """
    import os
    import networkx as nx
    import ast
    import yaml
    import pkg_resources
    from pynets.core.workflows import fmri_connectometry
    from multiprocessing import cpu_count

    base_dir = str(Path(__file__).parent/"examples")
    conf = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_desc-confounds_regressors.tsv"
    func_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    anat_file = f"{base_dir}/BIDS/sub-0025427/ses-1/anat/sub-0025427_desc-preproc_T1w.nii.gz"
    mask = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    roi = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    ID = '0025427_1'
    ref_txt = None
    nthreads = cpu_count()
    procmem = [int(nthreads), int(float(nthreads) * 2)]
    prune = 3
    use_AAL_naming = True
    plugin_type = 'MultiProc'
    norm = 6
    binary = False
    local_corr = 'tcorr'
    outdir = base_dir + '/outputs'
    vox_size = '2mm'
    template_name = 'MNI152_T1'
    plot_switch = True
    k = None
    k_list = None
    k_clustering = 0
    clust_mask = None
    clust_mask_list = None
    clust_type = None
    clust_type_list = None

    with open(pkg_resources.resource_filename("pynets", "runconfig.yaml"), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        runtime_dict = {}
        execution_dict = {}
        for i in range(len(hardcoded_params['resource_dict'])):
            runtime_dict[list(hardcoded_params['resource_dict'][i].keys())[0]] = ast.literal_eval(list(
                hardcoded_params['resource_dict'][i].values())[0][0])
        for i in range(len(hardcoded_params['execution_dict'])):
            execution_dict[list(hardcoded_params['execution_dict'][i].keys())[0]] = list(
                hardcoded_params['execution_dict'][i].values())[0][0]

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

    if isinstance(network, list) and len(network) > 1:
        multi_nets = network
        network = None
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

    fmri_connectometry_wf = fmri_connectometry(func_file, ID, atlas, network, node_size, roi, thr, uatlas, conn_model,
                                               dens_thresh, conf, plot_switch, parc, ref_txt, procmem, multi_thr,
                                               multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_list,
                                               k_clustering, user_atlas_list, clust_mask_list, node_size_list,
                                               conn_model_list, min_span_tree, use_AAL_naming, smooth, smooth_list,
                                               disp_filt, prune, multi_nets, clust_type, clust_type_list, plugin_type,
                                               mask, norm, binary, anat_file, runtime_dict, execution_dict, hpass,
                                               hpass_list, template_name, vox_size, local_corr, outdir)

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
@pytest.mark.parametrize("mask", [f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz", None])
@pytest.mark.parametrize("roi", [f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz", None])
@pytest.mark.parametrize("network", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("parc,node_size,node_size_list,atlas,multi_atlas,uatlas,user_atlas_list",
    [
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None, None),
        pytest.param(True, None, None, None, None, None, None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None,
                     f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz", None),
        pytest.param(True, None, None, None, None,
                     f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz", None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None,
                     [f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz",
                      f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"]),
        pytest.param(True, None, None, None, None, None,
                     [f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz",
                      f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200.nii.gz"])
    ]
)
@pytest.mark.parametrize("plot_switch", [True, False])
def test_func_clust(parc, uatlas, user_atlas_list, k, k_list, k_clustering, clust_mask, clust_mask_list,
                    clust_type, clust_type_list, plot_switch, roi, mask, network, node_size, node_size_list, atlas,
                    multi_atlas):
    """
    Test functional connectometry with clustering
    """
    import os
    import networkx as nx
    import ast
    import yaml
    import pkg_resources
    from pynets.core.workflows import fmri_connectometry
    from multiprocessing import cpu_count

    base_dir = str(Path(__file__).parent/"examples")
    conf = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_desc-confounds_regressors.tsv"
    func_file = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz"
    anat_file = f"{base_dir}/BIDS/sub-0025427/ses-1/anat/sub-0025427_desc-preproc_T1w.nii.gz"
    ID = '0025427_1'
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
    outdir = base_dir + '/outputs'
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

    with open(pkg_resources.resource_filename("pynets", "runconfig.yaml"), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        runtime_dict = {}
        execution_dict = {}
        for i in range(len(hardcoded_params['resource_dict'])):
            runtime_dict[list(hardcoded_params['resource_dict'][i].keys())[0]] = ast.literal_eval(list(
                hardcoded_params['resource_dict'][i].values())[0][0])
        for i in range(len(hardcoded_params['execution_dict'])):
            execution_dict[list(hardcoded_params['execution_dict'][i].keys())[0]] = list(
                hardcoded_params['execution_dict'][i].values())[0][0]

    dens_thresh = False
    min_span_tree = False
    disp_filt = False

    if isinstance(network, list) and len(network) > 1:
        multi_nets = network
        network = None
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

    fmri_connectometry_wf = fmri_connectometry(func_file, ID, atlas, network, node_size, roi, thr, uatlas, conn_model,
                                               dens_thresh, conf, plot_switch, parc, ref_txt, procmem, multi_thr,
                                               multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_list,
                                               k_clustering, user_atlas_list, clust_mask_list, node_size_list,
                                               conn_model_list, min_span_tree, use_AAL_naming, smooth, smooth_list,
                                               disp_filt, prune, multi_nets, clust_type, clust_type_list, plugin_type,
                                               mask, norm, binary, anat_file, runtime_dict, execution_dict, hpass,
                                               hpass_list, template_name, vox_size, local_corr, outdir)

#    fmri_connectometry_wf.write_graph(graph2use='hierarchical', simple_form=False)
    assert nx.is_directed_acyclic_graph(fmri_connectometry_wf._graph) is True
    # plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    # out = fmri_connectometry_wf.run(plugin=plugin_type, plugin_args=plugin_args)


@pytest.mark.parametrize("network", ['Default', ['Default', 'Limbic'], None])
@pytest.mark.parametrize("thr,max_thr,min_thr,step_thr,multi_thr,thr_type",
    [
        pytest.param(1.0, None, None, None, False, 'prop'),
        pytest.param(None, 0.80, 0.20, 0.10, True, 'prop'),
    ]
)
@pytest.mark.parametrize("directget", ['prob', ['det', 'boot']])
@pytest.mark.parametrize("min_length", [0, 5, [0, 5]])
@pytest.mark.parametrize("track_type,tiss_class,conn_model,conn_model_list",
    [
        pytest.param('local', 'wb', 'csd', None),
        pytest.param('local', 'act', 'csd', None),
        pytest.param('particle', 'cmc', 'csd', None),
        pytest.param('local', 'wb', None, ['csa', 'csd']),
        pytest.param('local', 'act', None, ['csa', 'csd']),
        pytest.param('particle', 'cmc', None, ['csa', 'csd']),
    ]
)
@pytest.mark.parametrize("parc,node_size,node_size_list,atlas,multi_atlas,uatlas,user_atlas_list",
    [
        pytest.param(False, None, [4, 8], None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, None, None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], None, None),
        pytest.param(False, 4, None, None, None, None, None, marks=pytest.mark.xfail),
        pytest.param(True, None, None, None, None, None, None),
        pytest.param(False, None, [4, 8], None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(False, None, [4, 8], 'coords_dosenbach_2010', None, f"{base_dir}/miscellaneous/whole_brain_cluster_labels_"
        f"PCA200.nii.gz", None),
        pytest.param(False, None, [4, 8], None, ['coords_dosenbach_2010', 'coords_power_2011'], f"{base_dir}/miscellaneous/whole_"
        f"brain_cluster_labels_PCA200.nii.gz", None),
        pytest.param(False, 4, None, 'coords_dosenbach_2010', None, f"{base_dir}/miscellaneous/whole_brain_cluster_labels_PCA200."
        f"nii.gz", None),
        pytest.param(False, 4, None, None, ['coords_dosenbach_2010', 'coords_power_2011'], f"{base_dir}/miscellaneous/whole_brain_"
        f"cluster_labels_PCA200.nii.gz", None),
        pytest.param(False, 4, None, None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
        pytest.param(True, None, None, None, None, f"{base_dir}/miscellaneous/triple_net_ICA_overlap_3_sig_bin.nii.gz", None),
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
def test_struct_all(node_size, parc, conn_model, conn_model_list, thr, max_thr, min_thr,
                    step_thr, multi_thr, thr_type, tiss_class, directget, min_length, track_type, node_size_list,
                    atlas, multi_atlas, uatlas, user_atlas_list, network):
    """
    Test structural connectometry
    """
    import os
    import networkx as nx
    import ast
    import yaml
    import pkg_resources
    from pynets.core.workflows import dmri_connectometry
    from multiprocessing import cpu_count

    base_dir = str(Path(__file__).parent/"examples")
    dwi_file = f"{base_dir}/BIDS/sub-0025427/ses-1/dwi/final_bval.bval"
    fbval = f"{base_dir}/BIDS/sub-0025427/ses-1/dwi/final_bvec.bvec"
    fbvec = f"{base_dir}/BIDS/sub-0025427/ses-1/dwi/final_preprocessed_dwi.nii.gz"
    anat_file = f"{base_dir}/BIDS/sub-0025427/ses-1/anat/sub-0025427_desc-preproc_T1w.nii.gz"
    mask = f"{base_dir}/BIDS/sub-0025427/ses-1/func/sub-0025427_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    roi = None
    ID = '0025427_1'
    ref_txt = None
    nthreads = cpu_count()
    procmem = [int(nthreads), int(float(nthreads) * 2)]
    prune = 3
    use_AAL_naming = True
    plugin_type = 'MultiProc'
    norm = 6
    binary = False
    waymask = None
    outdir = base_dir + '/outputs'
    vox_size = '2mm'
    template_name = 'MNI152_T1'
    plot_switch = True
    target_samples = 1000

    with open(pkg_resources.resource_filename("pynets", "runconfig.yaml"), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        runtime_dict = {}
        execution_dict = {}
        maxcrossing = hardcoded_params['maxcrossing'][0]
        overlap_thr = hardcoded_params['overlap_thr'][0]
        step_list = hardcoded_params['step_list']
        curv_thr_list = hardcoded_params['curv_thr_list']
        for i in range(len(hardcoded_params['resource_dict'])):
            runtime_dict[list(hardcoded_params['resource_dict'][i].keys())[0]] = ast.literal_eval(list(
                hardcoded_params['resource_dict'][i].values())[0][0])
        for i in range(len(hardcoded_params['execution_dict'])):
            execution_dict[list(hardcoded_params['execution_dict'][i].keys())[0]] = list(
                hardcoded_params['execution_dict'][i].values())[0][0]

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

    if isinstance(network, list) and len(network) > 1:
        multi_nets = network
        network = None
    else:
        multi_nets = None

    if isinstance(min_length, list) and len(min_length) > 1:
        min_length_list = min_length
        min_length = None
    else:
        min_length_list = None

    if isinstance(directget, list) and len(directget) > 1:
        multi_directget = directget
        directget = None
    else:
        multi_directget = None

    dmri_connectometry_wf = dmri_connectometry(ID, atlas, network, node_size, roi, uatlas, plot_switch, parc, ref_txt,
                                               procmem, dwi_file, fbval, fbvec, anat_file, thr, dens_thresh,
                                               conn_model, user_atlas_list, multi_thr, multi_atlas, max_thr, min_thr,
                                               step_thr, node_size_list, conn_model_list, min_span_tree,
                                               use_AAL_naming, disp_filt, plugin_type, multi_nets, prune, mask, norm,
                                               binary, target_samples, curv_thr_list, step_list, overlap_thr,
                                               track_type, min_length, maxcrossing, directget, tiss_class,
                                               runtime_dict, execution_dict, multi_directget, template_name,
                                               vox_size, waymask, min_length_list, outdir)

#    dmri_connectometry_wf.write_graph(graph2use='hierarchical', simple_form=False)
    assert nx.is_directed_acyclic_graph(dmri_connectometry_wf._graph) is True
    # plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    # out = dmri_connectometry_wf.run(plugin=plugin_type, plugin_args=plugin_args)
