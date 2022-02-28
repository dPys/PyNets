#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import warnings
import sys
import gc
import os
import shutil
import itertools
import pkg_resources
import numpy as np
import os.path as op
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

if sys.platform.startswith('win') is False:
    pass

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def workflow_selector(
    func_file,
    ID,
    atlas,
    subnet,
    node_radius,
    roi,
    thr,
    parcellation,
    multi_nets,
    conn_model,
    dens_thresh,
    conf,
    plot_switch,
    dwi_file,
    anat_file,
    parc,
    ref_txt,
    procmem,
    multi_thr,
    multi_atlas,
    max_thr,
    min_thr,
    step_thr,
    k,
    clust_mask,
    k_list,
    k_clustering,
    user_atlas_list,
    clust_mask_list,
    prune,
    node_radii_list,
    conn_model_list,
    min_span_tree,
    verbose,
    plugin_type,
    use_parcel_naming,
    smooth,
    smooth_list,
    disp_filt,
    clust_type,
    clust_type_list,
    mask,
    norm,
    binary,
    fbval,
    fbvec,
    curv_thr_list,
    step_list,
    track_type,
    min_length,
    maxcrossing,
    error_margin,
    traversal,
    tiss_class,
    runtime_dict,
    execution_dict,
    embed,
    multi_traversal,
    multimodal,
    hpass,
    hpass_list,
    vox_size,
    multiplex,
    waymask,
    local_corr,
    min_length_list,
    error_margin_list,
    signal,
    signal_list,
    outdir,
    clean=True
):
    """A meta-interface for selecting modality-specific workflows to nest
    into a single-subject workflow"""
    from pynets.core.utils import load_runconfig
    from nipype import Workflow
    from pynets.statistics.individual import spectral
    from pynets.core.utils import pass_meta_ins, pass_meta_outs, \
        pass_meta_ins_multi

    import_list = [
        "import gc",
        "import sys",
        "import os",
        "import yaml",
        "import shutil",
        "import itertools",
        "import pkg_resources",
        "import os.path as op",
        "import numpy as np",
        "import networkx as nx",
        "import nibabel as nib",
        "import warnings",
        "import matplotlib",
        'warnings.filterwarnings("ignore")',
        'np.warnings.filterwarnings("ignore")',
        'warnings.simplefilter("ignore")',
        'matplotlib.use("Agg")',
        "from pathlib import Path",
        "from nipype.pipeline import engine as pe",
        "from nipype.interfaces import utility as niu",
    ]

    # Available functional and structural connectivity models
    hardcoded_params = load_runconfig()
    template_name = hardcoded_params["template"][0]
    embedding_methods = hardcoded_params["embed"]
    try:
        func_models = hardcoded_params["available_models"][
            "func_models"]
    except KeyError as e:
        print(e,
              "available functional models not successfully extracted"
              " from advanced.yaml"
              )
    try:
        dwi_models = hardcoded_params["available_models"][
            "dwi_models"]
    except KeyError as e:
        print(e,
              "available structural models not successfully extracted"
              " from advanced.yaml"
              )

    # Handle modality logic
    if (func_file is not None) and (dwi_file is not None):
        # print("Parsing multimodal models...")
        func_model_list = []
        dwi_model_list = []
        if conn_model_list is not None:
            for conn_model in conn_model_list:
                if conn_model in func_models:
                    func_model_list.append(conn_model)
                    conn_model_func = None
                if conn_model in dwi_models:
                    dwi_model_list.append(conn_model)
                    conn_model_dwi = None
            if len(func_model_list) == 1:
                conn_model_func = func_model_list[0]
                func_model_list = None
                # print(f"conn_model_func: {conn_model_func}")
            if len(dwi_model_list) == 1:
                conn_model_dwi = dwi_model_list[0]
                dwi_model_list = None
                # print(f"conn_model_dwi: {conn_model_dwi}")
            # if len(func_model_list) > 0:
            #     print(f"func_model_list: {func_model_list}")
            # if len(dwi_model_list) > 0:
            #     print(f"dwi_model_list: {dwi_model_list}")
        else:
            raise RuntimeError(
                "Multimodal fMRI-dMRI pipeline specified, but "
                "only one connectivity model specified.")

    elif (dwi_file is not None) and (func_file is None):
        print("Parsing diffusion models...")
        conn_model_dwi = conn_model
        dwi_model_list = conn_model_list
        conn_model_func = None
        func_model_list = None
        # print(f"dwi_model_list: {dwi_model_list}")
    elif (func_file is not None) and (dwi_file is None):
        print("Parsing functional models...")
        conn_model_func = conn_model
        func_model_list = conn_model_list
        conn_model_dwi = None
        dwi_model_list = None
        # print(f"func_model_list: {func_model_list}")

    # for each file input, delete corresponding t1w anatomical copies.
    if clean is True:
        file_list = [dwi_file, func_file, anat_file]
        for _file in file_list:
            if _file is not None:
                if op.isdir(f"{outdir}{'/anat_tmp'}"):
                    shutil.rmtree(f"{outdir}{'/anat_tmp'}")

    # Workflow 1: Structural connectome
    if dwi_file is not None:
        outdir_mod_struct = f"{outdir}/dwi"
        os.makedirs(outdir_mod_struct, exist_ok=True)
        sub_struct_wf = dmri_connectometry(
            ID,
            atlas,
            subnet,
            node_radius,
            roi,
            parcellation,
            plot_switch,
            parc,
            ref_txt,
            procmem,
            dwi_file,
            fbval,
            fbvec,
            anat_file,
            thr,
            dens_thresh,
            conn_model_dwi,
            user_atlas_list,
            multi_thr,
            multi_atlas,
            max_thr,
            min_thr,
            step_thr,
            node_radii_list,
            dwi_model_list,
            min_span_tree,
            use_parcel_naming,
            disp_filt,
            plugin_type,
            multi_nets,
            prune,
            mask,
            norm,
            binary,
            curv_thr_list,
            step_list,
            track_type,
            min_length,
            maxcrossing,
            error_margin,
            traversal,
            tiss_class,
            runtime_dict,
            execution_dict,
            multi_traversal,
            template_name,
            vox_size,
            waymask,
            min_length_list,
            error_margin_list,
            outdir_mod_struct,
        )
        if func_file is None:
            sub_func_wf = None
        sub_struct_wf._n_procs = procmem[0]
        sub_struct_wf._mem_gb = procmem[1]
        sub_struct_wf.n_procs = procmem[0]
        sub_struct_wf.mem_gb = procmem[1]
        if verbose is True:
            from nipype import config, logging

            cfg_v = dict(
                logging={
                    "workflow_level": "INFO",
                    "utils_level": "INFO",
                    "log_to_file": False,
                    "interface_level": "DEBUG",
                    "filemanip_level": "DEBUG"
                }
            )
            logging.update_logging(config)
            config.update_config(cfg_v)
            config.enable_resource_monitor()
    else:
        outdir_mod_struct = None

    # Workflow 2: Functional connectome
    if func_file is not None:
        outdir_mod_func = f"{outdir}/func"
        os.makedirs(outdir_mod_func, exist_ok=True)
        sub_func_wf = fmri_connectometry(
            func_file,
            ID,
            atlas,
            subnet,
            node_radius,
            roi,
            thr,
            parcellation,
            conn_model_func,
            dens_thresh,
            conf,
            plot_switch,
            parc,
            ref_txt,
            procmem,
            multi_thr,
            multi_atlas,
            max_thr,
            min_thr,
            step_thr,
            k,
            clust_mask,
            k_list,
            k_clustering,
            user_atlas_list,
            clust_mask_list,
            node_radii_list,
            func_model_list,
            min_span_tree,
            use_parcel_naming,
            smooth,
            smooth_list,
            disp_filt,
            prune,
            multi_nets,
            clust_type,
            clust_type_list,
            plugin_type,
            mask,
            norm,
            binary,
            anat_file,
            runtime_dict,
            execution_dict,
            hpass,
            hpass_list,
            template_name,
            vox_size,
            local_corr,
            signal,
            signal_list,
            outdir_mod_func,
        )
        if dwi_file is None:
            sub_struct_wf = None
        sub_func_wf._n_procs = procmem[0]
        sub_func_wf._mem_gb = procmem[1]
        sub_func_wf.n_procs = procmem[0]
        sub_func_wf.mem_gb = procmem[1]
        if verbose is True:
            from nipype import config, logging

            cfg_v = dict(
                logging={
                    "workflow_level": "INFO",
                    "utils_level": "INFO",
                    "log_to_file": False,
                    "interface_level": "DEBUG",
                    "filemanip_level": "DEBUG",
                }
            )

            logging.update_logging(config)
            config.update_config(cfg_v)
            config.enable_resource_monitor()
    else:
        outdir_mod_func = None

    # Create meta-workflow to organize graph simulation sets in prep for
    # analysis
    base_dirname = f"{'ensemble'}"
    meta_wf = Workflow(name=base_dirname)

    if verbose is True:
        from nipype import config, logging

        cfg_v = dict(
            logging={
                "workflow_level": "INFO",
                "utils_level": "INFO",
                "log_to_file": False,
                "interface_level": "DEBUG",
                "filemanip_level": "DEBUG",
            }
        )
        logging.update_logging(config)
        config.update_config(cfg_v)
        config.enable_resource_monitor()

    execution_dict["plugin_args"] = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "mem_thread",
    }
    execution_dict["plugin"] = str(plugin_type)
    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            meta_wf.config[key][setting] = value

    meta_inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "func_file",
                "ID",
                "atlas",
                "subnet",
                "thr",
                "node_radius",
                "roi",
                "parcellation",
                "multi_nets",
                "conn_model_func",
                "conn_model_dwi",
                "dens_thresh",
                "conf",
                "plot_switch",
                "dwi_file",
                "anat_file",
                "parc",
                "ref_txt",
                "procmem",
                "multi_thr",
                "multi_atlas",
                "max_thr",
                "min_thr",
                "step_thr",
                "k",
                "clust_mask",
                "k_list",
                "k_clustering",
                "user_atlas_list",
                "clust_mask_list",
                "prune",
                "node_radii_list",
                "func_model_list",
                "dwi_model_list",
                "min_span_tree",
                "verbose",
                "plugin_type",
                "use_parcel_naming",
                "smooth",
                "smooth_list",
                "disp_filt",
                "clust_type",
                "clust_type_list",
                "mask",
                "norm",
                "binary",
                "fbval",
                "fbvec",
                "curv_thr_list",
                "step_list",
                "track_type",
                "min_length",
                "maxcrossing",
                "error_margin",
                "traversal",
                "tiss_class",
                "embed",
                "multi_traversal",
                "multimodal",
                "hpass",
                "hpass_list",
                "template_name",
                "vox_size",
                "multiplex",
                "waymask",
                "local_corr",
                "min_length_list",
                "error_margin_list",
                "signal",
                "signal_list",
                "outdir_mod_func",
                "outdir_mod_struct",
            ]
        ),
        name="meta_inputnode",
    )
    meta_inputnode.inputs.func_file = func_file
    meta_inputnode.inputs.ID = ID
    meta_inputnode.inputs.atlas = atlas
    meta_inputnode.inputs.subnet = subnet
    meta_inputnode.inputs.thr = thr
    meta_inputnode.inputs.node_radius = node_radius
    meta_inputnode.inputs.roi = roi
    meta_inputnode.inputs.parcellation = parcellation
    meta_inputnode.inputs.multi_nets = multi_nets
    meta_inputnode.inputs.conn_model_func = conn_model_func
    meta_inputnode.inputs.conn_model_dwi = conn_model_dwi
    meta_inputnode.inputs.dens_thresh = dens_thresh
    meta_inputnode.inputs.conf = conf
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
    meta_inputnode.inputs.k_list = k_list
    meta_inputnode.inputs.k_clustering = k_clustering
    meta_inputnode.inputs.user_atlas_list = user_atlas_list
    meta_inputnode.inputs.clust_mask_list = clust_mask_list
    meta_inputnode.inputs.prune = prune
    meta_inputnode.inputs.node_radii_list = node_radii_list
    meta_inputnode.inputs.func_model_list = func_model_list
    meta_inputnode.inputs.dwi_model_list = dwi_model_list
    meta_inputnode.inputs.min_span_tree = min_span_tree
    meta_inputnode.inputs.verbose = verbose
    meta_inputnode.inputs.plugin_type = plugin_type
    meta_inputnode.inputs.use_parcel_naming = use_parcel_naming
    meta_inputnode.inputs.smooth = smooth
    meta_inputnode.inputs.smooth_list = smooth_list
    meta_inputnode.inputs.hpass = hpass
    meta_inputnode.inputs.hpass_list = hpass_list
    meta_inputnode.inputs.disp_filt = disp_filt
    meta_inputnode.inputs.clust_type = clust_type
    meta_inputnode.inputs.clust_type_list = clust_type_list
    meta_inputnode.inputs.mask = mask
    meta_inputnode.inputs.norm = norm
    meta_inputnode.inputs.binary = binary
    meta_inputnode.inputs.curv_thr_list = curv_thr_list
    meta_inputnode.inputs.step_list = step_list
    meta_inputnode.inputs.track_type = track_type
    meta_inputnode.inputs.min_length = min_length
    meta_inputnode.inputs.maxcrossing = maxcrossing
    meta_inputnode.inputs.error_margin = error_margin
    meta_inputnode.inputs.traversal = traversal
    meta_inputnode.inputs.tiss_class = tiss_class
    meta_inputnode.inputs.embed = embed
    meta_inputnode.inputs.multimodal = multimodal
    meta_inputnode.inputs.multi_traversal = multi_traversal
    meta_inputnode.inputs.template_name = template_name
    meta_inputnode.inputs.vox_size = vox_size
    meta_inputnode.inputs.multiplex = multiplex
    meta_inputnode.inputs.waymask = waymask
    meta_inputnode.inputs.local_corr = local_corr
    meta_inputnode.inputs.min_length_list = min_length_list
    meta_inputnode.inputs.error_margin_list = error_margin_list
    meta_inputnode.inputs.signal = signal
    meta_inputnode.inputs.signal_list = signal_list
    meta_inputnode.inputs.outdir_mod_func = outdir_mod_func
    meta_inputnode.inputs.outdir_mod_struct = outdir_mod_struct

    if multimodal is True:
        # Create input/output nodes
        # print("Running Multimodal Workflow...")
        pass_meta_ins_multi_node = pe.Node(
            niu.Function(
                input_names=[
                    "conn_model_func",
                    "est_path_func",
                    "network_func",
                    "thr_func",
                    "prune_func",
                    "ID_func",
                    "roi_func",
                    "norm_func",
                    "binary_func",
                    "conn_model_struct",
                    "est_path_struct",
                    "network_struct",
                    "thr_struct",
                    "prune_struct",
                    "ID_struct",
                    "roi_struct",
                    "norm_struct",
                    "binary_struct",
                ],
                output_names=[
                    "conn_model_iterlist",
                    "est_path_iterlist",
                    "network_iterlist",
                    "thr_iterlist",
                    "prune_iterlist",
                    "ID_iterlist",
                    "roi_iterlist",
                    "norm_iterlist",
                    "binary_iterlist",
                ],
                function=pass_meta_ins_multi,
            ),
            name="pass_meta_ins_multi_node",
        )
        pass_meta_ins_multi_node._mem_gb = 2
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
        meta_wf.connect(
            [
                (
                    meta_inputnode,
                    sub_struct_wf,
                    [
                        ("ID", "inputnode.ID"),
                        ("dwi_file", "inputnode.dwi_file"),
                        ("fbval", "inputnode.fbval"),
                        ("fbvec", "inputnode.fbvec"),
                        ("anat_file", "inputnode.anat_file"),
                        ("atlas", "inputnode.atlas"),
                        ("subnet", "inputnode.subnet"),
                        ("thr", "inputnode.thr"),
                        ("node_radius", "inputnode.node_radius"),
                        ("roi", "inputnode.roi"),
                        ("parcellation", "inputnode.parcellation"),
                        ("multi_nets", "inputnode.multi_nets"),
                        ("conn_model_dwi", "inputnode.conn_model"),
                        ("dens_thresh", "inputnode.dens_thresh"),
                        ("plot_switch", "inputnode.plot_switch"),
                        ("parc", "inputnode.parc"),
                        ("ref_txt", "inputnode.ref_txt"),
                        ("procmem", "inputnode.procmem"),
                        ("multi_thr", "inputnode.multi_thr"),
                        ("multi_atlas", "inputnode.multi_atlas"),
                        ("max_thr", "inputnode.max_thr"),
                        ("min_thr", "inputnode.min_thr"),
                        ("step_thr", "inputnode.step_thr"),
                        ("user_atlas_list", "inputnode.user_atlas_list"),
                        ("prune", "inputnode.prune"),
                        ("dwi_model_list", "inputnode.conn_model_list"),
                        ("min_span_tree", "inputnode.min_span_tree"),
                        ("use_parcel_naming", "inputnode.use_parcel_naming"),
                        ("disp_filt", "inputnode.disp_filt"),
                        ("mask", "inputnode.mask"),
                        ("norm", "inputnode.norm"),
                        ("binary", "inputnode.binary"),
                        ("curv_thr_list", "inputnode.curv_thr_list"),
                        ("step_list", "inputnode.step_list"),
                        ("track_type", "inputnode.track_type"),
                        ("min_length", "inputnode.min_length"),
                        ("maxcrossing", "inputnode.maxcrossing"),
                        ("error_margin", "inputnode.error_margin"),
                        ("traversal", "inputnode.traversal"),
                        ("tiss_class", "inputnode.tiss_class"),
                        ("multi_traversal", "inputnode.multi_traversal"),
                        ("template_name", "inputnode.template_name"),
                        ("vox_size", "inputnode.vox_size"),
                        ("waymask", "inputnode.waymask"),
                        ("min_length_list", "inputnode.min_length_list"),
                        ("error_margin_list", "inputnode.error_margin_list"),
                        ("outdir_mod_struct", "inputnode.outdir"),
                    ],
                )
            ]
        )
        meta_wf.connect(
            [
                (
                    meta_inputnode,
                    sub_func_wf,
                    [
                        ("func_file", "inputnode.func_file"),
                        ("ID", "inputnode.ID"),
                        ("anat_file", "inputnode.anat_file"),
                        ("atlas", "inputnode.atlas"),
                        ("subnet", "inputnode.subnet"),
                        ("thr", "inputnode.thr"),
                        ("node_radius", "inputnode.node_radius"),
                        ("roi", "inputnode.roi"),
                        ("parcellation", "inputnode.parcellation"),
                        ("multi_nets", "inputnode.multi_nets"),
                        ("conn_model_func", "inputnode.conn_model"),
                        ("dens_thresh", "inputnode.dens_thresh"),
                        ("conf", "inputnode.conf"),
                        ("plot_switch", "inputnode.plot_switch"),
                        ("parc", "inputnode.parc"),
                        ("ref_txt", "inputnode.ref_txt"),
                        ("procmem", "inputnode.procmem"),
                        ("multi_thr", "inputnode.multi_thr"),
                        ("multi_atlas", "inputnode.multi_atlas"),
                        ("max_thr", "inputnode.max_thr"),
                        ("min_thr", "inputnode.min_thr"),
                        ("step_thr", "inputnode.step_thr"),
                        ("k", "inputnode.k"),
                        ("clust_mask", "inputnode.clust_mask"),
                        ("k_list", "inputnode.k_list"),
                        ("k_clustering", "inputnode.k_clustering"),
                        ("user_atlas_list", "inputnode.user_atlas_list"),
                        ("clust_mask_list", "inputnode.clust_mask_list"),
                        ("prune", "inputnode.prune"),
                        ("func_model_list", "inputnode.conn_model_list"),
                        ("min_span_tree", "inputnode.min_span_tree"),
                        ("use_parcel_naming", "inputnode.use_parcel_naming"),
                        ("smooth", "inputnode.smooth"),
                        ("hpass", "inputnode.hpass"),
                        ("hpass_list", "inputnode.hpass_list"),
                        ("disp_filt", "inputnode.disp_filt"),
                        ("clust_type", "inputnode.clust_type"),
                        ("clust_type_list", "inputnode.clust_type_list"),
                        ("mask", "inputnode.mask"),
                        ("norm", "inputnode.norm"),
                        ("binary", "inputnode.binary"),
                        ("template_name", "inputnode.template_name"),
                        ("vox_size", "inputnode.vox_size"),
                        ("local_corr", "inputnode.local_corr"),
                        ("signal", "inputnode.signal"),
                        ("signal_list",
                         "inputnode.signal_list"),
                        ("outdir_mod_func", "inputnode.outdir"),
                    ],
                )
            ]
        )

        # Connect outputs of nested workflow to parent wf
        meta_wf.connect(
            [
                (
                    sub_func_wf.get_node("outputnode"),
                    pass_meta_ins_multi_node,
                    [
                        ("conn_model", "conn_model_func"),
                        ("est_path", "est_path_func"),
                        ("subnet", "network_func"),
                        ("thr", "thr_func"),
                        ("prune", "prune_func"),
                        ("ID", "ID_func"),
                        ("roi", "roi_func"),
                        ("norm", "norm_func"),
                        ("binary", "binary_func"),
                    ],
                ),
                (
                    sub_struct_wf.get_node("outputnode"),
                    pass_meta_ins_multi_node,
                    [
                        ("conn_model", "conn_model_struct"),
                        ("est_path", "est_path_struct"),
                        ("subnet", "network_struct"),
                        ("thr", "thr_struct"),
                        ("prune", "prune_struct"),
                        ("ID", "ID_struct"),
                        ("roi", "roi_struct"),
                        ("norm", "norm_struct"),
                        ("binary", "binary_struct"),
                    ],
                ),
            ]
        )

        # # Handle case in which the structural atlas to be used was generated
        # # from an fmri-clustering based parcellation.
        # TODO: This will almost unavoidably require the split/combine
        #  semantics of Nipype 2.0/Pydra because parcellation iterables must be
        #  dynamically combined across sub-workflows.
        # if k_clustering > 0:
        #     meta_wf.disconnect(
        #         [
        #             (
        #                 sub_struct_wf.get_node("flexi_atlas_source"),
        #                 sub_struct_wf.get_node("fetch_nodes_and_labels_"
        #                                        "node"),
        #                 [
        #                     ("parcellation", "parcellation"),
        #                     ("atlas", "atlas")
        #                 ],
        #             )
        #         ]
        #     )
        #     meta_wf.connect(
        #         [
        #             (
        #                 sub_func_wf.get_node("fetch_nodes_and_labels_node"),
        #                 sub_struct_wf,
        #                 [
        #                     ("parcellation", "fetch_nodes_and_labels_node.
        #                     parcellation")
        #                 ],
        #             ),
        #             (
        #                 sub_func_wf.get_node("clustering_node"),
        #                 sub_struct_wf,
        #                 [
        #                     ("clustering",
        #                      "fetch_nodes_and_labels_node.clustering"),
        #                     ("atlas", "fetch_nodes_and_labels_node.atlas")
        #                 ],
        #             )
        #         ]
        #     )
    else:
        # print("Running Unimodal Workflow...")

        if dwi_file:
            pass_meta_ins_struct_node = pe.Node(
                niu.Function(
                    input_names=[
                        "conn_model",
                        "est_path",
                        "subnet",
                        "thr",
                        "prune",
                        "ID",
                        "roi",
                        "norm",
                        "binary",
                    ],
                    output_names=[
                        "conn_model_iterlist",
                        "est_path_iterlist",
                        "network_iterlist",
                        "thr_iterlist",
                        "prune_iterlist",
                        "ID_iterlist",
                        "roi_iterlist",
                        "norm_iterlist",
                        "binary_iterlist",
                    ],
                    function=pass_meta_ins,
                ),
                name="pass_meta_ins_struct_node",
            )

            meta_wf.add_nodes([sub_struct_wf])
            meta_wf.get_node(sub_struct_wf.name)._n_procs = procmem[0]
            meta_wf.get_node(sub_struct_wf.name)._mem_gb = procmem[1]
            meta_wf.get_node(sub_struct_wf.name).n_procs = procmem[0]
            meta_wf.get_node(sub_struct_wf.name).mem_gb = procmem[1]

            meta_wf.connect(
                [
                    (
                        meta_inputnode,
                        sub_struct_wf,
                        [
                            ("ID", "inputnode.ID"),
                            ("dwi_file", "inputnode.dwi_file"),
                            ("fbval", "inputnode.fbval"),
                            ("fbvec", "inputnode.fbvec"),
                            ("anat_file", "inputnode.anat_file"),
                            ("atlas", "inputnode.atlas"),
                            ("subnet", "inputnode.subnet"),
                            ("thr", "inputnode.thr"),
                            ("node_radius", "inputnode.node_radius"),
                            ("roi", "inputnode.roi"),
                            ("parcellation", "inputnode.parcellation"),
                            ("multi_nets", "inputnode.multi_nets"),
                            ("conn_model_dwi", "inputnode.conn_model"),
                            ("dens_thresh", "inputnode.dens_thresh"),
                            ("plot_switch", "inputnode.plot_switch"),
                            ("parc", "inputnode.parc"),
                            ("ref_txt", "inputnode.ref_txt"),
                            ("procmem", "inputnode.procmem"),
                            ("multi_thr", "inputnode.multi_thr"),
                            ("multi_atlas", "inputnode.multi_atlas"),
                            ("max_thr", "inputnode.max_thr"),
                            ("min_thr", "inputnode.min_thr"),
                            ("step_thr", "inputnode.step_thr"),
                            ("user_atlas_list", "inputnode.user_atlas_list"),
                            ("prune", "inputnode.prune"),
                            ("dwi_model_list", "inputnode.conn_model_list"),
                            ("min_span_tree", "inputnode.min_span_tree"),
                            ("use_parcel_naming",
                             "inputnode.use_parcel_naming"),
                            ("disp_filt", "inputnode.disp_filt"),
                            ("mask", "inputnode.mask"),
                            ("norm", "inputnode.norm"),
                            ("binary", "inputnode.binary"),
                            ("curv_thr_list", "inputnode.curv_thr_list"),
                            ("step_list", "inputnode.step_list"),
                            ("track_type", "inputnode.track_type"),
                            ("min_length", "inputnode.min_length"),
                            ("maxcrossing", "inputnode.maxcrossing"),
                            ("error_margin", "inputnode.error_margin"),
                            ("traversal", "inputnode.traversal"),
                            ("tiss_class", "inputnode.tiss_class"),
                            ("multi_traversal", "inputnode.multi_traversal"),
                            ("template_name", "inputnode.template_name"),
                            ("vox_size", "inputnode.vox_size"),
                            ("waymask", "inputnode.waymask"),
                            ("min_length_list", "inputnode.min_length_list"),
                            ("error_margin_list",
                             "inputnode.error_margin_list"),
                            ("outdir_mod_struct", "inputnode.outdir"),
                        ],
                    )
                ]
            )

            # Connect outputs of nested workflow to parent wf
            meta_wf.connect(
                [
                    (
                        sub_struct_wf.get_node("outputnode"),
                        pass_meta_ins_struct_node,
                        [
                            ("conn_model", "conn_model"),
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("thr", "thr"),
                            ("prune", "prune"),
                            ("ID", "ID"),
                            ("roi", "roi"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    )
                ]
            )

        if func_file:
            pass_meta_ins_func_node = pe.Node(
                niu.Function(
                    input_names=[
                        "conn_model",
                        "est_path",
                        "subnet",
                        "thr",
                        "prune",
                        "ID",
                        "roi",
                        "norm",
                        "binary",
                    ],
                    output_names=[
                        "conn_model_iterlist",
                        "est_path_iterlist",
                        "network_iterlist",
                        "thr_iterlist",
                        "prune_iterlist",
                        "ID_iterlist",
                        "roi_iterlist",
                        "norm_iterlist",
                        "binary_iterlist",
                    ],
                    function=pass_meta_ins,
                ),
                name="pass_meta_ins_func_node",
            )

            meta_wf.add_nodes([sub_func_wf])
            meta_wf.get_node(sub_func_wf.name)._n_procs = procmem[0]
            meta_wf.get_node(sub_func_wf.name)._mem_gb = procmem[1]
            meta_wf.get_node(sub_func_wf.name).n_procs = procmem[0]
            meta_wf.get_node(sub_func_wf.name).mem_gb = procmem[1]
            meta_wf.connect(
                [
                    (
                        meta_inputnode,
                        sub_func_wf,
                        [
                            ("func_file", "inputnode.func_file"),
                            ("ID", "inputnode.ID"),
                            ("anat_file", "inputnode.anat_file"),
                            ("atlas", "inputnode.atlas"),
                            ("subnet", "inputnode.subnet"),
                            ("thr", "inputnode.thr"),
                            ("node_radius", "inputnode.node_radius"),
                            ("roi", "inputnode.roi"),
                            ("parcellation", "inputnode.parcellation"),
                            ("multi_nets", "inputnode.multi_nets"),
                            ("conn_model_func", "inputnode.conn_model"),
                            ("dens_thresh", "inputnode.dens_thresh"),
                            ("conf", "inputnode.conf"),
                            ("plot_switch", "inputnode.plot_switch"),
                            ("parc", "inputnode.parc"),
                            ("ref_txt", "inputnode.ref_txt"),
                            ("procmem", "inputnode.procmem"),
                            ("multi_thr", "inputnode.multi_thr"),
                            ("multi_atlas", "inputnode.multi_atlas"),
                            ("max_thr", "inputnode.max_thr"),
                            ("min_thr", "inputnode.min_thr"),
                            ("step_thr", "inputnode.step_thr"),
                            ("k", "inputnode.k"),
                            ("clust_mask", "inputnode.clust_mask"),
                            ("k_list", "inputnode.k_list"),
                            ("k_clustering", "inputnode.k_clustering"),
                            ("user_atlas_list", "inputnode.user_atlas_list"),
                            ("clust_mask_list", "inputnode.clust_mask_list"),
                            ("prune", "inputnode.prune"),
                            ("func_model_list", "inputnode.conn_model_list"),
                            ("min_span_tree", "inputnode.min_span_tree"),
                            ("use_parcel_naming",
                             "inputnode.use_parcel_naming"),
                            ("smooth", "inputnode.smooth"),
                            ("hpass", "inputnode.hpass"),
                            ("hpass_list", "inputnode.hpass_list"),
                            ("disp_filt", "inputnode.disp_filt"),
                            ("clust_type", "inputnode.clust_type"),
                            ("clust_type_list", "inputnode.clust_type_list"),
                            ("mask", "inputnode.mask"),
                            ("norm", "inputnode.norm"),
                            ("binary", "inputnode.binary"),
                            ("template_name", "inputnode.template_name"),
                            ("vox_size", "inputnode.vox_size"),
                            ("local_corr", "inputnode.local_corr"),
                            ("signal", "inputnode.signal"),
                            (
                                "signal_list",
                                "inputnode.signal_list",
                            ),
                            ("outdir_mod_func", "inputnode.outdir"),
                        ],
                    )
                ]
            )

            # Connect outputs of nested workflow to parent wf
            meta_wf.connect(
                [
                    (
                        sub_func_wf.get_node("outputnode"),
                        pass_meta_ins_func_node,
                        [
                            ("conn_model", "conn_model"),
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("thr", "thr"),
                            ("prune", "prune"),
                            ("ID", "ID"),
                            ("roi", "roi"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    )
                ]
            )

    pass_meta_outs_node = pe.Node(
        niu.Function(
            input_names=[
                "conn_model_iterlist",
                "est_path_iterlist",
                "network_iterlist",
                "thr_iterlist",
                "prune_iterlist",
                "ID_iterlist",
                "roi_iterlist",
                "norm_iterlist",
                "binary_iterlist",
            ],
            output_names=[
                "conn_model_iterlist",
                "est_path_iterlist",
                "network_iterlist",
                "thr_iterlist",
                "prune_iterlist",
                "ID_iterlist",
                "roi_iterlist",
                "norm_iterlist",
                "binary_iterlist",
            ],
            function=pass_meta_outs,
        ),
        name="pass_meta_outs_node",
    )
    pass_meta_outs_node._mem_gb = 2

    if func_file and not dwi_file:
        meta_wf.connect(
            [
                (
                    pass_meta_ins_func_node,
                    pass_meta_outs_node,
                    [
                        ("conn_model_iterlist", "conn_model_iterlist"),
                        ("est_path_iterlist", "est_path_iterlist"),
                        ("network_iterlist", "network_iterlist"),
                        ("thr_iterlist", "thr_iterlist"),
                        ("prune_iterlist", "prune_iterlist"),
                        ("ID_iterlist", "ID_iterlist"),
                        ("roi_iterlist", "roi_iterlist"),
                        ("norm_iterlist", "norm_iterlist"),
                        ("binary_iterlist", "binary_iterlist"),
                    ],
                )
            ]
        )
        if embed is True:
            if 'OMNI' in embedding_methods:
                omni_embedding_node_func = pe.Node(
                    niu.Function(
                        input_names=["est_path_iterlist"],
                        output_names=["out_paths_dwi", "out_paths_func"],
                        function=spectral.build_omnetome,
                    ),
                    name="omni_embedding_node_func",
                    imports=import_list
                )
                meta_wf.connect(
                    [
                        (
                            pass_meta_ins_func_node,
                            omni_embedding_node_func,
                            [("est_path_iterlist", "est_path_iterlist")],
                        ),
                    ]
                )
            if 'ASE' in embedding_methods:
                ase_embedding_node_func = pe.Node(
                    niu.Function(
                        input_names=["est_path_iterlist"],
                        output_names=["out_paths"],
                        function=spectral.build_asetomes,
                    ),
                    name="ase_embedding_node_func",
                    imports=import_list
                )
                meta_wf.connect(
                    [
                        (
                            pass_meta_ins_func_node,
                            ase_embedding_node_func,
                            [("est_path_iterlist", "est_path_iterlist")],
                        ),
                    ]
                )
    if dwi_file and not func_file:
        meta_wf.connect(
            [
                (
                    pass_meta_ins_struct_node,
                    pass_meta_outs_node,
                    [
                        ("conn_model_iterlist", "conn_model_iterlist"),
                        ("est_path_iterlist", "est_path_iterlist"),
                        ("network_iterlist", "network_iterlist"),
                        ("thr_iterlist", "thr_iterlist"),
                        ("prune_iterlist", "prune_iterlist"),
                        ("ID_iterlist", "ID_iterlist"),
                        ("roi_iterlist", "roi_iterlist"),
                        ("norm_iterlist", "norm_iterlist"),
                        ("binary_iterlist", "binary_iterlist"),
                    ],
                )
            ]
        )
        if embed is True:
            if 'OMNI' in embedding_methods:
                omni_embedding_node_struct = pe.Node(
                    niu.Function(
                        input_names=["est_path_iterlist"],
                        output_names=["out_paths_dwi", "out_paths_func"],
                        function=spectral.build_omnetome,
                    ),
                    name="omni_embedding_node_struct",
                    imports=import_list
                )
                meta_wf.connect(
                    [
                        (
                            pass_meta_ins_struct_node,
                            omni_embedding_node_struct,
                            [("est_path_iterlist", "est_path_iterlist")],
                        ),
                    ]
                )
            if 'ASE' in embedding_methods:
                ase_embedding_node_struct = pe.Node(
                    niu.Function(
                        input_names=["est_path_iterlist"],
                        output_names=["out_paths"],
                        function=spectral.build_asetomes,
                    ),
                    name="ase_embedding_node_struct",
                    imports=import_list
                )
                meta_wf.connect(
                    [
                        (
                            pass_meta_ins_struct_node,
                            ase_embedding_node_struct,
                            [("est_path_iterlist", "est_path_iterlist")],
                        ),
                    ]
                )
    if multimodal is True:
        # print("Multiplex connectome modeling is temporarily unsupported.")
        mase_embedding_node = pe.Node(
            niu.Function(
                input_names=["est_path_iterlist"],
                output_names=["out_paths"],
                function=spectral.build_masetome,
            ),
            name="mase_embedding_node",
            imports=import_list
        )

        # Multiplex magic happens in the meta-workflow space.
        meta_wf.connect(
            [
                (
                    pass_meta_ins_multi_node,
                    pass_meta_outs_node,
                    [
                        ("conn_model_iterlist", "conn_model_iterlist"),
                        ("est_path_iterlist", "est_path_iterlist"),
                        ("network_iterlist", "network_iterlist"),
                        ("thr_iterlist", "thr_iterlist"),
                        ("prune_iterlist", "prune_iterlist"),
                        ("ID_iterlist", "ID_iterlist"),
                        ("roi_iterlist", "roi_iterlist"),
                        ("norm_iterlist", "norm_iterlist"),
                        ("binary_iterlist", "binary_iterlist"),
                    ],
                )
            ]
        )

        if float(multiplex) > 0:
            from pynets.statistics.individual.multiplex import \
                build_multigraphs
            build_multigraphs_node = pe.Node(
                niu.Function(
                    input_names=["est_path_iterlist"],
                    output_names=[
                        "multigraph_list_all",
                        "graph_path_list_all",
                        "namer_dir",
                    ],
                    function=build_multigraphs,
                ),
                name="build_multigraphs_node",
                imports=import_list
            )
            meta_wf.connect(
                [
                    (
                        pass_meta_ins_multi_node,
                        build_multigraphs_node,
                        [("est_path_iterlist", "est_path_iterlist")],
                    ),
                ]
            )

            if plot_switch is True:
                from pynets.plotting.brain import plot_all_struct_func

                plot_all_struct_func_node = pe.MapNode(
                    niu.Function(
                        input_names=[
                            "mG_paths",
                            "namer_dir",
                            "name",
                            "modality_paths",
                            "metadata",
                        ],
                        function=plot_all_struct_func,
                    ),
                    iterfield=[
                        "mG_paths",
                        "namer_dir",
                        "name",
                        "modality_paths",
                        "metadata",
                    ],
                    name="plot_all_struct_func_node",
                    imports=import_list
                )
                meta_wf.connect(
                    [
                        (
                            build_multigraphs_node,
                            plot_all_struct_func_node,
                            [
                                ("namer_dir", "namer_dir"),
                                ("multigraph_list_all", "mG_paths"),
                                ("graph_path_list_all", "modality_paths"),
                            ],
                        )
                    ]
                )

            if embed is True and 'MASE' in embedding_methods:
                meta_wf.connect(
                    [
                        (
                            build_multigraphs_node,
                            mase_embedding_node,
                            [("graph_path_list_all", "est_path_iterlist")],
                        ),
                    ]
                )

    # Set resource restrictions at level of the meta wf
    if func_file:
        wf_selected = 'fmri'
        for node_name in sub_func_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node(
                    f"{wf_selected}{'.'}{node_name}"
                )._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node(
                    f"{wf_selected}{'.'}{node_name}"
                )._mem_gb = runtime_dict[node_name][1]
                try:
                    meta_wf.get_node(
                        f"{wf_selected}{'.'}{node_name}"
                    ).interface.n_procs = runtime_dict[node_name][0]
                    meta_wf.get_node(
                        f"{wf_selected}{'.'}{node_name}"
                    ).interface.mem_gb = runtime_dict[node_name][1]
                except BaseException:
                    continue

    if dwi_file:
        wf_selected = 'dmri'
        for node_name in sub_struct_wf.list_node_names():
            if node_name in runtime_dict:
                meta_wf.get_node(
                    f"{wf_selected}{'.'}{node_name}"
                )._n_procs = runtime_dict[node_name][0]
                meta_wf.get_node(
                    f"{wf_selected}{'.'}{node_name}"
                )._mem_gb = runtime_dict[node_name][1]
                try:
                    meta_wf.get_node(
                        f"{wf_selected}{'.'}{node_name}"
                    ).interface.n_procs = runtime_dict[node_name][0]
                    meta_wf.get_node(
                        f"{wf_selected}{'.'}{node_name}"
                    ).interface.mem_gb = runtime_dict[node_name][1]
                except BaseException:
                    continue

    gc.collect()
    return meta_wf


def dmri_connectometry(
    ID,
    atlas,
    subnet,
    node_radius,
    roi,
    parcellation,
    plot_switch,
    parc,
    ref_txt,
    procmem,
    dwi_file,
    fbval,
    fbvec,
    anat_file,
    thr,
    dens_thresh,
    conn_model,
    user_atlas_list,
    multi_thr,
    multi_atlas,
    max_thr,
    min_thr,
    step_thr,
    node_radii_list,
    conn_model_list,
    min_span_tree,
    use_parcel_naming,
    disp_filt,
    plugin_type,
    multi_nets,
    prune,
    mask,
    norm,
    binary,
    curv_thr_list,
    step_list,
    track_type,
    min_length,
    maxcrossing,
    error_margin,
    traversal,
    tiss_class,
    runtime_dict,
    execution_dict,
    multi_traversal,
    template_name,
    vox_size,
    waymask,
    min_length_list,
    error_margin_list,
    outdir,
):
    """
    A function interface for generating a dMRI connectometry nested workflow
    """
    from pynets.core import nodemaker, utils, thresholding, interfaces
    from pynets import dmri, registration
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        run_dsn = hardcoded_params['tracking']["DSN"][0]
    except FileNotFoundError as e:
        print(e, "Failed to parse runconfig.yaml")

    import_list = [
        "import matplotlib",
        "import warnings",
        "import sys",
        "import os",
        "import numpy as np",
        "import networkx as nx",
        "import nibabel as nib",
        'matplotlib.use("Agg")',
        'warnings.filterwarnings("ignore")'
    ]
    base_dirname = f"dmri"
    dmri_wf = pe.Workflow(name=base_dirname)

    if template_name == "MNI152_T1" or template_name == "colin27" or \
            template_name == "CN200":
        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{template_name}_brain_"
                      f"{vox_size}.nii.gz"
        )
        template_mask = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{template_name}_brain_mask_"
                      f"{vox_size}.nii.gz"
        )
        utils.check_template_loads(template, template_mask, template_name)
    else:
        [template, template_mask, _] = utils.get_template_tf(
            template_name, vox_size)

    if not op.isfile(template) or not op.isfile(template_mask):
        raise FileNotFoundError("Template or mask not found!")

    # Create input/output nodes
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "ID",
                "atlas",
                "subnet",
                "node_radius",
                "roi",
                "parcellation",
                "plot_switch",
                "parc",
                "ref_txt",
                "procmem",
                "dwi_file",
                "fbval",
                "fbvec",
                "anat_file",
                "thr",
                "dens_thresh",
                "conn_model",
                "user_atlas_list",
                "multi_thr",
                "multi_atlas",
                "max_thr",
                "min_thr",
                "step_thr",
                "min_span_tree",
                "use_parcel_naming",
                "disp_filt",
                "multi_nets",
                "prune",
                "mask",
                "norm",
                "binary",
                "template_name",
                "template",
                "template_mask",
                "curv_thr_list",
                "step_list",
                "track_type",
                "min_length",
                "maxcrossing",
                "error_margin",
                "traversal",
                "tiss_class",
                "vox_size",
                "multi_traversal",
                "waymask",
                "min_length_list",
                "error_margin_list",
                "outdir",
            ]
        ),
        name="inputnode",
    )

    in_dir = op.dirname(anat_file)
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas = atlas
    inputnode.inputs.subnet = subnet
    inputnode.inputs.node_radius = node_radius
    inputnode.inputs.roi = roi
    inputnode.inputs.parcellation = parcellation
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
    inputnode.inputs.node_radii_list = node_radii_list
    inputnode.inputs.conn_model_list = conn_model_list
    inputnode.inputs.min_span_tree = min_span_tree
    inputnode.inputs.use_parcel_naming = use_parcel_naming
    inputnode.inputs.disp_filt = disp_filt
    inputnode.inputs.multi_nets = multi_nets
    inputnode.inputs.prune = prune
    inputnode.inputs.mask = mask
    inputnode.inputs.norm = norm
    inputnode.inputs.binary = binary
    inputnode.inputs.template_name = template_name
    inputnode.inputs.template = template
    inputnode.inputs.template_mask = template_mask
    inputnode.inputs.curv_thr_list = curv_thr_list
    inputnode.inputs.step_list = step_list
    inputnode.inputs.track_type = track_type
    inputnode.inputs.min_length = min_length
    inputnode.inputs.maxcrossing = maxcrossing
    inputnode.inputs.error_margin = error_margin
    inputnode.inputs.traversal = traversal
    inputnode.inputs.tiss_class = tiss_class
    inputnode.inputs.plugin_type = plugin_type
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.multi_traversal = multi_traversal
    inputnode.inputs.waymask = waymask
    inputnode.inputs.min_length_list = min_length_list
    inputnode.inputs.error_margin_list = error_margin_list
    inputnode.inputs.outdir = outdir

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('outdir: ', outdir))
    # print("%s%s" % ('dwi_file: ', dwi_file))
    # print("%s%s" % ('fbval: ', fbval))
    # print("%s%s" % ('fbvec: ', fbvec))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('atlas: ', atlas))
    # print("%s%s" % ('subnet: ', subnet))
    # print("%s%s" % ('node_radius: ', node_radius))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('parcellation: ', parcellation))
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
    # print("%s%s" % ('node_radii_list: ', node_radii_list))
    # print("%s%s" % ('mask: ', mask))
    # print("%s%s" % ('multi_nets: ', multi_nets))
    # print("%s%s" % ('norm: ', norm))
    # print("%s%s" % ('binary: ', binary))
    # print("%s%s" % ('template_name: ', template_name))
    # print("%s%s" % ('template: ', template))
    # print("%s%s" % ('template_mask: ', template_mask))
    # print("%s%s" % ('multi_traversal: ', multi_traversal))
    # print("%s%s" % ('min_length_list: ', min_length_list))
    # print("%s%s" % ('error_margin_list: ', error_margin_list))
    # print('\n\n\n\n\n')

    # Create function nodes
    orient_reslice_dwi_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size", "bvecs"],
            output_names=["outfile", "bvecs"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_dwi_node",
    )

    orient_reslice_dwi_node._n_procs = runtime_dict[
        "orient_reslice_dwi_node"
    ][0]
    orient_reslice_dwi_node._mem_gb = runtime_dict[
        "orient_reslice_dwi_node"
    ][1]

    orient_reslice_anat_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size"],
            output_names=["outfile"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_anat_node",
    )

    fetch_nodes_and_labels_node = pe.Node(
        interfaces.FetchNodesLabels(), name="fetch_nodes_and_labels_node"
    )

    fetch_nodes_and_labels_node.synchronize = True

    if parc is False:
        prep_spherical_nodes_node = pe.Node(
            niu.Function(
                input_names=["coords", "node_radius", "template_mask"],
                output_names=["parcel_list", "par_max", "node_radius", "parc"],
                function=nodemaker.create_spherical_roi_volumes,
                imports=import_list
            ),
            name="prep_spherical_nodes_node",
        )

        if node_radii_list:
            prep_spherical_nodes_node.inputs.node_radius = None
            prep_spherical_nodes_node.iterables = [
                ("node_radius", node_radii_list)]
        else:
            dmri_wf.connect(
                [(inputnode, prep_spherical_nodes_node,
                  [("node_radius", "node_radius")])
                 ])

        prep_spherical_nodes_node.synchronize = True

    save_nifti_parcels_node = pe.Node(
        niu.Function(
            input_names=["ID", "dir_path", "subnet", "net_parcels_map_nifti",
                         "vox_size"],
            output_names=["net_parcels_nii_path"],
            function=utils.save_nifti_parcels_map,
            imports=import_list
        ),
        name="save_nifti_parcels_node",
    )

    # Generate nodes
    if roi is not None:
        # Masking case
        node_gen_node = pe.Node(
            niu.Function(
                input_names=[
                    "roi",
                    "coords",
                    "parcel_list",
                    "labels",
                    "dir_path",
                    "ID",
                    "parc",
                    "atlas",
                    "parcellation",
                    "vox_size"
                ],
                output_names=[
                    "net_parcels_map_nifti",
                    "coords",
                    "labels",
                    "atlas",
                    "parcellation",
                    "dir_path",
                ],
                function=nodemaker.node_gen_masking,
                imports=import_list
            ),
            name="node_gen_node",
        )
        dmri_wf.connect(
            [
                (inputnode, node_gen_node, [("vox_size", "vox_size")]),
            ]
        )
    else:
        # Non-masking case
        node_gen_node = pe.Node(
            niu.Function(
                input_names=[
                    "coords",
                    "parcel_list",
                    "labels",
                    "dir_path",
                    "ID",
                    "parc",
                    "atlas",
                    "parcellation",
                ],
                output_names=[
                    "net_parcels_map_nifti",
                    "coords",
                    "labels",
                    "atlas",
                    "parcellation",
                    "dir_path",
                ],
                function=nodemaker.node_gen,
                imports=import_list
            ),
            name="node_gen_node",
        )
    node_gen_node._n_procs = runtime_dict["node_gen_node"][0]
    node_gen_node._mem_gb = runtime_dict["node_gen_node"][1]

    gtab_node = pe.Node(dmri.interfaces.MakeGtabBmask(), name="gtab_node")

    get_fa_node = pe.Node(
        niu.Function(
            input_names=["gtab_file", "dwi_file", "B0_mask"],
            output_names=["fa_path", "B0_mask", "gtab_file", "dwi_file"],
            function=dmri.estimation.tens_mod_fa_est,
            imports=import_list
        ),
        name="get_fa_node",
    )

    get_anisopwr_node = pe.Node(
        niu.Function(
            input_names=["gtab_file", "dwi_file", "B0_mask"],
            output_names=["anisopwr_path", "B0_mask", "gtab_file", "dwi_file"],
            function=dmri.estimation.create_anisopowermap,
            imports=import_list
        ),
        name="get_anisopwr_node",
    )

    register_node = pe.Node(registration.interfaces.RegisterDWI(in_dir=in_dir),
                            name="register_node")
    register_node._n_procs = runtime_dict["register_node"][0]
    register_node._mem_gb = runtime_dict["register_node"][1]

    # Check orientation and resolution
    orient_reslice_parcellation_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size"],
            output_names=["outfile"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_parcellation_node",
    )

    reg_nodes_node = pe.Node(
        registration.interfaces.RegisterAtlasDWI(),
        name="reg_nodes_node")
    reg_nodes_node._n_procs = runtime_dict["reg_nodes_node"][0]
    reg_nodes_node._mem_gb = runtime_dict["reg_nodes_node"][1]

    run_tracking_node = pe.Node(dmri.interfaces.Tracking(),
                                name="run_tracking_node")
    run_tracking_node.synchronize = True
    run_tracking_node._n_procs = runtime_dict["run_tracking_node"][0]
    run_tracking_node._mem_gb = runtime_dict["run_tracking_node"][1]

    # Set tracking iterable combinations
    if conn_model_list or multi_traversal or min_length_list:
        run_tracking_node_iterables = []

        if conn_model_list and not multi_traversal and min_length_list:
            conn_model_min_length_combo = list(
                itertools.product(min_length_list, conn_model_list)
            )
            min_length_list = [i[0] for i in conn_model_min_length_combo]
            conn_model_list = [i[1] for i in conn_model_min_length_combo]
            run_tracking_node_iterables.append(("min_length", min_length_list))
            run_tracking_node_iterables.append(("conn_model", conn_model_list))
            dmri_wf.connect(
                [(inputnode, run_tracking_node, [("traversal", "traversal")])]
            )
        elif conn_model_list and multi_traversal and not min_length_list:
            conn_model_traversal_combo = list(
                itertools.product(multi_traversal, conn_model_list)
            )
            multi_traversal = [i[0] for i in conn_model_traversal_combo]
            conn_model_list = [i[1] for i in conn_model_traversal_combo]
            run_tracking_node_iterables.append(("traversal", multi_traversal))
            run_tracking_node_iterables.append(("conn_model", conn_model_list))
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("min_length", "min_length")])]
            )
        elif not conn_model_list and multi_traversal and min_length_list:
            min_length_traversal_combo = list(
                itertools.product(multi_traversal, min_length_list)
            )
            multi_traversal = [i[0] for i in min_length_traversal_combo]
            min_length_list = [i[1] for i in min_length_traversal_combo]
            run_tracking_node_iterables.append(("min_length", min_length_list))
            run_tracking_node_iterables.append(("traversal", multi_traversal))
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("conn_model", "conn_model")])]
            )
        elif conn_model_list and not multi_traversal and not min_length_list:
            run_tracking_node_iterables.append(("conn_model", conn_model_list))
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("traversal", "traversal")])]
            )
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("min_length", "min_length")])]
            )
        elif not conn_model_list and not multi_traversal and min_length_list:
            run_tracking_node_iterables.append(("min_length", min_length_list))
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("conn_model", "conn_model")])]
            )
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("traversal", "traversal")])]
            )
        elif not conn_model_list and multi_traversal and not min_length_list:
            run_tracking_node_iterables.append(("traversal", multi_traversal))
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("conn_model", "conn_model")])]
            )
            dmri_wf.connect(
                [(inputnode, run_tracking_node,
                  [("min_length", "min_length")])]
            )
        elif conn_model_list and multi_traversal and min_length_list:
            all_combo = list(
                itertools.product(
                    multi_traversal,
                    min_length_list,
                    conn_model_list))
            multi_traversal = [i[0] for i in all_combo]
            min_length_list = [i[1] for i in all_combo]
            conn_model_list = [i[2] for i in all_combo]
            run_tracking_node_iterables.append(("min_length", min_length_list))
            run_tracking_node_iterables.append(("traversal", multi_traversal))
            run_tracking_node_iterables.append(("conn_model", conn_model_list))
        run_tracking_node.iterables = run_tracking_node_iterables
    else:
        dmri_wf.connect(
            [
                (
                    inputnode,
                    run_tracking_node,
                    [
                        ("conn_model", "conn_model"),
                        ("traversal", "traversal"),
                        ("min_length", "min_length"),
                    ],
                )
            ]
        )

    dsn_node = pe.Node(
        niu.Function(
            input_names=[
                "streams",
                "fa_path",
                "ap_path",
                "dir_path",
                "track_type",
                "conn_model",
                "subnet",
                "node_radius",
                "dens_thresh",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "labels_im_file",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "atlas_t1w",
                "basedir_path",
                "curv_thr_list",
                "step_list",
                "traversal",
                "min_length",
                "t1w_brain"
            ],
            output_names=[
                "streams_t1w",
                "dir_path",
                "track_type",
                "conn_model",
                "subnet",
                "node_radius",
                "dens_thresh",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "atlas_for_streams",
                "traversal",
                "warped_fa",
                "min_length",
            ],
            function=registration.register.direct_streamline_norm,
            imports=import_list
        ),
        name="dsn_node",
    )
    dsn_node.synchronize = True

    streams2graph_node = pe.Node(
        niu.Function(
            input_names=[
                "atlas_for_streams",
                "streams",
                "dir_path",
                "track_type",
                "conn_model",
                "subnet",
                "node_radius",
                "dens_thresh",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "traversal",
                "warped_fa",
                "min_length",
                "error_margin",
            ],
            output_names=[
                "atlas_for_streams",
                "streams",
                "conn_matrix",
                "track_type",
                "dir_path",
                "conn_model",
                "subnet",
                "node_radius",
                "dens_thresh",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "traversal",
                "min_length",
                "error_margin"
            ],
            function=dmri.estimation.streams2graph,
            imports=import_list
        ),
        name="streams2graph_node",
    )
    streams2graph_node.synchronize = True
    streams2graph_node._n_procs = runtime_dict["streams2graph_node"][0]
    streams2graph_node._mem_gb = runtime_dict["streams2graph_node"][1]

    if error_margin_list:
        streams2graph_node.iterables = [("error_margin", error_margin_list)]
    else:
        dmri_wf.connect(
            [
                (
                    inputnode, streams2graph_node,
                    [("error_margin", "error_margin")],
                )
            ]
        )

    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "est_path",
                "thr",
                "subnet",
                "prune",
                "ID",
                "roi",
                "conn_model",
                "norm",
                "binary",
            ]
        ),
        name="outputnode",
    )

    # Set atlas iterables and logic for multiple atlas useage
    if (multi_atlas is not None and user_atlas_list is None and parcellation is
        None) or (
            multi_atlas is None and atlas is None and user_atlas_list is not
            None):
        # print('\n\n\n\n')
        # print('No flexi-atlas1')
        # print('\n\n\n\n')
        atlas_iters = []
        flexi_atlas = False
        if multi_atlas:
            atlas_iters.append(("atlas", multi_atlas))
        elif user_atlas_list:
            atlas_iters.append(("parcellation", user_atlas_list))
        fetch_nodes_and_labels_node.iterables = atlas_iters

    elif (
        (atlas is not None and parcellation is None) or
        (atlas is None and parcellation is not None)
    ) and (multi_atlas is None and user_atlas_list is None):
        # print('\n\n\n\n')
        # print('No flexi-atlas2')
        # print('\n\n\n\n')
        flexi_atlas = False
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(
            niu.IdentityInterface(
                fields=[
                    "atlas",
                    "parcellation"]),
            name="flexi_atlas_source")
        flexi_atlas_source.synchronize = True
        if multi_atlas is not None and user_atlas_list is not None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: multiple nilearn atlases + multiple user
            # atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", len(user_atlas_list) * [None] + multi_atlas),
                ("parcellation", user_atlas_list + len(multi_atlas) * [None]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif multi_atlas is not None and parcellation is\
                not None and user_atlas_list is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single user atlas + multiple nilearn
            # atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", multi_atlas + [None]),
                ("parcellation", len(multi_atlas) * [None] + [parcellation]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and user_atlas_list is not None and multi_atlas\
                is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + multiple user
         # atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", len(user_atlas_list) * [None] + [atlas]),
                ("parcellation", user_atlas_list + [None]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif (
            atlas is not None
            and parcellation is not None
            and user_atlas_list is None
            and multi_atlas is None
        ):
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + single user atlas')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", [atlas, None]),
                ("parcellation", [None, parcellation]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables

    # Connect flexi_atlas inputs to definition node
    if flexi_atlas is True:
        dmri_wf.add_nodes([flexi_atlas_source])
        dmri_wf.connect(
            [
                (
                    flexi_atlas_source,
                    fetch_nodes_and_labels_node,
                    [("parcellation", "parcellation"), ("atlas", "atlas")],
                )
            ]
        )
    else:
        dmri_wf.connect(
            [
                (
                    inputnode,
                    fetch_nodes_and_labels_node,
                    [("atlas", "atlas"), ("parcellation", "parcellation")],
                )
            ]
        )

    # ROI case
    if roi is not None:
        orient_reslice_roi_node = pe.Node(
            niu.Function(
                input_names=["infile", "outdir", "vox_size"],
                output_names=["outfile"],
                function=registration.utils.orient_reslice,
                imports=import_list
            ),
            name="orient_reslice_roi_node",
        )

        register_roi_node = pe.Node(registration.interfaces.RegisterROIDWI(),
                                    name="register_roi_node")
        dmri_wf.connect([(inputnode,
                                        orient_reslice_roi_node,
                                        [("roi",
                                          "infile"),
                                         ("outdir",
                                          "outdir"),
                                         ("vox_size",
                                            "vox_size")],
                                        ),
                                       (orient_reslice_roi_node,
                                        register_roi_node),
                                       [("outfile", "roi")]
                                       ])
    save_coords_and_labels_node = pe.Node(
        niu.Function(
            input_names=["coords", "labels", "dir_path", "subnet", "indices"],
            function=utils.save_coords_and_labels_to_json,
            imports=import_list
        ),
        name="save_coords_and_labels_node",
    )

    # RSN case
    if subnet or multi_nets:
        get_node_membership_node = pe.Node(
            niu.Function(
                input_names=[
                    "subnet",
                    "infile",
                    "coords",
                    "labels",
                    "parc",
                    "parcel_list",
                    "perc_overlap",
                    "error",
                ],
                output_names=[
                    "net_coords",
                    "net_parcel_list",
                    "net_labels",
                    "subnet"],
                function=nodemaker.get_node_membership,
                imports=import_list
            ),
            name="get_node_membership_node",
        )
        get_node_membership_node._n_procs = runtime_dict[
            "get_node_membership_node"][0]
        get_node_membership_node._mem_gb = runtime_dict[
            "get_node_membership_node"][1]

        if multi_nets:
            get_node_membership_iterables = []
            get_node_membership_node.inputs.subnet = None
            get_node_membership_iterables.append(("subnet", multi_nets))
            get_node_membership_node.iterables = get_node_membership_iterables

        dmri_wf.connect(
            [
                (
                    inputnode,
                    get_node_membership_node,
                    [("subnet", "subnet"), ("template", "infile"),
                     ("parc", "parc")],
                ),
                (get_node_membership_node, run_tracking_node,
                 [("subnet", "subnet")]),
                (
                    get_node_membership_node,
                    save_nifti_parcels_node,
                    [("subnet", "subnet")],
                ),
                (
                    save_nifti_parcels_node,
                    dsn_node,
                    [("net_parcels_nii_path", "parcellation")],
                ),
                (
                    get_node_membership_node,
                    reg_nodes_node,
                    [("subnet", "subnet")],
                ),
                (
                    get_node_membership_node,
                    save_coords_and_labels_node,
                    [("subnet", "subnet")],
                ),
            ]
        )

        if parc is False:
            dmri_wf.connect(
                [
                    (
                        get_node_membership_node,
                        prep_spherical_nodes_node,
                        [("net_coords", "coords")],
                    ),
                    (
                        fetch_nodes_and_labels_node,
                        get_node_membership_node,
                        [
                            ("coords", "coords"),
                            ("labels", "labels"),
                            ("networks_list", "networks_list"),
                            ("parcel_list", "parcel_list"),
                        ],
                    ),
                    (
                        prep_spherical_nodes_node,
                        node_gen_node,
                        [("parc", "parc"), ("parcel_list", "parcel_list")],
                    ),
                    (
                        get_node_membership_node,
                        node_gen_node,
                        [("net_coords", "coords"), ("net_labels", "labels")],
                    ),
                ]
            )
        else:
            dmri_wf.connect(
                [
                    (
                        fetch_nodes_and_labels_node,
                        get_node_membership_node,
                        [
                            ("coords", "coords"),
                            ("labels", "labels"),
                            ("parcel_list", "parcel_list"),
                            ("par_max", "par_max"),
                            ("networks_list", "networks_list"),
                        ],
                    ),
                    (
                        get_node_membership_node,
                        node_gen_node,
                        [
                            ("net_coords", "coords"),
                            ("net_labels", "labels"),
                            ("net_parcel_list", "parcel_list"),
                        ],
                    ),
                ]
            )
    else:
        dmri_wf.connect(
            [
                (inputnode, save_nifti_parcels_node,
                 [("subnet", "subnet")]),
                (inputnode, run_tracking_node, [("subnet", "subnet")]),
                (run_tracking_node, dsn_node,
                 [("parcellation", "parcellation")]),
                (inputnode, reg_nodes_node, [("subnet", "subnet")]),
                (
                    fetch_nodes_and_labels_node,
                    node_gen_node,
                    [("coords", "coords"), ("labels", "labels")],
                ),
            ]
        )
        if parc is False:
            dmri_wf.connect(
                [
                    (
                        prep_spherical_nodes_node,
                        node_gen_node,
                        [("parcel_list", "parcel_list"), ("parc", "parc")],
                    ),
                    (
                        fetch_nodes_and_labels_node,
                        prep_spherical_nodes_node,
                        [("coords", "coords")],
                    ),
                ]
            )
        else:
            dmri_wf.connect(
                [
                    (
                        fetch_nodes_and_labels_node,
                        node_gen_node,
                        [("parcel_list", "parcel_list")],
                    ),
                ]
            )

    if parc is False:
        # register_node.inputs.simple = True
        dmri_wf.connect(
            [
                (
                    inputnode,
                    prep_spherical_nodes_node,
                    [("template_mask", "template_mask")],
                ),
                (
                    fetch_nodes_and_labels_node,
                    prep_spherical_nodes_node,
                    [("dir_path", "dir_path")],
                ),
                (
                    prep_spherical_nodes_node,
                    reg_nodes_node,
                    [("node_radius", "node_radius")],
                ),
                (reg_nodes_node, run_tracking_node,
                 [("node_radius", "node_radius")]),
                (node_gen_node, reg_nodes_node,
                 [("parcellation", "parcellation")]),
            ]
        )
    else:
        dmri_wf.connect(
            [
                (inputnode, run_tracking_node,
                 [("node_radius", "node_radius")]),
                (inputnode, node_gen_node, [("parc", "parc")]),
                (inputnode, reg_nodes_node, [("node_radius", "node_radius")]),
                (
                    inputnode,
                    orient_reslice_parcellation_node,
                    [("vox_size", "vox_size")],
                ),
                (
                    fetch_nodes_and_labels_node,
                    orient_reslice_parcellation_node,
                    [("parcellation", "infile"), ("dir_path", "outdir")],
                ),
                (
                    orient_reslice_parcellation_node,
                    reg_nodes_node,
                    [("outfile", "parcellation")],
                ),
            ]
        )
    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout
    # joins
    map_fields = [
        "conn_model",
        "dir_path",
        "conn_matrix",
        "node_radius",
        "dens_thresh",
        "subnet",
        "ID",
        "roi",
        "min_span_tree",
        "disp_filt",
        "parc",
        "prune",
        "thr",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "track_type",
        "atlas_for_streams",
        "streams",
        "traversal",
        "min_length",
        "error_margin",
    ]

    map_connects = [
        ("conn_model", "conn_model"),
        ("dir_path", "dir_path"),
        ("conn_matrix", "conn_matrix"),
        ("node_radius", "node_radius"),
        ("dens_thresh", "dens_thresh"),
        ("ID", "ID"),
        ("roi", "roi"),
        ("min_span_tree", "min_span_tree"),
        ("disp_filt", "disp_filt"),
        ("parc", "parc"),
        ("prune", "prune"),
        ("subnet", "subnet"),
        ("thr", "thr"),
        ("atlas", "atlas"),
        ("parcellation", "parcellation"),
        ("labels", "labels"),
        ("coords", "coords"),
        ("norm", "norm"),
        ("binary", "binary"),
        ("track_type", "track_type"),
        ("atlas_for_streams", "atlas_for_streams"),
        ("streams", "streams"),
        ("traversal", "traversal"),
        ("min_length", "min_length"),
        ("error_margin", "error_margin")
    ]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(
        niu.IdentityInterface(fields=map_fields), name="thr_info_node"
    )

    # Set iterables for thr on thresh_func, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(
            list(
                set(
                    [
                        str(i)
                        for i in np.round(
                            np.arange(float(min_thr), float(max_thr),
                                      float(step_thr)),
                            decimals=2,
                        ).tolist()
                    ]
                    + [str(float(max_thr))]
                )
            )
        )
        thr_info_node.iterables = ("thr", iter_thresh)
        thr_info_node.synchronize = True
    else:
        thr_info_node.iterables = ("thr", [thr])

    # Joinsource logic for atlas varieties
    if user_atlas_list or multi_atlas or flexi_atlas is True:
        if flexi_atlas is True:
            atlas_join_source = flexi_atlas_source
        else:
            atlas_join_source = fetch_nodes_and_labels_node
    else:
        atlas_join_source = None

    # Connect all streams2graph_node outputs to the "thr_info" node
    dmri_wf.connect(
        [
            (
                streams2graph_node,
                thr_info_node,
                [x for x in map_connects if x != ("thr", "thr")],
            )
        ]
    )

    # Begin joinnode chaining logic
    if (
        conn_model_list
        or multi_traversal
        or node_radii_list
        or user_atlas_list
        or multi_atlas
        or flexi_atlas is True
        or multi_thr is True
        or min_length_list
    ):
        if user_atlas_list or multi_atlas or flexi_atlas is True:
            join_iters_node = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_node_atlas",
                joinsource=atlas_join_source,
                joinfield=map_fields,
            )
        else:
            join_iters_node = pe.Node(
                niu.IdentityInterface(
                    fields=map_fields),
                name="join_iters_node")

        if (
            not conn_model_list
            and not multi_traversal
            and not min_length_list
            and (node_radii_list and parc is False)
        ):
            # print('Node extraction iterables...')
            join_iters_node_prep_spheres = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_prep_spheres_node",
                joinsource=prep_spherical_nodes_node,
                joinfield=map_fields,
            )
            if error_margin_list is not None:
                join_iters_node_em = pe.JoinNode(
                    niu.IdentityInterface(fields=map_fields),
                    name="join_iters_node_em",
                    joinsource=streams2graph_node,
                    joinfield=map_fields,
                )
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_em, map_connects),
                        (join_iters_node_em, join_iters_node_prep_spheres,
                         map_connects)
                    ]
                )
            else:
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_prep_spheres,
                         map_connects),
                    ]
                )
            dmri_wf.connect(
                [
                    (join_iters_node_prep_spheres, join_iters_node,
                     map_connects),
                ]
            )
        elif (
            conn_model_list or multi_traversal or min_length_list
        ) and not node_radii_list:
            # print('Multiple connectivity models...')
            join_iters_node_run_track = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_run_track_node",
                joinsource=run_tracking_node,
                joinfield=map_fields,
            )
            if error_margin_list is not None:
                join_iters_node_em = pe.JoinNode(
                    niu.IdentityInterface(fields=map_fields),
                    name="join_iters_node_em",
                    joinsource=streams2graph_node,
                    joinfield=map_fields,
                )
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_em, map_connects),
                        (join_iters_node_em, join_iters_node_run_track,
                         map_connects)
                    ]
                )
            else:
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_run_track,
                         map_connects),
                    ]
                )
            dmri_wf.connect(
                [
                    (join_iters_node_run_track, join_iters_node, map_connects),
                ]
            )
        elif (
            not conn_model_list
            and not multi_traversal
            and not min_length_list
            and not node_radii_list
        ):
            # print('No connectivity model or node extraction iterables...')
            if error_margin_list is not None:
                join_iters_node_em = pe.JoinNode(
                    niu.IdentityInterface(fields=map_fields),
                    name="join_iters_node_em",
                    joinsource=streams2graph_node,
                    joinfield=map_fields,
                )
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_em, map_connects),
                        (join_iters_node_em, join_iters_node,
                         map_connects)
                    ]
                )
            else:
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node, map_connects),
                    ]
                )
        elif (conn_model_list or multi_traversal or min_length_list) or (
            node_radii_list and parc is False
        ):
            # print('Connectivity model and node extraction iterables...')
            join_iters_node_prep_spheres = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_node_prep_spheres",
                joinsource=prep_spherical_nodes_node,
                joinfield=map_fields,
            )
            join_iters_node_run_track = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_run_track_node",
                joinsource=run_tracking_node,
                joinfield=map_fields,
            )
            if error_margin_list is not None:
                join_iters_node_em = pe.JoinNode(
                    niu.IdentityInterface(fields=map_fields),
                    name="join_iters_node_em",
                    joinsource=streams2graph_node,
                    joinfield=map_fields,
                )
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_em, map_connects),
                        (join_iters_node_em, join_iters_node_run_track,
                         map_connects)
                    ]
                )
            else:
                dmri_wf.connect(
                    [
                        (thr_info_node, join_iters_node_run_track,
                         map_connects),
                    ]
                )

            dmri_wf.connect([(join_iters_node_run_track,
                                            join_iters_node_prep_spheres,
                                            map_connects,
                                            ),
                                           (join_iters_node_prep_spheres,
                                            join_iters_node,
                                            map_connects),
                                           ])
        else:
            raise RuntimeError("\nUnknown join context.")

        no_iters = False
    else:
        if not multi_nets:
            # Minimal case of no iterables
            print("\nNo structural connectometry iterables...")
        join_iters_node = pe.Node(
            niu.IdentityInterface(fields=map_fields), name="join_iters_node"
        )
        dmri_wf.connect(
            [
                (
                    streams2graph_node,
                    join_iters_node,
                    [x for x in map_connects if x != ("thr", "thr")],
                ),
                (thr_info_node, join_iters_node, [("thr", "thr")]),
            ]
        )
        no_iters = True

    # Create final thresh_diff node that performs the thresholding
    thr_struct_fields = [
        "dens_thresh",
        "thr",
        "conn_matrix",
        "conn_model",
        "subnet",
        "ID",
        "dir_path",
        "roi",
        "node_radius",
        "min_span_tree",
        "disp_filt",
        "parc",
        "prune",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "track_type",
        "atlas_for_streams",
        "streams",
        "traversal",
        "min_length",
        "error_margin"
    ]
    thr_struct_iter_fields = [
        "edge_threshold",
        "est_path",
        "thr",
        "node_radius",
        "subnet",
        "conn_model",
        "roi",
        "prune",
        "ID",
        "dir_path",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "track_type",
        "atlas_for_streams",
        "streams",
        "traversal",
        "min_length",
        "error_margin"
    ]

    if no_iters is True:
        thresh_diff_node = pe.Node(
            niu.Function(
                input_names=thr_struct_fields,
                output_names=[
                    "edge_threshold",
                    "est_path",
                    "thr",
                    "node_radius",
                    "subnet",
                    "conn_model",
                    "roi",
                    "prune",
                    "ID",
                    "dir_path",
                    "atlas",
                    "parcellation",
                    "labels",
                    "coords",
                    "norm",
                    "binary",
                    "track_type",
                    "atlas_for_streams",
                    "streams",
                    "traversal",
                    "min_length",
                    "error_margin"
                ],
                function=thresholding.thresh_struct,
                imports=import_list
            ),
            name="thresh_diff_node",
        )
    else:
        thresh_diff_node = pe.MapNode(
            niu.Function(
                input_names=thr_struct_fields,
                output_names=[
                    "edge_threshold",
                    "est_path",
                    "thr",
                    "node_radius",
                    "subnet",
                    "conn_model",
                    "roi",
                    "prune",
                    "ID",
                    "dir_path",
                    "atlas",
                    "parcellation",
                    "labels",
                    "coords",
                    "norm",
                    "binary",
                    "track_type",
                    "atlas_for_streams",
                    "streams",
                    "traversal",
                    "min_length",
                    "error_margin"
                ],
                function=thresholding.thresh_struct,
                imports=import_list
            ),
            name="thresh_diff_node",
            iterfield=thr_struct_fields,
            nested=True,
        )
        thresh_diff_node.synchronize = True

    dmri_wf.connect(
        [
            (
                join_iters_node,
                thresh_diff_node,
                [
                    ("dens_thresh", "dens_thresh"),
                    ("thr", "thr"),
                    ("conn_matrix", "conn_matrix"),
                    ("conn_model", "conn_model"),
                    ("subnet", "subnet"),
                    ("ID", "ID"),
                    ("dir_path", "dir_path"),
                    ("roi", "roi"),
                    ("node_radius", "node_radius"),
                    ("min_span_tree", "min_span_tree"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("labels", "labels"),
                    ("coords", "coords"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                    ("track_type", "track_type"),
                    ("atlas_for_streams", "atlas_for_streams"),
                    ("streams", "streams"),
                    ("traversal", "traversal"),
                    ("min_length", "min_length"),
                    ("error_margin", "error_margin")
                ],
            )
        ]
    )

    if multi_thr is True:
        join_iters_node_thr = pe.JoinNode(
            niu.IdentityInterface(fields=thr_struct_iter_fields),
            name="join_iters_node_thr",
            joinsource=thr_info_node,
            joinfield=thr_struct_iter_fields,
        )
        dmri_wf.connect(
            [
                (
                    thresh_diff_node,
                    join_iters_node_thr,
                    [
                        ("edge_threshold", "edge_threshold"),
                        ("est_path", "est_path"),
                        ("thr", "thr"),
                        ("node_radius", "node_radius"),
                        ("subnet", "subnet"),
                        ("conn_model", "conn_model"),
                        ("roi", "roi"),
                        ("prune", "prune"),
                        ("ID", "ID"),
                        ("dir_path", "dir_path"),
                        ("atlas", "atlas"),
                        ("parcellation", "parcellation"),
                        ("labels", "labels"),
                        ("coords", "coords"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("track_type", "track_type"),
                        ("atlas_for_streams", "atlas_for_streams"),
                        ("streams", "streams"),
                        ("traversal", "traversal"),
                        ("min_length", "min_length"),
                        ("error_margin", "error_margin")
                    ],
                )
            ]
        )
        thr_out_node = join_iters_node_thr
    else:
        thr_out_node = thresh_diff_node

    # Plotting
    if plot_switch is True:
        from pynets import plotting
        plot_fields = [
            "conn_matrix",
            "conn_model",
            "atlas",
            "dir_path",
            "ID",
            "subnet",
            "labels",
            "roi",
            "coords",
            "thr",
            "node_radius",
            "edge_threshold",
            "prune",
            "parcellation",
            "norm",
            "binary",
            "track_type",
            "traversal",
            "min_length",
            "error_margin"
        ]

        # Plotting iterable graph solutions
        if (
            conn_model_list
            or node_radii_list
            or multi_traversal
            or min_length_list
            or multi_thr
            or user_atlas_list
            or multi_atlas
            or flexi_atlas is True
        ):
            plot_all_node = pe.MapNode(
                plotting.interfaces.PlotStruct(),
                iterfield=plot_fields,
                name="plot_all_node",
                nested=True)
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(plotting.interfaces.PlotStruct(),
                                    name="plot_all_node")

        # Connect thresh_diff_node outputs to plotting node
        dmri_wf.connect(
            [
                (
                    thr_out_node,
                    plot_all_node,
                    [
                        ("est_path", "conn_matrix"),
                        ("conn_model", "conn_model"),
                        ("atlas", "atlas"),
                        ("dir_path", "dir_path"),
                        ("ID", "ID"),
                        ("subnet", "subnet"),
                        ("labels", "labels"),
                        ("roi", "roi"),
                        ("coords", "coords"),
                        ("thr", "thr"),
                        ("node_radius", "node_radius"),
                        ("edge_threshold", "edge_threshold"),
                        ("prune", "prune"),
                        ("atlas_for_streams", "parcellation"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("track_type", "track_type"),
                        ("traversal", "traversal"),
                        ("min_length", "min_length"),
                        ("error_margin", "error_margin")
                    ],
                )
            ]
        )

    # Connect nodes of workflow
    dmri_wf.connect(
        [
            (
                inputnode,
                fetch_nodes_and_labels_node,
                [
                    ("parc", "parc"),
                    ("ref_txt", "ref_txt"),
                    ("use_parcel_naming", "use_parcel_naming"),
                    ("outdir", "outdir"),
                    ("vox_size", "vox_size"),
                ],
            ),
            (inputnode, node_gen_node, [("ID", "ID")]),
            (
                inputnode,
                orient_reslice_dwi_node,
                [
                    ("dwi_file", "infile"),
                    ("fbvec", "bvecs"),
                    ("outdir", "outdir"),
                    ("vox_size", "vox_size"),
                ],
            ),
            (
                orient_reslice_dwi_node,
                fetch_nodes_and_labels_node,
                [("outfile", "in_file")],
            ),
            (
                fetch_nodes_and_labels_node,
                node_gen_node,
                [
                    ("dir_path", "dir_path"),
                    ("par_max", "par_max"),
                    ("networks_list", "networks_list"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                ],
            ),
            (
                orient_reslice_dwi_node,
                gtab_node,
                [("bvecs", "fbvec"), ("outfile", "dwi_file")],
            ),
            (inputnode, gtab_node, [("fbval", "fbval")]),
            (
                inputnode,
                register_node,
                [("vox_size", "vox_size"),
                 ("template_name", "template_name")],
            ),
            (
                inputnode,
                orient_reslice_anat_node,
                [
                    ("anat_file", "infile"),
                    ("vox_size", "vox_size"),
                    ("outdir", "outdir"),
                ],
            ),
            (
                orient_reslice_anat_node,
                register_node,
                [("outfile", "anat_file")],
            ),
            (inputnode, save_nifti_parcels_node, [("ID", "ID"),
                                                  ("vox_size", "vox_size")]),
            (
                fetch_nodes_and_labels_node,
                save_coords_and_labels_node,
                [("dir_path", "dir_path")],
            ),
            (
                reg_nodes_node,
                save_coords_and_labels_node,
                [("coords", "coords"),
                 ("labels", "labels")],
            ),
            (
                node_gen_node,
                save_nifti_parcels_node,
                [("net_parcels_map_nifti",
                  "net_parcels_map_nifti"),
                 ("dir_path", "dir_path")],
            ),
            (
                inputnode,
                reg_nodes_node,
                [
                    ("vox_size", "vox_size"),
                    ("template_name", "template_name"),
                    ("mask", "mask"),
                ],
            ),
            (
                save_nifti_parcels_node,
                reg_nodes_node,
                [("net_parcels_nii_path", "parcellation4d")],
            ),
            (
                node_gen_node,
                reg_nodes_node,
                [("atlas", "atlas"),
                 ("coords", "coords"),
                 ("labels", "labels")],
            ),
            (
                register_node,
                reg_nodes_node,
                [
                    ("basedir_path", "basedir_path"),
                    ("anat_file", "anat_file"),
                    ("gm_in_dwi", "gm_in_dwi"),
                    ("vent_csf_in_dwi", "vent_csf_in_dwi"),
                    ("wm_in_dwi", "wm_in_dwi"),
                    ("ap_path", "ap_path"),
                    ("B0_mask", "B0_mask"),
                    ("gtab_file", "gtab_file"),
                    ("dwi_file", "dwi_file"),
                    ("t1w_brain", "t1w_brain"),
                    ("mni2t1w_warp", "mni2t1w_warp"),
                    ("t1wtissue2dwi_xfm", "t1wtissue2dwi_xfm"),
                    ("mni2t1_xfm", "mni2t1_xfm"),
                    ("t1w_brain_mask", "t1w_brain_mask"),
                    ("t1w2dwi_bbr_xfm", "t1w2dwi_bbr_xfm"),
                    ("t1w2dwi_xfm", "t1w2dwi_xfm"),
                    ("wm_gm_int_in_dwi", "wm_gm_int_in_dwi"),
                    ("t1_aligned_mni", "t1_aligned_mni"),
                ],
            ),
            (
                gtab_node,
                get_fa_node,
                [
                    ("B0_mask", "B0_mask"),
                    ("gtab_file", "gtab_file"),
                    ("dwi_file", "dwi_file"),
                ],
            ),
            (
                gtab_node,
                get_anisopwr_node,
                [
                    ("B0_mask", "B0_mask"),
                    ("gtab_file", "gtab_file"),
                    ("dwi_file", "dwi_file"),
                ],
            ),
            (
                get_anisopwr_node,
                register_node,
                [
                    ("anisopwr_path", "ap_path"),
                    ("B0_mask", "B0_mask"),
                    ("gtab_file", "gtab_file"),
                    ("dwi_file", "dwi_file"),
                ],
            ),
            (get_fa_node, register_node, [("fa_path", "fa_path")]),
            (get_fa_node, reg_nodes_node, [("fa_path", "fa_path")]),
            (get_fa_node, run_tracking_node, [("fa_path", "fa_path")]),
            (register_node, run_tracking_node, [("t1w2dwi", "t1w2dwi")]),
            (
                reg_nodes_node,
                run_tracking_node,
                [
                    ("dwi_aligned_atlas_wmgm_int", "labels_im_file_wm_gm_int"),
                    ("dwi_aligned_atlas", "labels_im_file"),
                    ("aligned_atlas_t1w", "atlas_t1w"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("coords", "coords"),
                    ("labels", "labels"),
                    ("gm_in_dwi", "gm_in_dwi"),
                    ("vent_csf_in_dwi", "vent_csf_in_dwi"),
                    ("wm_in_dwi", "wm_in_dwi"),
                    ("gtab_file", "gtab_file"),
                    ("B0_mask", "B0_mask"),
                    ("dwi_file", "dwi_file"),
                ],
            ),
            (
                inputnode,
                run_tracking_node,
                [
                    ("tiss_class", "tiss_class"),
                    ("dens_thresh", "dens_thresh"),
                    ("ID", "ID"),
                    ("min_span_tree", "min_span_tree"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                    ("curv_thr_list", "curv_thr_list"),
                    ("step_list", "step_list"),
                    ("track_type", "track_type"),
                    ("maxcrossing", "maxcrossing"),
                ],
            ),
            (get_anisopwr_node, dsn_node, [("anisopwr_path", "ap_path")]),
            (
                register_node,
                dsn_node,
                [
                    ("basedir_path", "basedir_path"),
                    ("t1w_brain", "t1w_brain"),
                ],
            ),
            (
                run_tracking_node,
                dsn_node,
                [
                    ("dir_path", "dir_path"),
                    ("streams", "streams"),
                    ("curv_thr_list", "curv_thr_list"),
                    ("step_list", "step_list"),
                    ("track_type", "track_type"),
                    ("conn_model", "conn_model"),
                    ("node_radius", "node_radius"),
                    ("dens_thresh", "dens_thresh"),
                    ("ID", "ID"),
                    ("roi", "roi"),
                    ("min_span_tree", "min_span_tree"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("atlas", "atlas"),
                    ("labels_im_file", "labels_im_file"),
                    ("labels", "labels"),
                    ("coords", "coords"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                    ("atlas_t1w", "atlas_t1w"),
                    ("fa_path", "fa_path"),
                    ("traversal", "traversal"),
                    ("min_length", "min_length"),
                    ("subnet", "subnet")
                ],
            ),
            (
                dsn_node,
                streams2graph_node,
                [
                    ("streams_t1w", "streams"),
                    ("dir_path", "dir_path"),
                    ("track_type", "track_type"),
                    ("conn_model", "conn_model"),
                    ("node_radius", "node_radius"),
                    ("dens_thresh", "dens_thresh"),
                    ("ID", "ID"),
                    ("roi", "roi"),
                    ("min_span_tree", "min_span_tree"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("labels", "labels"),
                    ("coords", "coords"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                    ("atlas_for_streams", "atlas_for_streams"),
                    ("traversal", "traversal"),
                    ("warped_fa", "warped_fa"),
                    ("min_length", "min_length"),
                    ("subnet", "subnet"),
                ],
            ),
        ]
    )

    if waymask is not None:
        orient_reslice_waymask_node = pe.Node(
            niu.Function(
                input_names=["infile", "outdir", "vox_size"],
                output_names=["outfile"],
                function=registration.utils.orient_reslice,
                imports=import_list
            ),
            name="orient_reslice_waymask_node",
        )
        dmri_wf.connect(
            [
                (
                    inputnode,
                    orient_reslice_waymask_node,
                    [
                        ("waymask", "infile"),
                        ("vox_size", "vox_size"),
                        ("outdir", "outdir"),
                    ],
                ),
                (
                    orient_reslice_waymask_node,
                    reg_nodes_node,
                    [("outfile", "waymask")],
                ),
                (
                    reg_nodes_node,
                    run_tracking_node,
                    [("waymask_in_dwi", "waymask")],
                ),
            ]
        )
    else:
        dmri_wf.connect(
            [
                (inputnode, reg_nodes_node, [("waymask", "waymask")]),
                (inputnode, run_tracking_node, [("waymask", "waymask")]),
            ]
        )

    # Handle mask scenarios
    if mask is not None:
        orient_reslice_mask_node = pe.Node(
            niu.Function(
                input_names=["infile", "outdir", "vox_size"],
                output_names=["outfile"],
                function=registration.utils.orient_reslice,
                imports=import_list
            ),
            name="orient_reslice_mask_node",
        )
        dmri_wf.connect([(inputnode,
                                        orient_reslice_mask_node,
                                        [("mask",
                                          "infile"),
                                         ("outdir",
                                            "outdir"),
                                         ("vox_size",
                                          "vox_size"),
                                         ],
                                        ),
                                       (orient_reslice_mask_node,
                                        register_node,
                                        [("outfile",
                                          "mask")]),
                                       ])
    else:
        dmri_wf.connect(
            [(inputnode, register_node, [("mask", "mask")]), ]
        )

    if roi:
        dmri_wf.connect(
            [
                (
                    inputnode,
                    register_roi_node,
                    [("vox_size", "vox_size"),
                     ("template_name", "template_name")],
                ),
                (
                    register_node,
                    register_roi_node,
                    [
                        ("basedir_path", "basedir_path"),
                        ("anat_file", "anat_file"),
                        ("gm_in_dwi", "gm_in_dwi"),
                        ("vent_csf_in_dwi", "vent_csf_in_dwi"),
                        ("wm_in_dwi", "wm_in_dwi"),
                        ("ap_path", "ap_path"),
                        ("B0_mask", "B0_mask"),
                        ("gtab_file", "gtab_file"),
                        ("dwi_file", "dwi_file"),
                        ("t1w_brain", "t1w_brain"),
                        ("mni2t1w_warp", "mni2t1w_warp"),
                        ("t1wtissue2dwi_xfm", "t1wtissue2dwi_xfm"),
                        ("mni2t1_xfm", "mni2t1_xfm"),
                    ],
                ),
                (get_fa_node, register_roi_node, [("fa_path", "fa_path")]),
                (register_roi_node, node_gen_node, [("roi", "roi")]),
                (register_roi_node, run_tracking_node, [("roi", "roi")]),
            ]
        )
    else:
        dmri_wf.connect(
            [
                (inputnode, node_gen_node, [("roi", "roi")]),
                (inputnode, run_tracking_node, [("roi", "roi")]),
            ]
        )

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(
            niu.IdentityInterface(
                fields=[
                    "est_path",
                    "thr",
                    "subnet",
                    "prune",
                    "ID",
                    "roi",
                    "conn_model",
                    "node_radius",
                    "track_type",
                    "norm",
                    "binary",
                    "atlas_for_streams",
                    "streams",
                    "traversal",
                    "min_length",
                    "error_margin"
                ]
            ),
            name="join_iters_node_nets",
            joinsource=get_node_membership_node,
            joinfield=[
                "est_path",
                "thr",
                "subnet",
                "prune",
                "ID",
                "roi",
                "conn_model",
                "node_radius",
                "track_type",
                "norm",
                "binary",
                "atlas_for_streams",
                "streams",
                "traversal",
                "min_length",
                "error_margin"
            ],
        )
        dmri_wf.connect(
            [
                (
                    thr_out_node,
                    join_iters_node_nets,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("node_radius", "node_radius"),
                        ("track_type", "track_type"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("atlas_for_streams", "atlas_for_streams"),
                        ("streams", "streams"),
                        ("traversal", "traversal"),
                        ("min_length", "min_length"),
                        ("error_margin", "error_margin")
                    ],
                ),
                (
                    join_iters_node_nets,
                    outputnode,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
            ]
        )
    else:
        dmri_wf.connect(
            [
                (
                    thr_out_node,
                    outputnode,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
            ]
        )

    for node_name in dmri_wf.list_node_names():
        if node_name in runtime_dict:
            dmri_wf.get_node(
                node_name).interface.n_procs = runtime_dict[node_name][0]
            dmri_wf.get_node(
                node_name).interface.mem_gb = runtime_dict[node_name][1]
            dmri_wf.get_node(
                node_name).n_procs = runtime_dict[node_name][0]
            dmri_wf.get_node(
                node_name)._mem_gb = runtime_dict[node_name][1]

    execution_dict["plugin_args"] = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "mem_thread",
    }
    execution_dict["logging"] = {
        "workflow_level": "INFO",
        "utils_level": "INFO",
        "log_to_file": False,
        "interface_level": "DEBUG",
        "filemanip_level": "DEBUG",
    }
    execution_dict["plugin"] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            dmri_wf.config[key][setting] = value

    gc.collect()
    return dmri_wf


def fmri_connectometry(
    func_file,
    ID,
    atlas,
    subnet,
    node_radius,
    roi,
    thr,
    parcellation,
    conn_model,
    dens_thresh,
    conf,
    plot_switch,
    parc,
    ref_txt,
    procmem,
    multi_thr,
    multi_atlas,
    max_thr,
    min_thr,
    step_thr,
    k,
    clust_mask,
    k_list,
    k_clustering,
    user_atlas_list,
    clust_mask_list,
    node_radii_list,
    conn_model_list,
    min_span_tree,
    use_parcel_naming,
    smooth,
    smooth_list,
    disp_filt,
    prune,
    multi_nets,
    clust_type,
    clust_type_list,
    plugin_type,
    mask,
    norm,
    binary,
    anat_file,
    runtime_dict,
    execution_dict,
    hpass,
    hpass_list,
    template_name,
    vox_size,
    local_corr,
    signal,
    signal_list,
    outdir,
):
    """
    A function interface for generating an fMRI connectometry nested workflow
    """
    from pynets.core import nodemaker, utils, thresholding, interfaces
    from pynets import fmri, registration

    import_list = [
        "import matplotlib",
        "import warnings",
        "import sys",
        "import os",
        "import numpy as np",
        "import networkx as nx",
        "import nibabel as nib",
        'matplotlib.use("Agg")',
        'warnings.filterwarnings("ignore")',
    ]
    base_dirname = f"fmri"
    fmri_wf = pe.Workflow(name=base_dirname)

    if template_name == "MNI152_T1" or template_name == "colin27" or \
            template_name == "CN200":
        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{template_name}_brain_"
                      f"{vox_size}.nii.gz"
        )
        template_mask = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{template_name}_brain_mask_"
                      f"{vox_size}.nii.gz"
        )
        utils.check_template_loads(template, template_mask, template_name)
    else:
        [template, template_mask, _] = utils.get_template_tf(
            template_name, vox_size)

    if not op.isfile(template) or not op.isfile(template_mask):
        raise FileNotFoundError("Template or mask not found!")

    # Create input/output nodes
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "func_file",
                "ID",
                "atlas",
                "subnet",
                "node_radius",
                "roi",
                "thr",
                "parcellation",
                "multi_nets",
                "conn_model",
                "dens_thresh",
                "conf",
                "plot_switch",
                "parc",
                "ref_txt",
                "procmem",
                "k",
                "clust_mask",
                "k_list",
                "k_clustering",
                "user_atlas_list",
                "min_span_tree",
                "use_parcel_naming",
                "smooth",
                "disp_filt",
                "prune",
                "clust_type",
                "mask",
                "norm",
                "binary",
                "template_name",
                "template",
                "template_mask",
                "vox_size",
                "anat_file",
                "hpass",
                "hpass_list",
                "local_corr",
                "signal",
                "signal_list",
                "outdir",
            ]
        ),
        name="inputnode",
    )

    in_dir = op.dirname(anat_file)
    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas = atlas
    inputnode.inputs.subnet = subnet
    inputnode.inputs.node_radius = node_radius
    inputnode.inputs.roi = roi
    inputnode.inputs.thr = thr
    inputnode.inputs.parcellation = parcellation
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
    inputnode.inputs.k_list = k_list
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
    inputnode.inputs.use_parcel_naming = use_parcel_naming
    inputnode.inputs.smooth = smooth
    inputnode.inputs.disp_filt = disp_filt
    inputnode.inputs.prune = prune
    inputnode.inputs.clust_type = clust_type
    inputnode.inputs.clust_type_list = clust_type_list
    inputnode.inputs.mask = mask
    inputnode.inputs.norm = norm
    inputnode.inputs.binary = binary
    inputnode.inputs.template_name = template_name
    inputnode.inputs.template = template
    inputnode.inputs.template_mask = template_mask
    inputnode.inputs.vox_size = vox_size
    inputnode.inputs.anat_file = anat_file
    inputnode.inputs.hpass = hpass
    inputnode.inputs.hpass_list = hpass_list
    inputnode.inputs.local_corr = local_corr
    inputnode.inputs.signal = signal
    inputnode.inputs.signal_list = signal_list
    inputnode.inputs.outdir = outdir

    # print('\n\n\n\n\n')
    # print("%s%s" % ('ID: ', ID))
    # print("%s%s" % ('outdir: ', outdir))
    # print("%s%s" % ('atlas: ', atlas))
    # print("%s%s" % ('subnet: ', subnet))
    # print("%s%s" % ('node_radius: ', node_radius))
    # print("%s%s" % ('smooth: ', smooth))
    # print("%s%s" % ('hpass: ', hpass))
    # print("%s%s" % ('hpass_list: ', hpass_list))
    # print("%s%s" % ('roi: ', roi))
    # print("%s%s" % ('thr: ', thr))
    # print("%s%s" % ('parcellation: ', parcellation))
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
    # print("%s%s" % ('k_list: ', k_list))
    # print("%s%s" % ('k_clustering: ', k_clustering))
    # print("%s%s" % ('user_atlas_list: ', user_atlas_list))
    # print("%s%s" % ('clust_mask_list: ', clust_mask_list))
    # print("%s%s" % ('prune: ', prune))
    # print("%s%s" % ('node_radii_list: ', node_radii_list))
    # print("%s%s" % ('smooth_list: ', smooth_list))
    # print("%s%s" % ('clust_type: ', clust_type))
    # print("%s%s" % ('clust_type_list: ', clust_type_list))
    # print("%s%s" % ('mask: ', mask))
    # print("%s%s" % ('norm: ', norm))
    # print("%s%s" % ('binary: ', binary))
    # print("%s%s" % ('template_name: ', template_name))
    # print("%s%s" % ('template: ', template))
    # print("%s%s" % ('template_mask: ', template_mask))
    # print("%s%s" % ('vox_size: ', vox_size))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('signal: ', signal))
    # print("%s%s" % ('signal_list: ', signal_list))
    # print("%s%s" % ('local_corr: ', local_corr))
    # print('\n\n\n\n\n')

    if (
        atlas is None
        and parcellation is None
        and multi_atlas is None
        and user_atlas_list is None
    ):
        all_clustering = True
    else:
        all_clustering = False

    # Create function nodes
    orient_reslice_func_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size"],
            output_names=["outfile"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_func_node",
    )

    orient_reslice_func_node._n_procs = runtime_dict[
        "orient_reslice_func_node"
    ][0]
    orient_reslice_func_node._mem_gb = runtime_dict[
        "orient_reslice_func_node"
    ][1]

    orient_reslice_anat_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size"],
            output_names=["outfile"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_anat_node",
    )

    register_node = pe.Node(
        registration.interfaces.RegisterFunc(in_dir=in_dir),
        name="register_node")

    register_node._n_procs = runtime_dict["register_node"][0]
    register_node._mem_gb = runtime_dict["register_node"][1]

    reg_nodes_node = pe.Node(
        registration.interfaces.RegisterAtlasFunc(),
        name="reg_nodes_node")

    # Clustering
    if float(k_clustering) > 0:
        reg_nodes_node = pe.Node(
            registration.interfaces.RegisterAtlasFunc(already_run=True),
            name="reg_nodes_node")

        clustering_info_node = pe.Node(
            niu.IdentityInterface(fields=["clust_mask", "clust_type", "k"]),
            name="clustering_info_node",
        )

        clustering_node = pe.Node(
            fmri.interfaces.IndividualClustering(),
            name="clustering_node")

        clustering_node.interface.n_procs = runtime_dict["clustering_node"][0]
        clustering_node.interface.mem_gb = runtime_dict["clustering_node"][1]
        clustering_node._n_procs = runtime_dict["clustering_node"][0]
        clustering_node._mem_gb = runtime_dict["clustering_node"][1]

        # Don't forget that this setting exists
        clustering_node.synchronize = True

        # clustering_node iterables and names
        if k_clustering == 1:
            mask_name = op.basename(clust_mask).split(".nii")[0]
            mask_name = utils.prune_suffices(mask_name)
            cluster_atlas_name = f"{mask_name}{'_'}{clust_type}{'_k'}{k}"
            cluster_atlas_file = (
                f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                f"{mask_name}_{clust_type}_k"
                f"{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list.append(cluster_atlas_file)
            elif parcellation and ((parcellation == cluster_atlas_file) is
                                   False):
                user_atlas_list = [parcellation, cluster_atlas_file]
            else:
                parcellation = cluster_atlas_file
        elif k_clustering == 2:
            k_cluster_iterables = []
            k_cluster_iterables.append(("k", k_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for k in k_list:
                mask_name = op.basename(clust_mask).split(".nii")[0]
                mask_name = utils.prune_suffices(mask_name)
                cluster_atlas_name = f"{mask_name}{'_'}{clust_type}{'_k'}{k}"
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append(
                    f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                    f"{mask_name}_"
                    f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 3:
            clustering_info_node.iterables = [("clust_mask", clust_mask_list)]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                mask_name = op.basename(clust_mask).split(".nii")[0]
                mask_name = utils.prune_suffices(mask_name)
                cluster_atlas_name = f"{mask_name}{'_'}{clust_type}{'_k'}{k}"
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append(
                    f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                    f"{mask_name}_"
                    f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 4:
            k_cluster_iterables = []
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_mask in clust_mask_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split(".nii")[0]
                    mask_name = utils.prune_suffices(mask_name)
                    cluster_atlas_name = f"{mask_name}{'_'}{clust_type}" \
                                         f"{'_k'}{k}"
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append(
                        f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                        f"{mask_name}_"
                        f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 5:
            clustering_info_node.iterables = [("clust_type", clust_type_list)]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                mask_name = op.basename(clust_mask).split(".nii")[0]
                mask_name = utils.prune_suffices(mask_name)
                cluster_atlas_name = f"{mask_name}{'_'}{clust_type}{'_k'}{k}"
                cluster_atlas_name_list.append(cluster_atlas_name)
                cluster_atlas_file_list.append(
                    f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                    f"{mask_name}_"
                    f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 6:
            k_cluster_iterables = []
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for k in k_list:
                    mask_name = op.basename(clust_mask).split(".nii")[0]
                    mask_name = utils.prune_suffices(mask_name)
                    cluster_atlas_name = f"{mask_name}{'_'}{clust_type}" \
                                         f"{'_k'}{k}"
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append(
                        f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                        f"{mask_name}_"
                        f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 7:
            clustering_info_node.iterables = [
                ("clust_type", clust_type_list),
                ("clust_mask", clust_mask_list),
            ]
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    mask_name = op.basename(clust_mask).split(".nii")[0]
                    mask_name = utils.prune_suffices(mask_name)
                    cluster_atlas_name = f"{mask_name}{'_'}" \
                                         f"{clust_type}{'_k'}{k}"
                    cluster_atlas_name_list.append(cluster_atlas_name)
                    cluster_atlas_file_list.append(
                        f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                        f"{mask_name}_"
                        f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list
        elif k_clustering == 8:
            k_cluster_iterables = []
            k_cluster_iterables.append(("k", k_list))
            k_cluster_iterables.append(("clust_mask", clust_mask_list))
            k_cluster_iterables.append(("clust_type", clust_type_list))
            clustering_info_node.iterables = k_cluster_iterables
            cluster_atlas_name_list = []
            cluster_atlas_file_list = []
            for clust_type in clust_type_list:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        mask_name = op.basename(clust_mask).split(".nii")[0]
                        mask_name = utils.prune_suffices(mask_name)
                        cluster_atlas_name = f"{mask_name}{'_'}" \
                                             f"{clust_type}{'_k'}{k}"
                        cluster_atlas_name_list.append(cluster_atlas_name)
                        cluster_atlas_file_list.append(
                            f"{utils.do_dir_path(cluster_atlas_name, outdir)}/"
                            f"{mask_name}_"
                            f"{clust_type}_k{str(k)}.nii.gz")
            if user_atlas_list:
                user_atlas_list = user_atlas_list + cluster_atlas_file_list
            elif parcellation:
                user_atlas_list = cluster_atlas_file_list + [parcellation]
            else:
                user_atlas_list = cluster_atlas_file_list

    # Define nodes
    # Create node definitions Node
    fetch_nodes_and_labels_node = pe.Node(
        interfaces.FetchNodesLabels(), name="fetch_nodes_and_labels_node"
    )
    fetch_nodes_and_labels_node.synchronize = True

    # Connect clustering solutions to node definition Node
    if float(k_clustering) > 0:

        RegisterParcellation2MNIFunc_node = pe.Node(
            registration.interfaces.RegisterParcellation2MNIFunc(),
            name="RegisterParcellation2MNIFunc_node"
        )

        fmri_wf.connect(
            [
                (
                    orient_reslice_anat_node,
                    clustering_node,
                    [("outfile", "anat_file")],
                ),
                (
                    inputnode,
                    clustering_node,
                    [("vox_size", "vox_size"),
                     ("template_name", "template_name")],
                ),
                (
                    inputnode,
                    RegisterParcellation2MNIFunc_node,
                    [("vox_size", "vox_size"),
                     ("template_name", "template_name"),
                     ("outdir", "dir_path")],
                ),
                (
                    register_node,
                    RegisterParcellation2MNIFunc_node,
                    [
                        ("t1w2mni_xfm", "t1w2mni_xfm"),
                        ("t1w_brain", "t1w_brain"),
                        ("t1w2mni_warp", "t1w2mni_warp"),
                    ],
                ),
                (
                    register_node,
                    clustering_node,
                    [
                        ("basedir_path", "basedir_path"),
                        ("t1w_brain", "t1w_brain"),
                        ("mni2t1w_warp", "mni2t1w_warp"),
                        ("mni2t1_xfm", "mni2t1_xfm"),
                    ],
                ),
                (
                    inputnode,
                    clustering_node,
                    [
                        ("ID", "ID"),
                        ("conf", "conf"),
                        ("local_corr", "local_corr"),
                        ("outdir", "outdir"),
                    ],
                ),
                (
                    orient_reslice_func_node,
                    clustering_node,
                    [("outfile", "func_file")],
                ),
                (
                    clustering_node,
                    fetch_nodes_and_labels_node,
                    [
                        ("atlas", "atlas"),
                        ("clustering", "clustering"),
                    ],
                ),
                (
                    clustering_node,
                    RegisterParcellation2MNIFunc_node,
                    [
                        ("parcellation", "parcellation"),
                    ],
                ),
                (
                    RegisterParcellation2MNIFunc_node,
                    fetch_nodes_and_labels_node,
                    [
                        ("aligned_atlas_mni", "parcellation"),
                    ],
                ),
                (
                    inputnode,
                    clustering_info_node,
                    [("clust_mask", "clust_mask"),
                     ("clust_type", "clust_type"),
                     ("k", "k")],
                ),
                (
                    clustering_info_node,
                    clustering_node,
                    [
                        ("clust_mask", "clust_mask"),
                        ("clust_type", "clust_type"),
                        ("k", "k"),
                    ],
                ),
                (
                    clustering_node,
                    fetch_nodes_and_labels_node,
                    [("func_file", "in_file")],
                ),
            ]
        )
    else:
        # Connect atlas input vars to node definition Node
        fmri_wf.connect(
            [
                (
                    inputnode,
                    fetch_nodes_and_labels_node,
                    [("atlas", "atlas"), ("parcellation", "parcellation")],
                ),
                (
                    orient_reslice_func_node,
                    fetch_nodes_and_labels_node,
                    [("outfile", "in_file")],
                ),
            ]
        )

    reg_nodes_node._n_procs = runtime_dict["reg_nodes_node"][0]
    reg_nodes_node._mem_gb = runtime_dict["reg_nodes_node"][1]

    # Set atlas iterables and logic for multiple atlas useage
    if all_clustering is True:
        flexi_atlas = False
    elif (
        (multi_atlas is not None and user_atlas_list is None and
         parcellation is None)
        or (multi_atlas is None and atlas is None and user_atlas_list is
            not None)
    ) and k_clustering == 0:
        # print('\n\n\n\n')
        # print('No flexi-atlas1')
        # print('\n\n\n\n')
        atlas_iters = []
        flexi_atlas = False
        if multi_atlas:
            atlas_iters.append(("atlas", multi_atlas))
        elif user_atlas_list:
            atlas_iters.append(("parcellation", user_atlas_list))
        fetch_nodes_and_labels_node.iterables = atlas_iters

    elif (
        (atlas is not None and parcellation is None and k_clustering == 0)
        or (atlas is None and parcellation is not None and k_clustering == 0)
        or (k_clustering > 0 and atlas is None and multi_atlas is None)
    ):
        # print('\n\n\n\n')
        # print('No flexi-atlas2')
        # print('\n\n\n\n')
        flexi_atlas = False
    else:
        flexi_atlas = True
        flexi_atlas_source = pe.Node(
            niu.IdentityInterface(fields=["atlas", "parcellation",
                                          "clustering"]),
            name="flexi_atlas_source",
        )
        flexi_atlas_source.synchronize = True
        if multi_atlas is not None and user_atlas_list is not None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: multiple nilearn atlases + multiple user '
            #       'atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", len(user_atlas_list) * [None] + multi_atlas),
                ("parcellation", user_atlas_list + len(multi_atlas) * [None]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif multi_atlas is not None and parcellation is not None and \
                user_atlas_list is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single user atlas + multiple nilearn '
            #       'atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", multi_atlas + [None]),
                ("parcellation", len(multi_atlas) * [None] + [parcellation]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif atlas is not None and user_atlas_list is not None and \
                multi_atlas is None:
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + multiple user '
            #       'atlases')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", len(user_atlas_list) * [None] + [atlas]),
                ("parcellation", user_atlas_list + [None]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables
        elif (
            atlas is not None
            and parcellation is not None
            and user_atlas_list is None
            and multi_atlas is None
        ):
            # print('\n\n\n\n')
            # print('Flexi-atlas: single nilearn atlas + single user atlas')
            # print('\n\n\n\n')
            flexi_atlas_source_iterables = [
                ("atlas", [atlas, None]),
                ("parcellation", [None, parcellation]),
            ]
            flexi_atlas_source.iterables = flexi_atlas_source_iterables

    # Connect flexi_atlas inputs to definition node
    if flexi_atlas is True:
        fmri_wf.add_nodes([flexi_atlas_source])
        if float(k_clustering) > 0:
            fmri_wf.disconnect(
                [
                    (
                        clustering_node,
                        fetch_nodes_and_labels_node,
                        [
                            ("atlas", "atlas"),
                            ("clustering", "clustering"),
                        ],
                    ),
                    (
                        RegisterParcellation2MNIFunc_node,
                        fetch_nodes_and_labels_node,
                        [
                            ("aligned_atlas_mni", "parcellation"),
                        ],
                    )
                ]
            )
            if float(k_clustering == 1):
                fmri_wf.connect(
                    [
                        (
                            clustering_node,
                            flexi_atlas_source,
                            [("clustering", "clustering")],
                        )
                    ]
                )
            else:
                clust_join_node = pe.JoinNode(
                    niu.IdentityInterface(
                        fields=["clustering", "k", "clust_mask", "clust_type"]
                    ),
                    name="clust_join_node",
                    joinsource=clustering_info_node,
                    joinfield=["k", "clust_mask", "clust_type"],
                )
                fmri_wf.connect(
                    [
                        (
                            clustering_node,
                            clust_join_node,
                            [
                                ("clustering", "clustering"),
                                ("k", "k"),
                                ("clust_mask", "clust_mask"),
                                ("clust_type", "clust_type"),
                            ],
                        )
                    ]
                )
                fmri_wf.connect(
                    [
                        (
                            clust_join_node,
                            flexi_atlas_source,
                            [("clustering", "clustering")],
                        )
                    ]
                )
            fmri_wf.connect(
                [
                    (
                        RegisterParcellation2MNIFunc_node,
                        flexi_atlas_source,
                        [
                            ("aligned_atlas_mni", "parcellation"),
                        ],
                    ),
                    (
                        flexi_atlas_source,
                        fetch_nodes_and_labels_node,
                        [
                            ("parcellation", "parcellation"),
                            ("atlas", "atlas"),
                            ("clustering", "clustering"),
                        ],
                    )
                ]
            )
        else:
            fmri_wf.disconnect(
                [
                    (
                        inputnode,
                        fetch_nodes_and_labels_node,
                        [
                            ("parcellation", "parcellation"),
                            ("atlas", "atlas"),
                            ("clustering", "clustering"),
                        ],
                    )
                ]
            )
            fmri_wf.connect(
                [
                    (
                        flexi_atlas_source,
                        fetch_nodes_and_labels_node,
                        [("parcellation", "parcellation"), ("atlas", "atlas")],
                    )
                ]
            )

    # Generate nodes
    if roi is not None:
        # Masking case
        node_gen_node = pe.Node(
            niu.Function(
                input_names=[
                    "roi",
                    "coords",
                    "parcel_list",
                    "labels",
                    "dir_path",
                    "ID",
                    "parc",
                    "atlas",
                    "parcellation",
                    "vox_size"
                ],
                output_names=[
                    "net_parcels_map_nifti",
                    "coords",
                    "labels",
                    "atlas",
                    "parcellation",
                    "dir_path",
                ],
                function=nodemaker.node_gen_masking,
                imports=import_list
            ),
            name="node_gen_node",
        )
        fmri_wf.connect(
            [
                (inputnode, node_gen_node, [("vox_size", "vox_size")]),
            ]
        )
    else:
        # Non-masking case
        node_gen_node = pe.Node(
            niu.Function(
                input_names=[
                    "coords",
                    "parcel_list",
                    "labels",
                    "dir_path",
                    "ID",
                    "parc",
                    "atlas",
                    "parcellation",
                ],
                output_names=[
                    "net_parcels_map_nifti",
                    "coords",
                    "labels",
                    "atlas",
                    "parcellation",
                    "dir_path",
                ],
                function=nodemaker.node_gen,
                imports=import_list
            ),
            name="node_gen_node",
        )
    node_gen_node._n_procs = runtime_dict["node_gen_node"][0]
    node_gen_node._mem_gb = runtime_dict["node_gen_node"][1]

    # Extract time-series from nodes
    extract_ts_info_iters = []
    extract_ts_info_node = pe.Node(
        name="extract_ts_info_node",
        interface=niu.IdentityInterface(
            fields=["node_radius", "smooth", "hpass", "signal"]
        ),
    )
    extract_ts_node = pe.Node(
        fmri.interfaces.ExtractTimeseries(),
        name="extract_ts_node",
    )

    extract_ts_node.interface.n_procs = runtime_dict["extract_ts_node"][0]
    extract_ts_node.interface.mem_gb = runtime_dict["extract_ts_node"][1]
    extract_ts_node._n_procs = runtime_dict["extract_ts_node"][0]
    extract_ts_node._mem_gb = runtime_dict["extract_ts_node"][1]

    if parc is True:
        # Parcels case
        extract_ts_node.inputs.parc = True
        extract_ts_node.inputs.node_radius = None
        reg_nodes_node.inputs.node_radius = None

    else:
        prep_spherical_nodes_node = pe.Node(
            niu.Function(
                input_names=["coords", "node_radius", "template_mask"],
                output_names=["parcel_list", "par_max", "node_radius", "parc"],
                function=nodemaker.create_spherical_roi_volumes,
                imports=import_list
            ),
            name="prep_spherical_nodes_node",
        )

        if node_radii_list:
            prep_spherical_nodes_node.inputs.node_radius = None
            prep_spherical_nodes_node.iterables = [
                ("node_radius", node_radii_list)]
        else:
            fmri_wf.connect(
                [
                    (
                        inputnode,
                        prep_spherical_nodes_node,
                        [("node_radius", "node_radius")],
                    ),
                    (
                        prep_spherical_nodes_node,
                        reg_nodes_node,
                        [("node_radius", "node_radius")],
                    ),
                ]
            )

        prep_spherical_nodes_node.synchronize = True

        # Coordinate case
        extract_ts_node.inputs.parc = False

    save_nifti_parcels_node = pe.Node(
        niu.Function(
            input_names=["ID", "dir_path", "subnet", "net_parcels_map_nifti",
                         "vox_size"],
            output_names=["net_parcels_nii_path"],
            function=utils.save_nifti_parcels_map,
            imports=import_list
        ),
        name="save_nifti_parcels_node",
    )
    fmri_wf.add_nodes([save_nifti_parcels_node])
    fmri_wf.connect(
        [
            (inputnode, save_nifti_parcels_node, [("ID", "ID"),
                                                  ("vox_size", "vox_size")]),
            (
                node_gen_node,
                save_nifti_parcels_node,
                [("net_parcels_map_nifti", "net_parcels_map_nifti"),
                 ("dir_path", "dir_path")],
            ),
            (
                save_nifti_parcels_node,
                reg_nodes_node,
                [("net_parcels_nii_path", "parcellation4d")],
            ),
            (
                reg_nodes_node,
                extract_ts_node,
                [("aligned_atlas_gm", "net_parcels_nii_path")],
            ),
        ]
    )

    # Set extract_ts iterables
    if not smooth_list and hpass_list and signal_list:
        signal_hpass_combo = list(
            itertools.product(hpass_list, signal_list)
        )
        hpass_list = [i[0] for i in signal_hpass_combo]
        signal_list = [i[1] for i in signal_hpass_combo]
    elif smooth_list and not hpass_list and signal_list:
        signal_tol_combo = list(
            itertools.product(smooth_list, signal_list)
        )
        smooth_list = [i[0] for i in signal_tol_combo]
        signal_list = [i[1] for i in signal_tol_combo]
    elif smooth_list and hpass_list and signal_list:
        signal_tol_hpass_combo = list(
            itertools.product(smooth_list, signal_list, hpass_list)
        )
        smooth_list = [i[0] for i in signal_tol_hpass_combo]
        signal_list = [i[1]
                                 for i in signal_tol_hpass_combo]
        hpass_list = [i[2] for i in signal_tol_hpass_combo]
    elif smooth_list and hpass_list and not signal_list:
        tol_hpass_combo = list(itertools.product(hpass_list, smooth_list))
        hpass_list = [i[0] for i in tol_hpass_combo]
        smooth_list = [i[1] for i in tol_hpass_combo]

    if signal_list:
        extract_ts_info_iters.append(
            ("signal", signal_list))
    else:
        fmri_wf.connect(
            [
                (
                    inputnode,
                    extract_ts_info_node,
                    [("signal", "signal")],
                )
            ]
        )

    if smooth_list:
        extract_ts_info_iters.append(("smooth", smooth_list))
    else:
        fmri_wf.connect(
            [(inputnode, extract_ts_info_node, [("smooth", "smooth")])]
        )

    if hpass_list:
        extract_ts_info_iters.append(("hpass", hpass_list))
    else:
        fmri_wf.connect(
            [(inputnode, extract_ts_info_node, [("hpass", "hpass")])]
        )

    if hpass_list or smooth_list or node_radii_list or signal_list:
        # print("%s%s" % ('Expanding within-node iterable combos:\n',
        #                 extract_ts_info_iters))
        extract_ts_info_node.iterables = extract_ts_info_iters

    extract_ts_info_node.synchronize = True

    fmri_wf.connect(
        [
            (
                extract_ts_info_node,
                extract_ts_node,
                [
                    ("hpass", "hpass"),
                    ("smooth", "smooth"),
                    ("node_radius", "node_radius"),
                    ("signal", "signal"),
                ],
            )
        ]
    )

    # Connectivity matrix model fit
    get_conn_matrix_node = pe.Node(
        niu.Function(
            input_names=[
                "time_series",
                "conn_model",
                "dir_path",
                "node_radius",
                "smooth",
                "dens_thresh",
                "subnet",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "hpass",
                "signal",
            ],
            output_names=[
                "conn_matrix",
                "conn_model",
                "dir_path",
                "node_radius",
                "smooth",
                "dens_thresh",
                "subnet",
                "ID",
                "roi",
                "min_span_tree",
                "disp_filt",
                "parc",
                "prune",
                "atlas",
                "parcellation",
                "labels",
                "coords",
                "norm",
                "binary",
                "hpass",
                "signal",
            ],
            function=fmri.estimation.get_conn_matrix,
            imports=import_list
        ),
        name="get_conn_matrix_node",
    )

    # Set get_conn_matrix_node iterables
    if conn_model_list:
        get_conn_matrix_node.iterables = ("conn_model", conn_model_list)
    else:
        fmri_wf.connect(
            [(inputnode, get_conn_matrix_node, [("conn_model", "conn_model")])]
        )

    get_conn_matrix_node.synchronize = True

    # ROI case
    if roi is not None:
        orient_reslice_roi_node = pe.Node(
            niu.Function(
                input_names=["infile", "outdir", "vox_size"],
                output_names=["outfile"],
                function=registration.utils.orient_reslice,
                imports=import_list
            ),
            name="orient_reslice_roi_node",
        )

        register_roi_node = pe.Node(registration.interfaces.RegisterROIEPI(),
                                    name="register_roi_node")

        fmri_wf.connect([(inputnode,
                                        orient_reslice_roi_node,
                                        [("roi",
                                          "infile"),
                                         ("outdir",
                                          "outdir"),
                                         ("vox_size",
                                            "vox_size")],
                                        ),
                                       (orient_reslice_roi_node,
                                        register_roi_node,
                                        [("outfile", "roi")],
                                        ),
                                       ])

    save_coords_and_labels_node = pe.Node(
        niu.Function(
            input_names=["coords", "labels", "dir_path", "subnet", "indices"],
            function=utils.save_coords_and_labels_to_json,
            imports=import_list
        ),
        name="save_coords_and_labels_node",
    )

    # RSN case
    if subnet or multi_nets:
        get_node_membership_node = pe.Node(
            niu.Function(
                input_names=[
                    "subnet",
                    "infile",
                    "coords",
                    "labels",
                    "parc",
                    "parcel_list",
                    "perc_overlap",
                    "error",
                ],
                output_names=[
                    "net_coords",
                    "net_parcel_list",
                    "net_labels",
                    "subnet"],
                function=nodemaker.get_node_membership,
                imports=import_list
            ),
            name="get_node_membership_node",
        )
        get_node_membership_node._n_procs = runtime_dict["get_node_"
                                                         "membership_node"][0]
        get_node_membership_node._mem_gb = runtime_dict["get_node_"
                                                        "membership_node"][1]

        if multi_nets:
            get_node_membership_iterables = []
            get_node_membership_node.inputs.subnet = None
            get_node_membership_iterables.append(("subnet", multi_nets))
            get_node_membership_node.iterables = get_node_membership_iterables

        fmri_wf.connect(
            [
                (
                    inputnode,
                    get_node_membership_node,
                    [("subnet", "subnet"), ("template", "infile"),
                     ("parc", "parc")],
                ),
                (
                    get_node_membership_node,
                    save_nifti_parcels_node,
                    [("subnet", "subnet")],
                ),
                (get_node_membership_node, extract_ts_node,
                 [("subnet", "subnet")]),
                (
                    get_node_membership_node,
                    get_conn_matrix_node,
                    [("subnet", "subnet")],
                ),
                (
                    get_node_membership_node,
                    reg_nodes_node,
                    [("subnet", "subnet")],
                ),
                (
                    get_node_membership_node,
                    save_coords_and_labels_node,
                    [("subnet", "subnet")],
                ),
            ]
        )

        if parc is False:
            fmri_wf.connect(
                [
                    (
                        get_node_membership_node,
                        prep_spherical_nodes_node,
                        [("net_coords", "coords")],
                    ),
                    (
                        fetch_nodes_and_labels_node,
                        get_node_membership_node,
                        [
                            ("coords", "coords"),
                            ("labels", "labels"),
                            ("networks_list", "networks_list"),
                            ("parcel_list", "parcel_list"),
                        ],
                    ),
                    (
                        prep_spherical_nodes_node,
                        node_gen_node,
                        [("parc", "parc"), ("parcel_list", "parcel_list")],
                    ),
                    (
                        get_node_membership_node,
                        node_gen_node,
                        [("net_coords", "coords"), ("net_labels", "labels")],
                    ),
                ]
            )
        else:
            fmri_wf.connect(
                [
                    (
                        fetch_nodes_and_labels_node,
                        get_node_membership_node,
                        [
                            ("coords", "coords"),
                            ("labels", "labels"),
                            ("parcel_list", "parcel_list"),
                            ("par_max", "par_max"),
                            ("networks_list", "networks_list"),
                        ],
                    ),
                    (
                        get_node_membership_node,
                        node_gen_node,
                        [
                            ("net_coords", "coords"),
                            ("net_labels", "labels"),
                            ("net_parcel_list", "parcel_list"),
                        ],
                    ),
                    (inputnode, node_gen_node, [("parc", "parc")]),
                ]
            )
    else:
        fmri_wf.connect(
            [
                (inputnode, save_nifti_parcels_node, [("subnet", "subnet")]),
                (inputnode, extract_ts_node, [("subnet", "subnet")]),
                (inputnode, get_conn_matrix_node, [("subnet", "subnet")]),
                (inputnode, reg_nodes_node, [("subnet", "subnet")]),
                (
                    fetch_nodes_and_labels_node,
                    node_gen_node,
                    [("coords", "coords"), ("labels", "labels")],
                ),
            ]
        )
        if parc is False:
            fmri_wf.connect(
                [
                    (
                        prep_spherical_nodes_node,
                        node_gen_node,
                        [("parcel_list", "parcel_list"), ("parc", "parc")],
                    ),
                    (
                        fetch_nodes_and_labels_node,
                        prep_spherical_nodes_node,
                        [("coords", "coords")],
                    ),
                ]
            )
        else:
            fmri_wf.connect(
                [
                    (
                        fetch_nodes_and_labels_node,
                        node_gen_node,
                        [("parcel_list", "parcel_list")],
                    ),
                    (inputnode, node_gen_node, [("parc", "parc")]),
                ]
            )

    # Begin joinnode chaining
    # Set lists of fields and connect statements for repeated use throughout
    # joins
    map_fields = [
        "conn_model",
        "dir_path",
        "conn_matrix",
        "node_radius",
        "smooth",
        "dens_thresh",
        "subnet",
        "ID",
        "roi",
        "min_span_tree",
        "disp_filt",
        "parc",
        "prune",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "hpass",
        "thr",
        "signal",
    ]

    map_connects = [
        ("conn_model", "conn_model"),
        ("dir_path", "dir_path"),
        ("conn_matrix", "conn_matrix"),
        ("node_radius", "node_radius"),
        ("smooth", "smooth"),
        ("dens_thresh", "dens_thresh"),
        ("ID", "ID"),
        ("roi", "roi"),
        ("min_span_tree", "min_span_tree"),
        ("disp_filt", "disp_filt"),
        ("parc", "parc"),
        ("prune", "prune"),
        ("subnet", "subnet"),
        ("atlas", "atlas"),
        ("thr", "thr"),
        ("parcellation", "parcellation"),
        ("labels", "labels"),
        ("coords", "coords"),
        ("norm", "norm"),
        ("binary", "binary"),
        ("hpass", "hpass"),
        ("signal", "signal"),
    ]

    # Create a "thr_info" node for iterating iterfields across thresholds
    thr_info_node = pe.Node(
        niu.IdentityInterface(fields=map_fields), name="thr_info_node"
    )

    # Set iterables for thr on thresh_func, else set thr to singular input
    if multi_thr is True:
        iter_thresh = sorted(
            list(
                set(
                    [
                        str(i)
                        for i in np.round(
                            np.arange(float(min_thr), float(max_thr),
                                      float(step_thr)),
                            decimals=2,
                        ).tolist()
                    ]
                    + [str(float(max_thr))]
                )
            )
        )
        thr_info_node.iterables = ("thr", iter_thresh)
        thr_info_node.synchronize = True
    else:
        thr_info_node.iterables = ("thr", [thr])

    # Joinsource logic for atlas varieties
    if user_atlas_list or multi_atlas or float(
            k_clustering) > 0 or flexi_atlas is True:
        if flexi_atlas is True:
            atlas_join_source = flexi_atlas_source
        elif float(k_clustering) > 1 and flexi_atlas is False:
            atlas_join_source = clustering_info_node
        else:
            atlas_join_source = fetch_nodes_and_labels_node
    else:
        atlas_join_source = None

    # Connect all get_conn_matrix_node outputs to the "join_info" node
    fmri_wf.connect(
        [
            (
                get_conn_matrix_node,
                thr_info_node,
                [x for x in map_connects if x != ("thr", "thr")],
            )
        ]
    )

    # Begin joinnode chaining logic
    if (
        conn_model_list
        or node_radii_list
        or smooth_list
        or user_atlas_list
        or multi_atlas
        or float(k_clustering) > 1
        or flexi_atlas is True
        or multi_thr is True
        or hpass_list is not None
        or signal_list
    ):
        if (
            user_atlas_list
            or multi_atlas
            or float(k_clustering) > 1
            or flexi_atlas is True
        ):
            join_iters_node = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_node_atlas",
                joinsource=atlas_join_source,
                joinfield=map_fields,
            )
        else:
            join_iters_node = pe.Node(
                niu.IdentityInterface(
                    fields=map_fields),
                name="join_iters_node")

        if not conn_model_list and (
                node_radii_list or smooth_list or hpass_list or
                signal_list):
            # print('Time-series node extraction iterables...')
            join_iters_node_ext_ts = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_extract_ts_node",
                joinsource=extract_ts_info_node,
                joinfield=map_fields,
            )
            fmri_wf.connect(
                [
                    (thr_info_node, join_iters_node_ext_ts, map_connects),
                    (join_iters_node_ext_ts, join_iters_node, map_connects),
                ]
            )
        elif conn_model_list and (
            not node_radii_list
            and not smooth_list
            and not hpass_list
            and not signal_list
        ):
            # print('Multiple connectivity models...')
            join_iters_node_get_conn_mx = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_get_conn_matrix_node",
                joinsource=get_conn_matrix_node,
                joinfield=map_fields,
            )
            fmri_wf.connect(
                [
                    (thr_info_node, join_iters_node_get_conn_mx,
                     map_connects),
                    (join_iters_node_get_conn_mx, join_iters_node,
                     map_connects),
                ]
            )
        elif (
            not conn_model_list
            and not node_radii_list
            and not smooth_list
            and not hpass_list
            and not signal_list
        ):
            # print('No connectivity model or time-series node extraction'
            #       ' iterables...')
            fmri_wf.connect(
                [(thr_info_node, join_iters_node, map_connects)]
            )
        elif conn_model_list and (
            node_radii_list or smooth_list or hpass_list or
            signal_list
        ):
            # print('Connectivity model and time-series node extraction'
            #       ' iterables...')
            join_iters_node_ext_ts = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_node_ext_ts",
                joinsource=extract_ts_info_node,
                joinfield=map_fields,
            )
            join_iters_node_get_conn_mx = pe.JoinNode(
                niu.IdentityInterface(fields=map_fields),
                name="join_iters_get_conn_matrix_node",
                joinsource=get_conn_matrix_node,
                joinfield=map_fields,
            )
            fmri_wf.connect(
                [
                    (thr_info_node, join_iters_node_get_conn_mx, map_connects),
                    (join_iters_node_get_conn_mx, join_iters_node_ext_ts,
                     map_connects),
                    (join_iters_node_ext_ts, join_iters_node, map_connects),
                ]
            )
        else:
            raise RuntimeError("\nUnknown join context.")

        no_iters = False
    else:
        if not multi_nets:
            # Minimal case of no iterables
            print("\nNo functional connectometry iterables...")
        join_iters_node = pe.Node(
            niu.IdentityInterface(fields=map_fields), name="join_iters_node"
        )
        fmri_wf.connect(
            [
                (
                    get_conn_matrix_node,
                    join_iters_node,
                    [x for x in map_connects if x != ("thr", "thr")],
                ),
                (thr_info_node, join_iters_node, [("thr", "thr")]),
            ]
        )
        no_iters = True

    # Create final thresh_func node that performs the thresholding
    thr_func_fields = [
        "dens_thresh",
        "thr",
        "conn_matrix",
        "conn_model",
        "subnet",
        "ID",
        "dir_path",
        "roi",
        "node_radius",
        "min_span_tree",
        "smooth",
        "disp_filt",
        "parc",
        "prune",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "hpass",
        "signal",
    ]
    thr_func_iter_fields = [
        "edge_threshold",
        "est_path",
        "thr",
        "node_radius",
        "subnet",
        "conn_model",
        "roi",
        "smooth",
        "prune",
        "ID",
        "dir_path",
        "atlas",
        "parcellation",
        "labels",
        "coords",
        "norm",
        "binary",
        "hpass",
        "signal",
    ]

    if no_iters is True:
        thresh_func_node = pe.Node(
            niu.Function(
                input_names=thr_func_fields,
                output_names=[
                    "edge_threshold",
                    "est_path",
                    "thr",
                    "node_radius",
                    "subnet",
                    "conn_model",
                    "roi",
                    "smooth",
                    "prune",
                    "ID",
                    "dir_path",
                    "atlas",
                    "parcellation",
                    "labels",
                    "coords",
                    "norm",
                    "binary",
                    "hpass",
                    "signal",
                ],
                function=thresholding.thresh_func,
                imports=import_list
            ),
            name="thresh_func_node",
        )
    else:
        thresh_func_node = pe.MapNode(
            niu.Function(
                input_names=thr_func_fields,
                output_names=[
                    "edge_threshold",
                    "est_path",
                    "thr",
                    "node_radius",
                    "subnet",
                    "conn_model",
                    "roi",
                    "smooth",
                    "prune",
                    "ID",
                    "dir_path",
                    "atlas",
                    "parcellation",
                    "labels",
                    "coords",
                    "norm",
                    "binary",
                    "hpass",
                    "signal",
                ],
                function=thresholding.thresh_func,
                imports=import_list
            ),
            name="thresh_func_node",
            iterfield=thr_func_fields,
            nested=True,
        )
        thresh_func_node.synchronize = True

    fmri_wf.connect(
        [
            (
                join_iters_node,
                thresh_func_node,
                [
                    ("dens_thresh", "dens_thresh"),
                    ("thr", "thr"),
                    ("conn_matrix", "conn_matrix"),
                    ("conn_model", "conn_model"),
                    ("subnet", "subnet"),
                    ("ID", "ID"),
                    ("dir_path", "dir_path"),
                    ("roi", "roi"),
                    ("node_radius", "node_radius"),
                    ("min_span_tree", "min_span_tree"),
                    ("smooth", "smooth"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("labels", "labels"),
                    ("coords", "coords"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                    ("hpass", "hpass"),
                    ("signal", "signal"),
                ],
            )
        ]
    )

    if multi_thr is True:
        join_iters_node_thr = pe.JoinNode(
            niu.IdentityInterface(fields=thr_func_iter_fields),
            name="join_iters_node_thr",
            joinsource=thr_info_node,
            joinfield=thr_func_iter_fields,
        )
        fmri_wf.connect(
            [
                (
                    thresh_func_node,
                    join_iters_node_thr,
                    [
                        ("edge_threshold", "edge_threshold"),
                        ("est_path", "est_path"),
                        ("thr", "thr"),
                        ("node_radius", "node_radius"),
                        ("subnet", "subnet"),
                        ("conn_model", "conn_model"),
                        ("roi", "roi"),
                        ("smooth", "smooth"),
                        ("prune", "prune"),
                        ("ID", "ID"),
                        ("dir_path", "dir_path"),
                        ("atlas", "atlas"),
                        ("parcellation", "parcellation"),
                        ("labels", "labels"),
                        ("coords", "coords"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("hpass", "hpass"),
                        ("signal", "signal"),
                    ],
                )
            ]
        )
        thr_out_node = join_iters_node_thr
    else:
        thr_out_node = thresh_func_node

    # Plotting
    if plot_switch is True:
        from pynets import plotting
        plot_fields = [
            "conn_matrix",
            "conn_model",
            "atlas",
            "dir_path",
            "ID",
            "subnet",
            "labels",
            "roi",
            "coords",
            "thr",
            "node_radius",
            "edge_threshold",
            "smooth",
            "prune",
            "parcellation",
            "norm",
            "binary",
            "hpass",
            "signal",
        ]

        # Plotting iterable graph solutions
        if (
            conn_model_list
            or node_radii_list
            or smooth_list
            or multi_thr
            or user_atlas_list
            or multi_atlas
            or float(k_clustering) > 1
            or flexi_atlas is True
            or hpass_list
            or signal_list
        ):

            plot_all_node = pe.MapNode(
                plotting.interfaces.PlotFunc(),
                iterfield=plot_fields,
                name="plot_all_node",
                nested=True)
        else:
            # Plotting singular graph solution
            plot_all_node = pe.Node(plotting.interfaces.PlotFunc(),
                                    name="plot_all_node")

        if user_atlas_list or multi_atlas or multi_nets:
            edge_color_override = True
        else:
            edge_color_override = False

        plot_all_node.inputs.edge_color_override = edge_color_override

        # Connect thr_out_node outputs to plotting node
        fmri_wf.connect(
            [
                (
                    thr_out_node,
                    plot_all_node,
                    [
                        ("ID", "ID"),
                        ("roi", "roi"),
                        ("subnet", "subnet"),
                        ("prune", "prune"),
                        ("node_radius", "node_radius"),
                        ("smooth", "smooth"),
                        ("dir_path", "dir_path"),
                        ("est_path", "conn_matrix"),
                        ("edge_threshold", "edge_threshold"),
                        ("thr", "thr"),
                        ("conn_model", "conn_model"),
                        ("atlas", "atlas"),
                        ("parcellation", "parcellation"),
                        ("labels", "labels"),
                        ("coords", "coords"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("hpass", "hpass"),
                        ("signal", "signal"),
                    ],
                )
            ]
        )

    # Create outputnode to capture results of nested workflow
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "est_path",
                "thr",
                "subnet",
                "prune",
                "ID",
                "roi",
                "conn_model",
                "norm",
                "binary",
            ]
        ),
        name="outputnode",
    )

    # Handle multiple RSN cases with multi_nets joinnode
    if multi_nets:
        join_iters_node_nets = pe.JoinNode(
            niu.IdentityInterface(
                fields=[
                    "est_path",
                    "thr",
                    "subnet",
                    "prune",
                    "ID",
                    "roi",
                    "conn_model",
                    "node_radius",
                    "smooth",
                    "norm",
                    "binary",
                    "hpass",
                    "signal",
                ]
            ),
            name="join_iters_node_nets",
            joinsource=get_node_membership_node,
            joinfield=[
                "est_path",
                "thr",
                "subnet",
                "prune",
                "ID",
                "roi",
                "conn_model",
                "node_radius",
                "smooth",
                "norm",
                "binary",
                "hpass",
                "signal",
            ],
        )
        fmri_wf.connect(
            [
                (
                    thr_out_node,
                    join_iters_node_nets,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("node_radius", "node_radius"),
                        ("smooth", "smooth"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                        ("hpass", "hpass"),
                        ("signal", "signal"),
                    ],
                ),
                (
                    join_iters_node_nets,
                    outputnode,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
            ]
        )
    else:
        fmri_wf.connect(
            [
                (
                    thr_out_node,
                    outputnode,
                    [
                        ("thr", "thr"),
                        ("subnet", "subnet"),
                        ("est_path", "est_path"),
                        ("roi", "roi"),
                        ("conn_model", "conn_model"),
                        ("ID", "ID"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                )
            ]
        )

    # Handle mask scenarios
    if mask is not None:
        orient_reslice_mask_node = pe.Node(
            niu.Function(
                input_names=["infile", "outdir", "vox_size"],
                output_names=["outfile"],
                function=registration.utils.orient_reslice,
                imports=import_list
            ),
            name="orient_reslice_mask_node",
        )
        fmri_wf.connect([(inputnode,
                                        orient_reslice_mask_node,
                                        [("mask",
                                          "infile"),
                                         ("outdir",
                                            "outdir"),
                                         ("vox_size",
                                          "vox_size"),
                                         ],
                                        ),
                                       (orient_reslice_mask_node,
                                        register_node,
                                        [("outfile",
                                          "mask")]),
                                       ])
    else:
        fmri_wf.connect(
            [(inputnode, register_node, [("mask", "mask")]), ]
        )

    if roi:
        fmri_wf.connect(
            [
                (
                    orient_reslice_anat_node,
                    register_roi_node,
                    [("outfile", "anat_file")],
                ),
                (
                    inputnode,
                    register_roi_node,
                    [("vox_size", "vox_size"),
                     ("template_name", "template_name")],
                ),
                (
                    register_node,
                    register_roi_node,
                    [
                        ("basedir_path", "basedir_path"),
                        ("t1w_brain", "t1w_brain"),
                        ("mni2t1w_warp", "mni2t1w_warp"),
                        ("mni2t1_xfm", "mni2t1_xfm"),
                    ],
                ),
                (register_roi_node, node_gen_node, [("roi", "roi")]),
                (register_roi_node, extract_ts_node, [("roi", "roi")]),
            ]
        )
    else:
        fmri_wf.connect(
            [
                (inputnode, node_gen_node, [("roi", "roi")]),
                (inputnode, extract_ts_node, [("roi", "roi")]),
            ]
        )

    if k_clustering > 0:
        fmri_wf.connect(
            [(register_node, clustering_node, [("t1w_brain_mask", "mask")])]
        )

    # Connect remaining nodes of workflow
    fmri_wf.connect(
        [
            (
                inputnode,
                fetch_nodes_and_labels_node,
                [
                    ("parc", "parc"),
                    ("ref_txt", "ref_txt"),
                    ("use_parcel_naming", "use_parcel_naming"),
                    ("outdir", "outdir"),
                    ("vox_size", "vox_size"),
                ],
            ),
            (
                inputnode,
                orient_reslice_func_node,
                [
                    ("func_file", "infile"),
                    ("vox_size", "vox_size"),
                    ("outdir", "outdir"),
                ],
            ),
            (
                orient_reslice_func_node,
                extract_ts_node,
                [("outfile", "func_file")],
            ),
            (
                register_node,
                reg_nodes_node,
                [
                    ("t1w_brain", "t1w_brain"),
                    ("mni2t1w_warp", "mni2t1w_warp"),
                    ("mni2t1_xfm", "mni2t1_xfm"),
                    ("t1w_brain_mask", "t1w_brain_mask"),
                    ("t1_aligned_mni", "t1_aligned_mni"),
                    ("gm_mask", "gm_mask"),
                ],
            ),
            (register_node, extract_ts_node, [("t1w_brain_mask", "mask")]),
            (inputnode, node_gen_node, [("ID", "ID")]),
            (
                fetch_nodes_and_labels_node,
                node_gen_node,
                [
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("dir_path", "dir_path"),
                    ("par_max", "par_max"),
                ],
            ),
            (inputnode, extract_ts_node, [("conf", "conf"), ("ID", "ID")]),
            (
                fetch_nodes_and_labels_node,
                save_coords_and_labels_node,
                [("dir_path", "dir_path")],
            ),
            (
                reg_nodes_node,
                save_coords_and_labels_node,
                [("coords", "coords"), ("labels", "labels")],
            ),
            (
                inputnode,
                get_conn_matrix_node,
                [
                    ("dens_thresh", "dens_thresh"),
                    ("ID", "ID"),
                    ("min_span_tree", "min_span_tree"),
                    ("disp_filt", "disp_filt"),
                    ("parc", "parc"),
                    ("prune", "prune"),
                    ("norm", "norm"),
                    ("binary", "binary"),
                ],
            ),
            (
                node_gen_node,
                extract_ts_node,
                [("atlas", "atlas"), ("dir_path", "dir_path")],
            ),
            (
                extract_ts_node,
                get_conn_matrix_node,
                [
                    ("ts_within_nodes", "time_series"),
                    ("dir_path", "dir_path"),
                    ("node_radius", "node_radius"),
                    ("smooth", "smooth"),
                    ("coords", "coords"),
                    ("labels", "labels"),
                    ("atlas", "atlas"),
                    ("parcellation", "parcellation"),
                    ("hpass", "hpass"),
                    ("signal", "signal"),
                    ("roi", "roi"),
                ],
            ),
        ]
    )

    # Check orientation and resolution
    orient_reslice_parcellation_node = pe.Node(
        niu.Function(
            input_names=["infile", "outdir", "vox_size"],
            output_names=["outfile"],
            function=registration.utils.orient_reslice,
            imports=import_list
        ),
        name="orient_reslice_parcellation_node",
    )
    fmri_wf.connect(
        [
            (
                inputnode,
                orient_reslice_anat_node,
                [
                    ("anat_file", "infile"),
                    ("vox_size", "vox_size"),
                    ("outdir", "outdir"),
                ],
            ),
            (
                orient_reslice_anat_node,
                register_node,
                [("outfile", "anat_file")],
            ),
            (
                orient_reslice_anat_node,
                reg_nodes_node,
                [("outfile", "anat_file")],
            ),
            (
                inputnode,
                register_node,
                [("vox_size", "vox_size"), ("template_name", "template_name")],
            ),
            (
                inputnode,
                reg_nodes_node,
                [("vox_size", "vox_size"), ("template_name", "template_name"),
                 ("outdir", "dir_path")],
            ),
            (
                register_node,
                reg_nodes_node,
                [
                    ("reg_fmri_complete", "reg_fmri_complete"),
                    ("basedir_path", "basedir_path"),
                ],
            ),
            (inputnode, orient_reslice_parcellation_node,
             [("vox_size", "vox_size")]),
            (
                fetch_nodes_and_labels_node,
                orient_reslice_parcellation_node,
                [("parcellation", "infile"), ("dir_path", "outdir")],
            ),
            (
                orient_reslice_parcellation_node,
                reg_nodes_node,
                [("outfile", "parcellation")],
            ),
            (
                node_gen_node,
                reg_nodes_node,
                [("coords", "coords"), ("labels", "labels"),
                 ("atlas", "atlas")],
            ),
            (
                reg_nodes_node,
                extract_ts_node,
                [
                    ("aligned_atlas_gm", "parcellation"),
                    ("coords", "coords"),
                    ("labels", "labels"),
                ],
            ),
        ]
    )

    if parc is False:
        # register_node.inputs.simple = True
        fmri_wf.connect(
            [
                (
                    inputnode,
                    prep_spherical_nodes_node,
                    [("template_mask", "template_mask")],
                ),
                (
                    fetch_nodes_and_labels_node,
                    prep_spherical_nodes_node,
                    [("dir_path", "dir_path")],
                ),
            ]
        )

    # Set cpu/memory reqs
    for node_name in fmri_wf.list_node_names():
        if node_name in runtime_dict:
            fmri_wf.get_node(
                node_name).interface.n_procs = runtime_dict[node_name][0]
            fmri_wf.get_node(
                node_name).interface.mem_gb = runtime_dict[node_name][1]
            fmri_wf.get_node(
                node_name).n_procs = runtime_dict[node_name][0]
            fmri_wf.get_node(
                node_name)._mem_gb = runtime_dict[node_name][1]

    # Set runtime/logging configurations
    execution_dict["plugin_args"] = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "mem_thread",
    }
    execution_dict["logging"] = {
        "workflow_level": "INFO",
        "utils_level": "INFO",
        "log_to_file": False,
        "interface_level": "DEBUG",
        "filemanip_level": "DEBUG",
    }
    execution_dict["plugin"] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            fmri_wf.config[key][setting] = value

    gc.collect()
    return fmri_wf


def raw_graph_workflow(
    multi_thr,
    thr,
    multi_graph,
    graph,
    ID,
    subnet,
    conn_model,
    roi,
    prune,
    norm,
    binary,
    min_span_tree,
    dens_thresh,
    disp_filt,
    min_thr,
    max_thr,
    step_thr,
    wf,
    net_mets_node,
    runtime_dict
):
    from pynets.core.utils import load_mat, load_mat_ext, save_mat_thresholded
    from pynets.core.thresholding import thresh_raw_graph

    import_list = [
        "import matplotlib",
        "import warnings",
        "import sys",
        "import os",
        "import numpy as np",
        "import networkx as nx",
        "import nibabel as nib",
        'matplotlib.use("Agg")',
        'warnings.filterwarnings("ignore")',
    ]

    if multi_thr is True or float(thr) != 1.0:
        thresholding_node = pe.Node(
            niu.Function(
                input_names=[
                    "conn_matrix",
                    "thr",
                    "min_span_tree",
                    "dens_thresh",
                    "disp_filt",
                    "est_path",
                ],
                output_names=[
                    "thr_type",
                    "edge_threshold",
                    "conn_matrix_thr",
                    "thr",
                    "est_path",
                ],
                function=thresh_raw_graph,
            ),
            name="thresholding_node",
            imports=import_list
        )

        thr_info_node = pe.Node(
            niu.IdentityInterface(
                fields=[
                    "conn_matrix",
                    "thr",
                    "min_span_tree",
                    "dens_thresh",
                    "disp_filt",
                    "ID",
                    "subnet",
                    "conn_model",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                    "est_path",
                ]
            ),
            name="thr_info_node",
        )

        save_mat_thresholded_node = pe.Node(
            niu.Function(
                input_names=[
                    "conn_matrix",
                    "est_path_orig",
                    "thr_type",
                    "ID",
                    "subnet",
                    "thr",
                    "conn_model",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                ],
                output_names=[
                    "est_path",
                    "ID",
                    "subnet",
                    "thr",
                    "conn_model",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                ],
                function=save_mat_thresholded,
            ),
            name="save_mat_thresholded_node",
            imports=import_list
        )
        save_mat_thresholded_node._n_procs = runtime_dict[
            "save_mat_thresholded_node"][0]
        save_mat_thresholded_node._mem_gb = runtime_dict[
            "save_mat_thresholded_node"][1]

    if multi_graph:
        inputinfo = pe.Node(
            interface=niu.IdentityInterface(
                fields=[
                    "ID",
                    "subnet",
                    "conn_model",
                    "est_path",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                    "thr",
                    "min_span_tree",
                    "dens_thresh",
                    "disp_filt",
                ]
            ),
            name="inputinfo",
        )

        inputinfo.iterables = ("est_path", multi_graph)
        inputinfo.inputs.ID = ID
        inputinfo.inputs.roi = roi
        inputinfo.inputs.thr = thr
        inputinfo.inputs.prune = prune
        inputinfo.inputs.subnet = subnet
        inputinfo.inputs.conn_model = conn_model
        inputinfo.inputs.norm = norm
        inputinfo.inputs.binary = binary
        inputinfo.inputs.min_span_tree = min_span_tree
        inputinfo.inputs.dens_thresh = dens_thresh
        inputinfo.inputs.disp_filt = disp_filt

        join_iters_node_g = pe.JoinNode(
            niu.IdentityInterface(
                fields=[
                    "est_path",
                    "subnet",
                    "ID",
                    "thr",
                    "conn_model",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                ]
            ),
            name="join_iters_node_g",
            joinsource=inputinfo,
            joinfield=[
                "est_path",
                "subnet",
                "ID",
                "thr",
                "conn_model",
                "roi",
                "prune",
                "norm",
                "binary",
            ],
        )
        join_iters_node_g._n_procs = runtime_dict["join_iters_node_g"][0]
        join_iters_node_g._mem_gb = runtime_dict["join_iters_node_g"][1]

        if multi_thr is True or float(thr) != 1.0:
            load_mat_node = pe.Node(
                niu.Function(
                    input_names=[
                        "est_path",
                        "ID",
                        "subnet",
                        "conn_model",
                        "roi",
                        "prune",
                        "norm",
                        "binary",
                        "min_span_tree",
                        "dens_thresh",
                        "disp_filt",
                    ],
                    output_names=[
                        "conn_matrix",
                        "est_path",
                        "ID",
                        "subnet",
                        "conn_model",
                        "roi",
                        "prune",
                        "norm",
                        "binary",
                        "min_span_tree",
                        "dens_thresh",
                        "disp_filt",
                    ],
                    function=load_mat_ext,
                ),
                name="load_mat_ext_node",
                imports=import_list
            )
            load_mat_node._n_procs = runtime_dict["load_mat_ext_node"][0]
            load_mat_node._mem_gb = runtime_dict["load_mat_ext_node"][1]

            wf.connect(
                [
                    (
                        inputinfo,
                        load_mat_node,
                        [
                            ("est_path", "est_path"),
                            ("min_span_tree", "min_span_tree"),
                            ("dens_thresh", "dens_thresh"),
                            ("disp_filt", "disp_filt"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                    (
                        load_mat_node,
                        thr_info_node,
                        [
                            ("conn_matrix", "conn_matrix"),
                            ("est_path", "est_path"),
                            ("min_span_tree", "min_span_tree"),
                            ("dens_thresh", "dens_thresh"),
                            ("disp_filt", "disp_filt"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                ]
            )

    else:
        inputinfo = pe.Node(
            interface=niu.IdentityInterface(
                fields=[
                    "ID",
                    "subnet",
                    "conn_model",
                    "est_path",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                    "thr",
                    "min_span_tree",
                    "dens_thresh",
                    "disp_filt",
                ]
            ),
            name="inputinfo",
        )

        inputinfo.inputs.est_path = graph
        inputinfo.inputs.ID = ID
        inputinfo.inputs.roi = roi
        inputinfo.inputs.thr = thr
        inputinfo.inputs.prune = prune
        inputinfo.inputs.subnet = subnet
        inputinfo.inputs.conn_model = conn_model
        inputinfo.inputs.norm = norm
        inputinfo.inputs.binary = binary
        inputinfo.inputs.min_span_tree = min_span_tree
        inputinfo.inputs.dens_thresh = dens_thresh
        inputinfo.inputs.disp_filt = disp_filt

        if multi_thr is True or float(thr) != 1.0:
            load_mat_node = pe.Node(
                niu.Function(
                    input_names=["est_path"],
                    output_names=["conn_matrix"],
                    function=load_mat,
                ),
                name="load_mat_node",
                imports=import_list
            )
            load_mat_node._n_procs = runtime_dict["load_mat_node"][0]
            load_mat_node._mem_gb = runtime_dict["load_mat_node"][1]

            wf.connect(
                [
                    (inputinfo, load_mat_node, [("est_path", "est_path")]),
                    (load_mat_node, thr_info_node,
                     [("conn_matrix", "conn_matrix")]),
                    (
                        inputinfo,
                        thr_info_node,
                        [
                            ("min_span_tree", "min_span_tree"),
                            ("dens_thresh", "dens_thresh"),
                            ("disp_filt", "disp_filt"),
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                ]
            )

    if multi_thr is True:
        iter_thresh = sorted(
            list(
                set(
                    [
                        str(i)
                        for i in np.round(
                            np.arange(float(min_thr), float(max_thr),
                                      float(step_thr)),
                            decimals=2,
                        ).tolist()
                    ]
                    + [str(float(max_thr))]
                )
            )
        )

        join_iters_node_thr = pe.JoinNode(
            niu.IdentityInterface(
                fields=[
                    "ID",
                    "subnet",
                    "thr",
                    "conn_model",
                    "est_path",
                    "roi",
                    "prune",
                    "norm",
                    "binary",
                ]
            ),
            name="join_iters_node_thr",
            joinsource=thr_info_node,
            joinfield=[
                "ID",
                "subnet",
                "thr",
                "conn_model",
                "est_path",
                "roi",
                "prune",
                "norm",
                "binary",
            ],
        )
        join_iters_node_thr._n_procs = runtime_dict["join_iters_node_thr"][0]
        join_iters_node_thr._mem_gb = runtime_dict["join_iters_node_thr"][1]

        thr_info_node.iterables = ("thr", iter_thresh)
        thr_info_node.synchronize = True
        wf.connect(
            [
                (
                    thr_info_node,
                    thresholding_node,
                    [
                        ("thr", "thr"),
                        ("min_span_tree", "min_span_tree"),
                        ("dens_thresh", "dens_thresh"),
                        ("disp_filt", "disp_filt"),
                        ("conn_matrix", "conn_matrix"),
                        ("est_path", "est_path"),
                    ],
                ),
                (
                    thr_info_node,
                    save_mat_thresholded_node,
                    [
                        ("subnet", "subnet"),
                        ("ID", "ID"),
                        ("conn_model", "conn_model"),
                        ("roi", "roi"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
                (
                    thresholding_node,
                    save_mat_thresholded_node,
                    [
                        ("est_path", "est_path_orig"),
                        ("conn_matrix_thr", "conn_matrix"),
                        ("thr_type", "thr_type"),
                        ("thr", "thr"),
                    ],
                ),
                (
                    save_mat_thresholded_node,
                    join_iters_node_thr,
                    [
                        ("est_path", "est_path"),
                        ("subnet", "subnet"),
                        ("ID", "ID"),
                        ("thr", "thr"),
                        ("conn_model", "conn_model"),
                        ("roi", "roi"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
            ]
        )
        if multi_graph:
            wf.connect(
                [
                    (
                        join_iters_node_thr,
                        join_iters_node_g,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                    (
                        join_iters_node_g,
                        net_mets_node,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                ]
            )
        else:
            wf.connect(
                [
                    (
                        join_iters_node_thr,
                        net_mets_node,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    )
                ]
            )
    elif float(thr) != 1.0:
        thr_info_node.iterables = ("thr", [thr])
        wf.connect(
            [
                (
                    thr_info_node,
                    thresholding_node,
                    [
                        ("thr", "thr"),
                        ("min_span_tree", "min_span_tree"),
                        ("dens_thresh", "dens_thresh"),
                        ("disp_filt", "disp_filt"),
                        ("conn_matrix", "conn_matrix"),
                        ("est_path", "est_path"),
                    ],
                ),
                (
                    thr_info_node,
                    save_mat_thresholded_node,
                    [
                        ("subnet", "subnet"),
                        ("ID", "ID"),
                        ("conn_model", "conn_model"),
                        ("roi", "roi"),
                        ("prune", "prune"),
                        ("norm", "norm"),
                        ("binary", "binary"),
                    ],
                ),
                (
                    thresholding_node,
                    save_mat_thresholded_node,
                    [
                        ("est_path", "est_path_orig"),
                        ("conn_matrix_thr", "conn_matrix"),
                        ("thr_type", "thr_type"),
                        ("thr", "thr"),
                    ],
                ),
            ]
        )
        if multi_graph:
            wf.connect(
                [
                    (
                        save_mat_thresholded_node,
                        join_iters_node_g,
                        [("est_path", "est_path")],
                    ),
                    (
                        thr_info_node,
                        join_iters_node_g,
                        [
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                    (
                        join_iters_node_g,
                        net_mets_node,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                ]
            )
        else:
            wf.connect(
                [
                    (
                        thr_info_node,
                        net_mets_node,
                        [
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                    (
                        save_mat_thresholded_node,
                        net_mets_node,
                        [("est_path", "est_path")],
                    ),
                ]
            )
    else:
        if multi_graph:
            wf.connect(
                [
                    (
                        inputinfo,
                        join_iters_node_g,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                    (
                        join_iters_node_g,
                        net_mets_node,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    ),
                ]
            )
        else:
            wf.connect(
                [
                    (
                        inputinfo,
                        net_mets_node,
                        [
                            ("est_path", "est_path"),
                            ("subnet", "subnet"),
                            ("ID", "ID"),
                            ("thr", "thr"),
                            ("conn_model", "conn_model"),
                            ("roi", "roi"),
                            ("prune", "prune"),
                            ("norm", "norm"),
                            ("binary", "binary"),
                        ],
                    )
                ]
            )

    gc.collect()
    return wf
