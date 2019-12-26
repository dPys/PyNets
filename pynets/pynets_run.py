#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
warnings.filterwarnings("ignore")


def get_parser():
    """Parse command-line inputs"""
    import argparse
    # Parse args
    parser = argparse.ArgumentParser(description='PyNets: A Fully-Automated Workflow for Reproducible Ensemble '
                                                 'Graph Analysis of Functional and Structural Connectomes')
    parser.add_argument('-id',
                        metavar='A subject id or other unique identifier',
                        default=None,
                        nargs='+',
                        required=True,
                        help='An subject identifier OR list of subject identifiers, separated by space and of '
                             'equivalent length to the list of input files indicated with the -func flag. This '
                             'parameter must be an alphanumeric string and can be arbitrarily chosen. If functional '
                             'and dmri connectomes are being generated simultaneously, then space-separated id\'s '
                             'need to be repeated to match the total input file count.\n')
    parser.add_argument('-mod',
                        metavar='Connectivity estimation/reconstruction method',
                        default='partcorr',
                        required=True,
                        nargs='+',
                        choices=['corr', 'sps', 'cov', 'partcorr', 'QuicGraphicalLasso', 'QuicGraphicalLassoCV',
                                 'QuicGraphicalLassoEBIC', 'AdaptiveQuicGraphicalLasso', 'csa', 'csd'],
                        help='Specify connectivity estimation model. For fMRI, possible models include: '
                             'corr for correlation, cov for covariance, sps for precision covariance, partcorr for '
                             'partial correlation. sps type is used by default. '
                             'If skgmm is installed (https://github.com/skggm/skggm), then QuicGraphicalLasso, '
                             'QuicGraphicalLassoCV, QuicGraphicalLassoEBIC, and AdaptiveQuicGraphicalLasso. '
                             'Default is partcorr for fMRI. For dMRI, models include csa and csd.\n')
    parser.add_argument('-g',
                        metavar='Path to graph file input.',
                        default=None,
                        nargs='+',
                        help='In either .txt or .npy format. This skips fMRI and dMRI graph estimation workflows and '
                             'begins at the graph analysis stage. Multiple graph files should be separated by space.\n')
    parser.add_argument('-func',
                        metavar='Path to input functional file (required for functional connectomes)',
                        default=None,
                        nargs='+',
                        help='Specify either a path to a preprocessed functional Nifti1Image in '
                             'MNI152 space OR multiple space-separated paths to multiple preprocessed functional '
                             'Nifti1Image files in MNI152 space and in .nii or .nii.gz format, '
                             'OR the path to a text file containing a list of paths '
                             'to subject files.\n')
    parser.add_argument('-conf',
                        metavar='Confound regressor file (.tsv/.csv format)',
                        default=None,
                        nargs='+',
                        help='Optionally specify a path to a confound regressor file to reduce noise in the '
                             'time-series estimation for the graph. This can also be a list of paths in the case of '
                             'running multiple subjects, which requires separation by space and of equivalent length '
                             'to the list of input files indicated with the -func flag.\n')
    parser.add_argument('-dwi',
                        metavar='Path to diffusion-weighted imaging data file (required for dmri connectomes)',
                        default=None,
                        nargs='+',
                        help='Specify either a path to a preprocessed dmri diffusion Nifti1Image in native diffusion '
                             'space and in .nii or .nii.gz format OR multiple space-separated paths to multiple '
                             'preprocessed dmri diffusion Nifti1Image files in native diffusion space and in .nii or '
                             '.nii.gz format.\n')
    parser.add_argument('-bval',
                        metavar='Path to b-values file (required for dmri connectomes)',
                        default=None,
                        nargs='+',
                        help='Specify either a path to a b-values text file containing gradient shell values per '
                             'diffusion direction OR multiple space-separated paths to multiple b-values text files in '
                             'the order of accompanying b-vectors and dwi files.\n')
    parser.add_argument('-bvec',
                        metavar='Path to b-vectors file (required for dmri connectomes)',
                        default=None,
                        nargs='+',
                        help='Specify either a path to a b-vectors text file containing gradient directions (x,y,z) '
                             'per diffusion direction OR multiple space-separated paths to multiple b-vectors text '
                             'files in the order of accompanying b-values and dwi files.\n')
    parser.add_argument('-anat',
                        metavar='Path to a skull-stripped anatomical Nifti1Image',
                        default=None,
                        nargs='+',
                        help='Required for dmri and/or functional connectomes. Multiple paths to multiple '
                             'anatomical files should be specified by space in the order of accompanying functional '
                             'and/or dmri files. If functional and dmri connectomes are both being generated '
                             'simultaneously, then anatomical Nifti1Image file paths need to be repeated, '
                             'but separated by comma.\n')
    parser.add_argument('-m',
                        metavar='Path to binarized mask Nifti1Image to apply to regions before extracting signals',
                        default=None,
                        nargs='+',
                        help='Specify either a path to a binarized brain mask Nifti1Image in MNI152 space '
                             'OR multiple paths to multiple brain mask Nifti1Image files in the case of running '
                             'multiple participants, in which case paths should be separated by a space. If no brain '
                             'mask is supplied, a default MNI152 template mask will be used\n')
    parser.add_argument('-roi',
                        metavar='Path to binarized Region-of-Interest (ROI) Nifti1Image',
                        default=None,
                        nargs='+',
                        help='Optionally specify a binarized ROI mask and retain only those nodes '
                             'of a parcellation contained within that mask for connectome estimation.\n')
    parser.add_argument('-way',
                        metavar='Path to binarized Nifti1Image to constrain tractography',
                        default=None,
                        nargs='+',
                        help='Optionally specify a binarized ROI mask in MNI-space to constrain tractography in the '
                             'case of dmri connectome estimation.\n')
    parser.add_argument('-cm',
                        metavar='Cluster mask',
                        default=None,
                        nargs='+',
                        help='Specify the path to a Nifti1Image mask file to constrained functional clustering. '
                             'If specifying a list of paths to multiple cluster masks, separate '
                             'them by space.\n')
    parser.add_argument('-ua',
                        metavar='Path to parcellation file in MNI-space',
                        default=None,
                        nargs='+',
                        help='Optionally specify a path to a parcellation/atlas Nifti1Image file in MNI152 space. '
                             'Labels should be spatially distinct across hemispheres and ordered with consecutive '
                             'integers with a value of 0 as the background label. If specifying a list of paths to '
                             'multiple user atlases, separate them by space.\n')
    parser.add_argument('-templ',
                        metavar='Path to template file',
                        default=None,
                        help='Optionally specify a path to a template Nifti1Image file. If none is specified, then '
                             'will use the MNI152 template by default.\n')
    parser.add_argument('-templm',
                        metavar='Path to template mask file',
                        default=None,
                        help='Optionally specify a path to a template mask Nifti1Image file. If none is specified, '
                             'then will use the MNI152 template mask by default.\n')
    parser.add_argument('-ref',
                        metavar='Atlas reference file path',
                        default=None,
                        help='Specify the path to the atlas reference .txt file that maps labels to '
                             'intensities corresponding to the atlas parcellation file specified with the -ua flag.\n')
    parser.add_argument('-a',
                        metavar='Atlas',
                        default=None,
                        nargs='+',
                        choices=['atlas_aal', 'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe',
                                 'atlas_harvard_oxford', 'atlas_destrieux_2009', 'atlas_msdl', 'coords_dosenbach_2010',
                                 'coords_power_2011', 'atlas_pauli_2017'],
                        help='Specify a coordinate atlas parcellation from those made publically available in nilearn. '
                             'If you wish to iterate your pynets run over multiple nilearn atlases, separate them by '
                             'space. Available nilearn atlases are:'
                             '\n\natlas_aal\natlas_talairach_gyrus\natlas_talairach_ba\natlas_talairach_lobe\n'
                             'atlas_harvard_oxford\natlas_destrieux_2009\natlas_msdl\ncoords_dosenbach_2010\n'
                             'coords_power_2011\natlas_pauli_2017.\n')
    parser.add_argument('-spheres',
                        default=False,
                        action='store_true',
                        help='Include this flag to use spheres instead of parcels as nodes.\n')
    parser.add_argument('-names',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to perform automated anatomical labeling of '
                             'nodes.\n')
    parser.add_argument('-ns',
                        metavar='Spherical centroid node size',
                        default=4,
                        nargs='+',
                        help='Optionally specify coordinate-based node radius size(s). Default is 4 mm for fMRI and '
                             '8mm for dMRI. If you wish to iterate the pipeline across multiple node sizes, separate '
                             'the list by space (e.g. 2 4 6).\n')
    parser.add_argument('-k',
                        metavar='Number of k clusters',
                        default=None,
                        help='Specify a number of clusters to produce.\n')
    parser.add_argument('-k_min',
                        metavar='Min k clusters',
                        default=None,
                        help='Specify the minimum k clusters.\n')
    parser.add_argument('-k_max',
                        metavar='Max k clusters',
                        default=None,
                        help='Specify the maximum k clusters.\n')
    parser.add_argument('-k_step',
                        metavar='K cluster step size',
                        default=None,
                        help='Specify the step size of k cluster iterables.\n')
    parser.add_argument('-ct',
                        metavar='Clustering type',
                        default='ward',
                        nargs='+',
                        choices=['ward', 'kmeans', 'complete', 'average', 'single'],
                        help='Specify the types of clustering to use. Recommended options are: '
                             'ward or kmeans. Note that imposing spatial constraints with a mask consisting of '
                             'disconnected components will leading to clustering instability in the case of complete, '
                             'average, or single clustering. If specifying a list of '
                             'clustering types, separate them by space.\n')
    parser.add_argument('-cc',
                        metavar='Clustering connectivity type',
                        default='allcorr',
                        nargs=1,
                        choices=['tcorr', 'scorr', 'allcorr'],
                        help='Include this flag if you are running agglomerative-type clustering and wish to specify a '
                             'spatially constrained connectivity method based on tcorr or scorr. Default is allcorr '
                             'which has no spatial constraints.\n')
    parser.add_argument('-n',
                        metavar='Resting-state network',
                        default=None,
                        nargs='+',
                        choices=['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'VisCent',
                                 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA',
                                 'SalVentAttnB', 'LimbicOFC', 'LimbicTempPole', 'ContA', 'ContB', 'ContC', 'DefaultA',
                                 'DefaultB', 'DefaultC', 'TempPar'],
                        help='Optionally specify the name of any of the 2017 Yeo-Schaefer RSNs (7-network or '
                             '17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, '
                             'VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, '
                             'LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar. If listing '
                             'multiple RSNs, separate them by space. (e.g. -n \'Default\' \'Cont\' \'SalVentAttn\')\'.'
                             '\n')
    parser.add_argument('-sm',
                        metavar='Smoothing value (mm fwhm)',
                        default=0,
                        nargs='+',
                        help='Optionally specify smoothing width(s). Default is 0 / no smoothing. '
                             'If you wish to iterate the pipeline across multiple smoothing '
                             'separate the list by space (e.g. 2 4 6).\n')
    parser.add_argument('-hp',
                        metavar='High-pass filter (Hz)',
                        default=None,
                        nargs='+',
                        help='Optionally specify high-pass filter values to apply to node-extracted time-series '
                             'for fMRI. Default is None. If you wish to iterate the pipeline across multiple high-pass '
                             'filter thresholds, values, separate the list by space (e.g. 0.008 0.01).\n')
    parser.add_argument('-b',
                        metavar='Number of bootstraps (integer)',
                        default=0,
                        nargs='+',
                        help='Optionally specify the number of bootstraps with this flag if you wish to apply '
                             'circular-block bootstrapped resampling of the node-extracted time-series. Size of '
                             'blocks can be specified using the -bs flag.\n')
    parser.add_argument('-bs',
                        metavar='Size bootstrap blocks (integer)',
                        default=None,
                        nargs='+',
                        help='If using the -b flag, you may manually specify a bootstrap block size for circular-block '
                             'resampling of the node-extracted time-series. sqrt(TR) rounded to the nearest integer is '
                             'recommended\n')
    parser.add_argument('-p',
                        metavar='Pruning strategy',
                        default=1,
                        nargs=1,
                        choices=['0', '1', '2'],
                        help='Include this flag to prune the resulting graph of (1) any isolated + fully '
                             'disconnected nodes or (2) any isolated + fully disconnected + non-important nodes. '
                             'Default pruning=1. Include -p 0 to disable pruning.\n')
    parser.add_argument('-bin',
                        default=False,
                        action='store_true',
                        help='Include this flag to binarize the resulting graph such that edges are boolean and not '
                             'weighted.\n')
    parser.add_argument('-s',
                        metavar='Number of samples',
                        default='1000000',
                        help='Include this flag to manually specify a number of cumulative streamline samples for '
                             'tractography. Default is 1000000.\n')
    parser.add_argument('-ml',
                        metavar='Maximum fiber length for tracking',
                        default=200,
                        help='Include this flag to manually specify a maximum tract length (mm) for dmri '
                             'connectome tracking. Default is 200.\n')
    parser.add_argument('-tt',
                        metavar='Tracking algorithm',
                        default='local',
                        nargs=1,
                        choices=['local', 'particle'],
                        help='Include this flag to manually specify a tracking algorithm for dmri connectome '
                             'estimation. Options are: local and particle. Default is local.\n')
    parser.add_argument('-dg',
                        metavar='Direction getter',
                        default='det',
                        nargs='+',
                        choices=['det', 'prob', 'clos', 'boot'],
                        help='Include this flag to manually specify the statistical approach to tracking for dmri '
                             'connectome estimation. Options are: det (deterministic), closest (clos), '
                             'boot (bootstrapped), and prob (probabilistic). '
                             'Default is det.\n')
    parser.add_argument('-tc',
                        metavar='Tissue classification method',
                        default='bin',
                        nargs=1,
                        choices=['wb', 'cmc', 'act', 'bin'],
                        help='Include this flag to manually specify a tissue classification method for dmri '
                             'connectome estimation. Options are: cmc (continuous), act (anatomically-constrained), '
                             'wb (whole-brain mask), and bin (binary to white-matter only). Default is bin.\n')
    parser.add_argument('-thr',
                        metavar='Graph threshold',
                        default=1.00,
                        help='Optionally specify a threshold indicating a proportion of weights to preserve in the '
                             'graph. Default is proportional thresholding. If omitted, no thresholding will be applied.'
                             '\n')
    parser.add_argument('-min_thr',
                        metavar='Multi-thresholding minimum threshold',
                        default=None,
                        help='Minimum threshold for multi-thresholding.\n')
    parser.add_argument('-max_thr',
                        metavar='Multi-thresholding maximum threshold',
                        default=None,
                        help='Maximum threshold for multi-thresholding.')
    parser.add_argument('-step_thr',
                        metavar='Multi-thresholding step size',
                        default=None,
                        help='Threshold step value for multi-thresholding. Default is 0.01.\n')
    parser.add_argument('-norm',
                        metavar='Normalization strategy for resulting graph(s)',
                        default=0,
                        nargs=1,
                        choices=['0', '1', '2', '3', '4', '5', '6'],
                        help='Include this flag to normalize the resulting graph by (1) maximum edge weight; '
                             '(2) using log10; (3) using pass-to-ranks for all non-zero edges; '
                             '(4) using pass-to-ranks for all non-zero edges relative to the number of nodes; (5) '
                             'using pass-to-ranks with zero-edge boost; and (6) which standardizes the matrix to '
                             'values [0, 1]. Default is (0) which is no normalization.\n')
    parser.add_argument('-dt',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to threshold to achieve a given density or '
                             'densities indicated by the -thr and -min_thr, -max_thr, -step_thr flags, respectively.\n')
    parser.add_argument('-mst',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to apply local thresholding via the Minimum '
                             'Spanning Tree approach. -thr values in this case correspond to a target density (if the '
                             '-dt flag is also included), otherwise a target proportional threshold.\n')
    parser.add_argument('-df',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to apply local thresholding via the disparity '
                             'filter approach. -thr values in this case correspond to α.\n')
    parser.add_argument('-mplx',
                        metavar='Perform various levels of multiplex graph analysis if both structural and diffusion '
                                'connectomes are provided.',
                        default=0,
                        nargs=1,
                        choices=['0', '1', '2', '3'],
                        help='Include this flag to perform multiplex graph analysis across structural-functional '
                             'connectome modalities. Options include level (1) Create and ensemble of multiplex graphs '
                             'using motif-matched adaptive thresholding; (2) Additionally perform multiplex graph '
                             'embedding and analysis; (3) Additionally perform plotting. '
                             'Default is (0) which is no multiplex analysis.\n')
    parser.add_argument('-embed',
                        default=None,
                        nargs=1,
                        choices=[None, 'omni', 'mase'],
                        help='Optionally use this flag if you wish to embed the ensemble(s) produced into '
                             'feature vector(s). Options include: omni or mase. Default is None.\n')
    parser.add_argument('-vox',
                        default='2mm',
                        nargs=1,
                        choices=['1mm', '2mm'],
                        help='Optionally use this flag if you wish to change the resolution of the images in the '
                             'workflow. Default is 2mm.\n')
    parser.add_argument('-plt',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to activate plotting of adjacency matrices, '
                             'connectomes, and time-series.\n')
    parser.add_argument('-pm',
                        metavar='Cores,memory',
                        default='2,4',
                        help='Number of cores to use, number of GB of memory to use for single subject run, entered as '
                             'two integers seperated by a comma.\n')
    parser.add_argument('-plug',
                        metavar='Scheduler type',
                        default='MultiProc',
                        nargs=1,
                        choices=['Linear', 'MultiProc', 'SGE', 'PBS', 'SLURM', 'SGEgraph', 'SLURMgraph',
                                 'LegacyMultiProc'],
                        help='Include this flag to specify a workflow plugin other than the default MultiProc.\n')
    parser.add_argument('-v',
                        default=False,
                        action='store_true',
                        help='Verbose print for debugging.\n')
    parser.add_argument('-work',
                        metavar='Working directory',
                        default='/tmp/work',
                        help='Specify the path to a working directory for pynets to run. Default is /tmp/work.\n')
    return parser


def build_workflow(args, retval):
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import glob
    import ast
    import os.path as op
    import sys
    import timeit
    import numpy as np
    from pathlib import Path
    import yaml
    import datetime
    try:
        import pynets
        print("%s%s%s" % ('\n\nPyNets Version:\n', pynets.__version__, '\n\n'))
    except ImportError:
        print('PyNets not installed! Ensure that you are using the correct python version.')
    from pynets.core.utils import do_dir_path

    # Start timer
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S\n\n"))
    start_time = timeit.default_timer()

    # Set Arguments to global variables
    func_file = args.func
    mask = args.m
    dwi_file = args.dwi
    fbval = args.bval
    fbvec = args.bvec
    graph = args.g
    if graph:
        if len(graph) > 1:
            multi_graph = graph
            graph = None
        elif graph == ['None']:
            graph = None
            multi_graph = None
        else:
            graph = graph[0]
            multi_graph = None
    else:
        multi_graph = None
    ID = args.id
    resources = args.pm
    if resources:
        procmem = list(eval(str(resources)))
    else:
        from multiprocessing import cpu_count
        nthreads = cpu_count()
        procmem = [int(nthreads), int(float(nthreads) * 2)]
    thr = float(args.thr)
    node_size = args.ns
    if node_size:
        if (type(node_size) is list) and (len(node_size) > 1):
            node_size_list = node_size
            node_size = None
        elif node_size == ['None']:
            node_size = None
            node_size_list = None
        elif type(node_size) is list:
            node_size = node_size[0]
            node_size_list = None
        else:
            node_size = None
            node_size_list = None
    else:
        node_size_list = None
    smooth = args.sm
    if smooth:
        if (type(smooth) is list) and (len(smooth) > 1):
            smooth_list = smooth
            smooth = 0
        elif smooth == ['None']:
            smooth = 0
            smooth_list = None
        elif type(smooth) is list:
            smooth = smooth[0]
            smooth_list = None
        else:
            smooth = 0
            smooth_list = None
    else:
        smooth_list = None
    hpass = args.hp
    if hpass:
        if (type(hpass) is list) and (len(hpass) > 1):
            hpass_list = hpass
            hpass = None
        elif hpass == ['None']:
            hpass = None
            hpass_list = None
        elif type(hpass) is list:
            hpass = hpass[0]
            hpass_list = None
        else:
            hpass = None
            hpass_list = None
    else:
        hpass_list = None
    c_boot = args.b
    block_size = args.bs
    roi = args.roi
    template = args.templ
    template_mask = args.templm
    conn_model = args.mod
    if conn_model:
        if (type(conn_model) is list) and (len(conn_model) > 1):
            conn_model_list = conn_model
        elif conn_model == ['None']:
            conn_model_list = None
        elif type(conn_model) is list:
            conn_model = conn_model[0]
            conn_model_list = None
        else:
            conn_model_list = None
    else:
        conn_model_list = None
    conf = args.conf
    dens_thresh = args.dt
    min_span_tree = args.mst
    disp_filt = args.df
    clust_type = args.ct
    if clust_type:
        if (type(clust_type) is list) and len(clust_type) > 1:
            clust_type_list = clust_type
            clust_type = None
        elif clust_type == ['None']:
            clust_type = None
            clust_type_list = None
        elif type(clust_type) is list:
            clust_type = clust_type[0]
            clust_type_list = None
        else:
            clust_type = None
            clust_type_list = None
    else:
        clust_type_list = None
    local_corr = args.cc
    if type(local_corr) is list:
        local_corr = local_corr[0]
    # adapt_thresh=args.at
    adapt_thresh = False
    plot_switch = args.plt
    min_thr = args.min_thr
    max_thr = args.max_thr
    step_thr = args.step_thr
    anat_file = args.anat
    num_total_samples = args.s
    spheres = args.spheres
    if spheres is True:
        parc = False
    else:
        parc = True

    if parc is True:
        node_size = None
        node_size_list = None
    else:
        if node_size:
            if node_size is None:
                if (func_file is not None) and (dwi_file is None):
                    node_size = 4
                elif (func_file is None) and (dwi_file is not None):
                    node_size = 8
    ref_txt = args.ref
    k = args.k
    k_min = args.k_min
    k_max = args.k_max
    k_step = args.k_step
    prune = args.p
    if type(prune) is list:
        prune = prune[0]
    norm = args.norm
    if type(norm) is list:
        norm = norm[0]
    binary = args.bin
    plugin_type = args.plug
    if type(plugin_type) is list:
        plugin_type = plugin_type[0]
    use_AAL_naming = args.names
    verbose = args.v
    clust_mask = args.cm
    if clust_mask:
        if len(clust_mask) > 1:
            clust_mask_list = clust_mask
            clust_mask = None
        elif clust_mask == ['None']:
            clust_mask = None
            clust_mask_list = None
        else:
            clust_mask = clust_mask[0]
            clust_mask_list = None
    else:
        clust_mask_list = None
    waymask = args.way
    if isinstance(waymask, list):
        waymask = waymask[0]
    network = args.n
    if network:
        if (type(network) is list) and (len(network) > 1):
            multi_nets = network
            network = None
        elif network == ['None']:
            network = None
            multi_nets = None
        elif type(network) is list:
            network = network[0]
            multi_nets = None
        else:
            network = None
            multi_nets = None
    else:
        multi_nets = None
    uatlas = args.ua
    if uatlas:
        if len(uatlas) > 1:
            user_atlas_list = uatlas
            uatlas = None
        elif uatlas == ['None']:
            uatlas = None
            user_atlas_list = None
        else:
            uatlas = uatlas[0]
            user_atlas_list = None
    else:
        user_atlas_list = None
    atlas = args.a
    if atlas:
        if (type(atlas) is list) and (len(atlas) > 1):
            multi_atlas = atlas
            atlas = None
        elif atlas == ['None']:
            multi_atlas = None
            atlas = None
        elif type(atlas) is list:
            atlas = atlas[0]
            multi_atlas = None
        else:
            atlas = None
            multi_atlas = None
    else:
        multi_atlas = None
    target_samples = args.s
    max_length = args.ml
    track_type = args.tt
    if type(track_type) is list:
        track_type = track_type[0]
    tiss_class = args.tc
    if type(tiss_class) is list:
        tiss_class = tiss_class[0]
    if track_type == 'particle':
        tiss_class = 'cmc'
    directget = args.dg
    if directget:
        if (type(directget) is list) and (len(directget) > 1):
            multi_directget = directget
        elif type(directget) is list:
            directget = directget[0]
            multi_directget = None
        else:
            multi_directget = None
    else:
        multi_directget = None
    embed = args.embed
    if embed is not None:
        embed = embed[0]
    multiplex = args.mplx
    vox_size = args.vox
    work_dir = args.work
    os.makedirs(work_dir, exist_ok=True)

    print('\n\n\n------------------------------------------------------------------------\n')

    # Hard-coded:
    with open("%s%s" % (str(Path(__file__).parent), '/runconfig.yaml'), 'r') as stream:
        try:
            hardcoded_params = yaml.load(stream)
            maxcrossing = hardcoded_params['maxcrossing'][0]
            min_length = hardcoded_params['min_length'][0]
            overlap_thr = hardcoded_params['overlap_thr'][0]
            overlap_thr_list = hardcoded_params['overlap_thr_list'][0]
            step_list = hardcoded_params['step_list']
            curv_thr_list = hardcoded_params['curv_thr_list']
            nilearn_parc_atlases = hardcoded_params['nilearn_parc_atlases']
            nilearn_coord_atlases = hardcoded_params['nilearn_coord_atlases']
            nilearn_prob_atlases = hardcoded_params['nilearn_prob_atlases']
            runtime_dict = {}
            execution_dict = {}
            for i in range(len(hardcoded_params['resource_dict'])):
                runtime_dict[list(hardcoded_params['resource_dict'][i].keys())[0]] = ast.literal_eval(list(
                    hardcoded_params['resource_dict'][i].values())[0][0])
            for i in range(len(hardcoded_params['execution_dict'])):
                execution_dict[list(hardcoded_params['execution_dict'][i].keys())[0]] = list(
                    hardcoded_params['execution_dict'][i].values())[0][0]
        except FileNotFoundError:
            print('Failed to parse runconfig.yaml')

    if (min_thr is not None) and (max_thr is not None) and (step_thr is not None):
        multi_thr = True
    elif (min_thr is not None) or (max_thr is not None) or (step_thr is not None):
        raise ValueError('Error: Missing either min_thr, max_thr, or step_thr flags!')
    else:
        multi_thr = False

    # Check required inputs for existence, and configure run
    if (func_file is None) and (dwi_file is None) and (graph is None) and (multi_graph is None):
        raise ValueError("\nError: You must include a file path to either an MNI152-normalized space functional image "
                         "in .nii or .nii.gz format with the -func flag.")

    if func_file:
        if isinstance(func_file, list) and len(func_file) > 1:
            func_file_list = func_file
            func_file = None
        elif isinstance(func_file, list):
            func_file = func_file[0]
            func_file_list = None
        elif func_file.endswith('.txt'):
            with open(func_file) as f:
                func_file_list = f.read().splitlines()
            func_file = None
        else:
            func_file = None
            func_file_list = None
    else:
        func_file_list = None

    if dwi_file and (not anat_file and not fbval and not fbvec):
        raise ValueError('ERROR: Anatomical image(s) (-anat), b-values file(s) (-fbval), and b-vectors file(s) '
                         '(-fbvec) must be specified for dmri_connectometry.')

    if dwi_file:
        if isinstance(dwi_file, list) and len(dwi_file) > 1:
            dwi_file_list = dwi_file
            dwi_file = None
        elif isinstance(dwi_file, list):
            dwi_file = dwi_file[0]
            dwi_file_list = None
        elif dwi_file.endswith('.txt'):
            with open(dwi_file) as f:
                dwi_file_list = f.read().splitlines()
            dwi_file = None
        else:
            dwi_file = None
            dwi_file_list = None
    else:
        dwi_file_list = None
        track_type = None
        tiss_class = None
        directget = None

    if (ID is None) and (func_file_list is None):
        raise ValueError("\nError: You must include a subject ID in your command line call.")

    if func_file_list and isinstance(ID, list):
        if len(ID) != len(func_file_list):
            raise ValueError("Error: Length of ID list does not correspond to length of input func file list.")

    if isinstance(ID, list) and len(ID) == 1:
        ID = ID[0]

    if conf:
        if isinstance(conf, list) and func_file_list:
            if len(conf) != len(func_file_list):
                raise ValueError("Error: Length of confound regressor list does not correspond to length of input file "
                                 "list.")
            else:
                conf_list = conf
                conf = None
        elif isinstance(conf, list):
            conf = conf[0]
            conf_list = None
        elif conf.endswith('.txt'):
            with open(conf) as f:
                conf_list = f.read().splitlines()
            conf = None
        else:
            conf = None
            conf_list = None
    else:
        conf_list = None

    if dwi_file_list and isinstance(ID, list):
        if len(ID) != len(dwi_file_list):
            raise ValueError("Error: Length of ID list does not correspond to length of input dwi file list.")

    if fbval:
        if isinstance(fbval, list) and dwi_file_list:
            if len(fbval) != len(dwi_file_list):
                raise ValueError("Error: Length of fbval list does not correspond to length of input dwi file list.")
            else:
                fbval_list = fbval
                fbval = None
        elif isinstance(fbval, list):
            fbval = fbval[0]
            fbval_list = None
        elif fbval.endswith('.txt'):
            with open(fbval) as f:
                fbval_list = f.read().splitlines()
            fbval = None
        else:
            fbval = None
            fbval_list = None
    else:
        fbval_list = None

    if fbvec:
        if isinstance(fbvec, list) and dwi_file_list:
            if len(fbvec) != len(dwi_file_list):
                raise ValueError("Error: Length of fbvec list does not correspond to length of input dwi file list.")
            else:
                fbvec_list = fbvec
                fbvec = None
        elif isinstance(fbvec, list):
            fbvec = fbvec[0]
            fbvec_list = None
        elif fbvec.endswith('.txt'):
            with open(fbvec) as f:
                fbvec_list = f.read().splitlines()
            fbvec = None
        else:
            fbvec = None
            fbvec_list = None
    else:
        fbvec_list = None

    if anat_file:
        if isinstance(anat_file, list) and dwi_file_list and func_file_list:
            if len(anat_file) != len(dwi_file_list) and len(anat_file) != len(dwi_file_list):
                raise ValueError("Error: Length of anat list does not correspond to length of input dwi and func file "
                                 "lists.")
            else:
                anat_file_list = anat_file
                anat_file = None
        elif isinstance(anat_file, list) and dwi_file_list:
            if len(anat_file) != len(dwi_file_list):
                raise ValueError("Error: Length of anat list does not correspond to length of input dwi file list.")
            else:
                anat_file_list = anat_file
                anat_file = None
        elif isinstance(anat_file, list) and func_file_list:
            if len(anat_file) != len(func_file_list):
                raise ValueError("Error: Length of anat list does not correspond to length of input func file list.")
            else:
                anat_file_list = anat_file
                anat_file = None
        elif isinstance(anat_file, list):
            anat_file = anat_file[0]
            anat_file_list = None
        else:
            anat_file_list = None
            anat_file = None
    else:
        anat_file_list = None

    if mask:
        if isinstance(mask, list) and func_file_list and dwi_file_list:
            if len(mask) != len(func_file_list) and len(mask) != len(dwi_file_list):
                raise ValueError("Error: Length of brain mask list does not correspond to length of input func "
                                 "and dwi file lists.")
            else:
                mask_list = mask
                mask = None
        elif isinstance(mask, list) and func_file_list:
            if len(mask) != len(func_file_list):
                raise ValueError("Error: Length of brain mask list does not correspond to length of input func "
                                 "file list.")
            else:
                mask_list = mask
                mask = None
        elif isinstance(mask, list) and dwi_file_list:
            if len(mask) != len(dwi_file_list):
                raise ValueError("Error: Length of brain mask list does not correspond to length of input dwi "
                                 "file list.")
            else:
                mask_list = mask
                mask = None
        elif isinstance(mask, list):
            mask = mask[0]
            mask_list = None
        else:
            mask_list = None
            mask = None
    else:
        mask_list = None

    if multi_thr is True:
        thr = None
    else:
        min_thr = None
        max_thr = None
        step_thr = None

    if (k_min is not None) and (k_max is not None) and (k is None) and (clust_mask_list is not
                                                                        None) and (clust_type_list is not None):
        k_clustering = 8
    elif (k is not None) and (k_min is None) and (k_max is None) and (clust_mask_list is not
                                                                      None) and (clust_type_list is not None):
        k_clustering = 7
    elif (k_min is not None) and (k_max is not None) and (k is None) and (clust_mask_list is
                                                                          None) and (clust_type_list is not None):
        k_clustering = 6
    elif (k is not None) and (k_min is None) and (k_max is None) and (clust_mask_list is
                                                                      None) and (clust_type_list is not None):
        k_clustering = 5
    elif (k_min is not None) and (k_max is not None) and (k is None) and (clust_mask_list is not
                                                                          None) and (clust_type_list is None):
        k_clustering = 4
    elif (k is not None) and (k_min is None) and (k_max is None) and (clust_mask_list is not
                                                                      None) and (clust_type_list is None):
        k_clustering = 3
    elif (k_min is not None) and (k_max is not None) and (k is None) and (clust_mask_list is
                                                                          None) and (clust_type_list is None):
        k_clustering = 2
    elif (k is not None) and (k_min is None) and (k_max is None) and (clust_mask_list is
                                                                      None) and (clust_type_list is None):
        k_clustering = 1
    else:
        k_clustering = 0

    if func_file_list or dwi_file_list:
        print('Running workflow of workflows across multiple subjects:')
    elif func_file_list is None and dwi_file_list is None:
        print('Running workflow for single subject:')
    print(str(ID))

    if graph is None and multi_graph is None:
        if network is not None:
            print("%s%s" % ("\nRunning pipeline for 1 RSN: ", network))
        elif multi_nets is not None:
            network = None
            print("%s%d%s%s%s" % ('\nIterating pipeline across ', len(multi_nets), ' RSN\'s: ',
                                  str(', '.join(str(n) for n in multi_nets)), '...'))
        else:
            print("\nUsing whole-brain pipeline...")

        if node_size_list:
            print("%s%s%s" % ('\nGrowing spherical nodes across multiple radius sizes: ',
                              str(', '.join(str(n) for n in node_size_list)), '...'))
            node_size = None
        elif parc is True:
            print("\nUsing parcels as nodes...")
        else:
            if node_size is None:
                node_size = 4
            print("%s%s%s" % ("\nUsing node size of: ", node_size, 'mm...'))

        if func_file or func_file_list:
            if smooth_list:
                print("%s%s%s" % ('\nApplying smoothing to node signal at multiple FWHM mm values: ',
                                  str(', '.join(str(n) for n in smooth_list)), '...'))
            elif float(smooth) > 0:
                print("%s%s%s" % ("\nApplying smoothing to node signal at: ", smooth, 'FWHM mm...'))
            else:
                smooth = 0

            if hpass_list:
                print("%s%s%s" % ('\nApplying high-pass filter to node signal at multiple Hz values: ',
                                  str(', '.join(str(n) for n in hpass_list)), '...'))
            elif hpass is not None:
                print("%s%s%s" % ("\nApplying high-pass filter to node signal at: ", hpass, 'Hz...'))
            else:
                hpass = None

            if isinstance(c_boot, list):
                c_boot = c_boot[0]
            if isinstance(block_size, list):
                block_size = block_size[0]

            if float(c_boot) > 0:
                try:
                    c_boot = int(c_boot)
                    try:
                        block_size = int(block_size)
                    except ValueError:
                        print('ERROR: size of bootstrap blocks indicated with the -bs flag must be an integer > 0.')
                except ValueError:
                    print('ERROR: number of boostraps indicated with the -b flag must be an integer > 0.')
                print("%s%s%s%s" % ('\nApplying circular-block bootstrapping to the node-extracted time-series using: ',
                                    int(c_boot), ' bootstraps with block size ', int(block_size)))
            if (c_boot and not block_size) or (block_size and not c_boot):
                raise ValueError("Error: Both number of bootstraps (-b) and block size (-bs) must be specified to run "
                                 "bootstrapped resampling.")

        if conn_model_list:
            print("%s%s%s" % ('\nIterating graph estimation across multiple connectivity models: ',
                              str(', '.join(str(n) for n in conn_model_list)), '...'))
            conn_model = None
        else:
            print("%s%s" % ("\nUsing connectivity model: ", conn_model))

    elif graph or multi_graph:
        network = 'custom_graph'
        thr = 0
        roi = 'None'
        k_clustering = 0
        node_size = 'None'
        hpass = 'None'
        conn_model = 'None'
        c_boot = 'None'
        if multi_graph:
            print('\nUsing multiple custom input graphs...')
            conn_model = None
            conn_model_list = []
            i = 1
            for graph in multi_graph:
                conn_model_list.append(str(i))
                if '.txt' in graph:
                    graph_name = op.basename(graph).split('.txt')[0]
                elif '.npy' in graph:
                    graph_name = op.basename(graph).split('.npy')[0]
                else:
                    print('Error: input graph file format not recognized. See -help for supported formats.')
                    sys.exit(0)
                print(graph_name)
                atlas = "%s%s%s" % (graph_name, '_', ID)
                do_dir_path(atlas, graph)
                i = i + 1
        else:
            if '.txt' in graph:
                graph_name = op.basename(graph).split('.txt')[0]
            elif '.npy' in graph:
                graph_name = op.basename(graph).split('.npy')[0]
            else:
                print('Error: input graph file format not recognized. See -help for supported formats.')
                sys.exit(0)
            print('\nUsing single custom graph input...')
            print(graph_name)
            atlas = "%s%s%s" % (graph_name, '_', ID)
            do_dir_path(atlas, graph)

    if func_file or func_file_list:
        if (uatlas is not None) and (k_clustering == 0) and (user_atlas_list is None):
            atlas_par = uatlas.split('/')[-1].split('.')[0]
            print("%s%s" % ("\nUser atlas: ", atlas_par))
        elif (uatlas is not None) and (user_atlas_list is None) and (k_clustering == 0):
            atlas_par = uatlas.split('/')[-1].split('.')[0]
            print("%s%s" % ("\nUser atlas: ", atlas_par))
        elif user_atlas_list is not None:
            print('\nIterating across multiple user atlases...')
            if func_file_list:
                for _uatlas in user_atlas_list:
                    atlas_par = _uatlas.split('/')[-1].split('.')[0]
                    print(atlas_par)
            else:
                for _uatlas in user_atlas_list:
                    atlas_par = _uatlas.split('/')[-1].split('.')[0]
                    print(atlas_par)

        if k_clustering == 1:
            cl_mask_name = op.basename(clust_mask).split('.nii')[0]
            atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
            print("%s%s" % ("\nCluster atlas: ", atlas_clust))
            print("\nClustering within mask at a single resolution...")
        elif k_clustering == 2:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            print("\nClustering within mask at multiple resolutions...")
            if func_file_list:
                for _k in k_list:
                    cl_mask_name = op.basename(clust_mask).split('.nii')[0]
                    atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', _k)
                    print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _k in k_list:
                    cl_mask_name = op.basename(clust_mask).split('.nii')[0]
                    atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', _k)
                    print("%s%s" % ("Cluster atlas: ", atlas_clust))
            k = None
        elif k_clustering == 3:
            print("\nClustering within multiple masks at a single resolution...")
            if func_file_list:
                for _clust_mask in clust_mask_list:
                    cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                    atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _clust_mask in clust_mask_list:
                    cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                    atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    print("%s%s" % ("Cluster atlas: ", atlas_clust))
            clust_mask = None
        elif k_clustering == 4:
            print("\nClustering within multiple masks at multiple resolutions...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            if func_file_list:
                for _clust_mask in clust_mask_list:
                    for _k in k_list:
                        cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', _k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _clust_mask in clust_mask_list:
                    for _k in k_list:
                        cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', _k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            clust_mask = None
            k = None
        elif k_clustering == 5:
            for _clust_type in clust_type_list:
                cl_mask_name = op.basename(clust_mask).split('.nii')[0]
                atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', k)
                print("%s%s" % ("\nCluster atlas: ", atlas_clust))
                print("\nClustering within mask at a single resolution using multiple clustering methods...")
            clust_type = None
        elif k_clustering == 6:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            print("\nClustering within mask at multiple resolutions using multiple clustering methods...")
            if func_file_list:
                for _clust_type in clust_type_list:
                    for _k in k_list:
                        cl_mask_name = op.basename(clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', _k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _clust_type in clust_type_list:
                    for _k in k_list:
                        cl_mask_name = op.basename(clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', _k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            clust_type = None
            k = None
        elif k_clustering == 7:
            print("\nClustering within multiple masks at a single resolution using multiple clustering methods...")
            if func_file_list:
                for _clust_type in clust_type_list:
                    for _clust_mask in clust_mask_list:
                        cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _clust_type in clust_type_list:
                    for _clust_mask in clust_mask_list:
                        cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                        atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_clust))
            clust_mask = None
            clust_type = None
        elif k_clustering == 8:
            print("\nClustering within multiple masks at multiple resolutions using multiple clustering methods...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            if func_file_list:
                for _clust_type in clust_type_list:
                    for _clust_mask in clust_mask_list:
                        for _k in k_list:
                            cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                            atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', _k)
                            print("%s%s" % ("Cluster atlas: ", atlas_clust))
            else:
                for _clust_type in clust_type_list:
                    for _clust_mask in clust_mask_list:
                        for _k in k_list:
                            cl_mask_name = op.basename(_clust_mask).split('.nii')[0]
                            atlas_clust = "%s%s%s%s%s" % (cl_mask_name, '_', _clust_type, '_k', _k)
                            print("%s%s" % ("Cluster atlas: ", atlas_clust))
            clust_mask = None
            clust_type = None
            k = None
        elif (user_atlas_list is not None or uatlas is not None) and (k_clustering == 4 or
                                                                      k_clustering == 3 or
                                                                      k_clustering == 2 or
                                                                      k_clustering == 1) and (atlas is None):
            print('Error: the -ua flag cannot be used alone with the clustering option. Use the -cm flag instead.')
            sys.exit(0)

        if multi_atlas is not None:
            print('\nIterating across multiple predefined atlases...')
            if func_file_list:
                for _func_file in func_file_list:
                    for _atlas in multi_atlas:
                        if (parc is True) and (_atlas in nilearn_coord_atlases or _atlas in
                                               nilearn_prob_atlases):
                            raise ValueError("%s%s%s" % ('\nERROR: ', _atlas,
                                                         ' is a coordinate atlas and must be used with the -spheres '
                                                         'flag.'))
                        else:
                            print(_atlas)
                            do_dir_path(_atlas, _func_file)
            else:
                for _atlas in multi_atlas:
                    if (parc is True) and (_atlas in nilearn_coord_atlases or _atlas in
                                           nilearn_prob_atlases):
                        raise ValueError("%s%s%s" % ('\nERROR: ', _atlas,
                                                     ' is a coordinate atlas and must be used with the -spheres '
                                                     'flag.'))
                    else:
                        print(_atlas)
                        do_dir_path(_atlas, func_file)
        elif atlas is not None:
            if (parc is True) and (atlas in nilearn_coord_atlases or atlas in nilearn_prob_atlases):
                raise ValueError("%s%s%s" % ('\nERROR: ', atlas,
                                             ' is a coordinate atlas and must be used with the -spheres flag.'))
            else:
                print("%s%s" % ("\nPredefined atlas: ", atlas))
                if func_file_list:
                    for _func_file in func_file_list:
                        do_dir_path(atlas, _func_file)
                else:
                    do_dir_path(atlas, func_file)
        else:
            if (uatlas is None) and (k == 0):
                raise KeyError('\nERROR: No atlas specified!')
            else:
                pass

    if dwi_file or dwi_file_list:
        if (conn_model == 'tensor') and (directget == 'prob'):
            raise ValueError('Cannot perform probabilistic tracking with tensor model estimation...')

        if user_atlas_list:
            print('\nIterating across multiple user atlases...')
            if dwi_file_list:
                for _dwi_file in dwi_file_list:
                    for _uatlas in user_atlas_list:
                        atlas_par = _uatlas.split('/')[-1].split('.')[0]
                        print(atlas_par)
            else:
                for _uatlas in user_atlas_list:
                    atlas_par = _uatlas.split('/')[-1].split('.')[0]
                    print(atlas_par)
        elif (uatlas is not None) and (user_atlas_list is None):
            atlas_par = uatlas.split('/')[-1].split('.')[0]
            print(atlas_par)
            ref_txt = "%s%s" % (uatlas.split('/')[-1:][0].split('.')[0], '.txt')
            print("%s%s" % ('Using label reference: ', ref_txt))
        if multi_atlas:
            print('\nIterating across multiple predefined atlases...')
            if dwi_file_list:
                for _dwi_file in dwi_file_list:
                    for _atlas in multi_atlas:
                        if (parc is True) and (_atlas in nilearn_coord_atlases):
                            raise ValueError("%s%s%s" % ('\nERROR: ', _atlas,
                                                         ' is a coordinate atlas and must be used with the -spheres '
                                                         'flag.'))
                        else:
                            print(_atlas)
                            do_dir_path(_atlas, _dwi_file)
            else:
                for _atlas in multi_atlas:
                    if (parc is True) and (_atlas in nilearn_coord_atlases):
                        raise ValueError("%s%s%s" % ('\nERROR: ', _atlas,
                                                     ' is a coordinate atlas and must be used with the -spheres '
                                                     'flag.'))
                    else:
                        print(_atlas)
                        do_dir_path(_atlas, dwi_file)
        elif atlas:
            if (parc is True) and (atlas in nilearn_coord_atlases):
                raise ValueError("%s%s%s" % ('\nERROR: ', atlas,
                                             ' is a coordinate atlas and must be used with the -spheres flag.'))
            else:
                print("%s%s" % ("\nNilearn atlas: ", atlas))
                if dwi_file_list:
                    for _dwi_file in dwi_file_list:
                        do_dir_path(atlas, _dwi_file)
                else:
                    do_dir_path(atlas, dwi_file)
        else:
            if uatlas is None:
                raise KeyError('\nERROR: No atlas specified!')
            else:
                pass
        if target_samples:
            print("%s%s%s" % ('Using ', target_samples, ' samples...'))
        if max_length:
            print("%s%s%s" % ('Using ', max_length, ' maximum length of streamlines...'))

    if (dwi_file or dwi_file_list) and not (func_file or func_file_list):
        print('\nRunning dmri connectometry only...')
        if dwi_file_list:
            for (_dwi_file, _fbval, _fbvec, _anat_file) in dwi_file_list:
                print("%s%s" % ('Diffusion-Weighted Image:\n', _dwi_file))
                print("%s%s" % ('B-Values:\n', _fbval))
                print("%s%s" % ('B-Vectors:\n', _fbvec))
                print("%s%s" % ('T1-weighted Image:\n', _anat_file))
                if waymask is not None:
                    print("%s%s" % ('Waymask:\n', waymask))
        else:
            print("%s%s" % ('Diffusion-Weighted Image:\n', dwi_file))
            print("%s%s" % ('B-Values:\n', fbval))
            print("%s%s" % ('B-Vectors:\n', fbvec))
            print("%s%s" % ('T1-weighted Image:\n', anat_file))
            if waymask is not None:
                print("%s%s" % ('Waymask:\n', waymask))
        conf = None
        k = None
        clust_mask = None
        k_min = None
        k_max = None
        k_step = None
        k_clustering = None
        clust_mask_list = None
        hpass = None
        clust_type = None
        local_corr = None
        clust_type_list = None
        c_boot = None
        block_size = None
        multimodal = False
    elif (func_file or func_file_list) and not (dwi_file or dwi_file_list):
        print('\nRunning fmri connectometry only...')
        if func_file_list:
            for _func_file in func_file_list:
                print("%s%s" % ('BOLD Image: ', _func_file))
        else:
            print("%s%s" % ('BOLD Image: ', func_file))
        multimodal = False
    elif (func_file or func_file_list) and (dwi_file or dwi_file_list):
        multimodal = True
        print('\nRunning joint fMRI-dMRI connectometry...')
        print("%s%s" % ('BOLD Image:\n', func_file))
        print("%s%s" % ('Diffusion-Weighted Image:\n', dwi_file))
        print("%s%s" % ('B-Values:\n', fbval))
        print("%s%s" % ('B-Vectors:\n', fbvec))
        print("%s%s" % ('T1-Weighted Image:\n', anat_file))
    else:
        multimodal = False
    print('\n-------------------------------------------------------------------------\n\n')

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
    # print("%s%s" % ('dens_thresh: ', dens_thresh))
    # print("%s%s" % ('conf: ', conf))
    # print("%s%s" % ('plot_switch: ', plot_switch))
    # print("%s%s" % ('multi_thr: ', multi_thr))
    # print("%s%s" % ('multi_atlas: ', multi_atlas))
    # print("%s%s" % ('min_thr: ', min_thr))
    # print("%s%s" % ('max_thr: ', max_thr))
    # print("%s%s" % ('step_thr: ', step_thr))
    # print("%s%s" % ('spheres: ', spheres))
    # print("%s%s" % ('ref_txt: ', ref_txt))
    # print("%s%s" % ('procmem: ', procmem))
    # print("%s%s" % ('waymask: ', waymask))
    # print("%s%s" % ('k: ', k))
    # print("%s%s" % ('clust_mask: ', clust_mask))
    # print("%s%s" % ('k_min: ', k_min))
    # print("%s%s" % ('k_max: ', k_max))
    # print("%s%s" % ('k_step: ', k_step))
    # print("%s%s" % ('k_clustering: ', k_clustering))
    # print("%s%s" % ('user_atlas_list: ', user_atlas_list))
    # print("%s%s" % ('clust_mask_list: ', clust_mask_list))
    # print("%s%s" % ('clust_type: ', clust_type))
    # print("%s%s" % ('local_corr: ', local_corr))
    # print("%s%s" % ('clust_type_list: ', clust_type_list))
    # print("%s%s" % ('prune: ', prune))
    # print("%s%s" % ('node_size_list: ', node_size_list))
    # print("%s%s" % ('smooth_list: ', smooth_list))
    # print("%s%s" % ('c_boot: ', c_boot))
    # print("%s%s" % ('block_size: ', block_size))
    # print("%s%s" % ('mask: ', mask))
    # print("%s%s" % ('norm: ', norm))
    # print("%s%s" % ('binary: ', binary))
    # print("%s%s" % ('embed: ', embed))
    # print("%s%s" % ('multiplex: ', multiplex))
    # print("%s%s" % ('track_type: ', track_type))
    # print("%s%s" % ('tiss_class: ', tiss_class))
    # print("%s%s" % ('directget: ', directget))
    # print("%s%s" % ('multi_directget: ', multi_directget))
    # print("%s%s" % ('template: ', template))
    # print("%s%s" % ('template_mask: ', template_mask))
    # print("%s%s" % ('func_file: ', func_file))
    # print("%s%s" % ('dwi_file: ', dwi_file))
    # print("%s%s" % ('fbval: ', fbval))
    # print("%s%s" % ('fbvec: ', fbvec))
    # print("%s%s" % ('anat_file: ', anat_file))
    # print("%s%s" % ('func_file_list: ', func_file_list))
    # print("%s%s" % ('dwi_file_list: ', dwi_file_list))
    # print("%s%s" % ('mask_list: ', mask_list))
    # print("%s%s" % ('fbvec_list: ', fbvec_list))
    # print("%s%s" % ('fbval_list: ', fbval_list))
    # print("%s%s" % ('conf_list: ', conf_list))
    # print("%s%s" % ('anat_file_list: ', anat_file_list))
    # print('\n\n\n\n\n')
    # import sys
    # sys.exit(0)

    # Import wf core and interfaces
    import warnings
    warnings.filterwarnings("ignore")
    from pynets.core.utils import collectpandasjoin
    from pynets.core.interfaces import CombinePandasDfs, ExtractNetStats
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.core.workflows import workflow_selector

    def init_wf_single_subject(ID, func_file, atlas, network, node_size, roi, thr, uatlas,
                               multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_file,
                               multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_file, parc, ref_txt, procmem, k,
                               clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                               node_size_list, num_total_samples, graph, conn_model_list, min_span_tree, verbose,
                               plugin_type, use_AAL_naming, multi_graph, smooth, smooth_list, disp_filt, clust_type,
                               clust_type_list, c_boot, block_size, mask, norm, binary, fbval, fbvec, target_samples,
                               curv_thr_list, step_list, overlap_thr, overlap_thr_list, track_type, max_length,
                               maxcrossing, min_length, directget, tiss_class, runtime_dict, execution_dict, embed,
                               multi_directget, multimodal, hpass, hpass_list, template, template_mask, vox_size,
                               multiplex, waymask, local_corr):
        """A function interface for generating a single-subject workflow"""
        import warnings
        warnings.filterwarnings("ignore")
        from time import strftime

        if (func_file is not None) and (dwi_file is None):
            wf = pe.Workflow(name="%s%s%s%s" % ('wf_single_sub_', ID, '_fmri_', strftime('%Y%m%d_%H%M%S')))
        elif (dwi_file is not None) and (func_file is None):
            wf = pe.Workflow(name="%s%s%s%s" % ('wf_single_sub_', ID, '_dmri_', strftime('%Y%m%d_%H%M%S')))
        else:
            wf = pe.Workflow(name="%s%s%s%s" % ('wf_single_sub_', ID, '_', strftime('%Y%m%d_%H%M%S')))
        import_list = ["import sys", "import os", "import numpy as np", "import networkx as nx", "import indexed_gzip",
                       "import nibabel as nib", "import warnings", "warnings.filterwarnings(\"ignore\")",
                       "np.warnings.filterwarnings(\"ignore\")", "warnings.simplefilter(\"ignore\")"]
        inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'network', 'thr', 'node_size', 'roi', 'multi_nets',
                                                          'conn_model', 'plot_switch', 'graph', 'prune',
                                                          'norm', 'binary', 'multimodal']),
                            name='inputnode', imports=import_list)
        if verbose is True:
            from nipype import config, logging
            cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'log_to_file': True,
                                  'interface_level': 'DEBUG'},
                         monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True})
            logging.update_logging(config)
            config.update_config(cfg_v)
            config.enable_debug_mode()
            config.enable_resource_monitor()

        execution_dict['crashdump_dir'] = str(wf.base_dir)
        execution_dict['plugin'] = str(plugin_type)
        cfg = dict(execution=execution_dict)
        for key in cfg.keys():
            for setting, value in cfg[key].items():
                wf.config[key][setting] = value

        inputnode.inputs.ID = ID
        inputnode.inputs.network = network
        inputnode.inputs.thr = thr
        inputnode.inputs.node_size = node_size
        inputnode.inputs.roi = roi
        inputnode.inputs.multi_nets = multi_nets
        inputnode.inputs.conn_model = conn_model
        inputnode.inputs.plot_switch = plot_switch
        inputnode.inputs.graph = graph
        inputnode.inputs.prune = prune
        inputnode.inputs.norm = norm
        inputnode.inputs.binary = binary
        inputnode.inputs.multimodal = multimodal

        meta_wf = workflow_selector(func_file, ID, atlas, network, node_size, roi, thr, uatlas,
                                    multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_file,
                                    anat_file, parc, ref_txt, procmem, multi_thr, multi_atlas, max_thr, min_thr,
                                    step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list,
                                    clust_mask_list, prune, node_size_list, num_total_samples, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, smooth, smooth_list, disp_filt,
                                    clust_type, clust_type_list, c_boot, block_size, mask, norm, binary, fbval, fbvec,
                                    target_samples, curv_thr_list, step_list, overlap_thr, overlap_thr_list, track_type,
                                    max_length, maxcrossing, min_length, directget, tiss_class, runtime_dict,
                                    execution_dict, embed, multi_directget, multimodal, hpass, hpass_list, template,
                                    template_mask, vox_size, multiplex, waymask, local_corr)

        meta_wf._n_procs = procmem[0]
        meta_wf._mem_gb = procmem[1]
        meta_wf.n_procs = procmem[0]
        meta_wf.mem_gb = procmem[1]
        wf.add_nodes([meta_wf])

        # Set resource restrictions at level of the meta-meta wf
        if func_file:
            wf_selected = "%s%s" % ('fmri_connectometry_', ID)
            for node_name in wf.get_node(meta_wf.name).get_node(wf_selected).list_node_names():
                if node_name in runtime_dict:
                    wf.get_node(meta_wf.name).get_node(wf_selected).get_node(node_name)._n_procs = runtime_dict[node_name][0]
                    wf.get_node(meta_wf.name).get_node(wf_selected).get_node(node_name)._mem_gb = runtime_dict[node_name][1]

        if dwi_file:
            wf_selected = "%s%s" % ('dmri_connectometry_', ID)
            for node_name in wf.get_node(meta_wf.name).get_node(wf_selected).list_node_names():
                if node_name in runtime_dict:
                    wf.get_node(meta_wf.name).get_node(wf_selected).get_node(node_name)._n_procs = runtime_dict[node_name][0]
                    wf.get_node(meta_wf.name).get_node(wf_selected).get_node(node_name)._mem_gb = runtime_dict[node_name][1]

        wf.get_node(meta_wf.name)._n_procs = procmem[0]
        wf.get_node(meta_wf.name)._mem_gb = procmem[1]
        wf.get_node(meta_wf.name).n_procs = procmem[0]
        wf.get_node(meta_wf.name).mem_gb = procmem[1]
        wf.get_node(meta_wf.name).get_node(wf_selected)._n_procs = procmem[0]
        wf.get_node(meta_wf.name).get_node(wf_selected)._mem_gb = procmem[1]
        wf.get_node(meta_wf.name).get_node(wf_selected).n_procs = procmem[0]
        wf.get_node(meta_wf.name).get_node(wf_selected).mem_gb = procmem[1]

        # Fully-automated graph analysis
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name="ExtractNetStats",
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path',
                                              'roi', 'prune', 'norm', 'binary'], nested=True,
                                   imports=import_list)
        net_mets_node._n_procs = 1
        net_mets_node._mem_gb = 1

        # Aggregate list of paths to pandas dataframe pickles
        join_net_mets = pe.JoinNode(niu.IdentityInterface(fields=['out_path_neat']),
                                    name='join_net_mets', joinsource=net_mets_node,
                                    joinfield=['out_path_neat'])

        collect_pd_list_net_csv_node = pe.Node(niu.Function(input_names=['net_mets_csv'],
                                                            output_names=['net_mets_csv_out'],
                                                            function=collectpandasjoin),
                                               name="AggregatePandasCSVs",
                                               imports=import_list)

        # Combine dataframes across models
        combine_pandas_dfs_node = pe.Node(interface=CombinePandasDfs(), name="CombinePandasDfs",
                                          input_names=['network', 'ID', 'net_mets_csv_list', 'plot_switch',
                                                       'multi_nets', 'multimodal'],
                                          output_names=['combination_complete'],
                                          imports=import_list)

        combine_pandas_dfs_node._n_procs = 1
        combine_pandas_dfs_node._mem_gb = 2

        handshake_node = meta_wf.get_node('pass_meta_outs_node')

        final_outputnode = pe.Node(niu.IdentityInterface(fields=['combination_complete']), name='final_outputnode')

        wf.connect([
            (handshake_node, net_mets_node, [('est_path_iterlist', 'est_path'),
                                             ('network_iterlist', 'network'),
                                             ('thr_iterlist', 'thr'),
                                             ('ID_iterlist', 'ID'),
                                             ('conn_model_iterlist', 'conn_model'),
                                             ('roi_iterlist', 'roi'),
                                             ('prune_iterlist', 'prune'),
                                             ('norm_iterlist', 'norm'),
                                             ('binary_iterlist', 'binary')]),
            (inputnode, combine_pandas_dfs_node, [('network', 'network'),
                                                  ('ID', 'ID'),
                                                  ('plot_switch', 'plot_switch'),
                                                  ('multi_nets', 'multi_nets'),
                                                  ('multimodal', 'multimodal')]),
            (net_mets_node, join_net_mets, [('out_path_neat', 'out_path_neat')]),
            (net_mets_node, collect_pd_list_net_csv_node, [('out_path_neat', 'net_mets_csv')]),
            (collect_pd_list_net_csv_node, combine_pandas_dfs_node, [('net_mets_csv_out', 'net_mets_csv_list')]),
            (combine_pandas_dfs_node, final_outputnode, [('combination_complete', 'combination_complete')])
        ])

        # Raw graph case
        if graph or multi_graph:
            wf.disconnect([(handshake_node, net_mets_node,
                            [('est_path_iterlist', 'est_path'),
                             ('network_iterlist', 'network'),
                             ('thr_iterlist', 'thr'),
                             ('ID_iterlist', 'ID'),
                             ('conn_model_iterlist', 'conn_model'),
                             ('roi_iterlist', 'roi'),
                             ('prune_iterlist', 'prune'),
                             ('norm_iterlist', 'norm'),
                             ('binary_iterlist', 'binary')])
                           ])
            wf.remove_nodes([meta_wf])

            # Multiple raw graphs
            if multi_graph:
                net_mets_node.inputs.est_path = multi_graph
                net_mets_node.inputs.ID = [ID] * len(multi_graph)
                net_mets_node.inputs.roi = [roi] * len(multi_graph)
                net_mets_node.inputs.thr = [thr] * len(multi_graph)
                net_mets_node.inputs.prune = [prune] * len(multi_graph)
                net_mets_node.inputs.network = [network] * len(multi_graph)
                net_mets_node.inputs.conn_model = conn_model_list
                net_mets_node.inputs.norm = [norm] * len(multi_graph)
                net_mets_node.inputs.binary = [binary] * len(multi_graph)
            else:
                wf.connect([(inputnode, net_mets_node, [('network', 'network'),
                                                        ('ID', 'ID'),
                                                        ('thr', 'thr'),
                                                        ('conn_model', 'conn_model'),
                                                        ('roi', 'roi'),
                                                        ('prune', 'prune'),
                                                        ('graph', 'est_path'),
                                                        ('norm', 'norm'),
                                                        ('binary', 'binary')])
                            ])

        return wf

    # Multi-subject pipeline
    def wf_multi_subject(ID, func_file_list, dwi_file_list, mask_list, fbvec_list, fbval_list, conf_list,
                         anat_file_list, atlas, network, node_size, roi, thr, uatlas, multi_nets, conn_model,
                         dens_thresh, conf, adapt_thresh, plot_switch, dwi_file, multi_thr, multi_atlas, min_thr,
                         max_thr, step_thr, anat_file, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step,
                         k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples,
                         graph, conn_model_list, min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph,
                         smooth, smooth_list, disp_filt, clust_type, clust_type_list, c_boot, block_size, mask, norm,
                         binary, fbval, fbvec, target_samples, curv_thr_list, step_list, overlap_thr, overlap_thr_list,
                         track_type, max_length, maxcrossing, min_length, directget, tiss_class, runtime_dict,
                         execution_dict, embed, multi_directget, multimodal, hpass, hpass_list, template, template_mask,
                         vox_size, multiplex, waymask, local_corr):
        """A function interface for generating multiple single-subject workflows -- i.e. a 'multi-subject' workflow"""
        import warnings
        warnings.filterwarnings("ignore")
        from time import strftime

        wf_multi = pe.Workflow(name="%s%s" % ('wf_multisub_', strftime('%Y%m%d_%H%M%S')))

        if (func_file_list is None) and dwi_file_list:
            func_file_list = len(dwi_file_list) * [None]
            conf_list = len(dwi_file_list) * [None]

        if (dwi_file_list is None) and func_file_list:
            dwi_file_list = len(func_file_list) * [None]
            fbvec_list = len(func_file_list) * [None]
            fbval_list = len(func_file_list) * [None]

        multi_iter_len = len(list(zip(dwi_file_list, func_file_list)))

        i = 0
        for dwi_file, func_file in zip(dwi_file_list, func_file_list):
            if conf_list and func_file:
                conf_sub = conf_list[i]
            else:
                conf_sub = None
            if fbval_list and dwi_file:
                fbval_sub = fbval_list[i]
            else:
                fbval_sub = None
            if fbvec_list and dwi_file:
                fbvec_sub = fbvec_list[i]
            else:
                fbvec_sub = None
            if mask_list:
                mask_sub = mask_list[i]
            else:
                mask_sub = None
            if anat_file_list:
                anat_file = anat_file_list[i]
            else:
                anat_file = None
            wf_single_subject = init_wf_single_subject(
                ID=ID[i], func_file=func_file, atlas=atlas,
                network=network, node_size=node_size, roi=roi, thr=thr, uatlas=uatlas,
                multi_nets=multi_nets, conn_model=conn_model, dens_thresh=dens_thresh, conf=conf_sub,
                adapt_thresh=adapt_thresh, plot_switch=plot_switch, dwi_file=dwi_file, multi_thr=multi_thr,
                multi_atlas=multi_atlas, min_thr=min_thr, max_thr=max_thr, step_thr=step_thr, anat_file=anat_file,
                parc=parc, ref_txt=ref_txt, procmem=procmem, k=k, clust_mask=clust_mask, k_min=k_min, k_max=k_max,
                k_step=k_step, k_clustering=k_clustering, user_atlas_list=user_atlas_list,
                clust_mask_list=clust_mask_list, prune=prune, node_size_list=node_size_list,
                num_total_samples=num_total_samples, graph=graph, conn_model_list=conn_model_list,
                min_span_tree=min_span_tree, verbose=verbose, plugin_type=plugin_type, use_AAL_naming=use_AAL_naming,
                multi_graph=multi_graph, smooth=smooth, smooth_list=smooth_list, disp_filt=disp_filt,
                clust_type=clust_type, clust_type_list=clust_type_list, c_boot=c_boot, block_size=block_size,
                mask=mask_sub, norm=norm, binary=binary, fbval=fbval_sub, fbvec=fbvec_sub,
                target_samples=target_samples, curv_thr_list=curv_thr_list, step_list=step_list,
                overlap_thr=overlap_thr, overlap_thr_list=overlap_thr_list, track_type=track_type,
                max_length=max_length, maxcrossing=maxcrossing, min_length=min_length,
                directget=directget, tiss_class=tiss_class, runtime_dict=runtime_dict, execution_dict=execution_dict,
                embed=embed, multi_directget=multi_directget, multimodal=multimodal, hpass=hpass, hpass_list=hpass_list,
                template=template, template_mask=template_mask, vox_size=vox_size, multiplex=multiplex,
                waymask=waymask, local_corr=local_corr)
            wf_single_subject._n_procs = procmem[0]
            wf_single_subject._mem_gb = procmem[1]
            wf_single_subject.n_procs = procmem[0]
            wf_single_subject.mem_gb = procmem[1]
            wf_multi.add_nodes([wf_single_subject])
            wf_multi.get_node(wf_single_subject.name)._n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name)._mem_gb = procmem[1]
            wf_multi.get_node(wf_single_subject.name).n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name).mem_gb = procmem[1]

            # Restrict nested meta-meta wf resources at the level of the group wf
            if func_file:
                wf_selected = "%s%s" % ('fmri_connectometry_', ID[i])
                meta_wf_name = "%s%s" % ('meta_wf_', ID[i])
                for node_name in wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).list_node_names():
                    if node_name in runtime_dict:
                        wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name)._n_procs = runtime_dict[node_name][0]
                        wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name)._mem_gb = runtime_dict[node_name][1]
                        try:
                            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
                            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
                        except:
                            continue
            if dwi_file:
                wf_selected = "%s%s" % ('dmri_connectometry_', ID[i])
                meta_wf_name = "%s%s" % ('meta_wf_', ID[i])
                for node_name in wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).list_node_names():
                    if node_name in runtime_dict:
                        wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name)._n_procs = runtime_dict[node_name][0]
                        wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name)._mem_gb = runtime_dict[node_name][1]
                        try:
                            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name).interface.n_procs = runtime_dict[node_name][0]
                            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node(node_name).interface.mem_gb = runtime_dict[node_name][1]
                        except:
                            continue

            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected)._n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected)._mem_gb = procmem[1]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).mem_gb = procmem[1]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name)._n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name)._mem_gb = procmem[1]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).n_procs = procmem[0]
            wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).mem_gb = procmem[1]

            wf_multi.get_node(wf_single_subject.name).get_node("ExtractNetStats")._n_procs = 1
            wf_multi.get_node(wf_single_subject.name).get_node("ExtractNetStats")._mem_gb = 1
            wf_multi.get_node(wf_single_subject.name).get_node("CombinePandasDfs")._n_procs = 1
            wf_multi.get_node(wf_single_subject.name).get_node("CombinePandasDfs")._mem_gb = 2

            i = i + 1

        return wf_multi

    # Workflow generation
    # Multi-subject workflow generator
    if (func_file_list or dwi_file_list) or (func_file_list and dwi_file_list):
        wf_multi = wf_multi_subject(ID, func_file_list, dwi_file_list, mask_list, fbvec_list, fbval_list,
                                    conf_list, anat_file_list, atlas, network, node_size, roi,
                                    thr, uatlas, multi_nets, conn_model, dens_thresh,
                                    conf, adapt_thresh, plot_switch, dwi_file, multi_thr,
                                    multi_atlas, min_thr, max_thr, step_thr, anat_file, parc,
                                    ref_txt, procmem, k, clust_mask, k_min, k_max, k_step,
                                    k_clustering, user_atlas_list, clust_mask_list, prune,
                                    node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph,
                                    smooth, smooth_list, disp_filt, clust_type, clust_type_list, c_boot,
                                    block_size, mask, norm, binary, fbval, fbvec, target_samples, curv_thr_list,
                                    step_list, overlap_thr, overlap_thr_list, track_type, max_length, maxcrossing,
                                    min_length, directget, tiss_class, runtime_dict, execution_dict, embed,
                                    multi_directget, multimodal, hpass, hpass_list, template, template_mask, vox_size,
                                    multiplex, waymask, local_corr)
        import warnings
        warnings.filterwarnings("ignore")
        import shutil

        os.makedirs("%s%s%s" % (work_dir, '/wf_multi_subject_', '_'.join(ID)), exist_ok=True)
        wf_multi.base_dir = "%s%s%s" % (work_dir, '/wf_multi_subject_', '_'.join(ID))

        func_dir_list = []
        if func_file_list:
            for func_file in func_file_list:
                if func_file is not None:
                    func_dir_list.append(os.path.dirname(func_file))

        dwi_dir_list = []
        if dwi_file_list:
            for dwi_file in dwi_file_list:
                if dwi_file is not None:
                    dwi_dir_list.append(os.path.dirname(dwi_file))

        if verbose is True:
            from nipype import config, logging
            cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'interface_level': 'DEBUG',
                                  'log_directory': str(wf_multi.base_dir), 'log_to_file': True},
                         monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True,
                                     'summary_file': str(wf_multi.base_dir)})
            logging.update_logging(config)
            config.update_config(cfg_v)
            config.enable_debug_mode()
            config.enable_resource_monitor()

            import logging
            callback_log_path = "%s%s" % (wf_multi.base_dir, '/run_stats.log')
            logger = logging.getLogger('callback')
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(callback_log_path)
            logger.addHandler(handler)

        execution_dict['crashdump_dir'] = str(wf_multi.base_dir)
        execution_dict['plugin'] = str(plugin_type)
        cfg = dict(execution=execution_dict)
        for key in cfg.keys():
            for setting, value in cfg[key].items():
                wf_multi.config[key][setting] = value
        try:
            wf_multi.write_graph(graph2use="colored", format='png')
        except:
            pass
        if verbose is True:
            from nipype.utils.profiler import log_nodes_cb
            plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                           'status_callback': log_nodes_cb, 'scheduler': 'mem_thread'}
        else:
            plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf_multi.run(plugin=plugin_type, plugin_args=plugin_args)
        if verbose is True:
            from nipype.utils.draw_gantt_chart import generate_gantt_chart
            print('Plotting resource profile from run...')
            generate_gantt_chart(callback_log_path, cores=int(procmem[0]))
            handler.close()
            logger.removeHandler(handler)

        # Clean up temporary directories
        if len(func_dir_list) > 0:
            for func_dir in func_dir_list:
                for cnfnd_tmp_dir in glob.glob("%s%s" % (func_dir, '/*/confounds_tmp')):
                    shutil.rmtree(cnfnd_tmp_dir)

    # Single-subject workflow generator
    else:
        # Single-subject pipeline
        wf = init_wf_single_subject(ID, func_file, atlas, network, node_size, roi, thr, uatlas,
                                    multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_file,
                                    multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_file, parc, ref_txt,
                                    procmem, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list,
                                    clust_mask_list, prune, node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph, smooth,
                                    smooth_list, disp_filt, clust_type, clust_type_list, c_boot, block_size, mask,
                                    norm, binary, fbval, fbvec, target_samples, curv_thr_list, step_list, overlap_thr,
                                    overlap_thr_list, track_type, max_length, maxcrossing, min_length,
                                    directget, tiss_class, runtime_dict, execution_dict, embed, multi_directget,
                                    multimodal, hpass, hpass_list, template, template_mask, vox_size, multiplex,
                                    waymask, local_corr)
        import warnings
        warnings.filterwarnings("ignore")
        import shutil
        import os
        import uuid
        from time import strftime
        if (func_file is not None) and (dwi_file is None):
            base_dirname = "%s%s" % ('wf_single_subject_fmri_', str(ID))
        elif (dwi_file is not None) and (func_file is None):
            base_dirname = "%s%s" % ('wf_single_subject_dmri_', str(ID))
        else:
            base_dirname = "%s%s" % ('wf_single_subject_', str(ID))

        run_uuid = '%s_%s' % (strftime('%Y%m%d_%H%M%S'), uuid.uuid4())
        if func_file:
            func_dir = os.path.dirname(func_file)
        if dwi_file:
            dwi_dir = os.path.dirname(dwi_file)
        os.makedirs("%s%s%s%s%s%s%s" % (work_dir, '/', ID, '_', run_uuid, '_', base_dirname), exist_ok=True)
        wf.base_dir = "%s%s%s%s%s%s%s" % (work_dir, '/', ID, '_', run_uuid, '_', base_dirname)

        if verbose is True:
            from nipype import config, logging
            cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'interface_level': 'DEBUG',
                                  'log_directory': str(wf.base_dir), 'log_to_file': True},
                         monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True,
                                     'summary_file': str(wf.base_dir)})
            logging.update_logging(config)
            config.update_config(cfg_v)
            config.enable_debug_mode()
            config.enable_resource_monitor()

            import logging
            callback_log_path = "%s%s" % (wf.base_dir, '/run_stats.log')
            logger = logging.getLogger('callback')
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(callback_log_path)
            logger.addHandler(handler)

        execution_dict['crashdump_dir'] = str(wf.base_dir)
        execution_dict['plugin'] = str(plugin_type)
        cfg = dict(execution=execution_dict)
        for key in cfg.keys():
            for setting, value in cfg[key].items():
                wf.config[key][setting] = value
        try:
            wf.write_graph(graph2use="colored", format='png')
        except:
            pass
        if verbose is True:
            from nipype.utils.profiler import log_nodes_cb
            plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                           'status_callback': log_nodes_cb, 'scheduler': 'mem_thread'}
        else:
            plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf.run(plugin=plugin_type, plugin_args=plugin_args)
        if verbose is True:
            from nipype.utils.draw_gantt_chart import generate_gantt_chart
            print('Plotting resource profile from run...')
            generate_gantt_chart(callback_log_path, cores=int(procmem[0]))
            handler.close()
            logger.removeHandler(handler)

        # Clean up temporary directories
        if func_file:
            for cnfnd_tmp_dir in glob.glob("%s%s" % (func_dir, '/*/confounds_tmp')):
                shutil.rmtree(cnfnd_tmp_dir)

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')

    return


def main():
    """Initializes main script from command-line call to generate single-subject or multi-subject workflow(s)"""
    import gc
    import sys
    try:
        from pynets.core.utils import do_dir_path
    except ImportError:
        print('PyNets not installed! Ensure that you are referencing the correct site-packages and using Python3.5+')

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    args = get_parser().parse_args()

    try:
        from multiprocessing import set_start_method, Process, Manager
        set_start_method('forkserver')
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(args, retval))
            p.start()
            p.join()

            if p.exitcode != 0:
                sys.exit(p.exitcode)

            # Clean up master process before running workflow, which may create forks
            gc.collect()
    except:
        print('\nWARNING: Forkserver failed to initialize. Are you using Python3 ?')
        retval = dict()
        build_workflow(args, retval)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
