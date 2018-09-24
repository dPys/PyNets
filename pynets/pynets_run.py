#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Tue Nov  7 10:40:07 2017
# Copyright (C) 2018
# @author: Derek Pisner
import warnings
warnings.simplefilter("ignore")


def get_parser():
    import argparse
    # Parse args
    parser = argparse.ArgumentParser(description='PyNets: A Fully-Automated Workflow for Reproducible Graph Analysis of Functional and Structural Connectomes')
    parser.add_argument('-i',
                        metavar='Path to input file',
                        default=None,
                        required=False,
                        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR multiple paths to multiple preprocessed functional images in standard space and in .nii or .nii.gz format, separated by commas OR the path to a text file containing a list of paths to subject files.\n')
    parser.add_argument('-g',
                        metavar='Path to graph',
                        default=None,
                        help='In either .txt or .npy format. This skips fMRI and dMRI graph estimation workflows and begins at the graph analysis stage.\n')
    parser.add_argument('-dwi',
                        metavar='Path to a directory containing diffusion data',
                        default=None,
                        help='Contains dwi.nii.gz, bval, bvec, nodif_brain_mask.nii.gz files, or the outputs from FSLs bedpostx Formatted according to the FSL default tree structure found at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX.\n')
    parser.add_argument('-id',
                        metavar='Subject ID',
                        default=None,
                        required=False,
                        help='An arbitrary subject identifier OR list of subject identifiers, separated by comma and of equivalent length to the list of input files indicated with the -i flag.\n')
    parser.add_argument('-a',
                        metavar='Atlas',
                        default=None,
                        help='Specify a coordinate atlas parcellation from those made publically available in nilearn. If you wish to iterate your pynets run over multiple nilearn atlases, separate them by comma. e.g. -a \'atlas_aal,atlas_destrieux_2009\' Available nilearn atlases are:\n\natlas_aal\natlas_talairach_gyrus\natlas_talairach_ba\natlas_talairach_lobe\natlas_harvard_oxford\natlas_destrieux_2009\natlas_msdl\ncoords_dosenbach_2010\ncoords_power_2011\natlas_pauli_2017.\n')
    parser.add_argument('-ua',
                        metavar='Path to parcellation file',
                        default=None,
                        help='Path to parcellation/atlas file in .nii format. If specifying a list of paths to multiple user atlases, separate them by comma.\n')
    parser.add_argument('-pm',
                        metavar='Cores,memory',
                        default=None,
                        help='Number of cores to use, number of GB of memory to use for single subject run, entered as two integers seperated by a comma.\n')
    parser.add_argument('-n',
                        metavar='Resting-state network',
                        default=None,
                        help='Optionally specify the name of any of the 2017 Yeo-Schaefer RSNs (7-network or 17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar. If listing multiple RSNs, separate them by comma. (e.g. -n \'Default,Cont,SalVentAttn)\'.\n')
    parser.add_argument('-thr',
                        metavar='Graph threshold',
                        default='0.00',
                        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is proportional thresholding. If omitted, no thresholding will be applied.\n')
    parser.add_argument('-ns',
                        metavar='Node size',
                        default=4,
                        help='Optionally specify coordinate-based node radius size(s). Default is 4 mm. If you wish to iterate the pipeline across multiple node sizes, separate the list by comma (e.g. 2,4,6).\n')
    parser.add_argument('-sm',
                        metavar='Smoothing value (mm fwhm)',
                        default=0,
                        help='Optionally specify smoothing width(s). Default is 0 / no smoothing. If you wish to iterate the pipeline across multiple smoothing values, separate the list by comma (e.g. 2,4,6).\n')
    parser.add_argument('-m',
                        metavar='Path to mask image',
                        default=None,
                        help='Optionally specify a thresholded binarized mask image and retain only those nodes contained within that mask for functional connectome estimation, or constrain the tractography in the case of structural connectome estimation.\n')
    parser.add_argument('-mod',
                        metavar='Graph estimator type',
                        default=None,
                        help='Specify matrix estimation type. For fMRI, options models include: corr for correlation, cov for covariance, sps for precision covariance, partcorr for partial correlation. sps type is used by default. If skgmm is installed (https://github.com/skggm/skggm), then QuicGraphicalLasso, QuicGraphicalLassoCV, QuicGraphicalLassoEBIC, and AdaptiveQuicGraphicalLasso. For dMRI, models include ball_and_stick, tensor, and csd.\n')
    parser.add_argument('-conf',
                        metavar='Confounds',
                        default=None,
                        help='Optionally specify a path to a confound regressor file to reduce noise in the time-series estimation for the graph. This can also be a list of paths, separated by comma and of equivalent length to the list of input files indicated with the -i flag.\n')
    parser.add_argument('-anat',
                        metavar='Path to preprocessed anatomical image',
                        default=None,
                        help='Optional with the -bpx flag to initiate probabilistic connectome estimation using parcels (recommended) as opposed to coordinate-based spherical volumes.\n')
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
    parser.add_argument('-ref',
                        metavar='Atlas reference file path',
                        default=None,
                        help='Specify the path to the atlas reference .txt file.\n')
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
    parser.add_argument('-cm',
                        metavar='Cluster mask',
                        default=None,
                        help='Specify the path to the mask within which to perform clustering. If specifying a list of paths to multiple cluster masks, separate them by comma.')
    parser.add_argument('-ct',
                        metavar='Clustering type',
                        default='ncut',
                        help='Specify the types of clustering to use. Options include ncut, ward, kmeans, complete, and average. If specifying a list of clustering types, separate them by comma.')
    parser.add_argument('-p',
                        metavar='Pruning strategy',
                        default=1,
                        help='Include this flag to prune the resulting graph of any isolated (1) or isolated + fully disconnected (2) nodes. Default pruning=1 and removes isolated nodes. Include -p 0 to disable pruning.\n')
    parser.add_argument('-s',
                        metavar='Number of samples',
                        default='5000',
                        help='Include this flag to manually specify number of fiber samples for probtrackx2 in structural connectome estimation (default is 500). PyNets parallelizes probtrackx2 by samples, but more samples can increase connectome estimation time considerably.\n')
    parser.add_argument('-plug',
                        metavar='Scheduler type',
                        default='MultiProc',
                        help='Include this flag to specify a workflow plugin other than the default MultiProc. Options include: Linear, SGE, PBS, SLURM, SGEgraph, SLURMgraph.\n')
    parser.add_argument('-parc',
                        default=False,
                        action='store_true',
                        help='Include this flag to use parcels instead of coordinates as nodes.\n')
    parser.add_argument('-dt',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to threshold to achieve a given density or densities indicated by the -thr and -min_thr, -max_thr, -step_thr flags, respectively.\n')
    parser.add_argument('-mst',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to apply local thresholding via the Minimum Spanning Tree approach. -thr values in this case correspond to a target density (if the -dt flag is also included), otherwise a target proportional threshold.\n')
    parser.add_argument('-df',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to apply local thresholding via the disparity filter approach. -thr values in this case correspond to α.\n')
    #    parser.add_argument('-at',
    #        default=False,
    #        action='store_true',
    #        help='Optionally use this flag if you wish to activate adaptive thresholding')
    parser.add_argument('-plt',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to activate plotting of adjacency matrices, connectomes, and time-series.\n')
    parser.add_argument('-names',
                        default=False,
                        action='store_true',
                        help='Optionally use this flag if you wish to map nodes to AAL labels.\n')
    parser.add_argument('-v',
                        default=False,
                        action='store_true',
                        help='Verbose print for debugging.\n')
    return parser


def build_workflow(args, retval):
    import os
    import sys
    import warnings
    import timeit
    import numpy as np
    warnings.simplefilter("ignore")
    try:
        import pynets
    except ImportError:
        print('PyNets not installed! Ensure that you are using the correct python version.')
    from pynets.utils import do_dir_path

    # Start time clock
    start_time = timeit.default_timer()

    # Set Arguments to global variables
    input_file = args.i
    dwi_dir = args.dwi
    graph_pre = args.g
    multi_graph = list(str(graph_pre).split(','))
    if len(multi_graph) > 1:
        graph = None
    elif multi_graph == ['None']:
        graph = None
        multi_graph = None
    else:
        graph = multi_graph[0]
        multi_graph = None
    ID = args.id
    resources = args.pm
    if resources:
        procmem = list(eval(str(resources)))
    else:
        from multiprocessing import cpu_count
        nthreads = cpu_count()
        procmem = [int(nthreads), int(float(nthreads)*2)]
    thr = float(args.thr)
    node_size_pre = args.ns
    node_size = list(str(node_size_pre).split(','))
    if len(node_size) > 1:
        node_size_list = node_size
        node_size = None
    elif node_size == ['None']:
        node_size = None
        node_size_list = None
    else:
        node_size = node_size[0]
        node_size_list = None
    smooth_pre = args.sm
    smooth = list(str(smooth_pre).split(','))
    if len(smooth) > 1:
        smooth_list = smooth
        smooth = 0
    elif smooth == ['None']:
        smooth = 0
        smooth_list = None
    else:
        smooth = smooth[0]
        smooth_list = None
    mask = args.m
    conn_model_pre = args.mod
    conn_model = list(str(conn_model_pre).split(','))
    if len(conn_model) > 1:
        conn_model_list = conn_model
        conn_model = None
    elif conn_model == ['None']:
        conn_model = None
        conn_model_list = None
    else:
        conn_model = conn_model[0]
        conn_model_list = None
    conf = args.conf
    dens_thresh = args.dt
    min_span_tree = args.mst
    disp_filt = args.df
    clust_type_pre = args.ct
    clust_type = list(str(clust_type_pre).split(','))
    if len(clust_type) > 1:
        clust_type_list = clust_type
        clust_type = None
    elif clust_type == ['None']:
        clust_type = None
        clust_type_list = None
    else:
        clust_type = clust_type[0]
        clust_type_list = None
#    adapt_thresh=args.at
    adapt_thresh = False
    plot_switch = args.plt
    min_thr = args.min_thr
    max_thr = args.max_thr
    step_thr = args.step_thr
    anat_loc = args.anat
    num_total_samples = args.s
    parc = args.parc
    ref_txt = args.ref
    k = args.k
    k_min = args.k_min
    k_max = args.k_max
    k_step = args.k_step
    clust_mask_pre = args.cm
    prune = args.p
    plugin_type = args.plug
    use_AAL_naming = args.names
    verbose = args.v
    clust_mask = list(str(clust_mask_pre).split(','))
    if len(clust_mask) > 1:
        clust_mask_list = clust_mask
        clust_mask = None
    elif clust_mask == ['None']:
        clust_mask = None
        clust_mask_list = None
    else:
        clust_mask = clust_mask[0]
        clust_mask_list = None
    network_pre = args.n
    network = list(str(network_pre).split(','))
    if len(network) > 1:
        multi_nets = network
        network = None
    elif network == ['None']:
        network = None
        multi_nets = None
    else:
        network = network[0]
        multi_nets = None
    uatlas_select_pre = args.ua
    atlas_select_pre = args.a
    uatlas_select = list(str(uatlas_select_pre).split(','))
    if len(uatlas_select) > 1:
        user_atlas_list = uatlas_select
        uatlas_select = user_atlas_list[0]
    elif uatlas_select == ['None']:
        uatlas_select = None
        user_atlas_list = None
    else:
        uatlas_select = uatlas_select[0]
        user_atlas_list = None
    atlas_select = list(str(atlas_select_pre).split(','))
    if len(atlas_select) > 1:
        multi_atlas = atlas_select
        atlas_select = atlas_select[0]
    elif len(atlas_select) > 1:
        multi_atlas = atlas_select
        atlas_select = atlas_select[0]
    elif atlas_select == ['None']:
        atlas_select = None
        multi_atlas = None
    else:
        atlas_select = atlas_select[0]
        multi_atlas = None
    print('\n\n\n------------------------------------------------------------------------\n')

    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009',
                            'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coord_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']

    if min_thr is not None and max_thr is not None and step_thr is not None:
        multi_thr = True
    elif min_thr is not None or max_thr is not None or step_thr is not None:
        raise ValueError('Error: Missing either min_thr, max_thr, or step_thr flags!')
    else:
        multi_thr = False

    # Check required inputs for existence, and configure run
    if input_file:
        if input_file.endswith('.txt'):
            with open(input_file) as f:
                subjects_list = f.read().splitlines()
        elif ',' in input_file:
            subjects_list = list(str(input_file).split(','))
        else:
            subjects_list = None
    else:
        subjects_list = None

    if input_file is None and dwi_dir is None and graph is None and multi_graph is None:
        raise ValueError("\nError: You must include a file path to either a standard space functional image in .nii or .nii.gz format with the -i flag.")

    if input_file and dwi_dir and subjects_list:
        raise ValueError("\nError: PyNets does not yet support joint functional-structural connectometry across multiple subjects.")

    if ID is None and subjects_list is None:
        raise ValueError("\nError: You must include a subject ID in your command line call.")

    if subjects_list and ',' in ID:
        ID = list(str(ID).split(','))
        if len(ID) != len(subjects_list):
            raise ValueError("Error: Length of ID list does not correspond to length of input file list.")

    if conf:
        if ',' in conf:
            conf = list(str(conf).split(','))
            if len(conf) != len(subjects_list):
                raise ValueError("Error: Length of confound regressor list does not correspond to length of input file list.")

    if anat_loc is not None and dwi_dir is None:
        raise RuntimeWarning('Warning: anatomical image specified, but no bedpostx directory specified. Anatomical images are only supported for structural connectome estimation at this time.')

    if multi_thr is True:
        thr = None
    else:
        min_thr = None
        max_thr = None
        step_thr = None

    if (k_min is not None and k_max is not None) and k is None and clust_mask_list is not None and clust_type_list is not None:
        k_clustering = 8
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is not None and clust_type_list is not None:
        k_clustering = 7
    elif (k_min is not None and k_max is not None) and k is None and clust_mask_list is None and clust_type_list is not None:
        k_clustering = 6
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is None and clust_type_list is not None:
        k_clustering = 5
    elif (k_min is not None and k_max is not None) and k is None and clust_mask_list is not None and clust_type_list is None:
        k_clustering = 4
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is not None and clust_type_list is None:
        k_clustering = 3
    elif (k_min is not None and k_max is not None) and k is None and clust_mask_list is None and clust_type_list is None:
        k_clustering = 2
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is None and clust_type_list is None:
        k_clustering = 1
    else:
        k_clustering = 0

    if subjects_list:
        print('\nRunning workflow of workflows across multiple subjects:')
    elif subjects_list is None:
        print('\nRunning workflow across single subject:')
    print(str(ID))

    if input_file:
        if uatlas_select is not None and k_clustering == 0 and user_atlas_list is None:
            atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
            print("%s%s" % ("\nUser atlas: ", atlas_select_par))
            if subjects_list:
                for input_file in subjects_list:
                    do_dir_path(atlas_select_par, input_file)
            else:
                do_dir_path(atlas_select_par, input_file)
        elif uatlas_select is not None and user_atlas_list is None and k_clustering == 0:
            atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
            print("%s%s" % ("\nUser atlas: ", atlas_select_par))
            if subjects_list:
                for input_file in subjects_list:
                    do_dir_path(atlas_select_par, input_file)
            else:
                do_dir_path(atlas_select_par, input_file)
        elif user_atlas_list is not None:
            print('\nIterating across multiple user atlases...')
            if subjects_list:
                for uatlas_select in user_atlas_list:
                    atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
                    print(atlas_select_par)
                    for input_file in subjects_list:
                        do_dir_path(atlas_select_par, input_file)
            else:
                for uatlas_select in user_atlas_list:
                    atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
                    print(atlas_select_par)
                    do_dir_path(atlas_select_par, input_file)
        if k_clustering == 1:
            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
            print("%s%s" % ("\nCluster atlas: ", atlas_select_clust))
            print("\nClustering within mask at a single resolution...")
            if subjects_list:
                for input_file in subjects_list:
                    do_dir_path(atlas_select_clust, input_file)
            else:
                do_dir_path(atlas_select_clust, input_file)
        elif k_clustering == 2:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            print("\nClustering within mask at multiple resolutions...")
            if subjects_list:
                for k in k_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                    for input_file in subjects_list:
                        do_dir_path(atlas_select_clust, input_file)
            else:
                for k in k_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                    do_dir_path(atlas_select_clust, input_file)
        elif k_clustering == 3:
            print("\nClustering within multiple masks at a single resolution...")
            if subjects_list:
                for clust_mask in clust_mask_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    for input_file in subjects_list:
                        do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_mask in clust_mask_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                    do_dir_path(atlas_select_clust, input_file)
            clust_mask = None
        elif k_clustering == 4:
            print("\nClustering within multiple masks at multiple resolutions...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            if subjects_list:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        for input_file in subjects_list:
                            do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        do_dir_path(atlas_select_clust, input_file)
            clust_mask = None
        elif k_clustering == 5:
            for clust_type in clust_type_list:
                cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                print("%s%s" % ("\nCluster atlas: ", atlas_select_clust))
                print("\nClustering within mask at a single resolution using multiple clustering methods...")
                if subjects_list:
                    for input_file in subjects_list:
                        do_dir_path(atlas_select_clust, input_file)
                else:
                    do_dir_path(atlas_select_clust, input_file)
            clust_type = None
        elif k_clustering == 6:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            print("\nClustering within mask at multiple resolutions using multiple clustering methods...")
            if subjects_list:
                for clust_type in clust_type_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                        for input_file in subjects_list:
                            do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_type in clust_type_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                        do_dir_path(atlas_select_clust, input_file)
            clust_type = None
        elif k_clustering == 7:
            print("\nClustering within multiple masks at a single resolution using multiple clustering methods...")
            if subjects_list:
                for clust_type in clust_type_list:
                    for clust_mask in clust_mask_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                        for input_file in subjects_list:
                            do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_type in clust_type_list:
                    for clust_mask in clust_mask_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                        do_dir_path(atlas_select_clust, input_file)
            clust_mask = None
            clust_type = None
        elif k_clustering == 8:
            print("\nClustering within multiple masks at multiple resolutions using multiple clustering methods...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            if subjects_list:
                for clust_type in clust_type_list:
                    for clust_mask in clust_mask_list:
                        for k in k_list:
                            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                            atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                            print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                            for input_file in subjects_list:
                                do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_type in clust_type_list:
                    for clust_mask in clust_mask_list:
                        for k in k_list:
                            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                            atlas_select_clust = "%s%s%s%s%s" % (cl_mask_name, '_', clust_type, '_k', k)
                            do_dir_path(atlas_select_clust, input_file)
            clust_mask = None
            clust_type = None
        elif (user_atlas_list is not None or uatlas_select is not None) and (k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1) and atlas_select is None:
            print('Error: the -ua flag cannot be used alone with the clustering option. Use the -cm flag instead.')
            sys.exit(0)
        if multi_atlas is not None:
            print('\nIterating across multiple predefined atlases...')
            if subjects_list:
                for input_file in subjects_list:
                    for atlas_select in multi_atlas:
                        if parc is True and (atlas_select in nilearn_coord_atlases or atlas_select in nilearn_prob_atlases):
                            raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
                        else:
                            print(atlas_select)
                            do_dir_path(atlas_select, input_file)
            else:
                for atlas_select in multi_atlas:
                    if parc is True and (atlas_select in nilearn_coord_atlases or atlas_select in nilearn_prob_atlases):
                        raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
                    else:
                        print(atlas_select)
                        do_dir_path(atlas_select, input_file)
        elif atlas_select is not None:
            if parc is True and (atlas_select in nilearn_coord_atlases or atlas_select in nilearn_prob_atlases):
                raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
            else:
                print("%s%s" % ("\nPredefined atlas: ", atlas_select))
                if subjects_list:
                    for input_file in subjects_list:
                        do_dir_path(atlas_select, input_file)
                else:
                    do_dir_path(atlas_select, input_file)
        else:
            if uatlas_select is None and k == 0:
                raise KeyError('\nERROR: No atlas specified!')
            else:
                pass
    elif graph or multi_graph:
        network = 'custom_graph'
        thr = 0
        mask = 'None'
        k_clustering = 0
        node_size = 'None'
        smooth = 'None'
        conn_model = 'None'
        if multi_graph:
            print('\nUsing multiple custom input graphs...')
            conn_model = None
            conn_model_list = []
            i = 1
            for graph in multi_graph:
                conn_model_list.append(str(i))
                if '.txt' in graph:
                    graph_name = os.path.basename(graph).split('.txt')[0]
                elif '.npy' in graph:
                    graph_name = os.path.basename(graph).split('.npy')[0]
                else:
                    print('Error: input graph file format not recognized. See -help for supported formats.')
                    sys.exit(0)
                print(graph_name)
                atlas_select = "%s%s%s" % (graph_name, '_', ID)
                do_dir_path(atlas_select, graph)
                i = i + 1
        else:
            if '.txt' in graph:
                graph_name = os.path.basename(graph).split('.txt')[0]
            elif '.npy' in graph:
                graph_name = os.path.basename(graph).split('.npy')[0]
            else:
                print('Error: input graph file format not recognized. See -help for supported formats.')
                sys.exit(0)
            print('\nUsing single custom graph input...')
            print(graph_name)
            atlas_select = "%s%s%s" % (graph_name, '_', ID)
            do_dir_path(atlas_select, graph)

    if graph is None and multi_graph is None:
        if network is not None:
            print("%s%s" % ('\nUsing resting-state network pipeline for: ', network))
        elif multi_nets is not None:
            network = multi_nets[0]
            print("%s%d%s%s%s" % ('\nIterating workflow across ', len(multi_nets), ' networks: ',
                                  str(', '.join(str(n) for n in multi_nets)), '...'))
        else:
            print("\nUsing whole-brain pipeline...")

        if node_size_list:
            print("%s%s%s" % ('\nGrowing spherical nodes across multiple radius sizes: ',
                              str(', '.join(str(n) for n in node_size_list)), '...'))
        elif parc is True:
            print("\nUsing parcels as nodes")
        else:
            print("%s%s%s" % ("\nUsing node size of: ", node_size, 'mm...'))

        if smooth_list:
            print("%s%s%s" % ('\nApplying smoothing to node signal at multiple FWHM mm values: ',
                              str(', '.join(str(n) for n in smooth_list)), '...'))
        elif float(smooth) > 0:
            print("%s%s%s" % ("\nApplying smoothing to node signal at: ", smooth, 'FWHM mm...'))
        else:
            smooth = 0

        if conn_model_list:
            print("%s%s%s" % ('\nIterating graph estimation across multiple connectivity models: ',
                              str(', '.join(str(n) for n in conn_model_list)), '...'))
        else:
            print("%s%s" % ("\nUsing connectivity model: ", conn_model))

    if dwi_dir:
        print("%s%s" % ('Bedpostx Directory: ', dwi_dir))
        print("%s%s" % ('Number of fiber samples for tracking: ', num_total_samples))
        if anat_loc is not None:
            print("%s%s" % ('Anatomical Image: ', anat_loc))
        if network is not None:
            print("%s%s" % ('RSN: ', network))
        # Set directory path containing input file
        nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
        input_file = nodif_brain_mask_path
        if user_atlas_list is not None:
            print('\nIterating across multiple user atlases...')
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    for uatlas_select in user_atlas_list:
                        atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
                        print(atlas_select_par)
                        do_dir_path(atlas_select_par, nodif_brain_mask_path)
            else:
                for uatlas_select in user_atlas_list:
                    atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
                    print(atlas_select_par)
                    do_dir_path(atlas_select_par, nodif_brain_mask_path)
        elif uatlas_select is not None and user_atlas_list is None:
            atlas_select_par = uatlas_select.split('/')[-1].split('.')[0]
            ref_txt = "%s%s" % (uatlas_select.split('/')[-1:][0].split('.')[0], '.txt')
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    do_dir_path(atlas_select_par, nodif_brain_mask_path)
            else:
                do_dir_path(atlas_select_par, nodif_brain_mask_path)
        if multi_atlas is not None:
            print('\nIterating across multiple predefined atlases...')
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    for atlas_select in multi_atlas:
                        if parc is True and atlas_select in nilearn_coord_atlases:
                            raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
                        else:
                            print(atlas_select)
                            do_dir_path(atlas_select, nodif_brain_mask_path)
            else:
                for atlas_select in multi_atlas:
                    if parc is True and atlas_select in nilearn_coord_atlases:
                        raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
                    else:
                        print(atlas_select)
                        do_dir_path(atlas_select, nodif_brain_mask_path)
        elif atlas_select is not None:
            if parc is True and atlas_select in nilearn_coord_atlases:
                raise ValueError("%s%s%s" % ('\nERROR: ', atlas_select, ' is a coordinate atlas and cannot be combined with the -parc flag.'))
            else:
                print("%s%s" % ("\nNilearn atlas: ", atlas_select))
                if subjects_list:
                    for dwi_dir in subjects_list:
                        nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                        do_dir_path(atlas_select, nodif_brain_mask_path)
                else:
                    do_dir_path(atlas_select, nodif_brain_mask_path)
        else:
            if uatlas_select is None:
                raise KeyError('\nERROR: No atlas specified!')
            else:
                pass

        merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
        if os.path.exists(merged_f_samples_path) is True:
            conn_model = 'ball_and_stick'
        elif conn_model is None:
            conn_model = 'tensor'
        else:
            conn_model = conn_model

    if dwi_dir and not input_file:
        print('\nRunning structural connectometry only...')
        if subjects_list:
            for dwi_dir in subjects_list:
                print("%s%s" % ('Diffusion directory: ', dwi_dir))
        else:
            print("%s%s" % ('Diffusion directory: ', dwi_dir))
    elif input_file and dwi_dir is None:
        print('\nRunning functional connectometry only...')
        if subjects_list:
            for input_file in subjects_list:
                print("%s%s" % ('Functional file: ', input_file))
        else:
            print("%s%s" % ('Functional file: ', input_file))
    elif input_file and dwi_dir:
        print('\nRunning joint structural-functional connectometry...')
        print("%s%s" % ('Functional file: ', input_file))
        print("%s%s" % ('Diffusion directory: ', dwi_dir))
    print('\n-------------------------------------------------------------------------\n\n')

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
    # import sys
    # sys.exit(0)

    # Import wf core and interfaces
    import random
    from pynets.utils import CollectPandasDfs, Export2Pandas, ExtractNetStats, collect_pandas_join
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets.workflows import workflow_selector

    def init_wf_single_subject(ID, input_file, atlas_select, network, node_size, mask, thr, uatlas_select,
                               multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir,
                               multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k,
                               clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                               node_size_list, num_total_samples, graph, conn_model_list, min_span_tree, verbose,
                               plugin_type, use_AAL_naming, multi_graph, smooth, smooth_list, disp_filt, clust_type,
                               clust_type_list):
        wf = pe.Workflow(name="%s%s%s%s" % ('Wf_single_subject_', ID, '_', random.randint(1000, 1000)))
        inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'network', 'thr', 'node_size', 'mask', 'multi_nets',
                                                          'conn_model', 'plot_switch', 'graph', 'prune', 'smooth']),
                            name='inputnode')
        inputnode.inputs.ID = ID
        inputnode.inputs.network = network
        inputnode.inputs.thr = thr
        inputnode.inputs.node_size = node_size
        inputnode.inputs.mask = mask
        inputnode.inputs.multi_nets = multi_nets
        inputnode.inputs.conn_model = conn_model
        inputnode.inputs.plot_switch = plot_switch
        inputnode.inputs.graph = graph
        inputnode.inputs.prune = prune
        inputnode.inputs.smooth = smooth

        meta_wf = workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, uatlas_select, multi_nets,
                                    conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir, anat_loc, parc,
                                    ref_txt, procmem, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k,
                                    clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                                    node_size_list, num_total_samples, conn_model_list, min_span_tree, verbose, plugin_type,
                                    use_AAL_naming, smooth, smooth_list, disp_filt, clust_type, clust_type_list)
        wf.add_nodes([meta_wf])

        # Set resource restrictions at level of the meta-meta wf
        if input_file:
            wf_selected = "%s%s" % ('functional_connectometry_', ID)
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('fetch_nodes_and_labels_node')._n_procs = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('fetch_nodes_and_labels_node')._mem_gb = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('extract_ts_node')._n_procs = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('extract_ts_node')._mem_gb = 4
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('node_gen_node')._n_procs = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('node_gen_node')._mem_gb = 1
            if k_clustering > 0:
                wf.get_node(meta_wf.name).get_node(wf_selected).get_node('clustering_node')._n_procs = 1
                wf.get_node(meta_wf.name).get_node(wf_selected).get_node('clustering_node')._mem_gb = 8
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('get_conn_matrix_node')._n_procs = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('get_conn_matrix_node')._mem_gb = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('thresh_func_node')._n_procs = 1
            wf.get_node(meta_wf.name).get_node(wf_selected).get_node('thresh_func_node')._mem_gb = 1

        # Fully-automated graph analysis
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name="ExtractNetStats",
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path',
                                              'mask', 'prune', 'node_size', 'smooth'], nested=True)

        # Export graph analysis results to pandas dataframes
        export_to_pandas_node = pe.MapNode(interface=Export2Pandas(), name="Export2Pandas",
                                           iterfield=['csv_loc', 'ID', 'network', 'mask'], nested=True)

        # Aggregate list of paths to pandas dataframe pickles
        collect_pd_list_net_pickles_node = pe.Node(niu.Function(input_names=['net_pickle_mt'],
                                                                output_names=['net_pickle_mt_out'],
                                                                function=collect_pandas_join),
                                                   name="AggregatePandasPickles")

        # Combine dataframes across models
        collect_pandas_dfs_node = pe.Node(interface=CollectPandasDfs(), name="CollectPandasDfs",
                                          input_names=['network', 'ID', 'net_pickle_mt_list', 'plot_switch',
                                                       'multi_nets'])

        wf.connect([
            (meta_wf.get_node('pass_meta_outs_node'), net_mets_node, [('est_path_iterlist', 'est_path'),
                                                                      ('network_iterlist', 'network'),
                                                                      ('thr_iterlist', 'thr'),
                                                                      ('ID_iterlist', 'ID'),
                                                                      ('conn_model_iterlist', 'conn_model'),
                                                                      ('mask_iterlist', 'mask'),
                                                                      ('prune_iterlist', 'prune'),
                                                                      ('node_size_iterlist', 'node_size'),
                                                                      ('smooth_iterlist', 'smooth')]),
            (meta_wf.get_node('pass_meta_outs_node'), export_to_pandas_node, [('network_iterlist', 'network'),
                                                                              ('ID_iterlist', 'ID'),
                                                                              ('mask_iterlist', 'mask')]),
            (net_mets_node, export_to_pandas_node, [('out_file', 'csv_loc')]),
            (inputnode, collect_pandas_dfs_node, [('network', 'network'),
                                                  ('ID', 'ID'),
                                                  ('plot_switch', 'plot_switch'),
                                                  ('multi_nets', 'multi_nets')]),
            (export_to_pandas_node, collect_pd_list_net_pickles_node, [('net_pickle_mt', 'net_pickle_mt')]),
            (collect_pd_list_net_pickles_node, collect_pandas_dfs_node, [('net_pickle_mt_out', 'net_pickle_mt_list')])
        ])

        # Raw graph case
        if graph or multi_graph:
            wf.disconnect([(meta_wf.get_node('pass_meta_outs_node'), net_mets_node,
                            [('est_path_iterlist', 'est_path'),
                             ('network_iterlist', 'network'),
                             ('thr_iterlist', 'thr'),
                             ('ID_iterlist', 'ID'),
                             ('conn_model_iterlist', 'conn_model'),
                             ('mask_iterlist', 'mask'),
                             ('prune_iterlist', 'prune'),
                             ('node_size_iterlist', 'node_size'),
                             ('smooth_iterlist', 'smooth')])
                           ])
            wf.disconnect([(meta_wf.get_node('pass_meta_outs_node'), export_to_pandas_node,
                            [('network_iterlist', 'network'),
                             ('ID_iterlist', 'ID'),
                             ('mask_iterlist', 'mask')])
                           ])
            wf.remove_nodes([meta_wf])
            # Multiple raw graphs
            if multi_graph:
                net_mets_node.inputs.est_path = multi_graph
                net_mets_node.inputs.ID = [ID] * len(multi_graph)
                net_mets_node.inputs.mask = [mask] * len(multi_graph)
                net_mets_node.inputs.node_size = [node_size] * len(multi_graph)
                net_mets_node.inputs.smooth = [smooth] * len(multi_graph)
                net_mets_node.inputs.thr = [thr] * len(multi_graph)
                net_mets_node.inputs.prune = [prune] * len(multi_graph)
                net_mets_node.inputs.network = [network] * len(multi_graph)
                net_mets_node.inputs.conn_model = conn_model_list

                export_to_pandas_node.inputs.ID = [ID] * len(multi_graph)
                export_to_pandas_node.inputs.mask = [mask] * len(multi_graph)
                export_to_pandas_node.inputs.network = [network] * len(multi_graph)
            else:
                wf.connect([(inputnode, net_mets_node, [('network', 'network'),
                                                        ('thr', 'thr'),
                                                        ('ID', 'ID'),
                                                        ('conn_model', 'conn_model'),
                                                        ('mask', 'mask'),
                                                        ('prune', 'prune'),
                                                        ('node_size', 'node_size'),
                                                        ('smooth', 'smooth'),
                                                        ('graph', 'est_path')])
                            ])
                wf.connect([(inputnode, export_to_pandas_node, [('network', 'network'),
                                                                ('ID', 'ID'),
                                                                ('mask', 'mask')])
                            ])

        return wf

    # Multi-subject pipeline
    def wf_multi_subject(ID, subjects_list, atlas_select, network, node_size, mask, thr, uatlas_select, multi_nets,
                         conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir, multi_thr,
                         multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask,
                         k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list,
                         num_total_samples, graph, conn_model_list, min_span_tree, verbose, plugin_type, use_AAL_naming,
                         multi_graph, smooth, smooth_list, disp_filt, clust_type, clust_type_list):

        wf_multi = pe.Workflow(name="%s%s" % ('PyNets_multisub_', random.randint(1000, 9000)))
        i = 0
        for _file in subjects_list:
            if conf:
                conf_sub = conf[i]
            else:
                conf_sub = None
            wf_single_subject = init_wf_single_subject(
                ID=ID[i], input_file=_file, atlas_select=atlas_select,
                network=network, node_size=node_size, mask=mask, thr=thr, uatlas_select=uatlas_select,
                multi_nets=multi_nets, conn_model=conn_model, dens_thresh=dens_thresh, conf=conf_sub,
                adapt_thresh=adapt_thresh, plot_switch=plot_switch, dwi_dir=dwi_dir, multi_thr=multi_thr,
                multi_atlas= multi_atlas, min_thr=min_thr, max_thr=max_thr, step_thr=step_thr, anat_loc=anat_loc,
                parc=parc, ref_txt=ref_txt, procmem='auto', k=k, clust_mask=clust_mask, k_min=k_min, k_max=k_max,
                k_step=k_step, k_clustering=k_clustering, user_atlas_list=user_atlas_list,
                clust_mask_list=clust_mask_list, prune=prune, node_size_list=node_size_list,
                num_total_samples=num_total_samples, graph=graph, conn_model_list=conn_model_list,
                min_span_tree=min_span_tree, verbose=verbose, plugin_type=plugin_type, use_AAL_naming=use_AAL_naming,
                multi_graph=multi_graph, smooth=smooth, smooth_list=smooth_list, disp_filt=disp_filt,
                clust_type=clust_type, clust_type_list=clust_type_list)
            wf_multi.add_nodes([wf_single_subject])
            # Restrict nested meta-meta wf resources at the level of the group wf
            if input_file:
                wf_selected = "%s%s" % ('functional_connectometry_', ID[i])
                meta_wf_name = "%s%s" % ('Meta_wf_', ID[i])
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('fetch_nodes_and_labels_node')._n_procs = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('fetch_nodes_and_labels_node')._mem_gb = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('extract_ts_node')._n_procs = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('extract_ts_node')._mem_gb = 4
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('node_gen_node')._n_procs = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('node_gen_node')._mem_gb = 1
                if k_clustering > 0:
                    wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('clustering_node')._n_procs = 1
                    wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('clustering_node')._mem_gb = 8
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('get_conn_matrix_node')._n_procs = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('get_conn_matrix_node')._mem_gb = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('thresh_func_node')._n_procs = 1
                wf_multi.get_node(wf_single_subject.name).get_node(meta_wf_name).get_node(wf_selected).get_node('thresh_func_node')._mem_gb = 1
            i = i + 1

        return wf_multi

    # Workflow generation
    # Multi-subject workflow generator

    if subjects_list:
        wf_multi = wf_multi_subject(ID, subjects_list, atlas_select, network, node_size, mask,
                                    thr, uatlas_select, multi_nets, conn_model, dens_thresh,
                                    conf, adapt_thresh, plot_switch, dwi_dir, multi_thr,
                                    multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc,
                                    ref_txt, procmem, k, clust_mask, k_min, k_max, k_step,
                                    k_clustering, user_atlas_list, clust_mask_list, prune,
                                    node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph,
                                    smooth, smooth_list, disp_filt, clust_type, clust_type_list)

        import shutil
        wf_multi.base_dir = "%s%s" % (os.getcwd(), '/Wf_multi_subject')
        if os.path.exists(wf_multi.base_dir):
            shutil.rmtree(wf_multi.base_dir)
        os.mkdir(wf_multi.base_dir)

        if verbose is True:
            from nipype import config, logging
            cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                        'hash_method': 'content'})
            config.update_config(cfg)
            config.update_config({'logging': {'log_directory': wf_multi.base_dir, 'log_to_file': True}})
            logging.update_logging(config)
            config.enable_debug_mode()
            wf_multi.config['logging']['workflow_level'] = 'DEBUG'
            wf_multi.config['logging']['utils_level'] = 'DEBUG'
            wf_multi.config['logging']['interface_level'] = 'DEBUG'
            wf_multi.config['logging']['log_directory'] = wf_multi.base_dir
            wf_multi.config['monitoring']['enabled'] = True
            wf_multi.config['monitoring']['sample_frequency'] = '0.5'
            wf_multi.config['monitoring']['summary_append'] = True

        wf_multi.config['execution']['crashdump_dir'] = wf_multi.base_dir
        wf_multi.config['execution']['display_variable'] = ':0'
        wf_multi.config['execution']['crashfile_format'] = 'txt'
        wf_multi.config['execution']['job_finished_timeout'] = 65
        wf_multi.config['execution']['stop_on_first_crash'] = False
        wf_multi.config['execution']['keep_inputs'] = True
        wf_multi.config['execution']['remove_unnecessary_outputs'] = False
        wf_multi.config['execution']['remove_node_directories'] = False
        plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'maxtasksperchild': 1}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf_multi.write_graph(graph2use="colored", format='png')
        wf_multi.run(plugin=plugin_type, plugin_args=plugin_args)
    # Single-subject workflow generator
    else:
        # Single-subject pipeline
        wf = init_wf_single_subject(ID, input_file, atlas_select, network, node_size, mask, thr, uatlas_select,
                                    multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir,
                                    multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt,
                                    procmem, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list,
                                    clust_mask_list, prune, node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph, smooth,
                                    smooth_list, disp_filt, clust_type, clust_type_list)

        import shutil
        base_dirname = "%s%s" % ('Wf_single_subject_', str(ID))
        if input_file:
            if os.path.exists("%s%s%s" % (os.path.dirname(input_file), '/', base_dirname)):
                shutil.rmtree("%s%s%s" % (os.path.dirname(input_file), '/', base_dirname))
            os.mkdir("%s%s%s" % (os.path.dirname(input_file), '/', base_dirname))
            wf.base_dir = os.path.dirname(input_file)
        elif dwi_dir:
            if os.path.exists("%s%s%s" % (dwi_dir, '/', base_dirname)):
                shutil.rmtree("%s%s%s" % (dwi_dir, '/', base_dirname))
            os.mkdir("%s%s%s" % (dwi_dir, '/', base_dirname))
            wf.base_dir = dwi_dir

        if verbose is True:
            from nipype import config, logging
            cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                        'hash_method': 'content'})
            config.update_config(cfg)
            config.update_config({'logging': {'log_directory': wf.base_dir, 'log_to_file': True}})
            logging.update_logging(config)
            config.enable_debug_mode()
            wf.config['logging']['workflow_level'] = 'DEBUG'
            wf.config['logging']['utils_level'] = 'DEBUG'
            wf.config['logging']['interface_level'] = 'DEBUG'
            wf.config['logging']['log_directory'] = wf.base_dir
            wf.config['monitoring']['enabled'] = True
            wf.config['monitoring']['sample_frequency'] = '0.5'
            wf.config['monitoring']['summary_append'] = True

        wf.config['execution']['crashdump_dir'] = wf.base_dir
        wf.config['execution']['crashfile_format'] = 'txt'
        wf.config['execution']['display_variable'] = ':0'
        wf.config['execution']['job_finished_timeout'] = 65
        wf.config['execution']['stop_on_first_crash'] = False
        wf.config['execution']['keep_inputs'] = True
        wf.config['execution']['remove_unnecessary_outputs'] = False
        wf.config['execution']['remove_node_directories'] = False
        wf.write_graph(graph2use="colored", format='png')
        if procmem != 'auto':
            plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1])}
            #plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'maxtasksperchild': 1}
            print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
            wf.run(plugin=plugin_type, plugin_args=plugin_args)
        else:
            wf.run(plugin=plugin_type)

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')
    return


def main():
    import sys
    try:
        from pynets.utils import do_dir_path
    except ImportError:
        print('PyNets not installed! Ensure that you are using the correct python version.')

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
    except RuntimeWarning:
        print('\nWARNING: Upgrade to python3 for forkserver functionality...')
        retval = dict()
        build_workflow(args, retval)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
