#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Tue Nov  7 10:40:07 2017
# Copyright (C) 2018
# @author: Derek Pisner

import sys
import argparse
import os
import timeit
import numpy as np
import warnings
warnings.simplefilter("ignore")
try:
    from pynets.utils import do_dir_path
except ImportError:
    print('PyNets not installed! Ensure that you are using the correct python version.')

# Parse args
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    parser = argparse.ArgumentParser()
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
                        help='Specify a coordinate atlas parcellation of those available in nilearn. If you wish to iterate your pynets run over multiple nilearn atlases, separate them by comma. e.g. -a \'atlas_aal,atlas_destrieux_2009\' Available nilearn atlases are:\n\natlas_aal\natlas_allen_2011\natlas_talairach_tissue\natlas_talairach_gyrus\natlas_talairach_ba\natlas_talairach_lobe\natlas_talairach_hemisphere\natlas_harvard_oxford\natlas_basc_multiscale_2015\natlas_craddock_2012\natlas_destrieux_2009\ncoords_dosenbach_2010\ncoords_power_2011\n')
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
                        help='Optionally specify the name of any of the 2017 Yeo-Schaefer RSNs (7-network or 17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar. If listing multiple RSNs, separate them by comma. (e.g. -n \'Default,Cont,SalVentAttn)\'\n')
    parser.add_argument('-thr',
                        metavar='Graph threshold',
                        default='0.00',
                        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is proportional thresholding. If omitted, no thresholding will be applied\n')
    parser.add_argument('-ns',
                        metavar='Node size',
                        default=4,
                        help='Optionally specify coordinate-based node radius size(s). Default is 4 mm. If you wish to iterate the pipeline across multiple node sizes, separate the list by comma (e.g. 2,4,6)\n')
    parser.add_argument('-sm',
                        metavar='Smoothing value (mm fwhm)',
                        default=0,
                        help='Optionally specify smoothing width(s). Default is 0 / no smoothing. If you wish to iterate the pipeline across multiple smoothing values, separate the list by comma (e.g. 2,4,6)\n')
    parser.add_argument('-m',
                        metavar='Path to mask image',
                        default=None,
                        help='Optionally specify a thresholded binarized mask image and retain only those nodes contained within that mask for functional connectome estimation, or constrain the tractography in the case of structural connectome estimation.\n')
    parser.add_argument('-mod',
                        metavar='Graph estimator type',
                        default=None,
                        help='Specify matrix estimation type. For fMRI, options models include: corr for correlation, cov for covariance, sps for precision covariance, partcorr for partial correlation. sps type is used by default. If skgmm is installed (https://github.com/skggm/skggm), then QuicGraphLasso, QuicGraphLassoCV, QuicGraphLassoEBIC, and AdaptiveGraphLasso. For dMRI, models include ball_and_stick, tensor, and csd.\n')
    parser.add_argument('-conf',
                        metavar='Confounds',
                        default=None,
                        help='Optionally specify a path to a confound regressor file to reduce noise in the time-series estimation for the graph. This can also be a list of paths, separated by comma and of equivalent length to the list of input files indicated with the -i flag. \n')
    parser.add_argument('-anat',
                        metavar='Path to preprocessed anatomical image',
                        default=None,
                        help='Optional with the -bpx flag to initiate probabilistic connectome estimation using parcels (recommended) as opposed to coordinate-based spherical volumes.\n')
    parser.add_argument('-min_thr',
                        metavar='Multi-thresholding minimum threshold',
                        default=None,
                        help='Minimum threshold for multi-thresholding\n')
    parser.add_argument('-max_thr',
                        metavar='Multi-thresholding maximum threshold',
                        default=None,
                        help='Maximum threshold for multi-thresholding')
    parser.add_argument('-step_thr',
                        metavar='Multi-thresholding step size',
                        default=None,
                        help='Threshold step value for multi-thresholding. Default is 0.01.\n')
    parser.add_argument('-ref',
                        metavar='Atlas reference file path',
                        default=None,
                        help='Specify the path to the atlas reference .txt file\n')
    parser.add_argument('-k',
                        metavar='Number of k clusters',
                        default=None,
                        help='Specify a number of clusters to produce\n')
    parser.add_argument('-k_min',
                        metavar='Min k clusters',
                        default=None,
                        help='Specify the minimum k clusters\n')
    parser.add_argument('-k_max',
                        metavar='Max k clusters',
                        default=None,
                        help='Specify the maximum k clusters\n')
    parser.add_argument('-k_step',
                        metavar='K cluster step size',
                        default=None,
                        help='Specify the step size of k cluster iterables\n')
    parser.add_argument('-cm',
                        metavar='Cluster mask',
                        default=None,
                        help='Specify the path to the mask within which to perform clustering. If specifying a list of paths to multiple cluster masks, separate them by comma.')
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
                        help='Optionally use this flag if you wish to apply local thresholding via the Minimum Spanning Tree approach.\n')
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
                        help='Optionally use this flag if you wish to map nodes to AAL labels\n')
    parser.add_argument('-v',
                        default=False,
                        action='store_true',
                        help='Verbose print for debugging\n')
    args = parser.parse_args()

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
    parlistfile_pre = args.ua
    atlas_select_pre = args.a
    parlistfile = list(str(parlistfile_pre).split(','))
    if len(parlistfile) > 1:
        user_atlas_list = parlistfile
        parlistfile = user_atlas_list[0]
    elif parlistfile == ['None']:
        parlistfile = None
        user_atlas_list = None
    else:
        parlistfile = parlistfile[0]
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
        raise ValueError("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format with the -i flag")

    if input_file and dwi_dir and subjects_list:
        raise ValueError("Error: PyNets does not yet support joint functional-structural connectometry across multiple subjects")

    if ID is None and subjects_list is None:
        raise ValueError("Error: You must include a subject ID in your command line call")

    if subjects_list and ',' in ID:
        ID = list(str(ID).split(','))
        if len(ID) != len(subjects_list):
            raise ValueError("Error: Length of ID list does not correspond to length of input file list")

    if conf:
        if ',' in conf:
            conf = list(str(conf).split(','))
            if len(conf) != len(subjects_list):
                raise ValueError("Error: Length of confound regressor list does not correspond to length of input file list")

    if anat_loc is not None and dwi_dir is None:
        raise RuntimeWarning('Warning: anatomical image specified, but not bedpostx directory specified. Anatomical images are only supported for structural connectome estimation at this time.')

    if multi_thr is True:
        thr = None
    else:
        min_thr = None
        max_thr = None
        step_thr = None

    if (k_min is not None and k_max is not None) and k is None and clust_mask_list is not None:
        k_clustering = 4
    elif (k_min is not None and k_max is not None) and k is None and clust_mask_list is None:
        k_clustering = 2
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is not None:
        k_clustering = 3
    elif k is not None and (k_min is None and k_max is None) and clust_mask_list is None:
        k_clustering = 1
    else:
        k_clustering = 0

    if subjects_list:
        print('\nRunning workflow of workflows across multiple subjects:')
    elif subjects_list is None:
        print('\nRunning workflow across single subject:')
    print(str(ID))

    if input_file:
        if parlistfile is not None and k_clustering == 0 and user_atlas_list is None:
            atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
            if subjects_list:
                for input_file in subjects_list:
                    dir_path = do_dir_path(atlas_select_par, input_file)
            else:
                dir_path = do_dir_path(atlas_select_par, input_file)
                print("%s%s" % ("\nUser atlas: ", atlas_select_par))
        elif parlistfile is not None and user_atlas_list is None and k_clustering == 0:
            atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
            if subjects_list:
                for input_file in subjects_list:
                    dir_path = do_dir_path(atlas_select_par, input_file)
            else:
                dir_path = do_dir_path(atlas_select_par, input_file)
            print("%s%s" % ("\nUser atlas: ", atlas_select_par))
        elif user_atlas_list is not None:
            print('\nIterating across multiple user atlases...')
            if subjects_list:
                for input_file in subjects_list:
                    for parlistfile in user_atlas_list:
                        atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
                        print(atlas_select_par)
                        dir_path = do_dir_path(atlas_select_par, input_file)
            else:
                for parlistfile in user_atlas_list:
                    atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
                    print(atlas_select_par)
                    dir_path = do_dir_path(atlas_select_par, input_file)
        elif k_clustering == 1:
            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
            print("%s%s" % ("\nCluster atlas: ", atlas_select_clust))
            print("\nClustering within mask at a single resolution...")
            if subjects_list:
                for input_file in subjects_list:
                    dir_path = do_dir_path(atlas_select_clust, input_file)
            else:
                dir_path = do_dir_path(atlas_select_clust, input_file)
        elif k_clustering == 2:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            print("\nClustering within mask at multiple resolutions...")
            if subjects_list:
                for input_file in subjects_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                        print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                        dir_path = do_dir_path(atlas_select_clust, input_file)
            else:
                for k in k_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                    print("%s%s" % ("Cluster atlas: ", atlas_select_clust))
                    dir_path = do_dir_path(atlas_select_clust, input_file)
        elif k_clustering == 3:
            print("\nClustering within multiple masks at a single resolution...")
            if subjects_list:
                for input_file in subjects_list:
                    for clust_mask in clust_mask_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                        dir_path = do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_mask in clust_mask_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                    dir_path = do_dir_path(atlas_select_clust, input_file)
        elif k_clustering == 4:
            print("\nClustering within multiple masks at multiple resolutions...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)), decimals=0).tolist() + [int(k_max)]
            if subjects_list:
                for input_file in subjects_list:
                    for clust_mask in clust_mask_list:
                        for k in k_list:
                            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                            atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                            dir_path = do_dir_path(atlas_select_clust, input_file)
            else:
                for clust_mask in clust_mask_list:
                    for k in k_list:
                        cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                        atlas_select_clust = "%s%s%s" % (cl_mask_name, '_k', k)
                        dir_path = do_dir_path(atlas_select_clust, input_file)
        elif (user_atlas_list is not None or parlistfile is not None) and (k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1) and atlas_select is None:
            print('Error: the -ua flag cannot be used with the clustering option. Use the -cm flag instead.')
            sys.exit(0)
        if multi_atlas is not None:
            print('\nIterating across multiple nilearn atlases...')
            if subjects_list:
                for input_file in subjects_list:
                    for atlas_select in multi_atlas:
                        print(atlas_select)
                        dir_path = do_dir_path(atlas_select, input_file)
            else:
                for atlas_select in multi_atlas:
                    print(atlas_select)
                    dir_path = do_dir_path(atlas_select, input_file)
        elif atlas_select is not None:
            print("%s%s" % ("\nNilearn atlas: ", atlas_select))
            if subjects_list:
                for input_file in subjects_list:
                    dir_path = do_dir_path(atlas_select, input_file)
            else:
                dir_path = do_dir_path(atlas_select, input_file)
        else:
            if parlistfile is None and k == 0:
                raise KeyError('ERROR: No atlas specified!')
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
                dir_path = do_dir_path(atlas_select, graph)
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
            dir_path = do_dir_path(atlas_select, graph)

    if graph is None and multi_graph is None:
        if network is not None:
            print("%s%s" % ('\nUsing RSN pipeline for: ', network))
        elif multi_nets is not None:
            network = multi_nets[0]
            print("%s%d%s%s%s" % ('\nIterating workflow across ', len(multi_nets), ' networks: ',
                                  str(', '.join(str(n) for n in multi_nets)), '...'))
        else:
            print("\nUsing whole-brain pipeline..." )

        if node_size_list:
            print("%s%s%s" % ('\nGrowing spherical nodes across multiple radius sizes: ',
                              str(', '.join(str(n) for n in node_size_list)), '...'))
        elif parc is True:
            print("\nUsing parcels as nodes")
        else:
            print("%s%s%s" % ("\nUsing node size of: ", node_size, 'mm'))

        if smooth_list:
            print("%s%s%s" % ('\nApplying smoothing to node signal at multiple FWHM mm values: ',
                              str(', '.join(str(n) for n in smooth_list)), '...'))
        elif smooth:
            print("%s%s%s" % ("\npplying smoothing to node signal at: ", smooth, 'FWHM mm'))

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
                    for parlistfile in user_atlas_list:
                        atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
                        print(atlas_select_par)
                        dir_path = do_dir_path(atlas_select_par, nodif_brain_mask_path)
            else:
                for parlistfile in user_atlas_list:
                    atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
                    print(atlas_select_par)
                    dir_path = do_dir_path(atlas_select_par, nodif_brain_mask_path)
        elif parlistfile is not None and user_atlas_list is None:
            atlas_select_par = parlistfile.split('/')[-1].split('.')[0]
            ref_txt = "%s%s" % (parlistfile.split('/')[-1:][0].split('.')[0], '.txt')
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    dir_path = do_dir_path(atlas_select_par, nodif_brain_mask_path)
            else:
                dir_path = do_dir_path(atlas_select_par, nodif_brain_mask_path)
        if multi_atlas is not None:
            print('\nIterating across multiple nilearn atlases...')
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    for atlas_select in multi_atlas:
                        print(atlas_select)
                        dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            else:
                for atlas_select in multi_atlas:
                    print(atlas_select)
                    dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
        elif atlas_select is not None:
            print("%s%s" % ("\nNilearn atlas: ", atlas_select))
            if subjects_list:
                for dwi_dir in subjects_list:
                    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
                    dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            else:
                dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
        else:
            if parlistfile is None:
                raise KeyError('ERROR: No atlas specified!')
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
    print('\n-------------------------------------------------------------------------\n\n\n')

    # print(str(ID))
    # print(str(input_file))
    # print(str(dir_path))
    # print(str(atlas_select))
    # print(str(network))
    # print(str(node_size))
    # print(str(smooth))
    # print(str(mask))
    # print(str(thr))
    # print(str(parlistfile))
    # print(str(multi_nets))
    # print(str(conn_model))
    # print(str(dens_thresh))
    # print(str(conf))
    # print(str(adapt_thresh))
    # print(str(plot_switch))
    # print(str(dwi_dir))
    # print(str(multi_thr))
    # print(str(multi_atlas))
    # print(str(min_thr))
    # print(str(max_thr))
    # print(str(step_thr))
    # print(str(anat_loc))
    # print(str(parc))
    # print(str(ref_txt))
    # print(str(procmem))
    # print(str(k))
    # print(str(clust_mask))
    # print(str(k_min))
    # print(str(k_max))
    # print(str(k_step))
    # print(str(k_clustering))
    # print(str(user_atlas_list))
    # print(str(clust_mask_list))
    # print(str(prune))
    # print(str(node_size_list))
    # print(str(smooth_list))
    # print(str(num_total_samples))
    # print(str(graph))
    # print(str(multi_graph))
    # import sys
    # sys.exit(0)

    # Import core modules
    from pynets.utils import export_to_pandas, collect_pandas_df, collect_pandas_join
    from pynets.netstats import extractnetstats
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface

    def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets,
                          conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir, anat_loc, parc,
                          ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k,
                          clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                          node_size_list, num_total_samples, conn_model_list, min_span_tree, verbose, plugin_type,
                          use_AAL_naming, smooth, smooth_list):
        import os
        from pynets import workflows
        from nipype import Workflow, Function
        from nipype.pipeline import engine as pe
        from nipype.interfaces import utility as niu

        # Workflow 1: Whole-brain functional connectome
        if dwi_dir is None and network is None:
            sub_func_wf = workflows.wb_functional_connectometry(input_file, ID, atlas_select, network, node_size,
                                                                mask, thr, parlistfile, conn_model, dens_thresh, conf,
                                                                plot_switch, parc, ref_txt, procmem, dir_path,
                                                                multi_thr, multi_atlas, max_thr, min_thr, step_thr,
                                                                k, clust_mask, k_min, k_max, k_step, k_clustering,
                                                                user_atlas_list, clust_mask_list, node_size_list,
                                                                conn_model_list, min_span_tree, use_AAL_naming, smooth,
                                                                smooth_list)
            sub_struct_wf = None
        # Workflow 2: RSN functional connectome
        elif dwi_dir is None and network is not None:
            sub_func_wf = workflows.rsn_functional_connectometry(input_file, ID, atlas_select, network, node_size,
                                                                 mask, thr, parlistfile, multi_nets, conn_model,
                                                                 dens_thresh, conf, plot_switch, parc, ref_txt,
                                                                 procmem, dir_path, multi_thr, multi_atlas,
                                                                 max_thr, min_thr, step_thr, k, clust_mask, k_min,
                                                                 k_max, k_step, k_clustering, user_atlas_list,
                                                                 clust_mask_list, node_size_list, conn_model_list,
                                                                 min_span_tree, use_AAL_naming, smooth, smooth_list)
            sub_struct_wf = None
        # Workflow 3: Whole-brain structural connectome
        elif dwi_dir is not None and network is None:
            sub_struct_wf = workflows.wb_structural_connectometry(ID, atlas_select, network, node_size, mask,
                                                                  parlistfile, plot_switch, parc, ref_txt, procmem,
                                                                  dir_path, dwi_dir, anat_loc, thr, dens_thresh,
                                                                  conn_model, user_atlas_list, multi_thr, multi_atlas,
                                                                  max_thr, min_thr, step_thr, node_size_list,
                                                                  num_total_samples, conn_model_list, min_span_tree,
                                                                  use_AAL_naming)
            sub_func_wf = None
        # Workflow 4: RSN structural connectome
        elif dwi_dir is not None and network is not None:
            sub_struct_wf = workflows.rsn_structural_connectometry(ID, atlas_select, network, node_size, mask,
                                                                   parlistfile, plot_switch, parc, ref_txt, procmem,
                                                                   dir_path, dwi_dir, anat_loc, thr, dens_thresh,
                                                                   conn_model, user_atlas_list, multi_thr, multi_atlas,
                                                                   max_thr, min_thr, step_thr, node_size_list,
                                                                   num_total_samples, conn_model_list, min_span_tree,
                                                                   multi_nets, use_AAL_naming)
            sub_func_wf = None

        base_wf = sub_func_wf if sub_func_wf else sub_struct_wf

        # Create meta-workflow to organize graph simulation sets in prep for analysis
        # Credit: @Mathias Goncalves
        base_dirname = "%s%s" % ('Meta_wf_', str(ID))
        meta_wf = Workflow(name=base_dirname)
        meta_wf.add_nodes([base_wf])

        if network is None and input_file:
            if k_clustering > 0:
                meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.clustering_node'))._n_procs = 1
                meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.clustering_node'))._mem_gb = 10
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.WB_fetch_nodes_and_labels_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.WB_fetch_nodes_and_labels_node'))._mem_gb = 2
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.extract_ts_wb_coords_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.extract_ts_wb_coords_node'))._mem_gb = 6
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.get_conn_matrix_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('wb_functional_connectometry_', ID, '.get_conn_matrix_node'))._mem_gb = 4
        elif network and input_file:
            if k_clustering > 0:
                meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.clustering_node'))._n_procs = 1
                meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.clustering_node'))._mem_gb = 10
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.RSN_fetch_nodes_and_labels_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.RSN_fetch_nodes_and_labels_node'))._mem_gb = 2
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.extract_ts_rsn_coords_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.extract_ts_rsn_coords_node'))._mem_gb = 6
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.get_conn_matrix_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('rsn_functional_connectometry_', ID, '.get_conn_matrix_node'))._mem_gb = 4
        elif dwi_dir and network is None:
            meta_wf.get_node("%s%s%s" % ('wb_structural_connectometry_', ID, '.WB_fetch_nodes_and_labels_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('wb_structural_connectometry_', ID, '.WB_fetch_nodes_and_labels_node'))._mem_gb = 2
            meta_wf.get_node("%s%s%s" % ('wb_structural_connectometry_', ID, '.thresh_diff_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('wb_structural_connectometry_', ID, '.thresh_diff_node'))._mem_gb = 1
        elif dwi_dir and network:
            meta_wf.get_node("%s%s%s" % ('rsn_structural_connectometry_', ID, '.RSN_fetch_nodes_and_labels_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('rsn_structural_connectometry_', ID, '.RSN_fetch_nodes_and_labels_node'))._mem_gb = 2
            meta_wf.get_node("%s%s%s" % ('rsn_structural_connectometry_', ID, '.thresh_diff_node'))._n_procs = 1
            meta_wf.get_node("%s%s%s" % ('rsn_structural_connectometry_', ID, '.thresh_diff_node'))._mem_gb = 1
        else:
            raise RuntimeError('ERROR: Either functional input file or dwi directory is missing!')

        if verbose is True:
            from nipype import config, logging
            cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                        'hash_method': 'content'})
            config.update_config(cfg)
            config.update_config({'logging': {'log_to_file': True}})
            logging.update_logging(config)
            config.enable_debug_mode()
            meta_wf.config['logging']['workflow_level'] = 'DEBUG'
            meta_wf.config['logging']['utils_level'] = 'DEBUG'
            meta_wf.config['logging']['interface_level'] = 'DEBUG'
            meta_wf.config['monitoring']['enabled'] = True
            meta_wf.config['monitoring']['sample_frequency'] = '0.5'
            meta_wf.config['monitoring']['summary_append'] = True

        meta_wf.config['execution']['crashfile_format'] = 'txt'
        meta_wf.config['execution']['display_variable'] = ':0'
        meta_wf.config['execution']['job_finished_timeout'] = 65
        meta_wf.config['execution']['stop_on_first_crash'] = False
        plugin_args = {'n_procs': int(procmem[0])-1, 'memory_gb': int(procmem[1])-1}
        egg = meta_wf.run(plugin=plugin_type, plugin_args=plugin_args)
        meta_wf_outputs = {}
        outputs = [n.result.outputs.get() for n in egg.nodes() if n.name == 'outputnode']
        for out in outputs:
            for k, v in out.items():
                if k in meta_wf_outputs:
                    meta_wf_outputs[k].extend(v)
                else:
                    meta_wf_outputs[k] = v
        conn_model_iterlist = meta_wf_outputs['conn_model']
        est_path_iterlist = meta_wf_outputs['est_path']
        network_iterlist = meta_wf_outputs['network']
        node_size_iterlist = meta_wf_outputs['node_size']
        smooth_iterlist = meta_wf_outputs['smooth']
        thr_iterlist = meta_wf_outputs['thr']
        prune_iterlist = [prune] * len(est_path_iterlist)
        ID_iterlist = [str(ID)] * len(est_path_iterlist)
        mask_iterlist = [mask] * len(est_path_iterlist)

        print('\n\nParameters:\n')
        print(conn_model_iterlist)
        print(est_path_iterlist)
        print(network_iterlist)
        print(node_size_iterlist)
        print(smooth_iterlist)
        print(thr_iterlist)
        print(prune_iterlist)
        print(ID_iterlist)
        print(mask_iterlist)
        print('\n\n')

        return thr_iterlist, est_path_iterlist, ID_iterlist, network_iterlist, conn_model_iterlist, mask_iterlist, prune_iterlist, node_size_iterlist, smooth_iterlist

    class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=False)
        thr = traits.Any(mandatory=True)
        conn_model = traits.Str(mandatory=True)
        est_path = File(exists=True, mandatory=True, desc="")
        mask = traits.Any(mandatory=False)
        prune = traits.Any(mandatory=False)
        node_size = traits.Any(mandatory=False)
        smooth = traits.Any(mandatory=False)

    class ExtractNetStatsOutputSpec(TraitedSpec):
        out_file = File()

    class ExtractNetStats(BaseInterface):
        input_spec = ExtractNetStatsInputSpec
        output_spec = ExtractNetStatsOutputSpec

        def _run_interface(self, runtime):
            out = extractnetstats(
                self.inputs.ID,
                self.inputs.network,
                self.inputs.thr,
                self.inputs.conn_model,
                self.inputs.est_path,
                self.inputs.mask,
                self.inputs.prune,
                self.inputs.node_size,
                self.inputs.smooth)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(getattr(self, '_outpath'))}

    class Export2PandasInputSpec(BaseInterfaceInputSpec):
        csv_loc = File(exists=True, mandatory=True, desc="")
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=False)
        mask = traits.Any(mandatory=False)

    class Export2PandasOutputSpec(TraitedSpec):
        net_pickle_mt = traits.Any(mandatory=True)

    class Export2Pandas(BaseInterface):
        input_spec = Export2PandasInputSpec
        output_spec = Export2PandasOutputSpec

        def _run_interface(self, runtime):
            out = export_to_pandas(
                self.inputs.csv_loc,
                self.inputs.ID,
                self.inputs.network,
                self.inputs.mask)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'net_pickle_mt': op.abspath(getattr(self, '_outpath'))}

    class CollectPandasDfsInputSpec(BaseInterfaceInputSpec):
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=True)
        net_pickle_mt_list = traits.List(mandatory=True)
        plot_switch = traits.Any(mandatory=True)
        multi_nets = traits.Any(mandatory=True)

    class CollectPandasDfs(SimpleInterface):
        input_spec = CollectPandasDfsInputSpec

        def _run_interface(self, runtime):
            collect_pandas_df(
                self.inputs.network,
                self.inputs.ID,
                self.inputs.net_pickle_mt_list,
                self.inputs.plot_switch,
                self.inputs.multi_nets)
            return runtime


    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, network, node_size, mask, thr, parlistfile,
                               multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir,
                               multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k,
                               clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune,
                               node_size_list, num_total_samples, graph, conn_model_list, min_span_tree, verbose,
                               plugin_type, use_AAL_naming, multi_graph, smooth, smooth_list):
        wf = pe.Workflow(name='Wf_single_subject_' + str(ID))
        # Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'network', 'thr',
                                                          'node_size', 'mask', 'parlistfile', 'multi_nets',
                                                          'conn_model', 'dens_thresh', 'conf', 'adapt_thresh',
                                                          'plot_switch', 'dwi_dir', 'anat_loc', 'parc', 'ref_txt',
                                                          'procmem', 'dir_path', 'multi_thr', 'multi_atlas', 'max_thr',
                                                          'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max',
                                                          'k_step', 'k_clustering', 'user_atlas_list',
                                                          'clust_mask_list', 'prune', 'node_size_list',
                                                          'num_total_samples', 'graph', 'conn_model_list',
                                                          'min_span_tree', 'verbose', 'plugin_type', 'use_AAL_naming',
                                                          'multi_graph', 'smooth', 'smooth_list']),
                            name='inputnode')

        #2) Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
        inputnode.inputs.in_file = input_file
        inputnode.inputs.ID = ID
        inputnode.inputs.atlas_select = atlas_select
        inputnode.inputs.network = network
        inputnode.inputs.thr = thr
        inputnode.inputs.node_size = node_size
        inputnode.inputs.mask = mask
        inputnode.inputs.parlistfile = parlistfile
        inputnode.inputs.multi_nets = multi_nets
        inputnode.inputs.conn_model = conn_model
        inputnode.inputs.dens_thresh = dens_thresh
        inputnode.inputs.conf = conf
        inputnode.inputs.adapt_thresh = adapt_thresh
        inputnode.inputs.plot_switch = plot_switch
        inputnode.inputs.dwi_dir = dwi_dir
        inputnode.inputs.anat_loc = anat_loc
        inputnode.inputs.parc = parc
        inputnode.inputs.ref_txt = ref_txt
        inputnode.inputs.procmem = procmem
        inputnode.inputs.dir_path = dir_path
        inputnode.inputs.multi_thr = multi_thr
        inputnode.inputs.multi_atlas = multi_atlas
        inputnode.inputs.max_thr = max_thr
        inputnode.inputs.min_thr = min_thr
        inputnode.inputs.step_thr = step_thr
        inputnode.inputs.k = k
        inputnode.inputs.clust_mask = clust_mask
        inputnode.inputs.k_min = k_min
        inputnode.inputs.k_max = k_max
        inputnode.inputs.k_step = k_step
        inputnode.inputs.k_clustering = k_clustering
        inputnode.inputs.user_atlas_list = user_atlas_list
        inputnode.inputs.clust_mask_list = clust_mask_list
        inputnode.inputs.prune = prune
        inputnode.inputs.node_size_list = node_size_list
        inputnode.inputs.num_total_samples = num_total_samples
        inputnode.inputs.graph = graph
        inputnode.inputs.conn_model_list = conn_model_list
        inputnode.inputs.min_span_tree = min_span_tree
        inputnode.inputs.verbose = verbose
        inputnode.inputs.plugin_type = plugin_type
        inputnode.inputs.use_AAL_naming = use_AAL_naming
        inputnode.inputs.multi_graph = multi_graph
        inputnode.inputs.smooth = smooth
        inputnode.inputs.smooth_list = smooth_list

        #3) Add variable to function nodes
        # Create function nodes
        imp_est = pe.Node(niu.Function(input_names=['input_file', 'ID', 'atlas_select', 'network', 'node_size', 'mask',
                                                    'thr', 'parlistfile', 'multi_nets', 'conn_model', 'dens_thresh',
                                                    'conf', 'adapt_thresh', 'plot_switch', 'dwi_dir', 'anat_loc',
                                                    'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr',
                                                    'multi_atlas', 'max_thr', 'min_thr', 'step_thr', 'k', 'clust_mask',
                                                    'k_min', 'k_max', 'k_step', 'k_clustering', 'user_atlas_list',
                                                    'clust_mask_list', 'prune', 'node_size_list', 'num_total_samples',
                                                    'conn_model_list', 'min_span_tree', 'verbose', 'plugin_type',
                                                    'use_AAL_naming', 'smooth', 'smooth_list'],
                                       output_names=['thr_iterlist', 'est_path_iterlist', 'ID_iterlist',
                                                     'network_iterlist', 'conn_model_iterlist', 'mask_iterlist',
                                                     'prune_iterlist', 'node_size_iterlist', 'smooth_iterlist'],
                                       function=workflow_selector), name="imp_est")
        imp_est._mem_gb = procmem[1]
        imp_est.n_procs = procmem[0]

        # Create MapNode types for net_mets_node and export_to_pandas_node
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name="ExtractNetStats",
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path',
                                              'mask', 'prune', 'node_size', 'smooth'])

        export_to_pandas_node = pe.MapNode(interface=Export2Pandas(), name="export_to_pandas",
                                           iterfield=['csv_loc', 'ID', 'network', 'mask'])

        collect_pd_list_net_pickles = pe.Node(niu.Function(input_names='net_pickle_mt',
                                                           output_names='net_pickle_mt_out',
                                                           function=collect_pandas_join),
                                              name='collect_pd_list_net_pickles')

        collect_pandas_dfs_node = pe.Node(interface=CollectPandasDfs(), name="CollectPandasDfs",
                                          input_names=['network', 'ID', 'net_pickle_mt_list', 'plot_switch',
                                                       'multi_nets'])

        # Connect nodes of workflow
        wf.connect([
            (inputnode, imp_est, [('in_file', 'input_file'),
                                  ('ID', 'ID'),
                                  ('atlas_select', 'atlas_select'),
                                  ('network', 'network'),
                                  ('node_size', 'node_size'),
                                  ('mask', 'mask'),
                                  ('thr', 'thr'),
                                  ('parlistfile', 'parlistfile'),
                                  ('multi_nets', 'multi_nets'),
                                  ('conn_model', 'conn_model'),
                                  ('dens_thresh', 'dens_thresh'),
                                  ('conf', 'conf'),
                                  ('adapt_thresh', 'adapt_thresh'),
                                  ('plot_switch', 'plot_switch'),
                                  ('dwi_dir', 'dwi_dir'),
                                  ('anat_loc', 'anat_loc'),
                                  ('parc', 'parc'),
                                  ('ref_txt', 'ref_txt'),
                                  ('procmem', 'procmem'),
                                  ('dir_path', 'dir_path'),
                                  ('multi_thr', 'multi_thr'),
                                  ('multi_atlas', 'multi_atlas'),
                                  ('max_thr', 'max_thr'),
                                  ('min_thr', 'min_thr'),
                                  ('step_thr', 'step_thr'),
                                  ('k', 'k'),
                                  ('clust_mask', 'clust_mask'),
                                  ('k_min', 'k_min'),
                                  ('k_max', 'k_max'),
                                  ('k_step', 'k_step'),
                                  ('k_clustering', 'k_clustering'),
                                  ('user_atlas_list', 'user_atlas_list'),
                                  ('clust_mask_list', 'clust_mask_list'),
                                  ('prune', 'prune'),
                                  ('node_size_list', 'node_size_list'),
                                  ('num_total_samples', 'num_total_samples'),
                                  ('conn_model_list', 'conn_model_list'),
                                  ('min_span_tree', 'min_span_tree'),
                                  ('verbose', 'verbose'),
                                  ('plugin_type', 'plugin_type'),
                                  ('use_AAL_naming', 'use_AAL_naming'),
                                  ('smooth', 'smooth'),
                                  ('smooth_list', 'smooth_list')]),
            (imp_est, net_mets_node, [('est_path_iterlist', 'est_path'),
                                      ('network_iterlist', 'network'),
                                      ('thr_iterlist', 'thr'),
                                      ('ID_iterlist', 'ID'),
                                      ('conn_model_iterlist', 'conn_model'),
                                      ('mask_iterlist', 'mask'),
                                      ('prune_iterlist', 'prune'),
                                      ('node_size_iterlist', 'node_size'),
                                      ('smooth_iterlist', 'smooth')]),
            (imp_est, export_to_pandas_node, [('network_iterlist', 'network'),
                                              ('ID_iterlist', 'ID'),
                                              ('mask_iterlist', 'mask')]),
            (net_mets_node, export_to_pandas_node, [('out_file', 'csv_loc')]),
            (inputnode, collect_pandas_dfs_node, [('network', 'network'),
                                                  ('ID', 'ID'),
                                                  ('plot_switch', 'plot_switch'),
                                                  ('multi_nets', 'multi_nets')]),
            (export_to_pandas_node, collect_pd_list_net_pickles, [('net_pickle_mt', 'net_pickle_mt')]),
            (collect_pd_list_net_pickles, collect_pandas_dfs_node, [('net_pickle_mt_out', 'net_pickle_mt_list')])
        ])
        if graph or multi_graph:
            wf.disconnect([
                (inputnode, imp_est, [('in_file', 'input_file'),
                                      ('ID', 'ID'),
                                      ('atlas_select', 'atlas_select'),
                                      ('network', 'network'),
                                      ('node_size', 'node_size'),
                                      ('mask', 'mask'),
                                      ('thr', 'thr'),
                                      ('parlistfile', 'parlistfile'),
                                      ('multi_nets', 'multi_nets'),
                                      ('conn_model', 'conn_model'),
                                      ('dens_thresh', 'dens_thresh'),
                                      ('conf', 'conf'),
                                      ('adapt_thresh', 'adapt_thresh'),
                                      ('plot_switch', 'plot_switch'),
                                      ('dwi_dir', 'dwi_dir'),
                                      ('anat_loc', 'anat_loc'),
                                      ('parc', 'parc'),
                                      ('ref_txt', 'ref_txt'),
                                      ('procmem', 'procmem'),
                                      ('dir_path', 'dir_path'),
                                      ('multi_thr', 'multi_thr'),
                                      ('multi_atlas', 'multi_atlas'),
                                      ('max_thr', 'max_thr'),
                                      ('min_thr', 'min_thr'),
                                      ('step_thr', 'step_thr'),
                                      ('k', 'k'),
                                      ('clust_mask', 'clust_mask'),
                                      ('k_min', 'k_min'),
                                      ('k_max', 'k_max'),
                                      ('k_step', 'k_step'),
                                      ('k_clustering', 'k_clustering'),
                                      ('user_atlas_list', 'user_atlas_list'),
                                      ('clust_mask_list', 'clust_mask_list'),
                                      ('prune', 'prune'),
                                      ('node_size_list', 'node_size_list'),
                                      ('num_total_samples', 'num_total_samples'),
                                      ('conn_model_list', 'conn_model_list'),
                                      ('min_span_tree', 'min_span_tree'),
                                      ('verbose', 'verbose'),
                                      ('plugin_type', 'plugin_type'),
                                      ('use_AAL_naming', 'use_AAL_naming'),
                                      ('smooth', 'smooth'),
                                      ('smooth_list', 'smooth_list')])
                            ])
            wf.disconnect([(imp_est, net_mets_node, [('est_path_iterlist', 'est_path'),
                                                     ('network_iterlist', 'network'),
                                                     ('thr_iterlist', 'thr'),
                                                     ('ID_iterlist', 'ID'),
                                                     ('conn_model_iterlist', 'conn_model'),
                                                     ('mask_iterlist', 'mask'),
                                                     ('prune_iterlist', 'prune'),
                                                     ('node_size_iterlist', 'node_size'),
                                                     ('smooth_iterlist', 'smooth')])
                           ])
            wf.disconnect([(imp_est, export_to_pandas_node, [('network_iterlist', 'network'),
                                                             ('ID_iterlist', 'ID'),
                                                             ('mask_iterlist', 'mask')])
                           ])
            wf.remove_nodes([imp_est])
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
                                                        ('smooth', 'smooth')])
                            ])
                wf.connect([(inputnode, net_mets_node, [('graph', 'est_path')])])
                wf.connect([(inputnode, export_to_pandas_node, [('network', 'network'),
                                                                ('ID', 'ID'),
                                                                ('mask', 'mask')])
                            ])

        return wf

    def wf_multi_subject(ID, subjects_list, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets,
                         conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir, multi_thr,
                         multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask,
                         k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list,
                         num_total_samples, graph, conn_model_list, min_span_tree, verbose, plugin_type, use_AAL_naming,
                         multi_graph, smooth, smooth_list):

        wf_multi = pe.Workflow(name='PyNets_multisubject')
        procmem_cores = int(np.round(float(procmem[0])/float(len(subjects_list)), 0))
        procmem_ram = int(np.round(float(procmem[1]) / float(len(subjects_list)), 0))
        procmem_indiv = [procmem_cores, procmem_ram]
        i = 0
        for _file in subjects_list:
            if conf:
                conf_sub = conf[i]
            else:
                conf_sub = None
            wf_single_subject = init_wf_single_subject(
                ID=ID[i], input_file=_file,
                dir_path=os.path.dirname(os.path.realpath(subjects_list[i])), atlas_select=atlas_select,
                network=network, node_size=node_size, mask=mask, thr=thr, parlistfile=parlistfile,
                multi_nets=multi_nets, conn_model=conn_model, dens_thresh=dens_thresh, conf=conf_sub,
                adapt_thresh=adapt_thresh, plot_switch=plot_switch, dwi_dir=dwi_dir, multi_thr=multi_thr,
                multi_atlas= multi_atlas, min_thr=min_thr, max_thr=max_thr, step_thr=step_thr, anat_loc=anat_loc,
                parc=parc, ref_txt=ref_txt, procmem=procmem_indiv, k=k, clust_mask=clust_mask, k_min=k_min, k_max=k_max,
                k_step=k_step, k_clustering=k_clustering, user_atlas_list=user_atlas_list,
                clust_mask_list=clust_mask_list, prune=prune, node_size_list=node_size_list,
                num_total_samples=num_total_samples, graph=graph, conn_model_list=conn_model_list,
                min_span_tree=min_span_tree, verbose=verbose, plugin_type=plugin_type, use_AAL_naming=use_AAL_naming,
                multi_graph=multi_graph, smooth=smooth, smooth_list=smooth_list)
            wf_multi.add_nodes([wf_single_subject])
            i = i + 1

        return wf_multi

    # Workflow generation
    # Multi-subject workflow generator

    if subjects_list:
        wf_multi = wf_multi_subject(ID, subjects_list, atlas_select, network, node_size, mask,
                                    thr, parlistfile, multi_nets, conn_model, dens_thresh,
                                    conf, adapt_thresh, plot_switch, dwi_dir, multi_thr,
                                    multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc,
                                    ref_txt, procmem, k, clust_mask, k_min, k_max, k_step,
                                    k_clustering, user_atlas_list, clust_mask_list, prune,
                                    node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph,
                                    smooth, smooth_list)

        import shutil
        if os.path.exists('/tmp/Wf_multi_subject'):
            shutil.rmtree('/tmp/Wf_multi_subject')
        os.mkdir('/tmp/Wf_multi_subject')
        wf_multi.base_dir = '/tmp/Wf_multi_subject'

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
        wf_multi.config['execution']['job_finished_timeout'] = 65
        wf_multi.config['execution']['stop_on_first_crash'] = False
        plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1])}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf_multi.run(plugin=plugin_type, plugin_args=plugin_args)
        #wf_multi.run()
    # Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, network, node_size, mask, thr, parlistfile,
                                    multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, dwi_dir,
                                    multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc, ref_txt,
                                    procmem, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list,
                                    clust_mask_list, prune, node_size_list, num_total_samples, graph, conn_model_list,
                                    min_span_tree, verbose, plugin_type, use_AAL_naming, multi_graph, smooth,
                                    smooth_list)

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
        else:
            if os.path.exists("%s%s%s" % (dir_path, '/', base_dirname)):
                shutil.rmtree("%s%s%s" % (dir_path, '/', base_dirname))
            os.mkdir("%s%s%s" % (dir_path, '/', base_dirname))
            wf.base_dir = dir_path

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
        #wf.write_graph(graph2use='flat', format='png', dotfilename='indiv_wf.dot')
        plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1])}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf.run(plugin=plugin_type, plugin_args=plugin_args)
        #wf.run(plugin='Linear')

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')
