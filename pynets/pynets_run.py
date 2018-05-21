#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Tue Nov  7 10:40:07 2017
# Copyright (C) 2018
# @author: Derek Pisner

import sys
import argparse
import os
import timeit
import warnings
warnings.simplefilter("ignore")
import numpy as np
try:
    from pynets.utils import do_dir_path
except ImportError:
    print('PyNets not installed! Ensure that you are using the correct python version.')

####Parse arguments####
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
        metavar='Path to input file',
        default=None,
        required=False,
        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR the path to a text file containing a list of paths to subject files.\n')
    parser.add_argument('-id',
        metavar='Subject ID',
        default=None,
        required=False,
        help='A subject ID that is also the name of the directory containing the input file.\n')
    parser.add_argument('-a',
        metavar='Atlas',
        default='coords_power_2011',
        help='Specify a coordinate atlas parcellation of those available in nilearn. Default is coords_power_2011. If you wish to iterate your pynets run over multiple nilearn atlases, separate them by comma. e.g. -a \'atlas_aal,atlas_destrieux_2009\' Available nilearn atlases are:\n\natlas_aal \natlas_destrieux_2009 \ncoords_dosenbach_2010 \ncoords_power_2011.\n')
    #parser.add_argument('-basc',
        #default=False,
        #action='store_true',
        #help='Specify whether you want to run BASC to calculate a group level set of nodes')
    parser.add_argument('-ua',
        metavar='Path to parcellation file',
        default=None,
        help='Path to parcellation/atlas file in .nii format. If specifying a list of paths to multiple user atlases, separate them by comma.\n')
    parser.add_argument('-pm',
        metavar='Cores,memory',
        default= '2,4',
        help='Number of cores to use, number of GB of memory to use for single subject run, entered as two integers seperated by a comma.\n')
    parser.add_argument('-n',
        metavar='Resting-state network',
        default=None,
        help='Optionally specify the name of any of the 2017 Yeo-Schaefer RSNs (7-network or 17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar. If listing multiple RSNs, separate them by comma. e.g. -n \'Default,Cont,SalVentAttn\'\n')
    parser.add_argument('-thr',
        metavar='graph threshold',
        default='1.00',
        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is no thresholding.\n')
    parser.add_argument('-ns',
        metavar='Node size',
        default=4,
        help='Optionally coordinate-based node radius size(s). Default is 4 mm. If you wish to iterate the pipeline across multiple node sizes, separate the list by comma (e.g. 2,4,6)\n')
    parser.add_argument('-m',
        metavar='Path to mask image',
        default=None,
        help='Optionally specify a thresholded binarized mask image (such as an ICA-derived mask) and retain only those nodes contained within that mask.\n')
    parser.add_argument('-mod',
        metavar='Graph estimator type',
        default='sps',
        help='Optionally specify matrix estimation type: corr, cov, sps, partcorr for correlation, covariance, sparse-inverse covariance, and partial correlation, respectively. sps type is used by default.\n')
    parser.add_argument('-conf',
        metavar='Confounds',
        default=None,
        help='Optionally specify a path to a confound regressor file to reduce noise in the time-series estimation for the graph.\n')
    parser.add_argument('-dt',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to threshold to achieve a given density or densities indicated by the -thr and -min_thr, -max_thr, -step_thr flags, respectively.\n')
#    parser.add_argument('-at',
#        default=False,
#        action='store_true',
#        help='Optionally use this flag if you wish to activate adaptive thresholding')
    parser.add_argument('-plt',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting of adjacency matrices, connectomes, and time-series.\n')
    parser.add_argument('-bpx',
        metavar='Path to bedpostx directory',
        default=None,
        help='Formatted according to the FSL default tree structure found at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX.\n')
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
        help='Maximum threshold for multi-thresholding.v')
    parser.add_argument('-step_thr',
        metavar='Multi-thresholding step size',
        default=None,
        help='Threshold step value for multi-thresholding. Default is 0.01.\n')
    parser.add_argument('-parc',
        default=False,
        action='store_true',
        help='Include this flag to use parcels instead of coordinates as nodes.\n')
    parser.add_argument('-ref',
        metavar='atlas reference file path',
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
        default=False,
        action='store_true',
        help='Include this flag to prune the resulting graph of any fully disconnected nodes.\n')
    parser.add_argument('-s',
        metavar='Number of samples',
        default= '500',
        help='Include this flag to manually specify number of fiber samples for probtrackx2 in structural connectome estimation (default is 500). PyNets parallelizes probtrackx2 by samples, but more samples can increase connectome estimation time considerably.\n')
    args = parser.parse_args()

    ##Start time clock
    start_time = timeit.default_timer()

    ###Set Arguments to global variables###
    input_file=args.i
    ID=args.id
    #basc=args.basc
    procmem=list(eval(str((args.pm))))
    thr=float(args.thr)
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
    mask=args.m
    conn_model=args.mod
    conf=args.conf
    dens_thresh=args.dt
#    adapt_thresh=args.at
    adapt_thresh=False
    plot_switch=args.plt
    bedpostx_dir=args.bpx
    min_thr=args.min_thr
    max_thr=args.max_thr
    step_thr=args.step_thr
    anat_loc=args.anat
    num_total_samples = args.s
    parc=args.parc
    ref_txt=args.ref
    k=args.k
    k_min=args.k_min
    k_max=args.k_max
    k_step=args.k_step
    clust_mask_pre = args.cm
    prune = args.p
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
    atlas_select_pre = args.a
    atlas_select = list(str(atlas_select_pre).split(','))
    if len(atlas_select) > 1:
        multi_atlas = atlas_select
        atlas_select = None
    elif atlas_select == ['None']:
        atlas_select = None
        multi_atlas = None
    else:
        atlas_select = atlas_select[0]
        multi_atlas = None
    print('\n\n\n------------------------------------------------------------------------\n')

    if min_thr is not None and max_thr is not None and step_thr is not None:
        multi_thr=True
    elif min_thr is not None or max_thr is not None or step_thr is not None:
        print('Error: Missing either min_thr, max_thr, or step_thr flags!')
        sys.exit(0)
    else:
        multi_thr=False

    ##Check required inputs for existence, and configure run
    if input_file:
        if input_file.endswith('.txt'):
            with open(input_file) as f:
                subjects_list = f.read().splitlines()
        else:
            subjects_list = None
    else:
        subjects_list = None

    if input_file is None and bedpostx_dir is None:
        print("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format with the -i flag")
        sys.exit()

    if ID is None and subjects_list is None:
        print("Error: You must include a subject ID in your command line call")
        sys.exit()

    if anat_loc is not None and bedpostx_dir is None:
        print('Warning: anatomical image specified, but not bedpostx directory specified. Anatomical images are only supported for structural connectome estimation at this time.')

    if multi_thr is True:
        adapt_thresh = False
    elif multi_thr is False and min_thr is not None and max_thr is not None:
        multi_thr = True
        adapt_thresh = False
    else:
        min_thr = None
        max_thr = None
        step_thr = None

    print("%s%s" % ('SUBJECT ID: ', ID))

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

    if input_file:
        if parlistfile is not None and k_clustering == 0 and user_atlas_list is None:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            dir_path = do_dir_path(atlas_select, input_file)
            print("%s%s" % ("ATLAS: ", atlas_select))
        elif parlistfile is not None and user_atlas_list is None and k_clustering == 0:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            dir_path = do_dir_path(atlas_select, input_file)
            print("%s%s" % ("ATLAS: ", atlas_select))
        elif user_atlas_list is not None:
            parlistfile = user_atlas_list[0]
            print('Iterating across multiple user atlases...')
            for parlistfile in user_atlas_list:
                atlas_select = parlistfile.split('/')[-1].split('.')[0]
                dir_path = do_dir_path(atlas_select, input_file)
            atlas_select = None
        elif multi_atlas is not None:
            print('Iterating across multiple nilearn atlases...')
            for atlas_select in multi_atlas:
                dir_path = do_dir_path(atlas_select, input_file)
            atlas_select = None
        elif k_clustering == 1:
            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
            atlas_select = "%s%s%s" % (cl_mask_name, '_k', k)
            print("%s%s" % ("ATLAS: ", atlas_select))
            dir_path = do_dir_path(atlas_select, input_file)
            print("Clustering within mask at a single resolution...")
        elif k_clustering == 2:
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            print("Clustering within mask at multiple resolutions...")
            for k in k_list:
                cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                atlas_select = "%s%s%s" % (cl_mask_name, '_k', k)
                print("%s%s" % ("ATLAS: ", atlas_select))
                dir_path = do_dir_path(atlas_select, input_file)
        elif k_clustering == 3:
            print("Clustering within multiple masks at a single resolution...")
            for clust_mask in clust_mask_list:
                cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                atlas_select = "%s%s%s" % (cl_mask_name, '_k', k)
                dir_path = do_dir_path(atlas_select, input_file)
        elif k_clustering == 4:
            print("Clustering within multiple masks at multiple resolutions...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            for clust_mask in clust_mask_list:
                for k in k_list:
                    cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                    atlas_select = "%s%s%s" % (cl_mask_name, '_k', k)
                    dir_path = do_dir_path(atlas_select, input_file)
        elif (user_atlas_list is not None or parlistfile is not None) and (k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1) and atlas_select is None:
            print('Error: the -ua flag cannot be used with the clustering option. Use the -cm flag instead.')
            sys.exit(0)
        else:
            dir_path = do_dir_path(atlas_select, input_file)

    if network is not None:
        print("%s%s" % ('Using RSN pipeline for: ', network))
    elif multi_nets is not None:
        network = multi_nets[0]
        print("%s%d%s%s%s" % ('Iterating workflow across ', len(multi_nets), ' networks: ', str(', '.join(str(n) for n in multi_nets)), '...'))
    else:
        print("Using whole-brain pipeline..." )

    if node_size_list:
        print("%s%s%s" % ('Growing spherical nodes across multiple radius sizes: ', str(', '.join(str(n) for n in node_size_list)), '...'))
    elif parc is True:
        print("Using parcels as nodes")
    else:
        print("%s%s" % ("Using node size of: ", node_size))

    if input_file and subjects_list:
        print('\nRunning workflow of workflows across subjects:\n')
        print(str(subjects_list))
        ##Set directory path containing input file
        dir_path = do_dir_path(atlas_select, subjects_list[0])
    elif input_file and bedpostx_dir:
        print('Running joint structural-functional connectometry...')
        print("%s%s" % ('Functional file: ', input_file))
        print("%s%s" % ('Bedpostx Directory: ', bedpostx_dir))
        if anat_loc is not None:
            print("%s%s" % ('Anatomical Image: ', anat_loc))
        if network is not None:
            print("%s%s" % ('RSN: ', network))
        ##Set directory path containing input file
        nodif_brain_mask_path = "%s%s" % (bedpostx_dir, '/nodif_brain_mask.nii.gz')
        if user_atlas_list is not None:
            parlistfile = user_atlas_list[0]
            print('Iterating across multiple user atlases...')
            for parlistfile in user_atlas_list:
                atlas_select = parlistfile.split('/')[-1].split('.')[0]
                dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = None
        elif multi_atlas is not None:
            print('Iterating across multiple nilearn atlases...')
            for atlas_select in multi_atlas:
                dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = None
        else:
            dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            ref_txt = parlistfile.split('/')[-1:][0].split('.')[0] + '.txt'
        conn_model = 'prob'
    elif input_file is None and bedpostx_dir:
        print('Running structural connectometry only...')
        print("%s%s" % ('Bedpostx Directory: ', bedpostx_dir))
        print("%s%s" % ('Number of fiber samples for tracking: ', num_total_samples))
        if anat_loc is not None:
            print("%s%s" % ('Anatomical Image: ', anat_loc))
        if network is not None:
            print("%s%s" % ('RSN: ', network))
        ##Set directory path containing input file
        nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
        input_file = nodif_brain_mask_path
        if user_atlas_list is not None:
            parlistfile = user_atlas_list[0]
            print('Iterating across multiple user atlases...')
            for parlistfile in user_atlas_list:
                atlas_select = parlistfile.split('/')[-1].split('.')[0]
                dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = None
        elif multi_atlas is not None:
            print('Iterating across multiple nilearn atlases...')
            for atlas_select in multi_atlas:
                dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = None
        else:
            dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            ref_txt = parlistfile.split('/')[-1:][0].split('.')[0] + '.txt'
        conn_model = 'prob'
    elif input_file and bedpostx_dir is None and subjects_list is None:
        print('Running functional connectometry only...')
        print("%s%s" % ('Functional file: ', input_file))
    print('\n-------------------------------------------------------------------------\n\n\n')

    ##Import core modules
    from pynets.utils import export_to_pandas, collect_pandas_df
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

    #if basc == True:
       #from pynets import basc_run
       #from pathlib import Path
       #basc_config=Path(__file__).parent/'basc_config.yaml'

       #print("\n\n\n-------------() > STARTING BASC < ()----------------------" + "\n\n\n")

       #basc_run.basc_run(subjects_list, basc_config)
       #parlistfile=Path(__file__)/'pynets'/'rsnrefs'/'group_stability_clusters.nii.gz'

    def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples):
        from pynets import workflows, utils
        from nipype import Node, Workflow, Function

        ##Workflow 1: Whole-brain functional connectome
        if bedpostx_dir is None and network is None:
            sub_func_wf = workflows.wb_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, node_size_list)
            sub_struct_wf = None
        ##Workflow 2: RSN functional connectome
        elif bedpostx_dir is None and network is not None:
            sub_func_wf = workflows.rsn_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, node_size_list)
            sub_struct_wf = None
        ##Workflow 3: Whole-brain structural connectome
        elif bedpostx_dir is not None and network is None:
            sub_struct_wf = workflows.wb_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, anat_loc, thr, dens_thresh, conn_model, user_atlas_list, multi_thr, multi_atlas, max_thr, min_thr, step_thr, node_size_list, num_total_samples)
            sub_func_wf = None
        ##Workflow 4: RSN structural connectome
        elif bedpostx_dir is not None and network is not None:
            sub_struct_wf = workflows.rsn_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, anat_loc, thr, dens_thresh, conn_model, user_atlas_list, multi_thr, multi_atlas, max_thr, min_thr, step_thr, node_size_list, num_total_samples)
            sub_func_wf = None

        base_wf = sub_func_wf if sub_func_wf else sub_struct_wf

        ##Create meta-workflow to organize graph simulation sets in prep for analysis
        ##Credit: @Mathias Goncalves
        meta_wf = Workflow(name='meta')
        meta_wf.add_nodes([base_wf])

        import_list = ['import sys',
                       'import os',
                       'import nibabel as nib'
                       'import numpy as np',
                       'from pynets import utils']

        comp_iter = Node(Function(function=utils.compile_iterfields,
                                  input_names=['input_file', 'ID', 'atlas_select',
                                                 'network', 'node_size', 'mask', 'thr',
                                                 'parlistfile', 'multi_nets', 'conn_model',
                                                 'dens_thresh', 'dir_path', 'multi_thr',
                                                 'multi_atlas', 'max_thr', 'min_thr', 'step_thr',
                                                 'k', 'clust_mask', 'k_min', 'k_max', 'k_step',
                                                 'k_clustering', 'user_atlas_list', 'clust_mask_list',
                                                 'prune', 'node_size_list', 'est_path'],
                                  output_names = ['est_path', 'thr', 'network', 'ID', 'mask', 'conn_model',
                                                  'k_clustering', 'prune', 'node_size']),
                         name='compile_iterfields', imports=import_list)

        comp_iter.inputs.input_file = input_file
        comp_iter.inputs.ID = ID
        comp_iter.inputs.atlas_select = atlas_select
        comp_iter.inputs.mask = mask
        comp_iter.inputs.parlistfile = parlistfile
        comp_iter.inputs.multi_nets = multi_nets
        comp_iter.inputs.conn_model = conn_model
        comp_iter.inputs.dens_thresh = dens_thresh
        comp_iter.inputs.multi_thr = multi_thr
        comp_iter.inputs.multi_atlas = multi_atlas
        comp_iter.inputs.max_thr = max_thr
        comp_iter.inputs.min_thr = min_thr
        comp_iter.inputs.step_thr = step_thr
        comp_iter.inputs.k = k
        comp_iter.inputs.clust_mask = clust_mask
        comp_iter.inputs.k_min = k_min
        comp_iter.inputs.k_max = k_max
        comp_iter.inputs.k_step = k_step
        comp_iter.inputs.k_clustering = k_clustering
        comp_iter.inputs.user_atlas_list = user_atlas_list
        comp_iter.inputs.clust_mask_list = clust_mask_list
        comp_iter.inputs.prune = prune
        comp_iter.inputs.node_size_list = node_size_list

        meta_wf.connect(base_wf, "outputnode.est_path", comp_iter, "est_path")
        meta_wf.connect(base_wf, "outputnode.thr", comp_iter, "thr")
        meta_wf.connect(base_wf, "outputnode.network", comp_iter, "network")
        meta_wf.connect(base_wf, "outputnode.node_size", comp_iter, "node_size")
        meta_wf.connect(base_wf, "outputnode.dir_path", comp_iter, "dir_path")
        meta_wf.config['logging']['log_directory']='/tmp'
        meta_wf.config['logging']['workflow_level']='DEBUG'
        meta_wf.config['logging']['utils_level']='DEBUG'
        meta_wf.config['logging']['interface_level']='DEBUG'
        #meta_wf.write_graph(graph2use='exec', format='png', dotfilename='meta_wf.dot')
        egg = meta_wf.run('MultiProc')
        outputs = [x for x in egg.nodes() if x.name == 'compile_iterfields'][0].result.outputs

        return outputs.thr, outputs.est_path, outputs.ID, outputs.network, outputs.conn_model, outputs.mask, outputs.prune, outputs.node_size

    class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=False)
        thr = traits.Any(mandatory=True)
        conn_model = traits.Str(mandatory=True)
        est_path = File(exists=True, mandatory=True, desc="")
        mask = traits.Any(mandatory=False)
        prune = traits.Any(mandatory=False)
        node_size = traits.Any(mandatory=False)

    class ExtractNetStatsOutputSpec(TraitedSpec):
        out_file = File()

    class ExtractNetStats(BaseInterface):
        input_spec = ExtractNetStatsInputSpec
        output_spec = ExtractNetStatsOutputSpec

        def _run_interface(self, runtime):
            from pynets.netstats import extractnetstats
            out = extractnetstats(
                self.inputs.ID,
                self.inputs.network,
                self.inputs.thr,
                self.inputs.conn_model,
                self.inputs.est_path,
                self.inputs.mask,
                self.inputs.prune,
                self.inputs.node_size)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(getattr(self, '_outpath'))}

    class Export2PandasInputSpec(BaseInterfaceInputSpec):
        in_csv = File(exists=True, mandatory=True, desc="")
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=False)
        mask = traits.Any(mandatory=False)
        out_file = File('output_export2pandas.csv', usedefault=True)

    class Export2PandasOutputSpec(TraitedSpec):
        out_file = File()

    class Export2Pandas(BaseInterface):
        input_spec = Export2PandasInputSpec
        output_spec = Export2PandasOutputSpec

        def _run_interface(self, runtime):
            export_to_pandas(
                self.inputs.in_csv,
                self.inputs.ID,
                self.inputs.network,
                self.inputs.mask,
                out_file=self.inputs.out_file)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(self.inputs.out_file)}

    class CollectPandasDfsInputSpec(BaseInterfaceInputSpec):
        input_file = traits.Any(mandatory=True)
        atlas_select = traits.Any(mandatory=True)
        clust_mask = traits.Any(mandatory=True)
        k_min = traits.Any(mandatory=True)
        k_max = traits.Any(mandatory=True)
        k = traits.Any(mandatory=True)
        k_step = traits.Any(mandatory=True)
        min_thr = traits.Any(mandatory=True)
        max_thr = traits.Any(mandatory=True)
        step_thr = traits.Any(mandatory=True)
        multi_thr = traits.Any(mandatory=True)
        thr = traits.Any(mandatory=True)
        mask = traits.Any(mandatory=True)
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=True)
        k_clustering = traits.Any(mandatory=True)
        conn_model = traits.Any(mandatory=True)
        in_csv = traits.Any(mandatory=True)
        user_atlas_list = traits.Any(mandatory=True)
        clust_mask_list = traits.Any(mandatory=True)
        multi_atlas = traits.Any(mandatory=True)
        node_size = traits.Any(mandatory=True)
        node_size_list = traits.Any(mandatory=True)
        parc = traits.Any(mandatory=True)
        out_file = File('output_collectpandasdf.csv', usedefault=True)

    class CollectPandasDfsOutputSpec(TraitedSpec):
        out_file = File()

    class CollectPandasDfs(BaseInterface):
        input_spec = CollectPandasDfsInputSpec
        output_spec = CollectPandasDfsOutputSpec

        def _run_interface(self, runtime):
            collect_pandas_df(
            self.inputs.input_file,
            self.inputs.atlas_select,
            self.inputs.clust_mask,
            self.inputs.k_min,
            self.inputs.k_max,
            self.inputs.k,
            self.inputs.k_step,
            self.inputs.min_thr,
            self.inputs.max_thr,
            self.inputs.step_thr,
            self.inputs.multi_thr,
            self.inputs.thr,
            self.inputs.mask,
            self.inputs.ID,
            self.inputs.network,
            self.inputs.k_clustering,
            self.inputs.conn_model,
            self.inputs.in_csv,
            self.inputs.user_atlas_list,
            self.inputs.clust_mask_list,
            self.inputs.multi_atlas,
            self.inputs.node_size,
            self.inputs.node_size_list,
            self.inputs.parc,
            out_file=self.inputs.out_file)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(self.inputs.out_file)}

    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
    node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
    adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
    max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min,
    k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples):
        wf = pe.Workflow(name='PyNets_' + str(ID))
        wf.base_directory='/tmp/pynets'
        ##Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID',
        'atlas_select', 'network', 'thr', 'node_size', 'mask', 'parlistfile',
        'multi_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch',
        'bedpostx_dir', 'anat_loc', 'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr',
        'multi_atlas', 'max_thr', 'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min',
        'k_max', 'k_step', 'k_clustering', 'user_atlas_list', 'clust_mask_list', 'prune',
        'node_size_list', 'num_total_samples']), name='inputnode')

        #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
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
        inputnode.inputs.bedpostx_dir = bedpostx_dir
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

        #3) Add variable to function nodes
        ##Create function nodes
        imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select',
        'network', 'node_size', 'mask', 'thr', 'parlistfile', 'multi_nets', 'conn_model',
        'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch', 'bedpostx_dir', 'anat_loc',
        'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr', 'multi_atlas', 'max_thr',
        'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max', 'k_step', 'k_clustering',
        'user_atlas_list', 'clust_mask_list', 'prune', 'node_size_list', 'num_total_samples'],
        output_names = ['outputs.thr', 'outputs.est_path', 'outputs.ID', 'outputs.network',
                        'outputs.conn_model', 'outputs.mask', 'outputs.prune', 'outputs.node_size'],
        function=workflow_selector),
        name = "imp_est")

        ##Create MapNode types for net_mets_node and export_to_pandas_node
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name = "ExtractNetStats",
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path',
                                              'mask', 'prune', 'node_size'])

        export_to_pandas_node = pe.MapNode(interface=Export2Pandas(), name = "export_to_pandas",
                                           iterfield=['in_csv', 'ID', 'network', 'mask'])

        collect_pandas_dfs_node = pe.Node(interface=CollectPandasDfs(), name = "CollectPandasDfs",
                                           input_files = ['input_file', 'atlas_select', 'clust_mask',
                                                          'k_min', 'k_max', 'k', 'k_step', 'min_thr',
                                                          'max_thr', 'step_thr', 'multi_thr', 'thr',
                                                          'mask', 'ID', 'network', 'k_clustering',
                                                          'conn_model', 'in_csv', 'user_atlas_list',
                                                          'clust_mask_list', 'multi_atlas', 'node_size',
                                                          'node_size_list', 'parc'])

        if multi_nets:
            collect_pandas_dfs_node_iterables = []
            collect_pandas_dfs_node_iterables.append(("network", multi_nets))
            collect_pandas_dfs_node.iterables = collect_pandas_dfs_node_iterables

        ##Connect nodes of workflow
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
                                  ('bedpostx_dir', 'bedpostx_dir'),
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
                                  ('num_total_samples', 'num_total_samples')]),
            (imp_est, net_mets_node, [('outputs.est_path', 'est_path'),
                                      ('outputs.network', 'network'),
                                      ('outputs.thr', 'thr'),
                                      ('outputs.ID', 'ID'),
                                      ('outputs.conn_model', 'conn_model'),
                                      ('outputs.mask', 'mask'),
                                      ('outputs.prune', 'prune'),
                                      ('outputs.node_size', 'node_size')]),
            (imp_est, export_to_pandas_node, [('outputs.network', 'network'),
                                              ('outputs.ID', 'ID'),
                                              ('outputs.mask', 'mask')]),
            (net_mets_node, export_to_pandas_node, [('out_file', 'in_csv')]),
            (export_to_pandas_node, collect_pandas_dfs_node, [('out_file', 'in_csv')]),
            (inputnode, collect_pandas_dfs_node, [('in_file', 'input_file'),
                                                  ('atlas_select', 'atlas_select'),
                                                  ('clust_mask', 'clust_mask'),
                                                  ('k_min', 'k_min'),
                                                  ('k_max', 'k_max'),
                                                  ('k', 'k'),
                                                  ('k_step', 'k_step'),
                                                  ('min_thr', 'min_thr'),
                                                  ('max_thr', 'max_thr'),
                                                  ('step_thr', 'step_thr'),
                                                  ('multi_thr', 'multi_thr'),
                                                  ('thr', 'thr'),
                                                  ('ID', 'ID'),
                                                  ('mask', 'mask'),
                                                  ('network', 'network'),
                                                  ('k_clustering', 'k_clustering'),
                                                  ('conn_model', 'conn_model'),
                                                  ('user_atlas_list','user_atlas_list'),
                                                  ('clust_mask_list','clust_mask_list'),
                                                  ('multi_atlas','multi_atlas'),
                                                  ('node_size','node_size'),
                                                  ('node_size_list','node_size_list'),
                                                  ('parc', 'parc')])
        ])
        return wf

    def wf_multi_subject(subjects_list, atlas_select, network, node_size, mask,
    thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
    plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr,
    anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering,
    user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples):
        wf_multi = pe.Workflow(name='PyNets_multisubject')
        cores=[]
        ram=[]
        i=0
        for _file in subjects_list:
            wf_single_subject = init_wf_single_subject(ID=os.path.dirname(os.path.realpath(subjects_list[i])).split('/')[-1],
                               input_file=_file,
                               dir_path=os.path.dirname(os.path.realpath(subjects_list[i])),
                               atlas_select=atlas_select,
                               network=network,
                               node_size=node_size,
                               mask=mask,
                               thr=thr,
                               parlistfile=parlistfile,
                               multi_nets=multi_nets,
                               conn_model=conn_model,
                               dens_thresh=dens_thresh,
                               conf=conf,
                               adapt_thresh=adapt_thresh,
                               plot_switch=plot_switch,
                               bedpostx_dir=bedpostx_dir,
                               multi_thr=multi_thr,
                               multi_atlas= multi_atlas,
                               min_thr=min_thr,
                               max_thr=max_thr,
                               step_thr=step_thr,
                               anat_loc=anat_loc,
                               parc=parc,
                               ref_txt=ref_txt,
                               procmem=procmem,
                               k=k,
                               clust_mask=clust_mask,
                               k_min=k_min,
                               k_max=k_max,
                               k_step=k_step,
                               k_clustering=k_clustering,
                               user_atlas_list=user_atlas_list,
                               clust_mask_list=clust_mask_list,
                               prune=prune,
                               node_size_list=node_size_list,
                               num_total_samples=num_total_samples)
            wf_multi.add_nodes([wf_single_subject])
            cores.append(int(procmem[0]))
            ram.append(int(procmem[1]))
            i = i + 1
        total_cores = sum(cores)
        total_ram = sum(ram)
        return wf_multi, total_cores, total_ram

    ##Workflow generation
    import logging
    from time import gmtime, strftime
    callback_log_path = "%s%s%s%s%s" % ('/tmp/pynets_run_stats_', ID, '_', strftime("%Y-%m-%d %H:%M:%S", gmtime()), '.log')
    logger = logging.getLogger('callback')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(callback_log_path)
    logger.addHandler(handler)

    ##Multi-subject workflow generator
    if subjects_list:
        [wf_multi, total_cores, total_ram] = wf_multi_subject(subjects_list, atlas_select, network, node_size,
        mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
        plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr,
        anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering,
        user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples)
        wf_multi.config['logging']['log_directory']='/tmp'
        wf_multi.config['logging']['workflow_level']='DEBUG'
        wf_multi.config['logging']['utils_level']='DEBUG'
        wf_multi.config['logging']['interface_level']='DEBUG'
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1])}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf_multi.run(plugin='MultiProc', plugin_args= plugin_args)
        #wf_multi.run()
    ##Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
        node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
        adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
        max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min,
        k_max, k_step, k_clustering, user_atlas_list, clust_mask_list, prune, node_size_list, num_total_samples)
        wf.config['logging']['log_directory']='/tmp'
        wf.config['logging']['workflow_level']='DEBUG'
        wf.config['logging']['utils_level']='DEBUG'
        wf.config['logging']['interface_level']='DEBUG'
        #wf.write_graph(graph2use='flat', format='png', dotfilename='indiv_wf.dot')
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1])}
        print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
        wf.run(plugin='MultiProc', plugin_args= plugin_args)
        #wf.run()

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')
