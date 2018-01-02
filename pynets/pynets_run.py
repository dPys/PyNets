# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import sys
import argparse
import os
import timeit
import pandas as pd
import numpy as np
from pynets.utils import do_dir_path

##Start time clock
start_time = timeit.default_timer()

####Parse arguments####
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag")
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
        metavar='Path to input file',
        default=None,
        required=False,
        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR the path to a text file containing a list of paths to subject files')
    parser.add_argument('-ID',
        metavar='Subject ID',
        default=None,
        required=False,
        help='A subject ID that is also the name of the directory containing the input file')
    parser.add_argument('-a',
        metavar='Atlas',
        default='coords_power_2011',
        help='Specify a coordinate atlas parcellation of those availabe in nilearn. Default is coords_power_2011. If you wish to iterate your pynets run over multiple nilearn atlases, separate by commas. e.g. -a \'atlas_aal,atlas_destrieux_2009\' Available nilearn atlases are:\n\natlas_aal \natlas_destrieux_2009 \ncoords_dosenbach_2010 \ncoords_power_2011')
    #parser.add_argument('-basc',
        #default=False,
        #action='store_true',
        #help='Specify whether you want to run BASC to calculate a group level set of nodes')
    parser.add_argument('-ua',
        metavar='Path to parcellation file',
        default=None,
        help='Path to nifti-formatted parcellation image file. If specifying a list of paths to multiple user atlases, separate by comma.')
    parser.add_argument('-pm',
        metavar='cores,memory',
        default= '2,4',
        help='Number of cores to use, number of GB of memory to use, please enter as two integers seperated by a comma')
    parser.add_argument('-n',
        metavar='Resting-state network',
        default=None,
        help='Optionally specify the name of any of the 2017 Yeo-Schaefer RSNs (7-network or 17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar. If listing multiple, separate by commas. e.g. -n \'Default,Cont,SalVentAttn\'')
    parser.add_argument('-thr',
        metavar='graph threshold',
        default='0.95',
        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is 0.95')
    parser.add_argument('-ns',
        metavar='Node size',
        default='2',
        help='Optionally specify a coordinate-based node radius size. Default is 3 voxels')
    parser.add_argument('-m',
        metavar='Path to mask image',
        default=None,
        help='Optionally specify a thresholded inverse-binarized mask image such as a group ICA-derived network volume, to retain only those network nodes contained within that mask')
    parser.add_argument('-model',
        metavar='Graph estimator',
        default='corr',
        help='Optionally specify matrix estimation type: corr, cov, sps, partcorr, or tangent for correlation, covariance, sparse-inverse covariance, partial correlation, and tangent, respectively')
    parser.add_argument('-confounds',
        metavar='Confounds',
        default=None,
        help='Optionally specify a path to a confound regressor file to improve in the signal estimation for the graph')
    parser.add_argument('-dt',
        metavar='Density threshold',
        default=None,
        help='Optionally indicate a target density of graph edges to be achieved through iterative absolute thresholding. In group analysis, this could be determined by finding the mean density of all unthresholded graphs across subjects, for instance.')
#    parser.add_argument('-at',
#        default=False,
#        action='store_true',
#        help='Optionally use this flag if you wish to activate adaptive thresholding')
    parser.add_argument('-plt',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting of adjacency matrices, connectomes, and time-series')
    parser.add_argument('-bpx',
        metavar='Path to bedpostx directory',
        default=None,
        help='Formatted according to the FSL default tree structure found at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX')
    parser.add_argument('-anat',
        metavar='Path to subject anatomical image (skull-stripped and normalized to MNI space)',
        default=None,
        help='Optional with the -bpx flag to initiate probabilistic connectome estimation using parcels (recommended) as opposed to coordinate-based spherical volumes')
    parser.add_argument('-min_thr',
        metavar='Multi-thresholding minimum threshold',
        default=None,
        help='Minimum threshold for multi-thresholding.')
    parser.add_argument('-max_thr',
        metavar='Multi-thresholding maximum threshold',
        default=None,
        help='Maximum threshold for multi-thresholding.')
    parser.add_argument('-step_thr',
        metavar='Multi-thresholding step size',
        default=None,
        help='Threshold step value for multi-thresholding. Default is 0.01.')
    parser.add_argument('-parc',
        default=False,
        action='store_true',
        help='Include this flag to use parcels instead of coordinates as nodes.')
    parser.add_argument('-ref',
        metavar='atlas reference file path',
        default=None,
        help='Specify the path to the atlas reference .txt file')
    parser.add_argument('-k',
        metavar='Number of clusters',
        default=None,
        help='Specify a number of clusters to produce')
    parser.add_argument('-k_min',
        metavar='Min k clusters',
        default=None,
        help='Specify the minimum k clusters')
    parser.add_argument('-k_max',
        metavar='Max k clusters',
        default=None,
        help='Specify the maximum k clusters')
    parser.add_argument('-k_step',
        metavar='k cluster step size',
        default=None,
        help='Specify the step size of k cluster iterables')
    parser.add_argument('-cm',
        metavar='Cluster mask',
        default=None,
        help='Specify the path to the mask within which to perform clustering. If specifying a list of paths to multiple cluster masks, separate by comma.')
    args = parser.parse_args()

    ###Set Arguments to global variables###
    input_file=args.i
    ID=args.ID
    #basc=args.basc
    procmem=list(eval(str((args.pm))))      
    thr=args.thr
    node_size=args.ns
    mask=args.m
    conn_model=args.model
    conf=args.confounds
    dens_thresh=args.dt
#    adapt_thresh=args.at
    adapt_thresh=False
    plot_switch=args.plt
    bedpostx_dir=args.bpx
    min_thr=args.min_thr
    max_thr=args.max_thr
    step_thr=args.step_thr
    anat_loc=args.anat
    parc=args.parc
    ref_txt=args.ref
    k=args.k
    k_min=args.k_min
    k_max=args.k_max
    k_step=args.k_step
    clust_mask_pre = args.cm
    clust_mask = list(str(clust_mask_pre).split(','))
    if len(clust_mask) > 1:  
        clust_mask_list=clust_mask
        clust_mask=None
    else:
        clust_mask = clust_mask[0]
        clust_mask_list = None      
    network_pre = args.n
    network = list(str(network_pre).split(','))
    if len(network) > 1:  
        multi_nets=network
        network=None
    else:
        network = network[0]
        multi_nets=None  
    parlistfile_pre = args.ua
    parlistfile = list(str(parlistfile_pre).split(','))
    if len(parlistfile) > 1:  
        user_atlas_list=parlistfile
        parlistfile=user_atlas_list[0]
    else:
        parlistfile = parlistfile[0]
        user_atlas_list=None
    atlas_select_pre = args.a
    atlas_select = list(str(atlas_select_pre).split(','))
    if len(atlas_select) > 1:  
        multi_atlas=atlas_select
        atlas_select=None
    else:
        atlas_select = atlas_select[0]
        multi_atlas=None  
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

    if dens_thresh is not None or adapt_thresh != False:
        thr=None
    else:
        thr=float(thr)

    if anat_loc is not None and bedpostx_dir is None:
        print('Warning: anatomical image specified, but not bedpostx directory specified. Anatomical images are only supported for structural connectome estimation at this time.')

    if multi_thr==True:
        dens_thresh=None
        adapt_thresh=False
    elif multi_thr==False and min_thr is not None and max_thr is not None:
        multi_thr=True
        dens_thresh=None
        adapt_thresh=False
    else:
        min_thr=None
        max_thr=None
        step_thr=None

    print ("SUBJECT ID: " + str(ID))

    if (k_min != None and k_max != None) and k == None and clust_mask_list != None:
        k_clustering = 4
    elif (k_min != None and k_max != None) and k == None and clust_mask_list == None:
        k_clustering = 2
    elif k != None and (k_min == None and k_max == None) and clust_mask_list != None:
        k_clustering = 3
    elif k != None and (k_min == None and k_max == None) and clust_mask_list == None:
        k_clustering = 1
    else:
        k_clustering = 0
            
    if input_file:
        if parlistfile != None and k_clustering == 0 and user_atlas_list == None:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            dir_path = do_dir_path(atlas_select, input_file)
            print ("ATLAS: " + str(atlas_select))
        elif parlistfile != None and user_atlas_list == None:
            atlas_select = parlistfile.split('/')[-1].split('.')[0]
            dir_path = do_dir_path(atlas_select, input_file)
            print ("ATLAS: " + str(atlas_select))
        elif user_atlas_list != None:
            parlistfile = user_atlas_list[0]
            print ('Iterating across multiple user atlases...')
            for parlistfile in user_atlas_list:
                atlas_select = parlistfile.split('/')[-1].split('.')[0]
                dir_path = do_dir_path(atlas_select, input_file)
            atlas_select = None
        elif (user_atlas_list != None or parlistfile != None) and (k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1):
            print('Error: the -ua flag cannot be used with the clustering option. Use the -cm flag instead.')
            sys.exit(0)
        elif k_clustering == 1:
            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0] 
            atlas_select = str(ID) + '_' + cl_mask_name + '_k' + str(k)
            dir_path = do_dir_path(atlas_select, input_file)    
            print ("Clustering within mask at a single resolution...")
        elif k_clustering == 2:
            cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            atlas_select = str(ID) + '_' + cl_mask_name + '_k' + str(k_list[0])
            dir_path = do_dir_path(atlas_select, input_file)
            print ("Clustering within mask at multiple resolutions...")
        elif k_clustering == 3:
            print ("Clustering within multiple masks at a single resolution...")
            for clust_mask in clust_mask_list:
                cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0] 
                atlas_select = str(ID) + '_' + cl_mask_name + '_k' + str(k)
                dir_path = do_dir_path(atlas_select, input_file)
        elif k_clustering == 4:
            print ("Clustering within multiple masks at multiple resolutions...")
            k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
            for clust_mask in clust_mask_list:
                cl_mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
                atlas_select = str(ID) + '_' + cl_mask_name + '_k' + str(k_list[0])
                dir_path = do_dir_path(atlas_select, input_file)
        else:
            dir_path = do_dir_path(atlas_select, input_file)
        
    if ref_txt != None and os.path.exists(ref_txt):
        atlas_select = os.path.basename(ref_txt).split('.txt')[0]
        dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
        indices = dict_df.Index.unique().tolist()
        label_names = dict_df['Region'].tolist()

    if network != None:
        print ("NETWORK: " + str(network))
    elif multi_nets is not None:
        network = multi_nets[0]
        print ('Iterating workflow across ' + str(len(multi_nets)) + ' networks: ' + str(', '.join(str(n) for n in multi_nets)) + '...')
    else:
        print("Using whole-brain pipeline..." )
    
    if input_file and subjects_list:
        print("\n")
        print('Running workflow of workflows across subjects:\n')
        print (str(subjects_list))
        ##Set directory path containing input file  
        dir_path = do_dir_path(atlas_select, subjects_list[0])
    elif input_file and bedpostx_dir:
        print('Running joint structural-functional connectometry...')
        print ("Functional file: " + input_file)
        print ("Bedpostx Directory: " + bedpostx_dir)
        if anat_loc is not None:
            print ("Anatomical Image: " + anat_loc)
        if network is not None:
            print('RSN: ' + network)
        ##Set directory path containing input file
        nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
        ref_txt = parlistfile.split('/')[-1:][0].split('.')[0] + '.txt'
        dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
    elif input_file is None and bedpostx_dir:
        print('Running structural connectometry only...')
        print ("Bedpostx Directory: " + bedpostx_dir)
        if anat_loc is not None:
            print ("Anatomical Image: " + anat_loc)
        if network is not None:
            print('RSN: ' + network)
        ##Set directory path containing input file
        nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
        ref_txt = parlistfile.split('/')[-1:][0].split('.')[0] + '.txt'
        dir_path = do_dir_path(atlas_select, nodif_brain_mask_path)
    elif input_file and bedpostx_dir is None and subjects_list is None:
        print('Running functional connectometry only...')
        print ("Functional file: " + input_file)
    print('\n-------------------------------------------------------------------------\n\n\n')
    
    ##Import core modules
    import warnings
    warnings.simplefilter("ignore")
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

    def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list):
        import os
        import numpy as np
        from pynets import workflows, utils
        
        ##Workflow 1: Whole-brain functional connectome
        if bedpostx_dir == None and network == None:
            [est_path, thr] = workflows.wb_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list)
        ##Workflow 2: RSN functional connectome
        elif bedpostx_dir == None:
            [est_path, thr] = workflows.rsn_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list)
        ##Workflow 3: Whole-brain structural connectome
        elif bedpostx_dir != None and network == None:
            est_path = workflows.wb_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, label_names, anat_loc)
            thr=None
        ##Workflow 4: RSN structural connectome
        elif bedpostx_dir != None:
            est_path = workflows.rsn_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, label_names, anat_loc)
            thr=None
            
        ##Build iterfields
        if multi_atlas is not None or multi_thr==True or multi_nets is not None or k_clustering == 2:
            ##Create est_path_list iterfield based on iterables across atlases, RSN's, k-values, and thresholding ranges          
            est_path_list = []
            if k_clustering == 2:
                print('\nIterating pipeline for ' + str(ID) + ' across multiple clustering resolutions...\n')
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
                k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                num_atlases = len(k_list)
                for k in k_list:
                    atlas_select = str(ID) + '_' + mask_name + '_k' + str(k)
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                        for network in multi_nets:
                            dir_path = utils.do_dir_path(atlas_select, input_file)
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                    else:
                        num_networks =  1
                        dir_path = utils.do_dir_path(atlas_select, input_file)
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
            elif k_clustering == 3:
                print('\nIterating pipeline for ' + str(ID) + ' across multiple masks at a single clustering resolution...\n')
                num_atlases = len(clust_mask_list)
                for clust_mask in clust_mask_list:                
                    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
                    atlas_select = str(ID) + '_' + mask_name + '_k' + str(k)
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                        for network in multi_nets:
                            dir_path = utils.do_dir_path(atlas_select, input_file)
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                    else:
                        num_networks =  1
                        dir_path = utils.do_dir_path(atlas_select, input_file)
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
            elif k_clustering == 4:
                print('\nIterating pipeline for ' + str(ID) + ' across multiple clustering resolutions and masks...\n')
                k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                num_atlases = len(k_list) * len(clust_mask_list)
                for clust_mask in clust_mask_list:                
                    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
                    for k in k_list:
                        atlas_select = str(ID) + '_' + mask_name + '_k' + str(k)
                        if multi_nets is not None:
                            num_networks = len(multi_nets)
                            print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                            for network in multi_nets:
                                dir_path = utils.do_dir_path(atlas_select, input_file)
                                [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                        else:
                            num_networks =  1
                            dir_path = utils.do_dir_path(atlas_select, input_file)
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
            elif multi_atlas is not None:
                num_atlases = len(multi_atlas)
                print('\nIterating pipeline for ' + str(ID) + ' across multiple atlases: ' + '\n'.join(str(n) for n in multi_atlas) + '...\n')
                for atlas_select in multi_atlas:
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                        for network in multi_nets:
                            dir_path = utils.do_dir_path(atlas_select, input_file)
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                    else:
                        num_networks =  1
                        dir_path = utils.do_dir_path(atlas_select, input_file)
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
            elif user_atlas_list:
                num_atlases = len(user_atlas_list)
                print('\nIterating pipeline for ' + str(ID) + ' across multiple atlases: ' + '\n'.join(str(a) for a in user_atlas_list) + '...\n')
                for parlistfile in user_atlas_list:
                    atlas_select = parlistfile.split('/')[-1].split('.')[0]
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                        for network in multi_nets:
                            dir_path = utils.do_dir_path(atlas_select, input_file)
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                    else:
                        num_networks =  1
                        dir_path = utils.do_dir_path(atlas_select, input_file)
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
            else:
                num_atlases = 1
                if multi_thr==True:
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                        for network in multi_nets:
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                    else:
                        num_networks =  1
                        [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                else:
                    if multi_nets is not None:
                        num_networks = len(multi_nets)
                        for network in multi_nets:
                            [iter_thresh, est_path_list] = utils.build_est_path_list(multi_thr, min_thr, max_thr, step_thr, ID, network, conn_model, thr, mask, dir_path, est_path_list)
                        iter_thresh = [thr]
                    else:
                        est_path_list = [est_path]
                        num_networks =  1
                        iter_thresh = [thr]
        
            ##Create network_list based on iterables across atlases, RSN's, k-values, and thresholding ranges      
            if multi_nets is not None:
                print('\nIterating pipeline for ' + str(ID) + ' across networks: ' + '\n'.join(str(n) for n in multi_nets) + '...\n')
                network_list = []
                if multi_atlas is not None:
                    for atlas in multi_atlas:
                        for network in multi_nets:
                            if multi_thr == True:
                                for thr in iter_thresh:
                                    network_list.append(network)
                            else:
                                network_list.append(network)
                elif user_atlas_list:
                    for atlas in user_atlas_list:
                        for network in multi_nets:
                            if multi_thr == True:
                                for thr in iter_thresh:
                                    network_list.append(network)
                            else:
                                network_list.append(network)
                elif k_clustering == 2:
                    k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                    for k in k_list:
                        for network in multi_nets:
                            if multi_thr == True:
                                for thr in iter_thresh:
                                    network_list.append(network)
                            else:
                                network_list.append(network)
                elif k_clustering == 4:
                    k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                    for clust_mask in clust_mask_list:
                        for k in k_list:
                            for network in multi_nets:
                                if multi_thr == True:
                                    for thr in iter_thresh:
                                        network_list.append(network)
                                else:
                                    network_list.append(network)
                elif k_clustering == 3:
                    for clust_mask in clust_mask_list:
                        for network in multi_nets:
                            if multi_thr == True:
                                for thr in iter_thresh:
                                    network_list.append(network)
                            else:
                                network_list.append(network)
                else:
                    for network in multi_nets:
                        if multi_thr == True:
                            for thr in iter_thresh:
                                network_list.append(network)
                        else:
                            network_list.append(network)
            elif network is not None and multi_nets is None:             
                network_list = [network] * len(est_path_list)
            else:             
                network_list = [None] * len(est_path_list)
            
            if multi_thr == True:
                thr = iter_thresh * num_atlases * num_networks
            else:
                thr = iter_thresh
            est_path = est_path_list
            network = network_list 
            ID = [str(ID)] * len(est_path_list)
            mask = [mask] * len(est_path_list)
            conn_model = [conn_model] * len(est_path_list)
            k_clustering = [k_clustering] * len(est_path_list)
            
            '''print('\n\n\n')
            print(thr)
            print(len(thr))
            print('\n\n\n')
            print(est_path)
            print(len(est_path))
            print('\n\n\n')
            print(network)
            print(len(network))
            print('\n\n\n')
            print(ID)
            print(len(ID))
            print('\n\n\n')
            print(mask)
            print(len(mask))
            print('\n\n\n')
            print(conn_model)
            print(len(conn_model))
            print('\n\n\n')
            print(k_clustering)
            print(len(k_clustering))
            print('\n\n\n')
            import sys
            sys.exit(0)'''
            
        return(est_path, thr, network, ID, mask, conn_model, k_clustering)
        
    class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
        ID = traits.Any(mandatory=True)
        network = traits.Any(mandatory=False)
        thr = traits.Any(mandatory=True)
        conn_model = traits.Str(mandatory=True)
        est_path = File(exists=False, mandatory=True, desc="")
        mask = traits.Any(mandatory=False)

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
                self.inputs.mask)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(getattr(self, '_outpath'))}

    class Export2PandasInputSpec(BaseInterfaceInputSpec):
        in_csv = File(exists=False, mandatory=True, desc="")
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
            out_file=self.inputs.out_file)
            return runtime
    
        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(self.inputs.out_file)}
        
    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
    node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
    adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
    max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, 
    k_max, k_step, k_clustering, user_atlas_list, clust_mask_list):
        wf = pe.Workflow(name='PyNets_' + str(ID))
        wf.base_directory='/tmp/pynets'
        ##Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID',
        'atlas_select', 'network', 'thr', 'node_size', 'mask', 'parlistfile',
        'multi_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch',
        'bedpostx_dir', 'anat_loc', 'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr', 
        'multi_atlas', 'max_thr', 'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 
        'k_max', 'k_step', 'k_clustering', 'user_atlas_list', 'clust_mask_list']), name='inputnode')

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

        #3) Add variable to function nodes
        ##Create function nodes
        imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select',
        'network', 'node_size', 'mask', 'thr', 'parlistfile', 'multi_nets', 'conn_model',
        'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch', 'bedpostx_dir', 'anat_loc', 
        'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr', 'multi_atlas', 'max_thr', 
        'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max', 'k_step', 'k_clustering', 
        'user_atlas_list', 'clust_mask_list'],
        output_names = ['est_path', 'thr', 'network', 'ID', 'mask', 'conn_model', 'k_clustering'], 
        function=workflow_selector),
        name = "imp_est")

        ##Create MapNode types for net_mets_node and export_to_pandas_node
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name = "ExtractNetStats", 
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path', 'mask'])
        
        export_to_pandas_node = pe.MapNode(interface=Export2Pandas(), name = "export_to_pandas", 
                                           iterfield=['in_csv', 'ID', 'network', 'mask'])
        
        collect_pandas_dfs_node = pe.Node(interface=CollectPandasDfs(), name = "CollectPandasDfs", 
                                           input_files = ['input_file', 'atlas_select', 'clust_mask', 
                                                          'k_min', 'k_max', 'k', 'k_step', 'min_thr', 
                                                          'max_thr', 'step_thr', 'multi_thr', 'thr', 
                                                          'mask', 'ID', 'network', 'k_clustering', 
                                                          'conn_model', 'in_csv', 'user_atlas_list', 
                                                          'clust_mask_list'])
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
                                  ('clust_mask_list', 'clust_mask_list')]),
            (imp_est, net_mets_node, [('est_path', 'est_path'),
                                      ('network', 'network'),
                                      ('thr', 'thr'),
                                      ('ID', 'ID'),        
                                      ('conn_model', 'conn_model'),
                                      ('mask', 'mask')]),
            (imp_est, export_to_pandas_node, [('network', 'network'),
                                              ('ID', 'ID'),
                                              ('mask', 'mask')]),    
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
                                                ('clust_mask_list','clust_mask_list')])
        ])
        return wf

    def wf_multi_subject(subjects_list, atlas_select, network, node_size, mask,
    thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
    plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr, 
    anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering, 
    user_atlas_list, clust_mask_list):
        wf_multi = pe.Workflow(name='PyNets_multisubject')
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
                               clust_mask_list=clust_mask_list)
            wf_multi.add_nodes([wf_single_subject])
            i = i + 1
        return wf_multi

    ##Workflow generation
    #import logging
    #from time import gmtime, strftime
    #from nipype.utils.profiler import log_nodes_cb
    #callback_log_path = '/tmp/run_stats' + '_' + str(ID) + '_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.log'
    #logger = logging.getLogger('callback')
    #logger.setLevel(logging.DEBUG)
    #handler = logging.FileHandler(callback_log_path)
    #logger.addHandler(handler)

    if subjects_list:
        wf_multi = wf_multi_subject(subjects_list, atlas_select, network, node_size,
        mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
        plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr,
        anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering, 
        user_atlas_list, clust_mask_list)
        wf_multi.config['logging']['log_directory']='/tmp'
        wf_multi.config['logging']['workflow_level']='DEBUG'
        wf_multi.config['logging']['utils_level']='DEBUG'
        wf_multi.config['logging']['interface_level']='DEBUG'
        #plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1]), 'status_callback' : log_nodes_cb}
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1])}
        print('\n' + 'Running with ' + str(plugin_args) + '\n')
        wf_multi.run(plugin='MultiProc', plugin_args= plugin_args)
        #wf_multi.run()
    ##Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
        node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
        adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
        max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, 
        k_max, k_step, k_clustering, user_atlas_list, clust_mask_list)
        #plugin_args = {'status_callback' : log_nodes_cb}
        wf.config['logging']['log_directory']='/tmp'
        wf.config['logging']['workflow_level']='DEBUG'
        wf.config['logging']['utils_level']='DEBUG'
        wf.config['logging']['interface_level']='DEBUG'
        wf.run()

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')