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

# Start time clock
start_time = timeit.default_timer()

####Parse arguments####
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag")
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
        metavar='path to input file',
        default=None,
        required=False,
        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR the path to a text file containing a list of paths to subject files')
    parser.add_argument('-ID',
        metavar='subject ID',
        default=None,
        required=False,
        help='A subject ID that is also the name of the directory containing the input file')
    parser.add_argument('-a',
        metavar='atlas',
        default='coords_power_2011',
        help='Specify a single coordinate atlas parcellation of those availabe in nilearn. Default is coords_power_2011. Available atlases are:\n\natlas_aal \natlas_destrieux_2009 \ncoords_dosenbach_2010 \ncoords_power_2011')
    parser.add_argument('-basc',
        default=False,
        action='store_true',
        help='Specify whether you want to run BASC to calculate a group level set of nodes')
    parser.add_argument('-ua',
        metavar='path to parcellation file',
        default=None,
        help='Path to nifti-formatted parcellation image file')
    parser.add_argument('-pm',
        metavar='cores,memory',
        default= '2,4',
        help='Number of cores to use, number of GB of memory to use, please enter as two integers seperated by a comma')
    parser.add_argument('-n',
        metavar='resting-state network',
        default=None,
        help='Optionally specify the name of one of the 2017 Yeo-Schaefer RSNs (7-network or 17-network): Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole, ContA, ContB, ContC, DefaultA, DefaultB, DefaultC, TempPar')
    parser.add_argument('-thr',
        metavar='graph threshold',
        default='0.95',
        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is 0.95')
    parser.add_argument('-ns',
        metavar='node size',
        default='2',
        help='Optionally specify a coordinate-based node radius size. Default is 3 voxels')
    parser.add_argument('-m',
        metavar='path to mask image',
        default=None,
        help='Optionally specify a thresholded inverse-binarized mask image such as a group ICA-derived network volume, to retain only those network nodes contained within that mask')
    parser.add_argument('-mn',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting designations and network statistic extraction for all Yeo 7 RSNs in the specified atlas')
    parser.add_argument('-model',
        metavar='connectivity',
        default='corr',
        help='Optionally specify matrix estimation type: corr, cov, or sps for correlation, covariance, or sparse-inverse covariance, respectively')
    parser.add_argument('-confounds',
        metavar='confounds',
        default=None,
        help='Optionally specify a path to a confound regressor file to improve in the signal estimation for the graph')
    parser.add_argument('-dt',
        metavar='density threshold',
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
    parser.add_argument('-ma',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to iterate your pynets run over all available nilearn atlases')
    parser.add_argument('-mt',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to iterate your pynets run over a range of proportional thresholds from 0.99 to 0.90')
    parser.add_argument('-bpx',
        metavar='path to bedpostx directory',
        default=None,
        help='Formatted according to the FSL default tree structure found at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX')
    parser.add_argument('-anat',
        metavar='path to subject anatomical image (skull-stripped and normalized to MNI space)',
        default=None,
        help='Optional with the -bpx flag to initiate probabilistic connectome estimation using parcels (recommended) as opposed to coordinate-based spherical volumes')
    parser.add_argument('-min_thr',
        metavar='multi-thresholding minimum threshold',
        default=0.90,
        help='Minimum threshold for multi-thresholding. Default is 0.90.')
    parser.add_argument('-max_thr',
        metavar='multi-thresholding maximum threshold',
        default=0.99,
        help='Maximum threshold for multi-thresholding. Default is 0.99.')
    parser.add_argument('-step_thr',
        metavar='multi-thresholding step size',
        default=0.01,
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
        default=False,
        help='Specify the path to the mask within which to perform clustering')    
    args = parser.parse_args()

    ###Set Arguments to global variables###
    input_file=args.i
    ID=args.ID
    atlas_select=args.a
    basc=args.basc
    parlistfile=args.ua
    procmem=list(eval(str((args.pm))))
    network=args.n
    thr=args.thr
    node_size=args.ns
    mask=args.m
    multi_nets=args.mn
    conn_model=args.model
    conf=args.confounds
    dens_thresh=args.dt
#    adapt_thresh=args.at
    adapt_thresh=False
    plot_switch=args.plt
    multi_atlas=args.ma
    multi_thr=args.mt
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
    clust_mask=args.cm

    print("\n\n\n" + "------------------------------------------------------------------------" + "\n")
    print('Starting up!')

    ##Check required inputs for existence, and configure run
    if input_file.endswith('.txt'):
        with open(input_file) as f:
            subjects_list = f.read().splitlines()
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
    else:
        min_thr=None
        max_thr=None
        step_thr=None

    print ("SUBJECT ID: " + str(ID))

    if k_min is not None and k_max is not None:
        k_clustering = 2
    elif k is not None:
        k_clustering = 1
    else:
        k_clustering = 0

    if parlistfile != None and k_clustering == 0:
        atlas_select = parlistfile.split('/')[-1].split('.')[0]
        dir_path = do_dir_path(atlas_select, input_file)
        print ("ATLAS: " + str(atlas_select))
    elif parlistfile != None and (k_clustering == 2 or k_clustering == 1):
        print('Error: the -ua flag cannot be used with the clustering option. Use the -cm flag instead.')
        sys.exit(0)
    elif k_clustering == 1:
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0] 
        atlas_select = mask_name + '_k' + str(k)
        dir_path = do_dir_path(atlas_select, input_file)    
        print ("Clustering within mask to build atlas...")
    elif k_clustering == 2:
        mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        atlas_select = mask_name + '_k' + str(k_list[0])
        dir_path = do_dir_path(atlas_select, input_file)
        print ("Iterative clustering within mask to build atlases...")
    else:
        dir_path = do_dir_path(atlas_select, input_file)

    if multi_atlas == True:
        print('\nIterating across multiple atlases...\n')
        
    if ref_txt != None and os.path.exists(ref_txt):
        atlas_select = os.path.basename(ref_txt).split('.txt')[0]
        dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
        indices = dict_df.Index.unique().tolist()
        label_names = dict_df['Region'].tolist()

    if network != None:
        print ("NETWORK: " + str(network))
    elif multi_nets == True:
        network = 'Vis'
        print ("Iterating workflow across all 7 Yeo networks...")
    else:
        print("USING WHOLE-BRAIN CONNECTOME..." )
    
    if input_file is not None and subjects_list is not None:
        print("\n")
        print('Running workflow of workflows across subjects:\n')
        print (str(subjects_list))
        ##Set directory path containing input file  
        dir_path = do_dir_path(atlas_select, subjects_list[0])
    elif input_file is not None and bedpostx_dir is not None and atlas_select != 'Clustered':
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
    elif input_file is None and bedpostx_dir is not None and atlas_select != 'Clustered':
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
    elif input_file is not None and bedpostx_dir is None and subjects_list is None and atlas_select != 'Clustered':
        print('Running functional connectometry only...')
        print ("Functional file: " + input_file)
    print("\n" + "-------------------------------------------------------------------------" + "\n\n\n")

    ##Import core modules
    import warnings
    warnings.simplefilter("ignore")
    from pynets.utils import export_to_pandas
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

    if basc == True:
       from pynets import basc_run
       from pathlib import Path
       basc_config=Path(__file__).parent/'basc_config.yaml'

       print("\n\n\n-------------() > STARTING BASC < ()----------------------" + "\n\n\n")

       basc_run.basc_run(subjects_list, basc_config)
       parlistfile=Path(__file__)/'pynets'/'rsnrefs'/'group_stability_clusters.nii.gz'

    def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering):
        import os
        import numpy as np
        from pynets import workflows, utils

        ##Case 1: Whole-brain functional connectome
        if bedpostx_dir == None and network == None:
            [est_path, thr] = workflows.wb_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering)
        ##Case 2: RSN functional connectome
        elif bedpostx_dir == None:
            [est_path, thr] = workflows.RSN_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask)
        ##Case 3: Whole-brain structural connectome
        elif bedpostx_dir != None and network == None:
            [est_path, thr] = workflows.wb_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parc, dict_df, anat_loc, ref_txt, int(procmem[0]), dir_path, multi_thr, multi_atlas, multi_nets, max_thr, min_thr, k, clust_mask)
        ##Case 4: RSN structural connectome
        elif bedpostx_dir != None:
            [est_path, thr] = workflows.RSN_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parc, dict_df, anat_loc, ref_txt, int(procmem[0]), dir_path, multi_thr, multi_atlas, multi_nets, max_thr, min_thr, k, clust_mask)

        ##Create est_path iterables for network extraction across multiple graph outputs             
        if multi_atlas==True or multi_thr==True or multi_nets==True or k_clustering == 2:
            est_path_list = []
            if k_clustering == 2 and multi_atlas==False:
                print('Iterating pipeline for ' + str(ID) + ' across multiple clustering resolutions...')
                mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]            
                k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
                for k in k_list:
                    atlas_select = mask_name + '_k' + str(k)
                    dir_path = utils.do_dir_path(atlas_select, input_file)
                    est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path)
                    est_path_list.append(est_path_tmp)      
            elif multi_atlas==True:
                print('Iterating pipeline for ' + str(ID) + ' across multiple atlases...')
                atlas_list = ['coords_power_2011', 'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal']
                for atlas_select in atlas_list:
                    dir_path = utils.do_dir_path(atlas_select, input_file)
                    est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path)
                    est_path_list.append(est_path_tmp)   
            if multi_thr==True:
                print('Iterating pipeline for ' + str(ID) + ' across multiple thresholds...')
                iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
                float(max_thr), float(step_thr)),decimals=2).tolist()]
                for thr in iter_thresh:
                    est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path)
                    est_path_list.append(est_path_tmp)
            else:
                iter_thresh = [thr] * len(est_path_list)
            if multi_nets==True:
                print('Iterating pipeline for ' + str(ID) + ' across all Yeo 7 networks...')
                network_list = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
                for network in network_list:
                    est_path_tmp = utils.create_est_path(ID, network, conn_model, thr, mask, dir_path)
                    est_path_list.append(est_path_tmp)
            else:
                network_list = [network]* len(est_path_list)    
                
            est_path = est_path_list
            thr = iter_thresh
            network = network_list
            ID = [ID] * len(est_path_list)
            mask = [mask] * len(est_path_list)
            conn_model = [conn_model] * len(est_path_list)
            
        return(est_path, thr, network, ID, mask, conn_model)

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

    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
    node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
    adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
    max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, 
    k_max, k_step, k_clustering):
        wf = pe.Workflow(name='PyNets_' + str(ID))
        wf.base_directory='/tmp/pynets'
        ##Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID',
        'atlas_select', 'network', 'thr', 'node_size', 'mask', 'parlistfile',
        'multi_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch',
        'bedpostx_dir', 'anat_loc', 'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr', 
        'multi_atlas', 'max_thr', 'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 
        'k_max', 'k_step', 'k_clustering']), name='inputnode')

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

        #3) Add variable to function nodes
        ##Create function nodes
        imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select',
        'network', 'node_size', 'mask', 'thr', 'parlistfile', 'multi_nets', 'conn_model',
        'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch', 'bedpostx_dir', 'anat_loc', 
        'parc', 'ref_txt', 'procmem', 'dir_path', 'multi_thr', 'multi_atlas', 'max_thr', 
        'min_thr', 'step_thr', 'k', 'clust_mask', 'k_min', 'k_max', 'k_step', 'k_clustering'],
        output_names = ['est_path', 'thr', 'network', 'ID', 'mask', 'conn_model'], function=workflow_selector),
        name = "imp_est")

        ##Create MapNode types for net_mets_node and export_to_pandas_node
        net_mets_node = pe.MapNode(interface=ExtractNetStats(), name = "ExtractNetStats", 
                                   iterfield=['ID', 'network', 'thr', 'conn_model', 'est_path', 'mask'])
        export_to_pandas_node = pe.MapNode(interface=Export2Pandas(), name = "export_to_pandas", 
                                           iterfield=['in_csv', 'ID', 'network', 'mask'])

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
                                  ('k_clustering', 'k_clustering')]),
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
        ])
        return wf

    def wf_multi_subject(subjects_list, atlas_select, network, node_size, mask,
    thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
    plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr, 
    anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering):
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
                               k_clustering=k_clustering)
            wf_multi.add_nodes([wf_single_subject])
            i = i + 1
        return wf_multi

    ##Workflow generation
    import logging
    from time import gmtime, strftime
    from nipype.utils.profiler import log_nodes_cb
    callback_log_path = '/tmp/run_stats' + '_' + str(ID) + '_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.log'
    logger = logging.getLogger('callback')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(callback_log_path)
    logger.addHandler(handler)

    if subjects_list is not None:
        wf_multi = wf_multi_subject(subjects_list, atlas_select, network, node_size,
        mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh,
        plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr,
        anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering)
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1]), 'status_callback' : log_nodes_cb}
        wf_multi.run()
        print('\n' + 'Running with ' + str(plugin_args) + '\n')
        #wf_multi.run(plugin='MultiProc', plugin_args= plugin_args)
    ##Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
        node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf,
        adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
        max_thr, step_thr, anat_loc, parc, ref_txt, procmem, k, clust_mask, k_min, k_max, k_step, k_clustering)
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1]), 'status_callback' : log_nodes_cb}
        wf.run()
        print('\n' + 'Running with ' + str(plugin_args) + '\n')
        #wf.run(plugin='MultiProc', plugin_args= plugin_args)

    print('\n\n------------NETWORK COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('---------------------------------------')
