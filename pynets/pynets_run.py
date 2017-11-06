# -*- coding: utf-8 -*-
import sys
import argparse
import os
import timeit
import string

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
        default='3',
        help='Optionally specify a coordinate-based node radius size. Default is 3 voxels')
    parser.add_argument('-m',
        metavar='path to mask image',
        default=None,
        help='Optionally specify a thresholded inverse-binarized mask image such as a group ICA-derived network volume, to retain only those network nodes contained within that mask')
    parser.add_argument('-an',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting designations and network statistic extraction for all Yeo RSNs in the specified atlas')
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
    parser.add_argument('-at',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate adaptive thresholding')
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
    all_nets=args.an
    conn_model=args.model
    conf=args.confounds
    dens_thresh=args.dt
    adapt_thresh=args.at
    plot_switch=args.plt
    multi_atlas=args.ma
    multi_thr=args.mt
    bedpostx_dir=args.bpx
    min_thr=args.min_thr
    max_thr=args.max_thr
    step_thr=args.step_thr
    anat_loc=args.anat
    parc=args.parc

    print('Starting up!')

    ##Check required inputs for existence, and configure run
    if input_file.endswith('.txt'):
        with open(input_file) as f:
            subjects_list = f.read().splitlines()
    else:
        subjects_list = None

    try:
        bedpostx_dir
    except:
        bedpostx_dir = None

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

    try:
        anat_loc
    except:
        anat_loc = None

    if anat_loc is not None and bedpostx_dir is None:
        print('Warning: anatomical image specified, but not bedpostx directory specified. Anatomical images are only supported for structural connectome estimation at this time.')

    if multi_thr==True:
        dens_thresh=None
        adapt_thresh=False
    else:
        min_thr=None
        max_thr=None
        step_thr=None

    ##Print inputs verbosely and set dir_path
    print("\n\n\n" + "------------------------------------------------------------------------" + "\n")
    if input_file is not None and subjects_list is not None:
        print("\n")
        print('Running workflow of workflows across subjects:\n')
        print (str(subjects_list))
    elif input_file is not None and bedpostx_dir is not None and subjects_list is None:
        print('Running joint structural-functional connectometry...')
        print ("INPUT FILE: " + input_file)
        print ("Bedpostx Directory: " + bedpostx_dir)
        if anat_loc is not None:
            print ("Anatomical Image: " + anat_loc)
        ##Set directory path containing input file
        dir_path = os.path.dirname(os.path.realpath(input_file))
    elif input_file is not None and bedpostx_dir is None and subjects_list is None:
        print('Running functional connectometry only...')
        print ("INPUT FILE: " + input_file)
        ##Set directory path containing input file
        dir_path = os.path.dirname(os.path.realpath(input_file))

    print ("SUBJECT ID: " + str(ID))

    if parlistfile != None:
        atlas_name = parlistfile.split('/')[-1].split('.')[0]
        print ("ATLAS: " + str(atlas_name))
    else:
        print ("ATLAS: " + str(atlas_select))

    if multi_atlas == True:
        atlas_select = None
        print('\nIterating across multiple atlases...\n')

    if network != None:
        print ("NETWORK: " + str(network))
    else:
        print("USING WHOLE-BRAIN CONNECTOME..." )
    print("\n" + "-------------------------------------------------------------------------" + "\n\n\n")

    ##Import core modules
    import nilearn
    import numpy as np
    import networkx as nx
    import pandas as pd
    import nibabel as nib
    import seaborn as sns
    import numpy.linalg as npl
    import matplotlib
    import sklearn
    import matplotlib
    import warnings
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    from numpy import genfromtxt
    from matplotlib import colors
    from nipype import Node, Workflow
    from nilearn import input_data, masking, datasets
    from nilearn import plotting as niplot
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces import io as nio
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure
    from nibabel.affines import apply_affine
    from nipype.interfaces.base import isdefined, Undefined
    from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso
    from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits


    if basc == True:
       from pynets import basc_run
       from pathlib import Path
       basc_config=Path(__file__)/'basc_config.yaml'

       print("\n\n\n-------------<(^.^<) STARTING BASC <(^.^<)----------------------" + "\n\n\n")

       basc_run.basc_runner(subjects_list, basc_config)
       parlistfile=Path(__file__)/'pynets'/'rsnrefs'/'group_stability_clusters.nii.gz'

    def workflow_selector(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc):
        import pynets
        from pynets import workflows

        nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']

        ##Case 1: Whole-brain connectome with user-specified atlas or nilearn atlas img
        if network == None and (parlistfile != None or atlas_select in nilearn_atlases):
            [est_path, thr] = workflows.wb_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc)

        ##Case 2: RSN connectome with nilearn atlas or user-specified atlas
        elif network != None:
            [est_path, thr] = workflows.network_functional_connectometry(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc, parc)

        return est_path, thr

    class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
        sub_id = traits.Str(mandatory=True)
        network = traits.Any(mandatory=True)
        thr = traits.Any(mandatory=True)
        conn_model = traits.Str(mandatory=True)
        est_path1 = File(exists=True, mandatory=True, desc="")

    class ExtractNetStatsOutputSpec(TraitedSpec):
        out_file = File()

    class ExtractNetStats(BaseInterface):
        input_spec = ExtractNetStatsInputSpec
        output_spec = ExtractNetStatsOutputSpec

        def _run_interface(self, runtime):
            from pynets.netstats import extractnetstats
            out = extractnetstats(
                self.inputs.sub_id,
                self.inputs.network,
                self.inputs.thr,
                self.inputs.conn_model,
                self.inputs.est_path1)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(getattr(self, '_outpath'))}

    ##save net metric files to pandas dataframes interface
    def export_to_pandas(csv_loc, ID, network, out_file=None):
        try:
            import cPickle
        except ImportError:
            import _pickle as cPickle
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/met_list_pickle_' + network
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/met_list_pickle_WB'
        metric_list_names = cPickle.load(open(met_list_picke_path, 'rb'))
        df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
        df = df.T
        column_headers={k: v for k, v in enumerate(metric_list_names)}
        df = df.rename(columns=column_headers)
        df['id'] = range(1, len(df) + 1)
        cols = df.columns.tolist()
        ix = cols.index('id')
        cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
        df = df[cols_ID]
        df['id'] = df['id'].astype('object')
        df['id'].values[0] = ID
        out_file = csv_loc.split('.csv')[0]
        df.to_pickle(out_file)
        return(out_file)

    class Export2PandasInputSpec(BaseInterfaceInputSpec):
        in_csv = File(exists=True, mandatory=True, desc="")
        sub_id = traits.Str(mandatory=True)
        network = traits.Any(mandatory=True)
        out_file = File('output_export2pandas.csv', usedefault=True)

    class Export2PandasOutputSpec(TraitedSpec):
        out_file = File()

    class Export2Pandas(BaseInterface):
        input_spec = Export2PandasInputSpec
        output_spec = Export2PandasOutputSpec

        def _run_interface(self, runtime):
            export_to_pandas(
                self.inputs.in_csv,
                self.inputs.sub_id,
                self.inputs.network,
                out_file=self.inputs.out_file)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(self.inputs.out_file)}

    import_list=[ "import sys", "import os", "from sklearn.model_selection import train_test_split",
    "import warnings", "import gzip", "import nilearn", "import numpy as np",
    "import networkx as nx", "import pandas as pd", "import nibabel as nib",
    "import seaborn as sns", "import numpy.linalg as npl", "import matplotlib",
    "matplotlib.use('Agg')", "import matplotlib.pyplot as plt", "from numpy import genfromtxt",
    "from matplotlib import colors", "from nipype import Node, Workflow",
    "from nilearn import input_data, masking, datasets", "from nilearn import plotting as niplot",
    "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu",
    "from nipype.interfaces import io as nio", "from nilearn.input_data import NiftiLabelsMasker",
    "from nilearn.connectome import ConnectivityMeasure", "from nibabel.affines import apply_affine",
    "from nipype.interfaces.base import isdefined, Undefined",
    "from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso",
    "from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits" ]

    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
    node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf,
    adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
    max_thr, step_thr, anat_loc, parc):
        wf = pe.Workflow(name='PyNets_' + str(ID))
        #wf.base_directory='/tmp/pynets'
        ##Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID',
        'atlas_select', 'network', 'thr', 'node_size', 'mask', 'parlistfile',
        'all_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch',
        'bedpostx_dir', 'anat_loc', 'parc']), name='inputnode')

        #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
        inputnode.inputs.in_file = input_file
        inputnode.inputs.ID = ID
        inputnode.inputs.atlas_select = atlas_select
        inputnode.inputs.network = network
        inputnode.inputs.thr = thr
        inputnode.inputs.node_size = node_size
        inputnode.inputs.mask = mask
        inputnode.inputs.parlistfile = parlistfile
        inputnode.inputs.all_nets = all_nets
        inputnode.inputs.conn_model = conn_model
        inputnode.inputs.dens_thresh = dens_thresh
        inputnode.inputs.conf = conf
        inputnode.inputs.adapt_thresh = adapt_thresh
        inputnode.inputs.plot_switch = plot_switch
        inputnode.inputs.bedpostx_dir = bedpostx_dir
        inputnode.inputs.anat_loc = anat_loc
        inputnode.inputs.parc = parc

        #3) Add variable to function nodes
        ##Create function nodes
        imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select',
        'network', 'node_size', 'mask', 'thr', 'parlistfile', 'all_nets', 'conn_model',
        'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch', 'bedpostx_dir', 'anat_loc', 'parc'],
        output_names = ['est_path', 'thr'], function=workflow_selector, imports=import_list),
        name = "imp_est")

        imp_est_iterables=[]
        if multi_thr==True:
            print('Iterating pipeline for ' + str(ID) + ' across multiple thresholds...')
            iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
            float(max_thr), float(step_thr)),decimals=2).tolist()]
            imp_est_iterables.append(("thr", iter_thresh))
        if multi_atlas==True:
            print('Iterating pipeline for ' + str(ID) + ' across multiple atlases...')
            atlas_iterables = ("atlas_select", ['coords_power_2011',
            'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal'])
            imp_est_iterables.append(atlas_iterables)

        if imp_est_iterables:
            imp_est.iterables = imp_est_iterables

        net_mets_node = pe.Node(ExtractNetStats(), name = "ExtractNetStats")
        export_to_pandas_node = pe.Node(Export2Pandas(), name = "export_to_pandas")

        ##Create data sink
        #datasink = pe.Node(nio.DataSink(), name='sinker')
        #datasink.inputs.base_directory = dir_path + '/DataSink'

        ##Add variable to workflow
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
                                  ('all_nets', 'all_nets'),
                                  ('conn_model', 'conn_model'),
                                  ('dens_thresh', 'dens_thresh'),
                                  ('conf', 'conf'),
                                  ('adapt_thresh', 'adapt_thresh'),
                                  ('plot_switch', 'plot_switch'),
                                  ('bedpostx_dir', 'bedpostx_dir'),
                                  ('anat_loc', 'anat_loc'),
                                  ('parc', 'parc')]),
            (inputnode, net_mets_node, [('ID', 'sub_id'),
                                       ('network', 'network'),
                                       ('conn_model', 'conn_model')]),
            (imp_est, net_mets_node, [('est_path', 'est_path1'),
                                      ('thr', 'thr')]),
            #(net_mets_cov_node, datasink, [('est_path', 'csv_loc')]),
            (inputnode, export_to_pandas_node, [('ID', 'sub_id'),
                                            ('network', 'network')]),
            (net_mets_node, export_to_pandas_node, [('out_file', 'in_csv')]),
            #(export_to_pandas1, datasink, [('out_file', 'pandas_df)]),
        ])
        return wf

    def wf_multi_subject(subjects_list, atlas_select, network, node_size, mask,
    thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh,
    plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc):
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
                               all_nets=all_nets,
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
                               parc=parc)
            wf_multi.add_nodes([wf_single_subject])
            i = i + 1
        return wf_multi

    ##Multi-subject workflow generator
    if subjects_list is not None:
        wf_multi = wf_multi_subject(subjects_list, atlas_select, network, node_size,
        mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh,
        plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr, anat_loc, parc)
        plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1])}
        wf_multi.run()
        #print('\n' + 'Running with ' + str(plugin_args) + '\n')
        #wf_multi.run(plugin='MultiProc', plugin_args= plugin_args)
    ##Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, network,
        node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf,
        adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
        max_thr, step_thr, anat_loc, parc)
        #plugin_args = { 'n_procs': int(procmem[0]),'memory_gb': int(procmem[1])}
        wf.run()
        #wf.run(plugin='MultiProc', plugin_args= plugin_args)


    print('-----------PYNETS COMPLETE-----------')
    print('Execution Time: ', timeit.default_timer() - start_time)
    print('-------------- o自自o ----------------')
