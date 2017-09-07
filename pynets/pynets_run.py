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
        help='Specify a single coordinate atlas parcellation of those availabe in nilearn. Default is coords_power_2011. Available atlases are:\n\natlas_aal \natlas_destrieux_2009 \ncoords_dosenbach_2010 \ncoords_power_2011')
    parser.add_argument('-basc',
       default=False,
       action='store_true',
       help='Specify whether you want to run BASC to calculate a group level set of nodes')
    parser.add_argument('-ua',
        metavar='Path to parcellation file',
        default=None,
        help='Path to nifti-formatted parcellation image file')
    parser.add_argument('-pm',
        metavar='Number of Cores and GB of Memory',
        default=[2,4],
        help='Number of cores to use, number of GB of memory to use, please enter as two integer seperated by a comma')
    parser.add_argument('-n',
        metavar='RSN',
        default=None,
        help='Optionally specify an atlas-defined network acronym from the following list of RSNs:\n\nDMN Default Mode\nFPTC Fronto-Parietal Task Control\nDA Dorsal Attention\nSN Salience\nVA Ventral Attention\nCOT Cingular-Opercular')
    parser.add_argument('-thr',
        metavar='Graph threshold',
        default='0.95',
        help='Optionally specify a threshold indicating a proportion of weights to preserve in the graph. Default is 0.95')
    parser.add_argument('-ns',
        metavar='Node size',
        default='3',
        help='Optionally specify a coordinate-based node radius size. Default is 4 voxels')
    parser.add_argument('-m',
        metavar='Path to mask image',
        default=None,
        help='Optionally specify a thresholded inverse-binarized mask image such as a group ICA-derived network volume, to retain only those network nodes contained within that mask')
    parser.add_argument('-an',
        default=False,
        action='store_true',
        help='Optionally use this flag if you wish to activate plotting designations and network statistic extraction for all Yeo RSNs in the specified atlas')
    parser.add_argument('-model',
        metavar='Connectivity',
        default='corr',
        help='Optionally specify matrix estimation type: corr, cov, or sps for correlation, covariance, or sparse-inverse covariance, respectively')
    parser.add_argument('-confounds',
        metavar='Confounds',
        default=None,
        help='Optionally specify a path to a confound regressor file to improve in the signal estimation for the graph')
    parser.add_argument('-dt',
        metavar='Density threshold',
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
        metavar='Path to bedpostx directory',
        default=None,
        help='Formatted according to the FSL default tree structure found at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide#BEDPOSTX')
    parser.add_argument('-bids',
        default=False,
        action='store_true',
        help='Initiate bids automation')
    parser.add_argument('-min_thr',
        metavar='Minimum threshold',
        default=0.90,
        help='Minimum threshold for multi-thresholding. Default is 0.90.')
    parser.add_argument('-max_thr',
        metavar='Maximum threshold',
        default=0.99,
        help='Maximum threshold for multi-thresholding. Default is 0.99.')
    parser.add_argument('-step_thr',
        metavar='Multithresholding step',
        default=0.01,
        help='Threshold step value for multi-thresholding. Default is 0.01.')
    args = parser.parse_args()

    ###Set Arguments to global variables###
    input_file=args.i
    ID=args.ID
    atlas_select=args.a
    basc=args.basc
    parlistfile=args.ua
    pm0=args.pm.split(',')
    procmem=list(pm0)
    NETWORK=args.n
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
    bids=args.bids
    min_thr=args.min_thr
    max_thr=args.max_thr
    step_thr=args.step_thr
    import pdb
    pdb.set_trace()
    #######################################

    ##Check required inputs for existence, and configure run
    if input_file.endswith('.txt'):
        with open(input_file) as f:
            subjects_list = f.read().splitlines()
    else:
        subjects_list = None

    if input_file is None and bedpostx_dir is None:
        print("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format or a path to a time-series text/csv file, with the -i flag")
        sys.exit()

    if ID is None and subjects_list is None:
        print("Error: You must include a subject ID in your command line call")
        sys.exit()

    if basc == True:
       from pynets import basc_run
       basc_run(subjects_list, basc_config)

    if dens_thresh is not None or adapt_thresh != False:
        thr=None
    else:
        thr=float(thr)

    if multi_thr==True:
        dens_thresh=None
        adapt_thresh=False
    else:
        min_thr=None
        max_thr=None
        step_thr=None

    ##Print inputs verbosely and set dir_path
    print("\n\n\n" + "------------------------------------------------------------------------")
    if input_file is not None and subjects_list is not None:
        print("\n")
        print('Running workflow of workflows across subjects:\n')
        print (str(subjects_list))
        print("\n")
    elif input_file is not None and bedpostx_dir is not None and subjects_list is None:
        print("\n")
        print('Running joint structural-functional connectometry...')
        print ("INPUT FILE: " + input_file)
        print("\n")
        ##Set directory path containing input file
        dir_path = os.path.dirname(os.path.realpath(input_file))
    elif input_file is not None and bedpostx_dir is None and subjects_list is None:
        print("\n")
        print('Running functional connectometry only...')
        print ("INPUT FILE: " + input_file)
        print("\n")
        ##Set directory path containing input file
        dir_path = os.path.dirname(os.path.realpath(input_file))

    print("\n")
    print ("SUBJECT ID: " + str(ID))
    print("\n")

    if parlistfile != None:
        atlas_name = parlistfile.split('/')[-1].split('.')[0]
        print ("ATLAS: " + str(atlas_name))
    else:
        print ("ATLAS: " + str(atlas_select))
        print("\n")

    if multi_atlas == True:
        atlas_select = None
        print('\nIterating across multiple atlases...\n')

    if NETWORK != None:
        print ("NETWORK: " + str(NETWORK))
    else:
        print("USING WHOLE-BRAIN CONNECTOME..." )
    print("-------------------------------------------------------------------------" + "\n\n\n")

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

    def workflow_selector(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir):
        import pynets
        from pynets import workflows

        nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']


        ##Case 1: Whole-brain connectome with nilearn coord atlas
        if parlistfile == None and NETWORK == None and atlas_select not in nilearn_atlases:
            [est_path, thr] = workflows.wb_connectome_with_nl_atlas_coords(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir)

        ##Case 2: Whole-brain connectome with user-specified atlas or nilearn atlas img
        elif NETWORK == None and (parlistfile != None or atlas_select in nilearn_atlases):
            [est_path, thr] = workflows.wb_connectome_with_us_atlas_coords(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir)

        ##Case 3: RSN connectome with nilearn atlas or user-specified atlas
        elif NETWORK != None:
            [est_path, thr] = workflows.network_connectome(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir)

        return est_path, thr

    class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
        sub_id = traits.Str(mandatory=True)
        NETWORK = traits.Any(mandatory=True)
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
                self.inputs.NETWORK,
                self.inputs.thr,
                self.inputs.conn_model,
                self.inputs.est_path1)
            setattr(self, '_outpath', out)
            return runtime

        def _list_outputs(self):
            import os.path as op
            return {'out_file': op.abspath(getattr(self, '_outpath'))}

    ##save net metric files to pandas dataframes interface
    def export_to_pandas(csv_loc, ID, NETWORK, out_file=None):
        try:
            import cPickle
        except ImportError:
            import _pickle as cPickle
        if NETWORK != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/met_list_pickle_' + NETWORK
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
        NETWORK = traits.Any(mandatory=True)
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
                self.inputs.NETWORK,
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

    def init_wf_single_subject(ID, input_file, dir_path, atlas_select, NETWORK,
    node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf,
    adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
    max_thr, step_thr):
        wf = pe.Workflow(name='PyNets_' + str(ID))
        #wf.base_directory='/tmp/pynets'
        ##Create input/output nodes
        #1) Add variable to IdentityInterface if user-set
        inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID',
        'atlas_select', 'NETWORK', 'thr', 'node_size', 'mask', 'parlistfile',
        'all_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch',
        'bedpostx_dir']), name='inputnode')

        #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
        inputnode.inputs.in_file = input_file
        inputnode.inputs.ID = ID
        inputnode.inputs.atlas_select = atlas_select
        inputnode.inputs.NETWORK = NETWORK
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

        #3) Add variable to function nodes
        ##Create function nodes
        imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select',
        'NETWORK', 'node_size', 'mask', 'thr', 'parlistfile', 'all_nets', 'conn_model',
        'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch', 'bedpostx_dir'],
        output_names = ['est_path', 'thr'], function=workflow_selector, imports=import_list),
        name = "imp_est")

        imp_est_iterables=[]
        if multi_thr==True:
            print('Iterating pipeline across multiple thresholds...')
            iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
            float(max_thr), float(step_thr)),decimals=2).tolist()]
            imp_est_iterables.append(("thr", iter_thresh))
        if multi_atlas==True:
            print('Iterating pipeline across multiple atlases...')
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
                                  ('NETWORK', 'NETWORK'),
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
                                  ('bedpostx_dir', 'bedpostx_dir')]),
            (inputnode, net_mets_node, [('ID', 'sub_id'),
                                       ('NETWORK', 'NETWORK'),
                                       ('conn_model', 'conn_model')]),
            (imp_est, net_mets_node, [('est_path', 'est_path1'),
                                      ('thr', 'thr')]),
            #(net_mets_cov_node, datasink, [('est_path', 'csv_loc')]),
            (inputnode, export_to_pandas_node, [('ID', 'sub_id'),
                                            ('NETWORK', 'NETWORK')]),
            (net_mets_node, export_to_pandas_node, [('out_file', 'in_csv')]),
            #(export_to_pandas1, datasink, [('out_file', 'pandas_df)]),
        ])
        return wf

    def wf_multi_subject(subjects_list, atlas_select, NETWORK, node_size, mask,
    thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh,
    plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr):
        wf_multi = pe.Workflow(name='PyNets_multisubject')
        i=0
        for _file in subjects_list:
            wf_single_subject = init_wf_single_subject(ID=os.path.dirname(os.path.realpath(subjects_list[i])).split('/')[-1],
                               input_file=_file,
                               dir_path=os.path.dirname(os.path.realpath(subjects_list[i])),
                               atlas_select=atlas_select,
                               NETWORK=NETWORK,
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
                               step_thr=step_thr)
            wf_multi.add_nodes([wf_single_subject])
            i = i + 1
        return wf_multi

    ##Multi-subject workflow generator
    if subjects_list is not None:
        wf_multi = wf_multi_subject(subjects_list, atlas_select, NETWORK, node_size,
        mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh,
        plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr, max_thr, step_thr)
        #wf_multi.run(plugin='MultiProc')
        plugin_args = { 'n_procs' : int(procmem[0]),'memory_gb': int(procmem[1])}
        wf_multi.run(plugin='MultiProc', plugin_args= plugin_args)
    ##Single-subject workflow generator
    else:
        wf = init_wf_single_subject(ID, input_file, dir_path, atlas_select, NETWORK,
        node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf,
        adapt_thresh, plot_switch, bedpostx_dir, multi_thr, multi_atlas, min_thr,
        max_thr, step_thr)
        #wf.run(plugin='MultiProc')
        plugin_args = { 'n_procs' : int(procmem[0]),'memory_gb': int(procmem[1])}
        wf.run(plugin='MultiProc', plugin_args= plugin_args)

    print('Time execution : ', timeit.default_timer() - start_time)
