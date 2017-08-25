import sys
import argparse
import os
import timeit

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
        required=True,
        help='Specify either a path to a preprocessed functional image in standard space and in .nii or .nii.gz format OR the path to an 4D time-series text/csv file OR the path of a pre-made graph that has been thresholded and standardized appropriately')
    parser.add_argument('-ID',
        metavar='Subject ID',
        default=None,
        required=True,
        help='A subject ID that is also the name of the directory containing the input file')
    parser.add_argument('-a',
        metavar='Atlas',
        default='coords_power_2011',
        help='Specify a single coordinate atlas parcellation of those availabe in nilearn. Default is coords_power_2011. Available atlases are:\n\natlas_aal \natlas_destrieux_2009 \ncoords_dosenbach_2010 \ncoords_power_2011')
# parser.add_argument('-ma', '--multiatlas',
#     default='All')
    parser.add_argument('-ua',
        metavar='Path to parcellation file',
        default=None,
        help='Path to nifti-formatted parcellation image file')
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
        help='Optionally specify a coordinate-based node radius size. Default is 3 voxels')
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
    args = parser.parse_args()

    ###Set Arguments to global variables###
    input_file=args.i
    ID=args.ID
    atlas_select=args.a
    NETWORK=args.n
    thr=args.thr
    node_size=args.ns
    mask=args.m
    conn_model=args.model
    all_nets=args.an
    parlistfile=args.ua
    dens_thresh=args.dt
    conf=args.confounds
    adapt_thresh=args.at
    plot_switch=args.plt
    multi_atlas=args.ma
    multi_thr=args.mt
    #######################################

    ##Check required inputs for existence, and configure run
    if input_file is None:
        print("Error: You must include a file path to either a standard space functional image in .nii or .nii.gz format or a path to a time-series text/csv file, with the -i flag")
        sys.exit()
    elif not os.path.isfile(input_file):
        print("Error: Input file does not exist.")
        sys.exit()
    if ID is None:
        print("Error: You must include a subject ID in your command line call")
        sys.exit()
    if dens_thresh is not None or adapt_thresh != False:
        thr=None
    else:
        thr=float(thr)
    if multi_thr==True:
        dens_thresh=None
        adapt_thresh=None

    ##Print inputs verbosely
    print("\n\n\n" + "------------------------------------------------------------------------")
    print ("INPUT FILE: " + input_file)
    print("\n")
    print ("SUBJECT ID: " + str(ID))
    print("\n")
    if parlistfile != None:
        atlas_name = parlistfile.split('/')[-1].split('.')[0]
        print ("ATLAS: " + str(atlas_name))
    else:
        print ("ATLAS: " + str(atlas_select))
        print("\n")
    if NETWORK != None:
        print ("NETWORK: " + str(NETWORK))
    else:
        print("USING WHOLE-BRAIN CONNECTOME..." )
    print("-------------------------------------------------------------------------" + "\n\n\n")

    ##Set directory path containing input file
    dir_path = os.path.dirname(os.path.realpath(input_file))

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

    def workflow_selector(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch):
        import pynets
        from pynets import workflows

        nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']

        ##Case 1: Whole-brain connectome with nilearn coord atlas
        if '.nii' in input_file and parlistfile == None and NETWORK == None and atlas_select not in nilearn_atlases:
            est_path = workflows.wb_connectome_with_nl_atlas_coords(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch)

        ##Case 2: Whole-brain connectome with user-specified atlas or nilearn atlas img
        elif '.nii' in input_file and NETWORK == None and (parlistfile != None or atlas_select in nilearn_atlases):
            est_path = workflows.wb_connectome_with_us_atlas_coords(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch)

        ##Case 3: RSN connectome with nilearn atlas or user-specified atlas
        elif '.nii' in input_file and NETWORK != None:
            est_path = workflows.network_connectome(input_file, ID, atlas_select, NETWORK, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch)

        return est_path

    ##Extract network metrics interface
    def extractnetstats(ID, NETWORK, thr, conn_model, est_path1, out_file=None):
        import pynets
        from pynets import netstats, thresholding

        ##Load and threshold matrix
        in_mat = np.array(genfromtxt(est_path1))
        in_mat = thresholding.autofix(in_mat)

        ##Get hyperbolic tangent of matrix if non-sparse (i.e. fischer r-to-z transform)
        if conn_model == 'corr':
            in_mat = np.arctanh(in_mat)

        ##Get dir_path
        dir_path = os.path.dirname(os.path.realpath(est_path1))

        ##Assign Weight matrix
        mat_wei = in_mat
        ##Load numpy matrix as networkx graph
        G=nx.from_numpy_matrix(mat_wei)

        ##Create Binary matrix
        #mat_bin = weight_conversion(in_mat, 'binarize')
        ##Load numpy matrix as networkx graph
        #G_bin=nx.from_numpy_matrix(mat_bin)

        ##Create Length matrix
        mat_len = thresholding.weight_conversion(in_mat, 'lengths')
        ##Load numpy matrix as networkx graph
        G_len=nx.from_numpy_matrix(mat_len)

        ##Save gephi files
        if NETWORK != None:
            nx.write_graphml(G, dir_path + '/' + ID + '_' + NETWORK + '.graphml')
        else:
            nx.write_graphml(G, dir_path + '/' + ID + '.graphml')

        ###############################################################
        ########### Calculate graph metrics from graph G ##############
        ###############################################################
        import random
        import itertools
        from itertools import permutations
        from networkx.algorithms import degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity, betweenness_centrality, rich_club_coefficient, eigenvector_centrality, communicability_centrality
        from pynets.netstats import efficiency, global_efficiency, local_efficiency, create_random_graph, smallworldness_measure, smallworldness, modularity
        ##For non-nodal scalar metrics from networkx.algorithms library, add the name of the function to metric_list for it to be automatically calculated.
        ##For non-nodal scalar metrics from custom functions, add the name of the function to metric_list and add the function  (with a G-only input) to the netstats module.
        metric_list = [global_efficiency, local_efficiency, smallworldness, degree_assortativity_coefficient, average_clustering, average_shortest_path_length, degree_pearson_correlation_coefficient, graph_number_of_cliques, transitivity]

        ##Iteratively run functions from above metric list
        num_mets = len(metric_list)
        net_met_arr = np.zeros([num_mets, 2], dtype='object')
        j=0
        for i in metric_list:
            met_name = str(i).split('<function ')[1].split(' at')[0]
            if NETWORK != None:
                net_met = NETWORK + '_' + met_name
            else:
                net_met = met_name
            try:
                net_met_val = float(i(G))
            except:
                net_met_val = np.nan
            net_met_arr[j,0] = net_met
            net_met_arr[j,1] = net_met_val
            print(net_met)
            print(str(net_met_val))
            print('\n')
            j = j + 1
        net_met_val_list = list(net_met_arr[:,1])

        ##Calculate modularity using the Louvain algorithm
        [community_aff, modularity] = modularity(mat_wei)

        ##betweenness_centrality
        try:
            bc_vector = betweenness_centrality(G_len)
            print('Extracting Betweeness Centrality vector for all network nodes...')
            bc_vals = list(bc_vector.values())
            bc_nodes = list(bc_vector.keys())
            num_nodes = len(bc_nodes)
            bc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j=0
            for i in range(num_nodes):
                if NETWORK != None:
                    bc_arr[j,0] = NETWORK + '_' + str(bc_nodes[j]) + '_betw_cent'
                    print('\n' + NETWORK + '_' + str(bc_nodes[j]) + '_betw_cent')
                else:
                    bc_arr[j,0] = 'WholeBrain_' + str(bc_nodes[j]) + '_betw_cent'
                    print('\n' + 'WholeBrain_' + str(bc_nodes[j]) + '_betw_cent')
                try:
                    bc_arr[j,1] = bc_vals[j]
                except:
                    bc_arr[j,1] = np.nan
                print(str(bc_vals[j]))
                j = j + 1
            bc_val_list = list(bc_arr[:,1])
            bc_arr[num_nodes,0] = NETWORK + '_MEAN_betw_cent'
            nonzero_arr_betw_cent = np.delete(bc_arr[:,1], [0])
            bc_arr[num_nodes,1] = np.mean(nonzero_arr_betw_cent)
            print('\n' + 'Mean Betweenness Centrality across all nodes: ' + str(bc_arr[num_nodes,1]) + '\n')
        except:
            print('Betweeness Centrality calculation failed. Skipping...')
            bc_val_list = []
            pass

        ##eigenvector_centrality
        try:
            ec_vector = eigenvector_centrality(G_len)
            print('Extracting Eigenvector Centrality vector for all network nodes...')
            ec_vals = list(ec_vector.values())
            ec_nodes = list(ec_vector.keys())
            num_nodes = len(ec_nodes)
            ec_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j=0
            for i in range(num_nodes):
                if NETWORK != None:
                    ec_arr[j,0] = NETWORK + '_' + str(ec_nodes[j]) + '_eig_cent'
                    print('\n' + NETWORK + '_' + str(ec_nodes[j]) + '_eig_cent')
                else:
                    ec_arr[j,0] = 'WholeBrain_' + str(ec_nodes[j]) + '_eig_cent'
                    print('\n' + 'WholeBrain_' + str(ec_nodes[j]) + '_eig_cent')
                try:
                    ec_arr[j,1] = ec_vals[j]
                except:
                    ec_arr[j,1] = np.nan
                print(str(ec_vals[j]))
                j = j + 1
            ec_val_list = list(ec_arr[:,1])
            ec_arr[num_nodes,0] = NETWORK + '_MEAN_eig_cent'
            nonzero_arr_eig_cent = np.delete(ec_arr[:,1], [0])
            ec_arr[num_nodes,1] = np.mean(nonzero_arr_eig_cent)
            print('\n' + 'Mean Eigenvector Centrality across all nodes: ' + str(ec_arr[num_nodes,1]) + '\n')
        except:
            print('Eigenvector Centrality calculation failed. Skipping...')
            ec_val_list = []
            pass

        ##communicability_centrality
        try:
            cc_vector = communicability_centrality(G_len)
            print('Extracting Communicability Centrality vector for all network nodes...')
            cc_vals = list(cc_vector.values())
            cc_nodes = list(cc_vector.keys())
            num_nodes = len(cc_nodes)
            cc_arr = np.zeros([num_nodes + 1, 2], dtype='object')
            j=0
            for i in range(num_nodes):
                if NETWORK != None:
                    cc_arr[j,0] = NETWORK + '_' + str(cc_nodes[j]) + '_comm_cent'
                    print('\n' + NETWORK + '_' + str(cc_nodes[j]) + '_comm_cent')
                else:
                    cc_arr[j,0] = 'WholeBrain_' + str(cc_nodes[j]) + '_comm_cent'
                    print('\n' + 'WholeBrain_' + str(cc_nodes[j]) + '_comm_cent')
                try:
                    cc_arr[j,1] = cc_vals[j]
                except:
                    cc_arr[j,1] = np.nan
                print(str(cc_vals[j]))
                j = j + 1
            cc_val_list = list(cc_arr[:,1])
            cc_arr[num_nodes,0] = NETWORK + '_MEAN_comm_cent'
            nonzero_arr_comm_cent = np.delete(cc_arr[:,1], [0])
            cc_arr[num_nodes,1] = np.mean(nonzero_arr_comm_cent)
            print('\n' + 'Mean Communicability Centrality across all nodes: ' + str(cc_arr[num_nodes,1]) + '\n')
        except:
            print('Communicability Centrality calculation failed. Skipping...')
            cc_val_list = []
            pass

        ##rich_club_coefficient
        try:
            rc_vector = rich_club_coefficient(G, normalized=True)
            print('Extracting Rich Club Coefficient vector for all network nodes...')
            rc_vals = list(rc_vector.values())
            rc_edges = list(rc_vector.keys())
            num_edges = len(rc_edges)
            rc_arr = np.zeros([num_edges + 1, 2], dtype='object')
            j=0
            for i in range(num_edges):
                if NETWORK != None:
                    rc_arr[j,0] = NETWORK + '_' + str(rc_edges[j]) + '_rich_club'
                    print('\n' + NETWORK + '_' + str(rc_edges[j]) + '_rich_club')
                else:
                    cc_arr[j,0] = 'WholeBrain_' + str(rc_nodes[j]) + '_rich_club'
                    print('\n' + 'WholeBrain_' + str(rc_nodes[j]) + '_rich_club')
                try:
                    rc_arr[j,1] = rc_vals[j]
                except:
                    rc_arr[j,1] = np.nan
                print(str(rc_vals[j]))
                j = j + 1
            ##Add mean
            rc_val_list = list(rc_arr[:,1])
            rc_arr[num_edges,0] = NETWORK + '_MEAN_rich_club'
            nonzero_arr_rich_club = np.delete(rc_arr[:,1], [0])
            rc_arr[num_edges,1] = np.mean(nonzero_arr_rich_club)
            print('\n' + 'Mean Rich Club Coefficient across all edges: ' + str(rc_arr[num_edges,1]) + '\n')
        except:
            print('Rich Club calculation failed. Skipping...')
            rc_val_list = []
            pass

        ##Create a list of metric names for scalar metrics
        metric_list_names = []
        net_met_val_list_final = net_met_val_list
        for i in net_met_arr[:,0]:
            metric_list_names.append(i)

        ##Add modularity measure
        try:
            if NETWORK != None:
                metric_list_names.append(NETWORK + '_Modularity')
            else:
                metric_list_names.append('WholeBrain_Modularity')
            net_met_val_list_final.append(modularity)
        except:
            pass

        ##Add centrality and rich club measures
        try:
            for i in bc_arr[:,0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(bc_arr[:,1])
        except:
            pass
        try:
            for i in ec_arr[:,0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(ec_arr[:,1])
        except:
            pass
        try:
            for i in cc_arr[:,0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(cc_arr[:,1])
        except:
            pass
        try:
            for i in rc_arr[:,0]:
                metric_list_names.append(i)
            net_met_val_list_final = net_met_val_list_final + list(rc_arr[:,1])
        except:
            pass

        ##Save metric names as pickle
        try:
            import cPickle
        except ImportError:
            import _pickle as cPickle
        if NETWORK != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/met_list_pickle_' + NETWORK
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(est_path1)) + '/met_list_pickle_WB'
        cPickle.dump(metric_list_names, open(met_list_picke_path, 'wb'))

        ##Save results to csv
        if 'inv' in est_path1:
            if NETWORK != None:
                out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_inv_sps_cov_' + str(thr) + '.csv'
            else:
                out_path = dir_path + '/' + ID + '_net_mets_inv_sps_cov_' + str(thr) + '.csv'
        else:
            if NETWORK != None:
                out_path = dir_path + '/' + ID + '_' + NETWORK + '_net_mets_corr_' + str(thr) + '.csv'
            else:
                out_path = dir_path + '/' + ID + '_net_mets_corr_' + str(thr) + '.csv'
        np.savetxt(out_path, net_met_val_list_final)

        return(out_path)
        ###############################################################
        ###############################################################


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
        out_file = csv_loc.replace('.', '')[:-3] + '_' + ID
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

    import_list=[ "import sys", "import os", "from sklearn.model_selection import train_test_split", "import warnings", "import gzip", "import nilearn", "import numpy as np", "import networkx as nx", "import pandas as pd", "import nibabel as nib", "import seaborn as sns", "import numpy.linalg as npl", "import matplotlib", "matplotlib.use('Agg')", "import matplotlib.pyplot as plt", "from numpy import genfromtxt", "from matplotlib import colors", "from nipype import Node, Workflow", "from nilearn import input_data, masking, datasets", "from nilearn import plotting as niplot", "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu", "from nipype.interfaces import io as nio", "from nilearn.input_data import NiftiLabelsMasker", "from nilearn.connectome import ConnectivityMeasure", "from nibabel.affines import apply_affine", "from nipype.interfaces.base import isdefined, Undefined", "from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso", "from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits" ]

    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID', 'atlas_select', 'NETWORK', 'thr', 'node_size', 'mask', 'parlistfile', 'all_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch']), name='inputnode')

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

    #3) Add variable to function nodes
    ##Create function nodes
    imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID', 'atlas_select', 'NETWORK', 'node_size', 'mask', 'thr', 'parlistfile', 'all_nets', 'conn_model', 'dens_thresh', 'conf', 'adapt_thresh', 'plot_switch'], output_names = ['est_path', 'thr'], function=workflow_selector, imports=import_list), name = "imp_est")
    if multi_thr==True:
        print('Iterating pipeline across multiple thresholds...')
        #imp_est.iterables = ("thr", ['0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '0.93', '0.92', '0.91', '0.90'])
    if multi_atlas==True:
        print('Iterating pipeline across multiple atlases...')
        #imp_est.iterables = ("atlas_select", ['coords_power_2011', 'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal'])
    net_mets_node = pe.Node(ExtractNetStats(), name = "ExtractNetStats")
    export_to_pandas_node = pe.Node(Export2Pandas(), name = "export_to_pandas")

    ##Create PyNets workflow
    wf = pe.Workflow(name='PyNets_WORKFLOW')
    wf.base_directory='/tmp/pynets'

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
                              ('plot_switch', 'plot_switch')]),
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

    #wf.run(plugin='SLURM')
    #wf.run(plugin='MultiProc')
    wf.run()

print('Time execution : ', timeit.default_timer() - start_time)
