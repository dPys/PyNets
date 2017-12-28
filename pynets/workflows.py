# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import warnings
warnings.simplefilter("ignore")
from pynets import nodemaker
from pynets.diffconnectometry import prepare_masks, run_struct_mapping
from pynets import utils, graphestimation, plotting, thresholding
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
import numpy as np

def wb_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list):

    import_list=[ "import sys", "import os", "from sklearn.model_selection import train_test_split",
    "from pynets.utils import export_to_pandas", "import warnings", "import gzip", "import nilearn", "import numpy as np",
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
    "from pynets import nodemaker, thresholding, graphestimation, plotting",
    "from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits",
    "import _pickle as pickle", "from pynets.utils import nilearn_atlas_helper", "import scipy as sp", 
    "import time", "from sklearn.feature_extraction import image",
    "from sklearn.cluster import FeatureAgglomeration", "from pynets.utils import do_dir_path"]

    wb_functional_connectometry_wf = pe.Workflow(name='wb_functional_connectometry')
    wb_functional_connectometry_wf.base_directory='/tmp/pynets'
    
    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 
                                                      'atlas_select', 'network', 
                                                      'node_size', 'mask', 'thr', 
                                                      'parlistfile', 'multi_nets', 
                                                      'conn_model', 'dens_thresh', 
                                                      'conf', 'adapt_thresh', 
                                                      'plot_switch', 'parc', 'ref_txt', 
                                                      'procmem', 'dir_path', 'k', 
                                                      'clust_mask', 'k_min', 'k_max', 
                                                      'k_step', 'k_clustering', 'user_atlas_list']), name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.thr = thr
    inputnode.inputs.parlistfile = parlistfile
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.dens_thresh = dens_thresh
    inputnode.inputs.conf = conf
    inputnode.inputs.adapt_thresh = adapt_thresh
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dir_path = dir_path
    inputnode.inputs.k = k
    inputnode.inputs.clust_mask = clust_mask
    inputnode.inputs.k_min = k_min
    inputnode.inputs.k_max = k_max
    inputnode.inputs.k_step = k_step
    inputnode.inputs.k_clustering = k_clustering
    inputnode.inputs.user_atlas_list = user_atlas_list

    #3) Add variable to function nodes
    ##Create function nodes                               
    clustering_node = pe.Node(niu.Function(input_names = ['func_file', 'clust_mask', 'ID', 'k'], 
                                                          output_names = ['parlistfile', 'atlas_select', 'dir_path'], 
                                                          function=utils.individual_tcorr_clustering, imports = import_list), name = "clustering_node")  
    
    WB_fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names = ['atlas_select', 'parlistfile', 'ref_txt', 'parc', 'func_file'], 
                                                          output_names = ['label_names', 'coords', 'atlas_select', 'networks_list', 'parcel_list', 'par_max', 'parlistfile', 'dir_path'], 
                                                          function=nodemaker.WB_fetch_nodes_and_labels, imports = import_list), name = "WB_fetch_nodes_and_labels_node")    
    
    ##Node generation
    if mask is not None:
        node_gen_node = pe.Node(niu.Function(input_names = ['mask', 'coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=nodemaker.node_gen_masking, imports = import_list), name = "node_gen_masking_node")   
    else:
        node_gen_node = pe.Node(niu.Function(input_names = ['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=nodemaker.node_gen, imports = import_list), name = "node_gen_node")
    
    ##Extract time-series from nodes
    if parc == True:
        save_nifti_parcels_node = pe.Node(niu.Function(input_names = ['ID', 'dir_path', 'mask', 'network', 'net_parcels_map_nifti'],  
                                                     function=utils.save_nifti_parcels_map, imports = import_list), name = "save_nifti_parcels_node")

        ##extract time series from whole brain parcellaions:        
        extract_ts_wb_node = pe.Node(niu.Function(input_names = ['net_parcels_map_nifti', 'conf', 'func_file', 'coords', 'mask', 'dir_path', 'ID', 'network'], 
                                                     output_names=['ts_within_nodes'], 
                                                     function=graphestimation.extract_ts_wb_parc, imports = import_list), name = "extract_ts_wb_parc_node")   
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_wb_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'thr', 'network'], 
                                             output_names=['ts_within_nodes'], 
                                             function=graphestimation.extract_ts_wb_coords, imports = import_list), name = "extract_ts_wb_coords_node")        

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'], 
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'], 
                                         function=thresholding.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold', 'plot_switch'],
                                 output_names='None',
                                 function=plotting.plot_all, imports = import_list), name = "plot_all_node")

    outputnode = pe.Node(niu.Function(function=utils.output_echo, input_names=['est_path', 'thr'], output_names=['est_path', 'thr']), name='outputnode')
   
    if multi_thr == True:
        thresh_and_fit_node_iterables = []
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
        float(max_thr), float(step_thr)),decimals=2).tolist()]
        thresh_and_fit_node_iterables.append(("thr", iter_thresh))
        thresh_and_fit_node.iterables = thresh_and_fit_node_iterables 
    if multi_atlas is not None:
        WB_fetch_nodes_and_labels_node_iterables = []
        atlas_iterables = ("atlas_select", multi_atlas)
        WB_fetch_nodes_and_labels_node_iterables.append(atlas_iterables)
        WB_fetch_nodes_and_labels_node.iterables = WB_fetch_nodes_and_labels_node_iterables
    elif user_atlas_list:
        WB_fetch_nodes_and_labels_node_iterables = []
        WB_fetch_nodes_and_labels_node_iterables.append(("parlistfile", user_atlas_list))
        WB_fetch_nodes_and_labels_node.iterables = WB_fetch_nodes_and_labels_node_iterables
    if k_clustering==2:
        k_cluster_iterables = []
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        k_cluster_iterables.append(("k", k_list))
        clustering_node.iterables = k_cluster_iterables
       
    ##Connect nodes of workflow
    wb_functional_connectometry_wf.connect([
        (inputnode, WB_fetch_nodes_and_labels_node, [('func_file', 'func_file'),
                                                    ('atlas_select', 'atlas_select'),
                                                    ('parlistfile', 'parlistfile'),
                                                    ('parc', 'parc'),
                                                    ('ref_txt', 'ref_txt')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc'),
                                    ('ref_txt', 'ref_txt'),
                                    ('atlas_select', 'atlas_select'),
                                    ('parlistfile', 'parlistfile')]),
        (WB_fetch_nodes_and_labels_node, node_gen_node, [('coords', 'coords'),
                                                        ('label_names', 'label_names'),
                                                        ('dir_path', 'dir_path'),
                                                        ('parcel_list', 'parcel_list'),
                                                        ('par_max', 'par_max'),
                                                        ('networks_list', 'networks_list')]),
        (inputnode, extract_ts_wb_node, [('conf', 'conf'),
                                        ('func_file', 'func_file'),
                                        ('mask', 'mask'),
                                        ('ID', 'ID'),
                                        ('network', 'network'),
                                        ('node_size', 'node_size'),
                                        ('thr', 'thr')]),
        (WB_fetch_nodes_and_labels_node, extract_ts_wb_node, [('dir_path', 'dir_path')]),
        (node_gen_node, extract_ts_wb_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                             ('coords', 'coords')]),
        (inputnode, thresh_and_fit_node, [('adapt_thresh', 'adapt_thresh'),
                                            ('dens_thresh', 'dens_thresh'),
                                            ('thr', 'thr'),
                                            ('ID', 'ID'),
                                            ('mask', 'mask'),
                                            ('network', 'network'),
                                            ('conn_model', 'conn_model')]),
        (WB_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
        (extract_ts_wb_node, thresh_and_fit_node, [('ts_within_nodes', 'ts_within_nodes')]),
        (inputnode, plot_all_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('network', 'network'),
                                    ('conn_model', 'conn_model'),
                                    ('atlas_select', 'atlas_select'),
                                    ('plot_switch', 'plot_switch')]),
        (WB_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path')]),
        (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                              ('coords', 'coords')]),
        (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                              ('edge_threshold', 'edge_threshold')]),        
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])
        
    if k_clustering == 2 or k_clustering == 1:
        wb_functional_connectometry_wf.add_nodes([clustering_node])
        wb_functional_connectometry_wf.disconnect([(inputnode, WB_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                   (inputnode, plot_all_node, [('atlas_select', 'atlas_select')]),
                                                   (WB_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path')]),
                                                   (inputnode, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                   (WB_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path')]),                                                   
                                                   (WB_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                   (WB_fetch_nodes_and_labels_node, extract_ts_wb_node, [('dir_path', 'dir_path')])
                                                   ])
        wb_functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'), ('func_file', 'func_file'), ('clust_mask', 'clust_mask'), ('k', 'k')]),
                                                (clustering_node, WB_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                (clustering_node, plot_all_node, [('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                (clustering_node, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                (clustering_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                (clustering_node, extract_ts_wb_node, [('dir_path', 'dir_path')])
                                                ])
    else:
        wb_functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        wb_functional_connectometry_wf.connect([(inputnode, save_nifti_parcels_node, [('ID', 'ID'),('mask', 'mask')]), 
                                                (inputnode, save_nifti_parcels_node, [('network', 'network')]),
                                                (WB_fetch_nodes_and_labels_node, save_nifti_parcels_node, [('dir_path', 'dir_path')]),
                                                (node_gen_node, save_nifti_parcels_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                                ])
           
    wb_functional_connectometry_wf.config['execution']['crashdump_dir']='/tmp'
    wb_functional_connectometry_wf.config['execution']['remove_unnecessary_outputs']='false'
    wb_functional_connectometry_wf.write_graph()
    res = wb_functional_connectometry_wf.run()

    try:
        thr=list(res.nodes())[-1].result.outputs.thr
        est_path=list(res.nodes())[-1].result.outputs.est_path
    except:
        thr=list(res.nodes())[-2].result.outputs.thr
        est_path=list(res.nodes())[-2].result.outputs.est_path    
    return(est_path, thr)

def RSN_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list):

    import_list=[ "import sys", "import os", "from sklearn.model_selection import train_test_split",
    "from pynets.utils import export_to_pandas", "import warnings", "import gzip", "import nilearn", "import numpy as np",
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
    "from pynets import nodemaker, thresholding, graphestimation, plotting",
    "from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits",
    "import _pickle as pickle", "import pkg_resources", "from pynets.nodemaker import get_sphere", 
    "from pynets.utils import nilearn_atlas_helper", "import scipy as sp", 
    "import time", "from sklearn.feature_extraction import image",
    "from sklearn.cluster import FeatureAgglomeration", "from pynets.utils import do_dir_path"]

    rsn_functional_connectometry_wf = pe.Workflow(name='rsn_functional_connectometry')
    rsn_functional_connectometry_wf.base_directory = '/tmp/pynets'
    
    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 
                                                      'atlas_select', 'network', 
                                                      'node_size', 'mask', 'thr', 
                                                      'parlistfile', 'multi_nets', 
                                                      'conn_model', 'dens_thresh', 
                                                      'conf', 'adapt_thresh', 
                                                      'plot_switch', 'parc', 'ref_txt', 
                                                      'procmem', 'dir_path', 'k', 
                                                      'clust_mask', 'k_min', 'k_max', 
                                                      'k_step', 'k_clustering', 'user_atlas_list']), name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.func_file = func_file
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.thr = thr
    inputnode.inputs.parlistfile = parlistfile
    inputnode.inputs.conn_model = conn_model
    inputnode.inputs.dens_thresh = dens_thresh
    inputnode.inputs.conf = conf
    inputnode.inputs.adapt_thresh = adapt_thresh
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dir_path = dir_path
    inputnode.inputs.k = k
    inputnode.inputs.clust_mask = clust_mask
    inputnode.inputs.k_min = k_min
    inputnode.inputs.k_max = k_max
    inputnode.inputs.k_step = k_step
    inputnode.inputs.k_clustering = k_clustering
    inputnode.inputs.user_atlas_list = user_atlas_list
    
    #3) Add variable to function nodes
    ##Create function nodes
    clustering_node = pe.Node(niu.Function(input_names = ['func_file', 'clust_mask', 'ID', 'k'], 
                                                          output_names = ['parlistfile', 'atlas_select', 'dir_path'], 
                                                          function=utils.individual_tcorr_clustering, imports = import_list), name = "clustering_node")

    RSN_fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names = ['atlas_select', 'parlistfile', 'ref_txt', 'parc', 'func_file'], 
                                                          output_names = ['label_names', 'coords', 'atlas_select', 'networks_list', 'parcel_list', 'par_max', 'parlistfile', 'dir_path'], 
                                                          function=nodemaker.RSN_fetch_nodes_and_labels, imports = import_list), name = "RSN_fetch_nodes_and_labels_node")    

    get_node_membership_node = pe.Node(niu.Function(input_names = ['network', 'func_file', 'coords', 'label_names', 'parc', 'parcel_list'], 
                                                      output_names = ['net_coords', 'net_parcel_list', 'net_label_names', 'network'], 
                                                      function=nodemaker.get_node_membership, imports = import_list), name = "get_node_membership_node")
    
    ##Node generation
    if mask is not None:
        node_gen_node = pe.Node(niu.Function(input_names = ['mask', 'coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=nodemaker.node_gen_masking, imports = import_list), name = "node_gen_masking_node")   
    else:
        node_gen_node = pe.Node(niu.Function(input_names = ['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=nodemaker.node_gen, imports = import_list), name = "node_gen_node")

    save_coords_and_labels_node = pe.Node(niu.Function(input_names = ['coords', 'label_names', 'dir_path', 'network'],  
                                                     function=utils.save_RSN_coords_and_labels_to_pickle, imports = import_list), name = "save_coords_and_labels_node")
    
    ##Extract time-series from nodes
    if parc == True:
        save_nifti_parcels_node = pe.Node(niu.Function(input_names = ['ID', 'dir_path', 'mask', 'network', 'net_parcels_map_nifti'],  
                                                     function=utils.save_nifti_parcels_map, imports = import_list), name = "save_nifti_parcels_node")

        ##extract time series from whole brain parcellaions:        
        extract_ts_rsn_node = pe.Node(niu.Function(input_names = ['net_parcels_map_nifti', 'conf', 'func_file', 'coords', 'mask', 'dir_path', 'ID', 'network'], 
                                                     output_names=['ts_within_nodes'], 
                                                     function=graphestimation.extract_ts_wb_parc, imports = import_list), name = "extract_ts_rsn_parc_node")   
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_rsn_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'thr', 'network'], 
                                             output_names=['ts_within_nodes'], 
                                             function=graphestimation.extract_ts_wb_coords, imports = import_list), name = "extract_ts_rsn_coords_node")        

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'], 
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'], 
                                         function=thresholding.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold', 'plot_switch'],
                                 output_names='None',
                                 function=plotting.plot_all, imports = import_list), name = "plot_all_node")

    outputnode = pe.Node(niu.Function(function=utils.output_echo, input_names=['est_path', 'thr'], output_names=['est_path', 'thr']), name='outputnode')
   
    if multi_thr == True:
        thresh_and_fit_node_iterables = []
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
        float(max_thr), float(step_thr)),decimals=2).tolist()]
        thresh_and_fit_node_iterables.append(("thr", iter_thresh))
        thresh_and_fit_node.iterables = thresh_and_fit_node_iterables 
    if multi_atlas is not None:
        RSN_fetch_nodes_and_labels_node_iterables = []
        atlas_iterables = ("atlas_select", multi_atlas)
        RSN_fetch_nodes_and_labels_node_iterables.append(atlas_iterables)
        RSN_fetch_nodes_and_labels_node.iterables = RSN_fetch_nodes_and_labels_node_iterables
    elif user_atlas_list:
        RSN_fetch_nodes_and_labels_node_iterables = []
        RSN_fetch_nodes_and_labels_node_iterables.append(("parlistfile", user_atlas_list))
        RSN_fetch_nodes_and_labels_node.iterables = RSN_fetch_nodes_and_labels_node_iterables
    if multi_nets is not None:
        get_node_membership_node_iterables = []
        network_iterables = ("network", multi_nets)
        get_node_membership_node_iterables.append(network_iterables)
        get_node_membership_node.iterables = get_node_membership_node_iterables
    if k_clustering==2:
        k_cluster_iterables = []
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        k_cluster_iterables.append(("k", k_list))                
        clustering_node.iterables = k_cluster_iterables
    
    ##Connect nodes of workflow
    rsn_functional_connectometry_wf.connect([
        (inputnode, RSN_fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                    ('parlistfile', 'parlistfile'),
                                                    ('parc', 'parc'),
                                                    ('ref_txt', 'ref_txt'),
                                                    ('func_file', 'func_file')]),
        (inputnode, get_node_membership_node, [('network', 'network'),
                                    ('func_file', 'func_file'),
                                    ('parc', 'parc')]),   
        (RSN_fetch_nodes_and_labels_node, get_node_membership_node, [('coords', 'coords'),
                                                        ('label_names', 'label_names'),
                                                        ('parcel_list', 'parcel_list'),
                                                        ('par_max', 'par_max'),
                                                        ('networks_list', 'networks_list')]),                                                
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc'),
                                    ('ref_txt', 'ref_txt'),
                                    ('atlas_select', 'atlas_select'),
                                    ('parlistfile', 'parlistfile')]),
        (RSN_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path')]),
        (get_node_membership_node, node_gen_node, [('net_coords', 'coords'),
                                                    ('net_label_names', 'label_names'),
                                                    ('net_parcel_list', 'parcel_list')]),
        (get_node_membership_node, save_coords_and_labels_node, [('net_coords', 'coords'),
                                                    ('net_label_names', 'label_names'),
                                                    ('network', 'network')]),
        (RSN_fetch_nodes_and_labels_node, save_coords_and_labels_node, [('dir_path', 'dir_path')]),
        (inputnode, extract_ts_rsn_node, [('conf', 'conf'),
                                        ('func_file', 'func_file'),
                                        ('mask', 'mask'),
                                        ('ID', 'ID'),
                                        ('network', 'network'),
                                        ('node_size', 'node_size'),
                                        ('thr', 'thr')]),
        (RSN_fetch_nodes_and_labels_node, extract_ts_rsn_node, [('dir_path', 'dir_path')]),
        (node_gen_node, extract_ts_rsn_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                             ('coords', 'coords')]),
        (inputnode, thresh_and_fit_node, [('adapt_thresh', 'adapt_thresh'),
                                            ('dens_thresh', 'dens_thresh'),
                                            ('thr', 'thr'),
                                            ('ID', 'ID'),
                                            ('mask', 'mask'),
                                            ('network', 'network'),
                                            ('conn_model', 'conn_model')]),
        (RSN_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
        (extract_ts_rsn_node, thresh_and_fit_node, [('ts_within_nodes', 'ts_within_nodes')]),
        (inputnode, plot_all_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('network', 'network'),
                                    ('conn_model', 'conn_model'),
                                    ('atlas_select', 'atlas_select'),
                                    ('plot_switch', 'plot_switch')]),
        (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                              ('coords', 'coords')]),
        (RSN_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path')]),
        (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                              ('edge_threshold', 'edge_threshold')]),        
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])
 
    if k_clustering == 2 or k_clustering == 1:
        rsn_functional_connectometry_wf.add_nodes([clustering_node])
        rsn_functional_connectometry_wf.disconnect([(inputnode, RSN_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                   (inputnode, plot_all_node, [('atlas_select', 'atlas_select')]),
                                                   (RSN_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path')]),
                                                   (inputnode, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                   (RSN_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path')]),
                                                   (RSN_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                   (RSN_fetch_nodes_and_labels_node, extract_ts_rsn_node, [('dir_path', 'dir_path')])
                                                   ])
        rsn_functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'), ('func_file', 'func_file'), ('clust_mask', 'clust_mask'), ('k', 'k')]),
                                                (clustering_node, RSN_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                (clustering_node, plot_all_node, [('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                (clustering_node, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                (clustering_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                (clustering_node, extract_ts_rsn_node, [('dir_path', 'dir_path')])
                                                ])
    else:
        rsn_functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        rsn_functional_connectometry_wf.connect([(inputnode, save_nifti_parcels_node, [('ID', 'ID'),('mask', 'mask')]), 
                                                (get_node_membership_node, save_nifti_parcels_node, [('network', 'network')]),
                                                (RSN_fetch_nodes_and_labels_node, save_nifti_parcels_node, [('dir_path', 'dir_path')]),
                                                (node_gen_node, save_nifti_parcels_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                                ])
    if multi_nets is not None:
        rsn_functional_connectometry_wf.disconnect([(inputnode, extract_ts_rsn_node, [('network', 'network')]), 
                                                    (inputnode, thresh_and_fit_node, [('network', 'network')]),
                                                    (inputnode, plot_all_node, [('network', 'network')])
                                                    ])
        rsn_functional_connectometry_wf.connect([(get_node_membership_node, extract_ts_rsn_node, [('network', 'network')]), 
                                                (get_node_membership_node, thresh_and_fit_node, [('network', 'network')]),
                                                (get_node_membership_node, plot_all_node, [('network', 'network')])
                                                ])
                                                   
    rsn_functional_connectometry_wf.config['execution']['crashdump_dir']='/tmp'
    rsn_functional_connectometry_wf.config['execution']['remove_unnecessary_outputs']='false'
    rsn_functional_connectometry_wf.write_graph()
    res = rsn_functional_connectometry_wf.run()
    
    try:
        thr=list(res.nodes())[-1].result.outputs.thr
        est_path=list(res.nodes())[-1].result.outputs.est_path
    except:
        thr=list(res.nodes())[-2].result.outputs.thr
        est_path=list(res.nodes())[-2].result.outputs.est_path

    return est_path, thr

def wb_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path, clust_mask):
    
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)
        
    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, coords, label_names] = nodemaker.node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parcels)
    else:
        [net_parcels_map_nifti, coords, label_names] = nodemaker.node_gen(coords, parcel_list, label_names, dir_path, ID, parcels)

    ##Prepare Volumes
    if parcels == True:
        print('\n' + 'Converting 3d atlas image file to 4d image of atlas volume masks...' + '\n')
        volumes_dir = utils.convert_atlas_to_volumes(parlistfile, parcel_list)
        coords=None
    else:
        volumes_dir=None

    ##Prepare seed, avoidance, and waypoint masks
    print('\n' + 'Running node preparation...' + '\n')
    [vent_CSF_diff_mask_path, WM_diff_mask_path] = prepare_masks(ID, bedpostx_dir, network, coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads)

    ##Run all stages of probabilistic structural connectometry
    print('\n' + 'Running probabilistic structural connectometry...' + '\n')
    est_path2 = run_struct_mapping(ID, bedpostx_dir, network, coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads, vent_CSF_diff_mask_path, WM_diff_mask_path)

    return est_path2
            
def RSN_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path, clust_mask):

    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)    
     
    ##Get coord membership dictionary
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, nodif_brain_mask_path, coords, label_names, parcels, parcel_list)

    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, net_coords, net_label_names] = nodemaker.node_gen_masking(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)            
    else:
        [net_parcels_map_nifti, net_coords, net_label_names] = nodemaker.node_gen(net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)
            
    ##Prepare Volumes
    if parcels == True:
        print('\n' + 'Converting 3d atlas image file to 4d image of atlas volume masks...' + '\n')
        volumes_dir = utils.convert_atlas_to_volumes(parlistfile, net_parcel_list)
        net_coords=None
    else:
        volumes_dir=None

    ##Prepare seed, avoidance, and waypoint masks
    print('\n' + 'Running node preparation...' + '\n')
    [vent_CSF_diff_mask_path, WM_diff_mask_path] = prepare_masks(ID, bedpostx_dir, network, net_coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads)

    ##Run all stages of probabilistic structural connectometry
    print('\n' + 'Running probabilistic structural connectometry...' + '\n')
    est_path2 = run_struct_mapping(ID, bedpostx_dir, network, net_coords, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads, vent_CSF_diff_mask_path, WM_diff_mask_path)

    return est_path2