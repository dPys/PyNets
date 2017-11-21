# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import warnings
warnings.simplefilter("ignore")
from pynets import nodemaker
from pynets.diffconnectometry import prepare_masks, run_struct_mapping
from pynets import utils
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
import numpy as np

def wb_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr):

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
    "import _pickle as pickle", "from pynets.utils import nilearn_atlas_helper"]

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
                                                      'procmem', 'dir_path']), name='inputnode')

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

    #3) Add variable to function nodes
    ##Create function nodes
    WB_fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names = ['atlas_select', 'parlistfile', 'ref_txt', 'parc'], 
                                                          output_names = ['label_names', 'coords', 'atlas_select', 'networks_list', 'parcel_list', 'par_max', 'parlistfile'], 
                                                          function=utils.WB_fetch_nodes_and_labels, imports = import_list), name = "WB_fetch_nodes_and_labels_node")    
    
    ##Node generation
    if mask is not None:
        node_gen_node = pe.Node(niu.Function(input_names = ['mask', 'coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=utils.node_gen_masking, imports = import_list), name = "node_gen_masking_node")   
    else:
        node_gen_node = pe.Node(niu.Function(input_names = ['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=utils.node_gen, imports = import_list), name = "node_gen_node")
    
    ##Extract time-series from nodes
    if parc == True:
        ##extract time series from whole brain parcellaions:        
        extract_ts_wb_node = pe.Node(niu.Function(input_names = ['net_parcels_map_nifti', 'conf', 'func_file', 'coords', 'mask', 'dir_path', 'ID', 'network'], 
                                                     output_names=['ts_within_nodes'], 
                                                     function=utils.extract_ts_wb_parc, imports = import_list), name = "extract_ts_wb_parc_node")   
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_wb_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'thr', 'network'], 
                                             output_names=['ts_within_nodes'], 
                                             function=utils.extract_ts_wb_coords, imports = import_list), name = "extract_ts_wb_coords_node")        

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'], 
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'], 
                                         function=utils.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold', 'plot_switch'],
                                 output_names='None',
                                 function=utils.plot_all, imports = import_list), name = "plot_all_node")

    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr']), name='outputnode')
   
    if multi_thr == True:
        thresh_and_fit_node_iterables = []
        print('Iterating pipeline for ' + str(ID) + ' across multiple thresholds...')
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
        float(max_thr), float(step_thr)),decimals=2).tolist()]
        thresh_and_fit_node_iterables.append(("thr", iter_thresh))
        thresh_and_fit_node.iterables = thresh_and_fit_node_iterables 
    if multi_atlas==True:
        WB_fetch_nodes_and_labels_node_iterables = []
        print('Iterating pipeline for ' + str(ID) + ' across multiple atlases...')
        atlas_iterables = ("atlas_select", ['coords_power_2011',
        'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal'])
        WB_fetch_nodes_and_labels_node_iterables.append(atlas_iterables)
        WB_fetch_nodes_and_labels_node.iterables = WB_fetch_nodes_and_labels_node_iterables

    ##Connect nodes of workflow
    wb_functional_connectometry_wf.connect([
        (inputnode, WB_fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                    ('parlistfile', 'parlistfile'),
                                                    ('parc', 'parc'),
                                                    ('ref_txt', 'ref_txt')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc'),
                                    ('ref_txt', 'ref_txt'),
                                    ('dir_path', 'dir_path'),
                                    ('atlas_select', 'atlas_select'),
                                    ('parlistfile', 'parlistfile')]),
        (WB_fetch_nodes_and_labels_node, node_gen_node, [('coords', 'coords'),
                                                        ('label_names', 'label_names'),
                                                        ('parcel_list', 'parcel_list'),
                                                        ('par_max', 'par_max'),
                                                        ('networks_list', 'networks_list')]),
        (inputnode, extract_ts_wb_node, [('conf', 'conf'),
                                        ('func_file', 'func_file'),
                                        ('mask', 'mask'),
                                        ('dir_path', 'dir_path'),
                                        ('ID', 'ID'),
                                        ('network', 'network'),
                                        ('node_size', 'node_size'),
                                        ('thr', 'thr')]),
        (node_gen_node, extract_ts_wb_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                             ('coords', 'coords')]),
        (inputnode, thresh_and_fit_node, [('adapt_thresh', 'adapt_thresh'),
                                            ('dens_thresh', 'dens_thresh'),
                                            ('thr', 'thr'),
                                            ('ID', 'ID'),
                                            ('dir_path', 'dir_path'),
                                            ('mask', 'mask'),
                                            ('network', 'network'),
                                            ('conn_model', 'conn_model')]),
        (extract_ts_wb_node, thresh_and_fit_node, [('ts_within_nodes', 'ts_within_nodes')]),
        (inputnode, plot_all_node, [('ID', 'ID'),
                                    ('dir_path', 'dir_path'),
                                    ('mask', 'mask'),
                                    ('network', 'network'),
                                    ('conn_model', 'conn_model'),
                                    ('atlas_select', 'atlas_select'),
                                    ('plot_switch', 'plot_switch')]),
        (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                              ('coords', 'coords')]),
        (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                              ('edge_threshold', 'edge_threshold')]),        
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])
    
    wb_functional_connectometry_wf.config['execution']['crashdump_dir']='/tmp'
    wb_functional_connectometry_wf.config['execution']['remove_unnecessary_outputs']='false'
    wb_functional_connectometry_wf.write_graph()
    res = wb_functional_connectometry_wf.run()
    
    thr=list(res.nodes())[3].result.outputs.thr
    est_path=list(res.nodes())[3].result.outputs.est_path
    
    return(est_path, thr)

def RSN_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr):

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
    "from pynets.utils import nilearn_atlas_helper"]

    rsn_functional_connectometry_wf = pe.Workflow(name='rsn_functional_connectometry')
    rsn_functional_connectometry_wf.base_directory='/tmp/pynets'
    
    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['func_file', 'ID', 
                                                      'atlas_select', 'network', 
                                                      'node_size', 'mask', 'thr', 
                                                      'parlistfile', 'multi_nets', 
                                                      'conn_model', 'dens_thresh', 
                                                      'conf', 'adapt_thresh', 
                                                      'plot_switch', 'parc', 'ref_txt', 
                                                      'procmem', 'dir_path']), name='inputnode')

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

    #3) Add variable to function nodes
    ##Create function nodes
    RSN_fetch_nodes_and_labels_node = pe.Node(niu.Function(input_names = ['atlas_select', 'parlistfile', 'ref_txt', 'parc'], 
                                                          output_names = ['label_names', 'coords', 'atlas_select', 'networks_list', 'parcel_list', 'par_max', 'parlistfile'], 
                                                          function=utils.WB_fetch_nodes_and_labels, imports = import_list), name = "RSN_fetch_nodes_and_labels_node")    

    get_node_membership_node = pe.Node(niu.Function(input_names = ['network', 'func_file', 'coords', 'label_names', 'parc', 'parcel_list'], 
                                                      output_names = ['net_coords', 'net_parcel_list', 'net_label_names'], 
                                                      function=nodemaker.get_node_membership, imports = import_list), name = "get_node_membership_node")
    
    ##Node generation
    if mask is not None:
        node_gen_node = pe.Node(niu.Function(input_names = ['mask', 'coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=utils.node_gen_masking, imports = import_list), name = "node_gen_masking_node")   
    else:
        node_gen_node = pe.Node(niu.Function(input_names = ['coords', 'parcel_list', 'label_names', 'dir_path', 'ID', 'parc'], 
                                                     output_names=['net_parcels_map_nifti', 'coords', 'label_names'], 
                                                     function=utils.node_gen, imports = import_list), name = "node_gen_node")
    
    ##Extract time-series from nodes
    if parc == True:
        ##extract time series from whole brain parcellaions:        
        extract_ts_rsn_node = pe.Node(niu.Function(input_names = ['net_parcels_map_nifti', 'conf', 'func_file', 'coords', 'mask', 'dir_path', 'ID', 'network'], 
                                                     output_names=['ts_within_nodes'], 
                                                     function=utils.extract_ts_wb_parc, imports = import_list), name = "extract_ts_wb_parc_node")   
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_rsn_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'thr', 'network'], 
                                             output_names=['ts_within_nodes'], 
                                             function=utils.extract_ts_wb_coords, imports = import_list), name = "extract_ts_wb_coords_node")        

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'], 
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'], 
                                         function=utils.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold', 'plot_switch'],
                                 output_names='None',
                                 function=utils.plot_all, imports = import_list), name = "plot_all_node")

    outputnode = pe.Node(niu.IdentityInterface(fields=['est_path', 'thr']), name='outputnode')
   
    if multi_thr == True:
        thresh_and_fit_node_iterables = []
        print('Iterating pipeline for ' + str(ID) + ' across multiple thresholds...')
        iter_thresh = [str(i) for i in np.round(np.arange(float(min_thr),
        float(max_thr), float(step_thr)),decimals=2).tolist()]
        thresh_and_fit_node_iterables.append(("thr", iter_thresh))
        thresh_and_fit_node.iterables = thresh_and_fit_node_iterables 
    if multi_atlas==True:
        RSN_fetch_nodes_and_labels_node_iterables = []
        print('Iterating pipeline for ' + str(ID) + ' across multiple atlases...')
        atlas_iterables = ("atlas_select", ['coords_power_2011',
        'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal'])
        RSN_fetch_nodes_and_labels_node_iterables.append(atlas_iterables)
        RSN_fetch_nodes_and_labels_node.iterables = RSN_fetch_nodes_and_labels_node_iterables
    if multi_nets==True:
        get_node_membership_node_iterables = []
        print('Iterating pipeline for ' + str(ID) + ' across all Yeo 7 networks...')
        network_iterables = ("network", ['SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'])
        get_node_membership_node_iterables.append(network_iterables)
        get_node_membership_node.iterables = get_node_membership_node_iterables

    ##Connect nodes of workflow
    rsn_functional_connectometry_wf.connect([
        (inputnode, RSN_fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                    ('parlistfile', 'parlistfile'),
                                                    ('parc', 'parc'),
                                                    ('ref_txt', 'ref_txt')]),
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
                                    ('dir_path', 'dir_path'),
                                    ('atlas_select', 'atlas_select'),
                                    ('parlistfile', 'parlistfile')]),
        (get_node_membership_node, node_gen_node, [('net_coords', 'coords'),
                                                    ('net_label_names', 'label_names'),
                                                    ('net_parcel_list', 'parcel_list')]),
        (inputnode, extract_ts_rsn_node, [('conf', 'conf'),
                                        ('func_file', 'func_file'),
                                        ('mask', 'mask'),
                                        ('dir_path', 'dir_path'),
                                        ('ID', 'ID'),
                                        ('network', 'network'),
                                        ('node_size', 'node_size'),
                                        ('thr', 'thr')]),
        (node_gen_node, extract_ts_rsn_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                             ('coords', 'coords')]),
        (inputnode, thresh_and_fit_node, [('adapt_thresh', 'adapt_thresh'),
                                            ('dens_thresh', 'dens_thresh'),
                                            ('thr', 'thr'),
                                            ('ID', 'ID'),
                                            ('dir_path', 'dir_path'),
                                            ('mask', 'mask'),
                                            ('network', 'network'),
                                            ('conn_model', 'conn_model')]),
        (extract_ts_rsn_node, thresh_and_fit_node, [('ts_within_nodes', 'ts_within_nodes')]),
        (inputnode, plot_all_node, [('ID', 'ID'),
                                    ('dir_path', 'dir_path'),
                                    ('mask', 'mask'),
                                    ('network', 'network'),
                                    ('conn_model', 'conn_model'),
                                    ('atlas_select', 'atlas_select'),
                                    ('plot_switch', 'plot_switch')]),
        (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                              ('coords', 'coords')]),
        (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                              ('edge_threshold', 'edge_threshold')]),        
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])
    
    rsn_functional_connectometry_wf.config['execution']['crashdump_dir']='/tmp'
    rsn_functional_connectometry_wf.config['execution']['remove_unnecessary_outputs']='false'
    rsn_functional_connectometry_wf.write_graph()
    res = rsn_functional_connectometry_wf.run()
    
    thr=list(res.nodes())[4].result.outputs.thr
    est_path=list(res.nodes())[4].result.outputs.est_path

    return est_path, thr

def wb_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path):
    
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)
        
    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parcels)
    else:
        [net_parcels_map_nifti, coords, label_names] = utils.node_gen(coords, parcel_list, label_names, dir_path, ID, parcels)

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
            
def RSN_structural_connectometry(ID, bedpostx_dir, network, node_size, atlas_select, parlistfile, label_names, plot_switch, parcels, dict_df, anat_loc, ref_txt, threads, mask, dir_path):

    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    
    [label_names, coords, atlas_select, networks_list, parcel_list, par_max, parlistfile] = utils.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parcels)    
     
    ##Get coord membership dictionary
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, nodif_brain_mask_path, coords, label_names, parcels, parcel_list)

    ##Node generation
    if mask is not None:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen_masking(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)            
    else:
        [net_parcels_map_nifti, net_coords, net_label_names] = utils.node_gen(net_coords, net_parcel_list, net_label_names, dir_path, ID, parcels)
            
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