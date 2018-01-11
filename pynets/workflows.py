# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import numpy as np

def wb_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, utils, graphestimation, plotting, thresholding
    import_list=[ "import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]

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
                                                     function=graphestimation.extract_ts_parc, imports = import_list), name = "extract_ts_wb_parc_node")
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_wb_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'network'],
                                             output_names=['ts_within_nodes'],
                                             function=graphestimation.extract_ts_coords, imports = import_list), name = "extract_ts_wb_coords_node")

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'],
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'],
                                         function=thresholding.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    if plot_switch == True:
        plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold'],
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
    elif user_atlas_list is not None:
        WB_fetch_nodes_and_labels_node_iterables = []
        WB_fetch_nodes_and_labels_node_iterables.append(("parlistfile", user_atlas_list))
        WB_fetch_nodes_and_labels_node.iterables = WB_fetch_nodes_and_labels_node_iterables
    if k_clustering==2:
        k_cluster_iterables = []
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        k_cluster_iterables.append(("k", k_list))
        clustering_node.iterables = k_cluster_iterables
    elif k_clustering==3:
        k_cluster_iterables = []
        k_cluster_iterables.append(("clust_mask", clust_mask_list))
        clustering_node.iterables = k_cluster_iterables
    elif k_clustering==4:
        k_cluster_iterables = []
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        k_cluster_iterables.append(("k", k_list))
        k_cluster_iterables.append(("clust_mask", clust_mask_list))
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
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])

    if plot_switch == True:
        wb_functional_connectometry_wf.add_nodes([plot_all_node])
        wb_functional_connectometry_wf.connect([(inputnode, plot_all_node, [('ID', 'ID'),
                                                                            ('mask', 'mask'),
                                                                            ('network', 'network'),
                                                                            ('conn_model', 'conn_model')]),
                                                (WB_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path'),
                                                                                                 ('atlas_select', 'atlas_select')]),
                                                (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                                                                ('coords', 'coords')]),
                                                (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                                                                      ('edge_threshold', 'edge_threshold')]),
                                                ])
    if k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1:
        wb_functional_connectometry_wf.add_nodes([clustering_node])
        if plot_switch == True:
            wb_functional_connectometry_wf.add_nodes([plot_all_node])
            wb_functional_connectometry_wf.disconnect([(inputnode, WB_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (WB_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path'), ('atlas_select', 'atlas_select')]),
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
            wb_functional_connectometry_wf.disconnect([(inputnode, WB_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (inputnode, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (WB_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path')]),
                                                       (WB_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                       (WB_fetch_nodes_and_labels_node, extract_ts_wb_node, [('dir_path', 'dir_path')])
                                                       ])
            wb_functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'), ('func_file', 'func_file'), ('clust_mask', 'clust_mask'), ('k', 'k')]),
                                                    (clustering_node, WB_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                    (clustering_node, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                    (clustering_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                    (clustering_node, extract_ts_wb_node, [('dir_path', 'dir_path')])
                                                    ])

    if parc == True:
        wb_functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        wb_functional_connectometry_wf.connect([(inputnode, save_nifti_parcels_node, [('ID', 'ID'),('mask', 'mask')]),
                                                (inputnode, save_nifti_parcels_node, [('network', 'network')]),
                                                (WB_fetch_nodes_and_labels_node, save_nifti_parcels_node, [('dir_path', 'dir_path')]),
                                                (node_gen_node, save_nifti_parcels_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                                ])
    else:
        wb_functional_connectometry_wf.disconnect([(node_gen_node, extract_ts_wb_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                                                                       ('coords', 'coords')])
                                                ])
        wb_functional_connectometry_wf.connect([(node_gen_node, extract_ts_wb_node, [('coords', 'coords')])
                                                ])
        wb_functional_connectometry_wf.connect([(inputnode, extract_ts_wb_node, [('node_size', 'node_size')])
                                                ])

    wb_functional_connectometry_wf.config['logging']['log_directory']='/tmp'
    wb_functional_connectometry_wf.config['logging']['workflow_level']='DEBUG'
    wb_functional_connectometry_wf.config['logging']['utils_level']='DEBUG'
    wb_functional_connectometry_wf.config['logging']['interface_level']='DEBUG'
    wb_functional_connectometry_wf.config['execution']['plugin']='MultiProc'
    #wb_functional_connectometry_wf.write_graph()
    res = wb_functional_connectometry_wf.run(plugin='MultiProc')
    #res = wb_functional_connectometry_wf.run()

    try:
        thr=list(res.nodes())[-1].result.outputs.thr
        est_path=list(res.nodes())[-1].result.outputs.est_path
    except AttributeError:
        try:
            thr=list(res.nodes())[-2].result.outputs.thr
            est_path=list(res.nodes())[-2].result.outputs.est_path
        except AttributeError:
            thr=list(res.nodes())[-3].result.outputs.thr
            est_path=list(res.nodes())[-3].result.outputs.est_path
    return(est_path, thr)

def rsn_functional_connectometry(func_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, multi_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, parc, ref_txt, procmem, dir_path, multi_thr, multi_atlas, max_thr, min_thr, step_thr, k, clust_mask, k_min, k_max, k_step, k_clustering, user_atlas_list, clust_mask_list):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import nodemaker, utils, graphestimation, plotting, thresholding
    import_list=[ "import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]

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
                                                     function=graphestimation.extract_ts_parc, imports = import_list), name = "extract_ts_rsn_parc_node")
    else:
        ##Extract within-spheres time-series from funct file
        extract_ts_rsn_node = pe.Node(niu.Function(input_names = ['node_size', 'conf', 'func_file', 'coords', 'dir_path', 'ID', 'mask', 'network'],
                                             output_names=['ts_within_nodes'],
                                             function=graphestimation.extract_ts_coords, imports = import_list), name = "extract_ts_rsn_coords_node")

    thresh_and_fit_node = pe.Node(niu.Function(input_names = ['adapt_thresh', 'dens_thresh', 'thr', 'ts_within_nodes', 'conn_model', 'network', 'ID', 'dir_path', 'mask'],
                                         output_names=['conn_matrix_thr', 'edge_threshold', 'est_path', 'thr'],
                                         function=thresholding.thresh_and_fit, imports = import_list), name = "thresh_and_fit_node")

    ##Plotting
    if plot_switch == True:
        plot_all_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_model', 'atlas_select', 'dir_path', 'ID', 'network', 'label_names', 'mask', 'coords', 'edge_threshold'],
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
    elif user_atlas_list is not None:
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
    elif k_clustering==3:
        k_cluster_iterables = []
        k_cluster_iterables.append(("clust_mask", clust_mask_list))
        clustering_node.iterables = k_cluster_iterables
    elif k_clustering==4:
        k_cluster_iterables = []
        k_list = np.round(np.arange(int(k_min), int(k_max), int(k_step)),decimals=0).tolist()
        k_cluster_iterables.append(("k", k_list))
        k_cluster_iterables.append(("clust_mask", clust_mask_list))
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
                                          ('network', 'network')]),
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
        (thresh_and_fit_node, outputnode, [('est_path', 'est_path'),
                                           ('thr', 'thr')]),
        ])

    if plot_switch == True:
        rsn_functional_connectometry_wf.add_nodes([plot_all_node])
        rsn_functional_connectometry_wf.connect([(inputnode, plot_all_node, [('ID', 'ID'),
                                                                            ('mask', 'mask'),
                                                                            ('network', 'network'),
                                                                            ('conn_model', 'conn_model')]),
                                                (RSN_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path'), 
                                                                                                  ('atlas_select', 'atlas_select')]),
                                                (node_gen_node, plot_all_node, [('label_names', 'label_names'),
                                                                                      ('coords', 'coords')]),
                                                (thresh_and_fit_node, plot_all_node, [('conn_matrix_thr', 'conn_matrix'),
                                                                                      ('edge_threshold', 'edge_threshold')])
                                                ])
    if k_clustering == 4 or k_clustering == 3 or k_clustering == 2 or k_clustering == 1:
        rsn_functional_connectometry_wf.add_nodes([clustering_node])
        if plot_switch == True:
            rsn_functional_connectometry_wf.add_nodes([plot_all_node])
            rsn_functional_connectometry_wf.disconnect([(inputnode, RSN_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (RSN_fetch_nodes_and_labels_node, plot_all_node, [('dir_path', 'dir_path'), ('atlas_select', 'atlas_select')]),
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
            rsn_functional_connectometry_wf.disconnect([(inputnode, RSN_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (inputnode, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                       (RSN_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path')]),
                                                       (RSN_fetch_nodes_and_labels_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                       (RSN_fetch_nodes_and_labels_node, extract_ts_rsn_node, [('dir_path', 'dir_path')])
                                                       ])
            rsn_functional_connectometry_wf.connect([(inputnode, clustering_node, [('ID', 'ID'), ('func_file', 'func_file'), ('clust_mask', 'clust_mask'), ('k', 'k')]),
                                                    (clustering_node, RSN_fetch_nodes_and_labels_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select')]),
                                                    (clustering_node, node_gen_node, [('parlistfile', 'parlistfile'), ('atlas_select', 'atlas_select'), ('dir_path', 'dir_path')]),
                                                    (clustering_node, thresh_and_fit_node, [('dir_path', 'dir_path')]),
                                                    (clustering_node, extract_ts_rsn_node, [('dir_path', 'dir_path')])
                                                    ])            
    if parc == True:
        rsn_functional_connectometry_wf.add_nodes([save_nifti_parcels_node])
        rsn_functional_connectometry_wf.connect([(inputnode, save_nifti_parcels_node, [('ID', 'ID'),('mask', 'mask')]),
                                                (get_node_membership_node, save_nifti_parcels_node, [('network', 'network')]),
                                                (RSN_fetch_nodes_and_labels_node, save_nifti_parcels_node, [('dir_path', 'dir_path')]),
                                                (node_gen_node, save_nifti_parcels_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')])
                                                ])
    else:
        rsn_functional_connectometry_wf.disconnect([(node_gen_node, extract_ts_rsn_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti'),
                                                                                       ('coords', 'coords')])
                                                ])
        rsn_functional_connectometry_wf.connect([(node_gen_node, extract_ts_rsn_node, [('coords', 'coords')])
                                                ])
        rsn_functional_connectometry_wf.connect([(inputnode, extract_ts_rsn_node, [('node_size', 'node_size')])
                                                ])

    if multi_nets is not None:
        if plot_switch == True:
            rsn_functional_connectometry_wf.disconnect([(inputnode, extract_ts_rsn_node, [('network', 'network')]),
                                                        (inputnode, thresh_and_fit_node, [('network', 'network')]),
                                                        (inputnode, plot_all_node, [('network', 'network')])
                                                        ])
            rsn_functional_connectometry_wf.connect([(get_node_membership_node, extract_ts_rsn_node, [('network', 'network')]),
                                                    (get_node_membership_node, thresh_and_fit_node, [('network', 'network')]),
                                                    (get_node_membership_node, plot_all_node, [('network', 'network')])
                                                    ])
        else:
            rsn_functional_connectometry_wf.disconnect([(inputnode, extract_ts_rsn_node, [('network', 'network')]),
                                                        (inputnode, thresh_and_fit_node, [('network', 'network')]),
                                                        ])
            rsn_functional_connectometry_wf.connect([(get_node_membership_node, extract_ts_rsn_node, [('network', 'network')]),
                                                    (get_node_membership_node, thresh_and_fit_node, [('network', 'network')]),
                                                    ])
    
    rsn_functional_connectometry_wf.config['logging']['log_directory']='/tmp'
    rsn_functional_connectometry_wf.config['logging']['workflow_level']='DEBUG'
    rsn_functional_connectometry_wf.config['logging']['utils_level']='DEBUG'
    rsn_functional_connectometry_wf.config['logging']['interface_level']='DEBUG'
    rsn_functional_connectometry_wf.config['execution']['plugin']='MultiProc'
    #rsn_functional_connectometry_wf.write_graph()
    res = rsn_functional_connectometry_wf.run(plugin='MultiProc')
    #res = rsn_functional_connectometry_wf.run()

    try:
        thr=list(res.nodes())[-1].result.outputs.thr
        est_path=list(res.nodes())[-1].result.outputs.est_path
    except AttributeError:
        try:
            thr=list(res.nodes())[-2].result.outputs.thr
            est_path=list(res.nodes())[-2].result.outputs.est_path
        except AttributeError:
            thr=list(res.nodes())[-3].result.outputs.thr
            est_path=list(res.nodes())[-3].result.outputs.est_path
    return est_path, thr

def wb_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, label_names, anat_loc):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import utils, nodemaker, diffconnectometry, plotting

    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'

    import_list=[ "import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]
    wb_structural_connectometry_wf = pe.Workflow(name='wb_structural_connectometry')
    wb_structural_connectometry_wf.base_directory='/tmp/pynets'

    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'atlas_select', 'network', 'node_size', 'mask', 'parlistfile', 'plot_switch', 'parc', 'ref_txt', 'procmem', 'dir_path', 'bedpostx_dir', 'label_names', 'anat_loc']), name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.parlistfile = parlistfile
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dir_path = dir_path
    inputnode.inputs.bedpostx_dir = bedpostx_dir
    inputnode.inputs.label_names = label_names
    inputnode.inputs.anat_loc = anat_loc
    inputnode.inputs.nodif_brain_mask_path = nodif_brain_mask_path


    #3) Add variable to function nodes
    ##Create function nodes
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

    prepare_masks_node = pe.Node(niu.Function(input_names = ['bedpostx_dir', 'anat_loc'],
                                              output_names=['WM_diff_mask_path', 'vent_CSF_diff_mask_path'],
                                         function=diffconnectometry.prepare_masks, imports = import_list), name = "prepare_masks_node")

    grow_nodes_node = pe.Node(niu.Function(input_names = ['bedpostx_dir', 'coords', 'node_size', 'parc', 'parcel_list', 'net_parcels_map_nifti', 'network'],
                                           output_names=['seeds_text', 'probtrackx_output_dir_path'],
                                         function=diffconnectometry.grow_nodes, imports = import_list), name = "grow_nodes_node")

    run_probtrackx2_node = pe.Node(niu.Function(input_names = ['i', 'seeds_text', 'bedpostx_dir', 'probtrackx_output_dir_path', 'vent_CSF_diff_mask_path', 'WM_diff_mask_path', 'procmem'],
                                                output_names=['max_i'],
                                         function=diffconnectometry.run_probtrackx2, imports = import_list), name = "run_probtrackx2_node")

    run_probtrackx2_iterables = []
    iter_i = range(int(procmem[0]))
    run_probtrackx2_iterables.append(("i", iter_i))
    run_probtrackx2_node.iterables = run_probtrackx2_iterables

    collect_struct_mapping_outputs_node = pe.Node(niu.Function(input_names = ['parc', 'bedpostx_dir', 'network', 'ID', 'probtrackx_output_dir_path', 'max_i'],
                                              output_names=['conn_matrix', 'conn_matrix_symm', 'est_path_struct'],
                                         function=diffconnectometry.collect_struct_mapping_outputs, imports = import_list), name = "collect_struct_mapping_outputs_node")

    if plot_switch == True:
        structural_plotting_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_matrix_symm', 'label_names', 'atlas_select', 'ID', 'bedpostx_dir', 'network', 'parc', 'coords'],
                                             function=plotting.structural_plotting, imports = import_list), name = "structural_plotting_node")

    outputnode = pe.Node(niu.Function(function=utils.output_echo, input_names=['est_path_struct'], output_names=['est_path_struct']), name='outputnode')

    ##Connect nodes of workflow
    wb_structural_connectometry_wf.connect([
        (inputnode, WB_fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                     ('parlistfile', 'parlistfile'),
                                                     ('parc', 'parc'),
                                                     ('ref_txt', 'ref_txt')]),
        (inputnode, node_gen_node, [('ID', 'ID'),
                                    ('mask', 'mask'),
                                    ('parc', 'parc'),
                                    ('ref_txt', 'ref_txt'),
                                    ('atlas_select', 'atlas_select'),
                                    ('parlistfile', 'parlistfile')]),
        (inputnode, WB_fetch_nodes_and_labels_node, [('nodif_brain_mask_path', 'func_file')]),
        (WB_fetch_nodes_and_labels_node, node_gen_node, [('coords', 'coords'),
                                                         ('label_names', 'label_names'),
                                                         ('dir_path', 'dir_path'),
                                                         ('parcel_list', 'parcel_list'),
                                                         ('par_max', 'par_max'),
                                                         ('networks_list', 'networks_list')]),
        (WB_fetch_nodes_and_labels_node, grow_nodes_node, [('parcel_list', 'parcel_list')]),
        (node_gen_node, grow_nodes_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
        (inputnode, prepare_masks_node, [('bedpostx_dir', 'bedpostx_dir'),
                                         ('anat_loc', 'anat_loc')]),
        (WB_fetch_nodes_and_labels_node, grow_nodes_node, [('coords', 'coords')]),
        (inputnode, grow_nodes_node, [('bedpostx_dir', 'bedpostx_dir'),
                                      ('node_size', 'node_size'),
                                      ('parc', 'parc'),
                                      ('network', 'network')]),
        (inputnode, run_probtrackx2_node, [('bedpostx_dir', 'bedpostx_dir'),
                                           ('procmem', 'procmem')]),
        (prepare_masks_node, run_probtrackx2_node, [('vent_CSF_diff_mask_path', 'vent_CSF_diff_mask_path'),
                                                    ('WM_diff_mask_path', 'WM_diff_mask_path')]),
        (grow_nodes_node, run_probtrackx2_node, [('seeds_text', 'seeds_text'),
                                                            ('probtrackx_output_dir_path','probtrackx_output_dir_path')]),
        (grow_nodes_node, collect_struct_mapping_outputs_node, [('probtrackx_output_dir_path','probtrackx_output_dir_path')]),
        (inputnode, collect_struct_mapping_outputs_node, [('bedpostx_dir', 'bedpostx_dir'),
                                                          ('node_size', 'node_size'),
                                                          ('parc', 'parc'),
                                                          ('network', 'network'),
                                                          ('ID', 'ID')]),
        (run_probtrackx2_node, collect_struct_mapping_outputs_node, [('max_i', 'max_i')]),
        (collect_struct_mapping_outputs_node, outputnode, [('est_path_struct', 'est_path_struct')]),
        ])

    if plot_switch == True:
        wb_structural_connectometry_wf.add_nodes([structural_plotting_node])
        wb_structural_connectometry_wf.connect([(collect_struct_mapping_outputs_node, structural_plotting_node, [('conn_matrix', 'conn_matrix'),
                                                                         ('conn_matrix_symm', 'conn_matrix_symm')]),
                                                (inputnode, structural_plotting_node, [('label_names', 'label_names'),
                                                                                       ('atlas_select', 'atlas_select'),
                                                                                       ('ID', 'ID'),
                                                                                       ('bedpostx_dir', 'bedpostx_dir'),
                                                                                       ('network', 'network'),
                                                                                       ('parc', 'parc'),
                                                                                       ('plot_switch', 'plot_switch')]),
                                                (WB_fetch_nodes_and_labels_node, structural_plotting_node, [('coords', 'coords')])
                                                ])
        
    wb_structural_connectometry_wf.config['execution']['crashdump_dir']='/tmp'
    wb_structural_connectometry_wf.config['logging']['log_directory']='/tmp'
    wb_structural_connectometry_wf.config['logging']['workflow_level']='DEBUG'
    wb_structural_connectometry_wf.config['logging']['utils_level']='DEBUG'
    wb_structural_connectometry_wf.config['logging']['interface_level']='DEBUG'
    wb_structural_connectometry_wf.config['execution']['plugin']='MultiProc'
    #wb_structural_connectometry_wf.write_graph()
    res = wb_structural_connectometry_wf.run(plugin='MultiProc')
    #res = wb_structural_connectometry_wf.run()

    try:
        est_path_struct=list(res.nodes())[-1].result.outputs.est_path_struct
    except AttributeError:
        est_path_struct=list(res.nodes())[-2].result.outputs.est_path_struct
        try:
            est_path_struct=list(res.nodes())[-2].result.outputs.est_path_struct
        except AttributeError:
            est_path_struct=list(res.nodes())[-3].result.outputs.est_path_struct

    return est_path_struct

def rsn_structural_connectometry(ID, atlas_select, network, node_size, mask, parlistfile, plot_switch, parc, ref_txt, procmem, dir_path, bedpostx_dir, label_names, anat_loc):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from pynets import utils, nodemaker, diffconnectometry, plotting

    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'

    import_list=[ "import sys", "import os", "import numpy as np", "import networkx as nx", "import nibabel as nib"]

    rsn_structural_connectometry_wf = pe.Workflow(name='wb_structural_connectometry')
    rsn_structural_connectometry_wf.base_directory='/tmp/pynets'

    ##Create input/output nodes
    #1) Add variable to IdentityInterface if user-set
    inputnode = pe.Node(niu.IdentityInterface(fields=['ID', 'atlas_select', 'network', 'node_size', 'mask', 'parlistfile', 'plot_switch', 'parc', 'ref_txt', 'procmem', 'dir_path', 'bedpostx_dir', 'label_names', 'anat_loc']), name='inputnode')

    #2)Add variable to input nodes if user-set (e.g. inputnode.inputs.WHATEVER)
    inputnode.inputs.ID = ID
    inputnode.inputs.atlas_select = atlas_select
    inputnode.inputs.network = network
    inputnode.inputs.node_size = node_size
    inputnode.inputs.mask = mask
    inputnode.inputs.parlistfile = parlistfile
    inputnode.inputs.plot_switch = plot_switch
    inputnode.inputs.parc = parc
    inputnode.inputs.ref_txt = ref_txt
    inputnode.inputs.procmem = procmem
    inputnode.inputs.dir_path = dir_path
    inputnode.inputs.bedpostx_dir = bedpostx_dir
    inputnode.inputs.label_names = label_names
    inputnode.inputs.anat_loc = anat_loc
    inputnode.inputs.nodif_brain_mask_path = nodif_brain_mask_path


    #3) Add variable to function nodes
    ##Create function nodes
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

    prepare_masks_node = pe.Node(niu.Function(input_names = ['bedpostx_dir', 'anat_loc'],
                                              output_names=['WM_diff_mask_path', 'vent_CSF_diff_mask_path'],
                                         function=diffconnectometry.prepare_masks, imports = import_list), name = "prepare_masks_node")

    grow_nodes_node = pe.Node(niu.Function(input_names = ['bedpostx_dir', 'coords', 'node_size', 'parc', 'parcel_list', 'net_parcels_map_nifti', 'network'],
                                           output_names=['seeds_text', 'probtrackx_output_dir_path'],
                                         function=diffconnectometry.grow_nodes, imports = import_list), name = "grow_nodes_node")

    run_probtrackx2_node = pe.Node(niu.Function(input_names = ['i', 'seeds_text', 'bedpostx_dir', 'probtrackx_output_dir_path', 'vent_CSF_diff_mask_path', 'WM_diff_mask_path', 'procmem'],
                                                output_names=['max_i'],
                                         function=diffconnectometry.run_probtrackx2, imports = import_list), name = "run_probtrackx2_node")

    run_probtrackx2_iterables = []
    iter_i = range(int(procmem[0]))
    run_probtrackx2_iterables.append(("i", iter_i))
    run_probtrackx2_node.iterables = run_probtrackx2_iterables

    collect_struct_mapping_outputs_node = pe.Node(niu.Function(input_names = ['parc', 'bedpostx_dir', 'network', 'ID', 'probtrackx_output_dir_path', 'max_i'],
                                              output_names=['conn_matrix', 'conn_matrix_symm', 'est_path_struct'],
                                         function=diffconnectometry.collect_struct_mapping_outputs, imports = import_list), name = "collect_struct_mapping_outputs_node")

    if plot_switch == True:
        structural_plotting_node = pe.Node(niu.Function(input_names = ['conn_matrix', 'conn_matrix_symm', 'label_names', 'atlas_select', 'ID', 'bedpostx_dir', 'network', 'parc', 'coords'],
                                             function=plotting.structural_plotting, imports = import_list), name = "structural_plotting_node")

    outputnode = pe.Node(niu.Function(function=utils.output_echo, input_names=['est_path_struct'], output_names=['est_path_struct']), name='outputnode')

    ##Connect nodes of workflow
    rsn_structural_connectometry_wf.connect([
        (inputnode, RSN_fetch_nodes_and_labels_node, [('atlas_select', 'atlas_select'),
                                                      ('parlistfile', 'parlistfile'),
                                                      ('parc', 'parc'),
                                                      ('ref_txt', 'ref_txt')]),
        (inputnode, get_node_membership_node, [('network', 'network'),
                                               ('nodif_brain_mask_path', 'func_file'),
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
        (inputnode, RSN_fetch_nodes_and_labels_node, [('nodif_brain_mask_path', 'func_file')]),
        (RSN_fetch_nodes_and_labels_node, node_gen_node, [('dir_path', 'dir_path'),
                                                          ('par_max', 'par_max'),
                                                          ('networks_list', 'networks_list')]),
        (get_node_membership_node, node_gen_node, [('net_coords', 'coords'),
                                                    ('net_label_names', 'label_names'),
                                                    ('net_parcel_list', 'parcel_list')]),
        (RSN_fetch_nodes_and_labels_node, grow_nodes_node, [('parcel_list', 'parcel_list')]),
        (node_gen_node, grow_nodes_node, [('net_parcels_map_nifti', 'net_parcels_map_nifti')]),
        (inputnode, prepare_masks_node, [('bedpostx_dir', 'bedpostx_dir'),
                                         ('anat_loc', 'anat_loc')]),
        (RSN_fetch_nodes_and_labels_node, grow_nodes_node, [('coords', 'coords')]),
        (inputnode, grow_nodes_node, [('bedpostx_dir', 'bedpostx_dir'),
                                      ('node_size', 'node_size'),
                                      ('parc', 'parc'),
                                      ('network', 'network')]),
        (inputnode, run_probtrackx2_node, [('bedpostx_dir', 'bedpostx_dir'),
                                           ('procmem', 'procmem')]),
        (prepare_masks_node, run_probtrackx2_node, [('vent_CSF_diff_mask_path', 'vent_CSF_diff_mask_path'),
                                                    ('WM_diff_mask_path', 'WM_diff_mask_path')]),
        (grow_nodes_node, run_probtrackx2_node, [('seeds_text', 'seeds_text'),
                                                            ('probtrackx_output_dir_path','probtrackx_output_dir_path')]),
        (grow_nodes_node, collect_struct_mapping_outputs_node, [('probtrackx_output_dir_path','probtrackx_output_dir_path')]),
        (inputnode, collect_struct_mapping_outputs_node, [('bedpostx_dir', 'bedpostx_dir'),
                                                          ('node_size', 'node_size'),
                                                          ('parc', 'parc'),
                                                          ('network', 'network'),
                                                          ('ID', 'ID')]),
        (run_probtrackx2_node, collect_struct_mapping_outputs_node, [('max_i', 'max_i')]),
        (collect_struct_mapping_outputs_node, outputnode, [('est_path_struct', 'est_path_struct')]),
        ])

    if plot_switch == True:
        rsn_structural_connectometry_wf.add_nodes([structural_plotting_node])
        rsn_structural_connectometry_wf.connect([(collect_struct_mapping_outputs_node, structural_plotting_node, [('conn_matrix', 'conn_matrix'),
                                                                         ('conn_matrix_symm', 'conn_matrix_symm')]),
                                                (inputnode, structural_plotting_node, [('label_names', 'label_names'),
                                                                                       ('atlas_select', 'atlas_select'),
                                                                                       ('ID', 'ID'),
                                                                                       ('bedpostx_dir', 'bedpostx_dir'),
                                                                                       ('network', 'network'),
                                                                                       ('parc', 'parc'),
                                                                                       ('plot_switch', 'plot_switch')]),
                                                (RSN_fetch_nodes_and_labels_node, structural_plotting_node, [('coords', 'coords')])
                                                ])
    
    rsn_structural_connectometry_wf.config['logging']['log_directory']='/tmp'
    rsn_structural_connectometry_wf.config['logging']['workflow_level']='DEBUG'
    rsn_structural_connectometry_wf.config['logging']['utils_level']='DEBUG'
    rsn_structural_connectometry_wf.config['logging']['interface_level']='DEBUG'
    rsn_structural_connectometry_wf.config['execution']['plugin']='MultiProc'
    #rsn_structural_connectometry_wf.write_graph()
    res = rsn_structural_connectometry_wf.run(plugin='MultiProc')
    #res = rsn_structural_connectometry_wf.run()

    try:
        est_path_struct=list(res.nodes())[-1].result.outputs.est_path_struct
    except AttributeError:
        est_path_struct=list(res.nodes())[-2].result.outputs.est_path_struct
        try:
            est_path_struct=list(res.nodes())[-2].result.outputs.est_path_struct
        except AttributeError:
            est_path_struct=list(res.nodes())[-3].result.outputs.est_path_struct
    return est_path_struct
