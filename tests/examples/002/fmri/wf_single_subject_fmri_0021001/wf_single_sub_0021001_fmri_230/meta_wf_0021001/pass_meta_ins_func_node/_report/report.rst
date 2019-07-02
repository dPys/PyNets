Node: meta_wf_0021001 (pass_meta_ins_func_node (utility)
========================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.pass_meta_ins_func_node
 Exec ID : pass_meta_ins_func_node


Original Inputs
---------------


* ID : [[['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']], [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]]
* binary : [[[False, False, False, False, False, False], [False, False, False, False, False, False]], [[False, False, False, False, False, False], [False, False, False, False, False, False]]]
* conn_model : [[['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']], [['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']]]
* est_path : [[['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.2dens4_mm.npy']], [['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.2dens4_mm.npy']]]
* function_str : def pass_meta_ins(conn_model, est_path, network, node_size, thr, prune, ID, roi, norm, binary):
    """
    Passes parameters as metadata.

    Parameters
    ----------
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.

    Returns
    -------
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    """
    est_path_iterlist = est_path
    conn_model_iterlist = conn_model
    network_iterlist = network
    node_size_iterlist = node_size
    thr_iterlist = thr
    prune_iterlist = prune
    ID_iterlist = ID
    roi_iterlist = roi
    norm_iterlist = norm
    binary_iterlist = binary
    # print('\n\nParam-iters:\n')
    # print(conn_model_iterlist)
    # print(est_path_iterlist)
    # print(network_iterlist)
    # print(node_size_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(roi_iterlist)
    # print(norm_iterlist)
    # print(binary_iterlist)
    # print('\n\n')
    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist

* network : [[['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']], [['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn'], ['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn']]]
* node_size : [[[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]], [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]]
* norm : [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]
* prune : [[['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']], [['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']]]
* roi : [[[None, None, None, None, None, None], [None, None, None, None, None, None]], [[None, None, None, None, None, None], [None, None, None, None, None, None]]]
* thr : [[['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']], [['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']]]

Execution Inputs
----------------


* ID : [[['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']], [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]]
* binary : [[[False, False, False, False, False, False], [False, False, False, False, False, False]], [[False, False, False, False, False, False], [False, False, False, False, False, False]]]
* conn_model : [[['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']], [['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']]]
* est_path : [[['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.2dens4_mm.npy']], [['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.2dens4_mm.npy']]]
* function_str : def pass_meta_ins(conn_model, est_path, network, node_size, thr, prune, ID, roi, norm, binary):
    """
    Passes parameters as metadata.

    Parameters
    ----------
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.

    Returns
    -------
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    est_path : str
        File path to .npy file containing graph with thresholding applied.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    """
    est_path_iterlist = est_path
    conn_model_iterlist = conn_model
    network_iterlist = network
    node_size_iterlist = node_size
    thr_iterlist = thr
    prune_iterlist = prune
    ID_iterlist = ID
    roi_iterlist = roi
    norm_iterlist = norm
    binary_iterlist = binary
    # print('\n\nParam-iters:\n')
    # print(conn_model_iterlist)
    # print(est_path_iterlist)
    # print(network_iterlist)
    # print(node_size_iterlist)
    # print(thr_iterlist)
    # print(prune_iterlist)
    # print(ID_iterlist)
    # print(roi_iterlist)
    # print(norm_iterlist)
    # print(binary_iterlist)
    # print('\n\n')
    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist

* network : [[['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']], [['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn'], ['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn']]]
* node_size : [[[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]], [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]]
* norm : [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]
* prune : [[['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']], [['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']]]
* roi : [[[None, None, None, None, None, None], [None, None, None, None, None, None]], [[None, None, None, None, None, None], [None, None, None, None, None, None]]]
* thr : [[['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']], [['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']]]


Execution Outputs
-----------------


* ID_iterlist : [[['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']], [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]]
* binary_iterlist : [[[False, False, False, False, False, False], [False, False, False, False, False, False]], [[False, False, False, False, False, False], [False, False, False, False, False, False]]]
* conn_model_iterlist : [[['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']], [['sps', 'sps', 'sps', 'sps', 'sps', 'sps'], ['partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr', 'partcorr']]]
* est_path_iterlist : [[['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_Default_est_partcorr_0.2dens4_mm.npy']], [['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.2dens4_mm.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.15dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.16dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.17dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.18dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.19dens4_mm.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_partcorr_0.2dens4_mm.npy']]]
* network_iterlist : [[['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']], [['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn'], ['SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn', 'SalVentAttn']]]
* node_size_iterlist : [[[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]], [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]]
* norm_iterlist : [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]
* prune_iterlist : [[['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']], [['1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1']]]
* roi_iterlist : [[[None, None, None, None, None, None], [None, None, None, None, None, None]], [[None, None, None, None, None, None], [None, None, None, None, None, None]]]
* thr_iterlist : [[['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']], [['0.15', '0.16', '0.17', '0.18', '0.19', '0.2'], ['0.15', '0.16', '0.17', '0.18', '0.19', '0.2']]]


Runtime info
------------


* duration : 0.001344
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/pass_meta_ins_func_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.VKfenSaB7x/Render
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include
* DISPLAY : dpys:0.0
* DYLD_LIBRARY_PATH : /Applications/freesurfer/lib/gcc/lib::/opt/X11/lib/flat_namespace
* FIX_VERTEX_AREA : 
* FMRI_ANALYSIS_DIR : /Applications/freesurfer/fsfast
* FREESURFER_HOME : /Applications/freesurfer
* FSFAST_HOME : /Applications/freesurfer/fsfast
* FSF_OUTPUT_FORMAT : nii.gz
* FSLDIR : /usr/local/fsl
* FSLGECUDAQ : cuda.q
* FSLLOCKDIR : 
* FSLMACHINELIST : 
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLREMOTECALL : 
* FSLTCLSH : /usr/local/fsl/bin/fsltclsh
* FSLWISH : /usr/local/fsl/bin/fslwish
* FSL_BIN : /usr/local/fsl/bin
* FSL_DIR : /usr/local/fsl
* FS_OVERRIDE : 0
* FUNCTIONALS_DIR : /Applications/freesurfer/sessions
* HOME : /Users/derekpisner
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/libxml2/lib
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : derekpisner
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* OLDPWD : /Users/derekpisner/Applications/PyNets_new_bak/pynets
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/Users/derekpisner/anaconda3/bin:/Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets
* SHELL : /bin/bash
* SHLVL : 3
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.qmAkE8F40f/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : 6432F315-D86A-4D51-A77C-DB02F4938E15
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

