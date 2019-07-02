Node: utility
=============


 Hierarchy : _thresh_func_node1
 Exec ID : _thresh_func_node1


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[-1.00952801  0.          0.         -0.          0.06115581  0.05460969
  -0.          0.         -0.          0.         -0.          0.
   0.        ]
 [ 0.         -1.70560904  0.          0.          0.55010506  0.
   0.08818284  0.02786552 -0.          0.48763399  0.13135885  0.
   0.1807707 ]
 [ 0.          0.         -2.10323819  0.3648176   0.          0.
   0.          0.29168192  0.29410093  0.34127086  0.          0.81837209
   0.        ]
 [-0.          0.          0.3648176  -2.03360665  0.          0.52889168
   0.2483378   0.08494683  0.          0.02635054  0.51311321  0.17767663
   0.10671197]
 [ 0.06115581  0.55010506  0.          0.         -1.89594385  0.25155992
   0.50764837  0.06629916  0.          0.38541002  0.02195094  0.00878754
   0.        ]
 [ 0.05460969  0.          0.          0.52889168  0.25155992 -1.81798113
   0.03662809  0.33071176  0.          0.20585683  0.2759881   0.06168298
   0.        ]
 [-0.          0.08818284  0.          0.2483378   0.50764837  0.03662809
  -1.39317243  0.          0.          0.          0.          0.
   0.0513585 ]
 [ 0.          0.02786552  0.29168192  0.08494683  0.06629916  0.33071176
   0.         -1.64800358 -0.          0.0138876   0.          0.
   0.57658514]
 [-0.         -0.          0.29410093  0.          0.          0.
   0.         -0.         -1.31742412  0.          0.21197075  0.28585196
  -0.        ]
 [ 0.          0.48763399  0.34127086  0.02635054  0.38541002  0.20585683
   0.          0.0138876   0.         -1.70783282  0.          0.01603164
   0.        ]
 [-0.          0.13135885  0.          0.51311321  0.02195094  0.2759881
   0.          0.          0.21197075  0.         -1.65841373  0.13259067
   0.17883609]
 [ 0.          0.          0.81837209  0.17767663  0.00878754  0.06168298
   0.          0.          0.28585196  0.01603164  0.13259067 -1.85940656
   0.1548196 ]
 [ 0.          0.1807707   0.          0.10671197  0.          0.
   0.0513585   0.57658514 -0.          0.          0.17883609  0.1548196
  -1.53741506]]
* conn_model : sps
* coords : [[-40.  32. -10.]
 [-54. -38.  34.]
 [  6.  22.  28.]
 [-50. -10.  -6.]
 [ 48.  16.  14.]
 [ 52.  -6.  -8.]
 [ 48.  32.   4.]
 [ 38.   2.   0.]
 [  6. -16.  40.]
 [ 54. -36.  36.]
 [ -6. -18.  40.]
 [ -6.  20.  32.]
 [-36.   2.   0.]]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* function_str : def thresh_func(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                smooth, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                hpass):
    """
    Threshold a functional connectivity matrix using any of a variety of methods.

    Parameters
    ----------
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.

    Returns
    -------
    conn_matrix_thr : array
        Weighted, thresholded, NxN matrix.
    edge_threshold : str
        The string percentage representation of thr.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets import utils, thresholding

    thr_perc = 100 * float(thr)
    edge_threshold = "%s%s" % (str(thr_perc), '%')
    if parc is True:
        node_size = 'parc'

    if np.count_nonzero(conn_matrix) == 0:
        raise ValueError('ERROR: Raw connectivity matrix contains only zeros.')

    # Save unthresholded
    unthr_path = utils.create_unthr_path(ID, network, conn_model, roi, dir_path)
    utils.save_mat(conn_matrix, unthr_path)

    if min_span_tree is True:
        print('Using local thresholding option with the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            thr_type = 'MSTprop'
            conn_matrix_thr = thresholding.local_thresholding_prop(conn_matrix, thr)
        else:
            thr_type = 'MSTdens'
            conn_matrix_thr = thresholding.local_thresholding_dens(conn_matrix, thr)
    elif disp_filt is True:
        thr_type = 'DISP_alpha'
        G1 = thresholding.disparity_filter(nx.from_numpy_array(conn_matrix))
        # G2 = nx.Graph([(u, v, d) for u, v, d in G1.edges(data=True) if d['alpha'] < thr])
        print('Computing edge disparity significance with alpha = %s' % thr)
        print('Filtered graph: nodes = %s, edges = %s' % (G1.number_of_nodes(), G1.number_of_edges()))
        # print('Backbone graph: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))
        # print(G2.edges(data=True))
        conn_matrix_thr = nx.to_numpy_array(G1)
    else:
        if dens_thresh is False:
            thr_type = 'prop'
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            thr_type = 'dens'
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('Warning: Fragmented graph')

    # Save thresholded mat
    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot,
                                          thr_type, hpass, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, smooth, prune, ID, dir_path, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass

* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* thr : 0.16
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[ 0.          0.          0.         -0.          0.06115581  0.05460969
  -0.          0.         -0.          0.         -0.          0.
   0.        ]
 [ 0.          0.          0.          0.          0.55010506  0.
   0.08818284  0.02786552 -0.          0.48763399  0.13135885  0.
   0.1807707 ]
 [ 0.          0.          0.          0.3648176   0.          0.
   0.          0.29168192  0.29410093  0.34127086  0.          0.81837209
   0.        ]
 [-0.          0.          0.3648176   0.          0.          0.52889168
   0.2483378   0.08494683  0.          0.02635054  0.51311321  0.17767663
   0.10671197]
 [ 0.06115581  0.55010506  0.          0.          0.          0.25155992
   0.50764837  0.06629916  0.          0.38541002  0.02195094  0.00878754
   0.        ]
 [ 0.05460969  0.          0.          0.52889168  0.25155992  0.
   0.03662809  0.33071176  0.          0.20585683  0.2759881   0.06168298
   0.        ]
 [-0.          0.08818284  0.          0.2483378   0.50764837  0.03662809
   0.          0.          0.          0.          0.          0.
   0.0513585 ]
 [ 0.          0.02786552  0.29168192  0.08494683  0.06629916  0.33071176
   0.          0.         -0.          0.0138876   0.          0.
   0.57658514]
 [-0.         -0.          0.29410093  0.          0.          0.
   0.         -0.          0.          0.          0.21197075  0.28585196
  -0.        ]
 [ 0.          0.48763399  0.34127086  0.02635054  0.38541002  0.20585683
   0.          0.0138876   0.          0.          0.          0.01603164
   0.        ]
 [-0.          0.13135885  0.          0.51311321  0.02195094  0.2759881
   0.          0.          0.21197075  0.          0.          0.13259067
   0.17883609]
 [ 0.          0.          0.81837209  0.17767663  0.00878754  0.06168298
   0.          0.          0.28585196  0.01603164  0.13259067  0.
   0.1548196 ]
 [ 0.          0.1807707   0.          0.10671197  0.          0.
   0.0513585   0.57658514 -0.          0.          0.17883609  0.1548196
   0.        ]]
* conn_model : sps
* coords : [[-40.  32. -10.]
 [-54. -38.  34.]
 [  6.  22.  28.]
 [-50. -10.  -6.]
 [ 48.  16.  14.]
 [ 52.  -6.  -8.]
 [ 48.  32.   4.]
 [ 38.   2.   0.]
 [  6. -16.  40.]
 [ 54. -36.  36.]
 [ -6. -18.  40.]
 [ -6.  20.  32.]
 [-36.   2.   0.]]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* function_str : def thresh_func(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                smooth, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                hpass):
    """
    Threshold a functional connectivity matrix using any of a variety of methods.

    Parameters
    ----------
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    thr : float
        A value, between 0 and 1, to threshold the graph using any variety of methods
        triggered through other options.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.

    Returns
    -------
    conn_matrix_thr : array
        Weighted, thresholded, NxN matrix.
    edge_threshold : str
        The string percentage representation of thr.
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy array in .npy format.
    thr : float
        The value, between 0 and 1, used to threshold the graph using any variety of methods
        triggered through other options.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    ID : str
        A subject id or other unique identifier.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets import utils, thresholding

    thr_perc = 100 * float(thr)
    edge_threshold = "%s%s" % (str(thr_perc), '%')
    if parc is True:
        node_size = 'parc'

    if np.count_nonzero(conn_matrix) == 0:
        raise ValueError('ERROR: Raw connectivity matrix contains only zeros.')

    # Save unthresholded
    unthr_path = utils.create_unthr_path(ID, network, conn_model, roi, dir_path)
    utils.save_mat(conn_matrix, unthr_path)

    if min_span_tree is True:
        print('Using local thresholding option with the Minimum Spanning Tree (MST)...\n')
        if dens_thresh is False:
            thr_type = 'MSTprop'
            conn_matrix_thr = thresholding.local_thresholding_prop(conn_matrix, thr)
        else:
            thr_type = 'MSTdens'
            conn_matrix_thr = thresholding.local_thresholding_dens(conn_matrix, thr)
    elif disp_filt is True:
        thr_type = 'DISP_alpha'
        G1 = thresholding.disparity_filter(nx.from_numpy_array(conn_matrix))
        # G2 = nx.Graph([(u, v, d) for u, v, d in G1.edges(data=True) if d['alpha'] < thr])
        print('Computing edge disparity significance with alpha = %s' % thr)
        print('Filtered graph: nodes = %s, edges = %s' % (G1.number_of_nodes(), G1.number_of_edges()))
        # print('Backbone graph: nodes = %s, edges = %s' % (G2.number_of_nodes(), G2.number_of_edges()))
        # print(G2.edges(data=True))
        conn_matrix_thr = nx.to_numpy_array(G1)
    else:
        if dens_thresh is False:
            thr_type = 'prop'
            print("%s%.2f%s" % ('\nThresholding proportionally at: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.threshold_proportional(conn_matrix, float(thr))
        else:
            thr_type = 'dens'
            print("%s%.2f%s" % ('\nThresholding to achieve density of: ', thr_perc, '% ...\n'))
            conn_matrix_thr = thresholding.density_thresholding(conn_matrix, float(thr))

    if not nx.is_connected(nx.from_numpy_matrix(conn_matrix_thr)):
        print('Warning: Fragmented graph')

    # Save thresholded mat
    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size, smooth, c_boot,
                                          thr_type, hpass, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, smooth, prune, ID, dir_path, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass

* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* thr : 0.16
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix_thr : [[0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.55010506 0.
  0.         0.         0.         0.48763399 0.         0.
  0.        ]
 [0.         0.         0.         0.3648176  0.         0.
  0.         0.         0.         0.34127086 0.         0.81837209
  0.        ]
 [0.         0.         0.3648176  0.         0.         0.52889168
  0.         0.         0.         0.         0.51311321 0.
  0.        ]
 [0.         0.55010506 0.         0.         0.         0.
  0.50764837 0.         0.         0.38541002 0.         0.
  0.        ]
 [0.         0.         0.         0.52889168 0.         0.
  0.         0.33071176 0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.50764837 0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.33071176
  0.         0.         0.         0.         0.         0.
  0.57658514]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.48763399 0.34127086 0.         0.38541002 0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.51311321 0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.81837209 0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.57658514 0.         0.         0.         0.
  0.        ]]
* conn_model : sps
* coords : [[-40.  32. -10.]
 [-54. -38.  34.]
 [  6.  22.  28.]
 [-50. -10.  -6.]
 [ 48.  16.  14.]
 [ 52.  -6.  -8.]
 [ 48.  32.   4.]
 [ 38.   2.   0.]
 [  6. -16.  40.]
 [ 54. -36.  36.]
 [ -6. -18.  40.]
 [ -6.  20.  32.]
 [-36.   2.   0.]]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* edge_threshold : 16.0%
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/0021001_SalVentAttn_est_sps_0.16dens4_mm.npy
* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan]
* network : SalVentAttn
* node_size : 4
* norm : 0
* prune : 1
* roi : None
* smooth : 0
* thr : 0.16
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 0.021006
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/thresh_func_node/mapflow/_thresh_func_node1


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

