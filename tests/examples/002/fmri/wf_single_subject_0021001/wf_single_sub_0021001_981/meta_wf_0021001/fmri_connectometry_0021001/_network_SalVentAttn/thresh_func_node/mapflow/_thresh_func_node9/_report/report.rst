Node: utility
=============


 Hierarchy : _thresh_func_node9
 Exec ID : _thresh_func_node9


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[ 1.          0.06227798  0.32326457  0.24302655  0.1213804  -0.03188517
   0.02481395  0.08537625  0.68171847 -0.23114818 -0.04812953 -0.00315837
   0.00359861 -0.06957115  0.08596162]
 [ 0.06227799  1.         -0.03717826  0.21330476  0.4097675   0.07464918
   0.01839136  0.06352065  0.00713484  0.71727407 -0.08799863 -0.12721597
  -0.15059516 -0.04714928 -0.16797777]
 [ 0.32326448 -0.03717822  1.         -0.07031117  0.06356345  0.05054064
  -0.02079634  0.17908119 -0.33067957  0.06457541  0.3399143   0.00969351
   0.09130301 -0.07294445 -0.11511344]
 [ 0.24302636  0.21330501 -0.07031115  1.          0.0451672   0.13451321
  -0.03349867  0.02670339 -0.00551622  0.04545967  0.03391681  0.61090386
   0.09749731 -0.10408276 -0.08069981]
 [ 0.12138028  0.40976748  0.06356354  0.04516723  1.          0.12676506
  -0.14194371  0.11451716 -0.09215576  0.03808055 -0.01994531 -0.29664618
   0.3554389   0.01351115  0.05419266]
 [-0.0318849   0.07464919  0.0505405   0.13451295  0.12676518  1.
   0.22502723  0.14174527 -0.04452883 -0.03213083 -0.15994282 -0.06350772
   0.03730256  0.70516825  0.20095474]
 [ 0.0248138   0.01839143 -0.02079621 -0.03349868 -0.1419438   0.22502758
   1.          0.4031862  -0.06000669  0.16925149  0.32316646 -0.03201042
   0.01140793 -0.12993984 -0.1426678 ]
 [ 0.08537645  0.06352062  0.17908107  0.02670343  0.11451707  0.14174512
   0.40318626  1.          0.04785818 -0.26522976 -0.04252403 -0.01489377
  -0.11166585 -0.05001271  0.42754248]
 [ 0.6817187   0.00713485 -0.3306797  -0.0055164  -0.09215587 -0.04452864
  -0.06000687  0.04785844  1.         -0.00531833  0.02699951  0.2194518
   0.1242663   0.04461207  0.20582199]
 [-0.2311482   0.7172742   0.06457546  0.04546003  0.03808057 -0.03213101
   0.16925162 -0.2652299  -0.00531844  1.          0.0783936   0.23674992
   0.08400593  0.00255005  0.23803   ]
 [-0.04812938 -0.0879988   0.33991432  0.03391672 -0.01994522 -0.15994297
   0.32316652 -0.0425241   0.02699943  0.07839379  1.         -0.06844413
   0.28218624  0.2452839   0.23289578]
 [-0.00315846 -0.12721612  0.00969351  0.6109037  -0.2966461  -0.06350784
  -0.03201044 -0.01489367  0.21945184  0.23675016 -0.06844414  1.
   0.22487392  0.07656612 -0.01048326]
 [ 0.00359871 -0.15059502  0.09130294  0.09749725  0.35543883  0.03730264
   0.01140788 -0.11166581  0.12426615  0.08400585  0.28218624  0.22487396
   1.          0.09178919 -0.2320783 ]
 [-0.06957128 -0.04714925 -0.07294433 -0.10408259  0.01351099  0.70516807
  -0.12993957 -0.05001294  0.04461206  0.00254996  0.24528377  0.07656607
   0.09178921  1.          0.17361528]
 [ 0.08596125 -0.16797762 -0.11511334 -0.08069976  0.0541926   0.20095505
  -0.14266779  0.4275425   0.20582238  0.23802973  0.23289578 -0.01048324
  -0.23207825  0.17361492  1.        ]]
* conn_model : partcorr
* coords : [[ 54.18698939 -35.04665191  36.22738032]
 [-13.29384318 -67.50893365  -5.87891203]
 [  7.53763026 -24.85488228  58.69490544]
 [ -5.21337127 -18.39260313  39.69630156]
 [ 22.6098635   -4.56230735 -31.95640687]
 [-29.8504807  -89.22470943   1.50272636]
 [ -6.17316943 -25.68163193  57.79975058]
 [ 20.5507772   -2.83523316  -0.70777202]
 [ -4.04270938 -57.24758091 -38.75942609]
 [  5.23006135  37.48432175 -17.26993865]
 [ 24.10399334 -29.37895175 -18.43427621]
 [-22.87226126  -5.03183134 -32.04257958]
 [ 34.79281102 -43.04760528 -20.99951817]
 [-35.83066274  12.09024484  47.08011793]
 [ 13.6988191  -66.42386874  -5.03918905]]
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
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* smooth : 0
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[ 0.          0.06227798  0.32326457  0.24302655  0.1213804  -0.03188517
   0.02481395  0.08537625  0.68171847 -0.23114818 -0.04812953 -0.00315837
   0.00359861 -0.06957115  0.08596162]
 [ 0.06227799  0.         -0.03717826  0.21330476  0.4097675   0.07464918
   0.01839136  0.06352065  0.00713484  0.71727407 -0.08799863 -0.12721597
  -0.15059516 -0.04714928 -0.16797777]
 [ 0.32326448 -0.03717822  0.         -0.07031117  0.06356345  0.05054064
  -0.02079634  0.17908119 -0.33067957  0.06457541  0.3399143   0.00969351
   0.09130301 -0.07294445 -0.11511344]
 [ 0.24302636  0.21330501 -0.07031115  0.          0.0451672   0.13451321
  -0.03349867  0.02670339 -0.00551622  0.04545967  0.03391681  0.61090386
   0.09749731 -0.10408276 -0.08069981]
 [ 0.12138028  0.40976748  0.06356354  0.04516723  0.          0.12676506
  -0.14194371  0.11451716 -0.09215576  0.03808055 -0.01994531 -0.29664618
   0.3554389   0.01351115  0.05419266]
 [-0.0318849   0.07464919  0.0505405   0.13451295  0.12676518  0.
   0.22502723  0.14174527 -0.04452883 -0.03213083 -0.15994282 -0.06350772
   0.03730256  0.70516825  0.20095474]
 [ 0.0248138   0.01839143 -0.02079621 -0.03349868 -0.1419438   0.22502758
   0.          0.4031862  -0.06000669  0.16925149  0.32316646 -0.03201042
   0.01140793 -0.12993984 -0.1426678 ]
 [ 0.08537645  0.06352062  0.17908107  0.02670343  0.11451707  0.14174512
   0.40318626  0.          0.04785818 -0.26522976 -0.04252403 -0.01489377
  -0.11166585 -0.05001271  0.42754248]
 [ 0.6817187   0.00713485 -0.3306797  -0.0055164  -0.09215587 -0.04452864
  -0.06000687  0.04785844  0.         -0.00531833  0.02699951  0.2194518
   0.1242663   0.04461207  0.20582199]
 [-0.2311482   0.7172742   0.06457546  0.04546003  0.03808057 -0.03213101
   0.16925162 -0.2652299  -0.00531844  0.          0.0783936   0.23674992
   0.08400593  0.00255005  0.23803   ]
 [-0.04812938 -0.0879988   0.33991432  0.03391672 -0.01994522 -0.15994297
   0.32316652 -0.0425241   0.02699943  0.07839379  0.         -0.06844413
   0.28218624  0.2452839   0.23289578]
 [-0.00315846 -0.12721612  0.00969351  0.6109037  -0.2966461  -0.06350784
  -0.03201044 -0.01489367  0.21945184  0.23675016 -0.06844414  0.
   0.22487392  0.07656612 -0.01048326]
 [ 0.00359871 -0.15059502  0.09130294  0.09749725  0.35543883  0.03730264
   0.01140788 -0.11166581  0.12426615  0.08400585  0.28218624  0.22487396
   0.          0.09178919 -0.2320783 ]
 [-0.06957128 -0.04714925 -0.07294433 -0.10408259  0.01351099  0.70516807
  -0.12993957 -0.05001294  0.04461206  0.00254996  0.24528377  0.07656607
   0.09178921  0.          0.17361528]
 [ 0.08596125 -0.16797762 -0.11511334 -0.08069976  0.0541926   0.20095505
  -0.14266779  0.4275425   0.20582238  0.23802973  0.23289578 -0.01048324
  -0.23207825  0.17361492  0.        ]]
* conn_model : partcorr
* coords : [[ 54.18698939 -35.04665191  36.22738032]
 [-13.29384318 -67.50893365  -5.87891203]
 [  7.53763026 -24.85488228  58.69490544]
 [ -5.21337127 -18.39260313  39.69630156]
 [ 22.6098635   -4.56230735 -31.95640687]
 [-29.8504807  -89.22470943   1.50272636]
 [ -6.17316943 -25.68163193  57.79975058]
 [ 20.5507772   -2.83523316  -0.70777202]
 [ -4.04270938 -57.24758091 -38.75942609]
 [  5.23006135  37.48432175 -17.26993865]
 [ 24.10399334 -29.37895175 -18.43427621]
 [-22.87226126  -5.03183134 -32.04257958]
 [ 34.79281102 -43.04760528 -20.99951817]
 [-35.83066274  12.09024484  47.08011793]
 [ 13.6988191  -66.42386874  -5.03918905]]
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
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* smooth : 0
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix_thr : [[0.         0.         0.32326457 0.24302655 0.         0.
  0.         0.         0.68171847 0.         0.         0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.4097675  0.
  0.         0.         0.         0.71727407 0.         0.
  0.         0.         0.        ]
 [0.32326448 0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.3399143  0.
  0.         0.         0.        ]
 [0.24302636 0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.61090386
  0.         0.         0.        ]
 [0.         0.40976748 0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.3554389  0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.70516825 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.4031862  0.         0.         0.32316646 0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.40318626 0.         0.         0.         0.         0.
  0.         0.         0.42754248]
 [0.6817187  0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.        ]
 [0.         0.7172742  0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.23674992
  0.         0.         0.23803   ]
 [0.         0.         0.33991432 0.         0.         0.
  0.32316652 0.         0.         0.         0.         0.
  0.28218624 0.2452839  0.23289578]
 [0.         0.         0.         0.6109037  0.         0.
  0.         0.         0.         0.23675016 0.         0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.35543883 0.
  0.         0.         0.         0.         0.28218624 0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.70516807
  0.         0.         0.         0.         0.24528377 0.
  0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.4275425  0.         0.23802973 0.23289578 0.
  0.         0.         0.        ]]
* conn_model : partcorr
* coords : [[ 54.18698939 -35.04665191  36.22738032]
 [-13.29384318 -67.50893365  -5.87891203]
 [  7.53763026 -24.85488228  58.69490544]
 [ -5.21337127 -18.39260313  39.69630156]
 [ 22.6098635   -4.56230735 -31.95640687]
 [-29.8504807  -89.22470943   1.50272636]
 [ -6.17316943 -25.68163193  57.79975058]
 [ 20.5507772   -2.83523316  -0.70777202]
 [ -4.04270938 -57.24758091 -38.75942609]
 [  5.23006135  37.48432175 -17.26993865]
 [ 24.10399334 -29.37895175 -18.43427621]
 [-22.87226126  -5.03183134 -32.04257958]
 [ 34.79281102 -43.04760528 -20.99951817]
 [-35.83066274  12.09024484  47.08011793]
 [ 13.6988191  -66.42386874  -5.03918905]]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* edge_threshold : 18.0%
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/graphs/0021001_SalVentAttn_est_partcorr_0.18densparc_mm.npy
* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan]
* network : SalVentAttn
* node_size : parc
* norm : 0
* prune : 1
* roi : None
* smooth : 0
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Runtime info
------------


* duration : 0.030583
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/thresh_func_node/mapflow/_thresh_func_node9


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
* OLDPWD : /Users/derekpisner/Applications/PyNets/tests
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

