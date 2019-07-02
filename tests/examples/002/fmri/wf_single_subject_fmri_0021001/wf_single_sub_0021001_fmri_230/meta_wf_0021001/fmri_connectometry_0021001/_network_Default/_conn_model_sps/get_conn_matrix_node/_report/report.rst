Node: meta_wf_0021001 (fmri_connectometry_0021001 (get_conn_matrix_node (utility)
=================================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.get_conn_matrix_node
 Exec ID : get_conn_matrix_node.bI.b0.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(-40.0, 32.0, -10.0), (46.0, -66.0, 30.0), (52.0, -6.0, -8.0), (-12.0, 34.0, 42.0), (-36.0, 12.0, 48.0), (42.0, 34.0, -12.0), (-50.0, -10.0, -6.0), (10.0, -58.0, 38.0), (8.0, -44.0, 20.0), (48.0, 32.0, 4.0), (6.0, -16.0, 40.0), (-6.0, -18.0, 40.0), (-46.0, 14.0, 12.0), (-46.0, 32.0, 6.0), (-8.0, -60.0, 38.0), (-22.0, -32.0, -18.0), (-58.0, -28.0, -12.0), (-42.0, -70.0, 32.0), (-6.0, 40.0, 6.0), (58.0, -26.0, -12.0), (6.0, 38.0, 2.0), (-6.0, -46.0, 20.0), (6.0, 38.0, -18.0), (-6.0, 42.0, -16.0)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* function_str : def get_conn_matrix(time_series, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
                    disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                    hpass):
    """
    Computes a functional connectivity matrix based on a node-extracted time-series array.
    Includes a library of routines across Nilearn, scikit-learn, and skggm packages, among others.

    Parameters
    ----------
    time_series : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
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
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
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
        List of string labels corresponding to graph nodes.
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
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphicalLassoCV

    conn_matrix = None
    if conn_model == 'corr' or conn_model == 'cor' or conn_model == 'correlation':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'partcorr' or conn_model == 'parcorr' or conn_model == 'partialcorrelation':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar' or conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
        # Fit estimator to matrix to get sparse matrix
        estimator_shrunk = None
        estimator = GraphicalLassoCV(cv=5)
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            try:
                from sklearn.covariance import GraphicalLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8, 0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphicalLasso(alpha)
                            estimator_shrunk.fit(shrunk_cov)
                            print("Retrying covariance matrix estimate with alpha=%s" % alpha)
                            if estimator_shrunk is None:
                                pass
                            else:
                                break
                        except:
                            print("Covariance estimation failed with shrinkage at alpha=%s" % alpha)
                            continue
            except ValueError:
                print('Unstable Lasso estimation! Shrinkage failed. A different connectivity model may be needed.')
        if estimator is None and estimator_shrunk is None:
            raise RuntimeError('\nERROR: Covariance estimation failed.')
        if conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
            if estimator_shrunk is None:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            else:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar':
            if estimator_shrunk is None:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            else:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphicalLasso':
        try:
            from inverse_covariance import QuicGraphicalLasso
        except ImportError:
            print('Cannot run QuicGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphicalLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphLassoCV':
        try:
            from inverse_covariance import QuicGraphicalLassoCV
        except ImportError:
            print('Cannot run QuicGraphLassoCV. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphicalLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphicalLassoEBIC':
        try:
            from inverse_covariance import QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run QuicGraphLassoEBIC. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphicalLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'AdaptiveQuicGraphLasso':
        try:
            from inverse_covariance import AdaptiveQuicGraphicalLasso, QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run AdaptiveGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveQuicGraphicalLasso(
            estimator=QuicGraphicalLassoEBIC(
                init_method='cov',
            ),
            method='binary',
        )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_
    else:
        raise ValueError('\nERROR! No connectivity model specified at runtime. Select a valid estimator using the '
                         '-mod flag.')

    if conn_matrix.shape < (2, 2):
        raise RuntimeError('\nERROR! Matrix estimation selection yielded an empty or 1-dimensional graph. '
                           'Check time-series for errors or try using a different atlas')

    coords = np.array(coords)
    labels = np.array(labels)
    return conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : Default
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 1.3127716   1.1574162   1.4838659  ...  0.67249227 -0.50939476
  -0.8282963 ]
 [-0.77208716 -0.22721355 -0.05433574 ...  0.80564415  0.7557712
   0.24824011]
 [ 0.7586903  -1.3740423  -0.9933575  ...  0.07429074 -0.32368442
  -1.6436886 ]
 ...
 [ 0.8830816   0.68389726 -0.02967551 ...  0.44515112  0.86113846
   0.32610065]
 [ 0.8627658   1.8771849   1.1543678  ...  1.9672911   0.56556803
   0.17502418]
 [ 0.11045323 -0.01386644  0.0496253  ...  2.1081576  -0.87343735
  -0.60447407]]
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(-40.0, 32.0, -10.0), (46.0, -66.0, 30.0), (52.0, -6.0, -8.0), (-12.0, 34.0, 42.0), (-36.0, 12.0, 48.0), (42.0, 34.0, -12.0), (-50.0, -10.0, -6.0), (10.0, -58.0, 38.0), (8.0, -44.0, 20.0), (48.0, 32.0, 4.0), (6.0, -16.0, 40.0), (-6.0, -18.0, 40.0), (-46.0, 14.0, 12.0), (-46.0, 32.0, 6.0), (-8.0, -60.0, 38.0), (-22.0, -32.0, -18.0), (-58.0, -28.0, -12.0), (-42.0, -70.0, 32.0), (-6.0, 40.0, 6.0), (58.0, -26.0, -12.0), (6.0, 38.0, 2.0), (-6.0, -46.0, 20.0), (6.0, 38.0, -18.0), (-6.0, 42.0, -16.0)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* function_str : def get_conn_matrix(time_series, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
                    disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                    hpass):
    """
    Computes a functional connectivity matrix based on a node-extracted time-series array.
    Includes a library of routines across Nilearn, scikit-learn, and skggm packages, among others.

    Parameters
    ----------
    time_series : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
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
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
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
        List of string labels corresponding to graph nodes.
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
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphicalLassoCV

    conn_matrix = None
    if conn_model == 'corr' or conn_model == 'cor' or conn_model == 'correlation':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'partcorr' or conn_model == 'parcorr' or conn_model == 'partialcorrelation':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar' or conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
        # Fit estimator to matrix to get sparse matrix
        estimator_shrunk = None
        estimator = GraphicalLassoCV(cv=5)
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            try:
                from sklearn.covariance import GraphicalLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8, 0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphicalLasso(alpha)
                            estimator_shrunk.fit(shrunk_cov)
                            print("Retrying covariance matrix estimate with alpha=%s" % alpha)
                            if estimator_shrunk is None:
                                pass
                            else:
                                break
                        except:
                            print("Covariance estimation failed with shrinkage at alpha=%s" % alpha)
                            continue
            except ValueError:
                print('Unstable Lasso estimation! Shrinkage failed. A different connectivity model may be needed.')
        if estimator is None and estimator_shrunk is None:
            raise RuntimeError('\nERROR: Covariance estimation failed.')
        if conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
            if estimator_shrunk is None:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            else:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar':
            if estimator_shrunk is None:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            else:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphicalLasso':
        try:
            from inverse_covariance import QuicGraphicalLasso
        except ImportError:
            print('Cannot run QuicGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphicalLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphLassoCV':
        try:
            from inverse_covariance import QuicGraphicalLassoCV
        except ImportError:
            print('Cannot run QuicGraphLassoCV. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphicalLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphicalLassoEBIC':
        try:
            from inverse_covariance import QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run QuicGraphLassoEBIC. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphicalLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'AdaptiveQuicGraphLasso':
        try:
            from inverse_covariance import AdaptiveQuicGraphicalLasso, QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run AdaptiveGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveQuicGraphicalLasso(
            estimator=QuicGraphicalLassoEBIC(
                init_method='cov',
            ),
            method='binary',
        )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_
    else:
        raise ValueError('\nERROR! No connectivity model specified at runtime. Select a valid estimator using the '
                         '-mod flag.')

    if conn_matrix.shape < (2, 2):
        raise RuntimeError('\nERROR! Matrix estimation selection yielded an empty or 1-dimensional graph. '
                           'Check time-series for errors or try using a different atlas')

    coords = np.array(coords)
    labels = np.array(labels)
    return conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : Default
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 1.3127716   1.1574162   1.4838659  ...  0.67249227 -0.50939476
  -0.8282963 ]
 [-0.77208716 -0.22721355 -0.05433574 ...  0.80564415  0.7557712
   0.24824011]
 [ 0.7586903  -1.3740423  -0.9933575  ...  0.07429074 -0.32368442
  -1.6436886 ]
 ...
 [ 0.8830816   0.68389726 -0.02967551 ...  0.44515112  0.86113846
   0.32610065]
 [ 0.8627658   1.8771849   1.1543678  ...  1.9672911   0.56556803
   0.17502418]
 [ 0.11045323 -0.01386644  0.0496253  ...  2.1081576  -0.87343735
  -0.60447407]]
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[-1.1043601   0.          0.0565755   0.          0.04192131  0.30921281
  -0.         -0.         -0.01544353 -0.         -0.         -0.
   0.          0.          0.          0.          0.          0.05159311
  -0.         -0.          0.          0.         -0.          0.        ]
 [ 0.         -1.42976628 -0.          0.          0.          0.09893272
  -0.          0.          0.         -0.09044454  0.04549355  0.
   0.         -0.          0.          0.          0.38498149  0.50926787
   0.          0.03046538  0.          0.         -0.          0.        ]
 [ 0.0565755  -0.         -1.73502564  0.          0.          0.0357415
   0.55515666  0.13314762  0.14565982  0.16179839  0.          0.26183318
   0.          0.          0.          0.01090732  0.         -0.
   0.0207631   0.          0.27123468  0.          0.          0.        ]
 [ 0.          0.          0.         -1.42897807  0.          0.
   0.          0.          0.          0.          0.          0.13754909
   0.22665762  0.          0.05130301  0.          0.30209679  0.03654139
   0.04170026  0.07886333  0.36784109 -0.          0.         -0.        ]
 [ 0.04192131  0.          0.          0.         -1.16011597  0.0261716
  -0.          0.07123918  0.          0.          0.          0.
   0.0903191   0.00867396  0.10761162  0.         -0.          0.31296789
   0.00245109 -0.          0.          0.          0.          0.02304423]
 [ 0.30921281  0.09893272  0.0357415   0.          0.0261716  -1.14036184
   0.          0.          0.          0.          0.          0.
   0.         -0.          0.          0.06618406  0.          0.04404083
   0.          0.04243663  0.          0.          0.          0.05988228]
 [-0.         -0.          0.55515666  0.         -0.          0.
  -2.05037071  0.30206754  0.          0.24842529  0.          0.55797326
   0.          0.          0.13023497  0.03041031  0.         -0.
   0.          0.25548575  0.07378266  0.          0.          0.        ]
 [-0.          0.          0.13314762  0.          0.07123918  0.
   0.30206754 -1.62467074  0.16139037  0.11259965  0.          0.
   0.          0.          0.40959456  0.          0.          0.
   0.          0.13126381  0.          0.17349245  0.          0.        ]
 [-0.01544353  0.          0.14565982  0.          0.          0.
   0.          0.16139037 -2.10217957  0.100512    0.18331692  0.24357845
   0.         -0.          0.28370345  0.          0.          0.
   0.26938497  0.          0.07127912  0.77849964  0.04032132  0.00505052]
 [-0.         -0.09044454  0.16179839  0.          0.          0.
   0.24842529  0.11259965  0.100512   -1.38995824  0.          0.
   0.16529553  0.37641869  0.          0.         -0.         -0.0735861
   0.          0.08656837  0.          0.          0.          0.        ]
 [-0.          0.04549355  0.          0.          0.          0.
   0.          0.          0.18331692  0.         -1.53161172  0.15628747
   0.         -0.          0.16694612  0.08997745  0.          0.
   0.02470645  0.          0.05877751  0.48196638  0.          0.14104226]
 [-0.          0.          0.26183318  0.13754909  0.          0.
   0.55797326  0.          0.24357845  0.          0.15628747 -1.69364273
   0.          0.          0.          0.          0.         -0.
   0.04777646  0.          0.10579463  0.          0.          0.        ]
 [ 0.          0.          0.          0.22665762  0.0903191   0.
   0.          0.          0.          0.16529553  0.          0.
  -1.17125314  0.1029374   0.10168142  0.          0.02376775  0.
   0.04833472  0.03219124  0.          0.         -0.          0.        ]
 [ 0.         -0.          0.          0.          0.00867396 -0.
   0.          0.         -0.          0.37641869 -0.          0.
   0.1029374  -1.15588965 -0.         -0.05296467 -0.11269795 -0.05971128
  -0.         -0.         -0.         -0.          0.          0.        ]
 [ 0.          0.          0.          0.05130301  0.10761162  0.
   0.13023497  0.40959456  0.28370345  0.          0.16694612  0.
   0.10168142 -0.         -1.6807008   0.          0.29465923  0.05671613
   0.          0.06880885  0.          0.04762927  0.          0.        ]
 [ 0.          0.          0.01090732  0.          0.          0.06618406
   0.03041031  0.          0.          0.          0.08997745  0.
   0.         -0.05296467  0.         -1.36520166  0.          0.00689807
   0.55269143  0.          0.          0.          0.0210227   0.17255287]
 [ 0.          0.38498149  0.          0.30209679 -0.          0.
   0.          0.          0.         -0.          0.          0.
   0.02376775 -0.11269795  0.29465923  0.         -1.63812159  0.13702306
   0.          0.42059064  0.          0.          0.         -0.        ]
 [ 0.05159311  0.50926787 -0.          0.03654139  0.31296789  0.04404083
  -0.          0.          0.         -0.0735861   0.         -0.
   0.         -0.05971128  0.05671613  0.00689807  0.13702306 -1.57014065
   0.22712283  0.          0.          0.          0.          0.21770009]
 [-0.          0.          0.0207631   0.04170026  0.00245109  0.
   0.          0.          0.26938497  0.          0.02470645  0.04777646
   0.04833472 -0.          0.          0.55269143  0.          0.22712283
  -1.80590844  0.24928333  0.23192777  0.          0.26193451  0.12744102]
 [-0.          0.03046538  0.          0.07886333 -0.          0.04243663
   0.25548575  0.13126381  0.          0.08656837  0.          0.
   0.03219124 -0.          0.06880885  0.          0.42059064  0.
   0.24928333 -1.51305062  0.         -0.          0.         -0.        ]
 [ 0.          0.          0.27123468  0.36784109  0.          0.
   0.07378266  0.          0.07127912  0.          0.05877751  0.10579463
   0.         -0.          0.          0.          0.          0.
   0.23192777  0.         -1.46668663  0.05319155  0.          0.00797935]
 [ 0.          0.          0.         -0.          0.          0.
   0.          0.17349245  0.77849964  0.          0.48196638  0.
   0.         -0.          0.04762927  0.          0.          0.
   0.         -0.          0.05319155 -1.82400159  0.          0.        ]
 [-0.         -0.          0.          0.          0.          0.
   0.          0.          0.04032132  0.          0.          0.
  -0.          0.          0.          0.0210227   0.          0.
   0.26193451  0.          0.          0.         -1.17184264  0.2544698 ]
 [ 0.          0.          0.         -0.          0.02304423  0.05988228
   0.          0.          0.00505052  0.          0.14104226  0.
   0.          0.          0.          0.17255287 -0.          0.21770009
   0.12744102 -0.          0.00797935  0.          0.2544698  -1.28146616]]
* conn_model : sps
* coords : [[-40.  32. -10.]
 [ 46. -66.  30.]
 [ 52.  -6.  -8.]
 [-12.  34.  42.]
 [-36.  12.  48.]
 [ 42.  34. -12.]
 [-50. -10.  -6.]
 [ 10. -58.  38.]
 [  8. -44.  20.]
 [ 48.  32.   4.]
 [  6. -16.  40.]
 [ -6. -18.  40.]
 [-46.  14.  12.]
 [-46.  32.   6.]
 [ -8. -60.  38.]
 [-22. -32. -18.]
 [-58. -28. -12.]
 [-42. -70.  32.]
 [ -6.  40.   6.]
 [ 58. -26. -12.]
 [  6.  38.   2.]
 [ -6. -46.  20.]
 [  6.  38. -18.]
 [ -6.  42. -16.]]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
 nan nan nan nan nan nan]
* min_span_tree : False
* network : Default
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 1.199158
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/_conn_model_sps/get_conn_matrix_node


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

