Node: meta_wf_0021001 (fmri_connectometry_0021001 (get_conn_matrix_node (utility)
=================================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.get_conn_matrix_node
 Exec ID : get_conn_matrix_node.bI.b1.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : partcorr
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
* conn_model : partcorr
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
* conn_matrix : [[ 1.00000000e+00  9.85934585e-02  2.78947890e-01 -4.81935106e-02
   2.30456074e-03  3.91249955e-01  1.77284554e-02 -2.82068532e-02
  -2.93370634e-01 -4.63284925e-02 -7.49200732e-02 -2.87241390e-04
   1.72864437e-01  7.41519257e-02  1.55661136e-01  1.53189018e-01
  -3.13304141e-02  1.38484836e-01 -7.08642080e-02 -2.03377545e-01
   2.10725926e-02  8.53452533e-02 -1.04868233e-01 -1.92561224e-02]
 [ 9.85935330e-02  1.00000000e+00 -1.34854555e-01 -1.91311957e-03
  -9.87252071e-02  1.49638563e-01 -1.38938800e-01  1.20703824e-01
  -3.64525504e-02 -2.13083610e-01  1.48372427e-01  1.10847063e-01
  -2.22783312e-02  1.19889513e-01 -2.81969812e-02  1.91361662e-02
   2.30173603e-01  2.77429044e-01 -4.98751178e-02  2.21271366e-01
   4.09134105e-02  3.48898843e-02 -1.34006247e-01  9.44997966e-02]
 [ 2.78947800e-01 -1.34854525e-01  1.00000000e+00  5.40004745e-02
   3.21644731e-02  1.09136462e-01  3.15061450e-01  1.60468191e-01
   1.53392732e-01  1.19386137e-01 -8.22728127e-02  2.03601390e-01
  -1.43782616e-01 -1.74196884e-02 -8.90278965e-02 -5.09186238e-02
   6.55580908e-02 -4.89867218e-02  1.16434887e-01 -1.27951398e-01
   1.47257462e-01 -4.55918722e-03 -5.53526282e-02  8.60927776e-02]
 [-4.81934957e-02 -1.91313541e-03  5.40004820e-02  1.00000000e+00
  -2.82737501e-02  6.50871396e-02  6.12838566e-02 -1.20987751e-01
   6.63420856e-02  8.23715031e-02 -8.38399157e-02  5.01140766e-02
   9.05970037e-02  1.01433441e-01  7.04932734e-02 -4.58824448e-02
   2.19745964e-01  1.63976654e-01  5.60026579e-02  1.02169942e-02
   3.49345177e-01 -1.06991336e-01 -4.19632122e-02 -1.28874108e-01]
 [ 2.30459752e-03 -9.87252519e-02  3.21645252e-02 -2.82737147e-02
   1.00000000e+00  1.56542212e-01 -3.84935215e-02  1.79550350e-01
  -8.91878735e-03 -1.41312303e-02  6.14549965e-02  2.06407905e-02
   1.51957944e-01  1.37857243e-01  1.72306076e-01 -3.75525877e-02
  -2.60064185e-01  3.99442852e-01  9.91950482e-02 -6.49313480e-02
  -8.83352831e-02 -5.76017424e-02 -6.97079077e-02  9.34466068e-03]
 [ 3.91250044e-01  1.49638548e-01  1.09136388e-01  6.50871769e-02
   1.56542212e-01  1.00000000e+00 -8.66331309e-02 -6.25303686e-02
  -3.90166626e-03  1.98265672e-01  2.59432150e-03 -8.55820104e-02
  -8.05570111e-02 -1.81652129e-01 -1.41930073e-01  6.60002744e-03
  -4.26381221e-03 -4.14001793e-02 -2.89945602e-02  2.21505836e-01
  -2.81798765e-02  9.07873064e-02  8.88037905e-02  1.08884715e-01]
 [ 1.77284870e-02 -1.38938785e-01  3.15061629e-01  6.12839609e-02
  -3.84934470e-02 -8.66332278e-02  1.00000000e+00  1.95135847e-01
  -5.12749441e-02  4.40958180e-02  1.09806202e-01  2.88385004e-01
  -5.57204150e-02 -3.68736573e-02  1.65233001e-01  1.30488694e-01
   3.58430184e-02 -2.03029156e-01  1.34774065e-03  2.54143089e-01
   9.89528224e-02 -7.58556798e-02 -6.49389327e-02  7.93260410e-02]
 [-2.82068010e-02  1.20703712e-01  1.60468027e-01 -1.20987698e-01
   1.79550245e-01 -6.25304058e-02  1.95135742e-01  1.00000000e+00
   7.09594712e-02  6.45228475e-02 -4.52979989e-02 -7.24954829e-02
  -9.84639525e-02  1.09817147e-01  2.47446314e-01  1.19195003e-02
  -1.10650405e-01  6.33818805e-02 -1.38196930e-01  2.40843982e-01
   8.72783363e-02  1.85303107e-01  1.78272519e-02 -1.36008516e-01]
 [-2.93370485e-01 -3.64526324e-02  1.53392762e-01  6.63420781e-02
  -8.91884696e-03 -3.90164927e-03 -5.12751341e-02  7.09595457e-02
   1.00000000e+00  1.49635047e-01  1.01801734e-02  1.85280129e-01
   1.43519416e-01 -1.47424072e-01  1.17793091e-01  4.03607031e-03
  -2.61004511e-02  3.73244174e-02  4.44281362e-02  3.57112437e-02
  -2.79107783e-02  5.47877967e-01  1.63783625e-01  9.54209343e-02]
 [-4.63284627e-02 -2.13083595e-01  1.19386122e-01  8.23713988e-02
  -1.41312266e-02  1.98265687e-01  4.40958291e-02  6.45227134e-02
   1.49635062e-01  1.00000000e+00  6.10000603e-02 -8.34827945e-02
   1.57730177e-01  4.09801632e-01  1.44678831e-01  8.21616128e-02
  -7.31210113e-02 -1.85962856e-01 -8.33873078e-03  2.25950047e-01
  -5.31116687e-03 -1.40610352e-01 -2.18350720e-02  2.43033152e-02]
 [-7.49201179e-02  1.48372412e-01 -8.22727829e-02 -8.38399231e-02
   6.14549443e-02  2.59432755e-03  1.09806120e-01 -4.52980064e-02
   1.01799667e-02  6.10001311e-02  1.00000000e+00  1.67389631e-01
   3.66515405e-02 -6.20923452e-02  8.45994651e-02  6.17322586e-02
  -8.77464097e-03  2.27137003e-03  3.36838304e-03 -3.45879458e-02
   4.53494154e-02  3.36632401e-01  3.03931581e-03  1.37563318e-01]
 [-2.87291594e-04  1.10847108e-01  2.03601271e-01  5.01140282e-02
   2.06408128e-02 -8.55819955e-02  2.88385212e-01 -7.24955723e-02
   1.85279936e-01 -8.34827796e-02  1.67389512e-01  1.00000000e+00
  -7.25798542e-03  2.03859895e-01 -2.19330620e-02  4.13620062e-02
   1.02530181e-01 -5.41124716e-02  5.56493104e-02  6.28441852e-03
   7.22843558e-02 -9.77558121e-02 -5.45843318e-03 -1.02470718e-01]
 [ 1.72864437e-01 -2.22782902e-02 -1.43782601e-01  9.05970633e-02
   1.51957974e-01 -8.05570111e-02 -5.57203889e-02 -9.84639153e-02
   1.43519431e-01  1.57730177e-01  3.66515629e-02 -7.25806411e-03
   1.00000000e+00  2.65461206e-01  5.41787632e-02 -2.35336050e-02
   1.66402742e-01 -5.67240361e-03  1.65738657e-01  8.49895626e-02
   1.22486629e-01 -1.12391241e-01 -1.98068991e-01 -5.60260899e-02]
 [ 7.41519406e-02  1.19889490e-01 -1.74196996e-02  1.01433493e-01
   1.37857243e-01 -1.81652129e-01 -3.68736908e-02  1.09817214e-01
  -1.47424057e-01  4.09801513e-01 -6.20923266e-02  2.03859940e-01
   2.65461206e-01  1.00000000e+00 -8.73960331e-02 -2.08246902e-01
  -1.60853639e-01 -1.13825209e-01 -9.13369060e-02 -1.28164053e-01
  -8.83670077e-02  1.19172364e-01  1.95543095e-01  1.43789470e-01]
 [ 1.55661091e-01 -2.81969588e-02 -8.90279114e-02  7.04931840e-02
   1.72306135e-01 -1.41930029e-01  1.65233180e-01  2.47446299e-01
   1.17792957e-01  1.44678861e-01  8.45993906e-02 -2.19330452e-02
   5.41787855e-02 -8.73960629e-02  1.00000000e+00 -8.38633627e-02
   2.62258232e-01  1.29522681e-01 -2.14203391e-02 -2.36410741e-03
  -1.03323549e-01  1.27959922e-01  2.04546191e-02 -5.28685690e-04]
 [ 1.53189078e-01  1.91361792e-02 -5.09186350e-02 -4.58824411e-02
  -3.75525765e-02  6.60001067e-03  1.30488634e-01  1.19194947e-02
   4.03614622e-03  8.21615756e-02  6.17322400e-02  4.13620137e-02
  -2.35336274e-02 -2.08246931e-01 -8.38633478e-02  1.00000000e+00
  -1.02967672e-01  7.61788487e-02  3.48232210e-01 -4.09621978e-03
  -1.92401595e-02  4.04994339e-02  8.17079917e-02  1.20373793e-01]
 [-3.13304029e-02  2.30173424e-01  6.55581132e-02  2.19745979e-01
  -2.60064185e-01 -4.26380709e-03  3.58429998e-02 -1.10650405e-01
  -2.61004306e-02 -7.31210262e-02 -8.77461769e-03  1.02530174e-01
   1.66402757e-01 -1.60853639e-01  2.62258112e-01 -1.02967694e-01
   1.00000000e+00  2.03797758e-01 -5.65579683e-02  2.53496081e-01
  -1.01265110e-01  6.32909983e-02  3.81200500e-02 -1.88806467e-02]
 [ 1.38484791e-01  2.77429193e-01 -4.89867292e-02  1.63976654e-01
   3.99442852e-01 -4.14001606e-02 -2.03029096e-01  6.33818507e-02
   3.73244844e-02 -1.85962886e-01  2.27133860e-03 -5.41125052e-02
  -5.67235937e-03 -1.13825195e-01  1.29522666e-01  7.61789531e-02
   2.03797683e-01  1.00000000e+00  1.43321380e-01  1.41134793e-02
   8.19668099e-02 -1.26833260e-01  8.91934410e-02  2.25055188e-01]
 [-7.08641782e-02 -4.98752072e-02  1.16434827e-01  5.60026765e-02
   9.91950706e-02 -2.89945789e-02  1.34782086e-03 -1.38196930e-01
   4.44280952e-02 -8.33869446e-03  3.36836139e-03  5.56493290e-02
   1.65738642e-01 -9.13369283e-02 -2.14203838e-02  3.48232210e-01
  -5.65578826e-02  1.43321484e-01  1.00000000e+00  2.50844806e-01
   1.15405701e-01  7.43931606e-02  2.41554976e-01  1.11007527e-01]
 [-2.03377590e-01  2.21271425e-01 -1.27951279e-01  1.02170343e-02
  -6.49314597e-02  2.21505865e-01  2.54143000e-01  2.40843952e-01
   3.57112847e-02  2.25949958e-01 -3.45879495e-02  6.28443062e-03
   8.49895477e-02 -1.28163978e-01 -2.36401847e-03 -4.09621513e-03
   2.53495961e-01  1.41135724e-02  2.50844777e-01  1.00000000e+00
  -8.67718831e-02 -1.63896918e-01  4.50722128e-02 -1.76392183e-01]
 [ 2.10725889e-02  4.09134068e-02  1.47257537e-01  3.49345148e-01
  -8.83352980e-02 -2.81798560e-02  9.89528298e-02  8.72783735e-02
  -2.79108118e-02 -5.31121530e-03  4.53494154e-02  7.22843856e-02
   1.22486651e-01 -8.83669928e-02 -1.03323542e-01 -1.92402098e-02
  -1.01265118e-01  8.19667876e-02  1.15405791e-01 -8.67718905e-02
   1.00000000e+00  1.39577135e-01  1.72894646e-03  7.68448934e-02]
 [ 8.53452459e-02  3.48899849e-02 -4.55919327e-03 -1.06991298e-01
  -5.76017015e-02  9.07872766e-02 -7.58555904e-02  1.85303003e-01
   5.47878265e-01 -1.40610442e-01  3.36632252e-01 -9.77559686e-02
  -1.12391241e-01  1.19172424e-01  1.27959892e-01  4.04995047e-02
   6.32909685e-02 -1.26833260e-01  7.43930563e-02 -1.63896888e-01
   1.39577150e-01  1.00000000e+00 -1.44269273e-01 -1.64877139e-02]
 [-1.04868248e-01 -1.34006262e-01 -5.53526096e-02 -4.19631787e-02
  -6.97078705e-02  8.88037756e-02 -6.49389625e-02  1.78272761e-02
   1.63783669e-01 -2.18350831e-02  3.03930230e-03 -5.45845833e-03
  -1.98068976e-01  1.95543081e-01  2.04546079e-02  8.17079842e-02
   3.81200314e-02  8.91933441e-02  2.41554976e-01  4.50722687e-02
   1.72895030e-03 -1.44269273e-01  1.00000000e+00  2.32524291e-01]
 [-1.92561261e-02  9.44997743e-02  8.60928446e-02 -1.28874138e-01
   9.34457127e-03  1.08884677e-01  7.93260410e-02 -1.36008546e-01
   9.54209343e-02  2.43033543e-02  1.37563333e-01 -1.02470770e-01
  -5.60260937e-02  1.43789500e-01 -5.28713397e-04  1.20373778e-01
  -1.88806728e-02  2.25055337e-01  1.11007504e-01 -1.76392138e-01
   7.68449008e-02 -1.64877232e-02  2.32524171e-01  1.00000000e+00]]
* conn_model : partcorr
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


* duration : 0.030315
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/_conn_model_partcorr/get_conn_matrix_node


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

