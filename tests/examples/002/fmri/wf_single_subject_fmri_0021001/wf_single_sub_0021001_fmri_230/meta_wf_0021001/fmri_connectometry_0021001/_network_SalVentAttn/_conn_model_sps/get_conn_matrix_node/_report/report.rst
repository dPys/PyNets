Node: meta_wf_0021001 (fmri_connectometry_0021001 (get_conn_matrix_node (utility)
=================================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.get_conn_matrix_node
 Exec ID : get_conn_matrix_node.bI.b0.c1


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
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
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 1.3127716   2.5494092   1.5156994  ...  3.0159454   1.6702434
   3.3613448 ]
 [-0.77208716 -0.16343977 -0.01717531 ...  0.15919366 -0.7728827
  -0.2856029 ]
 [ 0.7586903   0.66712046  1.1988986  ... -0.49962384  0.63434255
  -1.862055  ]
 ...
 [ 0.8830816  -0.01945488  0.54305464 ... -0.27885902 -0.10455992
   0.14930663]
 [ 0.8627658   1.2102736  -0.491194   ...  1.6579114   0.15735124
   0.14119335]
 [ 0.11045323  1.80482    -0.8675637  ...  0.8167391   0.4368118
  -2.0118477 ]]
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
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
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : SalVentAttn
* node_size : 4
* norm : 0
* parc : False
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 1.3127716   2.5494092   1.5156994  ...  3.0159454   1.6702434
   3.3613448 ]
 [-0.77208716 -0.16343977 -0.01717531 ...  0.15919366 -0.7728827
  -0.2856029 ]
 [ 0.7586903   0.66712046  1.1988986  ... -0.49962384  0.63434255
  -1.862055  ]
 ...
 [ 0.8830816  -0.01945488  0.54305464 ... -0.27885902 -0.10455992
   0.14930663]
 [ 0.8627658   1.2102736  -0.491194   ...  1.6579114   0.15735124
   0.14119335]
 [ 0.11045323  1.80482    -0.8675637  ...  0.8167391   0.4368118
  -2.0118477 ]]
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


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
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 0.516671
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/_conn_model_sps/get_conn_matrix_node


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

