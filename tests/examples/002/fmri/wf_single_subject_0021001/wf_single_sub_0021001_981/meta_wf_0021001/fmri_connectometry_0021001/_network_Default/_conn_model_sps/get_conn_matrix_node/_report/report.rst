Node: meta_wf_0021001 (fmri_connectometry_0021001 (get_conn_matrix_node (utility)
=================================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.get_conn_matrix_node
 Exec ID : get_conn_matrix_node.bI.b0.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : Default
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 0.74280095 -0.41899306 -0.28390482 ...  0.7304386   2.0709388
   1.4977248 ]
 [-1.5639161  -1.2458746   0.04724853 ...  0.33656812 -0.17132154
   0.5810947 ]
 [ 0.23354214 -0.06319581 -0.46292573 ... -0.28127933 -0.16899581
  -0.45163587]
 ...
 [ 0.5634819   1.116684    0.04043001 ...  0.01744784  0.19165032
  -0.01654548]
 [ 1.350044    1.603178    2.0741808  ...  0.7775223   1.6493644
   0.77825814]
 [ 1.0263946   0.6778529   1.1870908  ... -0.43348092  0.8611615
  -1.2669188 ]]
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_model : sps
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : Default
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* smooth : 0
* time_series : [[ 0.74280095 -0.41899306 -0.28390482 ...  0.7304386   2.0709388
   1.4977248 ]
 [-1.5639161  -1.2458746   0.04724853 ...  0.33656812 -0.17132154
   0.5810947 ]
 [ 0.23354214 -0.06319581 -0.46292573 ... -0.28127933 -0.16899581
  -0.45163587]
 ...
 [ 0.5634819   1.116684    0.04043001 ...  0.01744784  0.19165032
  -0.01654548]
 [ 1.350044    1.603178    2.0741808  ...  0.7775223   1.6493644
   0.77825814]
 [ 1.0263946   0.6778529   1.1870908  ... -0.43348092  0.8611615
  -1.2669188 ]]
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[-2.03920632  0.43106068  0.          0.04314351  0.          0.34331076
   0.          0.11852957  0.          0.27650819  0.          0.27466575
   0.33295579  0.          0.1678271   0.          0.          0.
   0.          0.          0.08714262  0.        ]
 [ 0.43106068 -1.90721664  0.08157102  0.1524147   0.          0.
   0.09456576  0.          0.          0.          0.35764581  0.
   0.         -0.          0.62555077  0.          0.          0.
  -0.          0.          0.         -0.        ]
 [ 0.          0.08157102 -2.6235266   0.          0.24954536  0.
  -0.         -0.          0.16966884  0.36461715  0.12429774  0.
   0.22961303  0.          0.          1.1801045   0.          0.
   0.          0.          0.20584319  0.        ]
 [ 0.04314351  0.1524147   0.         -2.12180543  0.04785933  0.27631466
  -0.          0.          0.          0.          0.         -0.
   0.29708672  0.11319843  0.          0.05034257  0.99336033  0.
   0.          0.          0.          0.        ]
 [ 0.          0.          0.24954536  0.04785933 -1.45376112  0.
   0.         -0.          0.13698582  0.          0.30156113  0.
   0.15918825  0.01157989  0.          0.          0.13988837  0.
  -0.          0.          0.          0.        ]
 [ 0.34331076  0.          0.          0.27631466  0.         -1.59645067
   0.15177792  0.46713137  0.          0.          0.         -0.
   0.09509693  0.03771282  0.          0.          0.          0.01062495
   0.         -0.          0.          0.        ]
 [ 0.          0.09456576 -0.         -0.          0.          0.15177792
  -1.4022526   0.24950514  0.         -0.          0.14602717  0.10670883
   0.          0.          0.         -0.06505858 -0.          0.41900489
  -0.          0.          0.         -0.        ]
 [ 0.11852957  0.         -0.          0.         -0.          0.46713137
   0.24950514 -1.41193867  0.          0.         -0.          0.13633707
  -0.          0.          0.         -0.         -0.          0.
   0.0859003  -0.         -0.          0.        ]
 [ 0.          0.          0.16966884  0.          0.13698582  0.
   0.          0.         -2.25602445  0.39560309  0.          0.34308124
   0.09813521  0.26806568  0.14320498  0.          0.          0.
   0.          0.31470402  0.44714846  0.        ]
 [ 0.27650819  0.          0.36461715  0.          0.          0.
  -0.          0.          0.39560309 -2.18434909  0.          0.10287198
   0.07344866  0.20669372  0.15651802  0.23811878  0.          0.
   0.05597093  0.          0.23560921  0.05200272]
 [ 0.          0.35764581  0.12429774  0.          0.30156113  0.
   0.14602717 -0.          0.          0.         -1.99946037  0.
   0.42285018  0.          0.          0.          0.          0.08907438
   0.          0.56075635  0.          0.        ]
 [ 0.27466575  0.          0.         -0.          0.         -0.
   0.10670883  0.13633707  0.34308124  0.10287198  0.         -1.46762918
   0.          0.          0.07562668  0.         -0.          0.
   0.          0.11816121  0.          0.        ]
 [ 0.33295579  0.          0.22961303  0.29708672  0.15918825  0.09509693
   0.         -0.          0.09813521  0.07344866  0.42285018  0.
  -2.73769541  0.05341289  0.          0.00554857  0.03686225  0.
   0.          0.          1.04496486  0.        ]
 [ 0.         -0.          0.          0.11319843  0.01157989  0.03771282
   0.          0.          0.26806568  0.20669372  0.          0.
   0.05341289 -2.60316846  0.          0.          0.01817835  0.
   0.22873531  0.          0.07045032  1.46512465]
 [ 0.1678271   0.62555077  0.          0.          0.          0.
   0.          0.          0.14320498  0.15651802  0.          0.07562668
   0.          0.         -1.93443057  0.          0.22907261  0.02711104
   0.          0.12817012  0.24311576  0.        ]
 [ 0.          0.          1.1801045   0.05034257  0.          0.
  -0.06505858 -0.          0.          0.23811878  0.          0.
   0.00554857  0.          0.         -2.35425875  0.29977121  0.
   0.          0.          0.344138    0.        ]
 [ 0.          0.          0.          0.99336033  0.13988837  0.
  -0.         -0.          0.          0.          0.         -0.
   0.03686225  0.01817835  0.22907261  0.29977121 -2.41706973  0.
   0.16096237  0.          0.2311509   0.35085835]
 [ 0.          0.          0.          0.          0.          0.01062495
   0.41900489  0.          0.          0.          0.08907438  0.
   0.          0.          0.02711104  0.          0.         -1.40671607
   0.3608923   0.15252059  0.          0.        ]
 [ 0.         -0.          0.          0.         -0.          0.
  -0.          0.0859003   0.          0.05597093  0.          0.
   0.          0.22873531  0.          0.          0.16096237  0.3608923
  -1.85633941  0.          0.14648113  0.6522768 ]
 [ 0.          0.          0.          0.          0.         -0.
   0.         -0.          0.31470402  0.          0.56075635  0.11816121
   0.          0.          0.12817012  0.          0.          0.15252059
   0.         -1.74566078  0.1448179   0.13400561]
 [ 0.08714262  0.          0.20584319  0.          0.          0.
   0.         -0.          0.44714846  0.23560921  0.          0.
   1.04496486  0.07045032  0.24311576  0.344138    0.2311509   0.
   0.14648113  0.1448179  -3.03178627  0.04615476]
 [ 0.         -0.          0.          0.          0.          0.
  -0.          0.          0.          0.05200272  0.          0.
   0.          1.46512465  0.          0.          0.35085835  0.
   0.6522768   0.13400561  0.04615476 -2.7901946 ]]
* conn_model : sps
* coords : [[ 41.46036719  -7.7177768   44.17087537]
 [ 22.6098635   -4.56230735 -31.95640687]
 [ 46.24613187 -65.41561533  30.40984289]
 [ -3.64342857 -66.68971429 -23.60742857]
 [ -4.46764253 -50.63995891 -13.20005136]
 [-50.25136791 -10.39928915  -6.63124912]
 [ 25.51007813 -60.26367188  53.30015625]
 [-29.8504807  -89.22470943   1.50272636]
 [ 32.80915227 -86.42231717   2.04755712]
 [-57.46055697 -27.80877621 -12.65871519]
 [  5.86218182  37.532        2.95709091]
 [ 45.29406347 -21.88872143  43.83195177]
 [ -6.17316943 -25.68163193  57.79975058]
 [ 10.77466562 -15.93674272  -9.79040126]
 [ 46.06022409 -17.97478992   8.11204482]
 [ 13.6988191  -66.42386874  -5.03918905]
 [ 49.02416244 -28.17928934 -27.29678511]
 [-13.29384318 -67.50893365  -5.87891203]
 [ 26.30807397 -20.53450609 -12.81235904]
 [  5.4625651   21.54329427  28.04296875]
 [-48.32543193 -28.42695232 -28.30117484]
 [  7.90322581 -44.98354839  19.30645161]]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* disp_filt : False
* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
 nan nan nan nan]
* min_span_tree : False
* network : Default
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Runtime info
------------


* duration : 4.991469
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/_conn_model_sps/get_conn_matrix_node


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

