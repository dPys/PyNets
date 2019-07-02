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
* conn_matrix : [[ 1.          0.21722063  0.17704847 -0.14270037  0.0724476   0.24523737
   0.02180628  0.21487068 -0.05245361  0.05048498 -0.18448937  0.33732185
   0.20162176 -0.14171204 -0.00570223 -0.10325853  0.2504256  -0.0094891
  -0.01877154 -0.0756686   0.02849883  0.02776617]
 [ 0.21722065  1.          0.22069594  0.42573127  0.05684003  0.04899087
   0.1342765   0.04576242  0.10477496  0.09428979  0.20792677  0.0569772
  -0.14732221 -0.17097726  0.3986473  -0.1780368  -0.08194586 -0.01344565
  -0.13957053  0.0427286  -0.04050763 -0.08055144]
 [ 0.17704877  0.22069567  1.          0.02430606  0.16649769 -0.02600118
  -0.149136   -0.08031796  0.10572147  0.32641917  0.15715905 -0.2174841
   0.00828737 -0.06218743 -0.05863119  0.70823056 -0.19401179  0.216715
  -0.05116839 -0.00198587  0.01609807  0.04386021]
 [-0.14270037  0.4257312   0.02430552  1.         -0.04917472  0.22864582
  -0.1201773   0.03784168 -0.11616265 -0.18677373 -0.06477907  0.01473012
   0.23099576  0.17586361 -0.05766912  0.06816862  0.5040683  -0.12980892
   0.03860641  0.00623524 -0.13649349 -0.00207813]
 [ 0.07244767  0.05683982  0.1664977  -0.04917467  1.         -0.17790885
   0.07782691  0.05358871  0.19128436 -0.08517019  0.19564137 -0.10497294
   0.04607405  0.13887383 -0.08617669 -0.05851131  0.20524956 -0.10035376
  -0.24786721 -0.03976106 -0.06236684  0.05556452]
 [ 0.2452376   0.0489908  -0.02600105  0.22864611 -0.17790899  1.
   0.14865467  0.48231792 -0.0335574   0.15705551  0.06264375 -0.41050655
   0.27925944  0.09744794 -0.02853176  0.02086505 -0.2006582   0.0912574
  -0.0985122  -0.12793756 -0.10816922  0.09509313]
 [ 0.02180623  0.1342765  -0.14913599 -0.12017724  0.07782688  0.14865461
   1.          0.04897049 -0.13237019 -0.08771995  0.3527296   0.29046258
   0.1504441   0.13749094  0.13986082 -0.12266624 -0.06503123  0.48006603
  -0.03653784 -0.27148822 -0.07397807 -0.04595331]
 [ 0.2148706   0.04576237 -0.08031799  0.0378415   0.05358883  0.4823183
   0.04897027  1.          0.11903244 -0.12917824  0.13275279  0.36235985
  -0.2826355   0.04579421 -0.14897852 -0.02704705  0.00917389 -0.04031504
   0.36082336 -0.12766238 -0.01235633 -0.10298776]
 [-0.05245371  0.10477507  0.10572159 -0.11616281  0.19128451 -0.03355714
  -0.13237004  0.11903232  1.          0.04352972 -0.02217566  0.10902831
   0.0933999   0.19323666  0.26676583  0.03191678 -0.2906796  -0.08356266
  -0.01308833  0.07934505  0.20593475  0.09288777]
 [ 0.05048481  0.09428991  0.32641906 -0.18677405 -0.08517025  0.15705556
  -0.08772007 -0.12917826  0.0435298   1.         -0.0567358   0.2583048
   0.05748328  0.15337013  0.13125208 -0.05697195  0.07879771 -0.23094958
   0.1696922  -0.05878058 -0.00105975  0.01771876]
 [-0.18448915  0.20792708  0.15715846 -0.06477921  0.19564113  0.06264333
   0.35272977  0.13275287 -0.02217545 -0.05673577  1.         -0.20033965
   0.21909097 -0.07444867 -0.13999705  0.01157818  0.07516738 -0.03835686
  -0.05065171  0.57392204  0.04837797 -0.03506817]
 [ 0.33732212  0.05697711 -0.21748415  0.01473049 -0.10497298 -0.4105066
   0.2904629   0.36235967  0.10902835  0.2583048  -0.20033927  1.
   0.054987    0.09551527  0.13030162  0.20157988 -0.28992003 -0.1129991
  -0.17624447  0.3018173   0.05101853  0.1048428 ]
 [ 0.20162098 -0.14732233  0.00828811  0.23099513  0.04607425  0.27926007
   0.15044396 -0.28263548  0.09340016  0.05748279  0.21909076  0.05498794
   1.          0.07755221 -0.15813859 -0.04188531  0.01261117 -0.11547563
  -0.04749952  0.01461324  0.62298054 -0.15224898]
 [-0.1417122  -0.17097701 -0.06218729  0.1758637   0.13887374  0.09744786
   0.137491    0.04579435  0.19323666  0.15336996 -0.07444927  0.09551521
   0.07755267  1.         -0.04060074  0.06027905 -0.02552993 -0.05058262
  -0.03334332  0.0280639  -0.03706608  0.66121894]
 [-0.00570235  0.39864716 -0.05863114 -0.05766955 -0.08617692 -0.02853176
   0.13986073 -0.14897852  0.26676607  0.13125208 -0.13999684  0.13030194
  -0.15813835 -0.04060066  1.         -0.05254452  0.42750755  0.09173843
   0.1476386   0.14341675  0.16336273 -0.22919206]
 [-0.1032588  -0.1780365   0.70823056  0.06816804 -0.0585113   0.02086523
  -0.12266625 -0.027047    0.03191702 -0.05697208  0.01157749  0.2015799
  -0.04188462  0.06027918 -0.05254459  1.          0.22186418 -0.02674251
   0.11903033 -0.10371353  0.1322768  -0.1472712 ]
 [ 0.25042573 -0.08194579 -0.1940115   0.5040686   0.20524974 -0.20065795
  -0.06503104  0.00917365 -0.29067984  0.07879753  0.07516735 -0.2899199
   0.01261008 -0.02552991  0.427507    0.22186372  1.         -0.01234349
   0.04210581 -0.06198287  0.13319586  0.3052074 ]
 [-0.00948931 -0.01344556  0.21671505 -0.12980923 -0.10035371  0.09125761
   0.48006627 -0.0403152  -0.08356238 -0.23094973 -0.03835709 -0.11299881
  -0.11547531 -0.05058273  0.09173823 -0.0267426  -0.01234297  1.
   0.32382345  0.22258529  0.12852019  0.09720125]
 [-0.0187715  -0.1395707  -0.0511683   0.0386066  -0.24786718 -0.09851211
  -0.03653779  0.36082342 -0.01308862  0.16969214 -0.05065158 -0.1762445
  -0.04749995 -0.03334317  0.14763872  0.11903024  0.04210546  0.32382312
   1.          0.01205036  0.14168434  0.31367958]
 [-0.07566873  0.04272836 -0.0019856   0.00623535 -0.03976103 -0.1279373
  -0.27148846 -0.12766239  0.07934476 -0.05878058  0.57392216  0.30181748
   0.01461318  0.02806348  0.1434171  -0.10371383 -0.06198312  0.2225852
   0.01205028  1.         -0.03375496  0.17982525]
 [ 0.02849954 -0.0405077   0.01609757 -0.13649271 -0.06236682 -0.10816988
  -0.07397819 -0.01235618  0.2059343  -0.00105925  0.04837813  0.05101769
   0.62298065 -0.037066    0.1633633   0.13227731  0.13319452  0.12852068
   0.14168373 -0.03375507  1.          0.05334796]
 [ 0.02776615 -0.08055156  0.04386012 -0.00207849  0.05556451  0.09509299
  -0.04595338 -0.10298771  0.09288792  0.01771893 -0.03506783  0.10484286
  -0.15224876  0.66121894 -0.22919193 -0.14727095  0.30520773  0.09720142
   0.31367958  0.17982478  0.05334735  1.        ]]
* conn_model : partcorr
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
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix : [[ 0.          0.21722063  0.17704847 -0.14270037  0.0724476   0.24523737
   0.02180628  0.21487068 -0.05245361  0.05048498 -0.18448937  0.33732185
   0.20162176 -0.14171204 -0.00570223 -0.10325853  0.2504256  -0.0094891
  -0.01877154 -0.0756686   0.02849883  0.02776617]
 [ 0.21722065  0.          0.22069594  0.42573127  0.05684003  0.04899087
   0.1342765   0.04576242  0.10477496  0.09428979  0.20792677  0.0569772
  -0.14732221 -0.17097726  0.3986473  -0.1780368  -0.08194586 -0.01344565
  -0.13957053  0.0427286  -0.04050763 -0.08055144]
 [ 0.17704877  0.22069567  0.          0.02430606  0.16649769 -0.02600118
  -0.149136   -0.08031796  0.10572147  0.32641917  0.15715905 -0.2174841
   0.00828737 -0.06218743 -0.05863119  0.70823056 -0.19401179  0.216715
  -0.05116839 -0.00198587  0.01609807  0.04386021]
 [-0.14270037  0.4257312   0.02430552  0.         -0.04917472  0.22864582
  -0.1201773   0.03784168 -0.11616265 -0.18677373 -0.06477907  0.01473012
   0.23099576  0.17586361 -0.05766912  0.06816862  0.5040683  -0.12980892
   0.03860641  0.00623524 -0.13649349 -0.00207813]
 [ 0.07244767  0.05683982  0.1664977  -0.04917467  0.         -0.17790885
   0.07782691  0.05358871  0.19128436 -0.08517019  0.19564137 -0.10497294
   0.04607405  0.13887383 -0.08617669 -0.05851131  0.20524956 -0.10035376
  -0.24786721 -0.03976106 -0.06236684  0.05556452]
 [ 0.2452376   0.0489908  -0.02600105  0.22864611 -0.17790899  0.
   0.14865467  0.48231792 -0.0335574   0.15705551  0.06264375 -0.41050655
   0.27925944  0.09744794 -0.02853176  0.02086505 -0.2006582   0.0912574
  -0.0985122  -0.12793756 -0.10816922  0.09509313]
 [ 0.02180623  0.1342765  -0.14913599 -0.12017724  0.07782688  0.14865461
   0.          0.04897049 -0.13237019 -0.08771995  0.3527296   0.29046258
   0.1504441   0.13749094  0.13986082 -0.12266624 -0.06503123  0.48006603
  -0.03653784 -0.27148822 -0.07397807 -0.04595331]
 [ 0.2148706   0.04576237 -0.08031799  0.0378415   0.05358883  0.4823183
   0.04897027  0.          0.11903244 -0.12917824  0.13275279  0.36235985
  -0.2826355   0.04579421 -0.14897852 -0.02704705  0.00917389 -0.04031504
   0.36082336 -0.12766238 -0.01235633 -0.10298776]
 [-0.05245371  0.10477507  0.10572159 -0.11616281  0.19128451 -0.03355714
  -0.13237004  0.11903232  0.          0.04352972 -0.02217566  0.10902831
   0.0933999   0.19323666  0.26676583  0.03191678 -0.2906796  -0.08356266
  -0.01308833  0.07934505  0.20593475  0.09288777]
 [ 0.05048481  0.09428991  0.32641906 -0.18677405 -0.08517025  0.15705556
  -0.08772007 -0.12917826  0.0435298   0.         -0.0567358   0.2583048
   0.05748328  0.15337013  0.13125208 -0.05697195  0.07879771 -0.23094958
   0.1696922  -0.05878058 -0.00105975  0.01771876]
 [-0.18448915  0.20792708  0.15715846 -0.06477921  0.19564113  0.06264333
   0.35272977  0.13275287 -0.02217545 -0.05673577  0.         -0.20033965
   0.21909097 -0.07444867 -0.13999705  0.01157818  0.07516738 -0.03835686
  -0.05065171  0.57392204  0.04837797 -0.03506817]
 [ 0.33732212  0.05697711 -0.21748415  0.01473049 -0.10497298 -0.4105066
   0.2904629   0.36235967  0.10902835  0.2583048  -0.20033927  0.
   0.054987    0.09551527  0.13030162  0.20157988 -0.28992003 -0.1129991
  -0.17624447  0.3018173   0.05101853  0.1048428 ]
 [ 0.20162098 -0.14732233  0.00828811  0.23099513  0.04607425  0.27926007
   0.15044396 -0.28263548  0.09340016  0.05748279  0.21909076  0.05498794
   0.          0.07755221 -0.15813859 -0.04188531  0.01261117 -0.11547563
  -0.04749952  0.01461324  0.62298054 -0.15224898]
 [-0.1417122  -0.17097701 -0.06218729  0.1758637   0.13887374  0.09744786
   0.137491    0.04579435  0.19323666  0.15336996 -0.07444927  0.09551521
   0.07755267  0.         -0.04060074  0.06027905 -0.02552993 -0.05058262
  -0.03334332  0.0280639  -0.03706608  0.66121894]
 [-0.00570235  0.39864716 -0.05863114 -0.05766955 -0.08617692 -0.02853176
   0.13986073 -0.14897852  0.26676607  0.13125208 -0.13999684  0.13030194
  -0.15813835 -0.04060066  0.         -0.05254452  0.42750755  0.09173843
   0.1476386   0.14341675  0.16336273 -0.22919206]
 [-0.1032588  -0.1780365   0.70823056  0.06816804 -0.0585113   0.02086523
  -0.12266625 -0.027047    0.03191702 -0.05697208  0.01157749  0.2015799
  -0.04188462  0.06027918 -0.05254459  0.          0.22186418 -0.02674251
   0.11903033 -0.10371353  0.1322768  -0.1472712 ]
 [ 0.25042573 -0.08194579 -0.1940115   0.5040686   0.20524974 -0.20065795
  -0.06503104  0.00917365 -0.29067984  0.07879753  0.07516735 -0.2899199
   0.01261008 -0.02552991  0.427507    0.22186372  0.         -0.01234349
   0.04210581 -0.06198287  0.13319586  0.3052074 ]
 [-0.00948931 -0.01344556  0.21671505 -0.12980923 -0.10035371  0.09125761
   0.48006627 -0.0403152  -0.08356238 -0.23094973 -0.03835709 -0.11299881
  -0.11547531 -0.05058273  0.09173823 -0.0267426  -0.01234297  0.
   0.32382345  0.22258529  0.12852019  0.09720125]
 [-0.0187715  -0.1395707  -0.0511683   0.0386066  -0.24786718 -0.09851211
  -0.03653779  0.36082342 -0.01308862  0.16969214 -0.05065158 -0.1762445
  -0.04749995 -0.03334317  0.14763872  0.11903024  0.04210546  0.32382312
   0.          0.01205036  0.14168434  0.31367958]
 [-0.07566873  0.04272836 -0.0019856   0.00623535 -0.03976103 -0.1279373
  -0.27148846 -0.12766239  0.07934476 -0.05878058  0.57392216  0.30181748
   0.01461318  0.02806348  0.1434171  -0.10371383 -0.06198312  0.2225852
   0.01205028  0.         -0.03375496  0.17982525]
 [ 0.02849954 -0.0405077   0.01609757 -0.13649271 -0.06236682 -0.10816988
  -0.07397819 -0.01235618  0.2059343  -0.00105925  0.04837813  0.05101769
   0.62298065 -0.037066    0.1633633   0.13227731  0.13319452  0.12852068
   0.14168373 -0.03375507  0.          0.05334796]
 [ 0.02776615 -0.08055156  0.04386012 -0.00207849  0.05556451  0.09509299
  -0.04595338 -0.10298771  0.09288792  0.01771893 -0.03506783  0.10484286
  -0.15224876  0.66121894 -0.22919193 -0.14727095  0.30520773  0.09720142
   0.31367958  0.17982478  0.05334735  0.        ]]
* conn_model : partcorr
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
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* binary : False
* c_boot : 0
* conn_matrix_thr : [[0.         0.21722063 0.         0.         0.         0.24523737
  0.         0.21487068 0.         0.         0.         0.33732185
  0.20162176 0.         0.         0.         0.2504256  0.
  0.         0.         0.         0.        ]
 [0.21722065 0.         0.22069594 0.42573127 0.         0.
  0.         0.         0.         0.         0.20792677 0.
  0.         0.         0.3986473  0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.22069567 0.         0.         0.         0.
  0.         0.         0.         0.32641917 0.         0.
  0.         0.         0.         0.70823056 0.         0.216715
  0.         0.         0.         0.        ]
 [0.         0.4257312  0.         0.         0.         0.22864582
  0.         0.         0.         0.         0.         0.
  0.23099576 0.         0.         0.         0.5040683  0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.20524956 0.
  0.         0.         0.         0.        ]
 [0.2452376  0.         0.         0.22864611 0.         0.
  0.         0.48231792 0.         0.         0.         0.
  0.27925944 0.         0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.3527296  0.29046258
  0.         0.         0.         0.         0.         0.48006603
  0.         0.         0.         0.        ]
 [0.2148706  0.         0.         0.         0.         0.4823183
  0.         0.         0.         0.         0.         0.36235985
  0.         0.         0.         0.         0.         0.
  0.36082336 0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.26676583 0.         0.         0.
  0.         0.         0.20593475 0.        ]
 [0.         0.         0.32641906 0.         0.         0.
  0.         0.         0.         0.         0.         0.2583048
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.20792708 0.         0.         0.         0.
  0.35272977 0.         0.         0.         0.         0.
  0.21909097 0.         0.         0.         0.         0.
  0.         0.57392204 0.         0.        ]
 [0.33732212 0.         0.         0.         0.         0.
  0.2904629  0.36235967 0.         0.2583048  0.         0.
  0.         0.         0.         0.20157988 0.         0.
  0.         0.3018173  0.         0.        ]
 [0.20162098 0.         0.         0.23099513 0.         0.27926007
  0.         0.         0.         0.         0.21909076 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.62298054 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.66121894]
 [0.         0.39864716 0.         0.         0.         0.
  0.         0.         0.26676607 0.         0.         0.
  0.         0.         0.         0.         0.42750755 0.
  0.         0.         0.         0.        ]
 [0.         0.         0.70823056 0.         0.         0.
  0.         0.         0.         0.         0.         0.2015799
  0.         0.         0.         0.         0.22186418 0.
  0.         0.         0.         0.        ]
 [0.25042573 0.         0.         0.5040686  0.20524974 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.427507   0.22186372 0.         0.
  0.         0.         0.         0.3052074 ]
 [0.         0.         0.21671505 0.         0.         0.
  0.48006627 0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.32382345 0.22258529 0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.36082342 0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.32382312
  0.         0.         0.         0.31367958]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.57392216 0.30181748
  0.         0.         0.         0.         0.         0.2225852
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.2059343  0.         0.         0.
  0.62298065 0.         0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.66121894 0.         0.         0.30520773 0.
  0.31367958 0.         0.         0.        ]]
* conn_model : partcorr
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
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* edge_threshold : 18.0%
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/graphs/0021001_Default_est_partcorr_0.18densparc_mm.npy
* hpass : None
* labels : [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
 nan nan nan nan]
* network : Default
* node_size : parc
* norm : 0
* prune : 1
* roi : None
* smooth : 0
* thr : 0.18
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Runtime info
------------


* duration : 0.033471
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/thresh_func_node/mapflow/_thresh_func_node9


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

