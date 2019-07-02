Node: utility
=============


 Hierarchy : _thresh_diff_node2
 Exec ID : _thresh_diff_node2


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_matrix : [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 2. 2.]
 [0. 0. 0. ... 2. 0. 3.]
 [0. 0. 0. ... 2. 3. 0.]]
* conn_model : csd
* coords : (7.537630258587413, -24.854882284832115, 58.69490544191433)
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* function_str : def thresh_struct(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                  disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary,
                  target_samples, track_type, atlas_mni, streams):
    """
    Threshold a structural connectivity matrix using any of a variety of methods.

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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.

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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.
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
    est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples,
                                          track_type, thr_type, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, prune, ID, dir_path, atlas, uatlas, labels, coords, norm, binary, target_samples, track_type, atlas_mni, streams

* labels : nan
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* thr : 0.07
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_matrix : [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 2. 2.]
 [0. 0. 0. ... 2. 0. 3.]
 [0. 0. 0. ... 2. 3. 0.]]
* conn_model : csd
* coords : (7.537630258587413, -24.854882284832115, 58.69490544191433)
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* function_str : def thresh_struct(dens_thresh, thr, conn_matrix, conn_model, network, ID, dir_path, roi, node_size, min_span_tree,
                  disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary,
                  target_samples, track_type, atlas_mni, streams):
    """
    Threshold a structural connectivity matrix using any of a variety of methods.

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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.

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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to save streamline array sequence in .trk format.
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
    est_path = utils.create_est_path_diff(ID, network, conn_model, thr, roi, dir_path, node_size, target_samples,
                                          track_type, thr_type, parc)

    utils.save_mat(conn_matrix_thr, est_path)

    return conn_matrix_thr, edge_threshold, est_path, thr, node_size, network, conn_model, roi, prune, ID, dir_path, atlas, uatlas, labels, coords, norm, binary, target_samples, track_type, atlas_mni, streams

* labels : nan
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* thr : 0.07
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_matrix_thr : [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 2. 2.]
 [0. 0. 0. ... 2. 0. 3.]
 [0. 0. 0. ... 2. 3. 0.]]
* conn_model : csd
* coords : (7.537630258587413, -24.854882284832115, 58.69490544191433)
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* edge_threshold : 7.000000000000001%
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_est_csd_0.07densparc_mm100000_samplesparticle_track.npy
* labels : nan
* network : None
* node_size : parc
* norm : 0
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* thr : 0.07
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 0.269536
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/thresh_diff_node/mapflow/_thresh_diff_node2


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
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/Users/derekpisner/anaconda3/bin:/Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner
* SHELL : /bin/bash
* SHLVL : 3
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.qmAkE8F40f/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : AFAF5DB1-79BD-4BC9-B7BB-C754B1B9AAB6
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

