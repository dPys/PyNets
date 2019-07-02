Node: meta_wf_0021001 (dmri_connectometry_0021001 (dsn_node (utility)
=====================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.dsn_node
 Exec ID : dsn_node.b1.c1


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* binary : False
* conn_model : tensor
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* curv_thr_list : [60, 30, 10]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def direct_streamline_norm(streams, fa_path, dir_path, track_type, target_samples, conn_model, network, node_size,
                           dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas,
                           labels, coords, norm, binary, atlas_mni, basedir_path, curv_thr_list, step_list,
                           overwrite=False):
    """
    A Function to perform normalization of streamlines tracked in native diffusion space to an
    FSL_HCP1065_FA_2mm.nii.gz template in MNI space.

    Parameters
    ----------
    streams : str
        File path to save streamline array sequence in .trk format.
    fa_path : str
        File path to FA Nifti1Image.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.

    Returns
    -------
    streams_warp : str
        File path to normalized streamline array sequence in .trk format.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.

    References
    ----------
    .. [1] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of different spatial normalization approaches on
           tractography and structural brain networks. Network Neuroscience, 1-19.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.tracking.streamline import Streamlines
    from pynets.registration import reg_utils as regutils
    from pynets.registration.register import Warp
    import pkg_resources

    template_path = pkg_resources.resource_filename("pynets", "templates/FSL_HCP1065_FA_2mm.nii.gz")

    dsn_dir = "%s%s" % (basedir_path, '/dmri_tmp/DSN')
    if not os.path.isdir(dsn_dir):
        os.mkdir(dsn_dir)

    streams_mni = "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/streamlines_mni_', conn_model, '_', target_samples,
                                                '_', node_size, 'mm_curv', str(curv_thr_list).replace(', ', '_'),
                                                '_step', str(step_list).replace(', ', '_'), '.trk')

    # Run ANTs reg
    t_aff = "%s%s" % (dsn_dir, '/0GenericAffine.mat')
    t_warp = "%s%s" % (dsn_dir, '/1Warp.nii.gz')

    fa_path_img = nib.load(fa_path)
    fa_flip_path = "%s%s" % (fa_path.split('.nii.gz')[0], '_flip.nii.gz')
    s_aff = fa_path_img.affine
    s_aff[0][0] = -s_aff[0][0]
    fa_path_img.set_sform(s_aff)
    fa_path_img.set_qform(s_aff)
    fa_path_img.update_header()
    nib.save(fa_path_img, fa_flip_path)

    if ((os.path.isfile(t_aff) is False) and (os.path.isfile(t_warp) is False)) or (overwrite is True):
        regutils.antssyn(template_path, fa_flip_path, dsn_dir)

    # Warp streamlines
    wS = Warp(streams, streams_mni, template_path, t_aff, t_warp, fa_path, dsn_dir)
    wS.streamlines()

    s_aff[:3, 3] = np.array([270, 0, 0])
    streamlines_mni = nib.streamlines.load(streams_mni)
    streamlines_mni_s = streamlines_mni.streamlines
    streamlines_trans = Streamlines(regutils.transform_to_affine(streamlines_mni_s, streamlines_mni.header, s_aff))
    streams_warp = "%s%s" % (streams_mni.split('.trk')[0], '_warped.trk')
    tractogram = nib.streamlines.Tractogram(streamlines_trans, affine_to_rasmm=np.eye(4))
    trkfile = nib.streamlines.trk.TrkFile(tractogram, header=streamlines_mni.header)
    nib.streamlines.save(trkfile, streams_warp)
    print(streams_warp)

    return streams_warp, dir_path, track_type, target_samples, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* step_list : [0.2, 0.3, 0.4, 0.5]
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_tensor_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5].trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* binary : False
* conn_model : tensor
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* curv_thr_list : [60, 30, 10]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def direct_streamline_norm(streams, fa_path, dir_path, track_type, target_samples, conn_model, network, node_size,
                           dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas,
                           labels, coords, norm, binary, atlas_mni, basedir_path, curv_thr_list, step_list,
                           overwrite=False):
    """
    A Function to perform normalization of streamlines tracked in native diffusion space to an
    FSL_HCP1065_FA_2mm.nii.gz template in MNI space.

    Parameters
    ----------
    streams : str
        File path to save streamline array sequence in .trk format.
    fa_path : str
        File path to FA Nifti1Image.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.

    Returns
    -------
    streams_warp : str
        File path to normalized streamline array sequence in .trk format.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
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
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.

    References
    ----------
    .. [1] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of different spatial normalization approaches on
           tractography and structural brain networks. Network Neuroscience, 1-19.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.tracking.streamline import Streamlines
    from pynets.registration import reg_utils as regutils
    from pynets.registration.register import Warp
    import pkg_resources

    template_path = pkg_resources.resource_filename("pynets", "templates/FSL_HCP1065_FA_2mm.nii.gz")

    dsn_dir = "%s%s" % (basedir_path, '/dmri_tmp/DSN')
    if not os.path.isdir(dsn_dir):
        os.mkdir(dsn_dir)

    streams_mni = "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/streamlines_mni_', conn_model, '_', target_samples,
                                                '_', node_size, 'mm_curv', str(curv_thr_list).replace(', ', '_'),
                                                '_step', str(step_list).replace(', ', '_'), '.trk')

    # Run ANTs reg
    t_aff = "%s%s" % (dsn_dir, '/0GenericAffine.mat')
    t_warp = "%s%s" % (dsn_dir, '/1Warp.nii.gz')

    fa_path_img = nib.load(fa_path)
    fa_flip_path = "%s%s" % (fa_path.split('.nii.gz')[0], '_flip.nii.gz')
    s_aff = fa_path_img.affine
    s_aff[0][0] = -s_aff[0][0]
    fa_path_img.set_sform(s_aff)
    fa_path_img.set_qform(s_aff)
    fa_path_img.update_header()
    nib.save(fa_path_img, fa_flip_path)

    if ((os.path.isfile(t_aff) is False) and (os.path.isfile(t_warp) is False)) or (overwrite is True):
        regutils.antssyn(template_path, fa_flip_path, dsn_dir)

    # Warp streamlines
    wS = Warp(streams, streams_mni, template_path, t_aff, t_warp, fa_path, dsn_dir)
    wS.streamlines()

    s_aff[:3, 3] = np.array([270, 0, 0])
    streamlines_mni = nib.streamlines.load(streams_mni)
    streamlines_mni_s = streamlines_mni.streamlines
    streamlines_trans = Streamlines(regutils.transform_to_affine(streamlines_mni_s, streamlines_mni.header, s_aff))
    streams_warp = "%s%s" % (streams_mni.split('.trk')[0], '_warped.trk')
    tractogram = nib.streamlines.Tractogram(streamlines_trans, affine_to_rasmm=np.eye(4))
    trkfile = nib.streamlines.trk.TrkFile(tractogram, header=streamlines_mni.header)
    nib.streamlines.save(trkfile, streams_warp)
    print(streams_warp)

    return streams_warp, dir_path, track_type, target_samples, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* step_list : [0.2, 0.3, 0.4, 0.5]
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_tensor_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5].trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : tensor
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* streams_warp : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_tensor_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 1.9293719999999999
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/_conn_model_tensor/dsn_node


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

