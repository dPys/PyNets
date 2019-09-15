Node: meta_wf_002 (pass_meta_ins_func_node (utility)
====================================================


 Hierarchy : wf_single_sub_002_fmri_419.meta_wf_002.pass_meta_ins_func_node
 Exec ID : pass_meta_ins_func_node.a0


Original Inputs
---------------


* ID : 002
* binary : False
* conn_model : partcorr
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/coords_dosenbach_2010/graphs/002_est_partcorr_0.2prop_4mm_func.npy
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

* network : None
* node_size : 4
* norm : 0
* prune : 2
* roi : None
* thr : 0.2

Execution Inputs
----------------


* ID : 002
* binary : False
* conn_model : partcorr
* est_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/coords_dosenbach_2010/graphs/002_est_partcorr_0.2prop_4mm_func.npy
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

* network : None
* node_size : 4
* norm : 0
* prune : 2
* roi : None
* thr : 0.2


Execution Outputs
-----------------


* ID_iterlist : 002
* binary_iterlist : False
* conn_model_iterlist : partcorr
* est_path_iterlist : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/coords_dosenbach_2010/graphs/002_est_partcorr_0.2prop_4mm_func.npy
* network_iterlist : None
* node_size_iterlist : 4
* norm_iterlist : 0
* prune_iterlist : 2
* roi_iterlist : None
* thr_iterlist : 0.2


Runtime info
------------


* duration : 0.001402
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_002/wf_single_sub_002_fmri_419/meta_wf_002/_thr_0.2/pass_meta_ins_func_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.v8x0bpN28D/Render
* CC : /usr/bin/clang
* CFLAGS :  -I/usr/local/opt/libomp/include
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_PYTHON_EXE : /usr/local/anaconda3/bin/python
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include -Xpreprocessor -fopenmp
* CXX : /usr/bin/clang++
* CXXFLAGS :  -I/usr/local/opt/libomp/include
* DISPLAY : dpys:0.0
* DYLD_LIBRARY_PATH : /usr/local/opt/libomp/lib
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
* KMP_DUPLICATE_LIB_OK : True
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/libxml2/lib -L/usr/local/opt/libomp/lib -lomp
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : derekpisner
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* OLDPWD : /Users/derekpisner/Applications/PyNets/tests/examples/002
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/Applications/autoDTI/Batch_scripts:/Users/derekpisner/Applications/autoDTI/Main_scripts:/Users/derekpisner/Applications/autoDTI/Stage_scripts:/Users/derekpisner/Applications/autoDTI/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/DTI_TK:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Conversion_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Motion_plotting_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/QAtools
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.0OgUYjH7Dp/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.2
* TERM_SESSION_ID : 035E9BD7-B8A4-4DE5-A5A5-FD6EC00B205C
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0
* autoDTI_HOME : /Users/derekpisner/Applications/autoDTI

