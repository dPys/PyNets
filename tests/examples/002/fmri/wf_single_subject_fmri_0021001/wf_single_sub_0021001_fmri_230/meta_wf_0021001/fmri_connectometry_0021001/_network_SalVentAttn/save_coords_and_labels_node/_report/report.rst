Node: meta_wf_0021001 (fmri_connectometry_0021001 (save_coords_and_labels_node (utility)
========================================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.save_coords_and_labels_node
 Exec ID : save_coords_and_labels_node.c1


Original Inputs
---------------


* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* function_str : def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """
    Save RSN coordinates and labels to pickle files.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.

    Returns
    -------
    coord_path : str
        Path to pickled coordinates list.
    labels_path : str
        Path to pickled labels list.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    # Save coords to pickle
    coord_path = "%s%s%s%s" % (dir_path, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)
    return coord_path, labels_path

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : SalVentAttn

Execution Inputs
----------------


* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* function_str : def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """
    Save RSN coordinates and labels to pickle files.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.

    Returns
    -------
    coord_path : str
        Path to pickled coordinates list.
    labels_path : str
        Path to pickled labels list.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    # Save coords to pickle
    coord_path = "%s%s%s%s" % (dir_path, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)
    return coord_path, labels_path

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : SalVentAttn


Execution Outputs
-----------------


* out : ('/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/SalVentAttn_coords_rsn.pkl', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/SalVentAttn_labels_rsn.pkl')


Runtime info
------------


* duration : 0.15741
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/save_coords_and_labels_node


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

