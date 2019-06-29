Node: meta_wf_0021001 (dmri_connectometry_0021001 (save_coords_and_labels_node (utility)
========================================================================================


 Hierarchy : wf_single_sub_0021001_dmri_271.meta_wf_0021001.dmri_connectometry_0021001.save_coords_and_labels_node
 Exec ID : save_coords_and_labels_node.c1


Original Inputs
---------------


* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* function_str : def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """

    :param coords:
    :param labels:
    :param dir_path:
    :param network:
    :return:
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

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : SalVentAttn

Execution Inputs
----------------


* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* function_str : def save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network):
    """

    :param coords:
    :param labels:
    :param dir_path:
    :param network:
    :return:
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

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : SalVentAttn


Execution Outputs
-----------------


* out : ('/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/SalVentAttn_coords_rsn.pkl', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/SalVentAttn_labels_rsn.pkl')


Runtime info
------------


* duration : 0.002362
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_271/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/save_coords_and_labels_node


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
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.qmAkE8F40f/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : FF5FFBDE-8277-4DEC-B281-B12FE6AE3D08
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

