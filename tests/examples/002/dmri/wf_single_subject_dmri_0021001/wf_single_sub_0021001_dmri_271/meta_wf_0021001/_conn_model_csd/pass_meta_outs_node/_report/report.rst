Node: meta_wf_0021001 (pass_meta_outs_node (utility)
====================================================


 Hierarchy : wf_single_sub_0021001_dmri_271.meta_wf_0021001.pass_meta_outs_node
 Exec ID : pass_meta_outs_node.b0


Original Inputs
---------------


* ID_iterlist : [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]
* binary_iterlist : [[False, False, False, False, False, False], [False, False, False, False, False, False]]
* conn_model_iterlist : [['csd', 'csd', 'csd', 'csd', 'csd', 'csd'], ['csd', 'csd', 'csd', 'csd', 'csd', 'csd']]
* embed : False
* est_path_iterlist : [['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy']]
* function_str : def pass_meta_outs(conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist,
                   prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist, embed=True,
                   multimodal=False):
    from pynets.utils import build_omnetome, flatten
    """

    :param conn_model_iterlist:
    :param est_path_iterlist:
    :param network_iterlist:
    :param node_size_iterlist:
    :param thr_iterlist:
    :param prune_iterlist:
    :param ID_iterlist:
    :param roi_iterlist:
    :param norm_iterlist:
    :param binary_iterlist:
    :return:
    """
    if embed is True:
        build_omnetome(list(flatten(est_path_iterlist)), list(flatten(ID_iterlist))[0], multimodal)

    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist

* multimodal : False
* network_iterlist : [['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']]
* node_size_iterlist : [['parc', 'parc', 'parc', 'parc', 'parc', 'parc'], ['parc', 'parc', 'parc', 'parc', 'parc', 'parc']]
* norm_iterlist : [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
* prune_iterlist : [['2', '2', '2', '2', '2', '2'], ['2', '2', '2', '2', '2', '2']]
* roi_iterlist : [[None, None, None, None, None, None], [None, None, None, None, None, None]]
* thr_iterlist : [['0.05', '0.06', '0.07', '0.08', '0.09', '0.1'], ['0.05', '0.06', '0.07', '0.08', '0.09', '0.1']]

Execution Inputs
----------------


* ID_iterlist : [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]
* binary_iterlist : [[False, False, False, False, False, False], [False, False, False, False, False, False]]
* conn_model_iterlist : [['csd', 'csd', 'csd', 'csd', 'csd', 'csd'], ['csd', 'csd', 'csd', 'csd', 'csd', 'csd']]
* embed : False
* est_path_iterlist : [['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy']]
* function_str : def pass_meta_outs(conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist,
                   prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist, embed=True,
                   multimodal=False):
    from pynets.utils import build_omnetome, flatten
    """

    :param conn_model_iterlist:
    :param est_path_iterlist:
    :param network_iterlist:
    :param node_size_iterlist:
    :param thr_iterlist:
    :param prune_iterlist:
    :param ID_iterlist:
    :param roi_iterlist:
    :param norm_iterlist:
    :param binary_iterlist:
    :return:
    """
    if embed is True:
        build_omnetome(list(flatten(est_path_iterlist)), list(flatten(ID_iterlist))[0], multimodal)

    return conn_model_iterlist, est_path_iterlist, network_iterlist, node_size_iterlist, thr_iterlist, prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist, binary_iterlist

* multimodal : False
* network_iterlist : [['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']]
* node_size_iterlist : [['parc', 'parc', 'parc', 'parc', 'parc', 'parc'], ['parc', 'parc', 'parc', 'parc', 'parc', 'parc']]
* norm_iterlist : [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
* prune_iterlist : [['2', '2', '2', '2', '2', '2'], ['2', '2', '2', '2', '2', '2']]
* roi_iterlist : [[None, None, None, None, None, None], [None, None, None, None, None, None]]
* thr_iterlist : [['0.05', '0.06', '0.07', '0.08', '0.09', '0.1'], ['0.05', '0.06', '0.07', '0.08', '0.09', '0.1']]


Execution Outputs
-----------------


* ID_iterlist : [['0021001', '0021001', '0021001', '0021001', '0021001', '0021001'], ['0021001', '0021001', '0021001', '0021001', '0021001', '0021001']]
* binary_iterlist : [[False, False, False, False, False, False], [False, False, False, False, False, False]]
* conn_model_iterlist : [['csd', 'csd', 'csd', 'csd', 'csd', 'csd'], ['csd', 'csd', 'csd', 'csd', 'csd', 'csd']]
* est_path_iterlist : [['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy'], ['/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy', '/Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/0021001_Default_est_csd_0.1dens_100000samples_particle_track.npy']]
* network_iterlist : [['Default', 'Default', 'Default', 'Default', 'Default', 'Default'], ['Default', 'Default', 'Default', 'Default', 'Default', 'Default']]
* node_size_iterlist : [['parc', 'parc', 'parc', 'parc', 'parc', 'parc'], ['parc', 'parc', 'parc', 'parc', 'parc', 'parc']]
* norm_iterlist : [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
* prune_iterlist : [['2', '2', '2', '2', '2', '2'], ['2', '2', '2', '2', '2', '2']]
* roi_iterlist : [[None, None, None, None, None, None], [None, None, None, None, None, None]]
* thr_iterlist : [['0.05', '0.06', '0.07', '0.08', '0.09', '0.1'], ['0.05', '0.06', '0.07', '0.08', '0.09', '0.1']]


Runtime info
------------


* duration : 0.005556
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_271/meta_wf_0021001/_conn_model_csd/pass_meta_outs_node


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

