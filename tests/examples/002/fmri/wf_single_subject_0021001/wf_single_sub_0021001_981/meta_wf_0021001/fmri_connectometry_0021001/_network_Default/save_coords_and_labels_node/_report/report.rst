Node: meta_wf_0021001 (fmri_connectometry_0021001 (save_coords_and_labels_node (utility)
========================================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.save_coords_and_labels_node
 Exec ID : save_coords_and_labels_node.c0


Original Inputs
---------------


* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    import os

    namer_dir = dir_path + '/nodes'
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    # Save coords to pickle
    coord_path = "%s%s%s%s" % (namer_dir, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (namer_dir, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)
    return coord_path, labels_path

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : Default

Execution Inputs
----------------


* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    import os

    namer_dir = dir_path + '/nodes'
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    # Save coords to pickle
    coord_path = "%s%s%s%s" % (namer_dir, '/', network, '_coords_rsn.pkl')
    with open(coord_path, 'wb') as f:
        pickle.dump(coords, f, protocol=2)
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (namer_dir, '/', network, '_labels_rsn.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)
    return coord_path, labels_path

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* network : Default


Execution Outputs
-----------------


* out : ('/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/nodes/Default_coords_rsn.pkl', '/Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012/nodes/Default_labels_rsn.pkl')


Runtime info
------------


* duration : 0.002353
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/save_coords_and_labels_node


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

