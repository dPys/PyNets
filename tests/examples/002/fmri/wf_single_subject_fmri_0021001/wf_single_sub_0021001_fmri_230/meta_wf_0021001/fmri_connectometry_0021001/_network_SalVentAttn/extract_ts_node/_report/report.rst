Node: meta_wf_0021001 (fmri_connectometry_0021001 (extract_ts_node (utility)
============================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.extract_ts_node
 Exec ID : extract_ts_node.c1


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* block_size : None
* c_boot : 0
* conf : None
* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* func_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* function_str : def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, roi, network, smooth, atlas,
                      uatlas, labels, c_boot, block_size, hpass, detrending=True):
    """
    API for employing Nilearn's NiftiSpheresMasker to extract fMRI time-series data from spherical ROI's based on a
    given list of seed coordinates. The resulting time-series can then optionally be resampled using circular-block
    bootrapping. The final 2D m x n array is ultimately saved to file in .npy format

    Parameters
    ----------
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    func_file : str
        File path to a preprocessed functional Nifti1Image in standard space.
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set (e.g. a coordinate atlas).
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    block_size : int
        Size bootstrap blocks if bootstrapping (c_boot) is performed.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    detrending : bool
        Indicates whether to remove linear trends from time-series when extracting across nodes. Default is True.

    Returns
    -------
    ts_within_nodes : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's, where ROI's are spheres.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
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
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from nilearn import input_data
    from pynets import utils

    if not op.isfile(func_file):
        raise ValueError('\nERROR: Functional data input not found! Check that the file(s) specified with the -i flag '
                         'exist(s)')

    if conf:
        if not op.isfile(conf):
            raise ValueError('\nERROR: Confound regressor file not found! Check that the file(s) specified with the '
                             '-conf flag exist(s)')

    if len(coords) > 0:
        spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
                                                       standardize=True, smoothing_fwhm=float(smooth), high_pass=hpass,
                                                       detrend=detrending, verbose=2)
        ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)
        if float(c_boot) > 0:
            print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
            ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
        if ts_within_nodes is None:
            raise RuntimeError('\nERROR: Time-series extraction failed!')
    else:
        raise RuntimeError(
            '\nERROR: Cannot extract time-series from an empty list of coordinates. \nThis usually means '
            'that no nodes were generated based on the specified conditions at runtime (e.g. atlas was '
            'overly restricted by an RSN or some user-defined mask.')

    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' coordinate ROI\'s'))
    print("%s%s%s" % ('Using node radius: ', node_size, ' mm'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    print("%s%s%s" % ('Applying high-pass filter: ', hpass, ' Hz\n'))

    # Save time series as txt file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    return ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas, labels, coords, c_boot, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* net_parcels_map_nifti : None
* network : SalVentAttn
* node_size : 4
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* block_size : None
* c_boot : 0
* conf : None
* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* func_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* function_str : def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, roi, network, smooth, atlas,
                      uatlas, labels, c_boot, block_size, hpass, detrending=True):
    """
    API for employing Nilearn's NiftiSpheresMasker to extract fMRI time-series data from spherical ROI's based on a
    given list of seed coordinates. The resulting time-series can then optionally be resampled using circular-block
    bootrapping. The final 2D m x n array is ultimately saved to file in .npy format

    Parameters
    ----------
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    func_file : str
        File path to a preprocessed functional Nifti1Image in standard space.
    coords : list
        List of (x, y, z) tuples corresponding to an a-priori defined set (e.g. a coordinate atlas).
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    block_size : int
        Size bootstrap blocks if bootstrapping (c_boot) is performed.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    detrending : bool
        Indicates whether to remove linear trends from time-series when extracting across nodes. Default is True.

    Returns
    -------
    ts_within_nodes : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's, where ROI's are spheres.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
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
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from nilearn import input_data
    from pynets import utils

    if not op.isfile(func_file):
        raise ValueError('\nERROR: Functional data input not found! Check that the file(s) specified with the -i flag '
                         'exist(s)')

    if conf:
        if not op.isfile(conf):
            raise ValueError('\nERROR: Confound regressor file not found! Check that the file(s) specified with the '
                             '-conf flag exist(s)')

    if len(coords) > 0:
        spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
                                                       standardize=True, smoothing_fwhm=float(smooth), high_pass=hpass,
                                                       detrend=detrending, verbose=2)
        ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)
        if float(c_boot) > 0:
            print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
            ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
        if ts_within_nodes is None:
            raise RuntimeError('\nERROR: Time-series extraction failed!')
    else:
        raise RuntimeError(
            '\nERROR: Cannot extract time-series from an empty list of coordinates. \nThis usually means '
            'that no nodes were generated based on the specified conditions at runtime (e.g. atlas was '
            'overly restricted by an RSN or some user-defined mask.')

    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' coordinate ROI\'s'))
    print("%s%s%s" % ('Using node radius: ', node_size, ' mm'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    print("%s%s%s" % ('Applying high-pass filter: ', hpass, ' Hz\n'))

    # Save time series as txt file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    return ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas, labels, coords, c_boot, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* net_parcels_map_nifti : None
* network : SalVentAttn
* node_size : 4
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* atlas : DesikanKlein2012
* c_boot : 0
* coords : [(-40.0, 32.0, -10.0), (-54.0, -38.0, 34.0), (6.0, 22.0, 28.0), (-50.0, -10.0, -6.0), (48.0, 16.0, 14.0), (52.0, -6.0, -8.0), (48.0, 32.0, 4.0), (38.0, 2.0, 0.0), (6.0, -16.0, 40.0), (54.0, -36.0, 36.0), (-6.0, -18.0, 40.0), (-6.0, 20.0, 32.0), (-36.0, 2.0, 0.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* node_size : 4
* smooth : 0
* ts_within_nodes : [[ 1.3127716   2.5494092   1.5156994  ...  3.0159454   1.6702434
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


Runtime info
------------


* duration : 14.315449
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/extract_ts_node


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

