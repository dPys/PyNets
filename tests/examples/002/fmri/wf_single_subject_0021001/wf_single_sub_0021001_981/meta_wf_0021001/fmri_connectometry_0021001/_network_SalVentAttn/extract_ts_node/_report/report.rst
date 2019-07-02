Node: meta_wf_0021001 (fmri_connectometry_0021001 (extract_ts_node (utility)
============================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.extract_ts_node
 Exec ID : extract_ts_node.c1


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* block_size : None
* c_boot : 0
* conf : None
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* func_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* function_str : def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, roi, dir_path, ID, network, smooth, atlas,
                    uatlas, labels, c_boot, block_size, hpass, detrending=True):
    """
    API for employing Nilearn's NiftiLabelsMasker to extract fMRI time-series data from spherical ROI's based on a
    given 3D atlas image of integer-based voxel intensities. The resulting time-series can then optionally be resampled
    using circular-block bootrapping. The final 2D m x n array is ultimately saved to file in .npy format

    Parameters
    ----------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    func_file : str
        File path to a preprocessed functional Nifti1Image in standard space.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
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
        n = number of ROI's, where ROI's are parcel volumes.
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
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
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

    parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
                                                 standardize=True, smoothing_fwhm=float(smooth), high_pass=hpass,
                                                 detrend=detrending, verbose=2, resampling_target='data')
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    if ts_within_nodes is None:
        raise RuntimeError('\nERROR: Time-series extraction failed!')
    if float(c_boot) > 0:
        print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
        ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' volumetric ROI\'s'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    print("%s%s%s" % ('Applying high-pass filter: ', hpass, ' Hz\n'))

    # Save time series as file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    node_size = None
    return ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas, labels, coords, c_boot, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* net_parcels_map_nifti : <class 'nibabel.nifti1.Nifti1Image'>
data shape (182, 218, 182)
affine: 
[[  -1.   -0.    0.   90.]
 [  -0.    1.   -0. -126.]
 [   0.    0.    1.  -72.]
 [   0.    0.    0.    1.]]
metadata:
<class 'nibabel.nifti1.Nifti1Header'> object, endian='<'
sizeof_hdr      : 348
data_type       : b''
db_name         : b''
extents         : 0
session_error   : 0
regular         : b''
dim_info        : 0
dim             : [  3 182 218 182   1   1   1   1]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : float64
bitpix          : 64
slice_start     : 0
pixdim          : [-1.  1.  1.  1.  1.  1.  1.  1.]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 0
slice_code      : unknown
xyzt_units      : 0
cal_max         : 0.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 0
glmin           : 0
descrip         : b''
aux_file        : b''
qform_code      : unknown
sform_code      : aligned
quatern_b       : 0.0
quatern_c       : 1.0
quatern_d       : 0.0
qoffset_x       : 90.0
qoffset_y       : -126.0
qoffset_z       : -72.0
srow_x          : [-1. -0.  0. 90.]
srow_y          : [  -0.    1.   -0. -126.]
srow_z          : [  0.   0.   1. -72.]
intent_name     : b''
magic           : b'n+1'
* network : SalVentAttn
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* block_size : None
* c_boot : 0
* conf : None
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* func_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* function_str : def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, roi, dir_path, ID, network, smooth, atlas,
                    uatlas, labels, c_boot, block_size, hpass, detrending=True):
    """
    API for employing Nilearn's NiftiLabelsMasker to extract fMRI time-series data from spherical ROI's based on a
    given 3D atlas image of integer-based voxel intensities. The resulting time-series can then optionally be resampled
    using circular-block bootrapping. The final 2D m x n array is ultimately saved to file in .npy format

    Parameters
    ----------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    func_file : str
        File path to a preprocessed functional Nifti1Image in standard space.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
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
        n = number of ROI's, where ROI's are parcel volumes.
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
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
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

    parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
                                                 standardize=True, smoothing_fwhm=float(smooth), high_pass=hpass,
                                                 detrend=detrending, verbose=2, resampling_target='data')
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    if ts_within_nodes is None:
        raise RuntimeError('\nERROR: Time-series extraction failed!')
    if float(c_boot) > 0:
        print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
        ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' volumetric ROI\'s'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    print("%s%s%s" % ('Applying high-pass filter: ', hpass, ' Hz\n'))

    # Save time series as file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    node_size = None
    return ts_within_nodes, node_size, smooth, dir_path, atlas, uatlas, labels, coords, c_boot, hpass

* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* net_parcels_map_nifti : <class 'nibabel.nifti1.Nifti1Image'>
data shape (182, 218, 182)
affine: 
[[  -1.   -0.    0.   90.]
 [  -0.    1.   -0. -126.]
 [   0.    0.    1.  -72.]
 [   0.    0.    0.    1.]]
metadata:
<class 'nibabel.nifti1.Nifti1Header'> object, endian='<'
sizeof_hdr      : 348
data_type       : b''
db_name         : b''
extents         : 0
session_error   : 0
regular         : b''
dim_info        : 0
dim             : [  3 182 218 182   1   1   1   1]
intent_p1       : 0.0
intent_p2       : 0.0
intent_p3       : 0.0
intent_code     : none
datatype        : float64
bitpix          : 64
slice_start     : 0
pixdim          : [-1.  1.  1.  1.  1.  1.  1.  1.]
vox_offset      : 0.0
scl_slope       : nan
scl_inter       : nan
slice_end       : 0
slice_code      : unknown
xyzt_units      : 0
cal_max         : 0.0
cal_min         : 0.0
slice_duration  : 0.0
toffset         : 0.0
glmax           : 0
glmin           : 0
descrip         : b''
aux_file        : b''
qform_code      : unknown
sform_code      : aligned
quatern_b       : 0.0
quatern_c       : 1.0
quatern_d       : 0.0
qoffset_x       : 90.0
qoffset_y       : -126.0
qoffset_z       : -72.0
srow_x          : [-1. -0.  0. 90.]
srow_y          : [  -0.    1.   -0. -126.]
srow_z          : [  0.   0.   1. -72.]
intent_name     : b''
magic           : b'n+1'
* network : SalVentAttn
* roi : None
* smooth : 0
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Execution Outputs
-----------------


* atlas : DesikanKlein2012
* c_boot : 0
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* hpass : None
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* node_size : None
* smooth : 0
* ts_within_nodes : [[ 1.8216035   1.8867463   2.428473   ...  2.0709388   1.4977248
   2.0285647 ]
 [-0.42965093 -0.47361463 -0.12245914 ... -0.17132154  0.5810947
   0.29684192]
 [ 0.4241159  -0.58999395  0.5557735  ... -0.16899581 -0.45163587
   0.26892996]
 ...
 [ 0.32354212 -0.34934202  0.3257923  ...  0.19165032 -0.01654548
  -0.08241437]
 [ 0.4459056   1.1703486   0.2037435  ...  1.6493644   0.77825814
   0.44683447]
 [-0.16782938  0.36983198 -0.65302557 ...  0.8611615  -1.2669188
  -1.1072674 ]]
* uatlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri/anat_reg/DesikanKlein2012_t1w_mni_gm.nii.gz


Runtime info
------------


* duration : 19.345368
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_SalVentAttn/extract_ts_node


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

