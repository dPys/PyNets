Node: meta_wf_0021001 (dmri_connectometry_0021001 (register_atlas_node (utility)
================================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.dmri_connectometry_0021001.register_atlas_node
 Exec ID : register_atlas_node.b1


Original Inputs
---------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_695.nii.gz
* atlas : DesikanKlein2012
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/tensor_fa.nii.gz
* function_str : def register_atlas_dwi(uatlas, atlas, node_size, basedir_path, fa_path, B0_mask, anat_file, wm_gm_int_in_dwi, coords,
                       labels, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, gtab_file, dwi_file, vox_size='2mm',
                       simple=False):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False). 
        Default is True.

    Returns
    -------
    dwi_aligned_atlas_t1mni_wmgm_int : str
        File path to atlas parcellation Nifti1Image in T1w-MNI warped native diffusion space, restricted only to wm-gm
        interface.
    dwi_aligned_atlas : str
        File path to atlas parcellation Nifti1Image in native diffusion space.
    aligned_atlas_t1mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    B0_mask : str
        File path to B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if node_size is not None:
        atlas = "%s%s%s" % (atlas, '_', node_size)

    # Apply warps/coregister atlas to dwi
    [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni] = reg.atlas2t1w2dwi_align(uatlas, atlas)

    return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni, uatlas, atlas, coords, labels, node_size, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, fa_path, gtab_file, B0_mask, dwi_file

* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* node_size : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* vox_size : 2mm
* wm_gm_int_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_gm_int_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz

Execution Inputs
----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_695.nii.gz
* atlas : DesikanKlein2012
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/tensor_fa.nii.gz
* function_str : def register_atlas_dwi(uatlas, atlas, node_size, basedir_path, fa_path, B0_mask, anat_file, wm_gm_int_in_dwi, coords,
                       labels, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, gtab_file, dwi_file, vox_size='2mm',
                       simple=False):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False). 
        Default is True.

    Returns
    -------
    dwi_aligned_atlas_t1mni_wmgm_int : str
        File path to atlas parcellation Nifti1Image in T1w-MNI warped native diffusion space, restricted only to wm-gm
        interface.
    dwi_aligned_atlas : str
        File path to atlas parcellation Nifti1Image in native diffusion space.
    aligned_atlas_t1mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    B0_mask : str
        File path to B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if node_size is not None:
        atlas = "%s%s%s" % (atlas, '_', node_size)

    # Apply warps/coregister atlas to dwi
    [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni] = reg.atlas2t1w2dwi_align(uatlas, atlas)

    return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni, uatlas, atlas, coords, labels, node_size, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, fa_path, gtab_file, B0_mask, dwi_file

* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* node_size : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* vox_size : 2mm
* wm_gm_int_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_gm_int_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz


Execution Outputs
-----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* aligned_atlas_t1mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* atlas : DesikanKlein2012
* coords : [(54.18698938688249, -35.04665190501642, 36.22738031610292), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (7.537630258587413, -24.854882284832115, 58.69490544191433), (-5.213371266002838, -18.392603129445234, 39.69630156472262), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (20.55077720207254, -2.8352331606217547, -0.7077720207253861), (-4.042709376042708, -57.24758091424758, -38.759426092759426), (5.230061349693258, 37.48432174505794, -17.26993865030675), (24.10399334442596, -29.37895174708818, -18.434276206322792), (-22.872261264985525, -5.03183133526251, -32.042579578338156), (34.79281102438084, -43.04760528090971, -20.99951816517298), (-35.830662735546724, 12.090244840405063, 47.08011793359826), (13.69881910335458, -66.42386874281534, -5.0391890479673975)]
* dwi_aligned_atlas : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track.nii.gz
* dwi_aligned_atlas_wmgm_int : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track_wmgm_int.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/tensor_fa.nii.gz
* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* node_size : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz


Runtime info
------------


* duration : 132.101636
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/register_atlas_node


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

