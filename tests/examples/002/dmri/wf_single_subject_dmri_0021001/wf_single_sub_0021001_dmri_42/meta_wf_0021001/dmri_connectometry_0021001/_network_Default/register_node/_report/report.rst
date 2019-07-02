Node: meta_wf_0021001 (dmri_connectometry_0021001 (register_node (utility)
==========================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.register_node
 Exec ID : register_node.c0


Original Inputs
---------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_1747.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def register_all_dwi(basedir_path, fa_path, B0_mask, anat_file, gtab_file, dwi_file, vox_size='2mm', simple=False,
                     overwrite=True):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False).
        Default is True.
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is True.

    Returns
    -------
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    csf_mask_dwi : str
        File path to CSF tissue segmentation Nifti1Image in native diffusion space.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.

    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1w2dwi) is False):
        # Align t1w to dwi
        reg.t1w2dwi_align()

    if (overwrite is True) or (op.isfile(reg.wm_gm_int_in_dwi) is False):
        # Align tissue
        reg.tissue2dwi_align()

    return reg.wm_gm_int_in_dwi, reg.wm_in_dwi, reg.gm_in_dwi, reg.vent_csf_in_dwi, reg.csf_mask_dwi, anat_file, B0_mask, fa_path, gtab_file, dwi_file

* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl

Execution Inputs
----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_1747.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def register_all_dwi(basedir_path, fa_path, B0_mask, anat_file, gtab_file, dwi_file, vox_size='2mm', simple=False,
                     overwrite=True):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False).
        Default is True.
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is True.

    Returns
    -------
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    csf_mask_dwi : str
        File path to CSF tissue segmentation Nifti1Image in native diffusion space.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.

    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1w2dwi) is False):
        # Align t1w to dwi
        reg.t1w2dwi_align()

    if (overwrite is True) or (op.isfile(reg.wm_gm_int_in_dwi) is False):
        # Align tissue
        reg.tissue2dwi_align()

    return reg.wm_gm_int_in_dwi, reg.wm_in_dwi, reg.gm_in_dwi, reg.vent_csf_in_dwi, reg.csf_mask_dwi, anat_file, B0_mask, fa_path, gtab_file, dwi_file

* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl


Execution Outputs
-----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_1747.nii.gz
* csf_mask_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_csf_mask_dwi.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* wm_gm_int_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_gm_int_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz


Runtime info
------------


* duration : 715.196675
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_Default/register_node


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

