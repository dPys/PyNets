Node: meta_wf_0021001 (fmri_connectometry_0021001 (register_node (utility)
==========================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.register_node
 Exec ID : register_node


Original Inputs
---------------


* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_5740.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri
* function_str : def register_all_fmri(basedir_path, anat_file, vox_size, overwrite=True):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is True.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration import register
    reg = register.FmriReg(basedir_path, anat_file, vox_size)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1_aligned_mni) is False):
        # Align t1w to dwi
        reg.t1w2mni_align()

    return

* vox_size : 2mm

Execution Inputs
----------------


* anat_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_5740.nii.gz
* basedir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/reg_fmri
* function_str : def register_all_fmri(basedir_path, anat_file, vox_size, overwrite=True):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is True.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration import register
    reg = register.FmriReg(basedir_path, anat_file, vox_size)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1_aligned_mni) is False):
        # Align t1w to dwi
        reg.t1w2mni_align()

    return

* vox_size : 2mm


Execution Outputs
-----------------


* out : None


Runtime info
------------


* duration : 509.405383
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/register_node


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

