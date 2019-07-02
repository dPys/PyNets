Node: meta_wf_0021001 (dmri_connectometry_0021001 (check_orient_and_dims_anat_node (utility)
============================================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.check_orient_and_dims_anat_node
 Exec ID : check_orient_and_dims_anat_node


Original Inputs
---------------


* function_str : def check_orient_and_dims(infile, vox_size, bvecs=None):
    """
    An API to reorient any image to RAS+ and resample any image to a given voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a dwi Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    reoriented = "%s%s%s%s" % (outdir, '/', infile.split('/')[-1].split('.nii.gz')[0], '_pre_reor.nii.gz')
    resampled = "%s%s%s%s" % (outdir, '/', os.path.basename(infile).split('.nii.gz')[0], '_pre_res.nii.gz')

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        outdir = "%s%s" % (outdir, '/std_dmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='dwi')
    elif (vols > 1) and (bvecs is None):
        # func case
        outdir = "%s%s" % (outdir, '/std_fmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='func')
    else:
        # t1w case
        outdir = "%s%s" % (outdir, '/std_anat_')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        if not os.path.isfile(resampled):
            # Check dimensions
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='t1w')

    print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs

* infile : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/s002_anat_brain.nii.gz
* vox_size : 2mm

Execution Inputs
----------------


* function_str : def check_orient_and_dims(infile, vox_size, bvecs=None):
    """
    An API to reorient any image to RAS+ and resample any image to a given voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a dwi Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    reoriented = "%s%s%s%s" % (outdir, '/', infile.split('/')[-1].split('.nii.gz')[0], '_pre_reor.nii.gz')
    resampled = "%s%s%s%s" % (outdir, '/', os.path.basename(infile).split('.nii.gz')[0], '_pre_res.nii.gz')

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        outdir = "%s%s" % (outdir, '/std_dmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='dwi')
    elif (vols > 1) and (bvecs is None):
        # func case
        outdir = "%s%s" % (outdir, '/std_fmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='func')
    else:
        # t1w case
        outdir = "%s%s" % (outdir, '/std_anat_')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        if not os.path.isfile(resampled):
            # Check dimensions
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='t1w')

    print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs

* infile : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/s002_anat_brain.nii.gz
* vox_size : 2mm


Execution Outputs
-----------------


* outfile : /Users/derekpisner/Applications/PyNets/tests/examples/002/anat/std_fmri/s002_anat_brain_pre_reor_1747.nii.gz


Runtime info
------------


* duration : 0.556405
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/check_orient_and_dims_anat_node


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

