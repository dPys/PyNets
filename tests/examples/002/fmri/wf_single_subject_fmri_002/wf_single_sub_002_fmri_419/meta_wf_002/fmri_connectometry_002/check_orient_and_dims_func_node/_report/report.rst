Node: meta_wf_002 (fmri_connectometry_002 (check_orient_and_dims_func_node (utility)
====================================================================================


 Hierarchy : wf_single_sub_002_fmri_419.meta_wf_002.fmri_connectometry_002.check_orient_and_dims_func_node
 Exec ID : check_orient_and_dims_func_node


Original Inputs
---------------


* function_str : def check_orient_and_dims(infile, vox_size, bvecs=None, overwrite=True):
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
    overwrite : bool
        Boolean indicating whether to overwrite existing outputs. Default is True.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    elif (vols > 1) and (bvecs is None):
        # func case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    else:
        # t1w case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        if ('reor' not in infile) or (overwrite is True):
            # Check dimensions
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs

* infile : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* vox_size : 2mm

Execution Inputs
----------------


* function_str : def check_orient_and_dims(infile, vox_size, bvecs=None, overwrite=True):
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
    overwrite : bool
        Boolean indicating whether to overwrite existing outputs. Default is True.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    elif (vols > 1) and (bvecs is None):
        # func case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if ('reor' not in infile) or (overwrite is True):
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)
    else:
        # t1w case
        # Check orientation
        if ('RAS' not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir)
        if ('reor' not in infile) or (overwrite is True):
            # Check dimensions
            outfile = match_target_vox_res(infile, vox_size, outdir)
            print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs

* infile : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* vox_size : 2mm


Execution Outputs
-----------------


* outfile : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002_reor_RAS_nores2mm.nii.gz


Runtime info
------------


* duration : 18.711184
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_002/wf_single_sub_002_fmri_419/meta_wf_002/fmri_connectometry_002/check_orient_and_dims_func_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.v8x0bpN28D/Render
* CC : /usr/bin/clang
* CFLAGS :  -I/usr/local/opt/libomp/include
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_PYTHON_EXE : /usr/local/anaconda3/bin/python
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include -Xpreprocessor -fopenmp
* CXX : /usr/bin/clang++
* CXXFLAGS :  -I/usr/local/opt/libomp/include
* DISPLAY : dpys:0.0
* DYLD_LIBRARY_PATH : /usr/local/opt/libomp/lib
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
* KMP_DUPLICATE_LIB_OK : True
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/libxml2/lib -L/usr/local/opt/libomp/lib -lomp
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : derekpisner
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* OLDPWD : /Users/derekpisner/Applications/PyNets/tests/examples/002
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/Applications/autoDTI/Batch_scripts:/Users/derekpisner/Applications/autoDTI/Main_scripts:/Users/derekpisner/Applications/autoDTI/Stage_scripts:/Users/derekpisner/Applications/autoDTI/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/DTI_TK:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Conversion_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Motion_plotting_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/QAtools
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.0OgUYjH7Dp/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.2
* TERM_SESSION_ID : 035E9BD7-B8A4-4DE5-A5A5-FD6EC00B205C
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0
* autoDTI_HOME : /Users/derekpisner/Applications/autoDTI

