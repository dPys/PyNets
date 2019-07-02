Node: meta_wf_0021001 (dmri_connectometry_0021001 (get_fa_node (utility)
========================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.dmri_connectometry_0021001.get_fa_node
 Exec ID : get_fa_node.b1


Original Inputs
---------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* function_str : def tens_mod_fa_est(gtab_file, dwi_file, B0_mask):
    '''
    Estimate a tensor FA image to use for registrations.

    Parameters
    ----------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted Nifti1Image.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    import os
    from dipy.io import load_pickle
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dti import fractional_anisotropy

    data = nib.load(dwi_file).get_fdata()
    gtab = load_pickle(gtab_file)

    print('Generating simple tensor FA image to use for registrations...')
    nodif_B0_img = nib.load(B0_mask)
    B0_mask_data = nodif_B0_img.get_fdata().astype('bool')
    nodif_B0_affine = nodif_B0_img.affine
    model = TensorModel(gtab)
    mod = model.fit(data, B0_mask_data)
    FA = fractional_anisotropy(mod.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA.astype(np.float32), nodif_B0_affine)
    fa_path = "%s%s" % (os.path.dirname(B0_mask), '/tensor_fa.nii.gz')
    nib.save(fa_img, fa_path)
    return fa_path, B0_mask, gtab_file, dwi_file

* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl

Execution Inputs
----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* function_str : def tens_mod_fa_est(gtab_file, dwi_file, B0_mask):
    '''
    Estimate a tensor FA image to use for registrations.

    Parameters
    ----------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted Nifti1Image.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    import os
    from dipy.io import load_pickle
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dti import fractional_anisotropy

    data = nib.load(dwi_file).get_fdata()
    gtab = load_pickle(gtab_file)

    print('Generating simple tensor FA image to use for registrations...')
    nodif_B0_img = nib.load(B0_mask)
    B0_mask_data = nodif_B0_img.get_fdata().astype('bool')
    nodif_B0_affine = nodif_B0_img.affine
    model = TensorModel(gtab)
    mod = model.fit(data, B0_mask_data)
    FA = fractional_anisotropy(mod.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA.astype(np.float32), nodif_B0_affine)
    fa_path = "%s%s" % (os.path.dirname(B0_mask), '/tensor_fa.nii.gz')
    nib.save(fa_img, fa_path)
    return fa_path, B0_mask, gtab_file, dwi_file

* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl


Execution Outputs
-----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/B0_bet_mask.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/tensor_fa.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/dmri_tmp/gtab.pkl


Runtime info
------------


* duration : 12.33027
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/get_fa_node


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

