Node: meta_wf_0021001 (dmri_connectometry_0021001 (gtab_node (utility)
======================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.gtab_node
 Exec ID : gtab_node.c1


Original Inputs
---------------


* atlas : DesikanKlein2012
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fbval : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/bval.bval
* fbvec : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/std_dmri/bvecs.bvec
* function_str : def make_gtab_and_bmask(fbval, fbvec, dwi_file, network, node_size, atlas):
    """
    Create gradient table from bval/bvec, and a mean B0 brain mask.

    Parameters
    ----------
    fbval : str
        File name of the b-values file.
    fbvec : str
        File name of the b-vectors file.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.

    Returns
    -------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    nodif_b0_bet : str
        File path to mean brain-extracted B0 image.
    B0_mask : str
        File path to mean B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import os
    from dipy.io import save_pickle
    import os.path as op
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from nilearn.image import mean_img
    from pynets.utils import rescale_bvec

    outdir = op.dirname(dwi_file)

    B0 = "%s%s" % (outdir, "/b0.nii.gz")
    B0_bet = "%s%s" % (outdir, "/b0_bet.nii.gz")
    B0_mask = "%s%s" % (outdir, "/B0_bet_mask.nii.gz")
    bvec_rescaled = "%s%s" % (outdir, "/bvec_scaled.bvec")
    gtab_file = "%s%s" % (outdir, "/gtab.pkl")

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[np.where(np.any(abs(bvecs) >= 10, axis=1) == True)] = [1, 0, 0]
    bvecs[np.where(bvals == 0)] = 0
    if len(bvecs[np.where(np.logical_and(bvals > 50, np.all(abs(bvecs) == np.array([0, 0, 0]), axis=1)))]) > 0:
        raise ValueError('WARNING: Encountered potentially corrupted bval/bvecs. Check to ensure volumes with a '
                         'diffusion weighting are not being treated as B0\'s along the bvecs')
    np.savetxt(fbval, bvals)
    np.savetxt(fbvec, bvecs)
    bvec_rescaled = rescale_bvec(fbvec, bvec_rescaled)
    if fbval and bvec_rescaled:
        bvals, bvecs = read_bvals_bvecs(fbval, bvec_rescaled)
    else:
        raise ValueError('Either bvals or bvecs files not found (or rescaling failed)!')

    # Creating the gradient table
    gtab = gradient_table(bvals, bvecs)

    # Correct b0 threshold
    gtab.b0_threshold = min(bvals)

    # Get b0 indices
    b0s = np.where(gtab.bvals == gtab.b0_threshold)[0]
    print("%s%s" % ('b0\'s found at: ', b0s))

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    # Extract and Combine all b0s collected
    print('Extracting b0\'s...')
    cmds = []
    b0s_bbr = []
    for b0 in b0s:
        print(b0)
        b0_bbr = "{}/{}_b0.nii.gz".format(outdir, str(b0))
        cmds.append('fslroi {} {} {} 1'.format(dwi_file, b0_bbr, str(b0), ' 1'))
        b0s_bbr.append(b0_bbr)

    for cmd in cmds:
        os.system(cmd)

    # Get mean b0
    mean_b0 = mean_img(b0s_bbr)
    nib.save(mean_b0, B0)

    # Get mean b0 brain mask
    os.system("bet {} {} -m -f 0.2".format(B0, B0_bet))
    return gtab_file, B0_bet, B0_mask, dwi_file

* network : SalVentAttn
* node_size : None

Execution Inputs
----------------


* atlas : DesikanKlein2012
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fbval : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/bval.bval
* fbvec : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/std_dmri/bvecs.bvec
* function_str : def make_gtab_and_bmask(fbval, fbvec, dwi_file, network, node_size, atlas):
    """
    Create gradient table from bval/bvec, and a mean B0 brain mask.

    Parameters
    ----------
    fbval : str
        File name of the b-values file.
    fbvec : str
        File name of the b-vectors file.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.

    Returns
    -------
    gtab_file : str
        File path to pickled DiPy gradient table object.
    nodif_b0_bet : str
        File path to mean brain-extracted B0 image.
    B0_mask : str
        File path to mean B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import os
    from dipy.io import save_pickle
    import os.path as op
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from nilearn.image import mean_img
    from pynets.utils import rescale_bvec

    outdir = op.dirname(dwi_file)

    B0 = "%s%s" % (outdir, "/b0.nii.gz")
    B0_bet = "%s%s" % (outdir, "/b0_bet.nii.gz")
    B0_mask = "%s%s" % (outdir, "/B0_bet_mask.nii.gz")
    bvec_rescaled = "%s%s" % (outdir, "/bvec_scaled.bvec")
    gtab_file = "%s%s" % (outdir, "/gtab.pkl")

    # loading bvecs/bvals
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[np.where(np.any(abs(bvecs) >= 10, axis=1) == True)] = [1, 0, 0]
    bvecs[np.where(bvals == 0)] = 0
    if len(bvecs[np.where(np.logical_and(bvals > 50, np.all(abs(bvecs) == np.array([0, 0, 0]), axis=1)))]) > 0:
        raise ValueError('WARNING: Encountered potentially corrupted bval/bvecs. Check to ensure volumes with a '
                         'diffusion weighting are not being treated as B0\'s along the bvecs')
    np.savetxt(fbval, bvals)
    np.savetxt(fbvec, bvecs)
    bvec_rescaled = rescale_bvec(fbvec, bvec_rescaled)
    if fbval and bvec_rescaled:
        bvals, bvecs = read_bvals_bvecs(fbval, bvec_rescaled)
    else:
        raise ValueError('Either bvals or bvecs files not found (or rescaling failed)!')

    # Creating the gradient table
    gtab = gradient_table(bvals, bvecs)

    # Correct b0 threshold
    gtab.b0_threshold = min(bvals)

    # Get b0 indices
    b0s = np.where(gtab.bvals == gtab.b0_threshold)[0]
    print("%s%s" % ('b0\'s found at: ', b0s))

    # Show info
    print(gtab.info)

    # Save gradient table to pickle
    save_pickle(gtab_file, gtab)

    # Extract and Combine all b0s collected
    print('Extracting b0\'s...')
    cmds = []
    b0s_bbr = []
    for b0 in b0s:
        print(b0)
        b0_bbr = "{}/{}_b0.nii.gz".format(outdir, str(b0))
        cmds.append('fslroi {} {} {} 1'.format(dwi_file, b0_bbr, str(b0), ' 1'))
        b0s_bbr.append(b0_bbr)

    for cmd in cmds:
        os.system(cmd)

    # Get mean b0
    mean_b0 = mean_img(b0s_bbr)
    nib.save(mean_b0, B0)

    # Get mean b0 brain mask
    os.system("bet {} {} -m -f 0.2".format(B0, B0_bet))
    return gtab_file, B0_bet, B0_mask, dwi_file

* network : SalVentAttn
* node_size : None


Execution Outputs
-----------------


* B0_bet : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/b0_bet.nii.gz
* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl


Runtime info
------------


* duration : 2.292402
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_SalVentAttn/gtab_node


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

