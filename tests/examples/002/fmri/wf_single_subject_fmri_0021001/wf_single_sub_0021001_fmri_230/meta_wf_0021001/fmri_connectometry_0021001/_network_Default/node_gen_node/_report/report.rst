Node: meta_wf_0021001 (fmri_connectometry_0021001 (node_gen_node (utility)
==========================================================================


 Hierarchy : wf_single_sub_0021001_fmri_230.meta_wf_0021001.fmri_connectometry_0021001.node_gen_node
 Exec ID : node_gen_node.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* coords : [(-40.0, 32.0, -10.0), (46.0, -66.0, 30.0), (52.0, -6.0, -8.0), (-12.0, 34.0, 42.0), (-36.0, 12.0, 48.0), (42.0, 34.0, -12.0), (-50.0, -10.0, -6.0), (10.0, -58.0, 38.0), (8.0, -44.0, 20.0), (48.0, 32.0, 4.0), (6.0, -16.0, 40.0), (-6.0, -18.0, 40.0), (-46.0, 14.0, 12.0), (-46.0, 32.0, 6.0), (-8.0, -60.0, 38.0), (-22.0, -32.0, -18.0), (-58.0, -28.0, -12.0), (-42.0, -70.0, 32.0), (-6.0, 40.0, 6.0), (58.0, -26.0, -12.0), (6.0, 38.0, 2.0), (-6.0, -46.0, 20.0), (6.0, 38.0, -18.0), (-6.0, 42.0, -16.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* function_str : def node_gen(coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas):
    """
    In the case that masking was not applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional roi masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coords_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* par_max : 96
* parc : False
* parcel_list : None
* roi : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* coords : [(-40.0, 32.0, -10.0), (46.0, -66.0, 30.0), (52.0, -6.0, -8.0), (-12.0, 34.0, 42.0), (-36.0, 12.0, 48.0), (42.0, 34.0, -12.0), (-50.0, -10.0, -6.0), (10.0, -58.0, 38.0), (8.0, -44.0, 20.0), (48.0, 32.0, 4.0), (6.0, -16.0, 40.0), (-6.0, -18.0, 40.0), (-46.0, 14.0, 12.0), (-46.0, 32.0, 6.0), (-8.0, -60.0, 38.0), (-22.0, -32.0, -18.0), (-58.0, -28.0, -12.0), (-42.0, -70.0, 32.0), (-6.0, 40.0, 6.0), (58.0, -26.0, -12.0), (6.0, 38.0, 2.0), (-6.0, -46.0, 20.0), (6.0, 38.0, -18.0), (-6.0, 42.0, -16.0)]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* function_str : def node_gen(coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas):
    """
    In the case that masking was not applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional roi masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coords_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* par_max : 96
* parc : False
* parcel_list : None
* roi : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* atlas : DesikanKlein2012
* coords : [(-40.0, 32.0, -10.0), (46.0, -66.0, 30.0), (52.0, -6.0, -8.0), (-12.0, 34.0, 42.0), (-36.0, 12.0, 48.0), (42.0, 34.0, -12.0), (-50.0, -10.0, -6.0), (10.0, -58.0, 38.0), (8.0, -44.0, 20.0), (48.0, 32.0, 4.0), (6.0, -16.0, 40.0), (-6.0, -18.0, 40.0), (-46.0, 14.0, 12.0), (-46.0, 32.0, 6.0), (-8.0, -60.0, 38.0), (-22.0, -32.0, -18.0), (-58.0, -28.0, -12.0), (-42.0, -70.0, 32.0), (-6.0, 40.0, 6.0), (58.0, -26.0, -12.0), (6.0, 38.0, 2.0), (-6.0, -46.0, 20.0), (6.0, 38.0, -18.0), (-6.0, 42.0, -16.0)]
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* net_parcels_map_nifti : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 0.001626
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_0021001/wf_single_sub_0021001_fmri_230/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/node_gen_node


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

