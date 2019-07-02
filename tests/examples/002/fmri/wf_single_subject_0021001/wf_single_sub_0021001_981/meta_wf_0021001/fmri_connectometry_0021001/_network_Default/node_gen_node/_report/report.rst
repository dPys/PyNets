Node: meta_wf_0021001 (fmri_connectometry_0021001 (node_gen_node (utility)
==========================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.node_gen_node
 Exec ID : node_gen_node.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* par_max : 96
* parc : True
* parcel_list : [<nibabel.nifti1.Nifti1Image object at 0xd1abe6e48>, <nibabel.nifti1.Nifti1Image object at 0xa16a647b8>, <nibabel.nifti1.Nifti1Image object at 0xa16a640f0>, <nibabel.nifti1.Nifti1Image object at 0xa16a64400>, <nibabel.nifti1.Nifti1Image object at 0xa16a60e80>, <nibabel.nifti1.Nifti1Image object at 0xa16a60c50>, <nibabel.nifti1.Nifti1Image object at 0xa16a60f60>, <nibabel.nifti1.Nifti1Image object at 0xa16a60860>, <nibabel.nifti1.Nifti1Image object at 0xa17b640f0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64240>, <nibabel.nifti1.Nifti1Image object at 0xa17b64390>, <nibabel.nifti1.Nifti1Image object at 0xa17b644e0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64630>, <nibabel.nifti1.Nifti1Image object at 0xa17b64780>, <nibabel.nifti1.Nifti1Image object at 0xa17b648d0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64a20>, <nibabel.nifti1.Nifti1Image object at 0xa17b64b70>, <nibabel.nifti1.Nifti1Image object at 0xa17b64cc0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64e10>, <nibabel.nifti1.Nifti1Image object at 0xa17b64f60>, <nibabel.nifti1.Nifti1Image object at 0xa17b5d0f0>, <nibabel.nifti1.Nifti1Image object at 0xa17b5d240>]
* roi : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
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

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* par_max : 96
* parc : True
* parcel_list : [<nibabel.nifti1.Nifti1Image object at 0xd1abe6e48>, <nibabel.nifti1.Nifti1Image object at 0xa16a647b8>, <nibabel.nifti1.Nifti1Image object at 0xa16a640f0>, <nibabel.nifti1.Nifti1Image object at 0xa16a64400>, <nibabel.nifti1.Nifti1Image object at 0xa16a60e80>, <nibabel.nifti1.Nifti1Image object at 0xa16a60c50>, <nibabel.nifti1.Nifti1Image object at 0xa16a60f60>, <nibabel.nifti1.Nifti1Image object at 0xa16a60860>, <nibabel.nifti1.Nifti1Image object at 0xa17b640f0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64240>, <nibabel.nifti1.Nifti1Image object at 0xa17b64390>, <nibabel.nifti1.Nifti1Image object at 0xa17b644e0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64630>, <nibabel.nifti1.Nifti1Image object at 0xa17b64780>, <nibabel.nifti1.Nifti1Image object at 0xa17b648d0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64a20>, <nibabel.nifti1.Nifti1Image object at 0xa17b64b70>, <nibabel.nifti1.Nifti1Image object at 0xa17b64cc0>, <nibabel.nifti1.Nifti1Image object at 0xa17b64e10>, <nibabel.nifti1.Nifti1Image object at 0xa17b64f60>, <nibabel.nifti1.Nifti1Image object at 0xa17b5d0f0>, <nibabel.nifti1.Nifti1Image object at 0xa17b5d240>]
* roi : None
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* atlas : DesikanKlein2012
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
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
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 3.07424
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/_network_Default/node_gen_node


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

