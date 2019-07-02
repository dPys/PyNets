Node: meta_wf_0021001 (dmri_connectometry_0021001 (run_tracking_node (utility)
==============================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.run_tracking_node
 Exec ID : run_tracking_node.bI.b0.c0


Original Inputs
---------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* curv_thr_list : [60, 30, 10]
* dens_thresh : True
* directget : prob
* disp_filt : False
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def run_track(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, labels_im_file_wm_gm_int,
              labels_im_file, target_samples, curv_thr_list, step_list, track_type, max_length, maxcrossing, directget,
              conn_model, gtab_file, dwi_file, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc,
              prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, life_run, min_length,
              fa_path):
    '''
    Run all ensemble tractography and filtering routines.

    Parameters
    ----------
    B0_mask : str
        File path to B0 brain mask.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image.
    tiss_class : str
        Tissue classification method.
    labels_im_file_wm_gm_int : str
        File path to atlas parcellation Nifti1Image in T1w-warped native diffusion space, restricted to wm-gm interface.
    labels_im_file : str
        File path to atlas parcellation Nifti1Image in T1w-warped native diffusion space.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    max_length : int
        Maximum fiber length threshold in mm to restrict tracking.
    maxcrossing : int
        Maximum number if diffusion directions that can be assumed per voxel while tracking.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    life_run : bool
        Indicates whether to perform Linear Fascicle Evaluation (LiFE).
    min_length : int
        Minimum fiber length threshold in mm.
    fa_path : str
        File path to FA Nifti1Image.

    Returns
    -------
    streams : str
        File path to save streamline array sequence in .trk format.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    fa_path : str
        File path to FA Nifti1Image.
    dm_path : str
        File path to fiber density map Nifti1Image.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.io import load_pickle
    from colorama import Fore, Style
    from dipy.data import get_sphere
    from pynets import utils
    from pynets.dmri.track import prep_tissues, reconstruction, filter_streamlines, track_ensemble

    # Load gradient table
    gtab = load_pickle(gtab_file)

    # Fit diffusion model
    mod_fit = reconstruction(conn_model, gtab, dwi_file, wm_in_dwi)

    # Load atlas parcellation (and its wm-gm interface reduced version for seeding)
    atlas_img = nib.load(labels_im_file)
    atlas_data = atlas_img.get_fdata().astype('int')
    atlas_img_wm_gm_int = nib.load(labels_im_file_wm_gm_int)
    atlas_data_wm_gm_int = atlas_img_wm_gm_int.get_fdata().astype('int')

    # Build mask vector from atlas for later roi filtering
    parcels = []
    i = 0
    for roi_val in np.unique(atlas_data)[1:]:
        parcels.append(atlas_data == roi_val)
        i = i + 1

    # Get sphere
    sphere = get_sphere('repulsion724')

    # Instantiate tissue classifier
    tiss_classifier = prep_tissues(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class)

    if np.sum(atlas_data) == 0:
        raise ValueError('ERROR: No non-zero voxels found in atlas. Check any roi masks and/or wm-gm interface images '
                         'to verify overlap with dwi-registered atlas.')

    # Iteratively build a list of streamlines for each ROI while tracking
    print("%s%s%s%s" % (Fore.GREEN, 'Target number of samples: ', Fore.BLUE, target_samples))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Using curvature threshold(s): ', Fore.BLUE, curv_thr_list))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Using step size(s): ', Fore.BLUE, step_list))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Tracking type: ', Fore.BLUE, track_type))
    print(Style.RESET_ALL)
    if directget == 'prob':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Probabilistic Direction...'))
    elif directget == 'boot':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Bootstrapped Direction...'))
    elif directget == 'closest':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Closest Peak Direction...'))
    elif directget == 'det':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Deterministic Maximum Direction...'))
    print(Style.RESET_ALL)

    # Commence Ensemble Tractography
    streamlines = track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, mod_fit, tiss_classifier, sphere,
                                 directget, curv_thr_list, step_list, track_type, maxcrossing, max_length)
    print('Tracking Complete')

    # Perform streamline filtering routines
    dir_path = utils.do_dir_path(atlas, dwi_file)
    [streams, dir_path, dm_path] = filter_streamlines(dwi_file, dir_path, gtab, streamlines, life_run, min_length,
                                                      conn_model, target_samples, node_size, curv_thr_list, step_list,
                                                      network, roi)

    return streams, track_type, target_samples, conn_model, dir_path, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, curv_thr_list, step_list, fa_path, dm_path

* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* labels_im_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track.nii.gz
* labels_im_file_wm_gm_int : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track_wmgm_int.nii.gz
* life_run : False
* max_length : 200
* maxcrossing : 2
* min_length : 20
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* step_list : [0.2, 0.3, 0.4, 0.5]
* target_samples : 100000
* tiss_class : cmc
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz

Execution Inputs
----------------


* B0_mask : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/B0_bet_mask.nii.gz
* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* curv_thr_list : [60, 30, 10]
* dens_thresh : True
* directget : prob
* disp_filt : False
* dwi_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* function_str : def run_track(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, labels_im_file_wm_gm_int,
              labels_im_file, target_samples, curv_thr_list, step_list, track_type, max_length, maxcrossing, directget,
              conn_model, gtab_file, dwi_file, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc,
              prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, life_run, min_length,
              fa_path):
    '''
    Run all ensemble tractography and filtering routines.

    Parameters
    ----------
    B0_mask : str
        File path to B0 brain mask.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image.
    tiss_class : str
        Tissue classification method.
    labels_im_file_wm_gm_int : str
        File path to atlas parcellation Nifti1Image in T1w-warped native diffusion space, restricted to wm-gm interface.
    labels_im_file : str
        File path to atlas parcellation Nifti1Image in T1w-warped native diffusion space.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    max_length : int
        Maximum fiber length threshold in mm to restrict tracking.
    maxcrossing : int
        Maximum number if diffusion directions that can be assumed per voxel while tracking.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    life_run : bool
        Indicates whether to perform Linear Fascicle Evaluation (LiFE).
    min_length : int
        Minimum fiber length threshold in mm.
    fa_path : str
        File path to FA Nifti1Image.

    Returns
    -------
    streams : str
        File path to save streamline array sequence in .trk format.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    fa_path : str
        File path to FA Nifti1Image.
    dm_path : str
        File path to fiber density map Nifti1Image.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.io import load_pickle
    from colorama import Fore, Style
    from dipy.data import get_sphere
    from pynets import utils
    from pynets.dmri.track import prep_tissues, reconstruction, filter_streamlines, track_ensemble

    # Load gradient table
    gtab = load_pickle(gtab_file)

    # Fit diffusion model
    mod_fit = reconstruction(conn_model, gtab, dwi_file, wm_in_dwi)

    # Load atlas parcellation (and its wm-gm interface reduced version for seeding)
    atlas_img = nib.load(labels_im_file)
    atlas_data = atlas_img.get_fdata().astype('int')
    atlas_img_wm_gm_int = nib.load(labels_im_file_wm_gm_int)
    atlas_data_wm_gm_int = atlas_img_wm_gm_int.get_fdata().astype('int')

    # Build mask vector from atlas for later roi filtering
    parcels = []
    i = 0
    for roi_val in np.unique(atlas_data)[1:]:
        parcels.append(atlas_data == roi_val)
        i = i + 1

    # Get sphere
    sphere = get_sphere('repulsion724')

    # Instantiate tissue classifier
    tiss_classifier = prep_tissues(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class)

    if np.sum(atlas_data) == 0:
        raise ValueError('ERROR: No non-zero voxels found in atlas. Check any roi masks and/or wm-gm interface images '
                         'to verify overlap with dwi-registered atlas.')

    # Iteratively build a list of streamlines for each ROI while tracking
    print("%s%s%s%s" % (Fore.GREEN, 'Target number of samples: ', Fore.BLUE, target_samples))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Using curvature threshold(s): ', Fore.BLUE, curv_thr_list))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Using step size(s): ', Fore.BLUE, step_list))
    print(Style.RESET_ALL)
    print("%s%s%s%s" % (Fore.GREEN, 'Tracking type: ', Fore.BLUE, track_type))
    print(Style.RESET_ALL)
    if directget == 'prob':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Probabilistic Direction...'))
    elif directget == 'boot':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Bootstrapped Direction...'))
    elif directget == 'closest':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Closest Peak Direction...'))
    elif directget == 'det':
        print("%s%s%s" % ('Using ', Fore.MAGENTA, 'Deterministic Maximum Direction...'))
    print(Style.RESET_ALL)

    # Commence Ensemble Tractography
    streamlines = track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, mod_fit, tiss_classifier, sphere,
                                 directget, curv_thr_list, step_list, track_type, maxcrossing, max_length)
    print('Tracking Complete')

    # Perform streamline filtering routines
    dir_path = utils.do_dir_path(atlas, dwi_file)
    [streams, dir_path, dm_path] = filter_streamlines(dwi_file, dir_path, gtab, streamlines, life_run, min_length,
                                                      conn_model, target_samples, node_size, curv_thr_list, step_list,
                                                      network, roi)

    return streams, track_type, target_samples, conn_model, dir_path, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, curv_thr_list, step_list, fa_path, dm_path

* gm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_gm_in_dwi.nii.gz
* gtab_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/gtab.pkl
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* labels_im_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track.nii.gz
* labels_im_file_wm_gm_int : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/DesikanKlein2012_dwi_track_wmgm_int.nii.gz
* life_run : False
* max_length : 200
* maxcrossing : 2
* min_length : 20
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* step_list : [0.2, 0.3, 0.4, 0.5]
* target_samples : 100000
* tiss_class : cmc
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* vent_csf_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_vent_csf_in_dwi.nii.gz
* wm_in_dwi : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/dmri_tmp/reg/imgs/t1w_wm_in_dwi.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* curv_thr_list : [60, 30, 10]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* dm_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/density_map_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5].nii.gz
* fa_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/tensor_fa.nii.gz
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* step_list : [0.2, 0.3, 0.4, 0.5]
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5].trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 89.090342
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_Default/_conn_model_csd/run_tracking_node


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

