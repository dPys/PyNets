#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import nibabel as nib
warnings.filterwarnings("ignore")


def reconstruction(conn_model, gtab, dwi_data, B0_mask):
    """
    Estimate a tensor model from dwi data.

    Parameters
    ----------
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    gtab : Obj
        DiPy object storing diffusion gradient information.
    dwi_data : array
        4D array of dwi data.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    mod : obj
        Connectivity reconstruction model.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.dmri.estimation import csa_mod_est, csd_mod_est
    if conn_model == 'csa':
        mod = csa_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == 'csd':
        mod = csd_mod_est(gtab, dwi_data, B0_mask)
    else:
        raise ValueError('Error: Either no seeds supplied, or no valid seeds found in white-matter interface')

    del dwi_data

    return mod


def prep_tissues(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, cmc_step_size=0.2):
    """
    Estimate a tissue classifier for tractography.

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
    cmc_step_size : float
        Step size from CMC tissue classification method.

    Returns
    -------
    tiss_classifier : obj
        Tissue classifier object.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.tracking.stopping_criterion import ActStoppingCriterion, CmcStoppingCriterion, BinaryStoppingCriterion
    # Loads mask and ensures it's a true binary mask
    mask_img = nib.load(B0_mask)
    # Load tissue maps and prepare tissue classifier
    gm_mask_data = nib.load(gm_in_dwi).get_fdata()
    wm_mask_data = nib.load(wm_in_dwi).get_fdata()
    vent_csf_in_dwi_data = nib.load(vent_csf_in_dwi).get_fdata()
    if tiss_class == 'act':
        background = np.ones(mask_img.shape)
        background[(gm_mask_data + wm_mask_data + vent_csf_in_dwi_data) > 0] = 0
        include_map = gm_mask_data
        include_map[background > 0] = 1
        tiss_classifier = ActStoppingCriterion(include_map, vent_csf_in_dwi_data)
        del background
        del include_map
    elif tiss_class == 'bin':
        tiss_classifier = BinaryStoppingCriterion(wm_mask_data.astype('bool'))
    elif tiss_class == 'cmc':
        voxel_size = np.average(mask_img.header['pixdim'][1:4])
        tiss_classifier = CmcStoppingCriterion.from_pve(wm_mask_data, gm_mask_data, vent_csf_in_dwi_data,
                                                        step_size=cmc_step_size, average_voxel_size=voxel_size)
    elif tiss_class == 'wb':
        tiss_classifier = BinaryStoppingCriterion(mask_img.get_fdata().astype('bool'))
    else:
        raise ValueError('Tissue Classifier cannot be none.')

    del gm_mask_data, wm_mask_data, vent_csf_in_dwi_data
    mask_img.uncache()

    return tiss_classifier


def create_density_map(dwi_img, dir_path, streamlines, conn_model, target_samples,
                       node_size, curv_thr_list, step_list, network, roi):
    """
    Create a density map of the list of streamlines.

    Parameters
    ----------
    dwi_img : Nifti1Image
        Dwi data stored as a Nifti1image object.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.

    Returns
    -------
    streams : str
        File path to saved streamline array sequence in DTK-compatible trackvis (.trk) format.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    dm_path : str
        File path to fiber density map Nifti1Image.
    """
    import os
    import os.path as op
    from dipy.tracking import utils
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram

    # Create density map
    dm = utils.density_map(streamlines, affine=np.eye(4), vol_dims=dwi_img.shape)

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype('int16'), dwi_img.affine)

    namer_dir = '{}/tractography'.format(dir_path)
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    dm_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/density_map_',
                                                '%s' % (network + '_' if network is not None else ''),
                                                '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None
                                                        else ''),
                                                conn_model, '_', target_samples, '_',
                                                '%s' % ("%s%s" % (node_size, 'mm_') if ((node_size != 'parc') and
                                                                                        (node_size is not None))
                                                        else 'parc_'),
                                                'curv', str(curv_thr_list).replace(', ', '_'),
                                                '_step', str(step_list).replace(', ', '_'), '.nii.gz')
    dm_img.to_filename(dm_path)

    # Save streamlines to trk
    streams = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/streamlines_',
                                                '%s' % (network + '_' if network is not None else ''),
                                                '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None
                                                        else ''),
                                                conn_model, '_', target_samples, '_',
                                                '%s' % ("%s%s" % (node_size, 'mm_') if ((node_size != 'parc') and
                                                                                        (node_size is not None))
                                                        else 'parc_'),
                                                'curv', str(curv_thr_list).replace(', ', '_'),
                                                '_step', str(step_list).replace(', ', '_'), '.trk')

    save_tractogram(StatefulTractogram(streamlines, reference=dwi_img, space=Space.RASMM, shifted_origin=True),
                    streams, bbox_valid_check=False)

    del streamlines
    dm_img.uncache()

    return streams, dir_path, dm_path


def track_ensemble(dwi_data, target_samples, atlas_data_wm_gm_int, parcels, mod_fit, tiss_classifier, sphere, directget,
                   curv_thr_list, step_list, track_type, maxcrossing, max_length, roi_neighborhood_tol, min_length,
                   waymask, n_seeds_per_iter=100, pft_back_tracking_dist=2, pft_front_tracking_dist=1,
                   particle_count=15):
    """
    Perform native-space ensemble tractography, restricted to a vector of ROI masks.

    dwi_data : array
        4D array of dwi data.
    target_samples : int
        Total number of streamline samples specified to generate streams.
    atlas_data_wm_gm_int : array
        3D int32 numpy array of atlas parcellation intensities from Nifti1Image in T1w-warped native diffusion space,
        restricted to wm-gm interface.
    parcels : list
        List of 3D boolean numpy arrays of atlas parcellation ROI masks from a Nifti1Image in T1w-warped native
        diffusion space.
    mod : obj
        Connectivity reconstruction model.
    tiss_classifier : str
        Tissue classification method.
    sphere : obj
        DiPy object for modeling diffusion directions on a sphere.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    maxcrossing : int
        Maximum number if diffusion directions that can be assumed per voxel while tracking.
    max_length : int
        Maximum fiber length threshold in mm to restrict tracking.
    roi_neighborhood_tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    min_length : int
        Minimum fiber length threshold in mm.
    waymask : str
        Path to a Nifti1Image in native diffusion space to constrain tractography.
    n_seeds_per_iter : int
        Number of seeds from which to initiate tracking for each unique ensemble combination.
        By default this is set to 200.
    particle_count
        pft_back_tracking_dist : float
        Distance in mm to back track before starting the particle filtering
        tractography. The total particle filtering tractography distance is
        equal to back_tracking_dist + front_tracking_dist. By default this is set to 2 mm.
    pft_front_tracking_dist : float
        Distance in mm to run the particle filtering tractography after the
        the back track distance. The total particle filtering tractography
        distance is equal to back_tracking_dist + front_tracking_dist. By
        default this is set to 1 mm.
    particle_count : int
        Number of particles to use in the particle filter.

    Returns
    -------
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.
    """
    import gc
    from colorama import Fore, Style
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines, select_by_rois
    from dipy.tracking.local_tracking import LocalTracking, ParticleFilteringTracking
    from dipy.direction import ProbabilisticDirectionGetter, BootDirectionGetter, ClosestPeakDirectionGetter, DeterministicMaximumDirectionGetter

    if waymask:
        waymask_data = nib.load(waymask).get_fdata().astype('bool')

    # Commence Ensemble Tractography
    parcel_vec = list(np.ones(len(parcels)).astype('bool'))
    streamlines = nib.streamlines.array_sequence.ArraySequence()
    ix = 0
    circuit_ix = 0
    stream_counter = 0
    while int(stream_counter) < int(target_samples):
        for curv_thr in curv_thr_list:
            print("%s%s" % ('Curvature: ', curv_thr))

            # Instantiate DirectionGetter
            if directget == 'prob':
                dg = ProbabilisticDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr), sphere=sphere)
            elif directget == 'boot':
                dg = BootDirectionGetter.from_data(dwi_data, mod_fit, max_angle=float(curv_thr), sphere=sphere)
            elif directget == 'clos':
                dg = ClosestPeakDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr), sphere=sphere)
            elif directget == 'det':
                dg = DeterministicMaximumDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr), sphere=sphere)
            else:
                raise ValueError('ERROR: No valid direction getter(s) specified.')

            for step in step_list:
                print("%s%s" % ('Step: ', step))

                # Perform wm-gm interface seeding, using n_seeds at a time
                seeds = utils.random_seeds_from_mask(atlas_data_wm_gm_int > 0, seeds_count=n_seeds_per_iter,
                                                     seed_count_per_voxel=False, affine=np.eye(4))
                if len(seeds) == 0:
                    raise RuntimeWarning('Warning: No valid seed points found in wm-gm interface...')

                print(seeds)

                # Perform tracking
                if track_type == 'local':
                    streamline_generator = LocalTracking(dg, tiss_classifier, seeds, np.eye(4),
                                                         max_cross=int(maxcrossing), maxlen=int(max_length),
                                                         step_size=float(step), return_all=True)
                elif track_type == 'particle':
                    streamline_generator = ParticleFilteringTracking(dg, tiss_classifier, seeds, np.eye(4),
                                                                     max_cross=int(maxcrossing),
                                                                     step_size=float(step),
                                                                     maxlen=int(max_length),
                                                                     pft_back_tracking_dist=pft_back_tracking_dist,
                                                                     pft_front_tracking_dist=pft_front_tracking_dist,
                                                                     particle_count=particle_count, return_all=True)
                else:
                    raise ValueError('ERROR: No valid tracking method(s) specified.')

                # Filter resulting streamlines by roi-intersection characteristics
                roi_proximal_streamlines = Streamlines(select_by_rois(streamline_generator, affine=np.eye(4),
                                                                      rois=parcels, include=parcel_vec,
                                                                      mode='any',
                                                                      tol=roi_neighborhood_tol))

                print("%s%s" % ('Qualifying Streamlines by node intersection: ', len(roi_proximal_streamlines)))

                roi_proximal_streamlines = nib.streamlines.array_sequence.ArraySequence([s for s in
                                                                                         roi_proximal_streamlines if
                                                                                         len(s) > float(min_length)])

                print("%s%s" % ('Qualifying Streamlines by minimum length criterion: ', len(roi_proximal_streamlines)))

                if waymask:
                    roi_proximal_streamlines = roi_proximal_streamlines[utils.near_roi(roi_proximal_streamlines,
                                                                                       np.eye(4),
                                                                                       waymask_data,
                                                                                       tol=roi_neighborhood_tol,
                                                                                       mode='any')]
                    print("%s%s" % ('Qualifying Streamlines by waymask proximity: ', len(roi_proximal_streamlines)))

                # Repeat process until target samples condition is met
                ix = ix + 1
                for s in roi_proximal_streamlines:
                    stream_counter = stream_counter + len(s)
                    streamlines.append(s)
                    if int(stream_counter) >= int(target_samples):
                        break
                    else:
                        continue

                # Cleanup memory
                del seeds, roi_proximal_streamlines, streamline_generator
                gc.collect()

            del dg

        circuit_ix = circuit_ix + 1
        print("%s%s%s%s%s" % ('Completed hyperparameter circuit: ', circuit_ix, '...\nCumulative Streamline Count: ',
                              Fore.CYAN, stream_counter))
        print(Style.RESET_ALL)

    print('\n')

    return streamlines


def run_track(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, labels_im_file_wm_gm_int,
              labels_im_file, target_samples, curv_thr_list, step_list, track_type, max_length, maxcrossing, directget,
              conn_model, gtab_file, dwi_file, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc,
              prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, min_length,
              fa_path, waymask, roi_neighborhood_tol=8):
    """
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
    min_length : int
        Minimum fiber length threshold in mm.
    fa_path : str
        File path to FA Nifti1Image.
    waymask : str
        Path to a Nifti1Image in native diffusion space to constrain tractography.
    roi_neighborhood_tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel. Default is 10 mm.

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
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.io import load_pickle
    from colorama import Fore, Style
    from dipy.data import get_sphere
    from pynets.core import utils
    from pynets.dmri.track import prep_tissues, reconstruction, create_density_map, track_ensemble

    # Load diffusion data
    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    # Fit diffusion model
    mod_fit = reconstruction(conn_model, load_pickle(gtab_file), dwi_data, B0_mask)

    # Load atlas parcellation (and its wm-gm interface reduced version for seeding)
    atlas_data = nib.load(labels_im_file).get_fdata().astype('uint8')
    atlas_data_wm_gm_int = nib.load(labels_im_file_wm_gm_int).get_fdata().astype('uint8')

    # Build mask vector from atlas for later roi filtering
    parcels = []
    i = 0
    for roi_val in np.unique(atlas_data)[1:]:
        parcels.append(atlas_data == roi_val)
        i = i + 1

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
        print("%s%s%s%s" % (Fore.GREEN, 'Direction-getting type: ', Fore.BLUE, 'Probabilistic'))
    elif directget == 'boot':
        print("%s%s%s%s" % (Fore.GREEN, 'Direction-getting type: ', Fore.BLUE, 'Bootstrapped'))
    elif directget == 'closest':
        print("%s%s%s%s" % (Fore.GREEN, 'Direction-getting type: ', Fore.BLUE, 'Closest Peak'))
    elif directget == 'det':
        print("%s%s%s%s" % (Fore.GREEN, 'Direction-getting type: ', Fore.BLUE, 'Deterministic Maximum'))
    print(Style.RESET_ALL)

    # Commence Ensemble Tractography
    streamlines = track_ensemble(dwi_data, target_samples, atlas_data_wm_gm_int, parcels, mod_fit,
                                 prep_tissues(B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class),
                                 get_sphere('repulsion724'), directget, curv_thr_list, step_list, track_type,
                                 maxcrossing, max_length, roi_neighborhood_tol, min_length, waymask)
    print('Tracking Complete')

    # Create streamline density map
    [streams, dir_path, dm_path] = create_density_map(dwi_img, utils.do_dir_path(atlas, dwi_file), streamlines,
                                                      conn_model, target_samples, node_size, curv_thr_list, step_list,
                                                      network, roi)

    del streamlines, dwi_data, atlas_data_wm_gm_int, atlas_data, mod_fit, parcels
    dwi_img.uncache()

    return streams, track_type, target_samples, conn_model, dir_path, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, curv_thr_list, step_list, fa_path, dm_path, directget, labels_im_file, roi_neighborhood_tol
