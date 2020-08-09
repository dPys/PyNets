#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import nibabel as nib
import indexed_gzip

warnings.filterwarnings("ignore")


def reconstruction(conn_model, gtab, dwi_data, B0_mask):
    """
    Estimate a tensor model from dwi data.

    Parameters
    ----------
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd',
        'sfm').
    gtab : Obj
        DiPy object storing diffusion gradient information.
    dwi_data : array
        4D array of dwi data.
    B0_mask : str
        File path to B0 brain mask.

    Returns
    -------
    mod_fit : ndarray
        Fitted connectivity reconstruction model.
    mod : obj
        Connectivity reconstruction model.

    References
    ----------
    .. [1] Soares, J. M., Marques, P., Alves, V., & Sousa, N. (2013).
      A hitchhiker’s guide to diffusion tensor imaging.
      Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2013.00031

    """
    from pynets.dmri.estimation import (
        csa_mod_est,
        csd_mod_est,
        sfm_mod_est,
        tens_mod_est,
    )

    if conn_model == "csa" or conn_model == "CSA":
        [mod_fit, mod] = csa_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == "csd" or conn_model == "CSD":
        [mod_fit, mod] = csd_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == "sfm" or conn_model == "SFM":
        [mod_fit, mod] = sfm_mod_est(gtab, dwi_data, B0_mask)
    elif conn_model == "ten" or conn_model == "tensor" or \
        conn_model == "TEN":
        [mod_fit, mod] = tens_mod_est(gtab, dwi_data, B0_mask)
    else:
        raise ValueError(
            "Error: No valid reconstruction model specified. See the `-mod` "
            "flag."
        )

    del dwi_data

    return mod_fit, mod


def prep_tissues(
    t1_mask,
    gm_in_dwi,
    vent_csf_in_dwi,
    wm_in_dwi,
    tiss_class,
    B0_mask,
    cmc_step_size=0.2):
    """
    Estimate a tissue classifier for tractography.

    Parameters
    ----------
    t1_mask : str
        File path to a T1w mask.
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

    References
    ----------
    .. [1] Zhang, Y., Brady, M. and Smith, S. Segmentation of Brain MR Images
      Through a Hidden Markov Random Field Model and the
      Expectation-Maximization Algorithm IEEE Transactions on Medical Imaging,
      20(1): 45-56, 2001
    .. [2] Avants, B. B., Tustison, N. J., Wu, J., Cook, P. A. and Gee, J. C.
      An open source multivariate framework for n-tissue segmentation with
      evaluation on public data. Neuroinformatics, 9(4): 381-400, 2011.

    """
    from dipy.tracking.stopping_criterion import (
        ActStoppingCriterion,
        CmcStoppingCriterion,
        BinaryStoppingCriterion,
    )
    from nilearn.masking import intersect_masks
    from nilearn.image import math_img

    # Load B0 mask
    B0_mask_img = math_img("img > 0.0", img=nib.load(B0_mask, mmap=True))

    # Load t1 mask
    mask_img = math_img("img > 0.0", img=nib.load(t1_mask, mmap=True))

    # Load tissue maps and prepare tissue classifier
    wm_img = nib.load(wm_in_dwi, mmap=True)
    wm_mask_img = math_img("img > 0.0", img=wm_img)
    gm_img = nib.load(gm_in_dwi, mmap=True)
    gm_data = np.asarray(gm_img.dataobj, dtype=np.float32)
    wm_data = np.asarray(wm_img.dataobj, dtype=np.float32)
    vent_csf_in_dwi_data = np.asarray(nib.load(vent_csf_in_dwi,
                                               mmap=True).dataobj,
                                      dtype=np.float32)
    if tiss_class == "act":
        background = np.ones(mask_img.shape)
        background[(gm_data + wm_data +
                    vent_csf_in_dwi_data) > 0] = 0
        gm_data[background > 0] = 1
        tiss_classifier = ActStoppingCriterion(
            gm_data, vent_csf_in_dwi_data)
        del background
    elif tiss_class == "wm":
        tiss_classifier = BinaryStoppingCriterion(
            np.asarray(
                intersect_masks(
                    [
                        mask_img,
                        wm_mask_img,
                        B0_mask_img,
                    ],
                    threshold=1,
                    connected=False,
                ).dataobj
            )
        )
    elif tiss_class == "cmc":
        voxel_size = np.average(mask_img.header["pixdim"][1:4])
        tiss_classifier = CmcStoppingCriterion.from_pve(
            wm_data,
            gm_data,
            vent_csf_in_dwi_data,
            step_size=cmc_step_size,
            average_voxel_size=voxel_size,
        )
    elif tiss_class == "wb":
        tiss_classifier = BinaryStoppingCriterion(
            np.asarray(
                intersect_masks(
                    [
                        mask_img,
                        B0_mask_img,
                        nib.Nifti1Image(np.invert(
                            vent_csf_in_dwi_data.astype('bool')).astype(
                            'int'),
                                        affine=mask_img.affine),
                    ],
                    threshold=1,
                    connected=False,
                ).dataobj
            )
        )
    else:
        raise ValueError("Tissue classifier cannot be none.")

    del gm_data, wm_data, vent_csf_in_dwi_data
    mask_img.uncache()
    gm_img.uncache()
    wm_img.uncache()
    B0_mask_img.uncache()

    return tiss_classifier


def create_density_map(
    dwi_img,
    dir_path,
    streamlines,
    conn_model,
    target_samples,
    node_size,
    curv_thr_list,
    step_list,
    network,
    roi,
    directget,
    min_length,
    namer_dir,
):
    """
    Create a density map of the list of streamlines.

    Parameters
    ----------
    dwi_img : Nifti1Image
        Dwi data stored as a Nifti1image object.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    node_size : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    Returns
    -------
    streams : str
        File path to saved streamline array sequence in DTK-compatible
        trackvis (.trk) format.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    dm_path : str
        File path to fiber density map Nifti1Image.
    """
    import os.path as op
    from dipy.tracking import utils

    # Create density map
    dm = utils.density_map(
        streamlines,
        affine=np.eye(4),
        vol_dims=dwi_img.shape)

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype("float32"), dwi_img.affine)

    dm_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
        namer_dir,
        "/density_map_",
        "%s" % (network + "_" if network is not None else ""),
        "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None else
                ""),
        conn_model,
        "_",
        target_samples,
        "_",
        "%s"
        % (
            "%s%s" % (node_size, "mm_")
            if ((node_size != "parc") and (node_size is not None))
            else "parc_"
        ),
        "curv-",
        str(curv_thr_list).replace(", ", "_"),
        "_step-",
        str(step_list).replace(", ", "_"),
        "_directget-",
        directget,
        "_minlength-",
        min_length,
        ".nii.gz",
    )
    dm_img.to_filename(dm_path)

    del streamlines
    dm_img.uncache()

    return dir_path, dm_path


def track_ensemble(
    target_samples,
    atlas_data_wm_gm_int,
    parcels,
    mod_fit,
    sphere,
    directget,
    curv_thr_list,
    step_list,
    track_type,
    maxcrossing,
    roi_neighborhood_tol,
    min_length,
    waymask_data,
    B0_mask_data,
    t1w2dwi,
    gm_in_dwi,
    vent_csf_in_dwi,
    wm_in_dwi,
    tiss_class,
    B0_mask
):
    """
    Perform native-space ensemble tractography, restricted to a vector of ROI
    masks.

    target_samples : int
        Total number of streamline samples specified to generate streams.
    atlas_data_wm_gm_int : array
        3D int32 numpy array of atlas parcellation intensities from Nifti1Image
        in T1w-warped native diffusion space, restricted to wm-gm interface.
    parcels : list
        List of 3D boolean numpy arrays of atlas parcellation ROI masks from a
        Nifti1Image in T1w-warped native diffusion space.
    mod : obj
        Connectivity reconstruction model.
    tiss_classifier : str
        Tissue classification method.
    sphere : obj
        DiPy object for modeling diffusion directions on a sphere.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), and prob (probabilistic).
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    maxcrossing : int
        Maximum number if diffusion directions that can be assumed per voxel
        while tracking.
    roi_neighborhood_tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    min_length : int
        Minimum fiber length threshold in mm.
    waymask_data : ndarray
        Tractography constraint mask array in native diffusion space.
    B0_mask_data : ndarray
        B0 brain mask data.
    n_seeds_per_iter : int
        Number of seeds from which to initiate tracking for each unique
        ensemble combination. By default this is set to 250.
    max_length : int
        Maximum number of steps to restrict tracking.
    particle_count
        pft_back_tracking_dist : float
        Distance in mm to back track before starting the particle filtering
        tractography. The total particle filtering tractography distance is
        equal to back_tracking_dist + front_tracking_dist. By default this is
        set to 2 mm.
    pft_front_tracking_dist : float
        Distance in mm to run the particle filtering tractography after the
        the back track distance. The total particle filtering tractography
        distance is equal to back_tracking_dist + front_tracking_dist. By
        default this is set to 1 mm.
    particle_count : int
        Number of particles to use in the particle filter.
    min_separation_angle : float
        The minimum angle between directions [0, 90].

    Returns
    -------
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.

    References
    ----------
    .. [1] Takemura, H., Caiafa, C. F., Wandell, B. A., & Pestilli, F. (2016).
      Ensemble Tractography. PLoS Computational Biology.
      https://doi.org/10.1371/journal.pcbi.1004692

    """
    import gc
    import time
    import pkg_resources
    import yaml
    from joblib import Parallel, delayed, Memory
    import itertools
    from pynets.dmri.track import run_tracking
    from colorama import Fore, Style
    import tempfile
    from pynets.dmri.dmri_utils import generate_sl
    from nibabel.streamlines.array_sequence import concatenate, ArraySequence
    cache_dir = tempfile.mkdtemp()
    memory = Memory(cache_dir, verbose=0)

    with open(
        pkg_resources.resource_filename("pynets", "runconfig.yaml"), "r"
    ) as stream:
        hardcoded_params = yaml.load(stream)
        nthreads = hardcoded_params["nthreads"][0]
        n_seeds_per_iter = hardcoded_params['tracking']["n_seeds_per_iter"][0]
        max_length = hardcoded_params['tracking']["max_length"][0]
        pft_back_tracking_dist = hardcoded_params['tracking']["pft_back_tracking_dist"][0]
        pft_front_tracking_dist = hardcoded_params['tracking']["pft_front_tracking_dist"][0]
        particle_count = hardcoded_params['tracking']["particle_count"][0]
        min_separation_angle = hardcoded_params['tracking']["min_separation_angle"][0]
    stream.close()

    parcel_vec = list(np.ones(len(parcels)).astype("bool"))

    all_combs = list(itertools.product(step_list, curv_thr_list))

    atlas_data_wm_gm_int = memory.cache(atlas_data_wm_gm_int)
    parcels = memory.cache(parcels)

    # Commence Ensemble Tractography
    start = time.time()
    stream_counter = 0

    all_streams = []
    ix = 0
    while float(stream_counter) < float(target_samples) and float(ix) < \
        len(all_combs):
        with Parallel(n_jobs=nthreads, backend='loky',
                      mmap_mode='r+', temp_folder=cache_dir,
                      verbose=10) as parallel:
            out_streams = parallel(
                delayed(run_tracking)(
                    i, atlas_data_wm_gm_int, mod_fit, n_seeds_per_iter,
                    directget, maxcrossing, max_length, pft_back_tracking_dist,
                    pft_front_tracking_dist, particle_count, B0_mask_data,
                    roi_neighborhood_tol, parcels, parcel_vec, waymask_data,
                    min_length, track_type, min_separation_angle, sphere,
                    t1w2dwi, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class,
                    B0_mask) for i in all_combs)

            out_streams = [i for i in out_streams if i is not None and
                           len(i) > 0]

            if len(out_streams) < 10:
                ix += 1
                continue
            else:
                ix = 0
                out_streams = concatenate(out_streams, axis=0)

            # Append streamline generators to prevent exponential growth
            # in memory consumption
            all_streams.extend([generate_sl(i) for i in out_streams])
            stream_counter += len(out_streams)
            del out_streams

            print(
                "%s%s%s%s"
                % (
                    "\nCumulative Streamline Count: ",
                    Fore.CYAN,
                    stream_counter,
                    "\n",
                )
            )
            gc.collect()
            print(Style.RESET_ALL)

    if ix >= len(all_combs):
        raise ValueError(f"Tractography failed. "
                         f">{len(all_combs)} consecutive sampling iterations "
                         f"with <10 streamlines. Are you using a waymask? "
                         f"If so, it may be too restrictive.")
    else:
        print("Tracking Complete: ", str(time.time() - start))

    del parallel, parcel_vec, atlas_data_wm_gm_int, parcels
    memory.clear(warn=False)

    if stream_counter != 0:
        print('Generating final ArraySequence...')
        return ArraySequence([ArraySequence(i) for i in all_streams])
    else:
        raise ValueError('No streamlines generated!')


def run_tracking(step_curv_combinations, atlas_data_wm_gm_int, mod_fit,
                 n_seeds_per_iter, directget, maxcrossing, max_length,
                 pft_back_tracking_dist, pft_front_tracking_dist,
                 particle_count, B0_mask_data, roi_neighborhood_tol,
                 parcels, parcel_vec, waymask_data, min_length, track_type,
                 min_separation_angle, sphere, t1w2dwi, gm_in_dwi,
                 vent_csf_in_dwi, wm_in_dwi, tiss_class, B0_mask):

    import gc
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines, select_by_rois
    from dipy.tracking.local_tracking import LocalTracking, \
        ParticleFilteringTracking
    from dipy.direction import (
        ProbabilisticDirectionGetter,
        ClosestPeakDirectionGetter,
        DeterministicMaximumDirectionGetter
    )
    from pynets.dmri.track import prep_tissues
    from nibabel.streamlines.array_sequence import ArraySequence

    tiss_classifier = prep_tissues(
        t1w2dwi,
        gm_in_dwi,
        vent_csf_in_dwi,
        wm_in_dwi,
        tiss_class,
        B0_mask)

    print("%s%s" % ("Curvature: ", step_curv_combinations[1]))

    # Instantiate DirectionGetter
    if directget == "prob" or directget == "probabilistic":
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            mod_fit,
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    elif directget == "clos" or directget == "closest":
        dg = ClosestPeakDirectionGetter.from_shcoeff(
            mod_fit,
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    elif directget == "det" or directget == "deterministic":
        maxcrossing = 1
        dg = DeterministicMaximumDirectionGetter.from_shcoeff(
            mod_fit,
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    else:
        raise ValueError(
            "ERROR: No valid direction getter(s) specified."
        )
    print("%s%s" % ("Step: ", step_curv_combinations[0]))

    # Perform wm-gm interface seeding, using n_seeds at a time
    seeds = utils.random_seeds_from_mask(
        atlas_data_wm_gm_int.func > 0,
        seeds_count=n_seeds_per_iter,
        seed_count_per_voxel=False,
        affine=np.eye(4),
    )
    if len(seeds) == 0:
        print(UserWarning(
            "No valid seed points found in wm-gm "
            "interface..."
        ))
        return None

    # print(seeds)

    # Perform tracking
    if track_type == "local":
        streamline_generator = LocalTracking(
            dg,
            tiss_classifier,
            seeds,
            np.eye(4),
            max_cross=int(maxcrossing),
            maxlen=int(max_length),
            step_size=float(step_curv_combinations[0]),
            fixedstep=False,
            return_all=True,
        )
    elif track_type == "particle":
        streamline_generator = ParticleFilteringTracking(
            dg,
            tiss_classifier,
            seeds,
            np.eye(4),
            max_cross=int(maxcrossing),
            step_size=float(step_curv_combinations[0]),
            maxlen=int(max_length),
            pft_back_tracking_dist=pft_back_tracking_dist,
            pft_front_tracking_dist=pft_front_tracking_dist,
            particle_count=particle_count,
            return_all=True,
        )
    else:
        raise ValueError(
            "ERROR: No valid tracking method(s) specified.")

    # Filter resulting streamlines by those that stay entirely
    # inside the brain
    try:
        roi_proximal_streamlines = utils.target(
            streamline_generator, np.eye(4),
            B0_mask_data, include=True
        )
    except BaseException:
        print('No streamlines found inside the brain! '
              'Check registrations.')
        return None

    # Filter resulting streamlines by roi-intersection
    # characteristics
    try:
        roi_proximal_streamlines = Streamlines(
            select_by_rois(
                roi_proximal_streamlines,
                affine=np.eye(4),
                rois=parcels.func,
                include=parcel_vec,
                mode="%s" % ("any" if waymask_data is not None else
                             "both_end"),
                tol=roi_neighborhood_tol,
            )
        )
        print("%s%s" % ("Filtering by: \nnode intersection: ",
                        len(roi_proximal_streamlines)))
    except BaseException:
        print('No streamlines found to connect any parcels! '
              'Check registrations.')
        return None

    try:
        roi_proximal_streamlines = nib.streamlines. \
            array_sequence.ArraySequence(
            [
                s for s in roi_proximal_streamlines
                if len(s) >= float(min_length)
            ]
        )
        print(f"Minimum fiber length >{min_length}mm: "
              f"{len(roi_proximal_streamlines)}")
    except BaseException:
        print('No streamlines remaining after minimal length criterion.')
        return None

    if waymask_data is not None:
        try:
            roi_proximal_streamlines = roi_proximal_streamlines[
                utils.near_roi(
                    roi_proximal_streamlines,
                    np.eye(4),
                    waymask_data,
                    tol=roi_neighborhood_tol,
                    mode="any",
                )
            ]
            print("%s%s" % ("Waymask proximity: ",
                            len(roi_proximal_streamlines)))
        except BaseException:
            print('No streamlines remaining in waymask\'s vacinity.')
            return None

    out_streams = [s.astype("float32")
                   for s in roi_proximal_streamlines]

    del dg, seeds, roi_proximal_streamlines, streamline_generator
    gc.collect()

    try:
        return ArraySequence(out_streams)
    except BaseException:
        return None
