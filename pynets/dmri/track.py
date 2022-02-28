#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import os
import matplotlib
import warnings
import numpy as np
import nibabel as nib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


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
    t1_mask : Nifti1Image
        T1w mask img.
    gm_in_dwi : Nifti1Image
        Grey-matter tissue segmentation Nifti1Image.
    vent_csf_in_dwi : Nifti1Image
        Ventricular CSF tissue segmentation Nifti1Image.
    wm_in_dwi : Nifti1Image
        White-matter tissue segmentation Nifti1Image.
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
    import gc
    from dipy.tracking.stopping_criterion import (
        ActStoppingCriterion,
        CmcStoppingCriterion,
        BinaryStoppingCriterion,
    )
    from nilearn.masking import intersect_masks
    from nilearn.image import math_img

    # Load B0 mask
    B0_mask_img = math_img("img > 0.01", img=B0_mask)

    # Load t1 mask
    mask_img = math_img("img > 0.01", img=t1_mask)

    # Load tissue maps and prepare tissue classifier
    wm_mask_img = math_img("img > 0.01", img=wm_in_dwi)
    gm_mask_img = math_img("img > 0.01", img=gm_in_dwi)
    vent_csf_in_dwi_img = math_img("img > 0.01", img=vent_csf_in_dwi)
    gm_data = np.asarray(gm_mask_img.dataobj, dtype=np.float32)
    wm_data = np.asarray(wm_mask_img.dataobj, dtype=np.float32)
    vent_csf_in_dwi_data = np.asarray(vent_csf_in_dwi_img.dataobj,
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
                        nib.Nifti1Image(np.invert(
                            vent_csf_in_dwi_data.astype('bool')).astype(
                            'int'), affine=mask_img.affine)
                    ],
                    threshold=1,
                    connected=False,
                ).dataobj
            )
        )
    elif tiss_class == "cmc":
        tiss_classifier = CmcStoppingCriterion.from_pve(
            wm_data,
            gm_data,
            vent_csf_in_dwi_data,
            step_size=cmc_step_size,
            average_voxel_size=np.average(mask_img.header["pixdim"][1:4]),
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
                            'int'), affine=mask_img.affine),
                    ],
                    threshold=1,
                    connected=False,
                ).dataobj
            )
        )
    else:
        raise ValueError("Tissue classifier cannot be none.")

    B0_mask_img.uncache()
    mask_img.uncache()
    wm_mask_img.uncache()
    gm_mask_img.uncache()
    del gm_data, wm_data, vent_csf_in_dwi_data
    gc.collect()

    return tiss_classifier


def create_density_map(
    fa_img,
    dir_path,
    streamlines,
    conn_model,
    node_radius,
    curv_thr_list,
    step_list,
    subnet,
    roi,
    traversal,
    min_length,
    namer_dir,
):
    """
    Create a density map of the list of streamlines.

    Parameters
    ----------
    fa_img : Nifti1Image
        Dwi data stored as a Nifti1image object.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    traversal : str
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
    from dipy.tracking._utils import _mapping_to_voxel

    # Remove streamlines with negative voxel indices
    lin_T, offset = _mapping_to_voxel(np.eye(4))
    streams_filt = []
    for sl in streamlines:
        inds = np.dot(sl, lin_T)
        inds += offset
        if not inds.min().round(decimals=6) < 0:
            streams_filt.append(sl)

    # Create density map
    dm = utils.density_map(
        streams_filt,
        affine=np.eye(4),
        vol_dims=fa_img.shape)

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype("float32"), fa_img.affine)

    dm_path = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
        namer_dir,
        "/density_map_",
        "%s" % (subnet + "_" if subnet is not None else ""),
        "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None else
                ""),
        conn_model,
        "_",
        "%s"
        % (
            "%s%s" % (node_radius, "mm_")
            if ((node_radius != "parc") and (node_radius is not None))
            else "parc_"
        ),
        "curv-",
        str(curv_thr_list).replace(", ", "_"),
        "_step-",
        str(step_list).replace(", ", "_"),
        "_traversal-",
        traversal,
        "_minlength-",
        min_length,
        ".nii.gz",
    )
    dm_img.to_filename(dm_path)

    del streamlines, streams_filt
    dm_img.uncache()

    return dir_path, dm_path


def track_ensemble(
    target_samples,
    atlas_data_wm_gm_int,
    labels_im_file,
    recon_path,
    sphere,
    traversal,
    curv_thr_list,
    step_list,
    track_type,
    maxcrossing,
    roi_neighborhood_tol,
    min_length,
    waymask,
    B0_mask,
    t1w2dwi,
    gm_in_dwi,
    vent_csf_in_dwi,
    wm_in_dwi,
    tiss_class
):
    """
    Perform native-space ensemble tractography, restricted to a vector of ROI
    masks.

    Parameters
    ----------
    target_samples : int
        Total number of streamline samples specified to generate streams.
    atlas_data_wm_gm_int : str
        File path to Nifti1Image in T1w-warped native diffusion space,
        restricted to wm-gm interface.
    parcels : list
        List of 3D boolean numpy arrays of atlas parcellation ROI masks from a
        Nifti1Image in T1w-warped native diffusion space.
    recon_path : str
        File path to diffusion reconstruction model.
    tiss_classifier : str
        Tissue classification method.
    sphere : obj
        DiPy object for modeling diffusion directions on a sphere.
    traversal : str
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
    import os
    import gc
    import time
    import warnings
    import time
    import tempfile
    from joblib import Parallel, delayed, Memory
    import itertools
    from pynets.dmri.track import run_tracking
    from colorama import Fore, Style
    from pynets.dmri.utils import generate_sl
    from nibabel.streamlines.array_sequence import concatenate, ArraySequence
    from pynets.core.utils import save_3d_to_4d
    from nilearn.masking import intersect_masks
    from nilearn.image import math_img
    from pynets.core.utils import load_runconfig
    from dipy.tracking import utils

    warnings.filterwarnings("ignore")

    joblib_dir = tempfile.mkdtemp()
    os.makedirs(joblib_dir, exist_ok=True)

    hardcoded_params = load_runconfig()
    nthreads = hardcoded_params["omp_threads"][0]
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)
    n_seeds_per_iter = \
        hardcoded_params['tracking']["n_seeds_per_iter"][0]
    max_length = \
        hardcoded_params['tracking']["max_length"][0]
    pft_back_tracking_dist = \
        hardcoded_params['tracking']["pft_back_tracking_dist"][0]
    pft_front_tracking_dist = \
        hardcoded_params['tracking']["pft_front_tracking_dist"][0]
    particle_count = \
        hardcoded_params['tracking']["particle_count"][0]
    min_separation_angle = \
        hardcoded_params['tracking']["min_separation_angle"][0]
    min_streams = \
        hardcoded_params['tracking']["min_streams"][0]
    seeding_mask_thr = hardcoded_params['tracking']["seeding_mask_thr"][0]
    timeout = hardcoded_params['tracking']["track_timeout"][0]

    all_combs = list(itertools.product(step_list, curv_thr_list))

    # Construct seeding mask
    seeding_mask = f"{os.path.dirname(labels_im_file)}/seeding_mask.nii.gz"
    if waymask is not None and os.path.isfile(waymask):
        waymask_img = math_img(f"img > {seeding_mask_thr}",
                               img=nib.load(waymask))
        waymask_img.to_filename(waymask)
        atlas_data_wm_gm_int_img = intersect_masks(
            [
                waymask_img,
                math_img("img > 0.001", img=nib.load(atlas_data_wm_gm_int)),
                math_img("img > 0.001", img=nib.load(labels_im_file))
            ],
            threshold=1,
            connected=False,
        )
        nib.save(atlas_data_wm_gm_int_img, seeding_mask)
    else:
        atlas_data_wm_gm_int_img = intersect_masks(
            [
                math_img("img > 0.001", img=nib.load(atlas_data_wm_gm_int)),
                math_img("img > 0.001", img=nib.load(labels_im_file))
            ],
            threshold=1,
            connected=False,
        )
        nib.save(atlas_data_wm_gm_int_img, seeding_mask)

    tissues4d = save_3d_to_4d([B0_mask, labels_im_file, seeding_mask,
                               t1w2dwi, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi])

    # Commence Ensemble Tractography
    start = time.time()
    stream_counter = 0

    all_streams = []
    ix = 0

    memory = Memory(location=joblib_dir, mmap_mode='r+', verbose=0)
    os.chdir(f"{memory.location}/joblib")

    @memory.cache
    def load_recon_data(recon_path):
        import h5py
        with h5py.File(recon_path, 'r') as hf:
            recon_data = hf['reconstruction'][:].astype('float32')
        hf.close()
        return recon_data

    recon_shelved = load_recon_data.call_and_shelve(recon_path)

    @memory.cache
    def load_tissue_data(tissues4d):
        return nib.load(tissues4d)

    tissue_shelved = load_tissue_data.call_and_shelve(tissues4d)

    try:
        while float(stream_counter) < float(target_samples) and \
                float(ix) < 0.50*float(len(all_combs)):
            with Parallel(n_jobs=nthreads, backend='loky',
                          mmap_mode='r+', verbose=0) as parallel:

                out_streams = parallel(
                    delayed(run_tracking)(
                        i, recon_shelved, n_seeds_per_iter, traversal,
                        maxcrossing, max_length, pft_back_tracking_dist,
                        pft_front_tracking_dist, particle_count,
                        roi_neighborhood_tol, min_length,
                        track_type, min_separation_angle, sphere, tiss_class,
                        tissue_shelved) for i in all_combs)

                out_streams = list(filter(None, out_streams))

                if len(out_streams) > 1:
                    out_streams = concatenate(out_streams, axis=0)
                else:
                    continue

                if waymask is not None and os.path.isfile(waymask):
                    try:
                        out_streams = out_streams[
                            utils.near_roi(
                                out_streams,
                                np.eye(4),
                                np.asarray(nib.load(waymask
                                                    ).dataobj).astype(
                                    "bool"),
                                tol=int(round(roi_neighborhood_tol * 0.50, 1)),
                                mode="all"
                            )
                        ]
                    except BaseException:
                        print(f"\n{Fore.RED}No streamlines generated in "
                              f"waymask vacinity\n")
                        print(Style.RESET_ALL)
                        #return None

                if len(out_streams) < min_streams:
                    ix += 1
                    print(f"\n{Fore.YELLOW}Fewer than {min_streams} "
                          f"streamlines tracked "
                          f"on last iteration...\n")
                    print(Style.RESET_ALL)
                    if ix > 5:
                        print(f"\n{Fore.RED}No streamlines generated\n")
                        print(Style.RESET_ALL)
                        #return None
                    continue
                else:
                    ix -= 1

                stream_counter += len(out_streams)
                all_streams.extend([generate_sl(i) for i in out_streams])
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

                if time.time() - start > timeout:
                    print(
                        f"\n{Fore.RED}Warning: Tractography timed "
                        f"out: {time.time() - start}")
                    print(Style.RESET_ALL)
                    memory.clear(warn=False)
                    #return None

    except RuntimeError as e:
        print(f"\n{Fore.RED}Error: Tracking failed due to:\n{e}\n")
        print(Style.RESET_ALL)
        memory.clear(warn=False)
        #return None

    memory.clear(warn=False)

    print("Tracking Complete: ", str(time.time() - start))

    del parallel, all_combs
    gc.collect()

    if stream_counter != 0:
        print('Generating final ...')
        return ArraySequence([ArraySequence(i) for i in all_streams])
    else:
        print(f"\n{Fore.RED}No streamlines generated!")
        print(Style.RESET_ALL)
        #return None


def run_tracking(step_curv_combinations, recon_shelved,
                 n_seeds_per_iter, traversal, maxcrossing, max_length,
                 pft_back_tracking_dist, pft_front_tracking_dist,
                 particle_count, roi_neighborhood_tol, min_length,
                 track_type, min_separation_angle, sphere, tiss_class,
                 tissue_shelved, verbose=False):
    """
    Create a density map of the list of streamlines.

    Parameters
    ----------
    step_curv_combinations : list
        List of tuples representing all pair combinations of step sizes and
        curvature thresholds from which to sample streamlines.
    recon_path : str
        File path to diffusion reconstruction model.
    n_seeds_per_iter : int
        Number of seeds from which to initiate tracking for each unique
        ensemble combination. By default this is set to 250.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    maxcrossing : int
        Maximum number if diffusion directions that can be assumed per voxel
        while tracking.
    max_length : int
        Maximum number of steps to restrict tracking.
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
    roi_neighborhood_tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    waymask_data : ndarray
        Tractography constraint mask array in native diffusion space.
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    min_separation_angle : float
        The minimum angle between directions [0, 90].
    sphere : obj
        DiPy object for modeling diffusion directions on a sphere.
    tiss_class : str
        Tissue classification method.
    tissue_shelved : str
        File path to joblib-shelved 4D T1w tissue segmentations in native
        diffusion space.

    Returns
    -------
    streamlines : ArraySequence
        DiPy list/array-like object of streamline points from tractography.
    """
    import gc
    import time
    import numpy as np
    from dipy.tracking import utils
    from dipy.tracking.streamline import select_by_rois
    from dipy.tracking.local_tracking import LocalTracking, \
        ParticleFilteringTracking
    from dipy.direction import (
        ProbabilisticDirectionGetter,
        ClosestPeakDirectionGetter,
        DeterministicMaximumDirectionGetter
    )
    from nilearn.image import index_img
    from pynets.dmri.utils import generate_seeds, random_seeds_from_mask
    from nibabel.streamlines.array_sequence import ArraySequence
    from nilearn.image import math_img

    start_time = time.time()

    if verbose is True:
        print("%s%s%s" % (
        'Preparing tissue constraints:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()

    tissue_img = tissue_shelved.get()

    # Order:
    B0_mask = index_img(tissue_img, 0)
    atlas_img = index_img(tissue_img, 1)
    t1w2dwi = index_img(tissue_img, 3)
    gm_in_dwi = index_img(tissue_img, 4)
    vent_csf_in_dwi = index_img(tissue_img, 5)
    wm_in_dwi = index_img(tissue_img, 6)
    tissue_img.uncache()

    tiss_classifier = prep_tissues(
        t1w2dwi,
        gm_in_dwi,
        vent_csf_in_dwi,
        wm_in_dwi,
        tiss_class,
        B0_mask
    )

    # if verbose is True:
    #     print("%s%s%s" % (
    #     'Fitting tissue classifier:',
    #     np.round(time.time() - start_time, 1), 's'))
    #     start_time = time.time()

    if verbose is True:
        print("%s%s%s" % (
        'Loading reconstruction:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()

        print("%s%s" % ("Curvature: ", step_curv_combinations[1]))

    # Instantiate DirectionGetter
    if traversal.lower() in ["probabilistic", "prob"]:
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            recon_shelved.get(),
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    elif traversal.lower() in ["closestpeaks", "cp"]:
        dg = ClosestPeakDirectionGetter.from_shcoeff(
            recon_shelved.get(),
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    elif traversal.lower() in ["deterministic", "det"]:
        maxcrossing = 1
        dg = DeterministicMaximumDirectionGetter.from_shcoeff(
            recon_shelved.get(),
            max_angle=float(step_curv_combinations[1]),
            sphere=sphere,
            min_separation_angle=min_separation_angle,
        )
    else:
        raise ValueError(
            "ERROR: No valid direction getter(s) specified."
        )

    if verbose is True:
        print("%s%s%s" % (
            'Extracting directions:',
            np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()
        print("%s%s" % ("Step: ", step_curv_combinations[0]))

    # Perform wm-gm interface seeding, using n_seeds at a time
    seeds = generate_seeds(random_seeds_from_mask(
            np.asarray(
                math_img("img > 0.01", img=index_img(tissue_img, 2)).dataobj
            ).astype("bool").astype("int16") > 0,
        seeds_count=n_seeds_per_iter, random_seed=42
    ))

    if verbose is True:
        print("%s%s%s" % (
        'Drawing random seeds:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()
        # print(seeds)

    # Perform tracking
    if track_type == "local":
        streamline_generator = LocalTracking(
            dg,
            tiss_classifier,
            np.stack([i for i in seeds]),
            np.eye(4),
            max_cross=int(maxcrossing),
            maxlen=int(max_length),
            step_size=float(step_curv_combinations[0]),
            fixedstep=False,
            return_all=True,
            random_seed=42
        )
    elif track_type == "particle":
        streamline_generator = ParticleFilteringTracking(
            dg,
            tiss_classifier,
            np.stack([i for i in seeds]),
            np.eye(4),
            max_cross=int(maxcrossing),
            step_size=float(step_curv_combinations[0]),
            maxlen=int(max_length),
            pft_back_tracking_dist=pft_back_tracking_dist,
            pft_front_tracking_dist=pft_front_tracking_dist,
            pft_max_trial=20,
            particle_count=particle_count,
            return_all=True,
            random_seed=42
        )
    else:
        raise ValueError(
            "ERROR: No valid tracking method(s) specified.")

    if verbose is True:
        print("%s%s%s" % (
        'Instantiating tracking:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()
        # print(seeds)

    del dg

    # Filter resulting streamlines by those that stay entirely
    # inside the brain
    try:
        roi_proximal_streamlines = utils.target(
            streamline_generator, np.eye(4),
            np.asarray(B0_mask.dataobj).astype('bool'), include=True
        )
    except BaseException:
        print('No streamlines found inside the brain! '
              'Check registrations.')
        #return None

    if verbose is True:
        print("%s%s%s" % (
        'Drawing streamlines:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()

    del seeds, tiss_classifier, streamline_generator

    B0_mask.uncache()
    atlas_img.uncache()
    t1w2dwi.uncache()
    gm_in_dwi.uncache()
    vent_csf_in_dwi.uncache()
    wm_in_dwi.uncache()
    gc.collect()

    # Filter resulting streamlines by roi-intersection
    # characteristics
    atlas_data = np.array(atlas_img.dataobj).astype("uint16")

    # Build mask vector from atlas for later roi filtering
    parcels = [atlas_data == roi_val for roi_val in
               [i for i in np.unique(atlas_data) if i != 0]]

    try:
        roi_proximal_streamlines = \
                select_by_rois(
                    roi_proximal_streamlines,
                    affine=np.eye(4),
                    rois=parcels,
                    include=list(np.ones(len(parcels)).astype("bool")),
                    mode="any",
                    tol=roi_neighborhood_tol,
                )
    except BaseException:
        print('No streamlines found to connect any parcels! '
              'Check registrations.')
        #return None

    del atlas_data

    if verbose is True:
        print("%s%s%s" % (
        'Selecting by parcellation:',
        np.round(time.time() - start_time, 1), 's'))
        start_time = time.time()

    del parcels

    gc.collect()

    if verbose is True:
        print("%s%s%s" % (
        'Selecting by minimum length criterion:',
        np.round(time.time() - start_time, 1), 's'))

    gc.collect()

    return ArraySequence([s.astype("float32") for s in
                          roi_proximal_streamlines if
                          len(s) > float(min_length)])
