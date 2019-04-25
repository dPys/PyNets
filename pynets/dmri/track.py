# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib


def reconstruction(conn_model, gtab, dwi, wm_in_dwi):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.dmri.estimation import tens_mod_est, odf_mod_est, csd_mod_est
    dwi_img = nib.load(dwi)
    data = dwi_img.get_data()
    if conn_model == 'tensor':
        mod = tens_mod_est(gtab, data, wm_in_dwi)
    elif conn_model == 'csa':
        mod = odf_mod_est(gtab, data, wm_in_dwi)
    elif conn_model == 'csd':
        mod = csd_mod_est(gtab, data, wm_in_dwi)
    else:
        raise ValueError('Error: Either no seeds supplied, or no valid seeds found in white-matter interface')

    return mod


def prep_tissues(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, dir_path, cmc_step_size=0.2):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.tracking.local import ActTissueClassifier, CmcTissueClassifier, BinaryTissueClassifier
    # Loads mask and ensures it's a true binary mask
    mask_img = nib.load(nodif_B0_mask)
    # Load tissue maps and prepare tissue classifier
    gm_mask = nib.load(gm_in_dwi)
    gm_mask_data = gm_mask.get_data()
    wm_mask = nib.load(wm_in_dwi)
    wm_mask_data = wm_mask.get_data()
    if tiss_class == 'act':
        vent_csf_in_dwi = nib.load(vent_csf_in_dwi)
        vent_csf_in_dwi_data = vent_csf_in_dwi.get_data()
        background = np.ones(mask_img.shape)
        background[(gm_mask_data + wm_mask_data + vent_csf_in_dwi_data) > 0] = 0
        include_map = gm_mask_data
        include_map[background > 0] = 1
        exclude_map = vent_csf_in_dwi_data
        tiss_classifier = ActTissueClassifier(include_map, exclude_map)
    elif tiss_class == 'bin':
        wm_in_dwi_data = nib.load(wm_in_dwi).get_data().astype('bool')
        tiss_classifier = BinaryTissueClassifier(wm_in_dwi_data)
    elif tiss_class == 'cmc':
        vent_csf_in_dwi = nib.load(vent_csf_in_dwi)
        vent_csf_in_dwi_data = vent_csf_in_dwi.get_data()
        voxel_size = np.average(wm_mask.get_header()['pixdim'][1:4])
        tiss_classifier = CmcTissueClassifier.from_pve(wm_mask_data, gm_mask_data, vent_csf_in_dwi_data,
                                                       step_size=cmc_step_size, average_voxel_size=voxel_size)
    else:
        raise ValueError("%s%s%s" % ('Error: tissuer classification method: ', tiss_class, 'not found'))

    return tiss_classifier


def run_LIFE_all(data, gtab, streamlines):
    import dipy.tracking.life as life
    import dipy.core.optimize as opt
    fiber_model = life.FiberModel(gtab)
    fiber_fit = fiber_model.fit(data, streamlines, affine=np.eye(4))
    streamlines_filt = list(np.array(streamlines)[np.where(fiber_fit.beta > 0)[0]])
    beta_baseline = np.zeros(fiber_fit.beta.shape[0])
    pred_weighted = np.reshape(opt.spdot(fiber_fit.life_matrix, beta_baseline),
                               (fiber_fit.vox_coords.shape[0], np.sum(~gtab.b0s_mask)))
    mean_pred = np.empty((fiber_fit.vox_coords.shape[0], gtab.bvals.shape[0]))
    S0 = fiber_fit.b0_signal
    mean_pred[..., gtab.b0s_mask] = S0[:, None]
    mean_pred[..., ~gtab.b0s_mask] = (pred_weighted + fiber_fit.mean_signal[:, None]) * S0[:, None]
    mean_error = mean_pred - fiber_fit.data
    mean_rmse = np.sqrt(np.mean(mean_error ** 2, -1))
    return streamlines_filt, mean_rmse


def save_streams(dwi_img, streamlines, dir_path):
    hdr = dwi_img.header
    streams = "%s%s" % (dir_path, '/streamlines.trk')

    # Save streamlines
    trk_affine = np.eye(4)
    trk_hdr = nib.streamlines.trk.TrkFile.create_empty_header()
    trk_hdr['hdr_size'] = 1000
    trk_hdr['dimensions'] = hdr['dim'][1:4].astype('float32')
    trk_hdr['voxel_sizes'] = hdr['pixdim'][1:4]
    trk_hdr['voxel_to_rasmm'] = trk_affine
    trk_hdr['voxel_order'] = 'LPS'
    trk_hdr['pad2'] = 'LPS'
    trk_hdr['image_orientation_patient'] = np.array([1., 0., 0., 0., 1., 0.]).astype('float32')
    trk_hdr['endianness'] = '<'
    trk_hdr['_offset_data'] = 1000
    trk_hdr['nb_streamlines'] = len(streamlines)
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=trk_affine)
    trkfile = nib.streamlines.trk.TrkFile(tractogram, header=trk_hdr)
    nib.streamlines.save(trkfile, streams)
    return streams


def run_track(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, dir_path, labels_im_file,
              target_samples, curv_thr_list, step_list, track_type, max_length,
              maxcrossing, directget, conn_model, gtab, dwi):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
    from dipy.data import get_sphere
    from dipy.direction import ProbabilisticDirectionGetter, BootDirectionGetter, ClosestPeakDirectionGetter, DeterministicMaximumDirectionGetter
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines
    from pynets.dmri.track import prep_tissues, reconstruction

    # Set max repetitions before assuming no further streamlines can be generated from seeds
    repetitions = 1000

    mod_fit = reconstruction(conn_model, gtab, dwi, wm_in_dwi)

    # Load atlas parcellation
    atlas_img = nib.load(labels_im_file)
    atlas_data = atlas_img.get_data().astype('int')

    # Get sphere
    sphere = get_sphere('repulsion724')

    tiss_classifier = prep_tissues(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, dir_path)

    if np.sum(atlas_data) == 0:
        raise ValueError('ERROR: No non-zero voxels found in atlas. Check any roi masks and/or wm-gm interface images to verify overlap with dwi-registered atlas.')

    # Iteratively build a list of streamlines for each ROI while tracking
    print("%s%s" % ('Target number of samples per ROI: ', target_samples))
    print("%s%s" % ('Using curvature threshold(s): ', curv_thr_list))
    print("%s%s" % ('Using step size(s): ', step_list))
    streamlines_list = []
    for roi in np.unique(atlas_data)[1:]:
        print("%s%s" % ('ROI: ', roi))
        streamlines = nib.streamlines.array_sequence.ArraySequence()
        ix = 0
        while (len(streamlines) < float(target_samples)) and (ix < int(repetitions)):
            for curv_thr in curv_thr_list:

                # Create ProbabilisticDirectionGetter whose name is confusing because it is not getting directions.
                if directget == 'prob':
                    print('Using Probabilistic Direction...')
                    dg = ProbabilisticDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=float(curv_thr),
                                                                   sphere=sphere)
                elif directget == 'boot':
                    print('Using Bootstrapped Direction...')
                    dg = BootDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=float(curv_thr),
                                                          sphere=sphere)
                elif directget == 'closest':
                    print('Using Closest Peak Direction...')
                    dg = ClosestPeakDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=float(curv_thr),
                                                                 sphere=sphere)
                elif directget == 'det':
                    print('Using Deterministic Maximum Direction...')
                    dg = DeterministicMaximumDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=float(curv_thr),
                                                                          sphere=sphere)
                else:
                    raise ValueError('ERROR: No valid direction getter(s) specified.')
                for step in step_list:
                    seed = utils.random_seeds_from_mask(atlas_data==roi, seeds_count=1, seed_count_per_voxel=True,
                                                        affine=np.eye(4))
                    #print(seed)
                    if track_type == 'local':
                        print('Using Local Tracking...')
                        streamline_generator = LocalTracking(dg, tiss_classifier, seed, np.eye(4),
                                                             max_cross=int(maxcrossing), maxlen=int(max_length),
                                                             step_size=float(step), return_all=True)
                    elif track_type == 'particle':
                        print('Using Particle Tracking...')
                        streamline_generator = ParticleFilteringTracking(dg, tiss_classifier, seed, np.eye(4),
                                                                         max_cross=int(maxcrossing), step_size=float(step),
                                                                         maxlen=int(max_length),
                                                                         pft_back_tracking_dist=2,
                                                                         pft_front_tracking_dist=1,
                                                                         particle_count=15, return_all=True)
                    else:
                        raise ValueError('ERROR: No valid tracking method(s) specified.')
                    streamlines_more = Streamlines(streamline_generator)

                    ix = ix + 1
                    for s in streamlines_more:
                        streamlines.append(s)
                        if len(streamlines) > float(target_samples):
                            break
                        elif ix > int(repetitions):
                            break
                        else:
                            continue

            print("%s%s" % ('Streams: ', len(streamlines)))

        print('\n')
        streamlines_list.append(streamlines)

    print('Tracking complete...')

    return streamlines_list


def filter_streamlines(dwi, dir_path, gtab, streamlines_list, life_run, min_length):
    from dipy.tracking import utils
    from pynets.dmri.track import save_streams, run_LIFE_all

    dwi_img = nib.load(dwi)
    data = dwi_img.get_data()

    # Flatten streamlines list, and apply min length filter
    print('Filtering streamlines...')
    streamlines = nib.streamlines.array_sequence.ArraySequence([s for s in np.concatenate(streamlines_list).ravel() if len(s) > float(min_length)])

    # Fit LiFE model
    if life_run is True:
        print('Fitting LiFE...')
        # Fit Linear Fascicle Evaluation (LiFE)
        [streamlines, rmse] = run_LIFE_all(data, gtab, streamlines)
        print("%s%s" % ('Mean RMSE: ', np.mean(rmse)))

    # Create density map
    dm = utils.density_map(streamlines, dwi_img.shape, affine=np.eye(4))

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype("int16"), dwi_img.affine)
    dm_img.to_filename("%s%s" % (dir_path, "/density_map.nii.gz"))

    # Save streamlines to trk
    streams = save_streams(dwi_img, streamlines, dir_path)

    return streamlines, streams
