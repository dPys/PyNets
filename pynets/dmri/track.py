# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import nibabel as nib


def reconstruction(conn_model, gtab, dwi_file, wm_in_dwi):
    """

    :param conn_model:
    :param gtab:
    :param dwi_file:
    :param wm_in_dwi:
    :return:
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.dmri.estimation import tens_mod_est, csa_mod_est, csd_mod_est
    dwi_img = nib.load(dwi_file)
    data = dwi_img.get_fdata()
    if conn_model == 'tensor':
        mod = tens_mod_est(gtab, data, wm_in_dwi)
    elif conn_model == 'csa':
        mod = csa_mod_est(gtab, data, wm_in_dwi)
    elif conn_model == 'csd':
        mod = csd_mod_est(gtab, data, wm_in_dwi)
    else:
        raise ValueError('Error: Either no seeds supplied, or no valid seeds found in white-matter interface')

    return mod


def prep_tissues(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, cmc_step_size=0.2):
    """

    :param nodif_B0_mask:
    :param gm_in_dwi:
    :param vent_csf_in_dwi:
    :param wm_in_dwi:
    :param tiss_class:
    :param cmc_step_size:
    :return:
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.tracking.local import ActTissueClassifier, CmcTissueClassifier, BinaryTissueClassifier
    # Loads mask and ensures it's a true binary mask
    mask_img = nib.load(nodif_B0_mask)
    # Load tissue maps and prepare tissue classifier
    gm_mask = nib.load(gm_in_dwi)
    gm_mask_data = gm_mask.get_fdata()
    wm_mask = nib.load(wm_in_dwi)
    wm_mask_data = wm_mask.get_fdata()
    if tiss_class == 'act':
        vent_csf_in_dwi = nib.load(vent_csf_in_dwi)
        vent_csf_in_dwi_data = vent_csf_in_dwi.get_fdata()
        background = np.ones(mask_img.shape)
        background[(gm_mask_data + wm_mask_data + vent_csf_in_dwi_data) > 0] = 0
        include_map = gm_mask_data
        include_map[background > 0] = 1
        exclude_map = vent_csf_in_dwi_data
        tiss_classifier = ActTissueClassifier(include_map, exclude_map)
    elif tiss_class == 'bin':
        wm_in_dwi_data = nib.load(wm_in_dwi).get_fdata().astype('bool')
        tiss_classifier = BinaryTissueClassifier(wm_in_dwi_data)
    elif tiss_class == 'cmc':
        vent_csf_in_dwi = nib.load(vent_csf_in_dwi)
        vent_csf_in_dwi_data = vent_csf_in_dwi.get_fdata()
        voxel_size = np.average(wm_mask.get_header()['pixdim'][1:4])
        tiss_classifier = CmcTissueClassifier.from_pve(wm_mask_data, gm_mask_data, vent_csf_in_dwi_data,
                                                       step_size=cmc_step_size, average_voxel_size=voxel_size)
    else:
        raise ValueError("%s%s%s" % ('Error: tissuer classification method: ', tiss_class, 'not found'))

    return tiss_classifier


def run_LIFE_all(data, gtab, streamlines):
    """

    :param data:
    :param gtab:
    :param streamlines:
    :return:
    """
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


def save_streams(dwi_img, streamlines, streams):
    """

    :param dwi_img:
    :param streamlines:
    :param streams:
    :return:
    """
    hdr = dwi_img.header

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


def filter_streamlines(dwi_file, dir_path, gtab, streamlines, life_run, min_length, conn_model, target_samples,
                       node_size, curv_thr_list, step_list):
    """

    :param dwi_file:
    :param dir_path:
    :param gtab:
    :param streamlines:
    :param life_run:
    :param min_length:
    :param conn_model:
    :param target_samples:
    :param node_size:
    :param curv_thr_list:
    :param step_list:
    :return:
    """
    from dipy.tracking import utils
    from pynets.dmri.track import save_streams, run_LIFE_all

    dwi_img = nib.load(dwi_file)
    data = dwi_img.get_fdata()

    # Flatten streamlines list, and apply min length filter
    print('Filtering streamlines...')
    streamlines = nib.streamlines.array_sequence.ArraySequence([s for s in streamlines if len(s) > float(min_length)])

    # Fit LiFE model
    if life_run is True:
        print('Fitting LiFE...')
        # Fit Linear Fascicle Evaluation (LiFE)
        [streamlines, rmse] = run_LIFE_all(data, gtab, streamlines)
        mean_rmse = np.mean(rmse)
        print("%s%s" % ('Mean RMSE: ', mean_rmse))
        if mean_rmse > 5:
            print(
                'WARNING: LiFE revealed high model error. Check streamlines output and review tracking parameters used.')

    # Create density map
    dm = utils.density_map(streamlines, dwi_img.shape, affine=np.eye(4))

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype('int16'), dwi_img.affine)
    dm_img.to_filename("%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/density_map_', conn_model, '_', target_samples, '_',
                                                     node_size, 'mm_curv', str(curv_thr_list).replace(', ', '_'),
                                                     '_step', str(step_list).replace(', ', '_'), '.nii.gz'))

    # Save streamlines to trk
    streams = "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/streamlines_', conn_model, '_', target_samples, '_',
                                            node_size, 'mm_curv', str(curv_thr_list).replace(', ', '_'),
                                            '_step', str(step_list).replace(', ', '_'), '.trk')
    streams = save_streams(dwi_img, streamlines, streams)

    return streams, dir_path


def track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, parcel_vec, mod_fit,
                   tiss_classifier, sphere, directget, curv_thr_list, step_list, track_type, maxcrossing, max_length,
                   n_seeds_per_iter=200):
    """

    :param target_samples:
    :param atlas_data_wm_gm_int:
    :param parcels:
    :param parcel_vec:
    :param mod_fit:
    :param tiss_classifier:
    :param sphere:
    :param directget:
    :param curv_thr_list:
    :param step_list:
    :param track_type:
    :param maxcrossing:
    :param max_length:
    :param n_seeds_per_iter:
    :return:
    """
    from colorama import Fore, Style
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines, select_by_rois
    from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
    from dipy.direction import ProbabilisticDirectionGetter, BootDirectionGetter, ClosestPeakDirectionGetter, \
        DeterministicMaximumDirectionGetter

    # Commence Ensemble Tractography
    streamlines = nib.streamlines.array_sequence.ArraySequence()
    ix = 0
    circuit_ix = 0
    stream_counter = 0
    while int(stream_counter) < int(target_samples):
        for curv_thr in curv_thr_list:
            print("%s%s" % ('Curvature: ', curv_thr))

            # Instantiate DirectionGetter
            if directget == 'prob':
                dg = ProbabilisticDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr),
                                                               sphere=sphere)
            elif directget == 'boot':
                dg = BootDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr),
                                                      sphere=sphere)
            elif directget == 'closest':
                dg = ClosestPeakDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr),
                                                             sphere=sphere)
            elif directget == 'det':
                dg = DeterministicMaximumDirectionGetter.from_shcoeff(mod_fit, max_angle=float(curv_thr),
                                                                      sphere=sphere)
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
                                                                     pft_back_tracking_dist=2,
                                                                     pft_front_tracking_dist=1,
                                                                     particle_count=15, return_all=True)
                else:
                    raise ValueError('ERROR: No valid tracking method(s) specified.')

                # Filter resulting streamlines by roi-intersection characteristics
                streamlines_more = Streamlines(select_by_rois(streamline_generator, parcels, parcel_vec.astype('bool'),
                                                              mode='any', affine=np.eye(4), tol=8))

                # Repeat process until target samples condition is met
                ix = ix + 1
                for s in streamlines_more:
                    stream_counter = stream_counter + len(s)
                    streamlines.append(s)
                    if int(stream_counter) >= int(target_samples):
                        break
                    else:
                        continue

        circuit_ix = circuit_ix + 1
        print("%s%s%s%s%s" % ('Completed hyperparameter circuit: ', circuit_ix, '...\nCumulative Streamline Count: ',
                              Fore.CYAN, stream_counter))
        print(Style.RESET_ALL)

    print('\n')
    return streamlines


def run_track(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, labels_im_file_wm_gm_int,
              labels_im_file, target_samples, curv_thr_list, step_list, track_type, max_length, maxcrossing, directget,
              conn_model, gtab_file, dwi_file, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc,
              prune, atlas_select, uatlas_select, label_names, coords, norm, binary, atlas_mni, life_run, min_length,
              fa_path):
    """

    :param nodif_B0_mask:
    :param gm_in_dwi:
    :param vent_csf_in_dwi:
    :param wm_in_dwi:
    :param tiss_class:
    :param labels_im_file_wm_gm_int:
    :param labels_im_file:
    :param target_samples:
    :param curv_thr_list:
    :param step_list:
    :param track_type:
    :param max_length:
    :param maxcrossing:
    :param directget:
    :param conn_model:
    :param gtab_file:
    :param dwi_file:
    :param network:
    :param node_size:
    :param dens_thresh:
    :param ID:
    :param roi:
    :param min_span_tree:
    :param disp_filt:
    :param parc:
    :param prune:
    :param atlas_select:
    :param uatlas_select:
    :param label_names:
    :param coords:
    :param norm:
    :param binary:
    :param atlas_mni:
    :param life_run:
    :param min_length:
    :param fa_path:
    :return:
    """
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
    parcel_vec = np.ones(len(parcels))

    # Get sphere
    sphere = get_sphere('repulsion724')

    # Instantiate tissue classifier
    tiss_classifier = prep_tissues(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class)

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
    streamlines = track_ensemble(target_samples, atlas_data_wm_gm_int, parcels, parcel_vec,
                                 mod_fit, tiss_classifier, sphere, directget, curv_thr_list, step_list, track_type,
                                 maxcrossing, max_length)
    print('Tracking Complete')

    # Perform streamline filtering routines
    dir_path = utils.do_dir_path(atlas_select, dwi_file)
    [streams, dir_path] = filter_streamlines(dwi_file, dir_path, gtab, streamlines, life_run, min_length, conn_model,
                                             target_samples, node_size, curv_thr_list, step_list)

    return streams, track_type, target_samples, conn_model, dir_path, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, norm, binary, atlas_mni, curv_thr_list, step_list, fa_path
