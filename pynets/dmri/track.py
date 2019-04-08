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


def reconstruction(conn_model, gtab, dwi, wm_in_dwi, dir_path):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.dmri.track import tens_mod_est, odf_mod_est, csd_mod_est
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

    mod_path = "%s%s%s%s" % (dir_path, '/recon_mod_', conn_model, '.pkl')

    # Create an variable to pickle and open it in write mode
    with open(mod_path, 'wb') as mod_pick:
        pickle.dump(mod, mod_pick)
    mod_pick.close()

    return mod_path


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


def tens_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.dti import TensorModel
    print('Fitting tensor model...')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_data().astype('bool')
    model = TensorModel(gtab)
    mod = model.fit(data, wm_in_dwi_mask)
    return mod


def odf_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.shm import CsaOdfModel
    print('Fitting CSA ODF model...')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_data().astype('bool')
    model = CsaOdfModel(gtab, sh_order=6)
    mod = model.fit(data, wm_in_dwi_mask)
    return mod


def csd_mod_est(gtab, data, wm_in_dwi):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
    print('Fitting CSD model...')
    wm_in_dwi_mask = nib.load(wm_in_dwi).get_data().astype('bool')
    try:
        print('Attempting to use spherical harmonic basis first...')
        model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
    except:
        print('Falling back to estimating recursive response...')
        response = recursive_response(gtab, data, mask=wm_in_dwi_mask, sh_order=8,
                                      peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8, convergence=0.001,
                                      parallel=False)
        print('CSD Reponse: ' + str(response))
        model = ConstrainedSphericalDeconvModel(gtab, response)
    mod = model.fit(data, wm_in_dwi_mask)
    return mod


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


def transform_to_affine(streams, header, affine):
    from dipy.tracking.utils import move_streamlines
    rotation, scale = np.linalg.qr(affine)
    streams = move_streamlines(streams, rotation)
    scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1. / header['voxel_sizes']))
    scale[0:3, 3] = abs(scale[0:3, 3])
    streams = move_streamlines(streams, scale)
    return streams


def save_streams(dwi_img, streamlines, dir_path, vox_size='2mm'):
    from dipy.tracking.streamline import Streamlines
    hdr = dwi_img.header
    streams = "%s%s" % (dir_path, '/streamlines.trk.gz')

    if vox_size == '1mm':
        zoom_set = (1.0, 1.0, 1.0)
    elif vox_size == '2mm':
        zoom_set = (2.0, 2.0, 2.0)
    else:
        raise ValueError('Voxel size not supported. Use 2mm or 1mm')

    # Save streamlines
    affine = np.eye(4)*np.array([-zoom_set[0], zoom_set[1], zoom_set[2],1])
    tract_affine = np.eye(4)*np.array([zoom_set[0], zoom_set[1], zoom_set[2],1])
    trk_hdr = nib.streamlines.trk.TrkFile.create_empty_header()
    trk_hdr['hdr_size'] = 1000
    trk_hdr['dimensions'] = hdr['dim'][1:4].astype('float32')
    trk_hdr['voxel_sizes'] = hdr['pixdim'][1:4]
    trk_hdr['voxel_to_rasmm'] = tract_affine
    trk_hdr['voxel_order'] = 'LPS'
    trk_hdr['pad2'] = 'LPS'
    trk_hdr['image_orientation_patient'] = np.array([1., 0., 0., 0., 1., 0.]).astype('float32')
    trk_hdr['endianness'] = '<'
    trk_hdr['_offset_data'] = 1000
    trk_hdr['nb_streamlines'] = streamlines.total_nb_rows
    streamlines_trans = Streamlines(transform_to_affine(streamlines, trk_hdr, affine))
    tractogram = nib.streamlines.Tractogram(streamlines_trans, affine_to_rasmm=affine)
    trkfile = nib.streamlines.trk.TrkFile(tractogram, header=trk_hdr)
    nib.streamlines.save(trkfile, streams)
    return streams


def run_track(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, dir_path, labels_im_file, mod_path,
              target_samples, curv_thr_list, step_list, overlap_thr_list, track_type, max_length,
              maxcrossing, directget):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
    from dipy.data import get_sphere
    from dipy.direction import ProbabilisticDirectionGetter, BootDirectionGetter, ClosestPeakDirectionGetter, DeterministicMaximumDirectionGetter
    from dipy.reconst.dti import quantize_evecs
    from dipy.tracking import utils
    from dipy.tracking.eudx import EuDX
    from dipy.tracking.streamline import Streamlines
    from pynets.dmri.track import prep_tissues

    with open(mod_path, 'rb') as mod_pick:
        mod_fit = pickle.load(mod_pick)

    tiss_classifier = prep_tissues(nodif_B0_mask, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, tiss_class, dir_path)

    # Load atlas parcellation
    atlas_img = nib.load(labels_im_file)
    atlas_data = atlas_img.get_data().astype('int')

    if np.sum(atlas_data) == 0:
        raise ValueError('ERROR: No non-zero voxels found in atlas. Check any roi masks and/or wm-gm interface images to verify overlap with dwi-registered atlas.')

    # Get sphere
    sphere = get_sphere('repulsion724')

    # Iteratively build a list of streamlines for each ROI while tracking
    streamlines_list = []
    for roi in np.unique(atlas_data)[1:]:
        print("%s%s" % ('ROI: ', roi))
        streamlines = nib.streamlines.array_sequence.ArraySequence()
        while len(streamlines) < target_samples:
            for curv_thr in curv_thr_list:
                print("%s%s" % ('Using curvature threshold: ', curv_thr))
                # Create ProbabilisticDirectionGetter whose name is confusing because it is not getting directions.
                if directget == 'prob':
                    dg = ProbabilisticDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=curv_thr, sphere=sphere)
                elif directget == 'boot':
                    dg = BootDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=curv_thr, sphere=sphere)
                elif directget == 'closest':
                    dg = ClosestPeakDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=curv_thr, sphere=sphere)
                elif directget == 'det':
                    dg = DeterministicMaximumDirectionGetter.from_shcoeff(mod_fit.shm_coeff, max_angle=curv_thr,
                                                                          sphere=sphere)
                elif directget == 'tensor':
                    fa = mod_fit.fa
                    fa[np.isnan(fa)] = 0
                    ind = quantize_evecs(mod_fit.evecs, sphere.vertices)
                for step in step_list:
                    print("%s%s" % ('Using step size: ', step))
                    for overlap_thr in overlap_thr_list:
                        print("%s%s" % ('Using ROI voxel overlap threshold: ', overlap_thr))
                        seed = utils.random_seeds_from_mask(atlas_data==roi, seeds_count=1, seed_count_per_voxel=True,
                                                            affine=np.eye(4))
                        print(seed)
                        if track_type == 'local':
                            if not directget != 'tensor':
                                streamline_generator = LocalTracking(dg, tiss_classifier, seed, np.eye(4),
                                                                     max_cross=maxcrossing, maxlen=max_length,
                                                                     step_size=step, return_all=True)
                            else:
                                raise ValueError('ERROR: Local tracking does not currently support tensor model')
                        elif track_type == 'particle':
                            if directget != 'tensor':
                                streamline_generator = ParticleFilteringTracking(dg, tiss_classifier, seed, np.eye(4),
                                                                                 max_cross=maxcrossing, step_size=step,
                                                                                 maxlen=max_length,
                                                                                 pft_back_tracking_dist=2,
                                                                                 pft_front_tracking_dist=1,
                                                                                 particle_count=15, return_all=True)
                            else:
                                raise ValueError('ERROR: Particle tracking does not currently support tensor model')
                        elif track_type == 'eudx':
                            if directget == 'tensor':
                                streamline_generator = EuDX(fa.astype('f8'), ind, odf_vertices=sphere.vertices,
                                                            a_low=float(0.2), seeds=seed, affine=np.eye(4))
                            else:
                                raise ValueError('ERROR: EuDX tracking is currently only supported for tensor model')
                        streamlines_more = Streamlines(streamline_generator)

                        for s in streamlines_more:
                            streamlines.append(s)
                            if len(streamlines) > target_samples:
                                break

        print('\n')
        streamlines_list.append(streamlines)

    return streamlines_list


def filter_streamlines(dwi, dir_path, gtab, streamlines_list, life_run, min_length):
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines
    from pynets.dmri.track import save_streams, run_LIFE_all
    dwi_img = nib.load(dwi)
    data = dwi_img.get_data()

    # Flatten streamlines list, and apply min length filter
    streamlines = Streamlines([s for stream in streamlines_list for s in stream if len(s) > min_length])
    print(streamlines)

    # Fit LiFE model
    if life_run is True:
        print('Fitting LiFE')
        # Fit Linear Fascicle Evaluation (LiFE)
        [streamlines, mean_rmse] = run_LIFE_all(data, gtab, streamlines)

    # Create density map
    dm = utils.density_map(streamlines, dwi_img.shape, affine=np.eye(4))

    # Save density map
    dm_img = nib.Nifti1Image(dm.astype("int16"), dwi_img.affine)
    dm_img.to_filename("%s%s" % (dir_path, "/density_map.nii.gz"))

    # Save streamlines to trk
    streams = save_streams(dwi, streamlines, dir_path)

    return streamlines, streams
