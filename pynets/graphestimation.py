# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np


def get_conn_matrix(time_series, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, mask, min_span_tree,
                    disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, vox_array):
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphLassoCV

    conn_matrix = None
    if conn_model == 'corr':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'partcorr':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'sps':
        # Fit estimator to matrix to get sparse matrix
        estimator_shrunk = None
        estimator = GraphLassoCV()
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            try:
                from sklearn.covariance import GraphLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8, 0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphLasso(alpha)
                            estimator_shrunk.fit(shrunk_cov)
                            print("Retrying covariance matrix estimate with alpha=%s" % alpha)
                            if estimator_shrunk is None:
                                pass
                            else:
                                break
                        except:
                            print("Covariance estimation failed with shrinkage at alpha=%s" % alpha)
                            continue
            except ValueError:
                print('Unstable Lasso estimation! Shrinkage failed. A different connectivity model may be needed.')
        if estimator is None and estimator_shrunk is None:
            raise RuntimeError('ERROR: Covariance estimation failed.')
        if conn_model == 'sps':
            if estimator_shrunk is None:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            else:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov':
            if estimator_shrunk is None:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            else:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphLasso':
        try:
            from inverse_covariance import QuicGraphLasso
        except ImportError:
            print('Cannot run QuicGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphLassoCV':
        try:
            from inverse_covariance import QuicGraphLassoCV
        except ImportError:
            print('Cannot run QuicGraphLassoCV. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphLassoEBIC':
        try:
            from inverse_covariance import QuicGraphLassoEBIC
        except ImportError:
            print('Cannot run QuicGraphLassoEBIC. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'AdaptiveQuicGraphLasso':
        try:
            from inverse_covariance import AdaptiveGraphLasso, QuicGraphLassoEBIC
        except ImportError:
            print('Cannot run AdaptiveGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveGraphLasso(
                estimator=QuicGraphLassoEBIC(
                    init_method='cov',
                ),
                method='binary',
            )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_

    # Weight reuslting matrix by voxels in each label if using parcels as nodes
    # if parc is True:
    #     norm_parcels = (vox_array - min(vox_array)) / (max(vox_array) - min(vox_array))
    #     conn_matrix_norm = normalize(conn_matrix)
    #     conn_matrix = norm_parcels * conn_matrix_norm

    coords = np.array(coords)
    label_names = np.array(label_names)
    return conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, mask, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords


def generate_mask_from_voxels(voxel_coords, volume_dims):
    mask = np.zeros(volume_dims)
    for voxel in voxel_coords:
        mask[tuple(voxel)] = 1
    return mask


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


# def extract_ts_coords_fast(node_size, conf, func_file, coords, dir_path):
#     import nibabel as nib
#     import time
#     import subprocess
#     try:
#         from StringIO import StringIO
#     except ImportError:
#         from io import BytesIO as StringIO
#     from pynets.nodemaker import get_sphere
#     from pynets.graphestimation import generate_mask_from_voxels, normalize
#     start_time = time.time()
#     data = nib.load(func_file)
#     activity_data = data.get_data()
#     volume_dims = np.shape(activity_data)[0:3]
#     ref_affine = data.get_qform()
#     ref_header = data.header
#     vox_dims = tuple(ref_header['pixdim'][1:4])
#     print("%s%s%s" % ('Data loaded: ', str(np.round(time.time() - start_time, 1)), 's'))
#     label_mask = np.zeros(volume_dims)
#     label_file = "%s%s" % (dir_path, '/label_file_tmp.nii.gz')
#
#     [x_vox, y_vox, z_vox] = vox_dims
#     # finding sphere voxels
#     #print(len(coords))
#     def mmToVox(mmcoords):
#         voxcoords = ['', '', '']
#         voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
#         voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
#         voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
#         return voxcoords
#
#     coords_vox = []
#     for i in coords:
#         coords_vox.append(mmToVox(i))
#     coords = list(tuple(x) for x in coords_vox)
#
#     for coord in range(len(coords)):
#         sphere_voxels = get_sphere(coords=coords[coord], r=node_size, vox_dims=vox_dims, dims=volume_dims)
#         # creating mask from found voxels
#         sphere_mask = generate_mask_from_voxels(sphere_voxels, volume_dims)
#         label_mask = label_mask + (coord+1)*sphere_mask
#
#     nib.save(nib.Nifti1Image(label_mask, ref_affine, header=ref_header), label_file)
#     print("%s%s%s" % ('Sphere mask saved: ', str(np.round(time.time() - start_time, 1)), 's'))
#     cmd = "%s%s%s%s" % ('fslmeants -i ', func_file, ' --label=', label_file)
#     stdout_extracted_ts = subprocess.check_output(cmd, shell=True)
#     ts_within_nodes = np.loadtxt(StringIO(stdout_extracted_ts))
#     ts_within_nodes = normalize(ts_within_nodes)
#     print("%s%s%s" % ('Mean time series extracted: ', str(np.round(time.time() - start_time, 1)), 's'))
#     print("%s%s" % ('Number of ROIs expected: ', str(len(coords))))
#     print("%s%s" % ('Number of ROIs found: ', str(ts_within_nodes.shape[1])))
#     return ts_within_nodes, node_size
#
#
# def extract_ts_parc_fast(label_file, conf, func_file, dir_path):
#     import time
#     import subprocess
#     from pynets.graphestimation import normalize
#     try:
#         from StringIO import StringIO
#     except ImportError:
#         from io import BytesIO as StringIO
#     start_time = time.time()
#     cmd = "%s%s%s%s" % ('fslmeants -i ', func_file, ' --label=', label_file)
#     stdout_extracted_ts = subprocess.check_output(cmd, shell=True)
#     ts_within_nodes = np.loadtxt(StringIO(stdout_extracted_ts))
#     ts_within_nodes = normalize(ts_within_nodes)
#     print("%s%s%s" % ('Mean time series extracted: ', str(np.round(time.time() - start_time, 1)), 's'))
#     print("%s%s" % ('Number of ROIs found: ', str(len(ts_within_nodes))))
#
#     node_size = None
#     return ts_within_nodes, node_size


def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network, smooth, atlas_select,
                    uatlas_select, label_names):
    from nilearn import input_data
    # from pynets.graphestimation import extract_ts_parc_fast
    from pynets import utils
    #from sklearn.externals.joblib import Memory

    # if fast is True:
    #     ts_within_nodes = extract_ts_parc_fast(net_parcels_map_nifti, conf, func_file, dir_path)
    # else:
    detrending = True
    # parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
    #                                              standardize=True, smoothing_fwhm=float(smooth),
    #                                              detrend=detrending,
    #                                              memory=Memory(cachedir="%s%s%s" % (dir_path,
    #                                                                                 '/SpheresMasker_cache_',
    #                                                                                 str(ID)), verbose=2),
    #                                              memory_level=1)
    parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
                                                 standardize=True, smoothing_fwhm=float(smooth),
                                                 detrend=detrending, verbose=2)
    # parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
    #                                              standardize=True)
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' volumetric ROI\'s'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    # Save time series as txt file
    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)

    node_size = None
    return ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select, label_names, coords


def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network, smooth, atlas_select, uatlas_select,
                      label_names):
    from nilearn import input_data
    # from pynets.graphestimation import extract_ts_coords_fast
    from pynets import utils
    #from sklearn.externals.joblib import Memory

    # if fast is True:
    #     ts_within_nodes = extract_ts_coords_fast(node_size, conf, func_file, coords, dir_path)
    # else:
    detrending = True
    # spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
    #                                                standardize=True, smoothing_fwhm=float(smooth),
    #                                                detrend=detrending,
    #                                                memory=Memory(cachedir="%s%s%s" % (dir_path,
    #                                                                                   '/SpheresMasker_cache_',
    #                                                                                   str(ID)), verbose=2),
    #                                                memory_level=1)
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
                                                   standardize=True, smoothing_fwhm=float(smooth),
                                                   detrend=detrending)
    # spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
    #                                                standardize=True, verbose=1)
    ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)

    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' coordinate ROI\'s'))
    print("%s%s%s" % ('Using node radius: ', node_size, ' mm'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    # Save time series as txt file
    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)
    return ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select, label_names, coords
