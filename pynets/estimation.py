# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import numpy as np
import warnings
warnings.simplefilter("ignore")


def get_conn_matrix(time_series, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
                    disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, c_boot):
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphicalLassoCV

    conn_matrix = None
    if conn_model == 'corr' or conn_model == 'cor' or conn_model == 'correlation':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'partcorr' or conn_model == 'parcorr' or conn_model == 'partialcorrelation':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar' or conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
        # Fit estimator to matrix to get sparse matrix
        estimator_shrunk = None
        estimator = GraphicalLassoCV(cv=5)
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            try:
                from sklearn.covariance import GraphicalLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8, 0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphicalLasso(alpha)
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
            raise RuntimeError('\nERROR: Covariance estimation failed.')
        if conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
            if estimator_shrunk is None:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            else:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar':
            if estimator_shrunk is None:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            else:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphicalLasso':
        try:
            from inverse_covariance import QuicGraphicalLasso
        except ImportError:
            print('Cannot run QuicGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphicalLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphLassoCV':
        try:
            from inverse_covariance import QuicGraphicalLassoCV
        except ImportError:
            print('Cannot run QuicGraphLassoCV. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphicalLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphicalLassoEBIC':
        try:
            from inverse_covariance import QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run QuicGraphLassoEBIC. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphicalLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'AdaptiveQuicGraphLasso':
        try:
            from inverse_covariance import AdaptiveQuicGraphicalLasso, QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run AdaptiveGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveQuicGraphicalLasso(
                estimator=QuicGraphicalLassoEBIC(
                    init_method='cov',
                ),
                method='binary',
            )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_
    else:
        raise ValueError('\nERROR! No connectivity model specified at runtime. Select a valid estimator using the -mod flag.')

    if conn_matrix.shape < (2, 2):
        raise RuntimeError('\nERROR! Matrix estimation selection yielded an empty or 1-dimensional graph. Check time-series for errors or try using a different atlas')

    coords = np.array(coords)
    label_names = np.array(label_names)
    return conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, c_boot


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
#     from pynets.estimation import generate_mask_from_voxels, normalize
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
#     from pynets.estimation import normalize
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


def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, roi, dir_path, ID, network, smooth, atlas_select,
                    uatlas_select, label_names, c_boot, block_size, mask):
    import os.path
    from nilearn import input_data
    # from pynets.estimation import extract_ts_parc_fast
    from pynets import utils
    #from sklearn.externals.joblib import Memory

    if not os.path.isfile(func_file):
        raise ValueError('\nERROR: Functional data input not found! Check that the file(s) specified with the -i flag exist(s)')

    if conf:
        if not os.path.isfile(conf):
            raise ValueError('\nERROR: Confound regressor file not found! Check that the file(s) specified with the -conf flag exist(s)')


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
                                                 detrend=detrending, verbose=2, mask_img=mask)
    # parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0,
    #                                              standardize=True)
    ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    if ts_within_nodes is None:
        raise RuntimeError('\nERROR: Time-series extraction failed!')
    if float(c_boot) > 0:
        print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
        ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' volumetric ROI\'s'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    # Save time series as txt file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    node_size = None
    return ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select, label_names, coords, c_boot


def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, roi, network, smooth, atlas_select,
                      uatlas_select, label_names, c_boot, block_size, mask):
    import os.path
    from nilearn import input_data
    # from pynets.estimation import extract_ts_coords_fast
    from pynets import utils
    #from sklearn.externals.joblib import Memory

    if not os.path.isfile(func_file):
        raise ValueError('\nERROR: Functional data input not found! Check that the file(s) specified with the -i flag exist(s)')

    if conf:
        if not os.path.isfile(conf):
            raise ValueError('\nERROR: Confound regressor file not found! Check that the file(s) specified with the -conf flag exist(s)')

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
    if len(coords) > 0:
        spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
                                                       standardize=True, smoothing_fwhm=float(smooth),
                                                       detrend=detrending, verbose=2, mask_img=mask)
        # spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True,
        #                                                standardize=True, verbose=1)
        ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)
        if float(c_boot) > 0:
            print("%s%s%s" % ('Performing circular block bootstrapping iteration: ', c_boot, '...'))
            ts_within_nodes = utils.timeseries_bootstrap(ts_within_nodes, block_size)[0]
        if ts_within_nodes is None:
            raise RuntimeError('\nERROR: Time-series extraction failed!')
    else:
        raise RuntimeError('\nERROR: Cannot extract time-series from an empty list of coordinates. \nThis usually means that no nodes were generated based on the specified conditions at runtime (e.g. atlas was overly restricted by an RSN or some user-defined mask.')

    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' mean extracted from ',
                        len(coords), ' coordinate ROI\'s'))
    print("%s%s%s" % ('Using node radius: ', node_size, ' mm'))
    print("%s%s%s" % ('Smoothing FWHM: ', smooth, ' mm\n'))
    # Save time series as txt file
    utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot)
    return ts_within_nodes, node_size, smooth, dir_path, atlas_select, uatlas_select, label_names, coords, c_boot
