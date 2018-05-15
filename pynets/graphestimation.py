# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np


def get_conn_matrix(time_series, conn_model):
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphLassoCV
    try:
        from brainiak.fcma.util import compute_correlation
    except ImportError:
        pass

    if conn_model == 'corr':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'corr_fast':
        # credit: brainiak
        try:
            print('\nComputing accelerated fcma correlation matrix...\n')
            conn_matrix = compute_correlation(time_series,time_series)
        except RuntimeError:
            print('Cannot run accelerated correlation computation due to a missing dependency. You need brainiak installed!')
    elif conn_model == 'partcorr':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'tangent':
        # credit: nilearn
        print('\nComputing tangent matrix...\n')
        conn_measure = ConnectivityMeasure(kind='tangent')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'sps':
        ##Fit estimator to matrix to get sparse matrix
        estimator = GraphLassoCV()
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            try:
                print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
                from sklearn.covariance import GraphLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8,0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphLasso(alpha)
                            estimator_shrunk.fit(shrunk_cov)
                            print("Calculated graph-lasso covariance matrix for alpha=%s"%alpha)
                            break
                        except FloatingPointError:
                            print("Failed at alpha=%s"%alpha)
                    if estimator_shrunk == None:
                        pass
                    else:
                        break
            except:
                raise ValueError('Unstable Lasso estimation! Shrinkage failed.')

        if conn_model == 'sps':
            try:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            except:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov':
            try:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            except:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphLasso':
        from inverse_covariance import QuicGraphLasso
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
        from inverse_covariance import QuicGraphLassoCV
        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_

    elif conn_model == 'QuicGraphLassoEBIC':
        from inverse_covariance import QuicGraphLassoEBIC
        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_

    elif conn_model == 'AdaptiveQuicGraphLasso':
        from inverse_covariance import AdaptiveGraphLasso, QuicGraphLassoEBIC
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

    return conn_matrix


def generate_mask_from_voxels(voxel_coords, volume_dims):
    mask = np.zeros(volume_dims)
    for voxel in voxel_coords:
        mask[tuple(voxel)] = 1
    return mask


def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def extract_ts_coords_fast(node_size, conf, func_file, coords, dir_path):
    import nibabel as nib
    #import time
    import subprocess
    try:
        from StringIO import StringIO
    except ImportError:
        from io import BytesIO as StringIO
    from pynets.nodemaker import get_sphere
    from pynets.graphestimation import generate_mask_from_voxels, normalize
    #start_time=time.time()
    data = nib.load(func_file)
    activity_data = data.get_data()
    volume_dims = np.shape(activity_data)[0:3]
    ref_affine = data.get_qform()
    ref_header = data.header
    vox_dims = tuple(ref_header['pixdim'][1:4])
    #print('Data loaded: t+'+str(time.time()-start_time)+'s')
    label_mask = np.zeros(volume_dims)
    label_file = "%s%s" % (dir_path, '/label_file_tmp.nii.gz')

    [x_vox, y_vox, z_vox] = vox_dims
    # finding sphere voxels
    print(len(coords))
    def mmToVox(mmcoords):
        voxcoords = ['','','']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords = list(tuple(x) for x in coords_vox)

    for coord in range(len(coords)):
        sphere_voxels = get_sphere(coords=coords[coord], r=node_size, vox_dims=vox_dims, dims=volume_dims)
        # creating mask from found voxels
        sphere_mask = generate_mask_from_voxels(sphere_voxels,volume_dims)
        label_mask = label_mask + (coord+1)*sphere_mask
        #print('Sphere voxels found: t+'+str(time.time()-start_time)+'s')

    nib.save(nib.Nifti1Image(label_mask, ref_affine, header=ref_header), label_file)
    #print('Sphere mask saved: t+'+str(time.time()-start_time)+'s')
    cmd ="%s%s%s%s" % ('fslmeants -i ', func_file, ' --label=', label_file)
    stdout_extracted_ts = subprocess.check_output(cmd, shell=True)
    ts_within_nodes = np.loadtxt(StringIO(stdout_extracted_ts))
    ts_within_nodes = normalize(ts_within_nodes)
    #print('Mean time series extracted: '+str(time.time()-start_time)+'s')
    #print('Number of ROIs expected: '+str(len(coords)))
    #print('Number of ROIs found: '+str(ts_within_nodes.shape[1]))
    return ts_within_nodes


def extract_ts_parc_fast(net_parcels_map_nifti, conf, func_file, dir_path):
    import nibabel as nib
    #import time
    import subprocess
    from pynets.graphestimation import normalize
    try:
        from StringIO import StringIO
    except ImportError:
        from io import BytesIO as StringIO
    #start_time=time.time()
    data = nib.load(func_file)
    activity_data = data.get_data()
    volume_dims = np.shape(activity_data)[0:3]
    ref_affine = data.get_qform()
    ref_header = data.header
    #print('Data loaded: t+'+str(time.time()-start_time)+'s')
    label_mask = np.zeros(volume_dims)

    nib.save(nib.Nifti1Image(label_mask, ref_affine, header=ref_header), net_parcels_map_nifti)
    #print('Sphere mask saved: t+'+str(time.time()-start_time)+'s')
    cmd ="%s%s%s%s" % ('fslmeants -i ', func_file, ' --label=', net_parcels_map_nifti)
    stdout_extracted_ts = subprocess.check_output(cmd, shell=True)
    ts_within_nodes = np.loadtxt(StringIO(stdout_extracted_ts))
    ts_within_nodes = normalize(ts_within_nodes)
    #print('Mean time series extracted: t+'+str(time.time()-start_time)+'s')
    #print('Number of ROIs found: '+str(ts_within_nodes.shape[1]))
    return ts_within_nodes


def extract_ts_parc(net_parcels_map_nifti, conf, func_file, coords, mask, dir_path, ID, network):
    from nilearn import input_data
    from pynets.graphestimation import extract_ts_parc_fast
    from pynets import utils
    ##extract time series from whole brain parcellaions:
    fast=False
    #import time
    #start_time = time.time()
    if fast==True:
        ts_within_nodes = extract_ts_parc_fast(net_parcels_map_nifti, conf, func_file, dir_path)
    else:
        parcel_masker = input_data.NiftiLabelsMasker(labels_img=net_parcels_map_nifti, background_label=0, standardize=True)
        ts_within_nodes = parcel_masker.fit_transform(func_file, confounds=conf)
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' and ', len(coords), ' volumetric ROI\'s\n'))
    ##Save time series as txt file
    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)
    return ts_within_nodes


def extract_ts_coords(node_size, conf, func_file, coords, dir_path, ID, mask, network):
    from nilearn import input_data
    from pynets.graphestimation import extract_ts_coords_fast
    from pynets import utils

    fast=False
    #import time
    #start_time = time.time()
    if fast==True:
        ts_within_nodes = extract_ts_coords_fast(node_size, conf, func_file, coords, dir_path)
    else:
        spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=float(node_size), allow_overlap=True, standardize=True, verbose=1)
        ts_within_nodes = spheres_masker.fit_transform(func_file, confounds=conf)

    #print(time.time()-start_time)
    print("%s%s%d%s" % ('\nTime series has {0} samples'.format(ts_within_nodes.shape[0]), ' and ', len(coords), ' coordinate ROI\'s\n'))
    ##Save time series as txt file
    utils.save_ts_to_file(mask, network, ID, dir_path, ts_within_nodes)
    return ts_within_nodes, node_size
