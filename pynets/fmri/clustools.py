#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def indx_1dto3d(idx, sz):
    """
    Translate 1D vector coordinates to 3D matrix coordinates for a 3D matrix of size sz.

    Parameters
    ----------
    idx : array
        A 1D numpy coordinate vector.
    sz : array
        Shape of 3D matrix idx.

    Returns
    -------
    x : int
        x-coordinate of 3D matrix coordinates.
    y : int
        y-coordinate of 3D matrix coordinates.
    z : int
        z-coordinate of 3D matrix coordinates.

    References
    ----------
    .. Adapted from PyClusterROI
    """
    from scipy import divide, prod
    x = divide(idx, prod(sz[1:3]))
    y = divide(idx - x * prod(sz[1:3]), sz[2])
    z = idx - x * prod(sz[1:3]) - y * sz[2]
    return x, y, z


def indx_3dto1d(idx, sz):
    """
    Translate 3D matrix coordinates to 1D vector coordinates for a 3D matrix of size sz.

    Parameters
    ----------
    idx : array
        A 3D numpy array of matrix coordinates.
    sz : array
        Shape of 3D matrix idx.

    Returns
    -------
    idx1 : array
        A 1D numpy coordinate vector.

    References
    ----------
    .. Adapted from PyClusterROI
    """
    from scipy import prod
    if np.linalg.matrix_rank(idx) == 1:
        idx1 = idx[0] * prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        idx1 = idx[:, 0] * prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]
    return idx1


def make_local_connectivity_scorr(func_file, clust_mask, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    spatial correlation between the whole brain FC maps generated from the
    time series from voxel i and voxel j. Connectivity is only calculated
    between a voxel and the 27 voxels in its 3D neighborhood
    (face touching and edge touching).

    Parameters
    ----------
    func_file : str
        File path to a 4D Nifti1Image containing fMRI data.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    Returns
    -------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the spatial correlation between the time series from
        voxel i and voxel j

    References
    ----------
    .. Adapted from PyClusterROI
    """
    from scipy.sparse import csc_matrix
    from scipy import prod
    from itertools import product
    from pynets.fmri.clustools import indx_1dto3d, indx_3dto1d

    neighbors = np.array(sorted(sorted(sorted([list(x) for x in list(set(product({-1, 0, 1}, repeat=3)))],
                                              key=lambda k: (k[0])), key=lambda k: (k[1])), key=lambda k: (k[2])))

    # Read in the mask
    msk = nib.load(clust_mask)
    msz = msk.shape

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(np.asarray(msk.dataobj), prod(msz))

    # Determine the 1D coordinates of the non-zero
    # elements of the mask
    iv = np.nonzero(mskdat)[0]

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    # nb.load('x.nii.gz').shape -> (x,y,z,t)
    nim = nib.load(func_file)
    sz = nim.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    imdat = np.reshape(np.asarray(nim.dataobj), (prod(sz[:3]), sz[3]))
    del nim

    # Mask the datset to only the in-mask voxels
    imdat = imdat[iv, :]
    imdat_sz = imdat.shape

    # Z-score fmri time courses, this makes calculation of the
    # correlation coefficient a simple matrix product
    imdat_s = np.tile(np.std(imdat, 1), (imdat_sz[1], 1)).T

    # Replace 0 with really large number to avoid div by zero
    imdat_s[imdat_s == 0] = 1000000
    imdat_m = np.tile(np.mean(imdat, 1), (imdat_sz[1], 1)).T
    imdat = (imdat - imdat_m) / imdat_s

    # Set values with no variance to zero
    imdat[imdat_s == 0] = 0
    imdat[np.isnan(imdat)] = 0

    # Remove voxels with zero variance, do this here
    # so that the mapping will be consistent across
    # subjects
    vndx = np.nonzero(np.var(imdat, 1) != 0)[0]
    iv = iv[vndx]

    m = len(iv)
    print(m, ' # of non-zero valued or non-zero variance voxels in the mask')

    # Construct a sparse matrix from the mask
    msk = csc_matrix((vndx + 1, (iv, np.zeros(m))), shape=(prod(msz), 1))

    sparse_i = []
    sparse_j = []
    sparse_w = [[]]

    for i in range(0, m):
        if i % 1000 == 0:
            print('voxel #', i)
        # Convert index into 3D and calculate neighbors
        ndx3d = indx_1dto3d(iv[i], sz[:-1]) + neighbors

        # Convert resulting 3D indices into 1D
        ndx1d = indx_3dto1d(ndx3d, sz[:-1])

        # Convert 1D indices into masked versions
        ondx1d = msk[ndx1d].todense()

        # Exclude indices not in the mask
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]]
        ndx1d = ndx1d.flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]])
        ondx1d = ondx1d.flatten() - 1

        # Keep track of the index corresponding to the "seed"
        nndx = np.nonzero(ndx1d == iv[i])[0]

        # Extract the time courses corresponding to the "seed"
        # and 3D neighborhood voxels
        tc = np.array(imdat[ondx1d.astype('int'), :])

        # Ensure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # Calculate functional connectivity maps for "seed"
        # and 3D neighborhood voxels
        fc = np.dot(tc, imdat.T) / (sz[3] - 1)

        R = np.corrcoef(fc)

        if np.linalg.matrix_rank(R) == 1:
            R = np.reshape(R, (1, 1))

        # Set nans to 0
        R[np.isnan(R)] = 0

        # Set values below thresh to 0
        R[R < thresh] = 0

        # Calculate the spatial correlation between FC maps
        if np.linalg.matrix_rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Keep track of the indices and the correlation weights
        # to construct sparse connectivity matrix
        sparse_i = np.append(sparse_i, ondx1d, 0)
        sparse_j = np.append(sparse_j, (ondx1d[nndx]) * np.ones(len(ondx1d)))
        sparse_w = np.append(sparse_w, R[nndx, :], 1)

    # Ensure that the weight vector is the correct shape
    sparse_w = np.reshape(sparse_w, prod(np.shape(sparse_w)))

    # Concatenate the i, j, and w_ij vectors
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(outlist) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    outlist = np.reshape(outlist, (3, int(n)))
    m = max(max(outlist[0, :]), max(outlist[1, :])) + 1

    # Make the sparse matrix, CSC format is supposedly efficient for matrix arithmetic
    W = csc_matrix((outlist[2, :], (outlist[0, :], outlist[1, :])), shape=(int(m), int(m)))

    del imdat
    del msk
    del mskdat
    del outlist
    del m
    del sparse_i
    del sparse_j
    del sparse_w

    return W


def make_local_connectivity_tcorr(func_file, clust_mask, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    temporal correlation between the time series from voxel i and voxel j.
    Connectivity is only calculated between a voxel and the 27 voxels in its 3D
    neighborhood (face touching and edge touching).

    References
    ----------
    .. Adapted from PyClusterROI

    Parameters
    ----------
    func_file : str
        File path to a 4D Nifti1Image containing fMRI data.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    Returns
    -------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the temporal correlation between the time series from
        voxel i and voxel j
    """
    from scipy.sparse import csc_matrix
    from scipy import prod
    from itertools import product
    from pynets.fmri.clustools import indx_1dto3d, indx_3dto1d

    # Index array used to calculate 3D neigbors
    neighbors = np.array(sorted(sorted(sorted([list(x) for x in list(set(product({-1, 0, 1}, repeat=3)))],
                                              key=lambda k: (k[0])), key=lambda k: (k[1])), key=lambda k: (k[2])))

    # Read in the mask
    msk = nib.load(clust_mask)
    msz = np.shape(np.asarray(msk.dataobj))

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(np.asarray(msk.dataobj), prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print("%s%s%s" % ('\nTotal non-zero voxels in the mask: ', m, '\n'))

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    nim = nib.load(func_file)
    sz = nim.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    imdat = np.reshape(np.asarray(nim.dataobj), (prod(sz[:3]), sz[3]))
    del nim

    # Construct a sparse matrix from the mask
    msk = csc_matrix((list(range(1, m + 1)), (iv, np.zeros(m))), shape=(prod(sz[:-1]), 1))
    sparse_i = []
    sparse_j = []
    sparse_w = []

    negcount = 0

    # Loop over all of the voxels in the mask
    print('Voxels:')
    for i in range(0, m):
        if i % 1000 == 0:
            print(str(i))
        # Calculate the voxels that are in the 3D neighborhood of the center voxel
        ndx3d = indx_1dto3d(iv[i], sz[:-1]) + neighbors
        ndx1d = indx_3dto1d(ndx3d, sz[:-1])

        # Restrict the neigborhood using the mask
        ondx1d = msk[ndx1d].todense()
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]]
        ndx1d = ndx1d.flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]])
        ondx1d = ondx1d.flatten()

        # Determine the index of the seed voxel in the neighborhood
        nndx = np.nonzero(ndx1d == iv[i])[0]

        # Extract the timecourses for all of the voxels in the neighborhood
        tc = np.array(imdat[ndx1d.astype('int'), :])

        # Ensure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # Calculate the correlation between all of the voxel TCs
        R = np.corrcoef(tc)

        if np.linalg.matrix_rank(R) == 1:
            R = np.reshape(R, (1, 1))

        # Set nans to 0
        R[np.isnan(R)] = 0

        # Set values below thresh to 0
        R[R < thresh] = 0

        if np.linalg.matrix_rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Extract just the correlations with the seed TC
        R = R[nndx, :].flatten()

        # Set NaN values to 0
        negcount = negcount + sum(R < 0)

        # Determine the non-zero correlations (matrix weights) and add their indices and values to the list
        nzndx = np.nonzero(R)[0]
        if len(nzndx) > 0:
            sparse_i = np.append(sparse_i, ondx1d[nzndx] - 1, 0)
            sparse_j = np.append(sparse_j, (ondx1d[nndx] - 1) * np.ones(len(nzndx)))
            sparse_w = np.append(sparse_w, R[nzndx], 0)

    # Concatenate the i, j and w_ij into a single vector
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(outlist) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    outlist = np.reshape(outlist, (3, int(n)))
    m = max(max(outlist[0, :]), max(outlist[1, :])) + 1

    # Make the sparse matrix
    W = csc_matrix((outlist[2, :], (outlist[0, :], outlist[1, :])), shape=(int(m), int(m)))

    del imdat
    del msk
    del mskdat
    del outlist
    del m
    del sparse_i
    del sparse_j
    del sparse_w

    return W


def nil_parcellate(func_file, clust_mask, k, clust_type, uatlas, dir_path, conf, local_corr, detrending=True,
                   standardize=True):
    """
    API for performing any of a variety of clustering routines available through NiLearn.

    Parameters
    ----------
    func_file : str
        File path to a 4D Nifti1Image containing fMRI data.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    k : int
        Numbers of clusters that will be generated.
    clust_type : str
        Type of clustering to be performed (e.g. 'ward', 'kmeans', 'complete', 'average').
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    local_corr : str
        Type of local connectivity estimation ('tcorr' or 'scorr').
    detrending : bool
        Indicates whether to remove linear trends from time-series when extracting across nodes. Default is True.
    standardize : bool
        If standardize is True, the time-series are centered and normed: their mean is put to 0 and their variance to 1
        in the time dimension.

    Returns
    -------
    region_labels : Nifti1Image
        A parcellation image.
    """
    import time
    import os
    from nilearn.regions import Parcellations, connected_label_regions
    from pynets.fmri.clustools import make_local_connectivity_tcorr, make_local_connectivity_scorr

    start = time.time()
    func_img = nib.load(func_file)
    clust_mask_img = nib.load(clust_mask)

    if local_corr == 'tcorr':
        local_conn = make_local_connectivity_tcorr(func_file, clust_mask, thresh=0.5)
    elif local_corr == 'scorr':
        local_conn = make_local_connectivity_scorr(func_file, clust_mask, thresh=0.5)
    else:
        raise ValueError('Local connectivity type not available')

    clust_est = Parcellations(method=clust_type, standardize=standardize, detrend=detrending, n_parcels=int(k),
                              mask=clust_mask_img, connectivity=local_conn)
    if conf is not None:
        import pandas as pd
        confounds = pd.read_csv(conf, sep='\t')
        if confounds.isnull().values.any():
            import uuid
            from time import strftime
            run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
            print('Warning: NaN\'s detected in confound regressor file. Filling these with mean values, but the '
                  'regressor file should be checked manually.')
            confounds_nonan = confounds.apply(lambda x: x.fillna(x.mean()), axis=0)
            os.makedirs("%s%s" % (dir_path, '/confounds_tmp'), exist_ok=True)
            conf_corr = "%s%s%s%s" % (dir_path, '/confounds_tmp/confounds_mean_corrected_', run_uuid, '.tsv')
            confounds_nonan.to_csv(conf_corr, sep='\t')
            clust_est.fit(func_img, confounds=conf_corr)
        else:
            clust_est.fit(func_img, confounds=conf)
    else:
        clust_est.fit(func_img)
    region_labels = connected_label_regions(clust_est.labels_img_)
    nib.save(region_labels, uatlas)

    print("%s%s%s" % (clust_type, k, " clusters: %.2fs" % (time.time() - start)))

    del clust_est
    del func_img
    del clust_mask_img

    return region_labels


def individual_clustering(func_file, conf, clust_mask, ID, k, clust_type, local_corr='tcorr'):
    """
    Meta-API for performing any of several types of fMRI clustering based on NiLearn Parcellations and tcorr/scorr
    spatial constraints.

    Parameters
    ----------
    func_file : str
        File path to a 4D Nifti1Image containing fMRI data.
    conf : str
        File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    ID : str
        A subject id or other unique identifier.
    k : int
        Numbers of clusters that will be generated.
    clust_type : str
        Type of clustering to be performed (e.g. 'ward', 'kmeans', 'complete', 'average').
    local_corr : str
        Type of local connectivity to use as the basis for clustering methods. Options are tcorr or scorr.
        Default is tcorr.

    Returns
    -------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of the atlas based on `clust_type`, `k`, and `clust_mask`
    clustering : bool
        A variable indicating that clustering was performed successfully.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricted the
        voxels used in the analysis.
    k : int
        Numbers of clusters that were generated.
    clust_type : str
        Type of clustering to be performed (e.g. 'ward', 'kmeans', 'complete', 'average').
    """
    import os
    from pynets.core import utils
    from pynets.fmri import clustools

    nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']

    mask_name = os.path.basename(clust_mask).split('.nii.gz')[0]
    atlas = "%s%s%s%s%s" % (mask_name, '_', clust_type, '_k', str(k))
    print("%s%s%s%s%s%s%s" % ('\nCreating atlas using ', clust_type, ' at cluster level ', str(k),
                              ' for ', str(atlas), '...\n'))
    dir_path = utils.do_dir_path(atlas, func_file)
    uatlas = "%s%s%s%s%s%s%s%s" % (dir_path, '/', mask_name, '_', clust_type, '_k', str(k), '.nii.gz')

    # Ensure mask does not inclue voxels outside of the brain
    mask_img = nib.load(clust_mask)
    mask_data = np.asarray(mask_img.dataobj).astype('bool').astype('int')
    func_img = nib.load(func_file)
    func_data = np.asarray(func_img.dataobj).astype('bool')
    func_data_sample_slice = func_data[:, :, :, 0]
    mask_data[~func_data_sample_slice] = 0
    clust_mask_corr = "%s%s%s%s" % (dir_path, '/', mask_name, '.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, affine=mask_img.affine, header=mask_img.header), clust_mask_corr)
    del mask_data
    del func_data

    if clust_type in nilearn_clust_list:
        clustools.nil_parcellate(func_file, clust_mask_corr, k, clust_type, uatlas, dir_path, conf,
                                 local_corr=local_corr)
        clustering = True
        del clust_mask_corr
    else:
        raise ValueError('Clustering method not recognized. '
                         'See: https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.html#nilearn.'
                         'regions.Parcellations')

    return uatlas, atlas, clustering, clust_mask, k, clust_type
