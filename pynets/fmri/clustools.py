#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
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


def make_local_connectivity_scorr(func_img, clust_mask_img, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    spatial correlation between the whole brain FC maps generated from the
    time series from voxel i and voxel j. Connectivity is only calculated
    between a voxel and the 27 voxels in its 3D neighborhood
    (face touching and edge touching).

    Parameters
    ----------
    func_img : Nifti1Image
        4D Nifti1Image containing fMRI data.
    clust_mask_img : Nifti1Image
        3D NIFTI file containing a mask, which restricts the voxels used in the analysis.
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
    msz = clust_mask_img.shape

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(np.asarray(clust_mask_img.dataobj), prod(msz))

    # Determine the 1D coordinates of the non-zero
    # elements of the mask
    iv = np.nonzero(mskdat)[0]

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    # nb.load('x.nii.gz').shape -> (x,y,z,t)
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    imdat = np.reshape(np.asarray(func_img.dataobj), (prod(sz[:3]), sz[3]))

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

        # Convert index into 3D and calculate neighbors, then convert resulting 3D indices into 1D
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])

        # Convert 1D indices into masked versions
        ondx1d = msk[ndx1d].todense()

        # Exclude indices not in the mask
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]].flatten()
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
        R = np.corrcoef(np.dot(tc, imdat.T) / (sz[3] - 1))

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

    if len(outlist) > 0:
        m = max(max(outlist[0, :]), max(outlist[1, :])) + 1
    else:
        raise ValueError('Outlist is empty. No spatially-constrained voxels found.')

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


def make_local_connectivity_tcorr(func_img, clust_mask_img, thresh):
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
    func_img : Nifti1Image
        4D Nifti1Image containing fMRI data.
    clust_mask_img : Nifti1Image
        3D NIFTI file containing a mask, which restricts the
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
    msz = np.shape(np.asarray(clust_mask_img.dataobj))

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(np.asarray(clust_mask_img.dataobj), prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print("%s%s%s" % ('\nTotal non-zero voxels in the mask: ', m, '\n'))

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    imdat = np.reshape(np.asarray(func_img.dataobj), (prod(sz[:3]), sz[3]))

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
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])

        # Restrict the neigborhood using the mask
        ondx1d = msk[ndx1d].todense()
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]].flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]]).flatten()

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

    if len(outlist) > 0:
        m = max(max(outlist[0, :]), max(outlist[1, :])) + 1
    else:
        raise ValueError('Outlist is empty. No spatially-constrained voxels found.')

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


class NilParcellate(object):
    """
    Class for implementing various clustering routines.
    """
    def __init__(self, func_file, clust_mask, k, clust_type, conf, vox_size, local_corr):
        """
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
        conf : str
            File path to a confound regressor file for reduce noise in the time-series when extracting from ROI's.
        vox_size : str
            Voxel size in mm. (e.g. 2mm).
        local_corr : str
            Type of local connectivity to use as the basis for clustering methods. Options are tcorr or scorr.
            Default is tcorr.
        """
        self.func_file = func_file
        self.clust_mask = clust_mask
        self.k = k
        self.clust_type = clust_type
        self.conf = conf
        self.detrending = True
        self.standardize = True
        self.func_img = nib.load(self.func_file)
        self.clust_mask_img = nib.load(self.clust_mask)
        self.vox_size = vox_size
        self.local_corr = local_corr
        self.local_conn_mat_path = None
        self.uatlas = None
        self.dir_path = None
        self.clust_mask_corr = None
        self.clust_mask_corr_img = None
        self.clust_est = None
        self.local_conn = None
        self.atlas = None

    def create_clean_mask(self):
        """
        Create a subject-refined version of the clustering mask.
        """
        import os
        import os.path as op
        from pynets.core import utils
        from pynets.registration.reg_utils import check_orient_and_dims

        mask_name = os.path.basename(self.clust_mask).split('.nii.gz')[0]
        self.atlas = "%s%s%s%s%s" % (mask_name, '_', self.clust_type, '_k', str(self.k))
        print("%s%s%s%s%s%s%s" % ('\nCreating atlas using ', self.clust_type, ' at cluster level ', str(self.k),
                                  ' for ', str(self.atlas), '...\n'))
        self.dir_path = utils.do_dir_path(self.atlas, self.func_file)
        self.uatlas = "%s%s%s%s%s%s%s%s" % (self.dir_path, '/', mask_name, '_', self.clust_type, '_k', str(self.k),
                                            '.nii.gz')

        # reorient and reslice mask
        self.clust_mask = check_orient_and_dims(utils.create_temporary_copy(self.clust_mask,
                                                                            op.basename(self.clust_mask).split('.nii.gz')[0],
                                                                            '.nii.gz'), self.vox_size)

        mask_data = np.asarray(self.clust_mask_img.dataobj).astype('bool').astype('int')

        # Ensure mask does not inclue voxels outside of the brain
        mask_data[~np.asarray(self.func_img.dataobj)[:, :, :, 0].astype('bool')] = 0
        self.clust_mask_corr = "%s%s%s%s" % (self.dir_path, '/', mask_name, '.nii.gz')
        self.clust_mask_corr_img = nib.Nifti1Image(mask_data, affine=self.clust_mask_img.affine,
                                                   header=self.clust_mask_img.header)
        nib.save(self.clust_mask_corr_img, self.clust_mask_corr)
        del mask_data
        return self.uatlas, self.atlas

    def create_local_clustering(self, overwrite=False, r_thresh=0.3):
        """
        API for performing any of a variety of clustering routines available through NiLearn.
        """
        import os.path as op
        from scipy.sparse import save_npz, load_npz
        from pynets.fmri.clustools import make_local_connectivity_tcorr, make_local_connectivity_scorr

        self.local_conn_mat_path = "%s%s%s%s" % (self.uatlas.split('.nii.gz')[0], '_', self.local_corr, '_conn.npz')

        if (not op.isfile(self.local_conn_mat_path)) or (overwrite is True):
            if self.local_corr == 'tcorr':
                self.local_conn = make_local_connectivity_tcorr(self.func_img, self.clust_mask_corr_img,
                                                                thresh=r_thresh)
            elif self.local_corr == 'scorr':
                self.local_conn = make_local_connectivity_scorr(self.func_img, self.clust_mask_corr_img,
                                                                thresh=r_thresh)
            else:
                raise ValueError('Local connectivity type not available')

            print("%s%s" % ('Saving spatially constrained connectivity structure to: ', self.local_conn_mat_path))
            save_npz(self.local_conn_mat_path, self.local_conn)
        elif op.isfile(self.local_conn_mat_path):
            self.local_conn = load_npz(self.local_conn_mat_path)

        return

    def parcellate(self):
        """
        API for performing any of a variety of clustering routines available through NiLearn.
        """
        import time
        import os
        import joblib
        from nilearn.regions import Parcellations, connected_label_regions
        import uuid
        import shutil
        from time import strftime

        start = time.time()

        if not os.path.isfile(self.local_conn_mat_path):
            raise FileNotFoundError('File containing sparse matrix of local connectivity structure not found.')

        tmp_dir = '%s%s_%s' % ('/tmp/clustering_', strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
        os.makedirs(tmp_dir, exist_ok=True)
        memory = joblib.Memory(tmp_dir, verbose=0)

        self.clust_est = Parcellations(method=self.clust_type, standardize=self.standardize, detrend=self.detrending,
                                       n_parcels=int(self.k), mask=self.clust_mask_corr_img,
                                       connectivity=self.local_conn, n_jobs=1, memory_level=3, memory=memory)

        if self.conf is not None:
            import pandas as pd
            confounds = pd.read_csv(self.conf, sep='\t')
            if confounds.isnull().values.any():
                run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
                print('Warning: NaN\'s detected in confound regressor file. Filling these with mean values, but the '
                      'regressor file should be checked manually.')
                confounds_nonan = confounds.apply(lambda x: x.fillna(x.mean()), axis=0)
                os.makedirs("%s%s" % (self.dir_path, '/confounds_tmp'), exist_ok=True)
                conf_corr = "%s%s%s%s" % (self.dir_path, '/confounds_tmp/confounds_mean_corrected_', run_uuid, '.tsv')
                confounds_nonan.to_csv(conf_corr, sep='\t')
                self.clust_est.fit(self.func_img, confounds=conf_corr)
            else:
                self.clust_est.fit(self.func_img, confounds=self.conf)
        else:
            self.clust_est.fit(self.func_img)

        nib.save(connected_label_regions(self.clust_est.labels_img_), self.uatlas)

        print("%s%s%s" % (self.clust_type, self.k, " clusters: %.2fs" % (time.time() - start)))

        memory.clear(warn=False)
        shutil.rmtree(tmp_dir)
        return
