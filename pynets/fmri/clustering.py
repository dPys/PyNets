#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
import numpy as np
import warnings

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def indx_1dto3d(idx, sz):
    """
    Translate 1D vector coordinates to 3D matrix coordinates for a 3D matrix
     of size sz.

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
    """

    x = np.divide(idx, np.prod(sz[1:3]))
    y = np.divide(idx - x * np.prod(sz[1:3]), sz[2])
    z = idx - x * np.prod(sz[1:3]) - y * sz[2]
    return x, y, z


def indx_3dto1d(idx, sz):
    """
    Translate 3D matrix coordinates to 1D vector coordinates for a 3D matrix
     of size sz.

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
    """

    if np.linalg.matrix_rank(idx) == 1:
        idx1 = idx[0] * np.prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        idx1 = idx[:, 0] * np.prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]
    return idx1


def ncut(W, nbEigenValues, offset=0.5, maxiterations=100,
         eigsErrorTolerence=1e-6, eps=2.2204e-16):
    """
    This function performs the first step of normalized cut spectral
    clustering. The normalized LaPlacian is calculated on the similarity
    matrix W, and top nbEigenValues eigenvectors are calculated. The number of
    eigenvectors corresponds to the maximum number of classes (K) that will be
    produced by the clustering algorithm.

    Parameters
    ----------
    W : array
        Numpy array containing a symmetric #feature x #feature sparse matrix
        representing the similarity between voxels, traditionally this matrix
        should be positive semidefinite, but regularization is employed to
        allow negative matrix entries (Yu 2001).
    nbEigenValues : int
        Number of eigenvectors that should be calculated, this determines the
        maximum number of clusters (K) that can be derived from the result.

    Returns
    -------
    eigen_val :  array
        Eigenvalues from the eigen decomposition of the LaPlacian of W.
    eigen_vec :  array
        Eigenvectors from the eigen decomposition of the LaPlacian of W.

    References
    ----------
    .. [1] Stella Yu and Jianbo Shi, "Understanding Popout through Repulsion,"
      Computer Vision and Pattern Recognition, December, 2001.
    .. [2] Shi, J., & Malik, J. (2000).  Normalized cuts and image
      segmentation. IEEE
      Transactions on Pattern Analysis and Machine Intelligence, 22(8),
      888-905. doi: 10.1109/34.868688.
    .. [3] Yu, S. X., & Shi, J. (2003). Multiclass spectral clustering.
      Proceedings Ninth
      IEEE International Conference on Computer Vision, (1), 313-319 vol.1.
      Ieee. doi: 10.1109/ICCV.2003.1238361

    """
    import gc
    import scipy.sparse as sps

    m = np.shape(W)[1]

    d = abs(W).sum(0) + offset * 2
    dr = 0.5 * (d - W.sum(0)) + offset

    # Calculation of the normalized LaPlacian
    Dinvsqrt = sps.spdiags((1.0 / np.sqrt(d + eps)), [0], m, m, "csc")

    # Perform the eigen decomposition
    eigen_val, eigen_vec = sps.linalg.eigsh(
        Dinvsqrt * ((W + sps.spdiags(dr, [0], m, m, "csc")) * Dinvsqrt),
        nbEigenValues, maxiter=maxiterations, tol=eigsErrorTolerence,
        which="LA")

    norm_ones = np.linalg.norm(np.ones((m, 1)))

    del d, dr, W, m
    gc.collect()

    # Sort the eigen_vals so that the first is the largest
    i = np.argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    # Normalize the returned eigenvectors
    eigen_vec = Dinvsqrt * np.array(eigen_vec)

    for i in range(0, np.shape(eigen_vec)[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / np.linalg.norm(
            eigen_vec[:, i])) * norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * np.sign(eigen_vec[0, i])

    return eigen_val, eigen_vec


def discretisation(eigen_vec, eps=2.2204e-16):
    """
    This function performs the second step of normalized cut clustering which
    assigns features to clusters based on the eigen vectors from the LaPlacian
    of a similarity matrix. There are a few different ways to perform this
    task. Shi and Malik (2000) iteratively bisect the features based on the
    positive and negative loadings of the eigenvectors. Ng, Jordan and Weiss
    (2001) proposed to perform K-means clustering on the rows of the
    eigenvectors. The method
    implemented here was proposed by Yu and Shi (2003) and it finds a discrete
    solution by iteratively rotating a binarised set of vectors until they are
    maximally similar to the the eigenvectors. An advantage of this method
    over K-means is that it is _more_ deterministic, i.e. you should get very
    similar results every time you run the algorithm on the same data.

    The number of clusters that the features are clustered into is determined
    by the number of eignevectors (number of columns) in the input array
    eigen_vec. A caveat of this method, is that number of resulting clusters
    is bound by the number of eignevectors, but it may contain less.

    Parameters
    ----------
    eigen_vec : array
        Eigenvectors of the normalized LaPlacian calculated from the
        similarity matrix for the corresponding clustering problem.

    Returns
    -------
    eigen_vec_discrete : array
        Discretised eigenvector outputs, i.e. vectors of 0
        and 1 which indicate whether or not a feature belongs
        to the cluster defined by the eigen vector. e.g. a one
        in the 10th row of the 4th eigenvector (column) means
        that feature 10 belongs to cluster #4.

    References
    ----------
    .. [1] Stella Yu and Jianbo Shi, "Understanding Popout through Repulsion,"
      Computer Vision and Pattern Recognition, December, 2001.
    .. [2] Shi, J., & Malik, J. (2000).  Normalized cuts and image
      segmentation. IEEE
      Transactions on Pattern Analysis and Machine Intelligence, 22(8),
      888-905. doi: 10.1109/34.868688.
    .. [3] Yu, S. X., & Shi, J. (2003). Multiclass spectral clustering.
      Proceedings Ninth IEEE International Conference on Computer Vision,
      (1), 313-319 vol.1. Ieee. doi: 10.1109/ICCV.2003.1238361

    """
    import gc
    import scipy as sp
    import scipy.sparse as sps

    # normalize the eigenvectors
    [n, k] = np.shape(eigen_vec)

    eigen_vec = np.divide(eigen_vec, np.reshape(np.kron(
        np.ones(
            (1, k)), np.sqrt(
            np.multiply(
                eigen_vec, eigen_vec).sum(1))), eigen_vec.shape))

    svd_restarts = 0
    exitLoop = 0

    # if there is an exception we try to randomize and rerun SVD again and do
    # this 30 times
    while (svd_restarts < 30) and (exitLoop == 0):
        # initialize algorithm with a random ordering of eigenvectors
        c = np.zeros((n, 1))
        R = np.matrix(np.zeros((k, k)))
        R[:, 0] = np.reshape(
            eigen_vec[int(sp.rand(1) * (n - 1)), :].transpose(), (k, 1)
        )

        for j in range(1, k):
            c = c + abs(eigen_vec * R[:, j - 1])
            R[:, j] = np.reshape(eigen_vec[c.argmin(), :].transpose(), (k, 1))

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        # Iteratively rotate the discretised eigenvectors until they are
        # maximally similar to the input eignevectors,
        # this converges when the differences between the current solution
        # and the previous solution differs by less
        # than eps or we have reached the maximum number of itarations
        while exitLoop == 0:
            nbIterationsDiscretisation = nbIterationsDiscretisation + 1

            # Rotate the original eigen_vectors
            tDiscrete = eigen_vec * R

            # Discretise the result by setting the max of each row=1 and other
            # values to 0
            j = np.reshape(np.asarray(tDiscrete.argmax(1)), n)
            eigenvec_discrete = sps.csc_matrix(
                (np.ones(
                    len(j)), (list(
                        range(
                            0, n)), np.array(j))), shape=(
                    n, k))

            # Calculate a rotation to bring the discrete eigenvectors cluster
            # to the original eigenvectors and catch a SVD convergence error
            # and restart
            try:
                [U, S, Vh] = sp.linalg.svd(
                    eigenvec_discrete.transpose() * eigen_vec)
            except sp.linalg.LinAlgError as e:
                # Catch exception and go back to the beginning of the loop
                print(e, "SVD did not converge. "
                         "Randomizing and trying again...")
                break

            # Test for convergence
            NcutValue = 2 * (n - S.sum())
            if (abs(NcutValue - lastObjectiveValue) < eps) or (
                nbIterationsDiscretisation > nbIterationsDiscretisationMax
            ):
                exitLoop = 1
            else:
                # Otherwise calculate rotation and continue
                lastObjectiveValue = NcutValue
                R = np.matrix(Vh).transpose() * np.matrix(U).transpose()

    if exitLoop == 0:
        raise ValueError("SVD did not converge after 30 retries")
    else:
        return eigenvec_discrete


def parcellate_ncut(W, k, mask_img):
    """
    Converts a connectivity matrix into a nifti file where each voxel
    intensity corresponds to the number of the cluster to which it belongs.
    Clusters are renumberd to be contiguous.

    Parameters
    ----------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the
        temporal/spatial correlation between the time series from voxel i
        and voxel j.
    k : int
        Numbers of clusters that will be generated.
    mask_img : Nifti1Image
        3D NIFTI file containing a mask, which restricts the voxels used in
        the analysis.

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333

    """
    import gc

    # We only have to calculate the eigendecomposition of the LaPlacian once,
    # for the largest number of clusters provided. This provides a significant
    # speedup, without any difference to the results.

    # Calculate each desired clustering result
    eigenvec_discrete = discretisation(ncut(W, k)[1][:, :k])

    # Transform the discretised eigenvectors into a single vector where the
    # value corresponds to the cluster # of the corresponding ROI
    a = eigenvec_discrete[:, 0].todense()

    for i in range(1, k):
        a = a + (i + 1) * eigenvec_discrete[:, i]

    del eigenvec_discrete
    gc.collect()

    unique_a = sorted(set(np.array(a.flatten().tolist()[0])))

    # Renumber clusters to make them non-contiguous
    b = np.zeros((len(a), 1))
    for i in range(0, len(unique_a)):
        b[a == unique_a[i]] = i + 1

    imdat = mask_img.get_fdata()
    mask_aff = mask_img.get_affine()
    mask_hdr = mask_img.get_header()
    imdat[imdat > 0] = 1
    imdat[imdat > 0] = np.short(b[0: int(np.sum(imdat))].flatten())

    del a, b, W
    mask_img.uncache()
    gc.collect()

    return nib.Nifti1Image(
        imdat.astype("uint16"), mask_aff, mask_hdr
    )


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
        3D NIFTI file containing a mask, which restricts the voxels used in
        the analysis.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    Returns
    -------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the spatial
        correlation between the time series from voxel i and voxel j

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333

    """
    import gc
    from scipy.sparse import csc_matrix
    from itertools import product

    neighbors = np.array(
        sorted(
            sorted(
                sorted(
                    [list(x) for x in list(set(product({-1, 0, 1},
                                                       repeat=3)))],
                    key=lambda k: (k[0]),
                ),
                key=lambda k: (k[1]),
            ),
            key=lambda k: (k[2]),
        )
    )

    # Read in the mask
    msz = clust_mask_img.shape

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(
        np.asarray(
            clust_mask_img.dataobj).astype("bool"),
        np.prod(msz))

    # Determine the 1D coordinates of the non-zero
    # elements of the mask
    iv = np.nonzero(mskdat)[0]
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = func_img.get_fdata(dtype=np.float32)
    imdat = np.reshape(func_data, (np.prod(sz[:3]), sz[3]))
    func_img.uncache()
    del func_data

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
    del imdat_s, imdat_sz, imdat_m

    imdat[np.isnan(imdat)] = 0

    # Remove voxels with zero variance, do this here
    # so that the mapping will be consistent across
    # subjects
    vndx = np.nonzero(np.var(imdat, 1) != 0)[0]
    iv = iv[vndx]
    m = len(iv)
    print(m, " # of non-zero valued or non-zero variance voxels in the mask")

    # Construct a sparse matrix from the mask
    msk = csc_matrix(
        (vndx + 1, (iv, np.zeros(m))), shape=(np.prod(msz), 1),
        dtype=np.float32
    )

    sparse_i = []
    sparse_j = []
    sparse_w = [[]]

    for i in range(0, m):
        if i % 1000 == 0:
            print("voxel #", i)

        # Convert index into 3D and calculate neighbors, then convert resulting
        # 3D indices into 1D
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])
        ndx1d = ndx1d[ndx1d<msk.shape[0]]

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
        tc = np.array(imdat[ondx1d.astype("int"), :])

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
    sparse_w = np.reshape(sparse_w, np.prod(np.shape(sparse_w)))

    # Concatenate the i, j, and w_ij vectors
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(outlist) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    outlist = np.reshape(outlist, (3, int(n)))

    m = max(max(outlist[0, :]), max(outlist[1, :])) + 1

    W = csc_matrix(
        (outlist[2, :], (outlist[0, :], outlist[1, :])),
        shape=(int(m), int(m)),
        dtype=np.float32,
    )

    del imdat, msk, mskdat, outlist, m, sparse_i, sparse_j, sparse_w
    gc.collect()

    return W


def make_local_connectivity_tcorr(func_img, clust_mask_img, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    temporal correlation between the time series from voxel i and voxel j.
    Connectivity is only calculated between a voxel and the 27 voxels in its 3D
    neighborhood (face touching and edge touching).

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
        A Scipy sparse matrix, with weights corresponding to the temporal
        correlation between the time series from voxel i and voxel j

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333

    """
    import gc
    from scipy.sparse import csc_matrix
    from itertools import product

    # Index array used to calculate 3D neigbors
    neighbors = np.array(
        sorted(
            sorted(
                sorted(
                    [list(x) for x in list(set(product({-1, 0, 1},
                                                       repeat=3)))],
                    key=lambda k: (k[0]),
                ),
                key=lambda k: (k[1]),
            ),
            key=lambda k: (k[2]),
        )
    )

    # Read in the mask
    msz = np.shape(np.asarray(clust_mask_img.dataobj).astype("bool"))

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(
        np.asarray(
            clust_mask_img.dataobj).astype("bool"),
        np.prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print(f"\nTotal non-zero voxels in the mask: {m}\n")
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = func_img.get_fdata(dtype=np.float32)
    imdat = np.reshape(func_data, (np.prod(sz[:3]), sz[3]))
    func_img.uncache()
    del func_data

    # Construct a sparse matrix from the mask
    msk = csc_matrix(
        (list(range(1, m + 1)), (iv, np.zeros(m))),
        shape=(np.prod(sz[:-1]), 1),
        dtype=np.float32,
    )
    sparse_i = []
    sparse_j = []
    sparse_w = []

    negcount = 0

    # Loop over all of the voxels in the mask
    print("Voxels:")
    for i in range(0, m):
        if i % 1000 == 0:
            print(str(i))
        # Calculate the voxels that are in the 3D neighborhood of the center
        # voxel
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])
        ndx1d = ndx1d[ndx1d<msk.shape[0]]

        # Restrict the neigborhood using the mask
        ondx1d = msk[ndx1d].todense()
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]].flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]]).flatten()

        # Determine the index of the seed voxel in the neighborhood
        nndx = np.nonzero(ndx1d == iv[i])[0]

        # Extract the timecourses for all of the voxels in the neighborhood
        tc = np.array(imdat[ndx1d.astype("int"), :])

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

        # Determine the non-zero correlations (matrix weights) and add their
        # indices and values to the list
        nzndx = np.nonzero(R)[0]
        if len(nzndx) > 0:
            sparse_i = np.append(sparse_i, ondx1d[nzndx] - 1, 0)
            sparse_j = np.append(sparse_j,
                                 (ondx1d[nndx] - 1) * np.ones(len(nzndx)))
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

    W = csc_matrix(
        (outlist[2, :], (outlist[0, :], outlist[1, :])),
        shape=(int(m), int(m)),
        dtype=np.float32,
    )

    del imdat, msk, mskdat, outlist, m, sparse_i, sparse_j, sparse_w
    gc.collect()

    return W


def ensemble_parcellate(infiles, k):
    from sklearn.feature_extraction import image

    # Read in the files, convert them to similarity matrices, and then average
    # them
    for i, file_ in enumerate(infiles):

        img = nib.load(file_)

        img_data = img.get_fdata().astype("int16")

        shape = img.shape
        conn = image.grid_to_graph(
            n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=img_data
        )

        if i == 0:
            W = conn
        else:
            W = W + conn
        del img_data, shape, conn

    # compute the average
    out_img = parcellate_ncut(W / len(infiles), k, img)
    out_img.set_data_dtype(np.uint16)

    return out_img


def proportional(k, voxels_list):
    """Hagenbach-Bischoff Quota"""
    quota = sum(voxels_list) / (1.0 + k)
    frac = [voxels / quota for voxels in voxels_list]
    res = [int(f) for f in frac]
    n = k - sum(res)
    if n == 0:
        return res
    if n < 0:
        return [min(x, k) for x in res]
    remainders = [ai - bi for ai, bi in zip(frac, res)]
    limit = sorted(remainders, reverse=True)[n - 1]
    for i, r in enumerate(remainders):
        if r >= limit:
            res[i] += 1
            n -= 1
            if n == 0:
                return res


def parcellate(func_boot_img, local_corr, clust_type, _local_conn_mat_path,
               num_conn_comps, _clust_mask_corr_img, _standardize,
               _detrending, k, _local_conn, conf, _dir_path, _conn_comps):
    """
    API for performing any of a variety of clustering routines available
    through NiLearn.
    """
    import time
    import os
    import numpy as np
    from nilearn.regions import Parcellations
    from pynets.fmri.estimation import fill_confound_nans
    # from joblib import Memory
    import tempfile

    cache_dir = tempfile.mkdtemp()
    # memory = Memory(cache_dir, verbose=0)

    start = time.time()

    if (clust_type == "ward") and (local_corr != "allcorr"):
        if _local_conn_mat_path is not None:
            if not os.path.isfile(_local_conn_mat_path):
                raise FileNotFoundError(
                    "File containing sparse matrix of local connectivity"
                    " structure not found."
                )
        else:
            raise FileNotFoundError(
                "File containing sparse matrix of local connectivity"
                " structure not found."
            )

    if (
        clust_type == "complete"
        or clust_type == "average"
        or clust_type == "single"
        or clust_type == "ward"
        or (clust_type == "rena" and num_conn_comps == 1)
        or (clust_type == "kmeans" and num_conn_comps == 1)
    ):
        _clust_est = Parcellations(
            method=clust_type,
            standardize=_standardize,
            detrend=_detrending,
            n_parcels=k,
            mask=_clust_mask_corr_img,
            connectivity=_local_conn,
            mask_strategy="background",
            random_state=42,
            memory=None,
            memory_level=0,
            n_jobs=1,
        )

        if conf is not None:
            import pandas as pd
            import random
            from nipype.utils.filemanip import fname_presuffix, copyfile

            out_name_conf = fname_presuffix(
                conf, suffix=f"_tmp{random.randint(1, 1000)}",
                newpath=cache_dir
            )
            copyfile(
                conf,
                out_name_conf,
                copy=True,
                use_hardlink=False)

            confounds = pd.read_csv(out_name_conf, sep="\t")
            if confounds.isnull().values.any():
                conf_corr = fill_confound_nans(confounds, _dir_path)
                try:
                    _clust_est.fit(func_boot_img, confounds=conf_corr)
                except UserWarning:
                    return None
                os.remove(conf_corr)
            else:
                try:
                    _clust_est.fit(func_boot_img, confounds=out_name_conf)
                except UserWarning:
                    return None
            os.remove(out_name_conf)
        else:
            try:
                _clust_est.fit(func_boot_img)
            except UserWarning:
                return None
        _clust_est.labels_img_.set_data_dtype(np.uint16)
        print(
            f"{clust_type}{k}"
            f"{(' clusters: %.2fs' % (time.time() - start))}"
        )

        return _clust_est.labels_img_

    elif clust_type == "ncut":
        out_img = parcellate_ncut(
            _local_conn, k, _clust_mask_corr_img
        )
        out_img.set_data_dtype(np.uint16)
        print(
            f"{clust_type}{k}"
            f"{(' clusters: %.2fs' % (time.time() - start))}"
        )
        return out_img

    elif (
        clust_type == "rena"
        or clust_type == "kmeans"
        and num_conn_comps > 1
    ):
        from pynets.core import nodemaker
        from nilearn.regions import Parcellations
        from nilearn.image import iter_img, new_img_like
        from pynets.core.utils import flatten

        mask_img_list = []
        mask_voxels_dict = dict()
        for i, mask_img in enumerate(iter_img(_conn_comps)):
            mask_voxels_dict[i] = np.int(
                np.sum(np.asarray(mask_img.dataobj)))
            mask_img_list.append(mask_img)

        # Allocate k across connected components using Hagenbach-Bischoff
        # Quota based on number of voxels
        k_list = proportional(k, list(mask_voxels_dict.values()))

        conn_comp_atlases = []
        print(
            f"Building {len(mask_img_list)} separate atlases with "
            f"voxel-proportional k clusters for each "
            f"connected component...")
        for i, mask_img in enumerate(iter_img(mask_img_list)):
            if k_list[i] < 5:
                print(f"Only {k_list[i]} voxels in component. Discarding...")
                continue
            _clust_est = Parcellations(
                method=clust_type,
                standardize=_standardize,
                detrend=_detrending,
                n_parcels=k_list[i],
                mask=mask_img,
                mask_strategy="background",
                random_state=i,
                memory=None,
                memory_level=0,
                n_jobs=1
            )

            if conf is not None:
                import pandas as pd
                import random
                from nipype.utils.filemanip import fname_presuffix, copyfile

                out_name_conf = fname_presuffix(
                    conf, suffix=f"_tmp{random.randint(1, 1000)}",
                    newpath=cache_dir
                )
                copyfile(
                    conf,
                    out_name_conf,
                    copy=True,
                    use_hardlink=False)

                confounds = pd.read_csv(out_name_conf, sep="\t")
                if confounds.isnull().values.any():
                    conf_corr = fill_confound_nans(
                        confounds, _dir_path)
                    try:
                        _clust_est.fit(func_boot_img, confounds=conf_corr)
                    except UserWarning:
                        continue
                else:
                    try:
                        _clust_est.fit(func_boot_img, confounds=conf)
                    except UserWarning:
                        continue
            else:
                try:
                    _clust_est.fit(func_boot_img)
                except UserWarning:
                    continue
            conn_comp_atlases.append(_clust_est.labels_img_)

        # Then combine the multiple atlases, corresponding to each
        # connected component, into a single atlas
        atlas_of_atlases = []
        for atlas in iter_img(conn_comp_atlases):
            bna_data = np.around(
                np.asarray(
                    atlas.dataobj)).astype("uint16")

            # Get an array of unique parcels
            bna_data_for_coords_uniq = np.unique(bna_data)

            # Number of parcels:
            par_max = len(bna_data_for_coords_uniq) - 1
            img_stack = []
            for idx in range(1, par_max + 1):
                roi_img = bna_data == bna_data_for_coords_uniq[idx].astype(
                    "uint16")
                img_stack.append(roi_img.astype("uint16"))
            img_stack = np.array(img_stack)

            img_list = []
            for idy in range(par_max):
                img_list.append(new_img_like(atlas, img_stack[idy]))
            atlas_of_atlases.append(img_list)
            del img_list, img_stack, bna_data

        super_atlas_ward = nodemaker.create_parcel_atlas(
            list(flatten(atlas_of_atlases)))[0]
        super_atlas_ward.set_data_dtype(np.uint16)
        del atlas_of_atlases, conn_comp_atlases, mask_img_list, \
            mask_voxels_dict

        print(
            f"{clust_type}{k}"
            f"{(' clusters: %.2fs' % (time.time() - start))}"
        )

        # memory.clear(warn=False)

        return super_atlas_ward
