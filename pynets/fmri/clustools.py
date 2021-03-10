#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
import numpy as np
import warnings

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
    from scipy import divide, prod

    x = divide(idx, prod(sz[1:3]))
    y = divide(idx - x * prod(sz[1:3]), sz[2])
    z = idx - x * prod(sz[1:3]) - y * sz[2]
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
    from scipy import prod

    if np.linalg.matrix_rank(idx) == 1:
        idx1 = idx[0] * prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        idx1 = idx[:, 0] * prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]
    return idx1


def ncut(W, nbEigenValues):
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
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import spdiags
    from numpy.linalg import norm

    # Parameters
    offset = 0.5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    eps = 2.2204e-16

    m = np.shape(W)[1]

    d = abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    # Calculation of the normalized LaPlacian
    W = W + spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0 / np.sqrt(d + eps)), [0], m, m, "csc")
    P = Dinvsqrt * (W * Dinvsqrt)

    # Perform the eigen decomposition
    eigen_val, eigen_vec = eigsh(
        P, nbEigenValues, maxiter=maxiterations, tol=eigsErrorTolerence,
        which="LA")

    # Sort the eigen_vals so that the first is the largest
    i = np.argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    # Normalize the returned eigenvectors
    eigen_vec = Dinvsqrt * np.array(eigen_vec)
    norm_ones = norm(np.ones((m, 1)))
    for i in range(0, np.shape(eigen_vec)[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / norm(eigen_vec[:, i])) * norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * np.sign(eigen_vec[0, i])

    return eigen_val, eigen_vec


def discretisation(eigen_vec):
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
    import scipy as sp
    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError, svd
    from scipy import divide

    eps = 2.2204e-16

    # normalize the eigenvectors
    [n, k] = np.shape(eigen_vec)
    vm = np.kron(
        np.ones(
            (1, k)), np.sqrt(
            np.multiply(
                eigen_vec, eigen_vec).sum(1)))
    out_vec = np.reshape(vm, eigen_vec.shape)
    eigen_vec = divide(eigen_vec, out_vec)

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
            eigenvec_discrete = csc_matrix(
                (np.ones(
                    len(j)), (list(
                        range(
                            0, n)), np.array(j))), shape=(
                    n, k))

            # Calculate a rotation to bring the discrete eigenvectors cluster
            # to the original eigenvectors
            tSVD = eigenvec_discrete.transpose() * eigen_vec

            # Catch a SVD convergence error and restart
            try:
                [U, S, Vh] = svd(tSVD)
            except LinAlgError as e:
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
    # We only have to calculate the eigendecomposition of the LaPlacian once,
    # for the largest number of clusters provided. This provides a significant
    # speedup, without any difference to the results.
    [_, eigenvec] = ncut(W, k)

    # Calculate each desired clustering result
    eigenvec_discrete = discretisation(eigenvec[:, :k])

    # Transform the discretised eigenvectors into a single vector where the
    # value corresponds to the cluster # of the corresponding ROI
    a = eigenvec_discrete[:, 0].todense()

    for i in range(1, k):
        a = a + (i + 1) * eigenvec_discrete[:, i]

    unique_a = sorted(set(np.array(a.flatten().tolist()[0])))

    # Renumber clusters to make them non-contiguous
    b = np.zeros((len(a), 1))
    for i in range(0, len(unique_a)):
        b[a == unique_a[i]] = i + 1

    imdat = mask_img.get_fdata()
    imdat[imdat > 0] = 1
    imdat[imdat > 0] = np.short(b[0: int(np.sum(imdat))].flatten())

    del a, b, W

    return nib.Nifti1Image(
        imdat.astype("uint16"), mask_img.get_affine(), mask_img.get_header()
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
    from scipy.sparse import csc_matrix
    from scipy import prod
    from itertools import product
    from pynets.fmri.clustools import indx_1dto3d, indx_3dto1d

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
        prod(msz))

    # Determine the 1D coordinates of the non-zero
    # elements of the mask
    iv = np.nonzero(mskdat)[0]
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = func_img.get_fdata(dtype=np.float32)
    imdat = np.reshape(func_data, (prod(sz[:3]), sz[3]))
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
        (vndx + 1, (iv, np.zeros(m))), shape=(prod(msz), 1), dtype=np.float32
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

    W = csc_matrix(
        (outlist[2, :], (outlist[0, :], outlist[1, :])),
        shape=(int(m), int(m)),
        dtype=np.float32,
    )

    del imdat, msk, mskdat, outlist, m, sparse_i, sparse_j, sparse_w

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
    from scipy.sparse import csc_matrix
    from scipy import prod
    from itertools import product
    from pynets.fmri.clustools import indx_1dto3d, indx_3dto1d

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
        prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print(f"\nTotal non-zero voxels in the mask: {m}\n")
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = func_img.get_fdata(dtype=np.float32)
    imdat = np.reshape(func_data, (prod(sz[:3]), sz[3]))
    del func_data

    # Construct a sparse matrix from the mask
    msk = csc_matrix(
        (list(range(1, m + 1)), (iv, np.zeros(m))),
        shape=(prod(sz[:-1]), 1),
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

    # compute the average
    W = W / len(infiles)

    out_img = parcellate_ncut(W, k, img)
    out_img.set_data_dtype(np.uint16)

    return out_img


class NiParcellate(object):
    """
    Class for implementing various clustering routines.
    """

    def __init__(
        self,
        func_file,
        clust_mask,
        k,
        clust_type,
        local_corr,
        outdir,
        conf=None,
        mask=None,
    ):
        """
        Parameters
        ----------
        func_file : str
            File path to a 4D Nifti1Image containing fMRI data.
        clust_mask : str
            File path to a 3D NIFTI file containing a mask, which restricts the
            voxels used in the clustering.
        k : int
            Numbers of clusters that will be generated.
        clust_type : str
            Type of clustering to be performed (e.g. 'ward', 'kmeans',
            'complete', 'average').
        local_corr : str
            Type of local connectivity to use as the basis for clustering
            methods. Options are tcorr or scorr. Default is tcorr.
        outdir : str
            Path to base derivatives directory.
        conf : str
            File path to a confound regressor file for reduce noise in the
            time-series when extracting from ROI's.
        mask : str
            File path to a 3D NIFTI file containing a mask, which restricts the
            voxels used in the analysis.

        References
        ----------
        .. [1] Thirion, B., Varoquaux, G., Dohmatob, E., & Poline, J. B.
          (2014). Which fMRI clustering gives good brain parcellations?
          Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2014.00167
        .. [2] Bellec, P., Rosa-Neto, P., Lyttelton, O. C., Benali, H., &
          Evans, A. C. (2010). Multi-level bootstrap analysis of stable
          clusters in resting-state fMRI. NeuroImage.
          https://doi.org/10.1016/j.neuroimage.2010.02.082
        .. [3] Garcia-Garcia, M., Nikolaidis, A., Bellec, P.,
          Craddock, R. C., Cheung, B., Castellanos, F. X., & Milham, M. P.
          (2018). Detecting stable individual differences in the functional
          organization of the human basal ganglia. NeuroImage.
          https://doi.org/10.1016/j.neuroimage.2017.07.029

        """
        self.func_file = func_file
        self.clust_mask = clust_mask
        self.k = int(k)
        self.clust_type = clust_type
        self.conf = conf
        self.local_corr = local_corr
        self.uatlas = None
        self.atlas = None
        self._detrending = True
        self._standardize = True
        self._func_img = nib.load(self.func_file)
        self.mask = mask
        self._mask_img = None
        self._local_conn_mat_path = None
        self._dir_path = None
        _clust_est = None
        self._local_conn = None
        self._clust_mask_corr_img = None
        self._func_img_data = None
        self._masked_fmri_vol = None
        self._conn_comps = None
        self.num_conn_comps = None
        self.outdir = outdir

    def create_clean_mask(self, num_std_dev=1.5):
        """
        Create a subject-refined version of the clustering mask.
        """
        import os
        from pynets.core import utils
        from nilearn.masking import intersect_masks
        from nilearn.image import index_img, math_img, resample_img

        mask_name = os.path.basename(self.clust_mask).split(".nii")[0]
        self.atlas = f"{mask_name}{'_'}{self.clust_type}{'_k'}{str(self.k)}"
        print(
            f"\nCreating atlas using {self.clust_type} at cluster level"
            f" {str(self.k)} for {str(self.atlas)}...\n"
        )
        self._dir_path = utils.do_dir_path(self.atlas, self.outdir)
        self.uatlas = f"{self._dir_path}/{mask_name}_clust-{self.clust_type}" \
                      f"_k{str(self.k)}.nii.gz"

        # Load clustering mask
        self._func_img.set_data_dtype(np.float32)
        func_vol_img = index_img(self._func_img, 1)
        func_vol_img.set_data_dtype(np.uint16)
        clust_mask_res_img = resample_img(
            nib.load(self.clust_mask),
            target_affine=func_vol_img.affine,
            target_shape=func_vol_img.shape,
            interpolation="nearest",
        )
        clust_mask_res_img.set_data_dtype(np.uint16)
        func_data = np.asarray(func_vol_img.dataobj, dtype=np.float32)
        func_int_thr = np.round(
            np.mean(func_data[func_data > 0])
            - np.std(func_data[func_data > 0]) * num_std_dev,
            3,
        )
        if self.mask is not None:
            self._mask_img = nib.load(self.mask)
            self._mask_img.set_data_dtype(np.uint16)
            mask_res_img = resample_img(
                self._mask_img,
                target_affine=func_vol_img.affine,
                target_shape=func_vol_img.shape,
                interpolation="nearest",
            )
            mask_res_img.set_data_dtype(np.uint16)
            self._clust_mask_corr_img = intersect_masks(
                [
                    math_img(f"img > {func_int_thr}", img=func_vol_img),
                    math_img("img > 0.01", img=clust_mask_res_img),
                    math_img("img > 0.01", img=mask_res_img),
                ],
                threshold=1,
                connected=False,
            )
            self._clust_mask_corr_img.set_data_dtype(np.uint16)
            self._mask_img.uncache()
            mask_res_img.uncache()
        else:
            self._clust_mask_corr_img = intersect_masks(
                [
                    math_img("img > " + str(func_int_thr), img=func_vol_img),
                    math_img("img > 0.01", img=clust_mask_res_img),
                ],
                threshold=1,
                connected=False,
            )
            self._clust_mask_corr_img.set_data_dtype(np.uint16)
        nib.save(self._clust_mask_corr_img,
                 f"{self._dir_path}{'/'}{mask_name}{'.nii.gz'}")

        del func_data
        func_vol_img.uncache()
        clust_mask_res_img.uncache()

        return self.atlas

    def create_local_clustering(self, overwrite, r_thresh, min_region_size=80):
        """
        API for performing any of a variety of clustering routines available
         through NiLearn.
        """
        import os.path as op
        from scipy.sparse import save_npz, load_npz
        from nilearn.regions import connected_regions

        try:
            conn_comps = connected_regions(
                self._clust_mask_corr_img,
                extract_type="connected_components",
                min_region_size=min_region_size,
            )
            self._conn_comps = conn_comps[0]
            self.num_conn_comps = len(conn_comps[1])
        except BaseException:
            raise ValueError("Clustering mask is empty!")

        if not self._conn_comps:
            if np.sum(np.asarray(self._clust_mask_corr_img.dataobj)) == 0:
                raise ValueError("Clustering mask is empty!")
            else:
                self._conn_comps = self._clust_mask_corr_img
                self.num_conn_comps = 1
        print(
            f"Detected {self.num_conn_comps} connected components in "
            f"clustering mask with a mininimum region "
            f"size of {min_region_size}")
        if (
            self.clust_type == "complete"
            or self.clust_type == "average"
            or self.clust_type == "single"
        ):
            if self.num_conn_comps > 1:
                raise ValueError(
                    "Clustering method unstable with spatial constrainsts "
                    "applied to multiple connected components.")

        if (
            self.clust_type == "ward" and self.num_conn_comps > 1
        ) or self.clust_type == "ncut":
            if self.k < self.num_conn_comps:
                raise ValueError(
                    "k must minimally be greater than the total number of "
                    "connected components in "
                    "the mask in the case of agglomerative clustering.")

            if self.local_corr == "tcorr" or self.local_corr == "scorr":
                self._local_conn_mat_path = (
                    f"{self.uatlas.split('.nii')[0]}_"
                    f"{self.local_corr}_conn.npz"
                )

                if (not op.isfile(self._local_conn_mat_path)) or (
                        overwrite is True):
                    from pynets.fmri.clustools import (
                        make_local_connectivity_tcorr,
                        make_local_connectivity_scorr,
                    )

                    if self.local_corr == "tcorr":
                        self._local_conn = make_local_connectivity_tcorr(
                            self._func_img, self._clust_mask_corr_img,
                            thresh=r_thresh)
                    elif self.local_corr == "scorr":
                        self._local_conn = make_local_connectivity_scorr(
                            self._func_img, self._clust_mask_corr_img,
                            thresh=r_thresh)
                    else:
                        raise ValueError(
                            "Local connectivity type not available")
                    print(
                        f"Saving spatially constrained connectivity structure"
                        f" to: {self._local_conn_mat_path}"
                    )
                    save_npz(self._local_conn_mat_path, self._local_conn)
                elif op.isfile(self._local_conn_mat_path):
                    self._local_conn = load_npz(self._local_conn_mat_path)
            elif self.local_corr == "allcorr":
                if self.clust_type == "ncut":
                    raise ValueError(
                        "Must select either `tcorr` or `scorr` local "
                        "connectivity option if you are using "
                        "`ncut` clustering method")

                self._local_conn = "auto"
            else:
                raise ValueError(
                    "Local connectivity method not recognized. Only tcorr,"
                    " scorr, and auto are currently "
                    "supported")
        else:
            self._local_conn = "auto"
        return

    def prep_boot(self, blocklength=1):
        from nilearn.masking import apply_mask

        ts_data = apply_mask(self._func_img, self._clust_mask_corr_img)
        return ts_data, int(int(np.sqrt(ts_data.shape[0])) * blocklength)


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
            random_state=42
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
        from nilearn.regions import connected_regions, Parcellations
        from nilearn.image import iter_img, new_img_like
        from pynets.core.utils import flatten, proportional

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
                random_state=i
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

        atlas_of_atlases = list(flatten(atlas_of_atlases))

        [super_atlas_ward, _] = nodemaker.create_parcel_atlas(
            atlas_of_atlases)
        super_atlas_ward.set_data_dtype(np.uint16)
        del atlas_of_atlases, conn_comp_atlases, mask_img_list, \
            mask_voxels_dict

        print(
            f"{clust_type}{k}"
            f"{(' clusters: %.2fs' % (time.time() - start))}"
        )

        # memory.clear(warn=False)

        return super_atlas_ward
