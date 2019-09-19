#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Recompiled, annotated, and updated to Python 3 by @dPys

Reference:
Craddock, R. C.; James, G. A.; Holtzheimer, P. E.; Hu, X. P. & Mayberg, H. S.
A whole brain fMRI atlas generated via spatially constrained spectral
clustering Human Brain Mapping, 2012, 33, 1914-1928 doi: 10.1002/hbm.21333.

ARTICLE{Craddock2012,
  author = {Craddock, R C and James, G A and Holtzheimer, P E and Hu, X P and
  Mayberg, H S},
  title = {{A whole brain fMRI atlas generated via spatially constrained
  spectral clustering}},
  journal = {Human Brain Mapping},
  year = {2012},
  volume = {33},
  pages = {1914--1928},
  number = {8},
  address = {Department of Neuroscience, Baylor College of Medicine, Houston,
      TX, United States},
  pmid = {21769991},
}
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
    from scipy import prod, rank
    if rank(idx) == 1:
        idx1 = idx[0] * prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        idx1 = idx[:, 0] * prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]
    return idx1


def make_local_connectivity_scorr(func_file, clust_mask, outfile, thresh):
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
    outfile : str
        Output file path, in .npy format, containing a single 3*N vector.
        The first N values are the i index, the second N values are the j index,
        and the last N values are the w_ij, connectivity weights between voxel i
        and voxel j.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    References
    ----------
    .. Adapted from PyClusterROI
    """
    from scipy.sparse import csc_matrix
    from scipy import prod, rank
    neighbors = np.array([[-1, -1, -1], [0, -1, -1], [1, -1, -1],
                          [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                          [-1, 1, -1], [0, 1, -1], [1, 1, -1],
                          [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                          [-1, 0, 0], [0, 0, 0], [1, 0, 0],
                          [-1, 1, 0], [0, 1, 0], [1, 1, 0],
                          [-1, -1, 1], [0, -1, 1], [1, -1, 1],
                          [-1, 0, 1], [0, 0, 1], [1, 0, 1],
                          [-1, 1, 1], [0, 1, 1], [1, 1, 1]])

    # Read in the mask
    msk = nib.load(clust_mask)
    msz = msk.shape

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(msk.get_data(), prod(msz))

    # Determine the 1D coordinates of the non-zero
    # elements of the mask
    iv = np.nonzero(mskdat)[0]

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    # nb.load('x.nii.gz').shape -> (x,y,z,t)
    nim = nib.load(func_file)
    sz = nim.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    imdat = np.reshape(nim.get_data(), (prod(sz[:3]), sz[3]))

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
        tc = imdat[ondx1d, :]

        # Calculate functional connectivity maps for "seed"
        # and 3D neighborhood voxels
        fc = np.dot(tc, imdat.T) / (sz[3] - 1)

        # Calculate the spatial correlation between FC maps
        R = np.corrcoef(fc)
        if rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Set NaN values to 0
        R[np.isnan(R)] = 0

        # Set values below thresh to 0
        R[R < thresh] = 0

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

    # Save the output file to a .NPY file
    np.save(outfile, outlist)

    print('finished ', func_file, ' len ', len(outlist))
    return outfile


def make_local_connectivity_tcorr(func_file, clust_mask, outfile, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    temporal correlation between the time series from voxel i and voxel j.
    Connectivity is only calculated between a voxel and the 27 voxels in its 3D
    neighborhood (face touching and edge touching). The resulting datafiles are
    suitable as inputs to the function binfile_parcellate.

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
    outfile : str
        Output file path, in .npy format, containing a single 3*N vector.
        The first N values are the i index, the second N values are the j index,
        and the last N values are the w_ij, connectivity weights between voxel i
        and voxel j.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).
    """
    from scipy.sparse import csc_matrix
    from scipy import prod, rank
    from itertools import product

    # Index array used to calculate 3D neigbors
    neighbors = np.array(sorted(sorted(sorted([list(x) for x in list(set(product({-1, 0, 1}, repeat=3)))],
                                              key=lambda k: (k[0])), key=lambda k: (k[1])), key=lambda k: (k[2])))

    # Read in the mask
    msk = nib.load(clust_mask)
    msz = np.shape(msk.get_data())
    msk_data = msk.get_data()

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(msk_data, prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print("%s%s%s" % ('\nTotal non-zero voxels in the mask: ', m, '\n'))

    # Read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    nim = nib.load(func_file)
    sz = nim.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    data = nim.get_data()
    imdat = np.reshape(data, (prod(sz[:3]), sz[3]))

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
        tc = np.matrix(imdat[ndx1d.astype('int'), :])

        # Ensure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # Calculate the correlation between all of the voxel TCs
        R = np.corrcoef(tc)
        if rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Extract just the correlations with the seed TC
        R = R[nndx, :].flatten()

        # Set NaN values to 0
        R[np.isnan(R)] = 0
        negcount = negcount + sum(R < 0)

        # Set values below thresh to 0
        R[R < thresh] = 0

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

    # Save the output file to a .NPY file
    np.save(outfile, outlist)

    print("%s%s" % ('Finished ', outfile))
    return outfile


def ncut(W, nbEigenValues):
    """
    This function performs the first step of normalized cut spectral clustering.
    The normalized LaPlacian is calculated on the similarity matrix W, and top
    nbEigenValues eigenvectors are calculated. The number of eigenvectors
    corresponds to the maximum number of classes (K) that will be produced by the
    clustering algorithm.

    Parameters
    ----------
    W : array
        Numpy array containing a symmetric #feature x #feature sparse matrix representing the
        similarity between voxels, traditionally this matrix should be positive semidefinite,
        but regularization is employed to allow negative matrix entries (Yu 2001).
    nbEigenValues : int
        Number of eigenvectors that should be calculated, this determines the maximum number
        of clusters (K) that can be derived from the result.

    Returns
    -------
    eigen_val :  array
        Eigenvalues from the eigen decomposition of the LaPlacian of W.
    eigen_vec :  array
        Eigenvectors from the eigen decomposition of the LaPlacian of W.

    References
    ----------
    .. Adapted from PyClusterROI
    .. [1] Stella Yu and Jianbo Shi, "Understanding Popout through Repulsion," Computer
       Vision and Pattern Recognition, December, 2001.
    .. [2] Shi, J., & Malik, J. (2000).  Normalized cuts and image segmentation. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888-905.
       doi: 10.1109/34.868688.
    .. [3] Yu, S. X., & Shi, J. (2003). Multiclass spectral clustering. Proceedings Ninth
       IEEE International Conference on Computer Vision, (1), 313-319 vol.1. Ieee.
       doi: 10.1109/ICCV.2003.1238361
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
    eigen_val, eigen_vec = eigsh(P, nbEigenValues, maxiter=maxiterations, tol=eigsErrorTolerence, which='LA')

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
    assigns features to clusters based on the eigen vectors from the LaPlacian of
    a similarity matrix. There are a few different ways to perform this task. Shi
    and Malik (2000) iteratively bisect the features based on the positive and
    negative loadings of the eigenvectors. Ng, Jordan and Weiss (2001) proposed to
    perform K-means clustering on the rows of the eigenvectors. The method
    implemented here was proposed by Yu and Shi (2003) and it finds a discrete
    solution by iteratively rotating a binarised set of vectors until they are
    maximally similar to the the eigenvectors. An advantage of this method over K-means
    is that it is _more_ deterministic, i.e. you should get very similar results
    every time you run the algorithm on the same data.

    The number of clusters that the features are clustered into is determined by
    the number of eignevectors (number of columns) in the input array eigen_vec. A
    caveat of this method, is that number of resulting clusters is bound by the
    number of eignevectors, but it may contain less.

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
    .. Adapted from PyClusterROI
    .. [1] Stella Yu and Jianbo Shi, "Understanding Popout through Repulsion," Computer
       Vision and Pattern Recognition, December, 2001.
    .. [2] Shi, J., & Malik, J. (2000).  Normalized cuts and image segmentation. IEEE
       Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888-905.
       doi: 10.1109/34.868688.
    .. [3] Yu, S. X., & Shi, J. (2003). Multiclass spectral clustering. Proceedings Ninth
       IEEE International Conference on Computer Vision, (1), 313-319 vol.1. Ieee.
       doi: 10.1109/ICCV.2003.1238361
    """
    import scipy as sp
    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError, svd
    from scipy import divide
    eps = 2.2204e-16

    # normalize the eigenvectors
    [n, k] = np.shape(eigen_vec)
    vm = np.kron(np.ones((1, k)), np.sqrt(np.multiply(eigen_vec, eigen_vec).sum(1)))
    eigen_vec = divide(eigen_vec, vm)

    svd_restarts = 0
    exitLoop = 0

    # if there is an exception we try to randomize and rerun SVD again and do this 30 times
    while (svd_restarts < 30) and (exitLoop == 0):
        # initialize algorithm with a random ordering of eigenvectors
        c = np.zeros((n, 1))
        R = np.matrix(np.zeros((k, k)))
        R[:, 0] = eigen_vec[int(sp.rand(1) * (n - 1)), :].transpose()

        for j in range(1, k):
            c = c + abs(eigen_vec * R[:, j - 1])
            R[:, j] = eigen_vec[c.argmin(), :].transpose()

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        # Iteratively rotate the discretised eigenvectors until they are maximally similar to the input eignevectors,
        # this converges when the differences between the current solution and the previous solution differs by less
        # than eps or we have reached the maximum number of itarations
        while exitLoop == 0:
            nbIterationsDiscretisation = nbIterationsDiscretisation + 1

            # Rotate the original eigen_vectors
            tDiscrete = eigen_vec * R

            # Discretise the result by setting the max of each row=1 and other values to 0
            j = np.reshape(np.asarray(tDiscrete.argmax(1)), n)
            eigenvec_discrete = csc_matrix((np.ones(len(j)), (list(range(0, n)), np.array(j))), shape=(n, k))

            # Calculate a rotation to bring the discrete eigenvectors cluster to the original eigenvectors
            tSVD = eigenvec_discrete.transpose() * eigen_vec

            # Catch a SVD convergence error and restart
            try:
                [U, S, Vh] = svd(tSVD)
            except LinAlgError:
                # Catch exception and go back to the beginning of the loop
                print("SVD did not converge. Randomizing and trying again...")
                break

            # Test for convergence
            NcutValue = 2 * (n - S.sum())
            if (abs(NcutValue - lastObjectiveValue) < eps) or (
                nbIterationsDiscretisation > nbIterationsDiscretisationMax):
                exitLoop = 1
            else:
                # Otherwise calculate rotation and continue
                lastObjectiveValue = NcutValue
                R = np.matrix(Vh).transpose() * np.matrix(U).transpose()

    if exitLoop == 0:
        raise ValueError("SVD did not converge after 30 retries")
    else:
        return eigenvec_discrete


def binfile_parcellate(infile, outfile, k):
    """
    This function performs normalized cut clustering on the connectivity matrix
    specified by infile into sets of K clusters.

    Parameters
    ----------
    infile : str
        Path to file in .npy or .bin format containing a representation of the connectivity
        matrix to be clustered. This file contains a single vector of length 3*N, in which
        the first N values correspond to the i indices, the second N values correspond to
        the j indices and the last N values correspond to the weights w_ij of the similarity
        matrix W.
    outfile : str
        Path to the output file.
    k : int
        Numbers of clusters that will be generated.

    References
    ----------
    .. Adapted from PyClusterROI
    """
    from scipy.sparse import csc_matrix

    # Read in the file
    if infile.endswith(".npy"):
        print("Reading", infile, "as a npy filetype")
        a = np.load(infile)
    else:
        print("Reading", infile, "as a binary file of doubles")
        fileobj = open(infile, 'rb')
        a = np.fromfile(fileobj)
        fileobj.close()

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(a) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    a = np.reshape(a, (3, int(n)))
    m = max(max(a[0, :]), max(a[1, :])) + 1

    # Make the sparse matrix, CSC format is supposedly efficient for matrix arithmetic
    W = csc_matrix((a[2, :], (a[0, :], a[1, :])), shape=(int(m), int(m)))

    # We only have to calculate the eigendecomposition of the LaPlacian once, for the largest number of clusters
    # provided. This provides a significant speedup, without any difference to the results.
    [_, eigenvec] = ncut(W, k)

    # Calculate each desired clustering result
    eigk = eigenvec[:, :k]
    eigenvec_discrete = discretisation(eigk)

    # Transform the discretised eigenvectors into a single vector where the value corresponds to the cluster # of the
    # corresponding ROI
    group_img = eigenvec_discrete[:, 0]

    for i in range(1, k):
        group_img = group_img + (i + 1) * eigenvec_discrete[:, i]

    # Apply the suffix to the output filename and write out results as a .npy file
    outname = "%s%s%s%s" % (outfile, '_', str(k), '.npy')
    np.save(outname, group_img.todense())
    return outname


def make_image_from_bin_renum(image, binfile, mask):
    """
    Converts a .npy file generated by binfile_parcellation.py into a
    nifti file where each voxel intensity corresponds to the number of the
    cluster to which it belongs. Clusters are renumberd to be contiguous.

    Parameters
    ----------
    image : str
        File path to the Nifti1Image file to be written.
    binfile : str
        The binfile to be converted. The file contains a n_voxel x 1 vector that
        is ultimately converted to a nifti file.
    mask : str
        Mask describing the space of the nifti file. This should
        correspond to the mask originally used to create the
        connectivity matrices used for parcellation.

    References
    ----------
    .. Adapted from PyClusterROI
    """
    # Read in the mask
    nim = nib.load(mask)

    # Read in the binary data
    if binfile.endswith(".npy"):
        print("Reading", binfile, "as a npy filetype")
        a = np.load(binfile)
    else:
        print("Reading", binfile, "as a binary file of doubles")
        a = np.fromfile(binfile)

    unique_a = list(set(a.flatten()))
    unique_a.sort()

    # Renumber clusters to make the contiguous
    b = np.zeros((len(a), 1))
    for i in range(0, len(unique_a)):
        b[a == unique_a[i]] = i + 1

    imdat = nim.get_data()
    # Map the binary data to mask
    imdat[imdat > 0] = 1
    imdat[imdat > 0] = np.short(b[0:int(np.sum(imdat))].flatten())

    # Write out the image as nifti
    nim_out = nib.Nifti1Image(imdat, nim.get_affine(), nim.get_header())
    # nim_out.set_data_dtype('int16')
    nim_out.to_filename(image)
    return image


def nil_parcellate(func_file, clust_mask, k, clust_type, uatlas):
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

    References
    ----------
    .. Adapted from PyClusterROI
    """
    import time
    import nibabel as nib
    from nilearn.regions import Parcellations
    from nilearn.regions import connected_label_regions
    detrending = True

    start = time.time()
    func_img = nib.load(func_file)
    mask_img = nib.load(clust_mask)
    clust_est = Parcellations(method=clust_type, detrend=detrending, n_parcels=int(k),
                              mask=mask_img)
    clust_est.fit(func_img)
    region_labels = connected_label_regions(clust_est.labels_img_)
    # TODO
    # Remove spurious regions < 1500 m^3 in size
    nib.save(region_labels, uatlas)
    print("%s%s%s" % (clust_type, k, " clusters: %.2fs" % (time.time() - start)))
    return region_labels


def individual_clustering(func_file, clust_mask, ID, k, clust_type, thresh=0.5):
    """
    Meta-API for performing any of several types of fMRI clustering based on NiLearn or NCUT.

    Parameters
    ----------
    func_file : str
        File path to a 4D Nifti1Image containing fMRI data.
    clust_mask : str
        File path to a 3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    ID : str
        A subject id or other unique identifier.
    k : int
        Numbers of clusters that will be generated.
    clust_type : str
        Type of clustering to be performed (e.g. 'ward', 'kmeans', 'complete', 'average').
    thresh : str
        Threshold value to be used for NCUT tcorr and scorr. Correlation coefficients
        lower than this value will be removed from the matrix (set to zero).
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

    if clust_type in nilearn_clust_list:
        clustools.nil_parcellate(func_file, clust_mask, k, clust_type, uatlas)
    elif (clust_type == 'ncut_tcorr') or (clust_type == 'ncut_scorr'):
        working_dir = "%s%s%s" % (os.path.dirname(func_file), '/', atlas)
        if clust_type == 'ncut_tcorr':
            outfile = "%s%s%s%s" % (working_dir, '/rm_tcorr_conn_', str(ID), '.npy')
            outfile_parc = "%s%s%s" % (working_dir, '/rm_tcorr_indiv_cluster_', str(ID))
            binfile = "%s%s%s%s%s%s" % (working_dir, '/rm_tcorr_indiv_cluster_', str(ID), '_', str(k), '.npy')
            clustools.make_local_connectivity_tcorr(func_file, clust_mask, outfile, thresh)
        elif clust_type == 'ncut_scorr':
            outfile = "%s%s%s%s" % (working_dir, '/rm_scorr_conn_', str(ID), '.npy')
            outfile_parc = "%s%s%s" % (working_dir, '/rm_scorr_indiv_cluster_', str(ID))
            binfile = "%s%s%s%s%s%s" % (working_dir, '/rm_scorr_indiv_cluster_', str(ID), '_', str(k), '.npy')
            clustools.make_local_connectivity_scorr(func_file, clust_mask, outfile, thresh)
        clustools.binfile_parcellate(outfile, outfile_parc, int(k))
        clustools.make_image_from_bin_renum(uatlas, binfile, clust_mask)

    clustering = True
    return uatlas, atlas, clustering, clust_mask, k, clust_type
