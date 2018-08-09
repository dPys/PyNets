#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:07:34 2017

@author: PSYC-dap3463
Adapted from Cameron Craddock's PyClusterROI and soon to be replaced with a pypi dependency
"""
import nibabel as nib
import numpy as np
nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = 'auto'

# Craddock, R. C.; James, G. A.; Holtzheimer, P. E.; Hu, X. P. & Mayberg, H. S.
# A whole brain fMRI atlas generated via spatially constrained spectral
# clustering Human Brain Mapping, 2012, 33, 1914-1928 doi: 10.1002/hbm.21333.
#
# ARTICLE{Craddock2012,
#   author = {Craddock, R C and James, G A and Holtzheimer, P E and Hu, X P and
#   Mayberg, H S},
#   title = {{A whole brain fMRI atlas generated via spatially constrained
#   spectral clustering}},
#   journal = {Human Brain Mapping},
#   year = {2012},
#   volume = {33},
#   pages = {1914--1928},
#   number = {8},
#   address = {Department of Neuroscience, Baylor College of Medicine, Houston,
#       TX, United States},
#   pmid = {21769991},
# }


# simple function to translate 1D vector coordinates to 3D matrix coordinates for a 3D matrix of size sz
def indx_1dto3d(idx, sz):
    from scipy import divide, prod
    x = divide(idx, prod(sz[1:3]))
    y = divide(idx-x*prod(sz[1:3]), sz[2])
    z = idx-x*prod(sz[1:3])-y*sz[2]
    return x, y, z


# simple function to translate 3D matrix coordinates to 1D vector coordinates for a 3D matrix of size sz
def indx_3dto1d(idx, sz):
    from scipy import prod, rank
    if rank(idx) == 1:
        idx1 = idx[0]*prod(sz[1:3])+idx[1]*sz[2]+idx[2]
    else:
        idx1 = idx[:, 0]*prod(sz[1:3])+idx[:, 1]*sz[2]+idx[:, 2]
    return idx1


def make_local_connectivity_tcorr(func_file, clust_mask, outfile, thresh):
    from scipy.sparse import csc_matrix
    from scipy import prod, rank
    from itertools import product

    # index array used to calculate 3D neigbors
    neighbors = np.array(sorted(sorted(sorted([list(x) for x in list(set(product({-1, 0, 1}, repeat=3)))],
                                              key=lambda k: (k[0])), key=lambda k: (k[1])), key=lambda k: (k[2])))

    # read in the mask
    msk = nib.load(clust_mask)
    msz = np.shape(msk.get_data())
    msk_data = msk.get_data()
    # convert the 3D mask array into a 1D vector
    mskdat = np.reshape(msk_data, prod(msz))

    # determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print("%s%s%s" % ('\nTotal non-zero voxels in the mask: ', m, '\n'))
    # read in the fmri data
    # NOTE the format of x,y,z axes and time dimension after reading
    nim = nib.load(func_file)
    sz = nim.shape

    # reshape fmri data to a num_voxels x num_timepoints array
    data = nim.get_data()
    imdat = np.reshape(data, (prod(sz[:3]), sz[3]))

    # construct a sparse matrix from the mask
    msk = csc_matrix((list(range(1, m+1)), (iv, np.zeros(m))), shape=(prod(sz[:-1]), 1))
    sparse_i = []
    sparse_j = []
    sparse_w = []

    negcount = 0

    # loop over all of the voxels in the mask
    print('Voxels:')
    for i in range(0, m):
        if i % 1000 == 0:
            print(str(i))
        # calculate the voxels that are in the 3D neighborhood of the center voxel
        ndx3d = indx_1dto3d(iv[i], sz[:-1])+neighbors
        ndx1d = indx_3dto1d(ndx3d, sz[:-1])

        # restrict the neigborhood using the mask
        ondx1d = msk[ndx1d].todense()
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]]
        ndx1d = ndx1d.flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]])
        ondx1d = ondx1d.flatten()

        # determine the index of the seed voxel in the neighborhood
        nndx = np.nonzero(ndx1d == iv[i])[0]
        # exctract the timecourses for all of the voxels in the neighborhood
        tc = np.matrix(imdat[ndx1d.astype('int'), :])

        # make sure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # calculate the correlation between all of the voxel TCs
        R = np.corrcoef(tc)
        if rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # extract just the correlations with the seed TC
        R = R[nndx, :].flatten()

        # set NaN values to 0
        R[np.isnan(R)] = 0
        negcount = negcount+sum(R < 0)

        # set values below thresh to 0
        R[R < thresh] = 0

        # determine the non-zero correlations (matrix weights) and add their indices and values to the list
        nzndx = np.nonzero(R)[0]
        if len(nzndx) > 0:
            sparse_i = np.append(sparse_i, ondx1d[nzndx]-1, 0)
            sparse_j = np.append(sparse_j, (ondx1d[nndx]-1)*np.ones(len(nzndx)))
            sparse_w = np.append(sparse_w, R[nzndx], 0)

    # concatenate the i, j and w_ij into a single vector
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # save the output file to a .NPY file
    np.save(outfile, outlist)

    print("%s%s" % ('Finished ', outfile))


def ncut(W, nbEigenValues):
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import spdiags
    from numpy.linalg import norm
    # parameters
    offset = 0.5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    eps = 2.2204e-16

    m = np.shape(W)[1]

    # make sure that W is symmetric, this is a computationally expensive operation, only use for debugging
    # if (W-W.transpose()).sum() != 0:
    #    print "W should be symmetric!"
    #    exit(0)

    # Degrees and regularization
    # S Yu Understanding Popout through Repulsion CVPR 2001
    # Allows negative values as well as improves invertability of d for small numbers i bet that this is what improves the stability of the eigen
    d = abs(W).sum(0)
    dr = 0.5*(d-W.sum(0))
    d = d+offset*2
    dr = dr+offset

    # calculation of the normalized LaPlacian
    W = W+spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0/np.sqrt(d+eps)), [0], m, m, "csc")
    P = Dinvsqrt*(W*Dinvsqrt)

    # perform the eigen decomposition
    eigen_val, eigen_vec = eigsh(P, nbEigenValues, maxiter=maxiterations, tol=eigsErrorTolerence, which='LA')

    # sort the eigen_vals so that the first is the largest
    i = np.argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    # normalize the returned eigenvectors
    eigen_vec = Dinvsqrt*np.matrix(eigen_vec)
    norm_ones = norm(np.ones((m, 1)))
    for i in range(0, np.shape(eigen_vec)[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / norm(eigen_vec[:, i]))*norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * np.sign(eigen_vec[0, i])

    return eigen_val, eigen_vec


def discretisation(eigen_vec):
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
        R[:, 0] = eigen_vec[int(sp.rand(1)*(n-1)), :].transpose()

        for j in range(1, k):
            c = c+abs(eigen_vec*R[:, j-1])
            R[:, j] = eigen_vec[c.argmin(), :].transpose()

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        # iteratively rotate the discretised eigenvectors until they are maximally similar to the input eignevectors, this converges when the differences between the current solution and the previous solution differs by less than eps or we have reached the maximum number of itarations
        while exitLoop == 0:
            nbIterationsDiscretisation = nbIterationsDiscretisation + 1

            # rotate the original eigen_vectors
            tDiscrete = eigen_vec*R

            # discretise the result by setting the max of each row=1 and other values to 0
            j = np.reshape(np.asarray(tDiscrete.argmax(1)), n)
            eigenvec_discrete = csc_matrix((np.ones(len(j)), (list(range(0, n)), np.array(j))), shape=(n, k))

            # calculate a rotation to bring the discrete eigenvectors cluster to the original eigenvectors
            tSVD = eigenvec_discrete.transpose()*eigen_vec
            # catch a SVD convergence error and restart
            try:
                [U, S, Vh] = svd(tSVD)
            except LinAlgError:
                # catch exception and go back to the beginning of the loop
                print("SVD did not converge. Randomizing and trying again...")
                break

            # test for convergence
            NcutValue = 2*(n-S.sum())
            if (abs(NcutValue-lastObjectiveValue) < eps) or (nbIterationsDiscretisation > nbIterationsDiscretisationMax):
                exitLoop = 1
            else:
                # otherwise calculate rotation and continue
                lastObjectiveValue = NcutValue
                R = np.matrix(Vh).transpose()*np.matrix(U).transpose()

    if exitLoop == 0:
        raise ValueError("SVD did not converge after 30 retries")
    else:
        return eigenvec_discrete


def binfile_parcellate(infile, outfile, k):
    from scipy.sparse import csc_matrix
    # check how long it takes

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
    n = len(a)/3

    # Reshape the 1D vector read in from infile in to a 3xN array
    a = np.reshape(a, (3, int(n)))
    m = max(max(a[0, :]), max(a[1, :]))+1

    #Make the sparse matrix, CSC format is supposedly efficient for matrix arithmetic
    W = csc_matrix((a[2, :], (a[0, :], a[1, :])), shape=(int(m), int(m)))

    #We only have to calculate the eigendecomposition of the LaPlacian once, for the largest number of clusters provided. This provides a significant speedup, without any difference to the results.
    [_, eigenvec] = ncut(W, k)

    # Calculate each desired clustering result
    eigk = eigenvec[:, :k]
    eigenvec_discrete = discretisation(eigk)

    # Transform the discretised eigenvectors into a single vector where the value corresponds to the cluster # of the corresponding ROI
    group_img = eigenvec_discrete[:, 0]

    for i in range(1, k):
        group_img = group_img+(i+1)*eigenvec_discrete[:, i]

    # Apply the suffix to the output filename and write out results as a .npy file
    outname = "%s%s%s%s" % (outfile, '_', str(k), '.npy')
    np.save(outname, group_img.todense())


def make_image_from_bin_renum(image, binfile, mask):
    # read in the mask
    nim = nib.load(mask)

    # read in the binary data
    if binfile.endswith(".npy"):
        print("Reading", binfile, "as a npy filetype")
        a = np.load(binfile)
    else:
        print("Reading", binfile, "as a binary file of doubles")
        a = np.fromfile(binfile)

    unique_a = list(set(a.flatten()))
    unique_a.sort()

    # renumber clusters to make the contiguous
    b = np.zeros((len(a), 1))
    for i in range(0, len(unique_a)):
        b[a == unique_a[i]] = i+1

    imdat = nim.get_data()
    # map the binary data to mask
    imdat[imdat > 0] = 1
    imdat[imdat > 0] = np.short(b[0:int(np.sum(imdat))].flatten())

    # write out the image as nifti
    nim_out = nib.Nifti1Image(imdat, nim.get_affine(), nim.get_header())
    #nim_out.set_data_dtype('int16')
    nim_out.to_filename(image)
