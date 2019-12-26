#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import warnings
import numpy as np
import indexed_gzip
import nibabel as nib
warnings.filterwarnings("ignore")


def get_sphere(coords, r, vox_dims, dims):
    """
    Return all points within r mm of coords. Generates a cube and then discards all points outside sphere.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    r : int
        Radius for sphere.
    vox_dims : array/tuple
        1D vector (x, y, z) of mm voxel resolution for sphere.
    dims : array/tuple
        1D vector (x, y, z) of image dimensions for sphere.

    Returns
    -------
    neighbors : list
        A list of indices, within the dimensions of the image, that fall within a spherical neighborhood defined by
        the specified error radius of the list of the input coordinates.

    References
    ----------
    .. Adapted from NeuroSynth
    """
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    neighbors = sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)

    return neighbors


def create_parcel_atlas(parcel_list):
    """
    Create a 3D Nifti1Image atlas parcellation of consecutive integer intensities from an input list of ROI's.

    Parameters
    ----------
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    parcel_list_exp : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks, prepended with a
        background image of zeros.
    """
    from nilearn.image import new_img_like, concat_imgs
    parcel_list_exp = [new_img_like(parcel_list[0], np.zeros(parcel_list[0].shape, dtype=bool))] + parcel_list
    concatted_parcels = concat_imgs(parcel_list_exp, dtype=np.float32)
    parcel_sum = np.sum(np.array(range(len(parcel_list_exp))) * np.asarray(concatted_parcels.dataobj), axis=3,
                        dtype=np.uint8)
    net_parcels_map_nifti = nib.Nifti1Image(parcel_sum, affine=parcel_list[0].affine)
    del concatted_parcels, parcel_sum, parcel_list

    return net_parcels_map_nifti, parcel_list_exp


def fetch_nilearn_atlas_coords(atlas):
    """
    Meta-API for nilearn's coordinate atlas fetching API to retrieve any publically-available coordinate
    atlas by string name.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted coordinate atlas supported for fetching. See Nilearn's datasets.atlas module for
        more detailed reference.

    Returns
    -------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas.
    atlas_name : str
        Name of atlas parcellation (can differ slightly from fetch API string).
    networks_list : list
        List of RSN's and their associated cooordinates, if predefined uniquely for a given atlas.
    labels : list
        List of string labels corresponding to atlas nodes.
    """
    from nilearn import datasets
    atlas = getattr(datasets, 'fetch_%s' % atlas)()
    atlas_name = atlas['description'].splitlines()[0]
    if atlas_name is None:
        atlas_name = atlas
    if "b'" in str(atlas_name):
        atlas_name = atlas_name.decode('utf-8')
    print("%s%s%s%s" % ('\n', atlas_name, ' comes with {0}'.format(atlas.keys()), '\n'))
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    print("%s%s" % ('\nStacked atlas coords in array of shape {0}.'.format(coords.shape), '\n'))
    try:
        networks_list = atlas.networks.astype('U').tolist()
    except:
        networks_list = None
    try:
        labels = np.array([s.strip('b\'') for s in atlas.labels.astype('U')]).tolist()
    except:
        labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    if len(coords) <= 1:
        raise ValueError('\nERROR: No coords returned for specified atlas! Ensure an active internet connection.')

    return coords, atlas_name, networks_list, labels


def nilearn_atlas_helper(atlas, parc):
    """
    Meta-API for nilearn's parcellation-based atlas fetching API to retrieve any publically-available parcellation-based
    atlas by string name.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed references.
    parc : bool
        Indicates whether to use the raw parcels as ROI nodes instead of coordinates at their center-of-mass.

    Returns
    -------
    labels : list
        List of string labels corresponding to atlas nodes.
    networks_list : list
        List of RSN's and their associated cooordinates, if predefined uniquely for a given atlas.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    from nilearn import datasets
    if atlas == 'atlas_harvard_oxford':
        atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'atlas_name')('cort-maxprob-thr0-1mm')
    elif atlas == 'atlas_pauli_2017':
        if parc is False:
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'version')('prob')
        else:
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'version')('det')
    elif 'atlas_talairach' in atlas:
        if atlas == 'atlas_talairach_lobe':
            atlas = 'atlas_talairach'
            print('Fetching level: lobe...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'level')('lobe')
        elif atlas == 'atlas_talairach_gyrus':
            atlas = 'atlas_talairach'
            print('Fetching level: gyrus...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'level')('gyrus')
        elif atlas == 'atlas_talairach_ba':
            atlas = 'atlas_talairach'
            print('Fetching level: ba...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas, 'level')('ba')
    else:
        atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas)()
    if len(list(atlas_fetch_obj.keys())) > 0:
        if 'maps' in list(atlas_fetch_obj.keys()):
            uatlas = atlas_fetch_obj.maps
        else:
            uatlas = None
        if 'labels' in list(atlas_fetch_obj.keys()):
            try:
                labels = [i.decode("utf-8") for i in atlas_fetch_obj.labels]
            except:
                labels = [i for i in atlas_fetch_obj.labels]
        else:
            raise ValueError('No labels found.')
        if 'networks' in list(atlas_fetch_obj.keys()):
            try:
                networks_list = [i.decode("utf-8") for i in atlas_fetch_obj.networks]
            except:
                networks_list = [i for i in atlas_fetch_obj.networks]
        else:
            networks_list = None
    else:
        raise RuntimeWarning('Extraction from nilearn datasets failed!')

    return labels, networks_list, uatlas


def mmToVox(img_affine, mmcoords):
    """
    Function to convert a list of mm coordinates to voxel coordinates.

    Parameters
    ----------
    img_affine : array
        4 x 4 2D Numpy array that is the affine of the image space that the coordinates inhabit.
    mmcoords : list
        List of [x, y, z] or (x, y, z) coordinates in mm-space.
    """
    return nib.affines.apply_affine(np.linalg.inv(img_affine), mmcoords)


def VoxTomm(img_affine, voxcoords):
    """
    Function to convert a list of voxel coordinates to mm coordinates.

    Parameters
    ----------
    img_affine : array
        4 x 4 2D Numpy array that is the affine of the image space that the coordinates inhabit.
    voxcoords : list
        List of [x, y, z] or (x, y, z) coordinates in voxel-space.
    """
    return nib.affines.apply_affine(img_affine, voxcoords)


def get_node_membership(network, infile, coords, labels, parc, parcel_list, perc_overlap=0.75, error=4):
    """
    Evaluate the affinity of any arbitrary list of coordinate or parcel nodes for a user-specified RSN based on
    Yeo-7 or Yeo-17 definitions.

    Parameters
    ----------
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    infile : str
        File path to Nifti1Image object whose affine will provide sampling reference for evaluation spatial proximity.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a spatial error cushion in the case of
        evaluating RSN membership from a given list of parcel masks. Default is 0.75.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a spatial error cushion in the case of
        evaluating RSN membership from a given list of coordinates. Default is 4.

    Returns
    -------
    coords_mm : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity for the specified RSN.
    RSN_parcels : list
        Filtered list of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks with a spatial
        affinity for the specified RSN.
    net_labels : list
        Filtered list of string labels corresponding to ROI nodes with a spatial affinity for the specified RSN.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    """
    from nilearn.image import resample_img
    from pynets.core.nodemaker import get_sphere, mmToVox, VoxTomm
    import pkg_resources
    import pandas as pd

    # Determine whether input is from 17-networks or 7-networks
    seven_nets = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    seventeen_nets = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA',
                      'SalVentAttnB', 'LimbicOFC', 'LimbicTempPole', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB',
                      'DefaultC', 'TempPar']

    # Load subject func data
    bna_img = nib.load(infile)
    bna_aff = bna_img.affine
    bna_img.uncache()
    x_vox = np.diagonal(bna_aff[:3, 0:3])[0]
    y_vox = np.diagonal(bna_aff[:3, 0:3])[1]
    z_vox = np.diagonal(bna_aff[:3, 0:3])[2]

    if network in seventeen_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <= 1:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF1mm.nii.gz")
        else:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF2mm.nii.gz")

        # Grab RSN reference file
        nets_ref_txt = pkg_resources.resource_filename("pynets", "rsnrefs/Schaefer2018_1000_17nets_ref.txt")
    elif network in seven_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <= 1:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/SMALLREF1mm.nii.gz")
        else:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/SMALLREF2mm.nii.gz")

        # Grab RSN reference file
        nets_ref_txt = pkg_resources.resource_filename("pynets", "rsnrefs/Schaefer2018_1000_7nets_ref.txt")
    else:
        nets_ref_txt = None

    if not nets_ref_txt:
        raise ValueError("%s%s%s" % ('Network: ', str(network), ' not found!\nSee valid network names using the --help '
                                                                'flag with pynets_run.py'))

    # Create membership dictionary
    dict_df = pd.read_csv(nets_ref_txt, sep="\t", header=None, names=["Index", "Region", "X", "Y", "Z"])
    dict_df.Region.unique().tolist()
    ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}
    par_img = nib.load(par_file)
    RSN_ix = list(ref_dict.keys())[list(ref_dict.values()).index(network)]
    RSNmask = np.asarray(par_img.dataobj)[:, :, :, RSN_ix]

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(bna_aff, i))
    coords_vox = list(tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x)) for x in coords_vox)
    if parc is False:
        i = -1
        RSN_parcels = None
        RSN_coords_vox = []
        net_labels = []
        for coords in coords_vox:
            sphere_vol = np.zeros(RSNmask.shape, dtype=bool)
            sphere_vol[tuple(coords)] = 1
            i = i + 1
            if (RSNmask.astype('bool') & sphere_vol).any():
                print("%s%s%s%s" % (coords, ' coords falls within ', network, '...'))
                RSN_coords_vox.append(coords)
                net_labels.append(labels[i])
                continue
            else:
                inds = get_sphere(coords, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape)
                sphere_vol[tuple(inds.T)] = 1
                if (RSNmask.astype('bool') & sphere_vol).any():
                    print("%s%s%.2f%s%s%s" % (coords, ' coords is within a + or - ', float(error),
                                              ' mm neighborhood of ',
                                              network, '...'))
                    RSN_coords_vox.append(coords)
                    net_labels.append(labels[i])

        coords_mm = []
        for i in RSN_coords_vox:
            coords_mm.append(VoxTomm(bna_aff, i))
        coords_mm = list(set(list(tuple(x) for x in coords_mm)))
    else:
        i = 0
        RSN_parcels = []
        coords_with_parc = []
        net_labels = []
        for parcel in parcel_list:
            parcel_vol = np.zeros(RSNmask.shape, dtype=bool)
            parcel_vol[np.asarray(resample_img(parcel, target_affine=par_img.affine,
                                               target_shape=RSNmask.shape).dataobj) == 1] = 1

            # Count number of unique voxels where overlap of parcel and mask occurs
            overlap_count = len(np.unique(np.where((RSNmask.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))

            # Count number of total unique voxels within the parcel
            total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))

            # Calculate % overlap
            try:
                overlap = float(overlap_count / total_count)
            except RuntimeWarning:
                print('\nWarning: No overlap with roi mask!\n')
                overlap = float(0)

            if overlap >= perc_overlap:
                print("%.2f%s%s%s%s%s" % (100 * overlap, '% of parcel ', labels[i], ' falls within ', str(network),
                                          ' mask...'))
                RSN_parcels.append(parcel)
                coords_with_parc.append(coords[i])
                net_labels.append(labels[i])
            i = i + 1
        coords_mm = list(set(list(tuple(x) for x in coords_with_parc)))

    par_img.uncache()

    if len(coords_mm) <= 1:
        raise ValueError("%s%s%s" % ('\nERROR: No coords from the specified atlas found within ', network, ' network.'))

    return coords_mm, RSN_parcels, net_labels, network


def parcel_masker(roi, coords, parcel_list, labels, dir_path, ID, perc_overlap):
    """
    Evaluate the affinity of any arbitrary list of parcel nodes for a user-specified ROI mask.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a spatial error cushion in the case of
        evaluating ROI-mask membership from a given list of parcel masks.

    Returns
    -------
    coords_adj : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity for the specified ROI mask.
    labels_adj : list
        Filtered list of string labels corresponding to ROI nodes with a spatial affinity for the specified ROI mask.
    parcel_list_adj : list
        Filtered list of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks with a spatial
        affinity to the specified ROI mask.
    """
    from pynets.core import nodemaker
    from nilearn.image import resample_img
    from nilearn import masking
    import os.path as op

    mask_img = nib.load(roi)
    mask_aff = mask_img.affine
    mask_data, _ = masking._load_mask_img(roi)
    mask_img.uncache()

    i = 0
    indices = []
    for parcel in parcel_list:
        parcel_vol = np.zeros(mask_data.shape, dtype=bool)
        parcel_data_reshaped = np.asarray(resample_img(parcel, target_affine=mask_aff,
                                                       target_shape=mask_data.shape).dataobj)
        parcel_vol[parcel_data_reshaped == 1] = 1

        # Count number of unique voxels where overlap of parcel and mask occurs
        overlap_count = len(np.unique(np.where((mask_data.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))

        # Count number of total unique voxels within the parcel
        total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))

        # Calculate % overlap
        try:
            overlap = float(overlap_count / total_count)
        except:
            print("%s%s%s" % ('\nWarning: No overlap of parcel', labels[i], 'with roi mask!\n'))
            overlap = float(0)

        if overlap >= perc_overlap:
            print("%.2f%s%s%s" % (100 * overlap, '% of parcel ', labels[i], ' falls within mask...'))
        else:
            indices.append(i)
        i = i + 1

    labels_adj = list(labels)
    coords_adj = list(tuple(x) for x in coords)
    parcel_list_adj = parcel_list
    try:
        for ix in sorted(indices, reverse=True):
            print("%s%s%s%s" % ('Removing: ', labels_adj[ix], ' at ', coords_adj[ix]))
            labels_adj.pop(ix)
            coords_adj.pop(ix)
            parcel_list_adj.pop(ix)
    except RuntimeError:
        print('ERROR: Restrictive masking. No parcels remain after masking with brain mask/roi...')

    # Create a resampled 3D atlas that can be viewed alongside mask img for QA
    resampled_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_parcels_resampled2roimask_',
                                                   op.basename(roi).split('.')[0], '.nii.gz')
    resampled_parcels_map_nifti = resample_img(nodemaker.create_parcel_atlas(parcel_list_adj)[0],
                                               target_affine=mask_aff, target_shape=mask_data.shape,
                                               interpolation='nearest')
    nib.save(resampled_parcels_map_nifti, resampled_parcels_nii_path)
    resampled_parcels_map_nifti.uncache()
    if not coords_adj:
        raise ValueError('\nERROR: ROI mask was likely too restrictive and yielded < 2 remaining parcels')

    return coords_adj, labels_adj, parcel_list_adj


def coords_masker(roi, coords, labels, error):
    """
    Evaluate the affinity of any arbitrary list of coordinate nodes for a user-specified ROI mask.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a spatial error cushion in the case of
        evaluating the spatial affinity of a given list of coordinates to the given ROI mask.

    Returns
    -------
    coords : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity for the specified ROI mask.
    labels : list
        Filtered list of string labels corresponding to ROI nodes with a spatial affinity for the specified ROI mask.
    """
    from nilearn import masking
    from pynets.core.nodemaker import mmToVox

    mask_data, mask_aff = masking._load_mask_img(roi)
    x_vox = np.diagonal(mask_aff[:3, 0:3])[0]
    y_vox = np.diagonal(mask_aff[:3, 0:3])[1]
    z_vox = np.diagonal(mask_aff[:3, 0:3])[2]

    #    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_aff, i))
    coords_vox = list(tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x)) for x in coords_vox)
    bad_coords = []
    for coord_vox in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coord_vox)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s" % (coord_vox, ' falls within mask...'))
            continue
        inds = get_sphere(coord_vox, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape)
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s%.2f%s" % (coord_vox, ' is within a + or - ', float(error), ' mm neighborhood...'))
            continue
        bad_coords.append(coord_vox)

    bad_coords = [x for x in bad_coords if x is not None]
    indices = []
    for bad_coords in bad_coords:
        indices.append(coords_vox.index(bad_coords))

    labels = list(labels)
    coords = list(tuple(x) for x in coords)
    try:
        for ix in sorted(indices, reverse=True):
            print("%s%s%s%s" % ('Removing: ', labels[ix], ' at ', coords[ix]))
            labels.pop(ix)
            coords.pop(ix)
    except RuntimeError:
        print('ERROR: Restrictive masking. No coords remain after masking with brain mask/roi...')

    if len(coords) <= 1:
        raise ValueError('\nERROR: ROI mask was likely too restrictive and yielded < 2 remaining coords')

    return coords, labels


def get_names_and_coords_of_parcels(uatlas, background_label=0):
    """
    Return list of coordinates and max label intensity for a 3D atlas parcellation image.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    atlas : str
        An arbitrary identified for the atlas based on the filename.
    par_max : int
        The maximum label intensity in the parcellation image.
    """
    import os.path as op
    from nilearn.plotting import find_parcellation_cut_coords

    if not op.isfile(uatlas):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    atlas = uatlas.split('/')[-1].split('.')[0]

    [coords, label_intensities] = find_parcellation_cut_coords(uatlas, background_label, return_label_names=True)
    print("%s%s" % ('Region intensities:\n', label_intensities))
    par_max = len(coords)

    return coords, atlas, par_max


def gen_img_list(uatlas):
    """
    Return list of boolean nifti masks where each masks corresponds to a unique atlas label for the provided atlas
    parcellation. Path string to Nifti1Image is input.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    img_list : list
        List of binarized Nifti1Images corresponding to ROI masks for each unique atlas label.
    """
    import os.path as op
    from nilearn.image import new_img_like
    if not op.isfile(uatlas):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    bna_img = nib.load(uatlas)
    bna_data = np.around(np.asarray(bna_img.dataobj)).astype('uint8')

    # Get an array of unique parcels
    bna_data_for_coords_uniq = np.unique(bna_data)

    # Number of parcels:
    par_max = len(bna_data_for_coords_uniq) - 1
    img_stack = []
    for idx in range(1, par_max + 1):
        roi_img = bna_data == bna_data_for_coords_uniq[idx].astype('uint8')
        img_stack.append(roi_img.astype('uint8'))
    img_stack = np.array(img_stack).astype('uint8')

    img_list = []
    for idy in range(par_max):
        img_list.append(new_img_like(bna_img, img_stack[idy]))

    del img_stack

    bna_img.uncache()
    return img_list


def gen_network_parcels(uatlas, network, labels, dir_path):
    """
    Return a modified verion of an atlas parcellation label, where labels have been filtered baseed on their spatial
    affinity for a specified RSN definition.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.

    Returns
    -------
    out_path : str
        File path to a new, RSN-filtered atlas parcellation Nifti1Image.
    """
    from nilearn.image import concat_imgs
    from pynets.core import nodemaker
    import os.path as op

    if not op.isfile(uatlas):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    img_list = nodemaker.gen_img_list(uatlas)
    print("%s%s%s" % ('\nExtracting parcels associated with ', network, ' network locations...\n'))
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    net_parcels_concatted = concat_imgs(net_parcels)
    net_parcels_sum = np.sum((np.array(range(len(net_parcels))) + 1) * np.asarray(net_parcels_concatted.dataobj),
                             axis=3, dtype=np.uint8)
    out_path = "%s%s%s%s" % (dir_path, '/', network, '_parcels.nii.gz')
    nib.save(nib.Nifti1Image(net_parcels_sum, affine=np.eye(4)), out_path)
    del net_parcels_concatted
    del img_list

    return out_path


def AAL_naming(coords):
    """
    Perform Automated-Anatomical Labeling of each coordinate from a list of a voxel coordinates.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in voxel-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.

    Returns
    -------
    labels : list
        List of string labels corresponding to each coordinates closest anatomical label based on AAL.
    """
    import pandas as pd
    import csv
    from pathlib import Path

    aal_coords_ix_path = "%s%s" % (str(Path(__file__).parent), '/labelcharts/aal_coords_ix.csv')
    aal_region_names_path = "%s%s" % (str(Path(__file__).parent), '/labelcharts/aal_dictionary.csv')
    try:
        aal_coords_ix = pd.read_csv(aal_coords_ix_path)
        with open(aal_region_names_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            aal_labs_dict = dict(reader)
    except FileNotFoundError:
        print('Loading AAL references failed!')

    labels_ix = []
    print('Building region index using AAL MNI coords...')
    for coord in coords:
        reg_lab = aal_coords_ix.loc[aal_coords_ix['coord_tuple'] == str(tuple(np.round(coord).astype('int'))),
                                    'Region_index']
        if len(reg_lab) > 0:
            labels_ix.append(reg_lab.values[0])
        else:
            labels_ix.append(np.nan)

    print('Building list of label names using AAL dictionary...')
    labels = []
    for region_ix in labels_ix:
        if region_ix is np.nan:
            labels.append('Unknown')
        else:
            labels.append(aal_labs_dict[str(region_ix)])

    return labels


def fetch_nodes_and_labels(atlas, uatlas, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    """
    General API for fetching, identifying, and defining atlas nodes based on coordinates and/or labels.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    ref_txt : str
        Path to an atlas reference .txt file that maps labels to intensities corresponding to uatlas.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    in_file : str
        File path to Nifti1Image object whose affine will provide sampling reference for fetching.
    use_AAL_naming : bool
        Indicates whether to perform Automated-Anatomical Labeling of each coordinate from a list of a voxel
        coordinates.
    clustering : bool
        Indicates whether clustering was performed. Default is False.

    Returns
    -------
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    atlas : str
        Name of atlas parcellation (can differ slightly from fetch API string).
    networks_list : list
        List of RSN's and their associated cooordinates, if predefined uniquely for a given atlas.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    par_max : int
        The maximum label intensity in the parcellation image.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    from pynets.core import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009', 'atlas_talairach_gyrus',
                            'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas is None and atlas in nilearn_parc_atlases:
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
    elif uatlas is None and parc is False and atlas in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
        parcel_list = None
        par_max = None
    elif uatlas is None and parc is False and atlas in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
        par_max = None
    elif uatlas:
        if clustering is True:
            while True:
                if op.isfile(uatlas):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas = uatlas.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or '
                  'you have not supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        labels = None
        networks_list = None
    else:
        networks_list = None
        labels = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas:
        if labels:
            pass
        else:
            if (ref_txt is not None) and (op.exists(ref_txt)) and (use_AAL_naming is False):
                labels = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas, '.txt')
                    if op.exists(ref_txt) and (use_AAL_naming is False):
                        try:
                            labels = pd.read_csv(ref_txt,
                                                 sep="\t", header=None, names=["Index", "Region"])['Region'].tolist()
                        except:
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
                                labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            labels = nodemaker.AAL_naming(coords)
                        except:
                            print('AAL reference labeling failed!')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', labels))
    dir_path = utils.do_dir_path(atlas.replace('reor_RAS_nores2mm_', ''), in_file)

    if len(coords) != len(labels):
        labels = len(coords) * [np.nan]
        if len(coords) != len(labels):
            raise ValueError('ERROR: length of coordinates is not equal to length of label names')

    return labels, coords, atlas, networks_list, parcel_list, par_max, uatlas, dir_path


def node_gen_masking(roi, coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas,
                     perc_overlap=0.75, error=4):
    """
    In the case that masking was applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a spatial error cushion in the case of
        evaluating mask/RSN membership from a given list of parcel masks. Default is 0.75.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a spatial error cushion in the case of
        evaluating mask/RSN membership from a given list of coordinates. Default is 4.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    from pynets.core import nodemaker
    import os.path as op
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    # Mask Parcels
    if parc is True:
        # For parcel masking, specify overlap thresh and error cushion in mm voxels
        [coords, labels, parcel_list_masked] = nodemaker.parcel_masker(roi, coords, parcel_list, labels,
                                                                       dir_path, ID, perc_overlap)
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list_masked)
    # Mask Coordinates
    else:
        [coords, labels] = nodemaker.coords_masker(roi, coords, labels, error)
        # Save coords to pickle
        coords_path = "%s%s%s%s" % (dir_path, '/atlas_coords_', op.basename(roi).split('.')[0], '.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)

        net_parcels_map_nifti = None
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/atlas_labelnames_', op.basename(roi).split('.')[0], '.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas, dir_path


def node_gen(coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas):
    """
    In the case that masking was not applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.core import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional roi masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coords_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)

        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas, dir_path


def mask_roi(dir_path, roi, mask, img_file):
    """
    Create derivative ROI based on intersection of roi and brain mask.

    Parameters
    ----------
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    mask : str
        Path to binarized/boolean brain mask Nifti1Image file.
    img_file : str
        File path to Nifti1Image to use to generate an epi-mask.

    Returns
    -------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file, reduced to the spatial intersection with
        the input brain mask.
    """
    import os
    import os.path as op
    from nilearn import masking

    img_mask_path = "%s%s%s%s" % (dir_path, '/', op.basename(img_file).split('.')[0], '_mask.nii.gz')
    nib.save(masking.compute_epi_mask(img_file), img_mask_path)

    if roi and mask:
        print('Refining ROI...')
        roi_red_path = "%s%s%s%s" % (dir_path, '/', op.basename(roi).split('.')[0], '_mask.nii.gz')
        os.system("fslmaths {} -mas {} -mas {} -bin {}".format(roi, mask, img_mask_path, roi_red_path))
        roi = roi_red_path

    return roi


def create_spherical_roi_volumes(node_size, coords, template_mask):
    """
    Create volume ROI mask of spheres from a given set of coordinates and radius.

    Parameters
    ----------
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    template_mask : str
        Path to binarized version of standard (MNI)-space template Nifti1Image file.

    Returns
    -------
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    par_max : int
        The maximum label intensity in the parcellation image.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    parc : bool
        Indicates whether to use the raw parcels as ROI nodes instead of coordinates at their center-of-mass.
    """
    from pynets.core.nodemaker import get_sphere, mmToVox
    mask_img = nib.load(template_mask)
    mask_aff = mask_img.affine
    mask_shape = mask_img.shape
    mask_img.uncache()

    print("%s%s" % ('Creating spherical ROI atlas with radius: ', node_size))

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_aff, i))
    coords_vox = list(set(list(tuple(x) for x in coords_vox)))

    x_vox = np.diagonal(mask_aff[:3, 0:3])[0]
    y_vox = np.diagonal(mask_aff[:3, 0:3])[1]
    z_vox = np.diagonal(mask_aff[:3, 0:3])[2]
    sphere_vol = np.zeros(mask_shape, dtype=bool)
    parcel_list = []
    i = 0
    for coord in coords_vox:
        sphere_vol[tuple(get_sphere(coord, node_size, (np.abs(x_vox), y_vox, z_vox), mask_shape).T)] = i * 1
        parcel_list.append(nib.Nifti1Image(sphere_vol.astype('uint8'), affine=mask_aff))
        i = i + 1

    par_max = len(coords)
    if par_max > 0:
        parc = True
    else:
        raise ValueError('Number of nodes is zero.')

    return parcel_list, par_max, node_size, parc
