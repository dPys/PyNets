# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")


def get_sphere(coords, r, vox_dims, dims):
    """

    :param coords:
    :param r:
    :param vox_dims:
    :param dims:
    :return:
    """
    # Adapted from Neurosynth
    # Return all points within r mm of coords. Generates a cube and then discards all points outside sphere. Only
    # returns values that fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)


def create_parcel_atlas(parcel_list):
    """

    :param parcel_list:
    :return:
    """
    from nilearn.image import new_img_like, concat_imgs
    parcel_background = new_img_like(parcel_list[0], np.zeros(parcel_list[0].shape, dtype=bool))
    parcel_list_exp = [parcel_background] + parcel_list
    parcellation = concat_imgs(parcel_list_exp).get_fdata()
    index_vec = np.array(range(len(parcel_list_exp))) + 1
    net_parcels_sum = np.sum(index_vec * parcellation, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=parcel_list[0].affine)
    return net_parcels_map_nifti, parcel_list_exp


def fetch_nilearn_atlas_coords(atlas_select):
    """

    :param atlas_select:
    :return:
    """
    from nilearn import datasets
    atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
    atlas_name = atlas['description'].splitlines()[0]
    if atlas_name is None:
        atlas_name = atlas_select
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
        label_names = np.array([s.strip('b\'') for s in atlas.labels.astype('U')]).tolist()
    except:
        label_names = None

    if len(coords) <= 1:
        raise ValueError('\nERROR: No coords returned for specified atlas! Be sure you have an active internet connection.')

    return coords, atlas_name, networks_list, label_names


def nilearn_atlas_helper(atlas_select, parc):
    """

    :param atlas_select:
    :param parc:
    :return:
    """
    from nilearn import datasets
    if atlas_select == 'atlas_harvard_oxford':
        atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'atlas_name')('cort-maxprob-thr0-1mm')
    elif atlas_select == 'atlas_pauli_2017':
        if parc is False:
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'version')('prob')
        else:
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'version')('det')
    elif 'atlas_talairach' in atlas_select:
        if atlas_select == 'atlas_talairach_lobe':
            atlas_select = 'atlas_talairach'
            print('Fetching level: lobe...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'level')('lobe')
        elif atlas_select == 'atlas_talairach_gyrus':
            atlas_select = 'atlas_talairach'
            print('Fetching level: gyrus...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'level')('gyrus')
        elif atlas_select == 'atlas_talairach_ba':
            atlas_select = 'atlas_talairach'
            print('Fetching level: ba...')
            atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select, 'level')('ba')
    else:
        atlas_fetch_obj = getattr(datasets, 'fetch_%s' % atlas_select)()
    if len(list(atlas_fetch_obj.keys())) > 0:
        if 'maps' in list(atlas_fetch_obj.keys()):
            uatlas_select = atlas_fetch_obj.maps
        else:
            uatlas_select = None
        if 'labels' in list(atlas_fetch_obj.keys()):
            try:
                label_names = [i.decode("utf-8") for i in atlas_fetch_obj.labels]
            except:
                label_names = [i for i in atlas_fetch_obj.labels]
        else:
            label_names = None
        if 'networks' in list(atlas_fetch_obj.keys()):
            try:
                networks_list = [i.decode("utf-8") for i in atlas_fetch_obj.networks]
            except:
                networks_list = [i for i in atlas_fetch_obj.networks]
        else:
            networks_list = None
    else:
        raise RuntimeWarning('Extraction from nilearn datasets failed!')

    return label_names, networks_list, uatlas_select


def get_node_membership(network, infile, coords, label_names, parc, parcel_list, perc_overlap=0.75, error=2):
    """

    :param network:
    :param infile:
    :param coords:
    :param label_names:
    :param parc:
    :param parcel_list:
    :param perc_overlap:
    :param error:
    :return:
    """
    from nilearn.image import resample_img
    from pynets.nodemaker import get_sphere
    import pkg_resources
    import pandas as pd

    # Determine whether input is from 17-networks or 7-networks
    seven_nets = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    seventeen_nets = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA',
                      'SalVentAttnB', 'LimbicOFC', 'LimbicTempPole', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB',
                      'DefaultC', 'TempPar']

    # Load subject func data
    bna_img = nib.load(infile)
    x_vox = np.diagonal(bna_img.affine[:3,0:3])[0]
    y_vox = np.diagonal(bna_img.affine[:3,0:3])[1]
    z_vox = np.diagonal(bna_img.affine[:3,0:3])[2]

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
    par_data = par_img.get_fdata()
    RSN_ix = list(ref_dict.keys())[list(ref_dict.values()).index(network)]
    RSNmask = par_data[:, :, :, RSN_ix]

    def mmToVox(nib_nifti, mmcoords):
        """

        :param nib_nifti:
        :param mmcoords:
        :return:
        """
        return nib.affines.apply_affine(np.linalg.inv(nib_nifti.affine), mmcoords)

    def VoxTomm(nib_nifti, voxcoords):
        """

        :param nib_nifti:
        :param voxcoords:
        :return:
        """
        return nib.affines.apply_affine(nib_nifti.affine, voxcoords)

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(bna_img, i))
    coords_vox = list(tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x)) for x in coords_vox)
    if parc is False:
        i = -1
        RSN_parcels = None
        RSN_coords_vox = []
        net_label_names = []
        for coords in coords_vox:
            sphere_vol = np.zeros(RSNmask.shape, dtype=bool)
            sphere_vol[tuple(coords)] = 1
            i = i + 1
            if (RSNmask.astype('bool') & sphere_vol).any():
                print("%s%s%s%s" % (coords, ' coords falls within ', network, '...'))
                RSN_coords_vox.append(coords)
                net_label_names.append(label_names[i])
                continue
            else:
                inds = get_sphere(coords, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape)
                sphere_vol[tuple(inds.T)] = 1
                if (RSNmask.astype('bool') & sphere_vol).any():
                    print("%s%s%.2f%s%s%s" % (coords, ' coords is within a + or - ', float(error), ' mm neighborhood of ',
                                              network, '...'))
                    RSN_coords_vox.append(coords)
                    net_label_names.append(label_names[i])

        coords_mm = []
        for i in RSN_coords_vox:
            coords_mm.append(VoxTomm(bna_img, i))
        coords_mm = list(set(list(tuple(x) for x in coords_mm)))
    else:
        i = 0
        RSN_parcels = []
        coords_with_parc = []
        net_label_names = []
        for parcel in parcel_list:
            parcel_vol = np.zeros(RSNmask.shape, dtype=bool)
            parcel_data_reshaped = resample_img(parcel, target_affine=par_img.affine, target_shape=RSNmask.shape).get_fdata()
            parcel_vol[parcel_data_reshaped == 1] = 1
            # Count number of unique voxels where overlap of parcel and mask occurs
            overlap_count = len(np.unique(np.where((RSNmask.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))
            # Count number of total unique voxels within the parcel
            total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))
            # Calculate % overlap
            try:
                overlap = float(overlap_count/total_count)
            except RuntimeWarning:
                print('\nWarning: No overlap with roi mask!\n')
                overlap = float(0)

            if overlap >= perc_overlap:
                print("%.2f%s%s%s%s%s" % (100*overlap, '% of parcel ', label_names[i], ' falls within ', str(network),
                                          ' mask...'))
                RSN_parcels.append(parcel)
                coords_with_parc.append(coords[i])
                net_label_names.append(label_names[i])
            i = i + 1
        coords_mm = list(set(list(tuple(x) for x in coords_with_parc)))

    bna_img.uncache()
    if len(coords_mm) <= 1:
        raise ValueError("%s%s%s" % ('\nERROR: No coords from the specified atlas found within ', network, ' network.'))

    return coords_mm, RSN_parcels, net_label_names, network


def parcel_masker(roi, coords, parcel_list, label_names, dir_path, ID, perc_overlap):
    """

    :param roi:
    :param coords:
    :param parcel_list:
    :param label_names:
    :param dir_path:
    :param ID:
    :param perc_overlap:
    :return:
    """
    from pynets import nodemaker
    from nilearn.image import resample_img
    from nilearn import masking
    import os.path as op

    mask_img = nib.load(roi)
    mask_data, _ = masking._load_mask_img(roi)

    i = 0
    indices = []
    for parcel in parcel_list:
        parcel_vol = np.zeros(mask_data.shape, dtype=bool)
        parcel_data_reshaped = resample_img(parcel, target_affine=mask_img.affine,
                                            target_shape=mask_data.shape).get_fdata()
        parcel_vol[parcel_data_reshaped == 1] = 1
        # Count number of unique voxels where overlap of parcel and mask occurs
        overlap_count = len(np.unique(np.where((mask_data.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))
        # Count number of total unique voxels within the parcel
        total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))
        # Calculate % overlap
        try:
            overlap = float(overlap_count/total_count)
        except RuntimeWarning:
            print('\nWarning: No overlap with roi mask!\n')
            overlap = float(0)

        if overlap >= perc_overlap:
            print("%.2f%s%s%s" % (100*overlap, '% of parcel ', label_names[i], ' falls within mask...'))
        else:
            indices.append(i)
        i = i + 1

    label_names_adj = list(label_names)
    coords_adj = list(tuple(x) for x in coords)
    parcel_list_adj = parcel_list
    try:
        for ix in sorted(indices, reverse=True):
            print("%s%s%s%s" % ('Removing: ', label_names_adj[ix], ' at ', coords_adj[ix]))
            label_names_adj.pop(ix)
            coords_adj.pop(ix)
            parcel_list_adj.pop(ix)
    except RuntimeError:
        print('ERROR: Restrictive masking. No parcels remain after masking with brain mask/roi...')

    # Create a resampled 3D atlas that can be viewed alongside mask img for QA
    resampled_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_parcels_resampled2roimask_',
                                                   op.basename(roi).split('.')[0], '.nii.gz')
    resampled_parcels_atlas, _ = nodemaker.create_parcel_atlas(parcel_list_adj)
    resampled_parcels_map_nifti = resample_img(resampled_parcels_atlas, target_affine=mask_img.affine,
                                               target_shape=mask_data.shape)
    nib.save(resampled_parcels_map_nifti, resampled_parcels_nii_path)
    mask_img.uncache()
    resampled_parcels_map_nifti.uncache()
    if not coords_adj:
        raise ValueError('\nERROR: ROI mask was likely too restrictive and yielded < 2 remaining parcels')

    return coords_adj, label_names_adj, parcel_list_adj


def coords_masker(roi, coords, label_names, error):
    """

    :param roi:
    :param coords:
    :param label_names:
    :param error:
    :return:
    """
    from nilearn import masking

    mask_data, mask_aff = masking._load_mask_img(roi)
    x_vox = np.diagonal(mask_aff[:3,0:3])[0]
    y_vox = np.diagonal(mask_aff[:3,0:3])[1]
    z_vox = np.diagonal(mask_aff[:3,0:3])[2]

    def mmToVox(mask_aff, mmcoords):
        """

        :param mask_aff:
        :param mmcoords:
        :return:
        """
        return nib.affines.apply_affine(np.linalg.inv(mask_aff), mmcoords)

#    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_aff, i))
    coords_vox = list(tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x)) for x in coords_vox)
    bad_coords = []
    for coords in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coords)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s" % (coords, ' falls within mask...'))
            continue
        inds = get_sphere(coords, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape)
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s%.2f%s" % (coords, ' is within a + or - ', float(error), ' mm neighborhood...'))
            continue
        bad_coords.append(coords)

    bad_coords = [x for x in bad_coords if x is not None]
    indices = []
    for bad_coords in bad_coords:
        indices.append(coords_vox.index(bad_coords))

    label_names = list(label_names)
    coords = list(tuple(x) for x in coords_vox)
    try:
        for ix in sorted(indices, reverse=True):
            print("%s%s%s%s" % ('Removing: ', label_names[ix], ' at ', coords[ix]))
            label_names.pop(ix)
            coords.pop(ix)
    except RuntimeError:
        print('ERROR: Restrictive masking. No coords remain after masking with brain mask/roi...')

    if len(coords) <= 1:
        raise ValueError('\nERROR: ROI mask was likely too restrictive and yielded < 2 remaining coords')

    return coords, label_names


def get_names_and_coords_of_parcels(uatlas_select):
    """

    :param uatlas_select:
    :return:
    """
    import os.path as op
    from nilearn.plotting import find_parcellation_cut_coords
    if not op.isfile(uatlas_select):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    atlas_select = uatlas_select.split('/')[-1].split('.')[0]
    [coords, label_intensities] = find_parcellation_cut_coords(uatlas_select, return_label_names=True)
    print("%s%s" % ('Region intensities:\n', label_intensities))
    par_max = len(coords)
    return coords, atlas_select, par_max


def gen_img_list(uatlas_select):
    """

    :param uatlas_select:
    :return:
    """
    import os.path as op
    from nilearn.image import new_img_like
    if not op.isfile(uatlas_select):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    bna_img = nib.load(uatlas_select)
    bna_data = np.round(bna_img.get_fdata(), 1)
    # Get an array of unique parcels
    bna_data_for_coords_uniq = np.unique(bna_data)
    # Number of parcels:
    par_max = len(bna_data_for_coords_uniq) - 1
    bna_data = bna_data.astype('int16')
    img_stack = []
    for idx in range(1, par_max+1):
        roi_img = bna_data == bna_data_for_coords_uniq[idx].astype('int16')
        roi_img = roi_img.astype('int16')
        img_stack.append(roi_img)
    img_stack = np.array(img_stack).astype('int16')
    img_list = []
    for idy in range(par_max):
        roi_img_nifti = new_img_like(bna_img, img_stack[idy])
        img_list.append(roi_img_nifti)
    bna_img.uncache()
    del img_stack

    return img_list


def gen_network_parcels(uatlas_select, network, labels, dir_path):
    """

    :param uatlas_select:
    :param network:
    :param labels:
    :param dir_path:
    :return:
    """
    from nilearn.image import concat_imgs
    from pynets import nodemaker
    import os.path as op

    if not op.isfile(uatlas_select):
        raise ValueError('\nERROR: User-specified atlas input not found! Check that the file(s) specified with the -ua '
                         'flag exist(s)')

    img_list = nodemaker.gen_img_list(uatlas_select)
    print("%s%s%s" % ('\nExtracting parcels associated with ', network, ' network locations...\n'))
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = concat_imgs(net_parcels).get_fdata()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = "%s%s%s%s" % (dir_path, '/', network, '_parcels.nii.gz')
    nib.save(net_parcels_map_nifti, out_path)
    net_parcels_map_nifti.uncache()
    return out_path


def AAL_naming(coords):
    """

    :param coords:
    :return:
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

    label_names_ix = []
    print('Building region index using AAL MNI coords...')
    for coords in coords:
        reg_lab = aal_coords_ix.loc[aal_coords_ix['coords_tuple'] == str(tuple(np.round(coords).astype('int'))),
                                    'Region_index']
        if len(reg_lab) > 0:
            label_names_ix.append(reg_lab.values[0])
        else:
            label_names_ix.append(np.nan)

    print('Building list of label names using AAL dictionary...')
    label_names = []
    for region_ix in label_names_ix:
        if region_ix is np.nan:
            label_names.append('Unknown')
        else:
            label_names.append(aal_labs_dict[str(region_ix)])

    return label_names


def fetch_nodes_and_labels(atlas_select, uatlas_select, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    """

    :param atlas_select:
    :param uatlas_select:
    :param ref_txt:
    :param parc:
    :param in_file:
    :param use_AAL_naming:
    :param clustering:
    :return:
    """
    from pynets import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009',
                            'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas_select is None and atlas_select in nilearn_parc_atlases:
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
    elif uatlas_select is None and parc is False and atlas_select in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    elif uatlas_select is None and parc is False and atlas_select in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas_select)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
        par_max = None
    elif uatlas_select:
        if clustering is True:
            while True:
                if op.isfile(uatlas_select):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas_select = uatlas_select.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas_select, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas_select, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or you have not '
                  'supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        label_names = None
        networks_list = None
    else:
        networks_list = None
        label_names = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas_select:
        if label_names:
            pass
        else:
            if ref_txt is not None and op.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas_select, '.txt')
                    if op.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            label_names = dict_df['Region'].tolist()
                            #print(label_names)
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. "
                                  "Attempting AAL naming...")
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            label_names = nodemaker.AAL_naming(coords)
                            #print(label_names)
                        except:
                            print('AAL reference labeling failed!')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', label_names))
    atlas_name = atlas_select
    dir_path = utils.do_dir_path(atlas_select, in_file)

    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, uatlas_select, dir_path


def node_gen_masking(roi, coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select, mask,
                     perc_overlap=0.75, error=4):
    """

    :param roi:
    :param coords:
    :param parcel_list:
    :param label_names:
    :param dir_path:
    :param ID:
    :param parc:
    :param atlas_select:
    :param uatlas_select:
    :param mask:
    :param perc_overlap:
    :param error:
    :return:
    """
    from pynets import nodemaker
    import os.path as op
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    # Mask Parcels
    if parc is True:
        # For parcel masking, specify overlap thresh and error cushion in mm voxels
        [coords, label_names, parcel_list_masked] = nodemaker.parcel_masker(roi, coords, parcel_list, label_names,
                                                                            dir_path, ID, perc_overlap)
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list_masked)
    # Mask Coordinates
    else:
        [coords, label_names] = nodemaker.coords_masker(roi, coords, label_names, error)
        # Save coords to pickle
        coords_path = "%s%s%s%s" % (dir_path, '/atlas_coords_', op.basename(roi).split('.')[0], '.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        net_parcels_map_nifti = None
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/atlas_labelnames_', op.basename(roi).split('.')[0], '.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select


def node_gen(coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select):
    """

    :param coords:
    :param parcel_list:
    :param label_names:
    :param dir_path:
    :param ID:
    :param parc:
    :param atlas_select:
    :param uatlas_select:
    :return:
    """
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
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
            pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select


def mask_roi(dir_path, roi, mask, img_file):
    """

    :param dir_path:
    :param roi:
    :param mask:
    :param img_file:
    :return:
    """
    import os
    import os.path as op
    from nilearn import masking

    img_mask_path = "%s%s%s%s" % (dir_path, '/', op.basename(img_file).split('.')[0], '_mask.nii.gz')
    img_mask = masking.compute_epi_mask(img_file)
    nib.save(img_mask, img_mask_path)

    if roi and mask:
        print('Refining ROI...')
        roi_red_path = "%s%s%s%s" % (dir_path, '/', op.basename(roi).split('.')[0], '_mask.nii.gz')
        cmd = 'fslmaths ' + roi + ' -mas ' + mask + ' -mas ' + img_mask_path + ' -bin ' + roi_red_path
        os.system(cmd)
        roi = roi_red_path
    return roi


def create_spherical_roi_volumes(node_size, coords, template_mask):
    """

    :param node_size:
    :param coords:
    :param template_mask:
    :return:
    """
    from pynets.nodemaker import get_sphere
    mask_img = nib.load(template_mask)
    mask_aff = mask_img.affine

    print("%s%s" % ('Creating spherical ROI atlas with radius: ', node_size))

    def mmToVox(nib_nifti, mmcoords):
        """

        :param nib_nifti:
        :param mmcoords:
        :return:
        """
        return nib.affines.apply_affine(np.linalg.inv(nib_nifti.affine), mmcoords)

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_img, i))
    coords_vox = list(set(list(tuple(x) for x in coords_vox)))

    x_vox = np.diagonal(mask_img.affine[:3,0:3])[0]
    y_vox = np.diagonal(mask_img.affine[:3,0:3])[1]
    z_vox = np.diagonal(mask_img.affine[:3,0:3])[2]
    sphere_vol = np.zeros(mask_img.shape, dtype=bool)
    parcel_list = []
    i = 0
    for coord in coords_vox:
        inds = get_sphere(coord, node_size, (np.abs(x_vox), y_vox, z_vox), mask_img.shape)
        sphere_vol[tuple(inds.T)] = i*1
        parcel_list.append(nib.Nifti1Image(sphere_vol.astype('int'), affine=mask_aff))
        i = i + 1

    par_max = len(coords)
    parc = True
    return parcel_list, par_max, node_size, parc
