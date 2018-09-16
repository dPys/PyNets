# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import numpy as np
import nibabel as nib
nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = 'auto'


def get_sphere(coords, r, vox_dims, dims):
    # Adapted from Neurosynth
    # Return all points within r mm of coordinates. Generates a cube and then discards all points outside sphere. Only returns values that fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)


def create_parcel_atlas(parcel_list):
    from nilearn.image import new_img_like, concat_imgs
    parcel_background = new_img_like(parcel_list[0], np.zeros(parcel_list[0].shape, dtype=bool))
    parcel_list_exp = [parcel_background] + parcel_list
    parcellation = concat_imgs(parcel_list_exp).get_data()
    index_vec = np.array(range(len(parcel_list_exp))) + 1
    net_parcels_sum = np.sum(index_vec * parcellation, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=parcel_list[0].affine)
    return net_parcels_map_nifti, parcel_list_exp


def fetch_nilearn_atlas_coords(atlas_select):
    from nilearn import datasets
    atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
    atlas_name = atlas['description'].splitlines()[0]
    if atlas_name is None:
        atlas_name = atlas_select
    if "b'" in str(atlas_name):
        atlas_name = atlas_name.decode('utf-8')
    print("%s%s%s%s" % ('\n', atlas_name, ' comes with {0}'.format(atlas.keys()), '\n'))
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    print("%s%s" % ('\nStacked atlas coordinates in array of shape {0}.'.format(coords.shape), '\n'))
    try:
        networks_list = atlas.networks.astype('U').tolist()
    except:
        networks_list = None
    try:
        label_names = np.array([s.strip('b\'') for s in atlas.labels.astype('U')]).tolist()
    except:
        label_names = None
    return coords, atlas_name, networks_list, label_names


def nilearn_atlas_helper(atlas_select, parc):
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


def get_node_membership(network, func_file, coords, label_names, parc, parcel_list):
    from nilearn.image import resample_img
    from pynets.nodemaker import get_sphere
    import pkg_resources
    import pandas as pd
    # For parcel membership determination, specify overlap thresh and error cushion in mm voxels
    perc_overlap = 0.75 # Default is >=90% overlap
    error = 2

    # Load subject func data
    bna_img = nib.load(func_file)

    x_vox = np.diagonal(bna_img.affine[:3,0:3])[0]
    y_vox = np.diagonal(bna_img.affine[:3,0:3])[1]
    z_vox = np.diagonal(bna_img.affine[:3,0:3])[2]

    # Determine whether input is from 17-networks or 7-networks
    seven_nets = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    seventeen_nets = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB', 'LimbicOFC', 'LimbicTempPole', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar']

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
        raise ValueError("%s%s%s" % ('Network: ', str(network), ' not found!\nSee valid network names using the --help flag with pynets_run.py'))

    # Create membership dictionary
    dict_df = pd.read_csv(nets_ref_txt, sep="\t", header=None, names=["Index", "Region", "X", "Y", "Z"])
    dict_df.Region.unique().tolist()
    ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}

    par_img = nib.load(par_file)
    par_data = par_img.get_data()

    RSN_ix = list(ref_dict.keys())[list(ref_dict.values()).index(network)]
    RSNmask = par_data[:, :, :, RSN_ix]

    def mmToVox(mmcoords):
        voxcoords = ['', '', '']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    def VoxTomm(voxcoords):
        mmcoords = ['', '', '']
        mmcoords[0] = int((round(int(voxcoords[0])-45)*x_vox))
        mmcoords[1] = int((round(int(voxcoords[1])-63)*y_vox))
        mmcoords[2] = int((round(int(voxcoords[2])-36)*z_vox))
        return mmcoords

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords_vox = list(tuple(x) for x in coords_vox)

    if parc is False:
        i = -1
        RSN_parcels = None
        RSN_coords_vox = []
        net_label_names = []
        for coord in coords_vox:
            sphere_vol = np.zeros(RSNmask.shape, dtype=bool)
            sphere_vol[tuple(coord)] = 1
            i = i + 1
            if (RSNmask.astype('bool') & sphere_vol).any():
                print("%s%s%s%s" % (coord, ' coord falls within ', network, '...'))
                RSN_coords_vox.append(coord)
                net_label_names.append(label_names[i])
                continue
            else:
                inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape)
                sphere_vol[tuple(inds.T)] = 1
                if (RSNmask.astype('bool') & sphere_vol).any():
                    print("%s%s%.2f%s%s%s" % (coord, ' coord is within a + or - ', float(error), ' mm neighborhood of ', network, '...'))
                    RSN_coords_vox.append(coord)
                    net_label_names.append(label_names[i])
        coords_mm = []
        for i in RSN_coords_vox:
            coords_mm.append(VoxTomm(i))
        coords_mm = list(set(list(tuple(x) for x in coords_mm)))
    else:
        i = 0
        RSN_parcels = []
        coords_with_parc = []
        net_label_names = []
        for parcel in parcel_list:
            parcel_vol = np.zeros(RSNmask.shape, dtype=bool)
            parcel_data_reshaped = resample_img(parcel, target_affine=par_img.affine, target_shape=RSNmask.shape).get_data()
            parcel_vol[parcel_data_reshaped == 1] = 1

            # Count number of unique voxels where overlap of parcel and mask occurs
            overlap_count = len(np.unique(np.where((RSNmask.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))

            # Count number of total unique voxels within the parcel
            total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))

            # Calculate % overlap
            try:
                overlap = float(overlap_count/total_count)
            except RuntimeWarning:
                print('\nWarning: No overlap with mask!\n')
                overlap = float(0)

            if overlap >= perc_overlap:
                print("%.2f%s%s%s%s%s" % (100*overlap, '% of parcel ', label_names[i], ' falls within ', str(network), ' mask...'))
                RSN_parcels.append(parcel)
                coords_with_parc.append(coords[i])
                net_label_names.append(label_names[i])
            i = i + 1
        coords_mm = list(set(list(tuple(x) for x in coords_with_parc)))

    return coords_mm, RSN_parcels, net_label_names, network


def parcel_masker(mask, coords, parcel_list, label_names, dir_path, ID, perc_overlap):
    from pynets import nodemaker
    from nilearn.image import resample_img
    from nilearn import masking

    mask_img = nib.load(mask)
    mask_data, _ = masking._load_mask_img(mask)

    i = 0
    indices = []
    for parcel in parcel_list:
        parcel_vol = np.zeros(mask_data.shape, dtype=bool)
        parcel_data_reshaped = resample_img(parcel, target_affine=mask_img.affine, target_shape=mask_data.shape).get_data()
        parcel_vol[parcel_data_reshaped == 1] = 1

        # Count number of unique voxels where overlap of parcel and mask occurs
        overlap_count = len(np.unique(np.where((mask_data.astype('uint8') == 1) & (parcel_vol.astype('uint8') == 1))))

        # Count number of total unique voxels within the parcel
        total_count = len(np.unique(np.where((parcel_vol.astype('uint8') == 1))))

        # Calculate % overlap
        try:
            overlap = float(overlap_count/total_count)
        except RuntimeWarning:
            print('\nWarning: No overlap with mask!\n')
            overlap = float(0)

        if overlap >= perc_overlap:
            print("%.2f%s%s%s" % (100*overlap, '% of parcel ', label_names[i], ' falls within mask...'))
        else:
            indices.append(i)
        i = i + 1

    label_names_adj = list(label_names)
    coords_adj = list(tuple(x) for x in coords)
    parcel_list_adj = parcel_list
    for ix in sorted(indices, reverse=True):
        print("%s%s%s%s" % ('Removing: ', label_names_adj[ix], ' at ', coords_adj[ix]))
        label_names_adj.pop(ix)
        coords_adj.pop(ix)
        parcel_list_adj.pop(ix)

    # Create a resampled 3D atlas that can be viewed alongside mask img for QA
    resampled_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_parcels_resampled2mask_', os.path.basename(mask).split('.')[0], '.nii.gz')
    resampled_parcels_atlas, _ = nodemaker.create_parcel_atlas(parcel_list_adj)
    resampled_parcels_map_nifti = resample_img(resampled_parcels_atlas, target_affine=mask_img.affine,
                                               target_shape=mask_data.shape)
    nib.save(resampled_parcels_map_nifti, resampled_parcels_nii_path)
    return coords_adj, label_names_adj, parcel_list_adj


def coord_masker(mask, coords, label_names, error):
    from nilearn import masking
    x_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[0]
    y_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[1]
    z_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[2]

    def mmToVox(mmcoords):
        voxcoords = ['', '', '']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)

        return voxcoords

    mask_data, _ = masking._load_mask_img(mask)
#    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords_vox = list(tuple(x) for x in coords_vox)

    bad_coords = []
    for coord in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coord)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s" % (coord, ' falls within mask...'))
            continue
        inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape)
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print("%s%s%.2f%s" % (coord, ' is within a + or - ', float(error), ' mm neighborhood...'))
            continue
        bad_coords.append(coord)

    bad_coords = [x for x in bad_coords if x is not None]
    indices = []
    for bad_coord in bad_coords:
        indices.append(coords_vox.index(bad_coord))

    label_names = list(label_names)
    coords = list(tuple(x) for x in coords)
    for ix in sorted(indices, reverse=True):
        print("%s%s%s%s" % ('Removing: ', label_names[ix], ' at ', coords[ix]))
        label_names.pop(ix)
        coords.pop(ix)
    return coords, label_names


def get_names_and_coords_of_parcels(uatlas_select):
    from nilearn.plotting import find_parcellation_cut_coords
    atlas_select = uatlas_select.split('/')[-1].split('.')[0]
    [coords, label_intensities] = find_parcellation_cut_coords(uatlas_select, return_label_names=True)
    print("%s%s" % ('Region intensities:\n', label_intensities))
    par_max = len(coords)
    return coords, atlas_select, par_max


def gen_img_list(uatlas_select):
    from nilearn.image import new_img_like
    bna_img = nib.load(uatlas_select)
    bna_data = np.round(bna_img.get_data(), 1)
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
    return img_list


def gen_network_parcels(uatlas_select, network, labels, dir_path):
    from nilearn.image import concat_imgs
    from pynets import nodemaker
    img_list = nodemaker.gen_img_list(uatlas_select)
    print("%s%s%s" % ('\nExtracting parcels associated with ', network, ' network locations...\n'))
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = concat_imgs(net_parcels).get_data()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = "%s%s%s%s" % (dir_path, '/', network, '_parcels.nii.gz')
    nib.save(net_parcels_map_nifti, out_path)
    return out_path


def AAL_naming(coords):
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
    print('Building region index using AAL MNI coordinates...')
    for coord in coords:
        reg_lab = aal_coords_ix.loc[aal_coords_ix['coord_tuple'] == str(tuple(np.round(coord).astype('int'))), 'Region_index']
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


def fetch_nodes_and_labels(atlas_select, uatlas_select, ref_txt, parc, func_file, use_AAL_naming, clustering=False):
    from pynets import utils, nodemaker
    import pandas as pd
    import os
    import time
    from pathlib import Path

    base_path = utils.get_file()

    # Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009',
                            'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coord_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
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
            raise ValueError("%s%s%s" % ('ERROR: Atlas file for ', atlas_select, ' not found!'))
    elif uatlas_select is None and parc is False and atlas_select in nilearn_coord_atlases:
        print('Fetching coordinates and labels from nilearn coordinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    elif uatlas_select is None and parc is False and atlas_select in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coordinates and labels from nilearn probabilistic atlas library...')
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
            raise ValueError("%s%s%s" % ('ERROR: Atlas file for ', atlas_select, ' not found!'))
        par_max = None
    elif uatlas_select:
        if clustering is True:
            while True:
                if os.path.isfile(uatlas_select):
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
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or you have not supplied a 3d atlas parcellation image!\n\n')
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
            if ref_txt is not None and os.path.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas_select, '.txt')
                    if os.path.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            label_names = dict_df['Region'].tolist()
                            #print(label_names)
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. Attempting AAL naming...")
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

    print(label_names)
    atlas_name = atlas_select
    dir_path = utils.do_dir_path(atlas_select, func_file)

    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, uatlas_select, dir_path


def node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select):
    from pynets import nodemaker
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    # Mask Parcels
    if parc is True:
        # For parcel masking, specify overlap thresh and error cushion in mm voxels
        if 'bedpostX' in dir_path:
            perc_overlap = 0.01
        else:
            perc_overlap = 0.75
        [coords, label_names, parcel_list_masked] = nodemaker.parcel_masker(mask, coords, parcel_list, label_names,
                                                                            dir_path, ID, perc_overlap)
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list_masked)
        vox_list = []
        for i in range(len(parcel_list)):
            vox_list.append(np.count_nonzero(parcel_list[i].get_data()))
        vox_array = np.array(vox_list).astype('float64')
    # Mask Coordinates
    else:
        if 'bedpostX' in dir_path:
            error = 60
        else:
            error = 2
        [coords, label_names] = nodemaker.coord_masker(mask, coords, label_names, error)
        # Save coords to pickle
        coord_path = "%s%s%s%s" % (dir_path, '/atlas_coords_', os.path.basename(mask).split('.')[0], '.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        net_parcels_map_nifti = None
        vox_array = None
    # Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/atlas_labelnames_', os.path.basename(mask).split('.')[0], '.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select, vox_array


def node_gen(coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
        vox_list = []
        for i in range(len(parcel_list)):
            vox_list.append(np.count_nonzero(parcel_list[i].get_data()))
        vox_array = np.array(vox_list).astype('float64')
    else:
        net_parcels_map_nifti = None
        vox_array = None
        print('No additional masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select, vox_array
