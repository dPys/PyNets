# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import numpy as np
import nibabel as nib


def get_sphere(coords, r, vox_dims, dims):
    """##Adapted from Neurosynth
    Return all points within r mm of coordinates. Generates a cube
    and then discards all points outside sphere. Only returns values that
    fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[
                        i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(
        vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) &
                  (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)


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
    print("%s%s%s%s" % ('\n', str(atlas_name.decode('utf-8')), ' comes with {0}'.format(atlas.keys()), '\n'))
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    print("%s%s" % ('\nStacked atlas coordinates in array of shape {0}.'.format(coords.shape), '\n'))
    try:
        networks_list = atlas.networks.astype('U').tolist()
    except:
        networks_list = None
    try:
        label_names=np.array([s.strip('b\'') for s in atlas.labels.astype('U')]).tolist()
    except:
        label_names=None
    return coords, atlas_name, networks_list, label_names


def get_node_membership(network, func_file, coords, label_names, parc, parcel_list):
    from nilearn.image import resample_img
    from pynets.nodemaker import get_sphere
    import pkg_resources
    import pandas as pd
    ##For parcel membership determination, specify overlap thresh and error cushion in mm voxels
    perc_overlap = 0.75 ##Default is >=90% overlap
    error = 2

    ##Load subject func data
    bna_img = nib.load(func_file)

    x_vox = np.diagonal(bna_img.affine[:3,0:3])[0]
    y_vox = np.diagonal(bna_img.affine[:3,0:3])[1]
    z_vox = np.diagonal(bna_img.affine[:3,0:3])[2]

    ##Determine whether input is from 17-networks or 7-networks
    seven_nets = [ 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default' ]
    seventeen_nets = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB', 'LimbicOFC', 'LimbicTempPole', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar']

    if network in seventeen_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <=1:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF1mm.nii.gz")
        else:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF2mm.nii.gz")

        ##Grab RSN reference file
        nets_ref_txt = pkg_resources.resource_filename("pynets", "rsnrefs/Schaefer2018_1000_17nets_ref.txt")
    elif network in seven_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <=1:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/SMALLREF1mm.nii.gz")
        else:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/SMALLREF2mm.nii.gz")

        ##Grab RSN reference file
        nets_ref_txt = pkg_resources.resource_filename("pynets", "rsnrefs/Schaefer2018_1000_7nets_ref.txt")
    else:
        nets_ref_txt = None

    if not nets_ref_txt:
        raise ValueError('Network: ' + str(network) + ' not found!\nSee valid network names using the --help flag with pynets_run.py')

    ##Create membership dictionary
    dict_df = pd.read_csv(nets_ref_txt, sep="\t", header=None, names=["Index", "Region", "X", "Y", "Z"])
    dict_df.Region.unique().tolist()
    ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}

    par_img = nib.load(par_file)
    par_data = par_img.get_data()

    RSN_ix=list(ref_dict.keys())[list(ref_dict.values()).index(network)]
    RSNmask = par_data[:,:,:,RSN_ix]

    def mmToVox(mmcoords):
        voxcoords = ['','','']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    def VoxTomm(voxcoords):
        mmcoords = ['','','']
        mmcoords[0] = int((round(int(voxcoords[0])-45)*x_vox))
        mmcoords[1] = int((round(int(voxcoords[1])-63)*y_vox))
        mmcoords[2] = int((round(int(voxcoords[2])-36)*z_vox))
        return mmcoords

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords_vox = list(tuple(x) for x in coords_vox)

    if parc == False:
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
            parcel_data_reshaped = resample_img(parcel, target_affine=par_img.affine,
                               target_shape=RSNmask.shape).get_data()
            parcel_vol[parcel_data_reshaped==1] = 1

            ##Count number of unique voxels where overlap of parcel and mask occurs
            overlap_count = len(np.unique(np.where((RSNmask.astype('uint8')==1) & (parcel_vol.astype('uint8')==1))))

            ##Count number of total unique voxels within the parcel
            total_count = len(np.unique(np.where(((parcel_vol.astype('uint8')==1)))))

            ##Calculate % overlap
            try:
                overlap = float(overlap_count/total_count)
            except:
                overlap = float(0)

            if overlap >=perc_overlap:
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
        parcel_data_reshaped = resample_img(parcel, target_affine=mask_img.affine,
                               target_shape=mask_data.shape).get_data()
        parcel_vol[parcel_data_reshaped==1] = 1

        ##Count number of unique voxels where overlap of parcel and mask occurs
        overlap_count = len(np.unique(np.where((mask_data.astype('uint8')==1) & (parcel_vol.astype('uint8')==1))))

        ##Count number of total unique voxels within the parcel
        total_count = len(np.unique(np.where(((parcel_vol.astype('uint8')==1)))))

        ##Calculate % overlap
        try:
            overlap = float(overlap_count/total_count)
        except:
            overlap = float(0)

        if overlap >= perc_overlap:
            print("%.2f%s%s%s" % (100*overlap, '% of parcel ', label_names[i], ' falls within mask...'))
        else:
            indices.append(i)
        i = i + 1

    label_names_adj=list(label_names)
    coords_adj = list(tuple(x) for x in coords)
    parcel_list_adj = parcel_list
    for ix in sorted(indices, reverse=True):
        print("%s%s%s%s" % ('Removing: ', label_names_adj[ix], ' at ', coords_adj[ix]))
        label_names_adj.pop(ix)
        coords_adj.pop(ix)
        parcel_list_adj.pop(ix)

    ##Create a resampled 3D atlas that can be viewed alongside mask img for QA
    resampled_parcels_nii_path = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_parcels_resampled2mask_', os.path.basename(mask).split('.')[0], '.nii.gz')
    resampled_parcels_atlas, _ = nodemaker.create_parcel_atlas(parcel_list_adj)
    resampled_parcels_map_nifti = resample_img(resampled_parcels_atlas, target_affine=mask_img.affine, target_shape=mask_data.shape)
    nib.save(resampled_parcels_map_nifti, resampled_parcels_nii_path)
    return(coords_adj, label_names_adj, parcel_list_adj)


def coord_masker(mask, coords, label_names, error):
    from nilearn import masking
    x_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[0]
    y_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[1]
    z_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[2]
    def mmToVox(mmcoords):
        voxcoords = ['','','']
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
    indices=[]
    for bad_coord in bad_coords:
        indices.append(coords_vox.index(bad_coord))

    label_names=list(label_names)
    coords = list(tuple(x) for x in coords)
    for ix in sorted(indices, reverse=True):
        print("%s%s%s%s" % ('Removing: ', label_names[ix], ' at ', coords[ix]))
        label_names.pop(ix)
        coords.pop(ix)
    return coords, label_names


def get_names_and_coords_of_parcels_from_img(bna_img):
    from nilearn.plotting import find_xyz_cut_coords
    from nilearn.image import new_img_like
    bna_data = np.round(bna_img.get_data(),1)
    ##Get an array of unique parcels
    bna_data_for_coords_uniq = np.unique(bna_data)
    ##Number of parcels:
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
    coords = []
    for roiin in img_list:
        coord = find_xyz_cut_coords(roiin)
        coords.append(coord)
    coords = list(tuple(x) for x in np.array(coords))
    return coords, par_max, img_list


def get_names_and_coords_of_parcels(parlistfile):
    from pynets import nodemaker
    try:
        atlas_select = parlistfile.split('/')[-1].split('.')[0]
    except:
        atlas_select = 'User_specified_atlas'
    bna_img = nib.load(parlistfile)
    [coords, par_max, img_list] = nodemaker.get_names_and_coords_of_parcels_from_img(bna_img)
    return coords, atlas_select, par_max, img_list


def gen_network_parcels(parlistfile, network, labels, dir_path):
    from nilearn.image import new_img_like, concat_imgs
    bna_img = nib.load(parlistfile)
    bna_data = np.round(bna_img.get_data(),1)
    ##Get an array of unique parcels
    bna_data_for_coords_uniq = np.unique(bna_data)
    ##Number of parcels:
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
    print("%s%s%s" % ('\nExtracting parcels associated with ', network, ' network locations...\n'))
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = concat_imgs(net_parcels).get_data()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = "%s%s%s%s" % (dir_path, '/', network, '_parcels.nii.gz')
    nib.save(net_parcels_map_nifti, out_path)
    return out_path


def WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file):
    from pynets import utils, nodemaker
    import pandas as pd
    from pathlib import Path
    ##Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    nilearn_coord_atlases=['harvard_oxford', 'msdl', 'coords_power_2011', 'smith_2009', 'basc_multiscale_2015', 'allen_2011', 'coords_dosenbach_2010']
    if atlas_select in nilearn_parc_atlases:
        [label_names, networks_list, parlistfile] = utils.nilearn_atlas_helper(atlas_select)

    ##Get coordinates and/or parcels from atlas
    if parlistfile is None and parc == False and atlas_select in nilearn_coord_atlases:
        print('Fetching coordinates and labels from nilearn coordinate-based atlases')
        ##Fetch nilearn atlas coords
        [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    else:
        try:
            ##Fetch user-specified atlas coords
            [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
            networks_list = None
            ##Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas_select, ' comes with {0} '.format(par_max), 'parcels\n'))
        except:
            raise ValueError('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or you have not supplied a 3d atlas parcellation image!\n\n')

    ##Labels prep
    try:
        label_names
    except:
        if ref_txt is not None and os.path.exists(ref_txt):
            atlas_select = os.path.basename(ref_txt).split('.txt')[0]
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            label_names = dict_df['Region'].tolist()
        else:
            try:
                atlas_ref_txt = atlas_select + '.txt'
                ref_txt = Path(__file__)/'atlases'/atlas_ref_txt
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            except:
                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    if label_names is None:
        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    try:
        atlas_name
    except:
        atlas_name = atlas_select

    dir_path = utils.do_dir_path(atlas_select, func_file)
    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path


def RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc, func_file):
    from pynets import utils, nodemaker
    import pandas as pd
    ##Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_atlases=['atlas_aal', 'atlas_craddock_2012', 'atlas_destrieux_2009']
    if atlas_select in nilearn_atlases:
        [label_names, networks_list, parlistfile] = utils.nilearn_atlas_helper(atlas_select)

    ##Get coordinates and/or parcels from atlas
    if parlistfile is None and parc == False:
        print('Fetching coordinates and labels from nilearn coordinate-based atlases')
        ##Fetch nilearn atlas coords
        [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    else:
        ##Fetch user-specified atlas coords
        [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
        networks_list = None

    ##Labels prep
    try:
        label_names
    except:
        if ref_txt is not None and os.path.exists(ref_txt):
            atlas_select = os.path.basename(ref_txt).split('.txt')[0]
            dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
            label_names = dict_df['Region'].tolist()
        else:
            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    if label_names is None:
        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    try:
        atlas_name
    except:
        atlas_name = atlas_select

    dir_path = utils.do_dir_path(atlas_select, func_file)
    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile, dir_path


def node_gen_masking(mask, coords, parcel_list, label_names, dir_path, ID, parc):
    from pynets import nodemaker
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    ##Mask Parcels
    if parc is True:
        ##For parcel masking, specify overlap thresh and error cushion in mm voxels
        if 'bedpostX' in dir_path:
            perc_overlap = 0.01
        else:
            perc_overlap = 0.75
        [coords, label_names, parcel_list_masked] = nodemaker.parcel_masker(mask, coords, parcel_list, label_names, dir_path, ID, perc_overlap)
        [net_parcels_map_nifti, parcel_list_adj] = nodemaker.create_parcel_atlas(parcel_list_masked)
    ##Mask Coordinates
    elif parc is False:
        if 'bedpostX' in dir_path:
            error = 10
        else:
            error = 2
        [coords, label_names] = nodemaker.coord_masker(mask, coords, label_names, error)
        ##Save coords to pickle
        coord_path = "%s%s%s%s" % (dir_path, '/whole_brain_atlas_coords_', os.path.basename(mask).split('.')[0], '.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        net_parcels_map_nifti = None
    ##Save labels to pickle
    labels_path = "%s%s%s%s" % (dir_path, '/whole_brain_atlas_labelnames_', os.path.basename(mask).split('.')[0], '.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(label_names, f, protocol=2)
    return net_parcels_map_nifti, coords, label_names


def node_gen(coords, parcel_list, label_names, dir_path, ID, parc):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, parcel_list_adj] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional masking...')

    if pick_dump is True:
        ##Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/whole_brain_atlas_coords_wb.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        ##Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/whole_brain_atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)
    return net_parcels_map_nifti, coords, label_names
