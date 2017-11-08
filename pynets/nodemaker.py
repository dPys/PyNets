# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import nilearn
import numpy as np
import pandas as pd
import nibabel as nib
import warnings
import pkg_resources
warnings.simplefilter("ignore")
from nilearn import datasets, masking
from nilearn.image import resample_img

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
    parcel_background = nilearn.image.new_img_like(parcel_list[0], np.zeros(parcel_list[0].shape, dtype=bool))
    parcel_list_exp = [parcel_background] + parcel_list
    parcellation = nilearn.image.concat_imgs(parcel_list_exp).get_data()
    index_vec = np.array(range(len(parcel_list_exp))) + 1
    net_parcels_sum = np.sum(index_vec * parcellation, axis=3)    
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=parcel_list[0].affine)
    return(net_parcels_map_nifti, parcel_list_exp)

def fetch_nilearn_atlas_coords(atlas_select):
    atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
    atlas_select = atlas['description'].splitlines()[0]
    print('\n' + str(atlas_select) + ' comes with {0}'.format(atlas.keys()) + '\n')
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    print('Stacked atlas coordinates in array of shape {0}.'.format(coords.shape) + '\n')
    try:
        networks_list = atlas.networks.astype('U')
    except:
        networks_list = None
    try:
        label_names=atlas.labels.astype('U')
        label_names=np.array([s.strip('b\'') for s in label_names]).astype('U')
    except:
        label_names=None
    return(coords, atlas_select, networks_list, label_names)

def get_node_membership(network, func_file, coords, label_names, parc, parcel_list):
    ##For parcel membership determination, specify overlap thresh and error cushion in mm voxels
    perc_overlap = 0.75 ##Default is >=75% overlap
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
                print(str(coord) + ' falls within mask...')
                RSN_coords_vox.append(coord)
                net_label_names.append(label_names[i])
                continue
            else:
                inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape)
                sphere_vol[tuple(inds.T)] = 1
                if (RSNmask.astype('bool') & sphere_vol).any():
                    print(str(coord) + ' is within a + or - ' + str(error) + ' mm neighborhood...')
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
            overlap = float(np.sum((RSNmask.astype('uint8') + parcel_vol.astype('uint8'))>1)/np.sum((parcel_vol.astype('uint8'))>=1))
            if overlap >=perc_overlap:
                print(str(round(100*overlap,1)) + '% of parcel ' + str(label_names[i]) + ' falls within ' + str(network) + ' mask...')
                RSN_parcels.append(parcel)
                coords_with_parc.append(coords[i])
                net_label_names.append(label_names[i])
            i = i + 1
        coords_mm = list(set(list(tuple(x) for x in coords_with_parc)))
    return(coords_mm, RSN_parcels, net_label_names)

def parcel_masker(mask, coords, parcel_list, label_names):
    ##For parcel masking, specify overlap thresh and error cushion in mm voxels
    perc_overlap = 0.75 ##Default is >=75% overlap

    mask_img = nib.load(mask)
    mask_data, _ = masking._load_mask_img(mask)

    i = 0
    bad_parcels = []
    for parcel in parcel_list:
        parcel_vol = np.zeros(mask_data.shape, dtype=bool)
        parcel_data_reshaped = resample_img(parcel, target_affine=mask_img.affine,
                               target_shape=mask_data.shape).get_data()
        parcel_vol[parcel_data_reshaped==1] = 1
        overlap = float(np.sum((mask_data.astype('uint8') + parcel_vol.astype('uint8'))>1)/np.sum((parcel_vol.astype('uint8'))>=1))
        if overlap >= perc_overlap:
            print(str(round(100*overlap,1)) + '% of parcel ' + str(label_names[i]) + ' falls within mask...')
        else:
            bad_parcels.append(parcel)
        i = i + 1

    bad_parcels = [x for x in bad_parcels if x is not None]
    indices=[]
    for bad_parcel in bad_parcels:
        indices.append(bad_parcels.index(bad_parcel))

    label_names_adj=list(label_names)
    coords_adj = list(tuple(x) for x in coords)
    parcel_list_adj = parcel_list
    for ix in sorted(indices, reverse=True):
        print('Removing: ' + str(label_names_adj[ix]) + ' at ' + str(coords_adj[ix]))
        label_names_adj.pop(ix)
        coords_adj.pop(ix)
        parcel_list_adj.pop(ix)
    return(coords_adj, label_names_adj, parcel_list_adj)

def coord_masker(mask, coords, label_names):
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
    error=2
    for coord in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coord)] = 1
        if (mask_data & sphere_vol).any():
            print(str(coord) + ' falls within mask...')
            continue
        inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape)
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print(str(coord) + ' is within a + or - ' + str(error) + ' mm neighborhood...')
            continue
        bad_coords.append(coord)

    bad_coords = [x for x in bad_coords if x is not None]
    indices=[]
    for bad_coord in bad_coords:
        indices.append(coords_vox.index(bad_coord))

    label_names=list(label_names)
    coords = list(tuple(x) for x in coords)
    for ix in sorted(indices, reverse=True):
        print('Removing: ' + str(label_names[ix]) + ' at ' + str(coords[ix]))
        label_names.pop(ix)
        coords.pop(ix)
    return(coords, label_names)

def get_names_and_coords_of_parcels(parlistfile):
    atlas_select = parlistfile.split('/')[-1].split('.')[0]
    bna_img = nib.load(parlistfile)
    bna_data = bna_img.get_data()
    if bna_img.get_data_dtype() != np.dtype(np.int):
        ##Get an array of unique parcels
        bna_data_for_coords_uniq = np.round(np.unique(bna_data))
        ##Number of parcels:
        par_max = len(bna_data_for_coords_uniq) - 1
        bna_data = bna_data.astype('int16')
    img_stack = []
    for idx in range(1, par_max+1):
        roi_img = bna_data == bna_data_for_coords_uniq[idx]
        img_stack.append(roi_img)
    img_stack = np.array(img_stack)
    img_list = []
    for idx in range(par_max):
        roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
        img_list.append(roi_img)
    coords = []
    for roi_img in img_list:
        coords.append(nilearn.plotting.find_xyz_cut_coords(roi_img))
    coords = np.array(coords)
    return(coords, atlas_select, par_max, img_list)

def gen_network_parcels(parlistfile, network, labels, dir_path):
    bna_img = nib.load(parlistfile)
    if bna_img.get_data_dtype() != np.dtype(np.int):
        bna_data_for_coords = bna_img.get_data()
        # Number of parcels:
        par_max = np.ceil(np.max(bna_data_for_coords)).astype('int')
    else:
        bna_data = bna_img.get_data()
        par_max = np.max(bna_data)
    img_stack = []
    ##Set indices
    for idx in range(1, par_max+1):
        roi_img = bna_data == idx
        img_stack.append(roi_img)
    img_stack = np.array(img_stack)
    img_list = []
    for idx in range(par_max):
        roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
        img_list.append(roi_img)
    print('Extracting parcels associated with ' + network + ' locations...')
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = nilearn.image.concat_imgs(net_parcels).get_data()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = dir_path + '/' + network + '_parcels.nii.gz'
    nib.save(net_parcels_map_nifti, out_path)
    return(out_path)
