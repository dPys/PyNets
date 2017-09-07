import sys
import argparse
import os
import nilearn
import numpy as np
import networkx as nx
import pandas as pd
import nibabel as nib
import seaborn as sns
import numpy.linalg as npl
import matplotlib
import sklearn
import matplotlib
import warnings
import pkg_resources
import pynets
import itertools
import multiprocessing
#warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib import colors
from nipype import Node, Workflow
from nilearn import input_data, masking, datasets
from nilearn import plotting as niplot
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nibabel.affines import apply_affine
from nipype.interfaces.base import isdefined, Undefined
from sklearn.covariance import GraphLassoCV, ShrunkCovariance, graph_lasso
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits

##Core node definition, graph estimation, and plotting functions
def check_neighborhood(coord, mask_coords):
    if coord not in mask_coords:
        error=4
        neighbors=[]
        ##Check range in case it's close by
        x=coord[0]
        y=coord[1]
        z=coord[2]
        x_min=x-error
        x_max=x+error
        y_min=y-error
        y_max=y+error
        z_min=z-error
        z_max=z+error
        x_range = list(range(x_min, x_max, 1))
        y_range = list(range(y_min, y_max, 1))
        z_range = list(range(z_min, z_max, 1))
        ##Check range in case it's close by
        tuple_range = list(itertools.product(x_range, y_range, z_range))
        for i in tuple_range:
            if tuple(i) in mask_coords:
                neighbors.append(i)
                print(str(coord) + ' is within a + or - ' + str(error) + ' voxel neighborhood...')
                break
        if len(neighbors)==0:
            bad_coord=coord
        elif len(neighbors)>0:
            bad_coord=None
    else:
        bad_coord=None
        print(str(coord) + ' falls within mask...')
    return bad_coord

def fetch_nilearn_atlas_coords(atlas_select):
    atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
    atlas_name = atlas['description'].splitlines()[0]
    print('\n' + atlas_name + ' comes with {0}'.format(atlas.keys()) + '\n')
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
    return(coords, atlas_name, networks_list, label_names)

def get_ref_net(bna_img, par_data, x, y, z):
    ref_dict = {0:'UNKNOWN', 1:'VIS', 2:'SENS', 3:'DA', 4:'VA', 5:'LIMBIC', 6:'FPTC', 7:'DMN'}
    ##apply_affine(aff, (x,y,z)) # vox to mni
    aff_inv=npl.inv(bna_img.affine)
    ##mni to vox
    vox_coord = apply_affine(aff_inv, (x, y, z))
    return ref_dict[int(par_data[int(vox_coord[0]),int(vox_coord[1]),int(vox_coord[2])])]

def get_mem_dict(func_file, coords, networks_list):
    bna_img = nib.load(func_file)
    if networks_list is None:
        #net_atlas = nilearn.datasets.fetch_atlas_yeo_2011()
        #par_path = net_atlas.thick_7
        par_file = pkg_resources.resource_filename("pynets", "rsnrefs/Yeo7.nii.gz")
        par_img = nib.load(par_file)
        par_data = par_img.get_data()
        membership = pd.Series(list(tuple(x) for x in coords), [get_ref_net(bna_img, par_data, coord[0],coord[1],coord[2]) for coord in coords])
        membership_plotting = pd.Series([get_ref_net(bna_img, par_data, coord[0],coord[1],coord[2]) for coord in coords])
    else:
        membership = pd.Series(list(tuple(x) for x in coords), networks_list)
        membership_plotting = pd.Series(networks_list)
    return(membership, membership_plotting)

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
    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords_vox = list(tuple(x) for x in coords_vox)

    bad_coords = []
    for coord in coords_vox:
        bad_coord = check_neighborhood(coord, mask_coords)
        bad_coords.append(bad_coord)
    bad_coords = [x for x in bad_coords if x is not None]

    #if __name__ == '__main__':
        #number_processes = (multiprocessing.cpu_count())
        #pool = multiprocessing.Pool(int(number_processes))
        #coord = coords_vox
        #result = pool.map_async(check_neighborhood, coord)
        #pool.close()
        #pool.join()
    #bad_coords = [x for x in result.get() if x is not None]

    indices=[]
    for bad_coord in bad_coords:
        indices.append(coords_vox.index(bad_coord))

    for ix in sorted(indices, reverse=True):
        print('Removing: ' + str(label_names[ix]) + ' at ' + str(coords[ix]))
        del label_names[ix]
        del coords[ix]
    return(coords, label_names)

def get_names_and_coords_of_parcels(parlistfile):
    atlas_name = parlistfile.split('/')[-1].split('.')[0]
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
    bna_4D = nilearn.image.concat_imgs(img_list).get_data()
    #nib.Nifti1Image(img_list, affine=np.eye(4))
    coords = []
    for roi_img in img_list:
        coords.append(nilearn.plotting.find_xyz_cut_coords(roi_img))
    coords = np.array(coords)
    return(coords, atlas_name, par_max)

def gen_network_parcels(parlistfile, NETWORK, labels, dir_path):
    bna_img = nib.load(parlistfile)
    if bna_img.get_data_dtype() != np.dtype(np.int):
        bna_data_for_coords = bna_img.get_data()
        # Number of parcels:
        par_max = np.ceil(np.max(bna_data_for_coords)).astype('int')
        bna_data = bna_data.astype('int16')
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
    print('Extracting parcels associated with ' + NETWORK + ' locations...')
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = nilearn.image.concat_imgs(net_parcels).get_data()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = dir_path + '/' + NETWORK + '_parcels.nii.gz'
    nib.save(net_parcels_map_nifti, out_path)
    return(out_path)
