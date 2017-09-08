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

nets_ref_txt = '/Users/PSYC-dap3463/Applications/PyNets/pynets/rsnrefs/Schaefer2018_1000_17nets_ref.txt'
dict_df = pd.read_csv(nets_ref_txt, sep="\t", header=None, names=["Index", "Region", "X", "Y", "Z"])
dict_df.Region.unique().tolist()
ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}

indices = []
for i in dict_df.Region.unique().tolist():
    indices.append(dict_df.index[dict_df['Region'] == i].tolist())

for i in range(len(indices)):
    for j in indices:
        print(dict_df['Region'][j])

par_file = '/Users/PSYC-dap3463/Downloads/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm.nii.gz'
par_img = nib.load(par_file)
par_data = par_img.get_data()

if par_img.get_data_dtype() != np.dtype(np.int):
    ##Get an array of unique parcels
    par_data_for_coords_uniq = np.round(np.unique(par_data))
    ##Number of parcels:
    par_max = len(par_data_for_coords_uniq) - 1
    par_data = par_data.astype('int16')

img_stack = []
for idx in range(1, par_max+1):
    roi_img = par_data == par_data_for_coords_uniq[idx]
    img_stack.append(roi_img)
img_stack = np.array(img_stack)
img_list = []
for idx in range(par_max):
    roi_img = nilearn.image.new_img_like(par_img, img_stack[idx])
    img_list.append(roi_img)

bna_4D_RSNS = []
for i in indices:
    print('Building network: ' + str(i))
    img_list_RSN = []
    for j in i:
        print('Adding image: ' + str(j) + ' to list for concatenation...')
        img_list_RSN.append(img_list[j])
    print('Concatenating images of network : ' + str(i))

    bna_4D = nib.Nifti1Image(np.sum([val.get_data() for val in img_list_RSN], axis=0), affine=par_img.affine)
    print('Appending 4D network image to list of 17 4D network images...')
    bna_4D_RSNS.append(bna_4D)

all4d = nilearn.image.concat_imgs(bna_4D_RSNS)

nib.save(all4d, '/Users/PSYC-dap3463/Downloads/BIGREF1mm.nii.gz')
