#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017

@author: PSYC-dap3463
"""
import sys
import os
import nibabel as nib
import pandas as pd
from nilearn import datasets
from nilearn.image import concat_imgs
try:
    import cPickle
except ImportError:
    import _pickle as cPickle

def nilearn_atlas_helper(atlas_select):
    try:
        parlistfile=getattr(datasets, 'fetch_%s' % atlas_select)().maps
        try:
            label_names=getattr(datasets, 'fetch_%s' % atlas_select)().labels
        except:
            label_names=None
        try:
            networks_list = getattr(datasets, 'fetch_%s' % atlas_select)().networks
        except:
            networks_list = None
    except:
        print('Extraction from nilearn datasets failed!')
        sys.exit()
    return(label_names, networks_list, parlistfile)

def convert_atlas_to_volumes(atlas_path, img_list):
    atlas_dir = os.path.dirname(atlas_path)
    ref_txt = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '.txt'
    fourd_file = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '_4d.nii.gz'
    if os.path.isfile(ref_txt):
        try:
            all4d = concat_imgs(img_list)
            nib.save(all4d, fourd_file)

            ##Save individual 3D volumes as individual files
            volumes_dir = atlas_dir + '/' + atlas_path.split('/')[-1:][0].split('.')[0] + '_volumes'
            if not os.path.exists(volumes_dir):
                os.makedirs(volumes_dir)

            j = 0
            for img in img_list:
                volume_path = volumes_dir + '/parcel_' + str(j)
                nib.save(img, volume_path)
                j = j + 1

        except:
            print('Image concatenation failed for: ' + str(atlas_path))
            volumes_dir = None
    else:
        print('Atlas reference file not found!')
        volumes_dir = None
    return(volumes_dir)

##save net metric files to pandas dataframes interface
def export_to_pandas(csv_loc, ID, network, mask, out_file=None):
    if mask != None:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network + '_' + str(os.path.basename(mask).split('.')[0])
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_WB' + '_' + str(os.path.basename(mask).split('.')[0])
    else:
        if network != None:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_' + network
        else:
            met_list_picke_path = os.path.dirname(os.path.abspath(csv_loc)) + '/net_metric_list_WB'

    metric_list_names = cPickle.load(open(met_list_picke_path, 'rb'))
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('')
    df = df.T
    column_headers={k: v for k, v in enumerate(metric_list_names)}
    df = df.rename(columns=column_headers)
    df['id'] = range(1, len(df) + 1)
    cols = df.columns.tolist()
    ix = cols.index('id')
    cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
    df = df[cols_ID]
    df['id'] = df['id'].astype('object')
    df['id'].values[0] = ID
    out_file = csv_loc.split('.csv')[0]
    df.to_pickle(out_file)
    return(out_file)
