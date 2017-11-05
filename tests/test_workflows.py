import sys
import argparse
import os
import timeit
import string
import pkgutil
import io
from pathlib import Path
import pynets
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
import matplotlib.pyplot as plt
import pytest
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
from pynets import workflows
from itertools import product
from pathlib import Path

def test_workflow_wb_connectome_with_us_atlas_coords():
    ##For fake testing
    input_file=Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz"
    dir_path =Path(__file__).parent/"examples"/"997"
    parlistfile=None
    conf=None
    network = None
    thr = '0.95'
    node_size = '2'
    conn_model = 'sps'
    atlas_select = 'coords_power_2011'
    ID = '997'
    adapt_thresh = False
    all_nets = False
    plot_switch = False
    multi_atlas = False
    multi_thr = False
    min_thr = None
    max_thr = None
    step_thr = None
    dens_thresh = None
    bedpostx_dir = None
    mask = None
    anat_loc = None
    [est_path, thr] = workflows.wb_connectome_with_nl_atlas_coords(input_file, ID, atlas_select, network, node_size, mask, thr, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc)

    assert est_path is not None

def test_workflow_wb_connectome_with_ua():
    ##For fake testing
    input_file=Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz"
    dir_path =Path(__file__).parent/"examples"/"997"
    parlistfile=Path(__file__).parent/"examples"/"whole_brain_cluster_labels_PCA200.nii.gz"
    conf=None
    atlas_select=None
    network = None
    thr = '0.95'
    node_size = '2'
    conn_model = 'sps'
    ID = '997'
    adapt_thresh = False
    all_nets = False
    plot_switch = False
    multi_atlas = False
    multi_thr = False
    min_thr = None
    max_thr = None
    step_thr = None
    dens_thresh = False
    bedpostx_dir = None
    mask = None
    anat_loc = None

    [est_path, thr] = workflows.wb_connectome_with_us_atlas_coords(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc)

    assert est_path is not None

def test_workflow_network_connectome():
    ##For fake testing
    input_file=Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz"
    dir_path =Path(__file__).parent/"examples"/"997"
    parlistfile = None
    conf=None
    atlas_select='coords_power_2011'
    network = 'DefaultA'
    thr = '0.95'
    node_size = '2'
    conn_model = 'sps'
    ID = '997'
    adapt_thresh = False
    all_nets = False
    plot_switch = False
    multi_atlas = False
    multi_thr = False
    min_thr = None
    max_thr = None
    step_thr = None
    dens_thresh = False
    bedpostx_dir = None
    mask = None
    anat_loc = None

    [est_path, thr] = workflows.network_connectome(input_file, ID, atlas_select, network, node_size, mask, thr, parlistfile, all_nets, conn_model, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir, anat_loc)
    assert est_path is not None
