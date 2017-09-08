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

@pytest.mark.skip(reason="no way of currently testing this")
def test_all_param_combs():
    ##For fake testing
    input_file='examples/997/sub-997_ses-01_task-REST_run-01_bold.nii.gz'
    parlistfile='examples/whole_brain_cluster_labels_PCA100.nii.gz'
    conf='examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
    ##

    #input_file=Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold.nii.gz"
    #parlistfile=Path(__file__).parent/"examples"/"whole_brain_cluster_labels_PCA100.nii.gz"
    #conf = Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_confounds.tsv"

    NETWORK = ['DMN', 'FPTC', 'DA', 'SN', 'VA', 'CON']
    thr = ['0.99', '0.95', '0.90']
    node_size = ['2', '4', '6']
    conn_model = ['corr', 'partcorr', 'sps', 'cov']
    atlas_select=['coords_power_2011', 'coords_dosenbach_2010', 'atlas_destrieux_2009', 'atlas_aal']
    ID='997'
    adapt_thresh = None
    all_nets = [True, False]
    plot_switch = [True, False]
    multi_atlas = [True, False]
    multi_thr = False
    min_thr = None
    max_thr = None
    step_thr = None
    dens_thresh = None
    bedpostx_dir = None
    mask=None

    pynets_iterables = list(product(conn_model, NETWORK, thr, node_size, atlas_select))
    print('Iterating over ' + str(len(list(product(conn_model, NETWORK, thr, node_size)))) + ' combinations of the pipeline...')

    for (conn_model_iter, NETWORK_iter, thr_iter, node_size_iter, atlas_select) in pynets_iterables:
        print(conn_model_iter)
        print(NETWORK_iter)
        print(thr_iter)
        print(node_size_iter)
        [est_path, thr] = workflows.network_connectome(input_file, ID, atlas_select, NETWORK_iter,
        node_size_iter, mask, thr_iter, parlistfile, all_nets, conn_model_iter, dens_thresh, conf, adapt_thresh, plot_switch, bedpostx_dir)

    assert est_path is not None
