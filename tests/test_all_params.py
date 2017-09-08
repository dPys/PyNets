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
from pynets import pynets_run

@pytest.mark.skip(reason="no way of currently testing this")

input_file=Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold.nii.gz"
ID='997'
parlistfile=Path(__file__).parent/"examples"/"whole_brain_cluster_labels_PCA100.nii.gz"
NETWORK = ['DMN', 'FPTC', 'DA', 'SN', 'VA', 'CON']
thr = ['0.99', '0.95', '0.90']
node_size = ['2', '4', '6']
mask=Path(__file__).parent/"examples"/"997"/"pDMN_3_bin.nii.gz"
all_nets = [True, False]
conn_model = ['corr', 'partcorr', 'sps', 'cov']
conf = Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_confounds.tsv"
adapt_thresh = None
plot_switch = [True, False]
multi_atlas = [True, False]
multi_thr = None
min_thr = None
max_thr = None
step_thr = None
dens_thresh = None

pynets_run(input_file, ID, atlas_select, parlistfile, NETWORK, thr, node_size, mask,
all_nets, conn_model, conf, dens_thresh, adapt_thresh, plot_switch, multi_atlas,
multi_thr, min_thr, max_thr, step_thr)
