#!/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import nilearn
import numpy as np
import bct
import os
import sys
from numpy import genfromtxt
from sklearn.covariance import GraphLassoCV
from matplotlib import pyplot as plt
from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

import_list=["import nilearn", "import numpy as np", "import os", "import bct", "from numpy import genfromtxt", "from matplotlib import pyplot as plt", "from nipype import Node, Workflow", "from nipype import Node, Workflow", "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu", "from nipype.interfaces import io as nio", "from nilearn import plotting", "from nilearn import datasets", "from nilearn.input_data import NiftiLabelsMasker", "from nilearn.connectome import ConnectivityMeasure", "from nilearn import plotting"]

######################
if len(sys.argv) > 1:
    input_file=sys.argv[1]
    ID=sys.argv[2]
else:
    print("Missing command-line inputs.\n\nYou musty include 1) Either a path to a functional image in standard space and .nii or .nii.gz format \nOR an input time-series text/csv file; \nand 2) A subject ID for those files")
    sys.exit()

######################
print("\n\n\n")
print ("INPUT FILE IS: " + input_file)
print ("SUBJECT ID IS: " + ID)
print("\n\n\n")
dir_path = os.path.dirname(os.path.realpath(input_file))

##Import ts and estimate cov
def import_mat_func(input_file, ID):
    if '.nii' in input_file:
        func_file=input_file
        dir_path = os.path.dirname(os.path.realpath(func_file))
        power = datasets.fetch_coords_power_2011()
        print('Power atlas comes with {0}.'.format(power.keys()))
        coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
        print('Stacked power coordinates in array of shape {0}.'.format(coords.shape))
        spheres_masker = input_data.NiftiSpheresMasker(
            seeds=coords, smoothing_fwhm=4, radius=5.,
            detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=2.5)
        time_series = spheres_masker.fit_transform(func_file)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        print('time series has {0} samples'.format(time_series.shape[0]))
        plt.imshow(matrix, vmin=-1., vmax=1., cmap='RdBu_r', interpolation='nearest')
        plt.colorbar()
        plt.title('Power correlation matrix')
        # Tweak edge_threshold to keep only the strongest connections.
        plotting.plot_connectome(matrix, coords, title='Power correlation graph',
                                 edge_threshold='99.8%', node_size=20, colorbar=True)
        out_path=dir_path + '/adj_mat_cov.png'
        plt.savefig(out_path)
        plt.close()
        np.savetxt(time_series, delimiter='\t')
        time_series_path = dir_path + '/' + ID + '_ts.txt'
        mx = genfromtxt(time_series_path, delimiter='')
    else:
        DR_st_1=input_file
        dir_path = os.path.dirname(os.path.realpath(DR_st_1))
        mx = genfromtxt(DR_st_1, delimiter='')
    from sklearn.covariance import GraphLassoCV
    estimator = GraphLassoCV()
    est = estimator.fit(mx)
    est_path1 = dir_path + '/' + ID + '_est_cov.txt'
    est_path2 = dir_path + '/' + ID + '_est_sps_inv_cov.txt'
    np.savetxt(est_path1, estimator.covariance_, delimiter='\t')
    np.savetxt(est_path2, estimator.precision_, delimiter='\t')
    return(mx, est_path1, est_path2)

##Display the covariance
def cov_plt_func(mx, est_path1, ID):
    rois_num=mx.shape[0]
    ts_num=mx.shape[1]
    dir_path = os.path.dirname(os.path.realpath(est_path1))
    est_cov = genfromtxt(est_path1)
    plt.figure(figsize=(10, 10))
    ##The covariance can be found at estimator.covariance_
    plt.imshow(est_cov, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    ##And display the labels
    x_ticks = plt.xticks(range(rois_num), rotation=90)
    y_ticks = plt.yticks(range(rois_num))
    plt.title('Covariance')
    out_path=dir_path + '/' + ID + '_adj_mat_cov.png'
    plt.savefig(out_path)
    plt.close()
    return(est_path1)

def sps_inv_cov_plt_func(mx, est_path2, ID):
    rois_num=mx.shape[0]
    ts_num=mx.shape[1]
    dir_path = os.path.dirname(os.path.realpath(est_path2))
    est_sps_inv_cov = genfromtxt(est_path2)
    plt.figure(figsize=(10, 10))
    ##The covariance can be found at estimator.precision_
    plt.imshow(-est_sps_inv_cov, interpolation="nearest",
               vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    ##And display the labels
    x_ticks = plt.xticks(range(rois_num), rotation=90)
    y_ticks = plt.yticks(range(rois_num))
    plt.title('Sparse inverse covariance')
    out_path=dir_path + '/' + ID + '_adj_mat_sps_inv_cov.png'
    plt.savefig(out_path)
    plt.close()
    return(est_path2)

def net_global_scalars_cov_func(est_path1, ID):
    in_mat = genfromtxt(est_path1)
    dir_path = os.path.dirname(os.path.realpath(est_path1))
    global_efficiency=float(bct.efficiency_wei(in_mat))
    modularity_und=float(bct.modularity_und(in_mat)[1])
    modularity_louvain_und=float(bct.modularity_louvain_und(in_mat)[1])
    out_path = dir_path + '/' + ID + '_net_global_scalars_cov.csv'
    np.savetxt(out_path, [global_efficiency, modularity_und, modularity_louvain_und])
    csv_loc1 = out_path
    return(csv_loc1)

def net_global_scalars_inv_sps_cov_func(est_path2, ID):
    in_mat = genfromtxt(est_path2)
    dir_path = os.path.dirname(os.path.realpath(est_path2))
    global_efficiency=float(bct.efficiency_wei(in_mat))
    modularity_und=float(bct.modularity_und(in_mat)[1])
    modularity_louvain_und=float(bct.modularity_louvain_und(in_mat)[1])
    out_path = dir_path + '/' + ID + '_net_global_scalars_inv_sps_cov.csv'
    csv_loc2 = out_path
    np.savetxt(out_path, [global_efficiency, modularity_und, modularity_louvain_und])
    return(csv_loc2)

##save global scalar files to pandas dataframes
def export_to_pandas1(csv_loc1, ID):
    import pandas as pd
    csv_loc = csv_loc1
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('').astype('float')
    df = df.T
    df = df.rename(columns={0: 'global_efficiency', 1: 'modularity_und', 2:'modularity_louvain_und'})
    df['id'] = range(1, len(df) + 1)
    if 'id' in df.columns:
        cols = df.columns.tolist()
        ix = cols.index('id')
        cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
        df = df[cols_ID]
    df['id'].values[0] = ID
    out_path_ren1 = csv_loc.replace('.', '')[:-3] + '_' + ID
    df.to_pickle(out_path_ren1)
    return(out_path_ren1)

##save global scalar files to pandas dataframes
def export_to_pandas2(csv_loc2, ID):
    import pandas as pd
    csv_loc = csv_loc2
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('').astype('float')
    df = df.T
    df = df.rename(columns={0: 'global_efficiency', 1: 'modularity_und', 2:'modularity_louvain_und'})
    df['id'] = range(1, len(df) + 1)
    if 'id' in df.columns:
        cols = df.columns.tolist()
        ix = cols.index('id')
        cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
        df = df[cols_ID]
    df['id'].values[0] = ID
    out_path_ren2 = csv_loc.replace('.', '')[:-3] + '_' + ID
    df.to_pickle(out_path_ren2)
    return(out_path_ren2)

##Create input/output nodes
inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'ID']),
                    name='inputnode')
inputnode.inputs.in_file = input_file
inputnode.inputs.ID = ID

##Create function nodes
imp_est = pe.Node(niu.Function(input_names = ['input_file', 'ID'], output_names = ['mx','est_path1', 'est_path2'], function=import_mat_func, imports=import_list), name = "imp_est")

cov_plt = pe.Node(niu.Function(input_names = ['mx', 'est_path1', 'ID'], output_names = ['est_path1'], function=cov_plt_func, imports=import_list), name = "cov_plt")

sps_inv_cov_plt = pe.Node(niu.Function(input_names=['mx', 'est_path2', 'ID'], output_names = ['est_path2'], function=sps_inv_cov_plt_func, imports=import_list), name = "sps_inv_cov_plt")

net_glob_scalars_cov = pe.Node(niu.Function(input_names=['est_path1', 'ID'], output_names = ['csv_loc1'], function=net_global_scalars_cov_func, imports=import_list), name = "net_glob_scalars_cov")

net_global_scalars_inv_sps_cov = pe.Node(niu.Function(input_names=['est_path2', 'ID'], output_names = ['csv_loc2'], function=net_global_scalars_inv_sps_cov_func, imports=import_list), name = "net_global_scalars_inv_sps_cov")

export_to_pandas1 = pe.Node(niu.Function(input_names=['csv_loc1', 'ID'], output_names = ['out_path_ren1'], function=export_to_pandas1, imports=import_list), name = "export_to_pandas1")

export_to_pandas2 = pe.Node(niu.Function(input_names=['csv_loc2', 'ID'], output_names = ['out_path_ren2'], function=export_to_pandas2, imports=import_list), name = "export_to_pandas2")

##Create PyNets workflow
wf = pe.Workflow(name='PyNets_WORKFLOW')
wf.base_directory='/tmp/nipype'

##Create data sink
#datasink = pe.Node(nio.DataSink(), name='sinker')
#datasink.inputs.base_directory = dir_path + '/DataSink'

##Connect nodes of workflow
wf.connect([
    (inputnode, imp_est, [('in_file', 'input_file'),
                          ('ID', 'ID')]),
    (inputnode, cov_plt, [('ID', 'ID')]),
    (imp_est, cov_plt, [('mx', 'mx'),
                        ('est_path1', 'est_path1')]),
    (imp_est, sps_inv_cov_plt, [('mx', 'mx'),
                                ('est_path2', 'est_path2')]),
    (inputnode, sps_inv_cov_plt, [('ID', 'ID')]),
    (imp_est, net_glob_scalars_cov, [('est_path1', 'est_path1')]),
    (inputnode, net_glob_scalars_cov, [('ID', 'ID')]),
    (imp_est, net_global_scalars_inv_sps_cov, [('est_path2', 'est_path2')]),
    (inputnode, net_global_scalars_inv_sps_cov, [('ID', 'ID')]),
#    (net_glob_scalars_cov, datasink, [('est_path1', 'csv_loc1')]),
#    (net_global_scalars_inv_sps_cov, datasink, [('est_path2', 'csv_loc2')]),
    (inputnode, export_to_pandas1, [('ID', 'ID')]),
    (net_glob_scalars_cov, export_to_pandas1, [('csv_loc1', 'csv_loc1')]),
    (inputnode, export_to_pandas2, [('ID', 'ID')]),
    (net_global_scalars_inv_sps_cov, export_to_pandas2, [('csv_loc2', 'csv_loc2')]),
#    (export_to_pandas1, datasink, [('out_path_ren1', 'pandas_df1')]),
#    (export_to_pandas2, datasink, [('out_path_ren2', 'pandas_df2')]),
])

wf.write_graph()
wf.run(plugin='MultiProc')
