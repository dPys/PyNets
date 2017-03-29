#!/bin/env python
import nilearn
import numpy as np
import os
import bct
from numpy import genfromtxt
from sklearn.covariance import GraphLassoCV
from matplotlib import pyplot as plt
from nilearn import plotting
from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
import_list=["import nilearn", "import numpy as np", "import os", "import bct", "from numpy import genfromtxt", "from sklearn.covariance import GraphLassoCV", "from matplotlib import pyplot as plt", "from nilearn import plotting", "from nipype import Node, Workflow", "from nipype import Node, Workflow", "from nipype.pipeline import engine as pe", "from nipype.interfaces import utility as niu", "from nipype.interfaces import io as nio"]

##User inputs##
input_ts='/Users/PSYC-dap3463/Desktop/PyNets/data/roi_CC200.1D'
ID='CC200'
###############

##Import ts and estimate cov
def import_mat_func(DR_st_1):
    dir_path = os.path.dirname(os.path.realpath(DR_st_1))
    mx = genfromtxt(DR_st_1, delimiter='\t')
    estimator = GraphLassoCV()
    est = estimator.fit(mx.transpose())
    est_path1 = dir_path + '/est_cov.txt'
    est_path2 = dir_path + '/est_sps_inv_cov.txt'
    np.savetxt(est_path1, estimator.covariance_, delimiter='\t')
    np.savetxt(est_path2, estimator.precision_, delimiter='\t')
    return mx, est_path1, est_path2

##Display the covariance
def cov_plt_func(mx, est_path1):
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
    out_path=dir_path + '/adj_mat_cov.png'
    plt.savefig(out_path)
    plt.close()
    return out_path

def sps_inv_cov_plt_func(mx, est_path2):
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
    out_path=dir_path + '/adj_mat_sps_inv_cov.png'
    plt.savefig(out_path)
    plt.close()
    return out_path

def net_global_scalars_cov_func(est_path1):
    in_mat = genfromtxt(est_path1)
    dir_path = os.path.dirname(os.path.realpath(est_path1))
    global_efficiency=float(bct.efficiency_wei(in_mat))
    modularity_und=float(bct.modularity_und(in_mat)[1])
    modularity_louvain_und=float(bct.modularity_louvain_und(in_mat)[1])
    out_path = dir_path + '/net_global_scalars_cov.csv'
    np.savetxt(out_path, [global_efficiency, modularity_und, modularity_louvain_und])
    return out_path

def net_global_scalars_inv_sps_cov_func(est_path2):
    in_mat = genfromtxt(est_path2)
    dir_path = os.path.dirname(os.path.realpath(est_path2))
    global_efficiency=float(bct.efficiency_wei(in_mat))
    modularity_und=float(bct.modularity_und(in_mat)[1])
    modularity_louvain_und=float(bct.modularity_louvain_und(in_mat)[1])
    out_path = dir_path + '/net_global_scalars_inv_sps_cov.csv'
    np.savetxt(out_path, [global_efficiency, modularity_und, modularity_louvain_und])
    return out_path

##save global scalar files to pandas dataframes
def import_to_pandas(out_path, ID)
    import pandas as pd
    dir_path = os.path.dirname(os.path.realpath(out_path))
    csv_loc = out_path
    df = pd.read_csv(csv_loc, delimiter='\t', header=None).fillna('').astype('float')
    df = df.T
    df = df.rename(columns={0: 'global_efficiency', 1: 'modularity_und', 2:'modularity_louvain_und'})
    df['id'] = range(1, len(df) + 1)

    ##Rearrange columns in dataframe so that ID is the first column (if ID exists)
    if 'id' in df.columns:
        cols = df.columns.tolist()
        ix = cols.index('id')
        cols_ID = cols[ix:ix+1]+cols[:ix]+cols[ix+1:]
        df = df[cols_ID]
    df['id'].values[0]=ID
    suffix=out_path.split("_",1)[1][:-4]
    out_path=dir_path + '/' + ID + '_' + suffix
    df.to_pickle(out_path)
    return df out_path

##Create input/output nodes
inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                    name='inputnode')
inputnode.inputs.in_file = input_ts
outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                     name='outputnode')

##Create function nodes
imp_est = pe.Node(niu.Function(input_names = ['DR_st_1'], output_names = ['mx','est_path1', 'est_path2'], function=import_mat_func, imports=import_list), name = "imp_est")

cov_plt = pe.Node(niu.Function(input_names = ['mx', 'est_path1'], output_names = ['out_path'], function=cov_plt_func, imports=import_list), name = "cov_plt")

sps_inv_cov_plt = pe.Node(niu.Function(input_names=['mx', 'est_path2'], output_names = ['out_path'], function=sps_inv_cov_plt_func, imports=import_list), name = "sps_inv_cov_plt")

net_glob_scalars_cov = pe.Node(niu.Function(input_names=['est_path1'], output_names = ['out_path1'], function=net_global_scalars_cov_func, imports=import_list), name = "net_glob_scalars_cov")

net_global_scalars_inv_sps_cov = pe.Node(niu.Function(input_names=['est_path2'], output_names = ['out_path2'], function=net_global_scalars_inv_sps_cov_func, imports=import_list), name = "net_global_scalars_inv_sps_cov")

##Create PyNets workflow
wf = pe.Workflow(name='PyNets')

##Create data sink
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = '/Users/PSYC-dap3463/Desktop/PyNets'

##Connect nodes of workflow
wf.connect([
    (inputnode, imp_est, [('in_file', 'DR_st_1')]),
    (imp_est, cov_plt, [('mx', 'mx'),
                        ('est_path1', 'est_path1')]),
    (imp_est, sps_inv_cov_plt, [('mx', 'mx'),
                                ('est_path2', 'est_path2')]),
    (imp_est, net_glob_scalars_cov, [('est_path1', 'est_path1')]),
    (imp_est, net_global_scalars_inv_sps_cov, [('est_path2', 'est_path2')]),
    (net_glob_scalars_cov, datasink, [('out_file', 'global_scalars1')]),
    (net_global_scalars_inv_sps_cov, datasink, [('out_file', 'global_scalars2')])
])
wf.write_graph()
wf.run()
