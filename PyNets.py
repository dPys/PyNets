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

##Import ts and estimate cov
def import_mat(DR_st_1):
    dir_path = os.path.dirname(os.path.realpath(DR_st_1))
    mx = genfromtxt(DR_st_1, delimiter='\t')
    estimator = GraphLassoCV()
    est = estimator.fit(mx.transpose())
    est_path1 = np.savetxt(dir_path + '/est_cov.txt',estimator.covariance_)
    est_path2 = np.savetxt(dir_path + '/est_sps_inv_cov.txt',estimator.precision_)
    return mx,est_path1,est_path2

##Display the covariance
def cov_plt(mx, est_path1):
    rois_num=mx.shape[0]
    ts_num=mx.shape[1]
    dir_path = os.path.dirname(os.path.realpath(est_path1))
    est_cov = genfromtxt(est_path1, delimiter='\t')
    plt.figure(figsize=(10, 10))
    ##The covariance can be found at estimator.covariance_
    plt.imshow(est_cov, interpolation="nearest",
               vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    ##And display the labels
    x_ticks = plt.xticks(range(rois_num), rotation=90)
    y_ticks = plt.yticks(range(rois_num))
    plt.title('Covariance')
    plt.savefig(dir_path + '/adj_mat_cov.png')
    plt.close()

def sps_inv_cov_plt(mx, est_path2):
    rois_num=mx.shape[0]
    ts_num=mx.shape[1]
    dir_path = os.path.dirname(os.path.realpath(est_path2))
    est_sps_inv_cov = genfromtxt(est_path2, delimiter='\t')
    plt.figure(figsize=(10, 10))
    ##The covariance can be found at estimator.precision_
    plt.imshow(-est_sps_inv_cov, interpolation="nearest",
               vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
    ##And display the labels
    x_ticks = plt.xticks(range(rois_num), rotation=90)
    y_ticks = plt.yticks(range(rois_num))
    plt.title('Sparse inverse covariance')
    plt.savefig(dir_path + '/adj_mat_sps_inv_cov.png')
    plt.close()

input_ts='/Users/PSYC-dap3463/Desktop/PyNets/roi_CC200.1D'

imp_est = Node(import_mat(input_ts), name = "imp_est")
cov_plt = Node(cov_plt(mx, est_path1), name = "cov_plt")
sps_inv_cov_plt = Node(sps_inv_cov_plt(mx, est_path2), name = "sps_inv_cov_plt")
wf = Workflow('PyNets')
wf.connect(imp_est, 'input_ts', [mx, est_path1, est_path2], cov_plt, sps_inv_cov_plt)
out = wf.run()

global_efficiency=bct.efficiency_wei(est_cov)
modularity_und=bct.modularity_und(est_cov)[1]
modularity_louvain_und=bct.modularity_louvain_und(est_cov)[1]
