import warnings
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import numpy as np
import os
import glob
import pandas as pd
import argparse
import sys
import seaborn as sns
import matplotlib
import multiprocessing
from matplotlib import pyplot as plt
from nipype.interfaces.fsl import FLIRT, ProbTrackX2, FNIRT, ConvertXFM, InvWarp, ApplyWarp
from subprocess import call
from sklearn.preprocessing import normalize
from nilearn import plotting, image, masking
from matplotlib import colors

def run_struct_mapping(FSLDIR, ID, bedpostx_dir, dir_path, NETWORK, coords_MNI, node_size):
    edge_threshold = 0.90
    connectome_fdt_thresh = 1000
    final_plot_path = dir_path + '/structural_connectome_fig_' + NETWORK + '_' + str(ID) + '.png'

    ####Auto-set INPUTS####
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    merged_th_samples_path = bedpostx_dir + '/merged_th1samples.nii.gz'
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    merged_ph_samples_path = bedpostx_dir + '/merged_ph1samples.nii.gz'
    input_MNI = FSLDIR + '/data/standard/MNI152_T1_2mm_brain.nii.gz'
    probtrackx_output_dir_path = bedpostx_dir + '/probtrackx_' + NETWORK
    ####Auto-set INPUTS####

    ##Delete any existing roi spheres
    del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*diff.nii.gz')
    try:
        for i in del_files_spheres:
            os.remove(i)
    except:
        pass

    ##Create transform matrix between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = input_MNI
    flirt.inputs.out_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.run()

    ##Apply transform between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = input_MNI
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.out_file = bedpostx_dir + '/xfms/MNI2diff_affine.nii.gz'
    flirt.run()

    x_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3,0:3])[0]
    y_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3,0:3])[1]
    z_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3,0:3])[2]

    def mmToVox(mmcoords):
        voxcoords = ['','','']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    ##Convert coords back to voxels
    coords_vox=[]
    for coord in coords_MNI:
        coords_vox.append(mmToVox(coord))
    coords = list(tuple(x) for x in coords_vox)

    j=0
    for i in coords:
        ##Grow spheres at ROI
        X = coords[j][0]
        Y = coords[j][1]
        Z = coords[j][2]
        out_file1 = bedpostx_dir + '/roi_point_' + str(j) +'.nii.gz'
        args = '-mul 0 -add 1 -roi ' + str(X) + ' 1 ' + str(Y) + ' 1 ' + str(Z) + ' 1 0 1'
        maths = fsl.ImageMaths(in_file=input_MNI, op_string=args, out_file=out_file1)
        os.system(maths.cmdline + ' -odt float')

        out_file2 = bedpostx_dir + '/roi_sphere_' + str(j) +'.nii.gz'
        args = '-kernel sphere ' + str(node_size) + ' -fmean -bin'
        maths = fsl.ImageMaths(in_file=out_file1, op_string=args, out_file=out_file2)
        os.system(maths.cmdline + ' -odt float')

        ##Map ROIs from Standard Space to diffusion Space:
        ##Applying xfm and input matrix to transform ROI's between diff and MNI using FLIRT,
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = nodif_brain_mask_path
        flirt.inputs.in_file = out_file2
        out_file_diff = out_file2.split('.nii')[0] + '_diff.nii.gz'
        flirt.inputs.out_file = out_file_diff
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
        flirt.run()
        j = j + 1

    if not os.path.exists(probtrackx_output_dir_path):
        os.makedirs(probtrackx_output_dir_path)

    seed_files = glob.glob(bedpostx_dir + '/*diff.nii.gz')
    seeds_text = probtrackx_output_dir_path + '/masks.txt'
    try:
        os.remove(seeds_text)
    except OSError:
        pass
    seeds_file_list = []
    for seed_file in seed_files:
        seeds_file_list.append(seed_file)
    f=open(seeds_text,'w')
    l1=map(lambda x:x+'\n', seeds_file_list)
    f.writelines(l1)
    f.close()

    del_files_points = glob.glob(bedpostx_dir + '/roi_point*.nii.gz')
    for i in del_files_points:
        os.remove(i)

    del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*[!diff].nii.gz')
    for i in del_files_spheres:
        os.remove(i)

    mx_path = dir_path + '/' + str(ID) + '_' + NETWORK + '_structural_mx.txt'
    probtrackx2 = pe.Node(interface=fsl.ProbTrackX2(),name='probtrackx2')
    probtrackx2.inputs.network=True
    probtrackx2.inputs.seed=seeds_text
    probtrackx2.inputs.onewaycondition=True
    probtrackx2.inputs.c_thresh=0.2
    probtrackx2.inputs.n_steps=2000
    probtrackx2.inputs.step_length=0.5
    probtrackx2.inputs.n_samples=5000
    probtrackx2.inputs.dist_thresh=0.0
    probtrackx2.inputs.opd=True
    probtrackx2.inputs.loop_check=True
    probtrackx2.inputs.omatrix1=True
    probtrackx2.overwrite=True
    probtrackx2.inputs.verbose=True
    probtrackx2.inputs.mask=nodif_brain_mask_path
    probtrackx2.inputs.out_dir=probtrackx_output_dir_path
    probtrackx2.inputs.thsamples=merged_th_samples_path
    probtrackx2.inputs.fsamples=merged_f_samples_path
    probtrackx2.inputs.phsamples=merged_ph_samples_path
    probtrackx2.iterables = ("seed", seed_files)
    try:
        probtrackx2.inputs.avoid_mp=vetricular_CSF_mask_path
    except:
        pass
    probtrackx2.run()
    del(probtrackx2)

    if os.path.exists(probtrackx_output_dir_path + '/fdt_network_matrix'):
        mx = np.genfromtxt(probtrackx_output_dir_path + '/fdt_network_matrix')

        waytotal = np.genfromtxt(probtrackx_output_dir_path + '/waytotal')
        np.seterr(divide='ignore', invalid='ignore')
        conn_matrix = np.divide(mx,waytotal)
        conn_matrix[np.isnan(conn_matrix)] = 0
        conn_matrix = np.nan_to_num(conn_matrix)
        conn_matrix = normalize(conn_matrix)

        ##Save matrix
        out_path_mx=dir_path + '/' + str(ID) + '_' + NETWORK + '_structural_mx.txt'
        np.savetxt(out_path_mx, conn_matrix, delimiter='\t')

        rois_num=conn_matrix.shape[0]
        print("Creating plot of dimensions:\n" + str(rois_num) + ' x ' + str(rois_num))
        plt.figure(figsize=(10, 10))
        plt.imshow(conn_matrix, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)

        ##And display the labels
        plt.colorbar()
        plt.title(atlas_select.upper() + ' ' + NETWORK + ' Structural Connectivity')

        out_path_fig=dir_path + '/' + str(ID) + '_' + NETWORK + '_structural_adj_mat.png'
        plt.savefig(out_path_fig)
        plt.close()

        fdt_paths_loc = probtrackx_output_dir_path + '/fdt_paths.nii.gz'

        ##Plotting with glass brain
        ##Create transform matrix between diff and MNI using FLIRT
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = input_MNI
        flirt.inputs.in_file = nodif_brain_mask_path
        flirt.inputs.out_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
        flirt.run()

        ##Apply transform between diff and MNI using FLIRT
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = input_MNI
        flirt.inputs.in_file = nodif_brain_mask_path
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
        flirt.inputs.out_file = bedpostx_dir + '/xfms/diff2MNI_affine.nii.gz'
        flirt.run()

        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = input_MNI
        flirt.inputs.in_file = fdt_paths_loc
        out_file_MNI = fdt_paths_loc.split('.nii')[0] + '_MNI.nii.gz'
        flirt.inputs.out_file = out_file_MNI
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/diff2MNI.mat'
        flirt.run()

        fdt_paths_MNI_loc = probtrackx_output_dir_path + '/fdt_paths_MNI.nii.gz'

        norm = colors.Normalize(vmin=-1, vmax=1)
        clust_pal = sns.color_palette("Blues_r", 4)
        clust_colors = colors.to_rgba_array(clust_pal)

        conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.transpose())
        connectome = plotting.plot_connectome(conn_matrix_symm, coords_MNI, edge_threshold=edge_threshold, node_color=clust_colors, edge_cmap=plotting.cm.black_blue_r)
        connectome.add_overlay(img=fdt_paths_MNI_loc, threshold=connectome_fdt_thresh, cmap=plotting.cm.cyan_copper_r)
        out_file_path = dir_path + '/structural_connectome_fig_' + NETWORK + '_' + str(ID) + '.png'
        plt.savefig(out_file_path)
        plt.close()

        if NETWORK != None:
            est_path = dir_path + '/' + ID + '_' + NETWORK + '_structural_est.txt'
        else:
            est_path = dir_path + '/' + ID + '_structural_est.txt'
        try:
            np.savetxt(est_path, conn_matrix_symm, delimiter='\t')
        except RuntimeError:
            print('Diffusion network connectome failed!')
    return(est_path)
