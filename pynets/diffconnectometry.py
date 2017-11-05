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
import pynets
import nibabel as nib
from nipype.interfaces.fsl import ExtractROI
from matplotlib import pyplot as plt
from nipype.interfaces.fsl import FLIRT, ProbTrackX2, FNIRT, ConvertXFM, InvWarp, ApplyWarp
from subprocess import call
from sklearn.preprocessing import normalize
from nilearn import plotting, image, masking
from matplotlib import colors
from nilearn import datasets
from pynets.nodemaker import fetch_nilearn_atlas_coords

def run_struct_mapping(ID, bedpostx_dir, network, coords_MNI, node_size, atlas_select, atlas_name, label_names, plot_switch, parcels, dict_df, indices, anat_loc):
    edge_threshold = 0.90
    connectome_fdt_thresh = 1000

    ####Auto-set INPUTS####
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    dir_path = os.path.dirname(bedpostx_dir)
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    merged_th_samples_path = bedpostx_dir + '/merged_th1samples.nii.gz'
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    merged_ph_samples_path = bedpostx_dir + '/merged_ph1samples.nii.gz'
    input_MNI = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    probtrackx_output_dir_path = bedpostx_dir + '/probtrackx_' + network
    ####Auto-set INPUTS####

    def create_mni2diff_transforms(merged_f_samples_path, input_MNI, bedpostx_dir):
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
        return

    create_mni2diff_transforms(merged_f_samples_path, input_MNI, bedpostx_dir)

    if parcels == False:
        ##Delete any existing roi spheres
        del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*diff.nii.gz')
        try:
            for i in del_files_spheres:
                os.remove(i)
        except:
            pass

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
    else:
        del_files_spheres = glob.glob(bedpostx_dir + '/roi_parcel*diff.nii.gz')
        try:
            for i in del_files_spheres:
                os.remove(i)
        except:
            pass

        def gen_anat_segs(anat_loc, FSLDIR):
            ##Create MNI ventricle mask
            print('Creating MNI Space ventricle mask...')
            anat_dir = os.path.dirname(anat_loc)
            lvent_out_file = anat_dir + '/LVentricle.nii.gz'
            rvent_out_file = anat_dir + '/RVentricle.nii.gz'
            MNI_atlas = FSLDIR + '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz'
            fslroi1 = ExtractROI(in_file=MNI_atlas, roi_file=lvent_out_file, t_min=2, t_size=1)
            os.system(fslroi1.cmdline)
            fslroi2 = ExtractROI(in_file=MNI_atlas, roi_file=rvent_out_file, t_min=13, t_size=1)
            os.system(fslroi2.cmdline)
            mni_csf_loc = anat_dir + '/VentricleMask.nii.gz'
            args = '-add ' + rvent_out_file + ' -thr 0.1 -bin -dilF'
            maths = fsl.ImageMaths(in_file=lvent_out_file, op_string=args, out_file=mni_csf_loc)
            os.system(maths.cmdline)

            ##Segment anatomical (should be in MNI space)
            print('Segmenting anatomical image using FAST...')
            fastr = fsl.FAST()
            fastr.inputs.in_files = anat_loc
            fastr.inputs.img_type = 1
            out = fastr.run()
            old_file_csf = anat_loc.split('.nii.gz')[0] + '_pve_0.nii.gz'
            new_file_csf = anat_dir + '/CSF.nii.gz'
            os.rename(old_file_csf, new_file_csf)
            old_file_wm = anat_loc.split('.nii.gz')[0] + '_pve_2.nii.gz'
            new_file_wm = anat_dir + '/WM.nii.gz'
            os.rename(old_file_wm, new_file_wm)

            return(new_file_csf, new_file_wm, mni_csf_loc)

        def coreg_vent_CSF_to_diff(nodif_brain_mask_path, bedpostx_dir, csf_loc, mni_csf_loc):
            print('Transforming CSF mask and ventricle mask to diffusion space...')
            csf_mask_diff_out = bedpostx_dir + '/csf_diff.nii.gz'
            flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
            flirt.inputs.reference = nodif_brain_mask_path
            flirt.inputs.in_file = csf_loc
            flirt.inputs.out_file = csf_mask_diff_out
            flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
            flirt.inputs.apply_xfm = True
            flirt.run()

            vent_mask_diff_out = bedpostx_dir + '/ventricle_diff.nii.gz'
            flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
            flirt.inputs.reference = nodif_brain_mask_path
            flirt.inputs.in_file = mni_csf_loc
            flirt.inputs.out_file = vent_mask_diff_out
            flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
            flirt.inputs.apply_xfm = True
            flirt.run()

            ##Mask CSF with MNI ventricle mask
            print('Masking CSF with ventricle mask...')
            vent_csf_diff_out = bedpostx_dir + '/vent_csf_diff.nii.gz'
            args = '-mas ' + vent_mask_diff_out
            maths = fsl.ImageMaths(in_file=csf_mask_diff_out, op_string=args, out_file=vent_csf_diff_out)
            os.system(maths.cmdline)

            ##Erode CSF mask
            out_file_final=vent_csf_diff_out.split('.nii.gz')[0] + '_ero.nii.gz'
            args = '-ero -bin'
            maths = fsl.ImageMaths(in_file=vent_csf_diff_out, op_string=args, out_file=out_file_final)
            os.system(maths.cmdline)
            return out_file_final

        def coreg_WM_mask_to_diff(nodif_brain_mask_path, bedpostx_dir, wm_mask_loc):
            print('Transforming white matter mask to diffusion space...')
            out_file = bedpostx_dir + '/wm_mask_diff.nii.gz'
            flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
            flirt.inputs.reference = nodif_brain_mask_path
            flirt.inputs.in_file = wm_mask_loc
            flirt.inputs.out_file = out_file
            flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
            flirt.inputs.apply_xfm = True
            flirt.run()
            return out_file

        [csf_loc, wm_mask_loc] = gen_anat_segs(anat_loc, FSLDIR)
        vent_CSF_diff_mask_path = coreg_vent_CSF_to_diff(nodif_brain_mask_path, bedpostx_dir, csf_loc, mni_csf_loc)
        WM_diff_mask_path = coreg_WM_mask_to_diff(nodif_brain_mask_path, bedpostx_dir, wm_mask_loc)

        i = 0
        for parcel in glob.glob('/scratch/04171/dpisner/HNU1/atlases/desikan/vol*'):
            out_file = bedpostx_dir + '/roi_parcel_' + str(i) +'.nii.gz'
            flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
            flirt.inputs.reference = nodif_brain_mask_path
            flirt.inputs.in_file = parcel
            out_file_diff = out_file.split('.nii')[0] + '_diff.nii.gz'
            flirt.inputs.out_file = out_file_diff
            flirt.inputs.apply_xfm = True
            flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
            flirt.run()
            i = i + 1

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

    if parcels == False:
        del_files_points = glob.glob(bedpostx_dir + '/roi_point*.nii.gz')
        for i in del_files_points:
            os.remove(i)

        del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*[!diff].nii.gz')
        for i in del_files_spheres:
            os.remove(i)

    mx_path = dir_path + '/' + str(ID) + '_' + network + '_structural_mx.txt'
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
    probtrackx2.inputs.use_anisotropy = True
    #probtrackx2.iterables = ("n_samples", seed_files)
    try:
        probtrackx2.inputs.avoid_mp=vent_CSF_diff_mask_path
    except:
        pass
    try:
        probtrackx2.inputs.waypoints=WM_diff_mask_path
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
        out_path_mx=dir_path + '/' + str(ID) + '_' + network + '_structural_mx.txt'
        np.savetxt(out_path_mx, conn_matrix, delimiter='\t')


        if plot_switch == True:
            rois_num=conn_matrix.shape[0]
            print("Creating plot of dimensions:\n" + str(rois_num) + ' x ' + str(rois_num))
            plt.figure(figsize=(10, 10))
            plt.imshow(conn_matrix, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)

            ##And display the labels
            plt.colorbar()
            plt.title(atlas_select.upper() + ' ' + network + ' Structural Connectivity')

            out_path_fig=dir_path + '/' + str(ID) + '_' + network + '_structural_adj_mat.png'
            plt.savefig(out_path_fig)
            plt.close()

            conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.transpose())

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

        if plot_switch == True:
            norm = colors.Normalize(vmin=-1, vmax=1)
            clust_pal = sns.color_palette("Blues_r", 4)
            clust_colors = colors.to_rgba_array(clust_pal)

            connectome = plotting.plot_connectome(conn_matrix_symm, coords_MNI, edge_threshold=edge_threshold, node_color=clust_colors, edge_cmap=plotting.cm.black_blue_r)
            connectome.add_overlay(img=fdt_paths_MNI_loc, threshold=connectome_fdt_thresh, cmap=plotting.cm.cyan_copper_r)
            out_file_path = dir_path + '/structural_connectome_fig_' + network + '_' + str(ID) + '.png'
            plt.savefig(out_file_path)
            plt.close()

            from pynets import plotting as pynplot
            network = network + '_structural'
            pynplot.plot_connectogram(conn_matrix, conn_model, atlas_name, dir_path, ID, network, label_names)

        if network != None:
            est_path = dir_path + '/' + ID + '_' + network + '_structural_est.txt'
        else:
            est_path = dir_path + '/' + ID + '_structural_est.txt'
        try:
            np.savetxt(est_path, conn_matrix_symm, delimiter='\t')
        except RuntimeError:
            print('Diffusion network connectome failed!')
    return(est_path)
