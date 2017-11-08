# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017

@author: Derek Pisner
"""
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import numpy as np
import os
import glob
import seaborn as sns
import random
import multiprocessing
import time
from nipype.interfaces.fsl import ExtractROI
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib import colors
from nilearn import plotting, masking

def prepare_masks(ID, bedpostx_dir, network, coords_MNI, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, indices, anat_loc, volumes_dir, threads):
    ####Auto-set INPUTS####
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    input_MNI = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
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

    print('\n' + 'Creating MNI-diffusion space transforms...' + '\n')
    create_mni2diff_transforms(merged_f_samples_path, input_MNI, bedpostx_dir)

    ##Create avoidance and waypoints masks
    def gen_anat_segs(anat_loc, FSLDIR):
        ##Create MNI ventricle mask
        print('Creating MNI space ventricle mask...')
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
        fastr.run()
        old_file_csf = anat_loc.split('.nii.gz')[0] + '_pve_0.nii.gz'
        new_file_csf = anat_dir + '/CSF.nii.gz'
        os.rename(old_file_csf, new_file_csf)
        old_file_wm = anat_loc.split('.nii.gz')[0] + '_pve_2.nii.gz'
        new_file_wm = anat_dir + '/WM.nii.gz'
        os.rename(old_file_wm, new_file_wm)

        return(new_file_csf, new_file_wm, mni_csf_loc)

    def coreg_vent_CSF_to_diff(nodif_brain_mask_path, bedpostx_dir, csf_loc, mni_csf_loc):
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

        print('Eroding CSF mask...')
        ##Erode CSF mask
        out_file_final=vent_csf_diff_out.split('.nii.gz')[0] + '_ero.nii.gz'
        args = '-ero -bin'
        maths = fsl.ImageMaths(in_file=vent_csf_diff_out, op_string=args, out_file=out_file_final)
        os.system(maths.cmdline)
        return out_file_final

    def coreg_WM_mask_to_diff(nodif_brain_mask_path, bedpostx_dir, wm_mask_loc):
        out_file = bedpostx_dir + '/wm_mask_diff.nii.gz'
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = nodif_brain_mask_path
        flirt.inputs.in_file = wm_mask_loc
        flirt.inputs.out_file = out_file
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
        flirt.inputs.apply_xfm = True
        flirt.run()
        return out_file

    print('\n' + 'Segmenting anatomical image to create White Matter and Ventricular CSF masks for contraining the tractography...')
    [csf_loc, wm_mask_loc, mni_csf_loc] = gen_anat_segs(anat_loc, FSLDIR)
    print('\n' + 'Transforming WM mask to diffusion space...')
    vent_CSF_diff_mask_path = coreg_vent_CSF_to_diff(nodif_brain_mask_path, bedpostx_dir, csf_loc, mni_csf_loc)
    print('\n' + 'Transforming CSF mask to diffusion space...')
    WM_diff_mask_path = coreg_WM_mask_to_diff(nodif_brain_mask_path, bedpostx_dir, wm_mask_loc)

    if parcels == False:
        ##Delete any existing roi spheres
        del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*')
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
        del_files_parcels = glob.glob(bedpostx_dir + '/roi_parcel*')
        try:
            for i in del_files_parcels:
                os.remove(i)
        except:
            pass

        i = 0
        for parcel in os.listdir(volumes_dir):
            out_file = bedpostx_dir + '/roi_parcel_' + str(i) +'.nii.gz'
            flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
            flirt.inputs.reference = nodif_brain_mask_path
            flirt.inputs.in_file = volumes_dir + '/' + parcel
            out_file_diff = out_file.split('.nii')[0] + '_diff.nii.gz'
            flirt.inputs.out_file = out_file_diff
            flirt.inputs.apply_xfm = True
            flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
            flirt.run()
            i = i + 1
    return(vent_CSF_diff_mask_path, WM_diff_mask_path)

def run_struct_mapping(ID, bedpostx_dir, network, coords_MNI, node_size, atlas_select, label_names, plot_switch, parcels, dict_df, anat_loc, volumes_dir, threads, vent_CSF_diff_mask_path, WM_diff_mask_path):
    edge_threshold = 0.90
    connectome_fdt_thresh = 1000
    ##Define probtrackx2 sample chunking for embaressing parallelization
    num_chunks=10
    sample_chunks=100

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

    def run_probtrackx2(i):
        '''
        Moises-- this is my band-aid solution at the moment for avoiding memory leaks
        '''
        ##Stagger start times to avoid memory errors
        time.sleep(random.randint(1,750))

        tmp_dir = probtrackx_output_dir_path + '/tmp_samples_' + str(i)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        probtrackx2 = fsl.ProbTrackX2()
        probtrackx2.inputs.network=True
        probtrackx2.inputs.seed=seeds_text
        probtrackx2.inputs.onewaycondition=True
        probtrackx2.inputs.c_thresh=0.2
        probtrackx2.inputs.n_steps=2000
        probtrackx2.inputs.step_length=0.5
        probtrackx2.inputs.n_samples=sample_chunks
        probtrackx2.inputs.dist_thresh=0.0
        probtrackx2.inputs.opd=True
        probtrackx2.inputs.loop_check=True
        probtrackx2.inputs.omatrix1=True
        probtrackx2.overwrite=True
        probtrackx2.inputs.verbose=True
        probtrackx2.inputs.mask=nodif_brain_mask_path
        probtrackx2.inputs.out_dir=tmp_dir
        probtrackx2.inputs.thsamples=merged_th_samples_path
        probtrackx2.inputs.fsamples=merged_f_samples_path
        probtrackx2.inputs.phsamples=merged_ph_samples_path
        probtrackx2.inputs.use_anisotropy=True
        try:
            probtrackx2.inputs.avoid_mp=vent_CSF_diff_mask_path
        except:
            pass
        try:
            probtrackx2.inputs.waypoints=WM_diff_mask_path
            probtrackx2.inputs.waycond='OR'
        except:
            pass
        rseed_arg=' --rseed=' + str(random.randint(1,1000))
        os.system(probtrackx2.cmdline + rseed_arg)
        del(probtrackx2)
        return

    if __name__ == "__main__":
        os.chdir(bedpostx_dir)
        if threads != None:
            number_processes = int(int(threads)-1)
        else:
            number_processes = int(multiprocessing.cpu_count()-1)
        pool = multiprocessing.Pool(number_processes)
        print('\n' + 'Running proxtrackx2, parallelized by chunks of n_samples...')
        print('Number of available cores: ' + str(number_processes))
        print('N_samples chunk size: ' + str(sample_chunks))
        print('Number of chunks: ' + str(num_chunks))
        num_cores_per_chunk = round(number_processes/num_chunks,0)
        print('Running with ~' + str(num_cores_per_chunk) + ' cores per chunk...')
        samples_split = range(1, num_chunks)
        result = pool.map_async(run_probtrackx2, samples_split)
        result.wait()
        pool.close()
        pool.join()

    output_fdts = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_paths.nii.gz')
    net_mats = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_network_matrix')
    waytotals = glob.glob(probtrackx_output_dir_path + '/tmp*/waytotal')

    ##Merge tmp fdt_path niftis
    out_file = probtrackx_output_dir_path + '/fdt_paths.nii.gz'
    args=''
    for i in output_fdts[1:]:
        new_arg = ' -add ' + str(i)
        args = args + new_arg
    maths = fsl.ImageMaths(in_file=output_fdts[1], op_string=args, out_file=out_file)
    os.system(maths.cmdline)

    ##Merge output mat files
    #mat_list_txt = probtrackx_output_dir_path + '/Mat1_list.txt'
    #merged_mat = probtrackx_output_dir_path + '/merged_matrix1.dot'
    #with open (mat_list_txt, 'w') as output_mats_file:
        #for eachfile in output_mats:
            #print(output_mats_file)
            #output_mats_file.write(eachfile+'\n')
    #cmd = 'fdt_matrix_merge ' + mat_list_txt + ' ' + merged_mat
    #os.system(cmd)

    mats_list = []
    for i in net_mats:
        mats_list.append(np.genfromtxt(i))

    waytotals_list = []
    for i in waytotals:
        waytotals_list.append(np.genfromtxt(i))

    conn_matrices = []
    i=0
    for mx in mats_list:
        waytotal = waytotals_list[i]
        np.seterr(divide='ignore', invalid='ignore')
        conn_matrix = np.divide(mx,waytotal)
        conn_matrix[np.isnan(conn_matrix)] = 0
        conn_matrix = np.nan_to_num(conn_matrix)
        conn_matrices.append(conn_matrix)
        i = i + 1

    conn_matri = conn_matrices[0]
    for i in range(len(conn_matrices))[1:]:
        conn_matri = np.add(conn_matri, conn_matrices[i])

    conn_matrix = normalize(conn_matri)
    conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.transpose())

    if plot_switch == True:
        plt.figure(figsize=(8, 8))
        plt.imshow(conn_matrix, interpolation="nearest", vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
        plt.xticks(range(len(label_names)), label_names, size='xx-small', rotation=90)
        plt.yticks(range(len(label_names)), label_names, size='xx-small')
        plt_title = atlas_select + ' Structural Connectivity of: ' + str(ID)
        plt.title(plt_title)
        plt.grid(False)
        plt.gcf().subplots_adjust(left=0.8)

        out_path_fig=dir_path + '/structural_adj_mat_' + str(ID) + '.png'
        plt.savefig(out_path_fig)
        plt.close()

        ##Prepare glass brain figure
        fdt_paths_loc = probtrackx_output_dir_path + '/fdt_paths.nii.gz'

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

        colors.Normalize(vmin=-1, vmax=1)
        clust_pal = sns.color_palette("Blues_r", 4)
        clust_colors = colors.to_rgba_array(clust_pal)

        ##Plotting with glass brain
        connectome = plotting.plot_connectome(conn_matrix_symm, coords_MNI, edge_threshold=edge_threshold, node_color=clust_colors, edge_cmap=plotting.cm.black_blue_r)
        connectome.add_overlay(img=fdt_paths_MNI_loc, threshold=connectome_fdt_thresh, cmap=plotting.cm.cyan_copper_r)
        out_file_path = dir_path + '/structural_connectome_fig_' + network + '_' + str(ID) + '.png'
        plt.savefig(out_file_path)
        plt.close()

        from pynets import plotting as pynplot
        network = network + '_structural'
        conn_model = 'struct'
        pynplot.plot_connectogram(conn_matrix, conn_model, atlas_select, dir_path, ID, network, label_names)

    if parcels == False:
        del_files_points = glob.glob(bedpostx_dir + '/roi_point*')
        for i in del_files_points:
            os.remove(i)

        del_files_spheres = glob.glob(bedpostx_dir + '/roi_sphere*')
        for i in del_files_spheres:
            os.remove(i)
    else:
        del_files_parcels = glob.glob(bedpostx_dir + '/roi_parcel*')
        try:
            for i in del_files_parcels:
                os.remove(i)
        except:
            pass

    if network != None:
        est_path = dir_path + '/' + ID + '_' + network + '_structural_est.txt'
    else:
        est_path = dir_path + '/' + ID + '_structural_est.txt'
    try:
        np.savetxt(est_path, conn_matrix_symm, delimiter='\t')
    except RuntimeError:
        print('Workflow error. Exiting...')
    return(est_path)