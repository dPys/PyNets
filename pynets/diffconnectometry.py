# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import nibabel as nib
import numpy as np
import os


def create_mni2diff_transforms(bedpostx_dir):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    ####Custom inputs####
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    ####Custom inputs####
    print('\nCreating MNI-diffusion space transforms...\n')
    input_MNI = FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz'
    out_aff = bedpostx_dir + '/xfms/MNI2diff_affine.nii.gz'

    ##Create transform matrix between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = input_MNI
    flirt.inputs.out_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.out_file = '/tmp/out_flirt.nii.gz'
    flirt.run()

    ##Apply transform between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = input_MNI
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.out_file = out_aff
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.run()
    return out_aff


##Create avoidance and waypoints masks
def gen_anat_segs(anat_loc, out_aff):
    ####Custom inputs####
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    import nipype.interfaces.fsl as fsl
    #import nibabel as nib
    #from nilearn.image import resample_img
    from nipype.interfaces.fsl import ExtractROI
    print('\nSegmenting anatomical image to create White Matter and Ventricular CSF masks for contraining the tractography...')
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
    old_file_wm = anat_loc.split('.nii.gz')[0] + '_pve_2.nii.gz'
    new_file_wm = anat_dir + '/WM.nii.gz'
    os.rename(old_file_csf, new_file_csf)
    os.rename(old_file_wm, new_file_wm)

    ##Reslice to 1x1x1mm voxels
    #img=nib.load(anat_loc)
    #vox_sz = img.affine[0][0]
    #targ_aff = img.affine/(np.array([[int(abs(vox_sz)),1,1,1],[1,int(abs(vox_sz)),1,1],[1,1,int(abs(vox_sz)),1],[1,1,1,1]]))
    #new_file_csf_res = resample_img(new_file_csf, target_affine=targ_aff)
    #new_file_wm_res = resample_img(new_file_wm, target_affine=targ_aff)
    #nib.save(new_file_csf_res, new_file_csf)
    #nib.save(new_file_wm_res, new_file_wm)
    return new_file_csf, mni_csf_loc, new_file_wm


def coreg_vent_CSF_to_diff(bedpostx_dir, csf_loc, mni_csf_loc):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    print('\nTransforming CSF mask to diffusion space...')
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    csf_mask_diff_out = bedpostx_dir + '/csf_diff.nii.gz'
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = csf_loc
    flirt.inputs.out_file = csf_mask_diff_out
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.apply_xfm = True
    flirt.run()

    vent_mask_diff_out = bedpostx_dir + '/ventricle_diff.nii.gz'
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = mni_csf_loc
    flirt.inputs.out_file = vent_mask_diff_out
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.apply_xfm = True
    flirt.run()

    ##Mask CSF with MNI ventricle mask
    print('Masking CSF with ventricle mask...')
    vent_csf_diff_out = bedpostx_dir + '/vent_csf_diff.nii.gz'
    args = '-mas ' + vent_mask_diff_out
    maths = fsl.ImageMaths(in_file=csf_mask_diff_out, op_string=args, out_file=vent_csf_diff_out)
    os.system(maths.cmdline)

    print('Eroding and binarizing CSF mask...')
    ##Erode CSF mask
    out_file_final=vent_csf_diff_out.split('.nii.gz')[0] + '_ero.nii.gz'
    args = '-ero -bin'
    maths = fsl.ImageMaths(in_file=vent_csf_diff_out, op_string=args, out_file=out_file_final)
    os.system(maths.cmdline)
    return out_file_final


def coreg_WM_mask_to_diff(bedpostx_dir, wm_mask_loc):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    print('\nTransforming WM mask to diffusion space...')
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    out_file = bedpostx_dir + '/wm_mask_diff.nii.gz'
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
    flirt.inputs.reference = merged_f_samples_path
    flirt.inputs.in_file = wm_mask_loc
    flirt.inputs.out_file = out_file
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
    flirt.inputs.apply_xfm = True
    flirt.run()

    args = '-bin'
    maths = fsl.ImageMaths(in_file=out_file, op_string=args, out_file=out_file)
    print('\nBinarizing WM mask...')
    os.system(maths.cmdline)
    return out_file


def reg_coords2diff(coords, bedpostx_dir, node_size, dir_path):
    try:
        FSLDIR = os.environ['FSLDIR']
    except NameError:
        print('FSLDIR environment variable not set!')
    import glob
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    from nilearn import masking
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    pick_dump = True
    volumes_dir = dir_path + '/volumes'
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'

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
    for coord in coords:
        coords_vox.append(mmToVox(coord))
    coords = list(tuple(x) for x in coords_vox)

    j=0
    for i in coords:
        ##Grow spheres at ROI
        X = coords[j][0]
        Y = coords[j][1]
        Z = coords[j][2]
        out_file1 = volumes_dir + '/roi_point_' + str(j) +'.nii.gz'
        args = '-mul 0 -add 1 -roi ' + str(X) + ' 1 ' + str(Y) + ' 1 ' + str(Z) + ' 1 0 1'
        maths = fsl.ImageMaths(in_file=FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz', op_string=args, out_file=out_file1)
        os.system(maths.cmdline + ' -odt float')

        out_file2 = volumes_dir + '/region_' + str(j) + '.nii.gz'
        args = '-kernel sphere ' + str(node_size) + ' -fmean -bin'
        maths = fsl.ImageMaths(in_file=out_file1, op_string=args, out_file=out_file2)
        os.system(maths.cmdline + ' -odt float')

        ##Map ROIs from Standard Space to diffusion Space:
        ##Applying xfm and input matrix to transform ROI's between diff and MNI using FLIRT,
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = merged_f_samples_path
        flirt.inputs.in_file = out_file2
        out_file_diff = out_file2.split('.nii')[0] + '_diff.nii.gz'
        flirt.inputs.out_file = out_file_diff
        flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
        flirt.run()
        j = j + 1

    del_files_points = glob.glob(volumes_dir + '/roi_point*')
    try:
        for i in del_files_points:
            os.remove(i)
    except:
        pass

    spheres_list = [i for i in os.listdir(volumes_dir) if 'diff.nii' in i]
    if pick_dump == True:
        ##Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/coords.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)

    return spheres_list


def reg_parcels2diff(bedpostx_dir, dir_path):
    import re
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    volumes_dir = dir_path + '/volumes'
    volumes_list = [i for i in os.listdir(volumes_dir) if '.nii' in i]

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]

    volumes_list.sort(key=natural_keys)
    i = 0
    for parcel in volumes_list:
        out_file = volumes_dir + '/region_' + str(i) + '_diff.nii.gz'
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'),name='coregister')
        flirt.inputs.reference = merged_f_samples_path
        flirt.inputs.in_file = volumes_dir + '/' + parcel
        flirt.inputs.out_file = out_file
        flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = bedpostx_dir + '/xfms/MNI2diff.mat'
        flirt.run()
        i = i + 1
    volumes_list = [i for i in os.listdir(volumes_dir) if 'diff.nii' in i]
    return volumes_list


def create_seed_mask_file(bedpostx_dir, network, dir_path):
    import glob
    import re
    import shutil

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]

    volumes_dir = dir_path + '/volumes'

    if network:
        probtrackx_output_dir_path = dir_path + '/probtrackx_' + network
    else:
        probtrackx_output_dir_path = dir_path + '/probtrackx_WB'

    if os.path.exists(probtrackx_output_dir_path):
        shutil.rmtree(probtrackx_output_dir_path)
    os.makedirs(probtrackx_output_dir_path)

    seed_files = glob.glob(volumes_dir + '/region*diff.nii.gz')
    seeds_text = probtrackx_output_dir_path + '/masks.txt'
    try:
        os.remove(seeds_text)
    except OSError:
        pass
    seeds_file_list = []
    for seed_file in seed_files:
        seeds_file_list.append(seed_file)
    seeds_file_list.sort(key=natural_keys)
    f=open(seeds_text,'w')
    l1=map(lambda x:x+'\n', seeds_file_list)
    f.writelines(l1)
    f.close()
    return seeds_text, probtrackx_output_dir_path


def save_parcel_vols(bedpostx_dir, parcel_list, net_parcels_map_nifti, dir_path):
    import os
    import time
    if net_parcels_map_nifti:
        net_parcels_map_file = dir_path + '/net_parcels_map_nifti.nii.gz'
        nib.save(net_parcels_map_nifti, net_parcels_map_file)
    volumes_dir = dir_path + '/volumes'
    if not os.path.exists(volumes_dir):
        os.mkdir(volumes_dir)
    num_vols = len(parcel_list)
    i = 0
    for vol in parcel_list:
        out_path = volumes_dir + '/region_' + str(i) + '.nii.gz'
        nib.save(vol, out_path)
        i = i + 1
    num_vols_pres = float(len(next(os.walk(volumes_dir))[2]))
    while num_vols_pres < float(num_vols):
        time.sleep(1)
        num_vols_pres = float(len(next(os.walk(volumes_dir))[2]))
    return num_vols_pres


def prepare_masks(bedpostx_dir, csf_loc, mni_csf_loc, wm_mask_loc):
    from pynets import diffconnectometry
    if (csf_loc and mni_csf_loc) is None:
        vent_CSF_diff_mask_path = None
    else:
        vent_CSF_diff_mask_path = diffconnectometry.coreg_vent_CSF_to_diff(bedpostx_dir, csf_loc, mni_csf_loc)

    if wm_mask_loc is None:
        WM_diff_mask_path = None
    else:
        WM_diff_mask_path = diffconnectometry.coreg_WM_mask_to_diff(bedpostx_dir, wm_mask_loc)
    return vent_CSF_diff_mask_path, WM_diff_mask_path


def grow_nodes(bedpostx_dir, coords, node_size, parc, parcel_list, net_parcels_map_nifti, network, dir_path):
    import shutil
    from pynets import diffconnectometry

    volumes_dir = dir_path + '/volumes'
    if os.path.exists(volumes_dir) is True:
        shutil.rmtree(volumes_dir)
    num_vols_pres = diffconnectometry.save_parcel_vols(bedpostx_dir, parcel_list, net_parcels_map_nifti, dir_path)

    if parc == False:
        spheres_list = diffconnectometry.reg_coords2diff(coords, bedpostx_dir, node_size, dir_path)
    else:
        volumes_list = diffconnectometry.reg_parcels2diff(bedpostx_dir, dir_path)

    [seeds_text, probtrackx_output_dir_path] = diffconnectometry.create_seed_mask_file(bedpostx_dir, network, dir_path)
    return seeds_text, probtrackx_output_dir_path


def run_probtrackx2(i, seeds_text, bedpostx_dir, probtrackx_output_dir_path, vent_CSF_diff_mask_path, WM_diff_mask_path, procmem, num_total_samples):
    import random
    import nipype.interfaces.fsl as fsl
    samples_i = int(round(float(num_total_samples) / float(procmem[0]),0))
    nodif_brain_mask_path = bedpostx_dir + '/nodif_brain_mask.nii.gz'
    merged_th_samples_path = bedpostx_dir + '/merged_th1samples.nii.gz'
    merged_f_samples_path = bedpostx_dir + '/merged_f1samples.nii.gz'
    merged_ph_samples_path = bedpostx_dir + '/merged_ph1samples.nii.gz'

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
    probtrackx2.inputs.n_samples=samples_i
    probtrackx2.inputs.dist_thresh=0.0
    probtrackx2.inputs.opd=True
    probtrackx2.inputs.loop_check=True
    probtrackx2.inputs.omatrix1=False
    probtrackx2.overwrite=True
    probtrackx2.inputs.verbose=True
    probtrackx2.inputs.mask=nodif_brain_mask_path
    probtrackx2.inputs.out_dir=tmp_dir
    probtrackx2.inputs.thsamples=merged_th_samples_path
    probtrackx2.inputs.fsamples=merged_f_samples_path
    probtrackx2.inputs.phsamples=merged_ph_samples_path
    probtrackx2.inputs.use_anisotropy=False
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
    os.chdir(bedpostx_dir)
    os.system(probtrackx2.cmdline + rseed_arg)
    del(probtrackx2)
    return


def collect_struct_mapping_outputs(parc, bedpostx_dir, network, ID, probtrackx_output_dir_path, dir_path, procmem):
    ##Wait for all probtrackx runs to complete
    import os
    import time
    import glob
    import nibabel as nib
    from sklearn.preprocessing import normalize

    tmp_files = []
    for i in [str(x) for x in range(int(procmem[0]))]:
        tmp_files.append(probtrackx_output_dir_path + '/tmp_samples_' + str(i) + '/fdt_paths.nii.gz')

    while True:
       if all([os.path.isfile(f) for f in tmp_files]):
          break
       else:
          print('Waiting on all fdt_paths!')
          print(str(sum([os.path.isfile(f) for f in tmp_files])) + str(' tmp files produced out of ') + str(len(tmp_files)) + ' expected...')
          time.sleep(10)

    output_fdts = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_paths.nii.gz')
    net_mats = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_network_matrix')
    waytotals = glob.glob(probtrackx_output_dir_path + '/tmp*/waytotal')

    try:
        ##Add the images
        out_file = probtrackx_output_dir_path + '/fdt_paths.nii.gz'
        fdt_imgs = []
        for img in output_fdts:
            fdt_imgs.append(nib.load(img).get_data())

        aff = nib.load(img).affine
        all_img = fdt_imgs[0]
        for img in fdt_imgs[1:]:
            all_img = all_img + img

        result_img = nib.Nifti1Image(all_img, affine=aff)
        print('\nSaving sum of all fdt_paths temporary images...\n')
        nib.save(result_img, out_file)

    except RuntimeError:
        print('fdt_paths wont merge!')

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
        try:
            conn_matri = conn_matri + conn_matrices[i]
        except:
            print('Matrix ' + str(i + 1) + ' is a different shape: ' + str(conn_matrices[i].shape) + '. Skipping...')
            continue

    try:
        print('Normalizing array...')
        conn_matrix = normalize(np.nan_to_num(conn_matri))
    except RuntimeError:
        print('Normalization failed...')
        pass
    conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.transpose())

    if parc is False:
        del_files_spheres = glob.glob(dir_path + '/volumes/roi_sphere*')
        for i in del_files_spheres:
            os.remove(i)
    else:
        del_files_parcels = glob.glob(dir_path + '/volumes/roi_parcel*')
        try:
            for i in del_files_parcels:
                os.remove(i)
        except:
            pass

    if network is not None:
        est_path_struct = dir_path + '/' + ID + '_' + network + '_struct_est_unthr_unsymm.npy'
    else:
        est_path_struct = dir_path + '/' + ID + '_struct_est_unthr_unsymm.npy'
    try:
        np.save(est_path_struct, conn_matrix)
    except RuntimeError:
        print('Workflow error. Exiting...')
    return conn_matrix_symm
