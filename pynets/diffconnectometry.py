# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import nibabel as nib
import numpy as np
import os
nib.arrayproxy.KEEP_FILE_OPEN_DEFAULT = 'auto'


def create_mni2diff_transforms(dwi_dir):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    try:
        FSLDIR = os.environ['FSLDIR']
    except KeyError:
        print('FSLDIR environment variable not set!')

    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    if not os.path.exists("%s%s" % (dwi_dir, '/xfms')):
        os.mkdir("%s%s" % (dwi_dir, '/xfms'))

    print('\nCreating MNI-diffusion space transforms...\n')
    input_MNI = "%s%s" % (FSLDIR, '/data/standard/MNI152_T1_1mm_brain.nii.gz')
    out_aff = "%s%s" % (dwi_dir, '/xfms/MNI2diff_affine.nii.gz')

    # Create transform matrix between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = input_MNI
    flirt.inputs.out_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.out_file = '/tmp/out_flirt.nii.gz'
    flirt.run()

    # Apply transform between diff and MNI using FLIRT
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = input_MNI
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.out_file = out_aff
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.run()
    return out_aff


# Create avoidance and waypoints masks
def gen_anat_segs(anat_loc, out_aff):
    # # Custom inputs# #
    try:
        FSLDIR = os.environ['FSLDIR']
    except KeyError:
        print('FSLDIR environment variable not set!')
    import nipype.interfaces.fsl as fsl
    #import nibabel as nib
    #from nilearn.image import resample_img
    from nipype.interfaces.fsl import ExtractROI
    print('\nSegmenting anatomical image to create White Matter and Ventricular CSF masks for contraining the tractography...')
    # Create MNI ventricle mask
    print('Creating MNI space ventricle mask...')
    anat_dir = os.path.dirname(anat_loc)
    lvent_out_file = "%s%s" % (anat_dir, '/LVentricle.nii.gz')
    rvent_out_file = "%s%s" % (anat_dir, '/RVentricle.nii.gz')
    MNI_atlas = "%s%s" % (FSLDIR, '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-1mm.nii.gz')
    fslroi1 = ExtractROI(in_file=MNI_atlas, roi_file=lvent_out_file, t_min=2, t_size=1)
    os.system(fslroi1.cmdline)
    fslroi2 = ExtractROI(in_file=MNI_atlas, roi_file=rvent_out_file, t_min=13, t_size=1)
    os.system(fslroi2.cmdline)
    mni_csf_loc = anat_dir + '/VentricleMask.nii.gz'
    args = "%s%s%s" % ('-add ', rvent_out_file, ' -thr 0.1 -bin -dilF')
    maths = fsl.ImageMaths(in_file=lvent_out_file, op_string=args, out_file=mni_csf_loc)
    os.system(maths.cmdline)

    # Segment anatomical (should be in MNI space)
    print('Segmenting anatomical image using FAST...')
    fastr = fsl.FAST()
    fastr.inputs.in_files = anat_loc
    fastr.inputs.img_type = 1
    fastr.run()
    old_file_csf = "%s%s" % (anat_loc.split('.nii.gz')[0], '_pve_0.nii.gz')
    new_file_csf = "%s%s" % (anat_dir, '/CSF.nii.gz')
    old_file_wm = "%s%s" % (anat_loc.split('.nii.gz')[0], '_pve_2.nii.gz')
    new_file_wm = "%s%s" % (anat_dir, '/WM.nii.gz')
    os.rename(old_file_csf, new_file_csf)
    os.rename(old_file_wm, new_file_wm)

    # Reslice to 1x1x1mm voxels
    #img=nib.load(anat_loc)
    #vox_sz = img.affine[0][0]
    #targ_aff = img.affine/(np.array([[int(abs(vox_sz)),1,1,1],[1,int(abs(vox_sz)),1,1],[1,1,int(abs(vox_sz)),1],[1,1,1,1]]))
    #new_file_csf_res = resample_img(new_file_csf, target_affine=targ_aff)
    #new_file_wm_res = resample_img(new_file_wm, target_affine=targ_aff)
    #nib.save(new_file_csf_res, new_file_csf)
    #nib.save(new_file_wm_res, new_file_wm)
    return new_file_csf, mni_csf_loc, new_file_wm


def coreg_vent_CSF_to_diff(dwi_dir, csf_loc, mni_csf_loc):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    print('\nTransforming CSF mask to diffusion space...')
    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    csf_mask_diff_out = "%s%s" % (dwi_dir, '/csf_diff.nii.gz')
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = csf_loc
    flirt.inputs.out_file = csf_mask_diff_out
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.apply_xfm = True
    flirt.run()

    vent_mask_diff_out = dwi_dir + '/ventricle_diff.nii.gz'
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = mni_csf_loc
    flirt.inputs.out_file = vent_mask_diff_out
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.apply_xfm = True
    flirt.run()

    # Mask CSF with MNI ventricle mask
    print('Masking CSF with ventricle mask...')
    vent_csf_diff_out = "%s%s" % (dwi_dir, '/vent_csf_diff.nii.gz')
    args = "%s%s" % ('-mas ', vent_mask_diff_out)
    maths = fsl.ImageMaths(in_file=csf_mask_diff_out, op_string=args, out_file=vent_csf_diff_out)
    os.system(maths.cmdline)

    print('Eroding and binarizing CSF mask...')
    # Erode CSF mask
    out_file_final = "%s%s" % (vent_csf_diff_out.split('.nii.gz')[0], '_ero.nii.gz')
    args = '-ero -bin'
    maths = fsl.ImageMaths(in_file=vent_csf_diff_out, op_string=args, out_file=out_file_final)
    os.system(maths.cmdline)
    return out_file_final


def coreg_WM_mask_to_diff(dwi_dir, wm_mask_loc):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    print('\nTransforming White-Matter waypoint mask to diffusion space...')
    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    out_file = "%s%s" % (dwi_dir, '/wm_mask_diff.nii.gz')
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = wm_mask_loc
    flirt.inputs.out_file = out_file
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.apply_xfm = True
    flirt.run()
    args = '-bin'
    maths = fsl.ImageMaths(in_file=out_file, op_string=args, out_file=out_file)
    print('\nBinarizing WM mask...')
    os.system(maths.cmdline)
    return out_file


def coreg_mask_to_diff(dwi_dir, mask):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    print('\nTransforming custom waypoint mask to diffusion space...')
    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    out_file = "%s%s" % (dwi_dir, '/mask_custom_diff.nii.gz')
    flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
    flirt.inputs.reference = dwi_infile
    flirt.inputs.in_file = mask
    flirt.inputs.out_file = out_file
    flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
    flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
    flirt.inputs.apply_xfm = True
    flirt.run()
    args = '-bin'
    maths = fsl.ImageMaths(in_file=out_file, op_string=args, out_file=out_file)
    print('\nBinarizing custom mask...')
    os.system(maths.cmdline)
    return out_file


def build_coord_list(coords, dwi_dir):
    from nilearn import masking
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')

    x_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3, 0:3])[0]
    y_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3, 0:3])[1]
    z_vox = np.diagonal(masking._load_mask_img(nodif_brain_mask_path)[1][:3, 0:3])[2]

    def mmToVox(mmcoords):
        voxcoords = ['', '', '']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    # Convert coords back to voxels
    coords_vox = []
    for coord in coords:
        coords_vox.append(mmToVox(coord))
    coords = list(tuple(x) for x in coords_vox)
    coord_num = len(coords)

    return coords, coord_num


def reg_coords2diff(coords, dwi_dir, node_size, seeds_dir):
    try:
        FSLDIR = os.environ['FSLDIR']
    except KeyError:
        print('FSLDIR environment variable not set!')

    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe

    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    # #print(coords)
    # # Grow spheres at ROI
    for coord in coords[0]:
        X = coord[0]
        Y = coord[1]
        Z = coord[2]

        out_file1 = "%s%s%s%s%s%s%s%s" % (seeds_dir, '/roi_point_', str(X), '_', str(Y), '_', str(Z), '.nii.gz')
        args = "%s%s%s%s%s%s%s" % ('-mul 0 -add 1 -roi ', str(X), ' 1 ', str(Y), ' 1 ', str(Z), ' 1 0 1')
        maths = fsl.ImageMaths(in_file=FSLDIR + '/data/standard/MNI152_T1_1mm_brain.nii.gz', op_string=args,
                               out_file=out_file1)
        os.system(maths.cmdline + ' -odt float')

        out_file2 = "%s%s%s%s%s%s%s%s" % (seeds_dir, '/region_', str(X), '_', str(Y), '_', str(Z), '.nii.gz')
        args = "%s%s%s" % ('-kernel sphere ', str(node_size), ' -fmean -bin')
        maths = fsl.ImageMaths(in_file=out_file1, op_string=args, out_file=out_file2)
        os.system(maths.cmdline + ' -odt float')

        # Map ROIs from Standard Space to diffusion Space:
        # Applying xfm and input matrix to transform ROI's between diff and MNI using FLIRT,
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
        flirt.inputs.reference = dwi_infile
        flirt.inputs.in_file = out_file2
        out_file_diff = "%s%s" % (out_file2.split('.nii')[0], '_diff.nii.gz')
        flirt.inputs.out_file = out_file_diff
        flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
        flirt.run()

        args = '-bin'
        maths = fsl.ImageMaths(in_file=out_file_diff, op_string=args, out_file=out_file_diff)
        os.system(maths.cmdline)

    done_nodes = True
    return done_nodes


def cleanup_tmp_nodes(done_nodes, seeds_dir, coords, dir_path):
    import glob
    import os
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle

    pick_dump = True

    del_files_points = glob.glob(seeds_dir + '/roi_point*')
    if len(del_files_points) > 0:
        for i in del_files_points:
            os.remove(i)

    seeds_list = [i for i in os.listdir(seeds_dir) if 'diff.nii.gz' in i]
    if pick_dump is True:
        # Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/coords.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)

    return seeds_list


def reg_parcels2diff(dwi_dir, seeds_dir):
    import re
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    if os.path.exists(merged_f_samples_path) is True:
        dwi_infile = merged_f_samples_path
    else:
        dwi_infile = "%s%s" % (dwi_dir, '/dwi.nii.gz')

    seeds_list = [i for i in os.listdir(seeds_dir) if '.nii' in i]

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    seeds_list.sort(key=natural_keys)
    i = 0
    for parcel in seeds_list:
        out_file = "%s%s%s%s" % (seeds_dir, '/region_', str(i), '_diff.nii.gz')
        flirt = pe.Node(interface=fsl.FLIRT(cost_func='mutualinfo'), name='coregister')
        flirt.inputs.reference = dwi_infile
        flirt.inputs.in_file = "%s%s%s" % (seeds_dir, '/', parcel)
        flirt.inputs.out_file = out_file
        flirt.inputs.out_matrix_file = '/tmp/out_flirt.mat'
        flirt.inputs.apply_xfm = True
        flirt.inputs.in_matrix_file = "%s%s" % (dwi_dir, '/xfms/MNI2diff.mat')
        flirt.run()
        i = i + 1
    seeds_list = [i for i in os.listdir(seeds_dir) if 'diff.nii' in i]
    return seeds_list


def create_seed_mask_file(node_size, network, dir_path, parc, seeds_list, atlas_select):
    import glob
    import re
    import shutil

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    if parc is True:
        node_size = 'parc'

    if network:
        seeds_dir = "%s%s%s%s%s%s%s" % (dir_path, '/seeds_', network, '_', atlas_select, '_', str(node_size))
    else:
        seeds_dir = "%s%s%s%s%s" % (dir_path, '/seeds_', atlas_select, '_', str(node_size))

    if network:
        probtrackx_output_dir_path = "%s%s%s%s%s" % (dir_path, '/probtrackx_', str(node_size), '_', network)
    else:
        probtrackx_output_dir_path = "%s%s%s" % (dir_path, '/probtrackx_WB_', str(node_size))

    if os.path.exists(probtrackx_output_dir_path):
        shutil.rmtree(probtrackx_output_dir_path)
    os.makedirs(probtrackx_output_dir_path)

    seed_files = glob.glob(seeds_dir + '/region*diff.nii.gz')
    seeds_text = "%s%s" % (probtrackx_output_dir_path, '/masks.txt')
    try:
        os.remove(seeds_text)
    except OSError:
        pass
    seeds_file_list = []
    for seed_file in seed_files:
        seeds_file_list.append(seed_file)
    seeds_file_list.sort(key=natural_keys)
    f = open(seeds_text, 'w')
    l1 = map(lambda x:x+'\n', seeds_file_list)
    f.writelines(l1)
    f.close()
    return seeds_text, probtrackx_output_dir_path


def save_parcel_vols(parcel_list, net_parcels_map_nifti, dir_path, seeds_dir):
    import os
    import time
    if net_parcels_map_nifti:
        net_parcels_map_file = "%s%s" % (dir_path, '/net_parcels_map_nifti.nii.gz')
        nib.save(net_parcels_map_nifti, net_parcels_map_file)

    if not os.path.exists(seeds_dir):
        os.mkdir(seeds_dir)
    num_vols = len(parcel_list)
    i = 0
    for vol in parcel_list:
        out_path = "%s%s%s%s" % (seeds_dir, '/region_', str(i), '.nii.gz')
        nib.save(vol, out_path)
        i = i + 1
    num_vols_pres = float(len(next(os.walk(seeds_dir))[2]))
    while num_vols_pres < float(num_vols):
        time.sleep(1)
        num_vols_pres = float(len(next(os.walk(seeds_dir))[2]))
    return num_vols_pres


def prepare_masks(dwi_dir, csf_loc, mni_csf_loc, wm_mask_loc, mask):
    from pynets import diffconnectometry
    if (csf_loc and mni_csf_loc) is None:
        vent_CSF_diff_mask_path = None
    else:
        vent_CSF_diff_mask_path = diffconnectometry.coreg_vent_CSF_to_diff(dwi_dir, csf_loc, mni_csf_loc)

    # Use custom waypoint mask (if present)
    if mask:
        way_mask = diffconnectometry.coreg_mask_to_diff(dwi_dir, mask)
    elif wm_mask_loc is None:
        way_mask = None
    else:
        way_mask = diffconnectometry.coreg_WM_mask_to_diff(dwi_dir, wm_mask_loc)

    return vent_CSF_diff_mask_path, way_mask


def prep_nodes(dwi_dir, node_size, parc, parcel_list, net_parcels_map_nifti, network, dir_path, mask, atlas_select):
    import shutil
    from pynets import diffconnectometry, nodemaker

    if parc is True:
        node_size = 'parc'

    if network:
        seeds_dir = "%s%s%s%s%s%s%s" % (dir_path, '/seeds_', network, '_', atlas_select, '_', str(node_size))
    else:
        seeds_dir = "%s%s%s%s%s" % (dir_path, '/seeds_', atlas_select, '_', str(node_size))

    if os.path.exists(seeds_dir) is True:
        shutil.rmtree(seeds_dir)
    if not os.path.exists(seeds_dir):
        os.mkdir(seeds_dir)

    if parc is True:
        # If masking was performed, get reduced list
        if mask or network:
            [_, _, parcel_list] = nodemaker.get_names_and_coords_of_parcels_from_img(net_parcels_map_nifti)

        diffconnectometry.save_parcel_vols(parcel_list, net_parcels_map_nifti, dir_path, seeds_dir)

    return parcel_list, seeds_dir, node_size


def run_probtrackx2(i, seeds_text, dwi_dir, probtrackx_output_dir_path, procmem, num_total_samples, vent_CSF_diff_mask_path=None, way_mask=None):
    import random
    import nipype.interfaces.fsl as fsl
    samples_i = int(round(float(num_total_samples) / float(procmem[0]), 0))
    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
    merged_th_samples_path = "%s%s" % (dwi_dir, '/merged_th1samples.nii.gz')
    merged_f_samples_path = "%s%s" % (dwi_dir, '/merged_f1samples.nii.gz')
    merged_ph_samples_path = "%s%s" % (dwi_dir, '/merged_ph1samples.nii.gz')

    tmp_dir = "%s%s%s" % (probtrackx_output_dir_path, '/tmp_samples_', str(i))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    probtrackx2 = fsl.ProbTrackX2()
    probtrackx2.inputs.network = True
    probtrackx2.inputs.seed = seeds_text
    probtrackx2.inputs.onewaycondition = True
    probtrackx2.inputs.c_thresh = 0.2
    probtrackx2.inputs.n_steps = 2000
    probtrackx2.inputs.step_length = 0.5
    probtrackx2.inputs.n_samples = samples_i
    probtrackx2.inputs.dist_thresh = 0.0
    probtrackx2.inputs.opd = True
    probtrackx2.inputs.loop_check = True
    probtrackx2.inputs.omatrix1 = False
    probtrackx2.overwrite = True
    probtrackx2.inputs.verbose = False
    probtrackx2.inputs.mask = nodif_brain_mask_path
    probtrackx2.inputs.out_dir = tmp_dir
    probtrackx2.inputs.thsamples = merged_th_samples_path
    probtrackx2.inputs.fsamples = merged_f_samples_path
    probtrackx2.inputs.phsamples = merged_ph_samples_path
    probtrackx2.inputs.use_anisotropy = False

    if vent_CSF_diff_mask_path:
        probtrackx2.inputs.avoid_mp = vent_CSF_diff_mask_path
    else:
        print('No ventricular CSF mask used. This is not recommended.')

    if way_mask:
        probtrackx2.inputs.waypoints = way_mask
        probtrackx2.inputs.waycond = 'OR'
        print('No waypointmask used. This will instantiate a computationally expensive probtrackx run and is generally not recommended.')

    rseed_arg = ' --rseed=' + str(random.randint(1, 1000))
    os.chdir(dwi_dir)
    os.system(probtrackx2.cmdline + rseed_arg)
    del probtrackx2
    return


def dwi_dipy_run(dwi_dir, node_size, dir_path, conn_model, parc, atlas_select, network, wm_mask=None):
    import os
    import glob
    import re
    import nipype.interfaces.fsl as fsl
    from dipy.reconst.dti import TensorModel, quantize_evecs
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response
    from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
    from dipy.tracking import utils
    from dipy.direction import peaks_from_model
    from dipy.tracking.eudx import EuDX
    from dipy.data import get_sphere
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    dwi_img = "%s%s" % (dwi_dir, '/dwi.nii.gz')
    nodif_brain_mask_path = "%s%s" % (dwi_dir, '/nodif_brain_mask.nii.gz')
    bvals = "%s%s" % (dwi_dir, '/bval')
    bvecs = "%s%s" % (dwi_dir, '/bvec')

    img = nib.load(dwi_img)
    data = img.get_data()

    # Loads mask and ensures it's a true binary mask
    img = nib.load(nodif_brain_mask_path)
    mask = img.get_data()
    mask = mask > 0

    [bvals, bvecs] = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table(bvals, bvecs)

    # Estimates some tensors
    model = TensorModel(gtab)
    ten = model.fit(data, mask)
    sphere = get_sphere('symmetric724')
    ind = quantize_evecs(ten.evecs, sphere.vertices)

    # Tractography
    if conn_model == 'csd':
        trac_mod = 'csd'
    else:
        conn_model = 'tensor'
        trac_mod = ten.fa

    affine = img.affine
    print('Tracking with tensor model...')
    if wm_mask is None:
        mask = nib.load(mask).get_data()
        mask[0, :, :] = False
        mask[:, 0, :] = False
        mask[:, :, 0] = False
        seeds = utils.seeds_from_mask(mask, density=2)
    else:
        wm_mask_data = nib.load(wm_mask).get_data()
        wm_mask_data[0, :, :] = False
        wm_mask_data[:, 0, :] = False
        wm_mask_data[:, :, 0] = False
        seeds = utils.seeds_from_mask(wm_mask_data, density=2)
    #seeds = random_seeds_from_mask(ten.fa > 0.3, seeds_count=num_total_samples)

    if conn_model == 'tensor':
        eu = EuDX(a=trac_mod, ind=ind, seeds=seeds, odf_vertices=sphere.vertices,
                  a_low=0.05, step_sz=.5)
        tracks = [e for e in eu]
    elif conn_model == 'csd':
        print('Tracking with CSD model...')
        if wm_mask is None:
            response = recursive_response(gtab, data, mask=mask.astype('bool'), sh_order=8, peak_thr=0.01, init_fa=0.08,
                                          init_trace=0.0021, iter=8, convergence=0.001, parallel=True)
        else:
            response = recursive_response(gtab, data, mask=wm_mask_data.astype('bool'), sh_order=8, peak_thr=0.01,
                                          init_fa=0.08, init_trace=0.0021, iter=8, convergence=0.001, parallel=True)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        csd_peaks = peaks_from_model(model=csd_model, data=data, sphere=sphere, relative_peak_threshold=.5,
                                     min_separation_angle=25, parallel=True)
        tissue_classifier = ThresholdTissueClassifier(ten.fa, 0.1)
        streamline_generator = LocalTracking(csd_peaks, tissue_classifier, seeds, affine=affine, step_size=0.5)
        tracks = [e for e in streamline_generator]

    if parc is True:
        node_size = 'parc'

    if network:
        seeds_dir = "%s%s%s%s%s%s%s" % (dir_path, '/seeds_', network, '_', atlas_select, '_', str(node_size))
    else:
        seeds_dir = "%s%s%s%s%s" % (dir_path, '/seeds_', atlas_select, '_', str(node_size))

    seed_files = glob.glob("%s%s" % (seeds_dir, '/*diff.nii.gz'))

    seed_files.sort(key=natural_keys)

    # Binarize ROIs
    print('\nBinarizing seed masks...')
    j = 1
    for i in seed_files:
        args = ' -bin '
        out_file = "%s%s" % (i.split('.nii.gz')[0], '_bin.nii.gz')
        maths = fsl.ImageMaths(in_file=i, op_string=args, out_file=out_file)
        os.system(maths.cmdline)
        args = ' -mul ' + str(j)
        maths = fsl.ImageMaths(in_file=out_file, op_string=args, out_file=out_file)
        os.system(maths.cmdline)
        j = j + 1

    # Create atlas from ROIs
    seed_files = glob.glob("%s%s" % (seeds_dir, '/*diff_bin.nii.gz'))

    seed_files.sort(key=natural_keys)

    print('\nMerging seed masks into single labels image...')
    label_sum = "%s%s" % (seeds_dir, '/all_rois.nii.gz')
    args = ' -add ' + i
    maths = fsl.ImageMaths(in_file=seed_files[0], op_string=args, out_file=label_sum)
    os.system(maths.cmdline)

    for i in seed_files:
        args = ' -add ' + i
        maths = fsl.ImageMaths(in_file=label_sum, op_string=args, out_file=label_sum)
        os.system(maths.cmdline)

    labels_im = nib.load(label_sum)
    labels_data = labels_im.get_data().astype('int')
    conn_matrix, grouping = utils.connectivity_matrix(tracks, labels_data, affine=affine, return_mapping=True,
                                                      mapping_as_streamlines=True)
    conn_matrix[:3, :] = 0
    conn_matrix[:, :3] = 0

    return conn_matrix


def collect_struct_mapping_outputs(parc, dwi_dir, network, ID, probtrackx_output_dir_path, dir_path, procmem, seeds_dir):
    import os
    import time
    import glob
    import nibabel as nib
    from sklearn.preprocessing import normalize

    tmp_files = []
    for i in [str(x) for x in range(int(procmem[0]))]:
        tmp_files.append("%s%s%s%s" % (probtrackx_output_dir_path, '/tmp_samples_', str(i), '/fdt_paths.nii.gz'))

    while True:
       if all([os.path.isfile(f) for f in tmp_files]):
          break
       else:
          print('Waiting on all fdt_paths!')
          print("%s%s%s%s" % (str(sum([os.path.isfile(f) for f in tmp_files])), str(' tmp files produced out of '), str(len(tmp_files)), ' expected...'))
          time.sleep(10)

    output_fdts = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_paths.nii.gz')
    net_mats = glob.glob(probtrackx_output_dir_path + '/tmp*/fdt_network_matrix')
    waytotals = glob.glob(probtrackx_output_dir_path + '/tmp*/waytotal')

    try:
        # Add the images
        out_file = "%s%s" % (probtrackx_output_dir_path, '/fdt_paths.nii.gz')
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

    # Merge output mat files
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
    i = 0
    for mx in mats_list:
        waytotal = waytotals_list[i]
        np.seterr(divide='ignore', invalid='ignore')
        conn_matrix = np.divide(mx, waytotal)
        conn_matrix[np.isnan(conn_matrix)] = 0
        conn_matrix[np.isinf(conn_matrix)] = 0
        conn_matrix = np.nan_to_num(conn_matrix)
        conn_matrices.append(conn_matrix)
        i = i + 1

    conn_matri = conn_matrices[0]
    j = 0
    for i in range(len(conn_matrices))[1:]:
        try:
            conn_matri = conn_matri + conn_matrices[i]
            j = j + 1
        except:
            print("%s%s%s%s%s" % ('Matrix ', str(i + 1), ' is a different shape: ', str(conn_matrices[i].shape),
                                  '. Skipping...'))
            continue

    mx_mean = conn_matri / float(j)

    try:
        print('Normalizing array...')
        conn_matrix = normalize(np.nan_to_num(mx_mean))
    except RuntimeError:
        print('Normalization failed...')
        pass
    conn_matrix_symm = np.maximum(conn_matrix, conn_matrix.transpose())

    if parc is False:
        del_files_spheres = glob.glob(seeds_dir + '/roi_sphere*')
        for i in del_files_spheres:
            os.remove(i)
    else:
        del_files_parcels = glob.glob(seeds_dir + '/roi_parcel*')
        if len(del_files_parcels) > 0:
            for i in del_files_parcels:
                os.remove(i)

    if network is not None:
        est_path_struct = "%s%s%s%s%s%s" % (dir_path, '/', ID, '_', network, '_struct_est_unthr_unsymm.npy')
    else:
        est_path_struct = "%s%s%s%s" % (dir_path, '/', ID, '_struct_est_unthr_unsymm.npy')
    try:
        np.save(est_path_struct, conn_matrix)
    except RuntimeError:
        print('Workflow error. Exiting...')
    return conn_matrix_symm
