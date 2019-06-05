# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import warnings
warnings.filterwarnings("ignore")
import os
import nibabel as nib
import numpy as np
from pynets.registration import reg_utils as mgru
from pynets import utils
from nilearn.image import load_img, math_img
try:
    FSLDIR = os.environ['FSLDIR']
except KeyError:
    print('FSLDIR environment variable not set!')


def transform_pts(pts, t_aff, t_warp, ref_img_path, ants_path, template_path, dsn_dir,
                  out_volume="", output_space="ras_voxels"):
    """

    :param pts:
    :param t_aff:
    :param t_warp:
    :param ref_img_path:
    :param ants_path:
    :param template_path:
    :param dsn_dir:
    :param out_volume:
    :param output_space:
    :return:
    """
    import random
    """
    return coordinates in
    "ras_voxels" if you want to streamlines in ras ijk coordinates or
    "lps_voxmm" if you want dsi studio streamline coordinates relative to the template
    """
    if not output_space in ("ras_voxels", "lps_voxmm"):
        raise ValueError("Must specify output space")

    run_hash = random.randint(1, 1000)
    warped_csv_out = dsn_dir + "/warped_output.csv"
    aattp_out = dsn_dir + '/aattp_' + str(run_hash) + '.csv'
    transforms = "-t [" + str(t_aff) + ", 1] " + "-t " + str(t_warp)

    # Load the volume from .trk
    ref_img = nib.load(ref_img_path)
    voxel_size = np.array(ref_img.header.get_zooms())
    extents = np.array(ref_img.shape)
    extents[-1] = 0

    # Convert the streamlines to voxel indices, then to ants points
    voxel_coords = abs(extents - pts / voxel_size)
    ants_mult = np.array([voxel_size[0], voxel_size[1], voxel_size[2]])
    ants_coord = voxel_coords * ants_mult - voxel_size[0]
    ants_coord[:, 0] = -ants_coord[:, 0]
    ants_coord[:, 1] = -ants_coord[:, 1]

    # Save the ants coordinates to a csv, then warp them
    np.savetxt(warped_csv_out, np.hstack([ants_coord, np.zeros((ants_coord.shape[0], 1))]),
               header="x,y,z,t", delimiter=",", fmt="%f")

    # Apply the trandforms to
    cmd = ants_path + "/antsApplyTransformsToPoints " + "-d 3 -i " + warped_csv_out + " -o " + aattp_out + " " + transforms
    os.system(cmd)

    # Load template to get output space
    template = nib.load(template_path)
    warped_affine = template.affine

    adjusted_affine = warped_affine.copy()
    adjusted_affine[0] = -adjusted_affine[0]
    adjusted_affine[1] = -adjusted_affine[1]

    ants_warped_coords = np.loadtxt(aattp_out, skiprows=1, delimiter=",")[:, :3]
    os.remove(aattp_out)
    to_transform = np.hstack([ants_warped_coords, np.ones((ants_warped_coords.shape[0], 1))])
    new_voxels = (np.dot(np.linalg.inv(adjusted_affine), to_transform.T) + warped_affine[0, 0])[:3]

    # Write out an image
    if out_volume:
        newdata = np.zeros(template.shape)
        ti, tj, tk = new_voxels.astype(np.int)
        newdata[ti, tj, tk] = 1
        warped_out = nib.Nifti1Image(newdata, warped_affine).to_filename(out_volume)
    if output_space == "ras_voxels":
        return new_voxels.astype(np.int).T

    elif output_space == "lps_voxmm":
        template_extents = template.shape
        lps_voxels = new_voxels.copy()
        lps_voxels[0] = template_extents[0] - lps_voxels[0]
        lps_voxels[1] = template_extents[1] - lps_voxels[1]
        lps_voxmm = lps_voxels.T * np.array(template.header.get_zooms())[:3]
        return lps_voxmm


class Warp(object):
    """

    """
    def __init__(self, ants_path="", file_in="", file_out="", template_path="", t_aff="", t_warp="", ref_img_path="",
                 dsn_dir=""):
        self.ants_path = ants_path
        self.file_in = file_in
        self.file_out = file_out
        self.template_path = template_path
        self.t_aff = t_aff
        self.t_warp = t_warp
        self.ref_img_path = ref_img_path
        self.dsn_dir = dsn_dir

    def streamlines(self):
        """

        :return:
        """
        from pynets.registration.register import transform_pts
        if not self.file_in.endswith((".trk", ".trk.gz")):
            print("File format currently unsupported.")
            return

        if self.ref_img_path == "":
            print("Specify reference image path: .ref_img_path = path to reference image")
            return

        print("Warping streamline file " + self.file_in)
        template = nib.load(self.template_path)
        warped_affine = template.affine
        dims = template.header.get_data_shape()
        template_trk_header = np.array(('TRACK',
                                        [dims[0], dims[1], dims[2]],
                                        [warped_affine[0][0], warped_affine[1][1], warped_affine[2][2]],
                                        [0.0, 0.0, 0.0], 0, ['', '', '', '', '', '', '', '', '', ''],
                                        0, ['', '', '', '', '', '', '', '', '', ''],
                                        [[1.0, 0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0, 1.0]], '', 'LPS', 'LPS',
                                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        '', '', '', '', '', '', '', 10000, 2, 1000),
                                       dtype=[('id_string', 'S6'), ('dim', '<i2', (3,)),
                                              ('voxel_size', '<f4', (3,)), ('origin', '<f4', (3,)),
                                              ('n_scalars', '<i2'), ('scalar_name', 'S20', (10,)),
                                              ('n_properties', '<i2'), ('property_name', 'S20', (10,)),
                                              ('vox_to_ras', '<f4', (4, 4)), ('reserved', 'S444'),
                                              ('voxel_order', 'S4'), ('pad2', 'S4'),
                                              ('image_orientation_patient', '<f4', (6,)),
                                              ('pad1', 'S2'), ('invert_x', 'S1'), ('invert_y', 'S1'),
                                              ('invert_z', 'S1'), ('swap_xy', 'S1'), ('swap_yz', 'S1'),
                                              ('swap_zx', 'S1'), ('n_count', '<i4'), ('version', '<i4'),
                                              ('hdr_size', '<i4')]
                                       )

        streams, hdr = nib.trackvis.read(self.file_in)
        offsets = []
        _streams = []
        for sl in streams:
            _streams.append(sl[0])
            offsets.append(_streams[-1].shape[0])
        allpoints = np.vstack(_streams)
        tx_points = transform_pts(allpoints, self.t_aff, self.t_warp, self.ref_img_path, self.ants_path,
                                  self.template_path, self.dsn_dir, output_space="lps_voxmm")
        offsets = np.cumsum([0] + offsets)
        starts = offsets[:-1]
        stops = offsets[1:]
        new_hdr = template_trk_header.copy()
        new_hdr["n_count"] = len(_streams)
        nib.trackvis.write(self.file_out, [(tx_points[a:b], None, None) for a, b in zip(starts, stops)],
                           hdr_mapping=new_hdr)
        print("Finished " + self.file_out)


def direct_streamline_norm(streams, fa_path, dir_path, track_type, target_samples, conn_model, network, node_size,
                           dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select,
                           label_names, coords, norm, binary, atlas_mni, basedir_path, curv_thr_list, step_list):
    """

    :param streams:
    :param fa_path:
    :param dir_path:
    :param track_type:
    :param target_samples:
    :param conn_model:
    :param network:
    :param node_size:
    :param dens_thresh:
    :param ID:
    :param roi:
    :param min_span_tree:
    :param disp_filt:
    :param parc:
    :param prune:
    :param atlas_select:
    :param uatlas_select:
    :param label_names:
    :param coords:
    :param norm:
    :param binary:
    :param atlas_mni:
    :param basedir_path:
    :param curv_thr_list:
    :param step_list:
    :return:
    """
    from dipy.tracking.streamline import Streamlines
    from pynets.registration import reg_utils as mgru
    from pynets.registration.register import Warp
    import pkg_resources
    '''Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of different spatial normalization approaches on 
    tractography and structural brain networks. Network Neuroscience, 1-19.'''
    template_path = pkg_resources.resource_filename("pynets", "templates/FSL_HCP1065_FA_2mm.nii.gz")
    try:
        ants_path = os.environ['ANTSPATH']
    except KeyError:
        print('ANTSDIR environment variable not set!')

    dsn_dir = "%s%s" % (basedir_path, '/tmp/DSN')
    if not os.path.isdir(dsn_dir):
        os.mkdir(dsn_dir)

    streams_mni = "%s%s%s%s%s%s%s%s%s%s%s%s" % (dir_path, '/streamlines_mni_', conn_model, '_', target_samples,
                                                '_', node_size, 'mm_curv', str(curv_thr_list).replace(', ', '_'),
                                                '_step', str(step_list).replace(', ', '_'), '.trk')

    # Run ANTs reg
    t_aff = "%s%s" % (basedir_path, '/0GenericAffine.mat')
    t_warp = "%s%s" % (basedir_path, '/1Warp.nii.gz')

    if (os.path.isfile(t_aff) is False) and (os.path.isfile(t_warp) is False):
        cmd = ants_path + '/antsRegistrationSyN.sh -d 3 -f ' + template_path + ' -m ' + fa_path + ' -o ' + basedir_path + '/'
        os.system(cmd)

    fa_path_img = nib.load(fa_path)
    s_aff = fa_path_img.affine

    # Warp streamlines
    wS = Warp(ants_path, streams, streams_mni, template_path, t_aff, t_warp, fa_path, dsn_dir)
    wS.streamlines()

    # Flip FOV and translate along x-plane
    s_aff[0][0] = np.negative(np.sign(s_aff[0][0])) * np.abs(s_aff[0][0])
    s_aff[:3, 3] = np.array([270, 0, 0])
    streamlines_mni = nib.streamlines.load(streams_mni)
    streamlines_mni_s = streamlines_mni.streamlines
    streamlines_trans1 = Streamlines(mgru.transform_to_affine(streamlines_mni_s, streamlines_mni.header, s_aff))

    # Flip back x-dimension and zero offsets
    s_aff[0][0] = np.negative(np.sign(s_aff[0][0])) * np.abs(s_aff[0][0])
    s_aff[:3, 3] = np.array([0, 0, 0])
    streamlines_trans2 = Streamlines(mgru.transform_to_affine(streamlines_trans1, streamlines_mni.header, s_aff))
    streams_warp = streams_mni.split('.trk')[0] + '_warped.trk'
    tractogram = nib.streamlines.Tractogram(streamlines_trans2, affine_to_rasmm=np.eye(4))
    trkfile = nib.streamlines.trk.TrkFile(tractogram, header=streamlines_mni.header)
    nib.streamlines.save(trkfile, streams_warp)
    print(streams_warp)

    return streams_warp, dir_path, track_type, target_samples, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas_select, uatlas_select, label_names, coords, norm, binary, atlas_mni


class DmriReg(object):
    def __init__(self, basedir_path, fa_path, nodif_B0_mask, anat_file, vox_size, simple):
        self.simple = simple
        self.fa_path = fa_path
        self.nodif_B0_mask = nodif_B0_mask
        self.t1w = anat_file
        self.vox_size = vox_size
        self.t1w_name = 't1w'
        self.dwi_name = 'dwi'
        self.basedir_path = basedir_path
        self.tmp_path = "%s%s" % (basedir_path, '/tmp')
        self.reg_path = "%s%s" % (basedir_path, '/tmp/reg')
        self.anat_path = "%s%s" % (basedir_path, '/anat_reg')
        self.reg_path_mat = "%s%s" % (self.reg_path, '/mats')
        self.reg_path_warp = "%s%s" % (self.reg_path, '/warps')
        self.reg_path_img = "%s%s" % (self.reg_path, '/imgs')
        self.t12mni_xfm_init = "{}/xfm_t1w2mni_init.mat".format(self.reg_path_mat)
        self.mni2t1_xfm_init = "{}/xfm_mni2t1w_init.mat".format(self.reg_path_mat)
        self.t12mni_xfm = "{}/xfm_t1w2mni.mat".format(self.reg_path_mat)
        self.mni2t1_xfm = "{}/xfm_mni2t1.mat".format(self.reg_path_mat)
        self.mni2t1w_warp = "{}/mni2t1w_warp.nii.gz".format(self.reg_path_warp)
        self.warp_t1w2mni = "{}/t1w2mni_warp.nii.gz".format(self.reg_path_warp)
        self.t1w2dwi = "{}/{}_in_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.t1_aligned_mni = "{}/{}_aligned_mni.nii.gz".format(self.anat_path, self.t1w_name)
        self.t1w_brain = "{}/{}_brain.nii.gz".format(self.anat_path, self.t1w_name)
        self.t1w_brain_mask = "{}/{}_brain_mask.nii.gz".format(self.anat_path, self.t1w_name)
        self.dwi2t1w_xfm = "{}/dwi2t1w_xfm.mat".format(self.reg_path_mat)
        self.t1w2dwi_xfm = "{}/t1w2dwi_xfm.mat".format(self.reg_path_mat)
        self.t1w2dwi_bbr_xfm = "{}/t1w2dwi_bbr_xfm.mat".format(self.reg_path_mat)
        self.dwi2t1w_bbr_xfm = "{}/dwi2t1w_bbr_xfm.mat".format(self.reg_path_mat)
        self.t1wtissue2dwi_xfm = "{}/t1wtissue2dwi_xfm.mat".format(self.reg_path_mat)
        self.xfm_atlas2t1w_init = "{}/{}_xfm_atlas2t1w_init.mat".format(self.reg_path_mat, self.t1w_name)
        self.xfm_atlas2t1w = "{}/{}_xfm_atlas2t1w.mat".format(self.reg_path_mat, self.t1w_name)
        self.temp2dwi_xfm = "{}/{}_xfm_temp2dwi.mat".format(self.reg_path_mat, self.dwi_name)
        self.temp2dwi_xfm = "{}/{}_xfm_temp2dwi.mat".format(self.reg_path_mat, self.dwi_name)
        self.map_path = "{}/{}_seg".format(self.anat_path, self.t1w_name)
        self.wm_mask = "{}/{}_wm.nii.gz".format(self.anat_path, self.t1w_name)
        self.wm_mask_thr = "{}/{}_wm_thr.nii.gz".format(self.anat_path, self.t1w_name)
        self.wm_edge = "{}/{}_wm_edge.nii.gz".format(self.anat_path, self.t1w_name)
        self.csf_mask = "{}/{}_csf.nii.gz".format(self.anat_path, self.t1w_name)
        self.gm_mask = "{}/{}_gm.nii.gz".format(self.anat_path, self.t1w_name)
        self.xfm_roi2mni_init = "{}/roi_2_mni.mat".format(self.reg_path_mat)
        self.lvent_out_file = "{}/LVentricle.nii.gz".format(self.reg_path_img)
        self.rvent_out_file = "{}/RVentricle.nii.gz".format(self.reg_path_img)
        self.mni_vent_loc = "{}/VentricleMask.nii.gz".format(self.reg_path_img)
        self.csf_mask_dwi = "{}/{}_csf_mask_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.gm_in_dwi = "{}/{}_gm_in_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.wm_in_dwi = "{}/{}_wm_in_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.csf_mask_dwi_bin = "{}/{}_csf_mask_dwi_bin.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.gm_in_dwi_bin = "{}/{}_gm_in_dwi_bin.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.wm_in_dwi_bin = "{}/{}_wm_in_dwi_bin.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.vent_mask_dwi = "{}/{}_vent_mask_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.vent_csf_in_dwi = "{}/{}_vent_csf_in_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.vent_mask_mni = "{}/vent_mask_mni.nii.gz".format(self.reg_path_img)
        self.vent_mask_t1w = "{}/vent_mask_t1w.nii.gz".format(self.reg_path_img)
        self.mni_atlas = "%s%s%s%s" % (FSLDIR, '/data/atlases/HarvardOxford/HarvardOxford-sub-prob-', vox_size, '.nii.gz')
        self.input_mni = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '.nii.gz')
        self.input_mni_brain = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain.nii.gz')
        self.input_mni_mask = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain_mask.nii.gz')
        self.wm_gm_int_in_dwi = "{}/{}_wm_gm_int_in_dwi.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.wm_gm_int_in_dwi_bin = "{}/{}_wm_gm_int_in_dwi_bin.nii.gz".format(self.reg_path_img, self.t1w_name)
        self.input_mni_sched = "%s%s" % (FSLDIR, '/etc/flirtsch/T1_2_MNI152_2mm.cnf')

        # Create empty tmp directories that do not yet exist
        reg_dirs = [self.tmp_path, self.reg_path, self.anat_path, self.reg_path_mat, self.reg_path_warp, self.reg_path_img]
        for i in range(len(reg_dirs)):
            if not os.path.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

        if os.path.isfile(self.t1w_brain) is False:
            import shutil
            shutil.copyfile(self.t1w, self.t1w_brain)

    def gen_tissue(self):
        # Segment the t1w brain into probability maps
        self.maps = mgru.segment_t1w(self.t1w_brain, self.map_path)
        self.wm_mask = self.maps['wm_prob']
        self.gm_mask = self.maps['gm_prob']
        self.csf_mask = self.maps['csf_prob']

        self.t1w_brain = utils.check_orient_and_dims(self.t1w_brain, self.vox_size)
        self.wm_mask = utils.check_orient_and_dims(self.wm_mask, self.vox_size)
        self.gm_mask = utils.check_orient_and_dims(self.gm_mask, self.vox_size)
        self.csf_mask = utils.check_orient_and_dims(self.csf_mask, self.vox_size)

        # Threshold WM to binary in dwi space
        self.t_img = load_img(self.wm_mask)
        self.mask = math_img('img > 0.2', img=self.t_img)
        self.mask.to_filename(self.wm_mask_thr)

        # Threshold T1w brain to binary in anat space
        self.t_img = load_img(self.t1w_brain)
        self.mask = math_img('img > 0.0', img=self.t_img)
        self.mask.to_filename(self.t1w_brain_mask)

        # Extract wm edge
        cmd = 'fslmaths ' + self.wm_mask_thr + ' -edge -bin -mas ' + self.wm_mask_thr + ' ' + self.wm_edge
        os.system(cmd)

        return

    def t1w2dwi_align(self):
        """
        alignment from T1w --> MNI and T1w_MNI --> DWI
        A function to perform self alignment. Uses a local optimisation
        cost function to get the two images close, and then uses bbr
        to obtain a good alignment of brain boundaries.
        Assumes input dwi is already preprocessed and brain extracted.
        """

        # Create linear transform/ initializer T1w-->MNI
        mgru.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm_init, bins=None, interp="spline", out=None,
                   dof=12, cost='mutualinfo', searchrad=True)

        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print('Running non-linear registration: T1w-->MNI ...')
                # Use FNIRT to nonlinearly align T1 to MNI template
                mgru.align_nonlinear(self.t1w_brain, self.input_mni, xfm=self.t12mni_xfm_init, out=self.t1_aligned_mni,
                                     warp=self.warp_t1w2mni, ref_mask=self.input_mni_mask,
                                     config=self.input_mni_sched)

                # Get warp from MNI -> T1
                mgru.inverse_warp(self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni)

                # Get mat from MNI -> T1
                cmd = 'convert_xfm -omat ' + self.mni2t1_xfm_init + ' -inverse ' + self.t12mni_xfm_init
                print(cmd)
                os.system(cmd)

            except RuntimeError('Error: FNIRT failed!'):
                pass
        else:
            # Falling back to linear registration
            mgru.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm, init=self.t12mni_xfm_init, bins=None,
                       dof=12, cost='mutualinfo', searchrad=True, interp="spline", out=self.t1_aligned_mni, sch=None)

        # Align T1w-->DWI
        mgru.align(self.fa_path, self.t1w_brain, xfm=self.t1w2dwi_xfm, bins=None, interp="spline", dof=6,
                   cost='mutualinfo', out=None, searchrad=True, sch=None)
        cmd = 'convert_xfm -omat ' + self.dwi2t1w_xfm + ' -inverse ' + self.t1w2dwi_xfm
        print(cmd)
        os.system(cmd)

        if self.simple is False:
            # Flirt bbr
            try:
                print('Running FLIRT BBR registration: T1w-->DWI ...')
                mgru.align(self.fa_path, self.t1w_brain, wmseg=self.wm_edge, xfm=self.dwi2t1w_bbr_xfm,
                           init=self.dwi2t1w_xfm, bins=256, dof=7, searchrad=True, interp="spline", out=None,
                           cost='bbr', finesearch=5, sch="${FSLDIR}/etc/flirtsch/bbr.sch")
                cmd = 'convert_xfm -omat ' + self.t1w2dwi_bbr_xfm + ' -inverse ' + self.dwi2t1w_bbr_xfm
                os.system(cmd)

                # Apply the alignment
                mgru.align(self.t1w_brain, self.fa_path, init=self.t1w2dwi_bbr_xfm, xfm=self.t1wtissue2dwi_xfm,
                           bins=None, interp="spline", dof=7, cost='mutualinfo', out=self.t1w2dwi, searchrad=True,
                           sch=None)
            except RuntimeError('Error: FLIRT BBR failed!'):
                pass
        else:
            # Apply the alignment
            mgru.align(self.t1w_brain, self.fa_path, init=self.t1w2dwi_xfm, xfm=self.t1wtissue2dwi_xfm, bins=None,
                       interp="spline", dof=6, cost='mutualinfo', out=self.t1w2dwi, searchrad=True, sch=None)

        return

    def atlas2t1w2dwi_align(self, uatlas_select, atlas_select):
        """
        alignment from atlas --> T1 --> dwi
        A function to perform atlas alignment.
        Tries nonlinear registration first, and if that fails,
        does a linear registration instead.
        NOTE: for this to work, must first have called t1w2dwi_align.
        """
        self.atlas = uatlas_select
        self.atlas_name = atlas_select
        self.aligned_atlas_t1mni = "{}/{}_t1w_mni.nii.gz".format(self.basedir_path, self.atlas_name)
        self.aligned_atlas_skull = "{}/{}_t1w_skull.nii.gz".format(self.anat_path, self.atlas_name)
        self.dwi_aligned_atlas = "{}/{}_dwi_track.nii.gz".format(self.reg_path_img, self.atlas_name)
        self.dwi_aligned_atlas_wmgm_int = "{}/{}_dwi_track_wmgm_int.nii.gz".format(self.reg_path_img, self.atlas_name)

        mgru.align(self.atlas, self.t1_aligned_mni, init=None, xfm=None, out=self.aligned_atlas_t1mni, dof=12,
                   searchrad=True, interp="nearestneighbour", cost='mutualinfo')

        if self.simple is False:
            try:
                # Apply warp resulting from the inverse of T1w-->MNI created earlier
                mgru.apply_warp(self.t1w_brain, self.aligned_atlas_t1mni, self.aligned_atlas_skull,
                                warp=self.mni2t1w_warp, interp='nn', sup=True)

                # Apply transform to dwi space
                mgru.align(self.aligned_atlas_skull, self.fa_path, init=self.t1wtissue2dwi_xfm, xfm=None,
                           out=self.dwi_aligned_atlas, dof=6, searchrad=True, interp="nearestneighbour",
                           cost='mutualinfo')
            except:
                print("Warning: Atlas is not in correct dimensions, or input is low quality,\nusing linear template registration.")

                # Create transform to align atlas to T1w using flirt
                mgru.align(self.atlas, self.t1w_brain, xfm=self.xfm_atlas2t1w_init, init=None, bins=None, dof=6,
                           cost='mutualinfo', searchrad=True, interp="spline", out=None, sch=None)
                mgru.align(self.atlas, self.t1_aligned_mni, xfm=self.xfm_atlas2t1w, out=None, dof=6, searchrad=True,
                           bins=None, interp="spline", cost='mutualinfo', init=self.xfm_atlas2t1w_init)

                # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a transform from atlas ->(-> t1w ->)-> dwi
                mgru.combine_xfms(self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm)

                # Apply linear transformation from template to dwi space
                mgru.applyxfm(self.fa_path, self.atlas, self.temp2dwi_xfm, self.dwi_aligned_atlas)
        else:
            # Create transform to align atlas to T1w using flirt
            mgru.align(self.atlas, self.t1w_brain, xfm=self.xfm_atlas2t1w_init, init=None, bins=None, dof=6,
                       cost='mutualinfo', searchrad=None, interp="spline", out=None, sch=None)
            mgru.align(self.atlas, self.t1w_brain, xfm=self.xfm_atlas2t1w, out=None, dof=6, searchrad=True, bins=None,
                       interp="spline", cost='mutualinfo', init=self.xfm_atlas2t1w_init)

            # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a transform from atlas ->(-> t1w ->)-> dwi
            mgru.combine_xfms(self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm)

            # Apply linear transformation from template to dwi space
            mgru.applyxfm(self.fa_path, self.atlas, self.temp2dwi_xfm, self.dwi_aligned_atlas)

        # Set intensities to int
        self.atlas_img = nib.load(self.dwi_aligned_atlas)
        self.atlas_data = self.atlas_img.get_fdata().astype('int')
        #node_num = len(np.unique(self.atlas_data))
        #self.atlas_data[self.atlas_data>node_num] = 0
        t_img = load_img(self.wm_gm_int_in_dwi)
        mask = math_img('img > 0', img=t_img)
        mask.to_filename(self.wm_gm_int_in_dwi_bin)
        nib.save(nib.Nifti1Image(self.atlas_data.astype(np.int32), affine=self.atlas_img.affine,
                                 header=self.atlas_img.header), self.dwi_aligned_atlas)
        cmd='fslmaths ' + self.dwi_aligned_atlas + ' -mas ' + self.nodif_B0_mask + ' -mas ' + self.wm_gm_int_in_dwi_bin + ' ' + self.dwi_aligned_atlas_wmgm_int
        os.system(cmd)

        return self.dwi_aligned_atlas_wmgm_int, self.dwi_aligned_atlas, self.aligned_atlas_t1mni

    def tissue2dwi_align(self):
        """
        alignment of ventricle ROI's from MNI space --> dwi and
        CSF from T1w space --> dwi
        A function to generate and perform dwi space alignment of avoidance/waypoint masks for tractography.
        First creates ventricle ROI. Then creates transforms from stock MNI template to dwi space.
        NOTE: for this to work, must first have called both t1w2dwi_align and atlas2t1w2dwi_align.
        """

        # Create MNI-space ventricle mask
        print('Creating MNI-space ventricle ROI...')
        if not os.path.isfile(self.mni_atlas):
            raise ValueError('FSL atlas for ventricle reference not found!')
        cmd='fslroi ' + self.mni_atlas + ' ' + self.rvent_out_file + ' 2 1'
        os.system(cmd)
        cmd='fslroi ' + self.mni_atlas + ' ' + self.lvent_out_file + ' 13 1'
        os.system(cmd)
        self.args = "%s%s%s" % (' -add ', self.rvent_out_file, ' -thr 0.1 -bin ')
        cmd='fslmaths ' + self.lvent_out_file + self.args + self.mni_vent_loc
        os.system(cmd)

        # Create transform to MNI atlas to T1w using flirt. This will be use to transform the ventricles to dwi space.
        mgru.align(self.mni_atlas, self.input_mni_brain, xfm=self.xfm_roi2mni_init, init=None, bins=None, dof=6,
                   cost='mutualinfo', searchrad=True, interp="spline", out=None)

        # Create transform to align roi to mni and T1w using flirt
        mgru.applyxfm(self.input_mni_brain, self.mni_vent_loc, self.xfm_roi2mni_init, self.vent_mask_mni)

        if self.simple is False:
            # Apply warp resulting from the inverse MNI->T1w created earlier
            mgru.apply_warp(self.t1w_brain, self.vent_mask_mni, self.vent_mask_t1w, warp=self.mni2t1w_warp,
                            interp='nn', sup=True)

        # Applyxfm tissue maps to dwi space
        mgru.applyxfm(self.fa_path, self.vent_mask_t1w, self.t1wtissue2dwi_xfm, self.vent_mask_dwi)
        mgru.applyxfm(self.fa_path, self.csf_mask, self.t1wtissue2dwi_xfm, self.csf_mask_dwi)
        mgru.applyxfm(self.fa_path, self.gm_mask, self.t1wtissue2dwi_xfm, self.gm_in_dwi)
        mgru.applyxfm(self.fa_path, self.wm_mask, self.t1wtissue2dwi_xfm, self.wm_in_dwi)

        # Threshold WM to binary in dwi space
        thr_img = nib.load(self.wm_in_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.2] = 0
        nib.save(thr_img, self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        thr_img = nib.load(self.gm_in_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.2] = 0
        nib.save(thr_img, self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        thr_img = nib.load(self.csf_mask_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.9] = 0
        nib.save(thr_img, self.csf_mask_dwi)

        # Threshold WM to binary in dwi space
        self.t_img = load_img(self.wm_in_dwi_bin)
        self.mask = math_img('img > 0', img=self.t_img)
        self.mask.to_filename(self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        self.t_img = load_img(self.gm_in_dwi_bin)
        self.mask = math_img('img > 0', img=self.t_img)
        self.mask.to_filename(self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        self.t_img = load_img(self.csf_mask_dwi)
        self.mask = math_img('img > 0', img=self.t_img)
        self.mask.to_filename(self.csf_mask_dwi_bin)

        # Create ventricular CSF mask
        print('Creating ventricular CSF mask...')
        cmd = 'fslmaths ' + self.vent_mask_dwi + ' -kernel sphere 10 -ero -bin ' + self.vent_mask_dwi
        os.system(cmd)
        cmd = 'fslmaths ' + self.csf_mask_dwi + ' -add ' + self.vent_mask_dwi + ' -bin ' + self.vent_csf_in_dwi
        os.system(cmd)

        # Create gm-wm interface image
        cmd = 'fslmaths ' + self.gm_in_dwi_bin + ' -mul ' + self.wm_in_dwi_bin + ' -mas ' + self.nodif_B0_mask + ' -bin ' + self.wm_gm_int_in_dwi
        os.system(cmd)

        return


class FmriReg(object):
    """

    """
    def __init__(self, basedir_path, anat_file, vox_size):
        self.t1w = anat_file
        self.vox_size = vox_size
        self.t1w_name = 't1w'
        self.basedir_path = basedir_path
        self.tmp_path = "%s%s" % (basedir_path, '/tmp')
        self.reg_path = "%s%s" % (basedir_path, '/tmp/reg')
        self.anat_path = "%s%s" % (basedir_path, '/anat_reg')
        self.reg_path_mat = "%s%s" % (self.reg_path, '/mats')
        self.reg_path_warp = "%s%s" % (self.reg_path, '/warps')
        self.reg_path_img = "%s%s" % (self.reg_path, '/imgs')
        self.t12mni_xfm_init = "{}/xfm_t1w2mni_init.mat".format(self.reg_path_mat)
        self.mni2t1_xfm_init = "{}/xfm_mni2t1w_init.mat".format(self.reg_path_mat)
        self.t12mni_xfm = "{}/xfm_t1w2mni.mat".format(self.reg_path_mat)
        self.mni2t1_xfm = "{}/xfm_mni2t1.mat".format(self.reg_path_mat)
        self.mni2t1w_warp = "{}/mni2t1w_warp.nii.gz".format(self.reg_path_warp)
        self.warp_t1w2mni = "{}/t1w2mni_warp.nii.gz".format(self.reg_path_warp)
        self.t1_aligned_mni = "{}/{}_aligned_mni.nii.gz".format(self.anat_path, self.t1w_name)
        self.t1w_brain = "{}/{}_brain.nii.gz".format(self.anat_path, self.t1w_name)
        self.t1w_brain_mask = "{}/{}_brain_mask.nii.gz".format(self.anat_path, self.t1w_name)
        self.map_path = "{}/{}_seg".format(self.anat_path, self.t1w_name)
        self.gm_mask = "{}/{}_gm.nii.gz".format(self.anat_path, self.t1w_name)
        self.gm_mask_thr = "{}/{}_gm_thr.nii.gz".format(self.anat_path, self.t1w_name)
        self.input_mni = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '.nii.gz')
        self.input_mni_brain = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain.nii.gz')
        self.input_mni_mask = "%s%s%s%s" % (FSLDIR, '/data/standard/MNI152_T1_', vox_size, '_brain_mask.nii.gz')
        self.input_mni_sched = "%s%s" % (FSLDIR, '/etc/flirtsch/T1_2_MNI152_2mm.cnf')

        # Create empty tmp directories that do not yet exist
        reg_dirs = [self.tmp_path, self.reg_path, self.anat_path, self.reg_path_mat, self.reg_path_warp, self.reg_path_img]
        for i in range(len(reg_dirs)):
            if not os.path.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

        if os.path.isfile(self.t1w_brain) is False:
            import shutil
            shutil.copyfile(self.t1w, self.t1w_brain)

    def gen_tissue(self):
        """

        :return:
        """
        # Segment the t1w brain into probability maps
        self.maps = mgru.segment_t1w(self.t1w_brain, self.map_path)
        self.gm_mask = self.maps['gm_prob']

        self.t1w_brain = utils.check_orient_and_dims(self.t1w_brain, self.vox_size)
        self.gm_mask = utils.check_orient_and_dims(self.gm_mask, self.vox_size)

        # Threshold WM to binary in dwi space
        self.t_img = load_img(self.gm_mask)
        self.mask = math_img('img > 0.1', img=self.t_img)
        self.mask.to_filename(self.gm_mask_thr)

        # Threshold T1w brain to binary in anat space
        self.t_img = load_img(self.t1w_brain)
        self.mask = math_img('img > 0.0', img=self.t_img)
        self.mask.to_filename(self.t1w_brain_mask)

        return

    def t1w2mni_align(self):
        """
        alignment from T1w --> MNI
        A function to perform self alignment.
        """

        # Create linear transform/ initializer T1w-->MNI
        mgru.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm_init, bins=None, interp="spline", out=None,
                   dof=12, cost='mutualinfo', searchrad=True)

        # Attempt non-linear registration of T1 to MNI template
        try:
            print('Running non-linear registration: T1w-->MNI ...')
            # Use FNIRT to nonlinearly align T1 to MNI template
            mgru.align_nonlinear(self.t1w_brain, self.input_mni, xfm=self.t12mni_xfm_init, out=self.t1_aligned_mni,
                                 warp=self.warp_t1w2mni, ref_mask=self.input_mni_mask,
                                 config=self.input_mni_sched)

            # Get warp from MNI -> T1
            mgru.inverse_warp(self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni)

            # Get mat from MNI -> T1
            cmd = 'convert_xfm -omat ' + self.mni2t1_xfm_init + ' -inverse ' + self.t12mni_xfm_init
            print(cmd)
            os.system(cmd)

        except RuntimeError('Error: FNIRT failed!'):
            pass

    def atlas2t1wmni_align(self, uatlas_select, atlas_select):
        """
        alignment from atlas --> T1_MNI
        A function to perform atlas alignment.
        Tries nonlinear registration first, and if that fails,
        does a linear registration instead.
        """
        self.atlas = uatlas_select
        self.atlas_name = atlas_select
        self.aligned_atlas_t1mni = "{}/{}_t1w_mni.nii.gz".format(self.basedir_path, self.atlas_name)
        self.aligned_atlas_skull = "{}/{}_t1w_skull.nii.gz".format(self.anat_path, self.atlas_name)
        self.gm_mask_mni = "{}/{}_gm_mask_t1w_mni.nii.gz".format(self.anat_path, self.t1w_name)
        self.aligned_atlas_t1mni_gm = "{}/{}_t1w_mni_gm.nii.gz".format(self.basedir_path, self.atlas_name)

        mgru.align(self.atlas, self.t1_aligned_mni, init=None, xfm=None, out=self.aligned_atlas_t1mni, dof=12,
                   searchrad=True, interp="nearestneighbour", cost='mutualinfo')

        # Apply warp resulting from the inverse of T1w-->MNI created earlier
        mgru.apply_warp(self.t1w_brain, self.aligned_atlas_t1mni, self.aligned_atlas_skull,
                        warp=self.mni2t1w_warp, interp='nn', sup=True)

        # Apply warp resulting from the inverse MNI->T1w created earlier
        mgru.apply_warp(self.t1w_brain, self.gm_mask_mni, self.gm_mask_thr, warp=self.mni2t1w_warp,
                        interp='nn', sup=True)


        # Set intensities to int
        self.atlas_img = nib.load(self.aligned_atlas_t1mni)
        self.atlas_data = self.atlas_img.get_fdata().astype('int')
        #node_num = len(np.unique(self.atlas_data))
        #self.atlas_data[self.atlas_data>node_num] = 0
        nib.save(nib.Nifti1Image(self.atlas_data.astype(np.int32), affine=self.atlas_img.affine,
                                 header=self.atlas_img.header), self.aligned_atlas_t1mni)
        cmd='fslmaths ' + self.aligned_atlas_t1mni + ' -mas ' + self.gm_mask_mni + ' ' + self.aligned_atlas_t1mni_gm
        os.system(cmd)

        return self.aligned_atlas_t1mni_gm


def register_all_dwi(basedir_path, fa_path, nodif_B0_mask, anat_file, gtab_file, dwi_file, vox_size='2mm', simple=False,
                 overwrite=False):
    """

    :param basedir_path:
    :param fa_path:
    :param nodif_B0_mask:
    :param anat_file:
    :param gtab_file:
    :param dwi_file:
    :param vox_size:
    :param simple:
    :param overwrite:
    :return:
    """
    import os.path as op
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, nodif_B0_mask, anat_file, vox_size, simple)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1w2dwi) is False):
        # Align t1w to dwi
        reg.t1w2dwi_align()

    if (overwrite is True) or (op.isfile(reg.wm_gm_int_in_dwi) is False):
        # Align tissue
        reg.tissue2dwi_align()

    return reg.wm_gm_int_in_dwi, reg.wm_in_dwi, reg.gm_in_dwi, reg.vent_csf_in_dwi, reg.csf_mask_dwi, anat_file, nodif_B0_mask, fa_path, gtab_file, dwi_file


def register_atlas_dwi(uatlas_select, atlas_select, node_size, basedir_path, fa_path, nodif_B0_mask, anat_file,
                   wm_gm_int_in_dwi, coords, label_names, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, gtab_file, dwi_file,
                   vox_size='2mm', simple=False):
    """

    :param uatlas_select:
    :param atlas_select:
    :param node_size:
    :param basedir_path:
    :param fa_path:
    :param nodif_B0_mask:
    :param anat_file:
    :param wm_gm_int_in_dwi:
    :param coords:
    :param label_names:
    :param gm_in_dwi:
    :param vent_csf_in_dwi:
    :param wm_in_dwi:
    :param gtab_file:
    :param dwi_file:
    :param vox_size:
    :param simple:
    :return:
    """
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, nodif_B0_mask, anat_file, vox_size, simple)

    if node_size is not None:
        atlas_select = "%s%s%s" % (atlas_select, '_', node_size)

    # Apply warps/coregister atlas to dwi
    [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni] = reg.atlas2t1w2dwi_align(uatlas_select,
                                                                                                   atlas_select)

    return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni, uatlas_select, atlas_select, coords, label_names, node_size, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, fa_path, gtab_file, nodif_B0_mask, dwi_file


def register_all_fmri(basedir_path, anat_file, vox_size='2mm', overwrite=False):
    """

    :param basedir_path:
    :param anat_file:
    :param vox_size:
    :param overwrite:
    :return:
    """
    import os.path as op
    from pynets.registration import register
    reg = register.FmriReg(basedir_path, anat_file, vox_size)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1_aligned_mni) is False):
        # Align t1w to dwi
        reg.t1w2mni_align()

    return


def register_atlas_fmri(uatlas_select, atlas_select, node_size, basedir_path, anat_file, vox_size='2mm'):
    """

    :param uatlas_select:
    :param atlas_select:
    :param node_size:
    :param basedir_path:
    :param anat_file:
    :param vox_size:
    :return:
    """
    from pynets.registration import register
    reg = register.FmriReg(basedir_path, anat_file, vox_size)

    if node_size is not None:
        atlas_select = "%s%s%s" % (atlas_select, '_', node_size)

    # Apply warps/coregister atlas to t1w_mni
    aligned_atlas_t1mni_gm = reg.atlas2t1wmni_align(uatlas_select, atlas_select)

    return aligned_atlas_t1mni_gm
