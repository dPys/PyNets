#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import warnings
warnings.filterwarnings("ignore")
import nibabel as nib
import numpy as np
from pynets.registration import reg_utils as regutils
from nilearn.image import load_img, math_img
try:
    FSLDIR = os.environ['FSLDIR']
except KeyError:
    print('FSLDIR environment variable not set!')


def direct_streamline_norm(streams, fa_path, dir_path, track_type, target_samples, conn_model, network, node_size,
                           dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, labels_im_file, uatlas,
                           labels, coords, norm, binary, atlas_mni, basedir_path, curv_thr_list, step_list, directget):
    """
    A Function to perform normalization of streamlines tracked in native diffusion space to an
    FA template in MNI space.

    Parameters
    ----------
    streams : str
        File path to save streamline array sequence in .trk format.
    fa_path : str
        File path to FA Nifti1Image.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    labels_im_file : str
        File path to atlas parcellation Nifti1Image aligned to dwi space.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).

    Returns
    -------
    streams_warp : str
        File path to normalized streamline array sequence in .trk format.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    directget : str
        The statistical approach to tracking. Options are: det (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    warped_fa : str
        File path to MNI-space warped FA Nifti1Image.

    References
    ----------
    .. [1] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of different spatial normalization approaches on
           tractography and structural brain networks. Network Neuroscience, 1-19.
    """
    from dipy.tracking.streamline import deform_streamlines
    from pynets.registration import reg_utils as regutils
    from pynets.dmri.track import save_streams
    from pynets.plotting import plot_gen
    from dipy.tracking import utils
    import pkg_resources
    import os.path as op
    from nilearn.image import resample_to_img
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space

    dsn_dir = "%s%s" % (basedir_path, '/dmri_tmp/DSN')
    if not os.path.isdir(dsn_dir):
        os.mkdir(dsn_dir)

    namer_dir = dir_path + '/tractography'
    if not os.path.isdir(namer_dir):
        os.mkdir(namer_dir)

    # Run SyN and normalize streamlines
    fa_img = nib.load(fa_path)
    vox_size = fa_img.header.get_zooms()[0]
    template_path = pkg_resources.resource_filename("pynets", "%s%s%s" % ('templates/FA_', int(vox_size), 'mm.nii.gz'))
    template_img = nib.load(template_path)

    streams_mni = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/streamlines_mni_',
                                                  '%s' % (network + '_' if network is not None else ''),
                                                  '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                  conn_model, '_', target_samples,
                                                  '%s' % ("%s%s" % ('_' + str(node_size), 'mm_') if ((node_size != 'parc') and (node_size is not None)) else '_'),
                                                  'curv', str(curv_thr_list).replace(', ', '_'),
                                                  'step', str(step_list).replace(', ', '_'), '.trk')

    density_mni = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (namer_dir, '/density_map_mni_',
                                                  '%s' % (network + '_' if network is not None else ''),
                                                  '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                  conn_model, '_', target_samples,
                                                  '%s' % ("%s%s" % ('_' + str(node_size), 'mm_') if ((node_size != 'parc') and (node_size is not None)) else '_'),
                                                  'curv', str(curv_thr_list).replace(', ', '_'),
                                                  'step', str(step_list).replace(', ', '_'), '.nii.gz')

    streams_warp_png = "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dsn_dir, '/streamlines_mni_warp_',
                                                       '%s' % (network + '_' if network is not None else ''),
                                                       '%s' % (op.basename(roi).split('.')[0] + '_' if roi is not None else ''),
                                                       conn_model, '_', target_samples,
                                                       '%s' % ("%s%s" % ('_' + str(node_size), 'mm_') if ((node_size != 'parc') and (node_size is not None)) else '_'),
                                                       'curv', str(curv_thr_list).replace(', ', '_'),
                                                       'step', str(step_list).replace(', ', '_'), '.png')

    # SyN FA->Template
    [mapping, affine_map, warped_fa] = regutils.wm_syn(template_path, fa_path, dsn_dir)

    tractogram = load_tractogram(streams, fa_img, to_space=Space.VOXMM, shifted_origin=True, bbox_valid_check=True)
    streamlines = tractogram.streamlines
    warped_fa_img = nib.load(warped_fa)

    # Warp streamlines
    adjusted_affine = affine_map.affine.copy()
    adjusted_affine[0][3] = adjusted_affine[0][3] / vox_size
    adjusted_affine[1][3] = adjusted_affine[1][3] / vox_size
    adjusted_affine[2][3] = adjusted_affine[2][3] / vox_size
    mni_streamlines = deform_streamlines(streamlines, deform_field=mapping.get_forward_field(),
                                         stream_to_current_grid=warped_fa_img.affine,
                                         current_grid_to_world=adjusted_affine,
                                         stream_to_ref_grid=template_img.affine,
                                         ref_grid_to_world=np.eye(4))

    # Save streamlines
    save_streams(warped_fa_img, mni_streamlines, streams_mni)

    # Create and save MNI density map
    nib.save(nib.Nifti1Image(utils.density_map(mni_streamlines, affine=np.eye(4), vol_dims=warped_fa_img.shape),
             warped_fa_img.affine), density_mni)

    # DSN QC plotting
    plot_gen.show_template_bundles(mni_streamlines, template_path, streams_warp_png)

    # Map parcellation from native space back to MNI-space and create an 'uncertainty-union' parcellation
    # with original mni-space uatlas
    atlas_img = nib.load(labels_im_file)
    uatlas_mni_img = nib.load(uatlas)
    warped_uatlas = affine_map.transform_inverse(mapping.transform(atlas_img.get_data().astype('int16'),
                                                                   interpolation='nearestneighbour'),
                                                 interp='nearest')
    union_ua = "%s%s%s%s" % (os.path.dirname(uatlas), '/', os.path.basename(uatlas).split('.nii.gz')[0],
                             '_UNION.nii.gz')

    warped_uatlas_img = nib.Nifti1Image(warped_uatlas, affine=warped_fa_img.affine)
    warped_uatlas_img_res = resample_to_img(warped_uatlas_img, uatlas_mni_img, interpolation='nearest', clip=False)
    warped_uatlas_img_res_data = warped_uatlas_img_res.get_data()
    uatlas_mni_data = uatlas_mni_img.get_data()
    overlap_mask = np.invert(warped_uatlas_img_res_data.astype('bool') * uatlas_mni_data.astype('bool'))
    union_atlas = warped_uatlas_img_res_data * overlap_mask.astype('int') + uatlas_mni_data * overlap_mask.astype('int')
    union_atlas_corr = union_atlas + np.invert(overlap_mask).astype('int') * uatlas_mni_data
    nib.save(nib.Nifti1Image(union_atlas_corr, affine=warped_fa_img.affine), union_ua)
    atlas_mni = union_ua

    return streams_mni, dir_path, track_type, target_samples, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary, atlas_mni, directget, warped_fa


class DmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in native diffusion space.
    """

    def __init__(self, basedir_path, fa_path, B0_mask, anat_file, vox_size, simple):
        import pkg_resources
        self.simple = simple
        self.fa_path = fa_path
        self.B0_mask = B0_mask
        self.t1w = anat_file
        self.vox_size = vox_size
        self.t1w_name = 't1w'
        self.dwi_name = 'dwi'
        self.basedir_path = basedir_path
        self.tmp_path = "%s%s" % (basedir_path, '/dmri_tmp')
        self.reg_path = "%s%s" % (basedir_path, '/dmri_tmp/reg')
        self.anat_path = "%s%s" % (basedir_path, '/anat_reg')
        self.reg_path_mat = "%s%s" % (self.reg_path, '/mats')
        self.reg_path_warp = "%s%s" % (self.reg_path, '/warps')
        self.reg_path_img = "%s%s" % (self.reg_path, '/imgs')
        self.t12mni_xfm_init = "%s%s" % (self.reg_path_mat, "/xfm_t1w2mni_init.mat")
        self.mni2t1_xfm_init = "%s%s" % (self.reg_path_mat, "/xfm_mni2t1w_init.mat")
        self.t12mni_xfm = "%s%s" % (self.reg_path_mat, "/xfm_t1w2mni.mat")
        self.mni2t1_xfm = "%s%s" % (self.reg_path_mat, "/xfm_mni2t1.mat")
        self.mni2t1w_warp = "%s%s" % (self.reg_path_warp, "/mni2t1w_warp.nii.gz")
        self.warp_t1w2mni = "%s%s" % (self.reg_path_warp, "/t1w2mni_warp.nii.gz")
        self.t1w2dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_in_dwi.nii.gz")
        self.t1_aligned_mni = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_aligned_mni.nii.gz")
        self.t1w_brain = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_brain.nii.gz")
        self.t1w_brain_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_brain_mask.nii.gz")
        self.dwi2t1w_xfm = "%s%s" % (self.reg_path_mat, "/dwi2t1w_xfm.mat")
        self.t1w2dwi_xfm = "%s%s" % (self.reg_path_mat, "/t1w2dwi_xfm.mat")
        self.t1w2dwi_bbr_xfm = "%s%s" % (self.reg_path_mat, "/t1w2dwi_bbr_xfm.mat")
        self.dwi2t1w_bbr_xfm = "%s%s" % (self.reg_path_mat, "/dwi2t1w_bbr_xfm.mat")
        self.t1wtissue2dwi_xfm = "%s%s" % (self.reg_path_mat, "/t1wtissue2dwi_xfm.mat")
        self.xfm_atlas2t1w_init = "%s%s%s%s" % (self.reg_path_mat, '/', self.t1w_name, "_xfm_atlas2t1w_init.mat")
        self.xfm_atlas2t1w = "%s%s%s%s" % (self.reg_path_mat, '/', self.t1w_name, "_xfm_atlas2t1w.mat")
        self.temp2dwi_xfm = "%s%s%s%s" % (self.reg_path_mat, '/', self.dwi_name, "_xfm_temp2dwi.mat")
        self.temp2dwi_xfm = "%s%s%s%s" % (self.reg_path_mat, '/', self.dwi_name, "_xfm_temp2dwi.mat")
        self.map_path = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_seg")
        self.wm_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_wm.nii.gz")
        self.wm_mask_thr = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_wm_thr.nii.gz")
        self.wm_edge = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_wm_edge.nii.gz")
        self.csf_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_csf.nii.gz")
        self.gm_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_gm.nii.gz")
        self.xfm_roi2mni_init = "%s%s" % (self.reg_path_mat, "/roi_2_mni.mat")
        self.mni_vent_loc = pkg_resources.resource_filename("pynets", "templates/LateralVentricles_" + vox_size +
                                                            ".nii.gz")
        self.csf_mask_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_csf_mask_dwi.nii.gz")
        self.gm_in_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_gm_in_dwi.nii.gz")
        self.wm_in_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_wm_in_dwi.nii.gz")
        self.csf_mask_dwi_bin = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_csf_mask_dwi_bin.nii.gz")
        self.gm_in_dwi_bin = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_gm_in_dwi_bin.nii.gz")
        self.wm_in_dwi_bin = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_wm_in_dwi_bin.nii.gz")
        self.vent_mask_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_vent_mask_dwi.nii.gz")
        self.vent_csf_in_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_vent_csf_in_dwi.nii.gz")
        self.vent_mask_mni = "%s%s" % (self.reg_path_img, "/vent_mask_mni.nii.gz")
        self.vent_mask_t1w = "%s%s" % (self.reg_path_img, "/vent_mask_t1w.nii.gz")
        self.input_mni = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size + ".nii.gz")
        self.input_mni_brain = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                               "_brain.nii.gz")
        self.input_mni_mask = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                              "_brain_mask.nii.gz")
        self.mni_atlas = pkg_resources.resource_filename("pynets", "core/atlases/HarvardOxford-sub-prob-" + vox_size +
                                                         ".nii.gz")
        self.wm_gm_int_in_dwi = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_wm_gm_int_in_dwi.nii.gz")
        self.wm_gm_int_in_dwi_bin = "%s%s%s%s" % (self.reg_path_img, '/', self.t1w_name, "_wm_gm_int_in_dwi_bin.nii.gz")
        self.corpuscallosum = pkg_resources.resource_filename("pynets", "templates/CorpusCallosum_" + vox_size +
                                                              ".nii.gz")
        self.corpuscallosum_mask_t1w = ("%s%s" % (self.reg_path_img, '/CorpusCallosum_t1wmask.nii.gz'))
        self.corpuscallosum_dwi = ("%s%s" % (self.reg_path_img, '/CorpusCallosum_dwi.nii.gz'))

        # Create empty tmp directories that do not yet exist
        reg_dirs = [self.tmp_path, self.reg_path, self.anat_path, self.reg_path_mat, self.reg_path_warp,
                    self.reg_path_img]
        for i in range(len(reg_dirs)):
            if not os.path.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

        if os.path.isfile(self.t1w_brain) is False:
            import shutil
            shutil.copyfile(self.t1w, self.t1w_brain)

    def gen_tissue(self):
        """
        A function to segment and threshold tissue types from T1w.
        """
        # Segment the t1w brain into probability maps
        maps = regutils.segment_t1w(self.t1w_brain, self.map_path)
        self.wm_mask = maps['wm_prob']
        self.gm_mask = maps['gm_prob']
        self.csf_mask = maps['csf_prob']

        self.t1w_brain = regutils.check_orient_and_dims(self.t1w_brain, self.vox_size)
        self.wm_mask = regutils.check_orient_and_dims(self.wm_mask, self.vox_size)
        self.gm_mask = regutils.check_orient_and_dims(self.gm_mask, self.vox_size)
        self.csf_mask = regutils.check_orient_and_dims(self.csf_mask, self.vox_size)

        # Threshold WM to binary in dwi space
        t_img = load_img(self.wm_mask)
        mask = math_img('img > 0.2', img=t_img)
        mask.to_filename(self.wm_mask_thr)

        # Threshold T1w brain to binary in anat space
        t_img = load_img(self.t1w_brain)
        mask = math_img('img > 0.0', img=t_img)
        mask.to_filename(self.t1w_brain_mask)

        # Extract wm edge
        os.system("fslmaths {} -edge -bin -mas {} {}".format(self.wm_mask_thr, self.wm_mask_thr, self.wm_edge))

        return

    def t1w2dwi_align(self):
        """
        A function to perform alignment from T1w --> MNI and T1w_MNI --> DWI. Uses a local optimisation
        cost function to get the two images close, and then uses bbr to obtain a good alignment of brain boundaries.
        Assumes input dwi is already preprocessed and brain extracted.
        """

        # Create linear transform/ initializer T1w-->MNI
        regutils.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm_init, bins=None, interp="spline",
                       out=None, dof=12, cost='mutualinfo', searchrad=True)

        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print('Running non-linear registration: T1w-->MNI ...')
                # Use FNIRT to nonlinearly align T1 to MNI template
                regutils.align_nonlinear(self.t1w_brain, self.input_mni, xfm=self.t12mni_xfm_init,
                                         out=self.t1_aligned_mni, warp=self.warp_t1w2mni, ref_mask=self.input_mni_mask)

                # Get warp from MNI -> T1
                regutils.inverse_warp(self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni)

                # Get mat from MNI -> T1
                os.system("convert_xfm -omat {} -inverse {}".format(self.mni2t1_xfm_init, self.t12mni_xfm_init))

            except RuntimeError('Error: FNIRT failed!'):
                pass
        else:
            # Falling back to linear registration
            regutils.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm, init=self.t12mni_xfm_init,
                           bins=None, dof=12, cost='mutualinfo', searchrad=True, interp="spline",
                           out=self.t1_aligned_mni, sch=None)
            # Get mat from MNI -> T1
            os.system("convert_xfm -omat {} -inverse {}".format(self.t12mni_xfm, self.mni2t1_xfm))

        # Align T1w-->DWI
        regutils.align(self.fa_path, self.t1w_brain, xfm=self.t1w2dwi_xfm, bins=None, interp="spline", dof=6,
                       cost='mutualinfo', out=None, searchrad=True, sch=None)
        os.system("convert_xfm -omat {} -inverse {}".format(self.dwi2t1w_xfm, self.t1w2dwi_xfm))

        if self.simple is False:
            # Flirt bbr
            try:
                print('Running FLIRT BBR registration: T1w-->DWI ...')
                regutils.align(self.fa_path, self.t1w_brain, wmseg=self.wm_edge, xfm=self.dwi2t1w_bbr_xfm,
                               init=self.dwi2t1w_xfm, bins=256, dof=7, searchrad=True, interp="spline", out=None,
                               cost='bbr', sch="${FSLDIR}/etc/flirtsch/bbr.sch")
                os.system("convert_xfm -omat {} -inverse {}".format(self.t1w2dwi_bbr_xfm, self.dwi2t1w_bbr_xfm))

                # Apply the alignment
                regutils.align(self.t1w_brain, self.fa_path, init=self.t1w2dwi_bbr_xfm, xfm=self.t1wtissue2dwi_xfm,
                               bins=None, interp="spline", dof=7, cost='mutualinfo', out=self.t1w2dwi, searchrad=True,
                               sch=None)
            except RuntimeError('Error: FLIRT BBR failed!'):
                pass
        else:
            # Apply the alignment
            regutils.align(self.t1w_brain, self.fa_path, init=self.t1w2dwi_xfm, xfm=self.t1wtissue2dwi_xfm, bins=None,
                           interp="spline", dof=6, cost='mutualinfo', out=self.t1w2dwi, searchrad=True, sch=None)

        return

    def atlas2t1w2dwi_align(self, uatlas, atlas):
        """
        A function to perform atlas alignment atlas --> T1 --> dwi.
        Tries nonlinear registration first, and if that fails, does a linear registration instead. For this to succeed,
        must first have called t1w2dwi_align.
        """
        aligned_atlas_t1mni = "%s%s%s%s" % (self.anat_path, '/', atlas, "_t1w_mni.nii.gz")
        aligned_atlas_skull = "%s%s%s%s" % (self.anat_path, '/', atlas, "_t1w_skull.nii.gz")
        dwi_aligned_atlas = "%s%s%s%s" % (self.reg_path_img, '/', atlas, "_dwi_track.nii.gz")
        dwi_aligned_atlas_wmgm_int = "%s%s%s%s" % (self.reg_path_img, '/', atlas, "_dwi_track_wmgm_int.nii.gz")

        regutils.align(uatlas, self.t1_aligned_mni, init=None, xfm=None, out=aligned_atlas_t1mni, dof=12,
                       searchrad=True, interp="nearestneighbour", cost='mutualinfo')

        if self.simple is False:
            try:
                # Apply warp resulting from the inverse of T1w-->MNI created earlier
                regutils.apply_warp(self.t1w_brain, aligned_atlas_t1mni, aligned_atlas_skull,
                                    warp=self.mni2t1w_warp, interp='nn', sup=True)

                # Apply transform to dwi space
                regutils.align(aligned_atlas_skull, self.fa_path, init=self.t1wtissue2dwi_xfm, xfm=None,
                               out=dwi_aligned_atlas, dof=6, searchrad=True, interp="nearestneighbour",
                               cost='mutualinfo')
            except:
                print("Warning: Atlas is not in correct dimensions, or input is low quality,\nusing linear template "
                      "registration.")

                # Create transform to align atlas to T1w using flirt
                regutils.align(uatlas, self.t1w_brain, xfm=self.xfm_atlas2t1w_init, init=None, bins=None, dof=6,
                               cost='mutualinfo', searchrad=True, interp="spline", out=None, sch=None)
                regutils.align(uatlas, self.t1_aligned_mni, xfm=self.xfm_atlas2t1w, out=None, dof=6, searchrad=True,
                               bins=None, interp="spline", cost='mutualinfo', init=self.xfm_atlas2t1w_init)

                # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a
                # transform from atlas ->(-> t1w ->)-> dwi
                regutils.combine_xfms(self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm)

                # Apply linear transformation from template to dwi space
                regutils.applyxfm(self.fa_path, uatlas, self.temp2dwi_xfm, dwi_aligned_atlas)
        else:
            # Create transform to align atlas to T1w using flirt
            regutils.align(uatlas, self.t1w_brain, xfm=self.xfm_atlas2t1w_init, init=None, bins=None, dof=6,
                           cost='mutualinfo', searchrad=None, interp="spline", out=None, sch=None)
            regutils.align(uatlas, self.t1w_brain, xfm=self.xfm_atlas2t1w, out=None, dof=6, searchrad=True,
                           bins=None, interp="spline", cost='mutualinfo', init=self.xfm_atlas2t1w_init)

            # Combine our linear transform from t1w to template with our transform from dwi to t1w space to get a
            # transform from atlas ->(-> t1w ->)-> dwi
            regutils.combine_xfms(self.xfm_atlas2t1w, self.t1wtissue2dwi_xfm, self.temp2dwi_xfm)

            # Apply linear transformation from template to dwi space
            regutils.applyxfm(self.fa_path, uatlas, self.temp2dwi_xfm, dwi_aligned_atlas)

        # Set intensities to int
        atlas_img = nib.load(dwi_aligned_atlas)
        atlas_data = np.around(atlas_img.get_fdata()).astype('int16')
        t_img = load_img(self.wm_gm_int_in_dwi)
        mask = math_img('img > 0', img=t_img)
        mask.to_filename(self.wm_gm_int_in_dwi_bin)
        nib.save(nib.Nifti1Image(atlas_data.astype('int16'), affine=atlas_img.affine, header=atlas_img.header),
                 dwi_aligned_atlas)
        os.system("fslmaths {} -mas {} {}".format(dwi_aligned_atlas, self.wm_gm_int_in_dwi_bin,
                                                  dwi_aligned_atlas_wmgm_int))

        return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni

    def tissue2dwi_align(self):
        """
        A function to perform alignment of ventricle ROI's from MNI space --> dwi and CSF from T1w space --> dwi.
        First generates and performs dwi space alignment of avoidance/waypoint masks for tractography.
        First creates ventricle ROI. Then creates transforms from stock MNI template to dwi space.
        For this to succeed, must first have called both t1w2dwi_align and atlas2t1w2dwi_align.
        """

        # Register Lateral Ventricles and Corpus Callosum rois to t1w
        if not os.path.isfile(self.mni_atlas):
            raise ValueError('FSL atlas for ventricle reference not found!')

        # Create transform to MNI atlas to T1w using flirt. This will be use to transform the ventricles to dwi space.
        regutils.align(self.mni_atlas, self.input_mni_brain, xfm=self.xfm_roi2mni_init, init=None, bins=None, dof=6,
                       cost='mutualinfo', searchrad=True, interp="spline", out=None)

        # Create transform to align roi to mni and T1w using flirt
        regutils.applyxfm(self.input_mni_brain, self.mni_vent_loc, self.xfm_roi2mni_init, self.vent_mask_mni)

        if self.simple is False:
            # Apply warp resulting from the inverse MNI->T1w created earlier
            regutils.apply_warp(self.t1w_brain, self.vent_mask_mni, self.vent_mask_t1w, warp=self.mni2t1w_warp,
                                interp='nn', sup=True)

            regutils.apply_warp(self.t1w_brain, self.corpuscallosum, self.corpuscallosum_mask_t1w,
                                warp=self.mni2t1w_warp, interp="nn", sup=True)

        else:
            regutils.applyxfm(self.vent_mask_mni, self.t1w_brain, self.mni2t1_xfm, self.vent_mask_t1w)
            regutils.applyxfm(self.corpuscallosum, self.t1w_brain, self.mni2t1_xfm, self.corpuscallosum_mask_t1w)

        # Applyxfm tissue maps to dwi space
        regutils.applyxfm(self.fa_path, self.vent_mask_t1w, self.t1wtissue2dwi_xfm, self.vent_mask_dwi)
        regutils.applyxfm(self.fa_path, self.csf_mask, self.t1wtissue2dwi_xfm, self.csf_mask_dwi)
        regutils.applyxfm(self.fa_path, self.gm_mask, self.t1wtissue2dwi_xfm, self.gm_in_dwi)
        regutils.applyxfm(self.fa_path, self.wm_mask, self.t1wtissue2dwi_xfm, self.wm_in_dwi)
        regutils.applyxfm(self.fa_path, self.corpuscallosum_mask_t1w, self.t1wtissue2dwi_xfm, self.corpuscallosum_dwi)

        # Threshold WM to binary in dwi space
        thr_img = nib.load(self.wm_in_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.1] = 0
        nib.save(thr_img, self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        thr_img = nib.load(self.gm_in_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.2] = 0
        nib.save(thr_img, self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        thr_img = nib.load(self.csf_mask_dwi)
        thr_img.get_fdata()[thr_img.get_fdata() < 0.95] = 0
        nib.save(thr_img, self.csf_mask_dwi)

        # Threshold WM to binary in dwi space
        t_img = load_img(self.wm_in_dwi_bin)
        mask = math_img('img > 0', img=t_img)
        mask.to_filename(self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        t_img = load_img(self.gm_in_dwi_bin)
        mask = math_img('img > 0', img=t_img)
        mask.to_filename(self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        t_img = load_img(self.csf_mask_dwi)
        mask = math_img('img > 0', img=t_img)
        mask.to_filename(self.csf_mask_dwi_bin)

        # Create ventricular CSF mask
        print('Creating ventricular CSF mask...')
        os.system("fslmaths {} -kernel sphere 10 -ero -bin {}".format(self.vent_mask_dwi, self.vent_mask_dwi))
        os.system("fslmaths {} -add {} -bin {} ".format(self.csf_mask_dwi, self.vent_mask_dwi, self.vent_csf_in_dwi))

        print("Creating Corpus Callosum mask...")
        os.system("fslmaths {} -mas {} -sub {} -bin {}".format(self.corpuscallosum_dwi, self.wm_in_dwi_bin,
                                                               self.vent_csf_in_dwi, self.corpuscallosum_dwi))

        # Create gm-wm interface image
        os.system("fslmaths {} -mul {} -add {} -mas {} -bin {}".format(self.gm_in_dwi_bin, self.wm_in_dwi_bin,
                                                                       self.corpuscallosum_dwi, self.B0_mask,
                                                                       self.wm_gm_int_in_dwi))

        return


class FmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image.
    """

    def __init__(self, basedir_path, anat_file, vox_size):
        import pkg_resources
        self.t1w = anat_file
        self.vox_size = vox_size
        self.t1w_name = 't1w'
        self.basedir_path = basedir_path
        self.tmp_path = "%s%s" % (basedir_path, '/fmri_tmp')
        self.reg_path = "%s%s" % (basedir_path, '/fmri_tmp/reg')
        self.anat_path = "%s%s" % (basedir_path, '/anat_reg')
        self.reg_path_mat = "%s%s" % (self.reg_path, '/mats')
        self.reg_path_warp = "%s%s" % (self.reg_path, '/warps')
        self.reg_path_img = "%s%s" % (self.reg_path, '/imgs')
        self.t12mni_xfm_init = "%s%s" % (self.reg_path_mat, "/xfm_t1w2mni_init.mat")
        self.mni2t1_xfm_init = "%s%s" % (self.reg_path_mat, "/xfm_mni2t1w_init.mat")
        self.t12mni_xfm = "%s%s" % (self.reg_path_mat, "/xfm_t1w2mni.mat")
        self.mni2t1_xfm = "%s%s" % (self.reg_path_mat, "/xfm_mni2t1.mat")
        self.mni2t1w_warp = "%s%s" % (self.reg_path_warp, "/mni2t1w_warp.nii.gz")
        self.warp_t1w2mni = "%s%s" % (self.reg_path_warp, "/t1w2mni_warp.nii.gz")
        self.t1_aligned_mni = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_aligned_mni.nii.gz")
        self.t1w_brain = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_brain.nii.gz")
        self.t1w_brain_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_brain_mask.nii.gz")
        self.map_path = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_seg")
        self.gm_mask = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_gm.nii.gz")
        self.gm_mask_thr = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_gm_thr.nii.gz")
        self.input_mni = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size + ".nii.gz")
        self.input_mni_brain = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                               "_brain.nii.gz")
        self.input_mni_mask = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                              "_brain_mask.nii.gz")

        # Create empty tmp directories that do not yet exist
        reg_dirs = [self.tmp_path, self.reg_path, self.anat_path, self.reg_path_mat, self.reg_path_warp,
                    self.reg_path_img]
        for i in range(len(reg_dirs)):
            if not os.path.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

        if os.path.isfile(self.t1w_brain) is False:
            import shutil
            shutil.copyfile(self.t1w, self.t1w_brain)

    def gen_tissue(self):
        """
        A function to segment and threshold tissue types from T1w.
        """
        # Segment the t1w brain into probability maps
        maps = regutils.segment_t1w(self.t1w_brain, self.map_path)
        self.gm_mask = maps['gm_prob']

        self.t1w_brain = regutils.check_orient_and_dims(self.t1w_brain, self.vox_size)
        self.gm_mask = regutils.check_orient_and_dims(self.gm_mask, self.vox_size)

        # Threshold GM to binary in dwi space
        t_img = load_img(self.gm_mask)
        mask = math_img('img > 0.05', img=t_img)
        mask.to_filename(self.gm_mask_thr)

        # Threshold T1w brain to binary in anat space
        t_img = load_img(self.t1w_brain)
        mask = math_img('img > 0.0', img=t_img)
        mask.to_filename(self.t1w_brain_mask)

        return

    def t1w2mni_align(self):
        """
        A function to perform alignment from T1w --> MNI.
        """

        # Create linear transform/ initializer T1w-->MNI
        regutils.align(self.t1w_brain, self.input_mni_brain, xfm=self.t12mni_xfm_init, bins=None, interp="spline",
                       out=None, dof=12, cost='mutualinfo', searchrad=True)

        # Attempt non-linear registration of T1 to MNI template
        try:
            print('Running non-linear registration: T1w-->MNI ...')
            # Use FNIRT to nonlinearly align T1 to MNI template
            regutils.align_nonlinear(self.t1w_brain, self.input_mni, xfm=self.t12mni_xfm_init, out=self.t1_aligned_mni,
                                     warp=self.warp_t1w2mni, ref_mask=self.input_mni_mask)

            # Get warp from MNI -> T1
            regutils.inverse_warp(self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni)

            # Get mat from MNI -> T1
            os.system("convert_xfm -omat {} -inverse {}".format(self.mni2t1_xfm_init, self.t12mni_xfm_init))

        except RuntimeError('Error: FNIRT failed!'):
            pass

    def atlas2t1wmni_align(self, uatlas, atlas):
        """
        A function to perform atlas alignment from atlas --> T1_MNI.
        Tries nonlinear registration first, and if that fails, does a linear registration instead.
        """
        aligned_atlas_t1mni = "%s%s%s%s" % (self.anat_path, '/', atlas, "_t1w_mni.nii.gz")
        aligned_atlas_skull = "%s%s%s%s" % (self.anat_path, '/', atlas, "_t1w_skull.nii.gz")
        gm_mask_mni = "%s%s%s%s" % (self.anat_path, '/', self.t1w_name, "_gm_mask_t1w_mni.nii.gz")
        aligned_atlas_t1mni_gm = "%s%s%s%s" % (self.anat_path, '/', atlas, "_t1w_mni_gm.nii.gz")
        regutils.align(uatlas, self.t1_aligned_mni, init=None, xfm=None, out=aligned_atlas_t1mni, dof=12,
                       searchrad=True, interp="nearestneighbour", cost='mutualinfo')

        # Apply warp resulting from the inverse of T1w-->MNI created earlier
        regutils.apply_warp(self.t1w_brain, aligned_atlas_t1mni, aligned_atlas_skull, warp=self.mni2t1w_warp,
                            interp='nn', sup=True)

        # Apply warp resulting from the inverse MNI->T1w created earlier
        regutils.apply_warp(self.t1w_brain, self.gm_mask_thr, gm_mask_mni, warp=self.mni2t1w_warp, interp='nn',
                            sup=True)

        # Set intensities to int
        atlas_img = nib.load(aligned_atlas_t1mni)
        atlas_data = np.around(atlas_img.get_fdata()).astype('int16')
        nib.save(nib.Nifti1Image(atlas_data.astype('int16'), affine=atlas_img.affine,
                                 header=atlas_img.header), aligned_atlas_t1mni)
        os.system("fslmaths {} -mas {} {}".format(aligned_atlas_t1mni, gm_mask_mni, aligned_atlas_t1mni_gm))

        return aligned_atlas_t1mni_gm


def register_all_dwi(basedir_path, fa_path, B0_mask, anat_file, gtab_file, dwi_file, vox_size, simple=False,
                     overwrite=False):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False).
        Default is True.
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is False.

    Returns
    -------
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    csf_mask_dwi : str
        File path to CSF tissue segmentation Nifti1Image in native diffusion space.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    """
    import os.path as op
    from pynets.registration import register
    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if (overwrite is True) or (op.isfile(reg.map_path) is False):
        # Perform anatomical segmentation
        reg.gen_tissue()

    if (overwrite is True) or (op.isfile(reg.t1w2dwi) is False):
        # Align t1w to dwi
        reg.t1w2dwi_align()

    if (overwrite is True) or (op.isfile(reg.wm_gm_int_in_dwi) is False):
        # Align tissue
        reg.tissue2dwi_align()

    return reg.wm_gm_int_in_dwi, reg.wm_in_dwi, reg.gm_in_dwi, reg.vent_csf_in_dwi, reg.csf_mask_dwi, anat_file, B0_mask, fa_path, gtab_file, dwi_file


def register_atlas_dwi(uatlas, atlas, node_size, basedir_path, fa_path, B0_mask, anat_file, wm_gm_int_in_dwi, coords,
                       labels, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, gtab_file, dwi_file, vox_size='2mm',
                       simple=False):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    fa_path : str
        File path to FA Nifti1Image.
    B0_mask : str
        File path to B0 brain mask.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    wm_gm_int_in_dwi : st
        Path to wm-gm interface Nifti1Image file in native diffusion space.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    dwi_file : str
        File path to diffusion weighted image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    simple : bool
        Indicates whether to use non-linear registration and BBR (True) or entirely linear methods (False). 
        Default is True.
        
    Returns
    -------
    dwi_aligned_atlas_t1mni_wmgm_int : str
        File path to atlas parcellation Nifti1Image in T1w-MNI warped native diffusion space, restricted only to wm-gm
        interface.
    dwi_aligned_atlas : str
        File path to atlas parcellation Nifti1Image in native diffusion space.
    aligned_atlas_t1mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to graph nodes.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    gm_in_dwi : str
        File path to grey-matter tissue segmentation Nifti1Image in native diffusion space.
    vent_csf_in_dwi : str
        File path to ventricular CSF tissue segmentation Nifti1Image in native diffusion space.
    wm_in_dwi : str
        File path to white-matter tissue segmentation Nifti1Image in native diffusion space.
    fa_path : str
        File path to FA Nifti1Image.
    gtab_file : str
        File path to pickled DiPy gradient table object.
    B0_mask : str
        File path to B0 brain mask.
    dwi_file : str
        File path to diffusion weighted image.
    """
    from pynets.registration import register
    from pynets.registration import reg_utils as regutils

    reg = register.DmriReg(basedir_path, fa_path, B0_mask, anat_file, vox_size, simple)

    if node_size is not None:
        atlas = "%s%s%s" % (atlas, '_', node_size)

    # Check orientation and resolution
    uatlas = regutils.check_orient_and_dims(uatlas, vox_size)

    # Apply warps/coregister atlas to dwi
    [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni] = reg.atlas2t1w2dwi_align(uatlas, atlas)

    return dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni, uatlas, atlas, coords, labels, node_size, gm_in_dwi, vent_csf_in_dwi, wm_in_dwi, fa_path, gtab_file, B0_mask, dwi_file


def register_all_fmri(basedir_path, anat_file, vox_size, overwrite=False):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    overwrite : bool
        Indicates whether to overwrite existing registration files. Default is False.
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


def register_atlas_fmri(uatlas, atlas, node_size, basedir_path, anat_file, vox_size):
    """
    A Function to register an atlas to T1w-warped MNI-space, and restrict the atlas to grey-matter only.

    Parameters
    ----------
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    atlas : str
        Name of atlas parcellation used.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files and outputs.
    anat_file : str
        Path to a skull-stripped anatomical Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).

    Returns
    -------
    aligned_atlas_t1mni_gm : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space, restricted only to grey-matter.
    """
    from pynets.registration import register
    from pynets.registration import reg_utils as regutils

    reg = register.FmriReg(basedir_path, anat_file, vox_size)

    if node_size is not None:
        atlas = "%s%s%s" % (atlas, '_', node_size)

    # Check orientation and resolution
    uatlas = regutils.check_orient_and_dims(uatlas, vox_size)

    # Apply warps/coregister atlas to t1w_mni
    aligned_atlas_t1mni_gm = reg.atlas2t1wmni_align(uatlas, atlas)

    return aligned_atlas_t1mni_gm
