#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner
"""
import os
import nibabel as nib
import warnings
import indexed_gzip
import numpy as np
from pynets.registration import reg_utils as regutils
from nilearn.image import math_img

warnings.filterwarnings("ignore")
try:
    FSLDIR = os.environ["FSLDIR"]
except KeyError:
    print("FSLDIR environment variable not set!")


def direct_streamline_norm(
    streams,
    fa_path,
    ap_path,
    dir_path,
    track_type,
    target_samples,
    conn_model,
    network,
    node_size,
    dens_thresh,
    ID,
    roi,
    min_span_tree,
    disp_filt,
    parc,
    prune,
    atlas,
    labels_im_file,
    uatlas,
    labels,
    coords,
    norm,
    binary,
    atlas_mni,
    basedir_path,
    curv_thr_list,
    step_list,
    directget,
    min_length,
    t1_aligned_mni,
    error_margin,
):
    """
    A Function to perform normalization of streamlines tracked in native diffusion space
    to an MNI-space template.
    Parameters
    ----------
    streams : str
        File path to save streamline array sequence in .trk format.
    fa_path : str
        File path to FA Nifti1Image.
    ap_path : str
        File path to the anisotropic power Nifti1Image.
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
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    t1_aligned_mni : str
        File path to the T1w Nifti1Image in template MNI space.
    error_margin : int
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
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
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    error_margin : int
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    References
    ----------
    .. [1] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of different
      spatial normalization approaches on tractography and structural
      brain networks. Network Neuroscience, 1-19.
    """
    import gc
    from dipy.tracking.streamline import transform_streamlines
    from pynets.registration import reg_utils as regutils
    # from pynets.plotting import plot_gen
    import pkg_resources
    import yaml
    import os.path as op
    from pynets.registration.reg_utils import vdc
    from nilearn.image import resample_to_img
    from dipy.io.streamline import load_tractogram
    from dipy.tracking import utils
    from dipy.tracking._utils import _mapping_to_voxel
    from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
    from dipy.io.streamline import save_tractogram

    # from pynets.core.utils import missing_elements

    with open(
        pkg_resources.resource_filename("pynets", "runconfig.yaml"), "r"
    ) as stream:
        try:
            hardcoded_params = yaml.load(stream)
            run_dsn = hardcoded_params["DSN"][0]
        except FileNotFoundError:
            print("Failed to parse runconfig.yaml")
    stream.close()

    if run_dsn is True:
        dsn_dir = f"{basedir_path}/dmri_reg/DSN"
        if not op.isdir(dsn_dir):
            os.mkdir(dsn_dir)

        namer_dir = f"{dir_path}/tractography"
        if not op.isdir(namer_dir):
            os.mkdir(namer_dir)

        atlas_img = nib.load(labels_im_file)

        # Run SyN and normalize streamlines
        fa_img = nib.load(fa_path)
        vox_size = fa_img.header.get_zooms()[0]
        template_path = pkg_resources.resource_filename(
            "pynets", f"templates/FA_{int(vox_size)}mm.nii.gz"
        )
        try:
            template_img = nib.load(template_path)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load FA template. Do you have git-lfs "
                  f"installed?")
        uatlas_mni_img = nib.load(atlas_mni)
        t1_aligned_mni_img = nib.load(t1_aligned_mni)
        brain_mask = np.asarray(t1_aligned_mni_img.dataobj).astype("bool")

        streams_mni = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
            namer_dir,
            "/streamlines_mni_",
            "%s" % (network + "_" if network is not None else ""),
            "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None else ""),
            conn_model,
            "_",
            target_samples,
            "%s"
            % (
                "%s%s" % ("_" + str(node_size), "mm_")
                if ((node_size != "parc") and (node_size is not None))
                else "_"
            ),
            "curv",
            str(curv_thr_list).replace(", ", "_"),
            "step",
            str(step_list).replace(", ", "_"),
            "tracktype-",
            track_type,
            "_directget-",
            directget,
            "_minlength-",
            min_length,
            ".trk",
        )

        density_mni = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
            namer_dir,
            "/density_map_mni_",
            "%s" % (network + "_" if network is not None else ""),
            "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None else ""),
            conn_model,
            "_",
            target_samples,
            "%s"
            % (
                "%s%s" % ("_" + str(node_size), "mm_")
                if ((node_size != "parc") and (node_size is not None))
                else "_"
            ),
            "curv",
            str(curv_thr_list).replace(", ", "_"),
            "step",
            str(step_list).replace(", ", "_"),
            "tracktype-",
            track_type,
            "_directget-",
            directget,
            "_minlength-",
            min_length,
            ".nii.gz",
        )

        # streams_warp_png = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (dsn_dir, '/streamlines_mni_warp_',
        #                                                                '%s' % (network + '_' if network is not
        #                                                                                         None else ''),
        #                                                                '%s' % (op.basename(roi).split('.')[0] + '_' if
        #                                                                        roi is not None else ''),
        #                                                                conn_model, '_', target_samples,
        #                                                                '%s' % ("%s%s" %
        #                                                                        ('_' + str(node_size),
        #                                                                         'mm_') if ((node_size != 'parc') and
        #                                                                                    (node_size is not None)) else
        #                                                                        '_'),
        #                                                                'curv', str(curv_thr_list).replace(', ', '_'),
        #                                                                'step', str(step_list).replace(', ', '_'), 'tracktype-',
        #                                                                track_type,  '_directget-', directget, '_minlength-', min_length,
        #                                                                '.png')

        # SyN FA->Template
        [mapping, affine_map, warped_fa] = regutils.wm_syn(
            template_path, fa_path, t1_aligned_mni, ap_path, dsn_dir
        )

        tractogram = load_tractogram(
            streams,
            fa_img,
            to_origin=Origin.NIFTI,
            to_space=Space.VOXMM,
            bbox_valid_check=False,
        )

        fa_img.uncache()
        streamlines = tractogram.streamlines
        warped_fa_img = nib.load(warped_fa)
        warped_fa_affine = warped_fa_img.affine
        warped_fa_shape = warped_fa_img.shape

        streams_in_curr_grid = transform_streamlines(
            streamlines, warped_fa_affine)

        # Create isocenter mapping where we anchor the origin transformation affine
        # to the corner of the FOV by scaling x, y, z offsets according to a multiplicative
        # van der Corput sequence with a base value equal to the voxel
        # resolution
        [x_mul, y_mul, z_mul] = [vdc(i, vox_size) for i in range(1, 4)]

        ref_grid_aff = vox_size * np.eye(4)
        ref_grid_aff[3][3] = 1

        streams_final_filt = []
        i = 0
        # Test for various types of voxel-grid configurations
        combs = [(-x_mul, -y_mul, -z_mul), (-x_mul, -y_mul, z_mul),
                 (-x_mul, y_mul, -z_mul), (x_mul, -y_mul, -z_mul),
                 (x_mul, y_mul, z_mul)]
        while len(streams_final_filt)/len(streams_in_curr_grid) < 0.90:
            print(f"Warping streamlines to MNI space. Attempt {i}...")
            print(len(streams_final_filt)/len(streams_in_curr_grid))
            adjusted_affine = affine_map.affine.copy()
            if i > len(combs) - 1:
                raise ValueError('DSN failed. Header orientation information '
                                 'may be corrupted. Is your dataset oblique?')
            adjusted_affine[0][3] = adjusted_affine[0][3] * combs[i][0]
            adjusted_affine[1][3] = adjusted_affine[1][3] * combs[i][1]
            adjusted_affine[2][3] = adjusted_affine[2][3] * combs[i][2]

            streams_final_filt = regutils.warp_streamlines(adjusted_affine,
                                                           ref_grid_aff,
                                                           mapping,
                                                           warped_fa_img,
                                                           streams_in_curr_grid,
                                                           brain_mask)

            i += 1

        # Remove streamlines with negative voxel indices
        lin_T, offset = _mapping_to_voxel(np.eye(4))
        streams_final_filt_final = []
        for sl in streams_final_filt:
            inds = np.dot(sl, lin_T)
            inds += offset
            if not inds.min().round(decimals=6) < 0:
                streams_final_filt_final.append(sl)

        # Save streamlines
        stf = StatefulTractogram(
            streams_final_filt_final,
            reference=uatlas_mni_img,
            space=Space.VOXMM,
            origin=Origin.NIFTI,
        )
        stf.remove_invalid_streamlines()
        streams_final_filt_final = stf.streamlines
        save_tractogram(stf, streams_mni, bbox_valid_check=True)
        warped_fa_img.uncache()

        # DSN QC plotting
        # plot_gen.show_template_bundles(streams_final_filt_final, atlas_mni, streams_warp_png)
        # plot_gen.show_template_bundles(streamlines, fa_path, streams_warp_png)

        # Create and save MNI density map
        nib.save(
            nib.Nifti1Image(
                utils.density_map(
                    streams_final_filt_final,
                    affine=np.eye(4),
                    vol_dims=warped_fa_shape),
                warped_fa_affine,
            ),
            density_mni,
        )

        # Map parcellation from native space back to MNI-space and create an 'uncertainty-union' parcellation
        # with original mni-space uatlas
        warped_uatlas = affine_map.transform_inverse(
            mapping.transform(
                np.asarray(atlas_img.dataobj).astype("int"),
                interpolation="nearestneighbour",
            ),
            interp="nearest",
        )
        atlas_img.uncache()
        warped_uatlas_img_res_data = np.asarray(
            resample_to_img(
                nib.Nifti1Image(warped_uatlas, affine=warped_fa_affine),
                uatlas_mni_img,
                interpolation="nearest",
                clip=False,
            ).dataobj
        )
        uatlas_mni_data = np.asarray(uatlas_mni_img.dataobj)
        uatlas_mni_img.uncache()
        overlap_mask = np.invert(
            warped_uatlas_img_res_data.astype("bool") *
            uatlas_mni_data.astype("bool"))
        os.makedirs(f"{dir_path}/parcellations", exist_ok=True)
        atlas_mni = f"{dir_path}/parcellations/{op.basename(uatlas).split('.nii')[0]}_liberal.nii.gz"

        nib.save(
            nib.Nifti1Image(
                warped_uatlas_img_res_data * overlap_mask.astype("int")
                + uatlas_mni_data * overlap_mask.astype("int")
                + np.invert(overlap_mask).astype("int")
                * warped_uatlas_img_res_data.astype("int"),
                affine=uatlas_mni_img.affine,
            ),
            atlas_mni,
        )

        del (
            tractogram,
            streamlines,
            warped_uatlas_img_res_data,
            uatlas_mni_data,
            overlap_mask,
            stf,
            streams_final_filt_final,
            streams_final_filt,
            streams_in_curr_grid,
            brain_mask,
        )

        gc.collect()

        assert len(coords) == len(labels)

    else:
        print(
            "Skipping Direct Streamline Normalization (DSN). Will proceed to "
            "define fiber connectivity in native diffusion space...")
        streams_mni = streams
        warped_fa = fa_path
        atlas_mni = labels_im_file

    return (
        streams_mni,
        dir_path,
        track_type,
        target_samples,
        conn_model,
        network,
        node_size,
        dens_thresh,
        ID,
        roi,
        min_span_tree,
        disp_filt,
        parc,
        prune,
        atlas,
        uatlas,
        labels,
        coords,
        norm,
        binary,
        atlas_mni,
        directget,
        warped_fa,
        min_length,
        error_margin,
    )


class DmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in
    native diffusion space.

    References
    ----------
    .. [1] Adluru, N., Zhang, H., Tromp, D. P. M., & Alexander, A. L. (2013).
      Effects of DTI spatial normalization on white matter tract reconstructions.
      Medical Imaging 2013: Image Processing. https://doi.org/10.1117/12.2007130
    .. [2] Greve DN, Fischl B. Accurate and robust brain image alignment using
      boundary-based registration. Neuroimage. 2009 Oct;48(1):63–72.
      doi:10.1016/j.neuroimage.2009.06.060.
    .. [3] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a
      hidden Markov random field model and the expectation-maximization algorithm.
      IEEE Trans Med Imaging. 2001 Jan;20(1):45–57. doi:10.1109/42.906424.
    """

    def __init__(
        self,
        basedir_path,
        fa_path,
        ap_path,
        B0_mask,
        anat_file,
        vox_size,
        template_name,
        simple,
    ):
        import pkg_resources
        import os.path as op

        self.simple = simple
        self.ap_path = ap_path
        self.fa_path = fa_path
        self.B0_mask = B0_mask
        self.t1w = anat_file
        self.vox_size = vox_size
        self.template_name = template_name
        self.t1w_name = "t1w"
        self.dwi_name = "dwi"
        self.basedir_path = basedir_path
        self.tmp_path = f"{basedir_path}{'/dmri_reg'}"
        self.reg_path = f"{basedir_path}{'/dmri_reg/reg'}"
        self.reg_path_mat = f"{self.reg_path}{'/mats'}"
        self.reg_path_warp = f"{self.reg_path}{'/warps'}"
        self.reg_path_img = f"{self.reg_path}{'/imgs'}"
        self.t12mni_xfm_init = f"{self.reg_path_mat}{'/xfm_t1w2mni_init.mat'}"
        self.t12mni_xfm = f"{self.reg_path_mat}{'/xfm_t1w2mni.mat'}"
        self.mni2t1_xfm = f"{self.reg_path_mat}{'/xfm_mni2t1.mat'}"
        self.mni2t1w_warp = f"{self.reg_path_warp}{'/mni2t1w_warp.nii.gz'}"
        self.warp_t1w2mni = f"{self.reg_path_warp}{'/t1w2mni_warp.nii.gz'}"
        self.t1w2dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_in_dwi.nii.gz'}"
        self.t1_aligned_mni = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_aligned_mni.nii.gz'}"
        )
        self.t1w_brain = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain.nii.gz'}"
        self.t1w_head = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_head.nii.gz'}"
        self.t1w_brain_mask = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain_mask.nii.gz'}"
        )
        self.t1w_brain_mask_in_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain_mask_in_dwi.nii.gz'}"
        )
        self.dwi2t1w_xfm = f"{self.reg_path_mat}{'/dwi2t1w_xfm.mat'}"
        self.t1w2dwi_xfm = f"{self.reg_path_mat}{'/t1w2dwi_xfm.mat'}"
        self.t1w2dwi_bbr_xfm = f"{self.reg_path_mat}{'/t1w2dwi_bbr_xfm.mat'}"
        self.dwi2t1w_bbr_xfm = f"{self.reg_path_mat}{'/dwi2t1w_bbr_xfm.mat'}"
        self.t1wtissue2dwi_xfm = f"{self.reg_path_mat}{'/t1wtissue2dwi_xfm.mat'}"
        self.temp2dwi_xfm = (
            f"{self.reg_path_mat}{'/'}{self.dwi_name}{'_xfm_temp2dwi.mat'}"
        )
        self.map_name = f"{self.t1w_name}{'_seg'}"
        self.wm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm.nii.gz'}"
        self.wm_mask_thr = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_thr.nii.gz'}"
        self.wm_edge = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_edge.nii.gz'}"
        self.csf_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_csf.nii.gz'}"
        self.gm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_gm.nii.gz'}"
        self.xfm_roi2mni_init = f"{self.reg_path_mat}{'/roi_2_mni.mat'}"
        self.mni_vent_loc = pkg_resources.resource_filename(
            "pynets", f"templates/LateralVentricles_{vox_size}.nii.gz"
        )
        self.csf_mask_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_csf_mask_dwi.nii.gz'}"
        )
        self.gm_in_dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_gm_in_dwi.nii.gz'}"
        self.wm_in_dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_in_dwi.nii.gz'}"
        self.csf_mask_dwi_bin = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_csf_mask_dwi_bin.nii.gz'}"
        )
        self.gm_in_dwi_bin = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_gm_in_dwi_bin.nii.gz'}"
        )
        self.wm_in_dwi_bin = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_in_dwi_bin.nii.gz'}"
        )
        self.vent_mask_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_vent_mask_dwi.nii.gz'}"
        )
        self.vent_csf_in_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_vent_csf_in_dwi.nii.gz'}"
        )
        self.vent_mask_mni = f"{self.reg_path_img}{'/vent_mask_mni.nii.gz'}"
        self.vent_mask_t1w = f"{self.reg_path_img}{'/vent_mask_t1w.nii.gz'}"
        self.input_mni = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_{vox_size}.nii.gz"
        )
        self.input_mni_brain = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_" f"brain_{vox_size}.nii.gz"
        )
        self.input_mni_mask = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_"
            f"brain_mask_{vox_size}.nii.gz")
        self.mni_atlas = pkg_resources.resource_filename(
            "pynets", f"core/atlases/HarvardOxford-sub-prob-{vox_size}.nii.gz"
        )
        self.wm_gm_int_in_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_gm_int_in_dwi.nii.gz'}"
        )
        self.wm_gm_int_in_dwi_bin = (
            f"{self.reg_path_img}/{self.t1w_name}_wm_gm_int_in_dwi_bin.nii.gz"
        )
        self.corpuscallosum = pkg_resources.resource_filename(
            "pynets", f"templates/CorpusCallosum_{vox_size}.nii.gz"
        )
        self.corpuscallosum_mask_t1w = (
            f"{self.reg_path_img}{'/CorpusCallosum_t1wmask.nii.gz'}"
        )
        self.corpuscallosum_dwi = f"{self.reg_path_img}{'/CorpusCallosum_dwi.nii.gz'}"

        # Create empty tmp directories that do not yet exist
        reg_dirs = [
            self.tmp_path,
            self.reg_path,
            self.reg_path_mat,
            self.reg_path_warp,
            self.reg_path_img,
        ]
        for i in range(len(reg_dirs)):
            if not op.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

    def gen_mask(self, mask):
        import os.path as op

        if op.isfile(self.t1w_brain) is False:
            import shutil

            shutil.copyfile(self.t1w, self.t1w_head)

        [self.t1w_brain, self.t1w_brain_mask] = regutils.gen_mask(
            self.t1w_head, self.t1w_brain, mask
        )
        return

    def gen_tissue(
        self, wm_mask_existing, gm_mask_existing, csf_mask_existing, overwrite
    ):
        """
        A function to segment and threshold tissue types from T1w.
        """
        import shutil

        # Segment the t1w brain into probability maps
        if (
            wm_mask_existing is not None
            and gm_mask_existing is not None
            and csf_mask_existing is not None
            and overwrite is False
        ):
            print("Existing segmentations detected...")
            wm_mask = regutils.check_orient_and_dims(
                wm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            gm_mask = regutils.check_orient_and_dims(
                gm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            csf_mask = regutils.check_orient_and_dims(
                csf_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
        else:
            try:
                maps = regutils.segment_t1w(self.t1w_brain, self.map_name)
                wm_mask = maps["wm_prob"]
                gm_mask = maps["gm_prob"]
                csf_mask = maps["csf_prob"]
            except RuntimeError:
                print(
                    "Segmentation failed. Does the input anatomical image "
                    "still contained skull?"
                )

        # Threshold WM to binary in dwi space
        t_img = nib.load(wm_mask)
        mask = math_img("img > 0.20", img=t_img)
        mask.to_filename(self.wm_mask_thr)

        # Extract wm edge
        self.wm_edge = regutils.get_wm_contour(wm_mask, self.wm_mask_thr,
                                               self.wm_edge)

        shutil.copyfile(wm_mask, self.wm_mask)
        shutil.copyfile(gm_mask, self.gm_mask)
        shutil.copyfile(csf_mask, self.csf_mask)

        return

    def t1w2mni_align(self):
        """
        A function to perform alignment from T1w --> MNI template.
        """

        # Create linear transform/ initializer T1w-->MNI
        regutils.align(
            self.t1w_brain,
            self.input_mni_brain,
            xfm=self.t12mni_xfm_init,
            bins=None,
            interp="spline",
            out=None,
            dof=12,
            cost="mutualinfo",
            searchrad=True,
        )

        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print(
                    f"Learning a non-linear mapping from T1w --> "
                    f"{self.template_name} ..."
                )
                # Use FNIRT to nonlinearly align T1 to MNI template
                regutils.align_nonlinear(
                    self.t1w_brain,
                    self.input_mni,
                    xfm=self.t12mni_xfm_init,
                    out=self.t1_aligned_mni,
                    warp=self.warp_t1w2mni,
                    ref_mask=self.input_mni_mask,
                )

                # Get warp from MNI -> T1
                regutils.inverse_warp(
                    self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni
                )

                # Get mat from MNI -> T1
                self.mni2t1_xfm = regutils.invert_xfm(self.t12mni_xfm_init,
                                                      self.mni2t1_xfm)

            except BaseException:
                # Falling back to linear registration
                regutils.align(
                    self.t1w_brain,
                    self.input_mni_brain,
                    xfm=self.mni2t1_xfm,
                    init=self.t12mni_xfm_init,
                    bins=None,
                    dof=12,
                    cost="mutualinfo",
                    searchrad=True,
                    interp="spline",
                    out=self.t1_aligned_mni,
                    sch=None,
                )
                # Get mat from MNI -> T1
                self.mni2t1_xfm = regutils.invert_xfm(self.t12mni_xfm,
                                                      self.mni2t1_xfm)
        else:
            # Falling back to linear registration
            regutils.align(
                self.t1w_brain,
                self.input_mni_brain,
                xfm=self.t12mni_xfm,
                init=self.t12mni_xfm_init,
                bins=None,
                dof=12,
                cost="mutualinfo",
                searchrad=True,
                interp="spline",
                out=self.t1_aligned_mni,
                sch=None,
            )
            # Get mat from MNI -> T1
            self.t12mni_xfm = regutils.invert_xfm(self.mni2t1_xfm,
                                                  self.t12mni_xfm)

    def t1w2dwi_align(self):
        """
        A function to perform alignment from T1w_MNI --> DWI. Uses a local
        optimization cost function to get the two images close, and then uses
        bbr to obtain a good alignment of brain boundaries.
        Assumes input dwi is already preprocessed and brain extracted.
        """

        # Align T1w-->DWI
        regutils.align(
            self.ap_path,
            self.t1w_brain,
            xfm=self.t1w2dwi_xfm,
            bins=None,
            interp="spline",
            dof=6,
            cost="mutualinfo",
            out=None,
            searchrad=True,
            sch=None,
        )
        self.dwi2t1w_xfm = regutils.invert_xfm(self.t1w2dwi_xfm,
                                               self.dwi2t1w_xfm)

        if self.simple is False:
            # Flirt bbr
            try:
                print("Learning a Boundary-Based Mapping from T1w-->DWI ...")
                regutils.align(
                    self.fa_path,
                    self.t1w_brain,
                    wmseg=self.wm_edge,
                    xfm=self.dwi2t1w_bbr_xfm,
                    init=self.dwi2t1w_xfm,
                    bins=256,
                    dof=7,
                    searchrad=True,
                    interp="spline",
                    out=None,
                    cost="bbr",
                    sch="${FSLDIR}/etc/flirtsch/bbr.sch",
                )
                self.t1w2dwi_bbr_xfm = regutils.invert_xfm(
                    self.dwi2t1w_bbr_xfm, self.t1w2dwi_bbr_xfm)

                # Apply the alignment
                regutils.align(
                    self.t1w_brain,
                    self.ap_path,
                    init=self.t1w2dwi_bbr_xfm,
                    xfm=self.t1wtissue2dwi_xfm,
                    bins=None,
                    interp="spline",
                    dof=7,
                    cost="mutualinfo",
                    out=self.t1w2dwi,
                    searchrad=True,
                    sch=None,
                )
            except BaseException:
                # Apply the alignment
                regutils.align(
                    self.t1w_brain,
                    self.ap_path,
                    init=self.t1w2dwi_xfm,
                    xfm=self.t1wtissue2dwi_xfm,
                    bins=None,
                    interp="spline",
                    dof=7,
                    cost="mutualinfo",
                    out=self.t1w2dwi,
                    searchrad=True,
                    sch=None,
                )
        else:
            # Apply the alignment
            regutils.align(
                self.t1w_brain,
                self.ap_path,
                init=self.t1w2dwi_xfm,
                xfm=self.t1wtissue2dwi_xfm,
                bins=None,
                interp="spline",
                dof=6,
                cost="mutualinfo",
                out=self.t1w2dwi,
                searchrad=True,
                sch=None,
            )

        return

    def tissue2dwi_align(self):
        """
        A function to perform alignment of ventricle ROI's from MNI space --> dwi and CSF from T1w space --> dwi.
        First generates and performs dwi space alignment of avoidance/waypoint masks for tractography.
        First creates ventricle ROI. Then creates transforms from stock MNI template to dwi space.
        For this to succeed, must first have called both t1w2dwi_align.
        """
        import os.path as op

        # Register Lateral Ventricles and Corpus Callosum rois to t1w
        if not op.isfile(self.mni_atlas):
            raise ValueError("FSL atlas for ventricle reference not found!")

        # Create transform to MNI atlas to T1w using flirt. This will be use to
        # transform the ventricles to dwi space.
        regutils.align(
            self.mni_atlas,
            self.input_mni_brain,
            xfm=self.xfm_roi2mni_init,
            init=None,
            bins=None,
            dof=6,
            cost="mutualinfo",
            searchrad=True,
            interp="spline",
            out=None,
        )

        try:
            nib.load(self.mni_vent_loc)
        except indexed_gzip.ZranError as e:
            print(e,
                  f"\nCannot load ventricle ROI. Do you have git-lfs "
                  f"installed?")

        # Create transform to align roi to mni and T1w using flirt
        regutils.applyxfm(
            self.input_mni_brain,
            self.mni_vent_loc,
            self.xfm_roi2mni_init,
            self.vent_mask_mni,
        )

        if self.simple is False:
            # Apply warp resulting from the inverse MNI->T1w created earlier
            regutils.apply_warp(
                self.t1w_brain,
                self.vent_mask_mni,
                self.vent_mask_t1w,
                warp=self.mni2t1w_warp,
                interp="nn",
                sup=True,
            )

            try:
                nib.load(self.corpuscallosum)
            except indexed_gzip.ZranError as e:
                print(e,
                      f"\nCannot load Corpus Callosum ROI. Do you have "
                      f"git-lfs installed?")

            regutils.apply_warp(
                self.t1w_brain,
                self.corpuscallosum,
                self.corpuscallosum_mask_t1w,
                warp=self.mni2t1w_warp,
                interp="nn",
                sup=True,
            )

        else:
            regutils.applyxfm(
                self.vent_mask_mni,
                self.t1w_brain,
                self.mni2t1_xfm,
                self.vent_mask_t1w)
            regutils.applyxfm(
                self.corpuscallosum,
                self.t1w_brain,
                self.mni2t1_xfm,
                self.corpuscallosum_mask_t1w,
            )

        # Applyxfm tissue maps to dwi space
        if self.t1w_brain_mask is not None:
            regutils.applyxfm(
                self.ap_path,
                self.t1w_brain_mask,
                self.t1wtissue2dwi_xfm,
                self.t1w_brain_mask_in_dwi,
            )
        regutils.applyxfm(
            self.ap_path,
            self.vent_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.vent_mask_dwi)
        regutils.applyxfm(
            self.ap_path,
            self.csf_mask,
            self.t1wtissue2dwi_xfm,
            self.csf_mask_dwi)
        regutils.applyxfm(
            self.ap_path, self.gm_mask, self.t1wtissue2dwi_xfm, self.gm_in_dwi
        )
        regutils.applyxfm(
            self.ap_path, self.wm_mask, self.t1wtissue2dwi_xfm, self.wm_in_dwi
        )
        regutils.applyxfm(
            self.ap_path,
            self.corpuscallosum_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.corpuscallosum_dwi,
        )

        # Threshold WM to binary in dwi space
        thr_img = nib.load(self.wm_in_dwi)
        thr_img = math_img("img > 0.20", img=thr_img)
        nib.save(thr_img, self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        thr_img = nib.load(self.gm_in_dwi)
        thr_img = math_img("img > 0.15", img=thr_img)
        nib.save(thr_img, self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        thr_img = nib.load(self.csf_mask_dwi)
        thr_img = math_img("img > 0.95", img=thr_img)
        nib.save(thr_img, self.csf_mask_dwi_bin)

        # Threshold WM to binary in dwi space
        self.wm_in_dwi = regutils.apply_mask_to_image(self.wm_in_dwi,
                                                      self.wm_in_dwi_bin,
                                                      self.wm_in_dwi)

        # Threshold GM to binary in dwi space
        self.gm_in_dwi = regutils.apply_mask_to_image(self.gm_in_dwi,
                                                      self.gm_in_dwi_bin,
                                                      self.gm_in_dwi)

        # Threshold CSF to binary in dwi space
        self.csf_mask = regutils.apply_mask_to_image(self.csf_mask_dwi,
                                                     self.csf_mask_dwi_bin,
                                                     self.csf_mask_dwi)

        # Create ventricular CSF mask
        print("Creating Ventricular CSF mask...")
        os.system(
            f"fslmaths {self.vent_mask_dwi} -kernel sphere 10 -ero -bin {self.vent_mask_dwi}"
        )
        os.system(
            f"fslmaths {self.csf_mask_dwi} -add {self.vent_mask_dwi} -bin {self.vent_csf_in_dwi}"
        )

        print("Creating Corpus Callosum mask...")
        os.system(
            f"fslmaths {self.corpuscallosum_dwi} -mas {self.wm_in_dwi_bin} -sub {self.vent_csf_in_dwi} "
            f"-bin {self.corpuscallosum_dwi}")

        # Create gm-wm interface image
        os.system(
            f"fslmaths {self.gm_in_dwi_bin} -mul {self.wm_in_dwi_bin} -add {self.corpuscallosum_dwi} "
            f"-mas {self.B0_mask} -bin {self.wm_gm_int_in_dwi}")

        return


class FmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in native epi space.
    References
    ----------
    .. [1] Brett M, Leff AP, Rorden C, Ashburner J (2001) Spatial Normalization
      of Brain Images with Focal Lesions Using Cost Function Masking.
      NeuroImage 14(2) doi:10.006/nimg.2001.0845.
    .. [2] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a
      hidden Markov random field model and the expectation-maximization algorithm.
      IEEE Trans Med Imaging. 2001 Jan;20(1):45–57. doi:10.1109/42.906424.
    """

    def __init__(
            self,
            basedir_path,
            anat_file,
            vox_size,
            template_name,
            simple):
        import os.path as op
        import pkg_resources

        self.t1w = anat_file
        self.vox_size = vox_size
        self.template_name = template_name
        self.t1w_name = "t1w"
        self.simple = simple
        self.basedir_path = basedir_path
        self.reg_path = f"{basedir_path}{'/reg'}"
        self.reg_path_mat = f"{self.reg_path}{'/mats'}"
        self.reg_path_warp = f"{self.reg_path}{'/warps'}"
        self.reg_path_img = f"{self.reg_path}{'/imgs'}"
        self.t1w2epi_xfm = f"{self.reg_path_mat}{'/t1w2epi_xfm.mat'}"
        self.t12mni_xfm_init = f"{self.reg_path_mat}{'/xfm_t1w2mni.mat'}"
        self.t12mni_xfm = f"{self.reg_path_mat}{'/xfm_t1w2mni.mat'}"
        self.mni2t1_xfm = f"{self.reg_path_mat}{'/xfm_mni2t1.mat'}"
        self.mni2t1w_warp = f"{self.reg_path_warp}{'/mni2t1w_warp.nii.gz'}"
        self.warp_t1w2mni = f"{self.reg_path_warp}{'/t1w2mni_warp.nii.gz'}"
        self.t1_aligned_mni = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_aligned_mni.nii.gz'}"
        )
        self.t1w_brain = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain.nii.gz'}"
        self.t1w_head = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_head.nii.gz'}"
        self.t1w_brain_mask = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain_mask.nii.gz'}"
        )
        self.map_name = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_seg'}"
        self.gm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_gm.nii.gz'}"
        self.gm_mask_thr = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_gm_thr.nii.gz'}"
        self.wm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm.nii.gz'}"
        self.wm_mask_thr = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_thr.nii.gz'}"
        self.wm_edge = f"{self.reg_path_img}{'/'}{self.t1w_name}{'_wm_edge.nii.gz'}"
        self.input_mni = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_{vox_size}.nii.gz"
        )
        self.input_mni_brain = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_" f"brain_{vox_size}.nii.gz"
        )
        self.input_mni_mask = pkg_resources.resource_filename(
            "pynets", f"templates/{self.template_name}_"
            f"brain_mask_{vox_size}.nii.gz")

        # Create empty tmp directories that do not yet exist
        reg_dirs = [
            self.reg_path,
            self.reg_path_mat,
            self.reg_path_warp,
            self.reg_path_img,
        ]
        for i in range(len(reg_dirs)):
            if not op.isdir(reg_dirs[i]):
                os.mkdir(reg_dirs[i])

    def gen_mask(self, mask):
        import os.path as op

        if op.isfile(self.t1w_brain) is False:
            import shutil

            shutil.copyfile(self.t1w, self.t1w_head)
        [self.t1w_brain, self.t1w_brain_mask] = regutils.gen_mask(
            self.t1w_head, self.t1w_brain, mask
        )
        return

    def gen_tissue(self, wm_mask_existing, gm_mask_existing, overwrite):
        """
        A function to segment and threshold tissue types from T1w.
        """

        # Segment the t1w brain into probability maps
        if (
            wm_mask_existing is not None
            and gm_mask_existing is not None
            and overwrite is False
        ):
            print("Existing segmentations detected...")
            gm_mask = regutils.check_orient_and_dims(
                gm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            wm_mask = regutils.check_orient_and_dims(
                wm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
        else:
            try:
                maps = regutils.segment_t1w(self.t1w_brain, self.map_name)
                gm_mask = maps["gm_prob"]
                wm_mask = maps["wm_prob"]
            except RuntimeError:
                print(
                    "Segmentation failed. Does the input anatomical image "
                    "still contained skull?"
                )

        # Threshold GM to binary in func space
        t_img = nib.load(gm_mask)
        mask = math_img("img > 0.01", img=t_img)
        mask.to_filename(self.gm_mask_thr)
        self.gm_mask = regutils.apply_mask_to_image(gm_mask,
                                                    self.gm_mask_thr,
                                                    self.gm_mask)

        # Threshold WM to binary in dwi space
        t_img = nib.load(wm_mask)
        mask = math_img("img > 0.50", img=t_img)
        mask.to_filename(self.wm_mask_thr)
        self.wm_mask = regutils.apply_mask_to_image(wm_mask,
                                                    self.wm_mask_thr,
                                                    self.wm_mask)
        # Extract wm edge
        self.wm_edge = regutils.get_wm_contour(wm_mask, self.wm_mask_thr,
                                               self.wm_edge)

        return

    def t1w2mni_align(self):
        """
        A function to perform alignment from T1w --> MNI.
        """

        # Create linear transform/ initializer T1w-->MNI
        regutils.align(
            self.t1w_brain,
            self.input_mni_brain,
            xfm=self.t12mni_xfm_init,
            bins=None,
            interp="spline",
            out=None,
            dof=12,
            cost="mutualinfo",
            searchrad=True,
        )

        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print(
                    f"Learning a non-linear mapping from T1w --> "
                    f"{self.template_name} ..."
                )
                # Use FNIRT to nonlinearly align T1w to MNI template
                regutils.align_nonlinear(
                    self.t1w_brain,
                    self.input_mni,
                    xfm=self.t12mni_xfm_init,
                    out=self.t1_aligned_mni,
                    warp=self.warp_t1w2mni,
                    ref_mask=self.input_mni_mask,
                )

                # Get warp from T1w --> MNI
                regutils.inverse_warp(
                    self.t1w_brain,  self.mni2t1w_warp, self.warp_t1w2mni
                )

                # Get mat from MNI -> T1w
                self.mni2t1_xfm = regutils.invert_xfm(self.t12mni_xfm_init,
                                                      self.mni2t1_xfm)

            except BaseException:
                # Falling back to linear registration
                regutils.align(
                    self.t1w_brain,
                    self.input_mni_brain,
                    xfm=self.t12mni_xfm,
                    init=self.t12mni_xfm_init,
                    bins=None,
                    dof=12,
                    cost="mutualinfo",
                    searchrad=True,
                    interp="spline",
                    out=self.t1_aligned_mni,
                    sch=None,
                )
                # Get mat from MNI -> T1w
                self.t12mni_xfm = regutils.invert_xfm(self.mni2t1_xfm,
                                                      self.t12mni_xfm)
        else:
            # Falling back to linear registration
            regutils.align(
                self.t1w_brain,
                self.input_mni_brain,
                xfm=self.t12mni_xfm,
                init=self.t12mni_xfm_init,
                bins=None,
                dof=12,
                cost="mutualinfo",
                searchrad=True,
                interp="spline",
                out=self.t1_aligned_mni,
                sch=None,
            )
            # Get mat from MNI -> T1w
            self.t12mni_xfm = regutils.invert_xfm(self.mni2t1_xfm,
                                                  self.t12mni_xfm)
        return

