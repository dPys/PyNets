#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import matplotlib
import os
import nibabel as nib
import warnings
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import numpy as np
from pynets.registration import utils as regutils
from nilearn.image import math_img

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
try:
    FSLDIR = os.environ["FSLDIR"]
except KeyError as e:
    print(e, "FSLDIR environment variable not set!")
    sys.exit(1)


def direct_streamline_norm(
    streams,
    fa_path,
    ap_path,
    dir_path,
    track_type,
    conn_model,
    subnet,
    node_radius,
    dens_thresh,
    ID,
    roi,
    min_span_tree,
    disp_filt,
    parc,
    prune,
    atlas,
    labels_im_file,
    parcellation,
    labels,
    coords,
    norm,
    binary,
    atlas_t1w,
    basedir_path,
    curv_thr_list,
    step_list,
    traversal,
    min_length,
    t1w_brain,
    run_dsn=False
):
    """
    A Function to perform normalization of streamlines tracked in native
    diffusion space to an MNI-space template.

    Parameters
    ----------
    streams : str
        File path to save streamline array sequence in .trk format.
    fa_path : str
        File path to FA Nifti1Image.
    ap_path : str
        File path to the anisotropic power Nifti1Image.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
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
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    labels_im_file : str
        File path to atlas parcellation Nifti1Image aligned to dwi space.
    parcellation : str
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
    atlas_t1w : str
        File path to atlas parcellation Nifti1Image in T1w-conformed space.
    basedir_path : str
        Path to directory to output direct-streamline normalized temp files
        and outputs.
    curv_thr_list : list
        List of integer curvature thresholds used to perform ensemble tracking.
    step_list : list
        List of float step-sizes used to perform ensemble tracking.
    traversal : str
        The statistical approach to tracking. Options are: det (deterministic),
        closest (clos), boot (bootstrapped), and prob (probabilistic).
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.
    t1w_brain : str
        File path to the T1w Nifti1Image.

    Returns
    -------
    streams_warp : str
        File path to normalized streamline array sequence in .trk format.
    dir_path : str
        Path to directory containing subject derivative data for a given
        pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
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
        'backbone subnet' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    parcellation : str
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
    atlas_for_streams : str
        File path to atlas parcellation Nifti1Image in the same
        morphological space as the streamlines.
    traversal : str
        The statistical approach to tracking. Options are: det
        (deterministic), closest (clos), boot (bootstrapped),
        and prob (probabilistic).
    warped_fa : str
        File path to MNI-space warped FA Nifti1Image.
    min_length : int
        Minimum fiber length threshold in mm to restrict tracking.

    References
    ----------
    .. [1] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of
      different spatial normalization approaches on tractography and structural
      brain subnets. subnet Neuroscience, 1-19.
    """
    import gc
    from dipy.tracking.streamline import transform_streamlines
    from pynets.registration import utils as regutils
    from pynets.plotting.brain import show_template_bundles
    import os.path as op
    from dipy.io.streamline import load_tractogram
    from dipy.tracking._utils import _mapping_to_voxel
    from dipy.tracking.utils import density_map
    from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
    from dipy.io.streamline import save_tractogram

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

        atlas_t1w_img = nib.load(atlas_t1w)
        t1w_brain_img = nib.load(t1w_brain)
        brain_mask = np.asarray(t1w_brain_img.dataobj).astype("bool")

        streams_t1w = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
            namer_dir,
            "/streamlines_t1w_",
            "%s" % (subnet + "_" if subnet is not None else ""),
            "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None
                    else ""),
            conn_model,
            "%s"
            % (
                "%s%s" % ("_" + str(node_radius), "mm_")
                if ((node_radius != "parc") and (node_radius is not None))
                else "_"
            ),
            "curv",
            str(curv_thr_list).replace(", ", "_"),
            "step",
            str(step_list).replace(", ", "_"),
            "tracktype-",
            track_type,
            "_traversal-",
            traversal,
            "_minlength-",
            min_length,
            ".trk",
        )

        density_t1w = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
            namer_dir,
            "/density_map_t1w_",
            "%s" % (subnet + "_" if subnet is not None else ""),
            "%s" % (op.basename(roi).split(".")[0] + "_" if roi is not None
                    else ""),
            conn_model,
            "%s"
            % (
                "%s%s" % ("_" + str(node_radius), "mm_")
                if ((node_radius != "parc") and (node_radius is not None))
                else "_"
            ),
            "curv",
            str(curv_thr_list).replace(", ", "_"),
            "step",
            str(step_list).replace(", ", "_"),
            "tracktype-",
            track_type,
            "_traversal-",
            traversal,
            "_minlength-",
            min_length,
            ".nii.gz",
        )

        streams_warp_png = '/tmp/dsn.png'

        # SyN FA->Template
        [mapping, affine_map, warped_fa] = regutils.wm_syn(
            t1w_brain, ap_path, dsn_dir
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
            streamlines, affine_map.affine_inv)

        streams_final_filt = regutils.warp_streamlines(t1w_brain_img.affine,
                                                       fa_img.affine,
                                                       mapping,
                                                       warped_fa_img,
                                                       streams_in_curr_grid,
                                                       brain_mask)

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
            reference=t1w_brain_img,
            space=Space.VOXMM,
            origin=Origin.NIFTI,
        )
        stf.remove_invalid_streamlines()
        streams_final_filt_final = stf.streamlines
        save_tractogram(stf, streams_t1w, bbox_valid_check=True)
        warped_fa_img.uncache()

        # DSN QC plotting
        show_template_bundles(streams_final_filt_final, atlas_t1w,
        streams_warp_png)

        nib.save(
            nib.Nifti1Image(
                density_map(
                    streams_final_filt_final,
                    affine=np.eye(4),
                    vol_dims=warped_fa_shape),
                warped_fa_affine,
            ),
            density_t1w,
        )

        del (
            tractogram,
            streamlines,
            stf,
            streams_final_filt_final,
            streams_final_filt,
            streams_in_curr_grid,
            brain_mask,
        )

        gc.collect()

        assert len(coords) == len(labels)

        atlas_for_streams = atlas_t1w

    else:
        print(
            "Skipping Direct Streamline Normalization (DSN). Will proceed to "
            "define fiber connectivity in native diffusion space...")
        streams_t1w = streams
        warped_fa = fa_path
        atlas_for_streams = labels_im_file

    return (
        streams_t1w,
        dir_path,
        track_type,
        conn_model,
        subnet,
        node_radius,
        dens_thresh,
        ID,
        roi,
        min_span_tree,
        disp_filt,
        parc,
        prune,
        atlas,
        parcellation,
        labels,
        coords,
        norm,
        binary,
        atlas_for_streams,
        traversal,
        warped_fa,
        min_length
    )


class DmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in
    native diffusion space.

    References
    ----------
    .. [1] Adluru, N., Zhang, H., Tromp, D. P. M., & Alexander, A. L. (2013).
      Effects of DTI spatial normalization on white matter tract
      reconstructions. Medical Imaging 2013: Image Processing.
      https://doi.org/10.1117/12.2007130
    .. [2] Greve DN, Fischl B. Accurate and robust brain image alignment using
      boundary-based registration. Neuroimage. 2009 Oct;48(1):63–72.
      doi:10.1016/j.neuroimage.2009.06.060.
    .. [3] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a
      hidden Markov random field model and the expectation-maximization
      algorithm. IEEE Trans Med Imaging. 2001 Jan;20(1):45–57.
      doi:10.1109/42.906424.
    """
    __slots__ = ('simple', 'ap_path', 'fa_path', 'B0_mask', 't1w',
                 'vox_size', 'template_name', 't1w_name', 'dwi_name',
                 'basedir_path', 'tmp_path', 'reg_path', 'reg_path_mat',
                 'reg_path_warp', 'reg_path_img', 't12mni_xfm_init',
                 't12mni_xfm', 'mni2t1_xfm', 'mni2t1w_warp', 'warp_t1w2mni',
                 't1w2dwi', 't1_aligned_mni', 't1w_brain', 't1w_head',
                 't1w_brain_mask', 't1w_brain_mask_in_dwi', 'dwi2t1w_xfm',
                 't1w2dwi_xfm', 't1w2dwi_bbr_xfm', 'dwi2t1w_bbr_xfm',
                 't1wtissue2dwi_xfm', 'temp2dwi_xfm', 'map_name', 'wm_mask',
                 'wm_mask_thr', 'wm_edge', 'csf_mask', 'gm_mask',
                 'xfm_roi2mni_init', 'csf_mask_dwi', 'gm_in_dwi', 'wm_in_dwi',
                 'csf_mask_dwi_bin', 'gm_in_dwi_bin', 'wm_in_dwi_bin',
                 'vent_mask_dwi', 'vent_csf_in_dwi', 'mni_vent_loc',
                 'vent_mask_mni', 'vent_mask_t1w', 'input_mni',
                 'input_mni_brain', 'input_mni_mask', 'mni_atlas',
                 'mni_roi_ref', 'wm_gm_int_in_dwi', 'wm_gm_int_in_dwi_bin',
                 'corpuscallosum', 'corpuscallosum_mask_t1w',
                 'corpuscallosum_dwi', 'fa_template_res', 'fa_template_t1w')

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
        self.t1w2dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                       f"{'_in_dwi.nii.gz'}"
        self.t1_aligned_mni = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_aligned_mni.nii.gz'}"
        )
        self.t1w_brain = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                         f"{'_brain.nii.gz'}"
        self.t1w_head = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                        f"{'_head.nii.gz'}"
        self.t1w_brain_mask = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_brain_mask.nii.gz'}"
        )
        self.t1w_brain_mask_in_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}"
            f"{'_brain_mask_in_dwi.nii.gz'}"
        )
        self.dwi2t1w_xfm = f"{self.reg_path_mat}{'/dwi2t1w_xfm.mat'}"
        self.t1w2dwi_xfm = f"{self.reg_path_mat}{'/t1w2dwi_xfm.mat'}"
        self.t1w2dwi_bbr_xfm = f"{self.reg_path_mat}{'/t1w2dwi_bbr_xfm.mat'}"
        self.dwi2t1w_bbr_xfm = f"{self.reg_path_mat}{'/dwi2t1w_bbr_xfm.mat'}"
        self.t1wtissue2dwi_xfm = f"{self.reg_path_mat}" \
                                 f"{'/t1wtissue2dwi_xfm.mat'}"
        self.temp2dwi_xfm = (
            f"{self.reg_path_mat}{'/'}{self.dwi_name}{'_xfm_temp2dwi.mat'}"
        )
        self.map_name = f"{self.t1w_name}{'_seg'}"
        self.wm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                       f"{'_wm.nii.gz'}"
        self.wm_mask_thr = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                           f"{'_wm_thr.nii.gz'}"
        self.wm_edge = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                       f"{'_wm_edge.nii.gz'}"
        self.csf_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                        f"{'_csf.nii.gz'}"
        self.gm_mask = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                       f"{'_gm.nii.gz'}"
        self.xfm_roi2mni_init = f"{self.reg_path_mat}{'/roi_2_mni.mat'}"
        self.csf_mask_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}{'_csf_mask_dwi.nii.gz'}"
        )
        self.gm_in_dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                         f"{'_gm_in_dwi.nii.gz'}"
        self.wm_in_dwi = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                         f"{'_wm_in_dwi.nii.gz'}"
        self.csf_mask_dwi_bin = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}"
            f"{'_csf_mask_dwi_bin.nii.gz'}"
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
            f"{self.reg_path_img}{'/'}{self.t1w_name}"
            f"{'_vent_csf_in_dwi.nii.gz'}"
        )
        self.mni_vent_loc = pkg_resources.resource_filename(
            "pynets", f"templates/rois/LateralVentricles_{vox_size}.nii.gz"
        )
        self.vent_mask_mni = f"{self.reg_path_img}{'/vent_mask_mni.nii.gz'}"
        self.vent_mask_t1w = f"{self.reg_path_img}{'/vent_mask_t1w.nii.gz'}"
        self.input_mni = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
                      f"{vox_size}.nii.gz"
        )
        self.input_mni_brain = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
                      f"brain_{vox_size}.nii.gz"
        )
        self.input_mni_mask = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
            f"brain_mask_{vox_size}.nii.gz")
        self.mni_atlas = pkg_resources.resource_filename(
            "pynets", f"templates/atlases/HarvardOxford-sub-prob-"
                      f"{vox_size}.nii.gz"
        )
        self.mni_roi_ref = (
            f"{self.reg_path_img}/HarvardOxford_rois_in-"
            f"{self.template_name}_{vox_size}.nii.gz"
        )
        self.wm_gm_int_in_dwi = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}"
            f"{'_wm_gm_int_in_dwi.nii.gz'}"
        )
        self.wm_gm_int_in_dwi_bin = (
            f"{self.reg_path_img}/{self.t1w_name}_wm_gm_int_in_dwi_bin.nii.gz"
        )
        self.corpuscallosum = pkg_resources.resource_filename(
            "pynets", f"templates/rois/CorpusCallosum_{vox_size}.nii.gz"
        )
        self.corpuscallosum_mask_t1w = (
            f"{self.reg_path_img}{'/CorpusCallosum_t1wmask.nii.gz'}"
        )
        self.corpuscallosum_dwi = f"{self.reg_path_img}" \
                                  f"{'/CorpusCallosum_dwi.nii.gz'}"
        self.fa_template_res = f"{self.reg_path_img}" \
            f"{'/FA_template_res.nii.gz'}"
        self.fa_template_t1w = f"{self.reg_path_img}" \
            f"{'/FA_template_T1w.nii.gz'}"

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
        import time
        import shutil

        # Segment the t1w brain into probability maps
        if (
            wm_mask_existing is not None
            and gm_mask_existing is not None
            and csf_mask_existing is not None
            and overwrite is False
        ):
            print("Existing segmentations detected...")
            wm_mask = regutils.orient_reslice(
                wm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            gm_mask = regutils.orient_reslice(
                gm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            csf_mask = regutils.orient_reslice(
                csf_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
        else:
            try:
                maps = regutils.segment_t1w(self.t1w_brain, self.map_name)
                wm_mask = maps["wm_prob"]
                gm_mask = maps["gm_prob"]
                csf_mask = maps["csf_prob"]
            except RuntimeError as e:
                print(e,
                      "Segmentation failed. Does the input anatomical image "
                      "still contained skull?"
                      )

        # Threshold WM to binary in dwi space
        math_img("img > 0.10", img=nib.load(wm_mask)
                 ).to_filename(self.wm_mask_thr)

        # Extract wm edge
        self.wm_edge = regutils.get_wm_contour(wm_mask, self.wm_mask_thr,
                                               self.wm_edge)
        time.sleep(0.5)
        shutil.copyfile(wm_mask, self.wm_mask)
        shutil.copyfile(gm_mask, self.gm_mask)
        shutil.copyfile(csf_mask, self.csf_mask)

        return

    def t1w2mni_align(self):
        """
        A function to perform alignment from T1w --> MNI template.
        """
        import time

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
        time.sleep(0.5)
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
                time.sleep(0.5)
                # Get warp from MNI -> T1
                regutils.inverse_warp(
                    self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni
                )
                time.sleep(0.5)
                # Get mat from MNI -> T1
                self.mni2t1_xfm = regutils.invert_xfm(self.t12mni_xfm_init,
                                                      self.mni2t1_xfm)
                time.sleep(0.5)
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
                time.sleep(0.5)
                # Get mat from MNI -> T1
                self.mni2t1_xfm = regutils.invert_xfm(self.t12mni_xfm,
                                                      self.mni2t1_xfm)
                time.sleep(0.5)
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
            time.sleep(0.5)
            # Get mat from MNI -> T1
            self.t12mni_xfm = regutils.invert_xfm(self.mni2t1_xfm,
                                                  self.t12mni_xfm)
            time.sleep(0.5)

    def t1w2dwi_align(self):
        """
        A function to perform alignment from T1w_MNI --> DWI. Uses a local
        optimization cost function to get the two images close, and then uses
        bbr to obtain a good alignment of brain boundaries.
        Assumes input dwi is already preprocessed and brain extracted.
        """
        import time

        self.ap_path = regutils.apply_mask_to_image(self.ap_path,
                                                    self.B0_mask,
                                                    self.ap_path)

        self.fa_path = regutils.apply_mask_to_image(self.fa_path,
                                                    self.B0_mask,
                                                    self.fa_path)

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
        time.sleep(0.5)
        self.dwi2t1w_xfm = regutils.invert_xfm(self.t1w2dwi_xfm,
                                               self.dwi2t1w_xfm)
        time.sleep(0.5)
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
                time.sleep(0.5)
                self.t1w2dwi_bbr_xfm = regutils.invert_xfm(
                    self.dwi2t1w_bbr_xfm, self.t1w2dwi_bbr_xfm)
                time.sleep(0.5)
                # Apply the alignment
                regutils.align(
                    self.t1w_brain,
                    self.ap_path,
                    init=self.t1w2dwi_bbr_xfm,
                    xfm=self.t1wtissue2dwi_xfm,
                    bins=None,
                    interp="spline",
                    dof=6,
                    cost="mutualinfo",
                    out=self.t1w2dwi,
                    searchrad=True,
                    sch=None,
                )
                time.sleep(0.5)
            except BaseException:
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
                time.sleep(0.5)
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
            time.sleep(0.5)

        self.t1w2dwi = regutils.apply_mask_to_image(self.t1w2dwi,
                                                    self.B0_mask,
                                                    self.t1w2dwi)
        return

    def tissue2dwi_align(self):
        """
        A function to perform alignment of ventricle ROI's from MNI
        space --> dwi and CSF from T1w space --> dwi. First generates and
        performs dwi space alignment of avoidance/waypoint masks for
        tractography. First creates ventricle ROI. Then creates transforms
        from stock MNI template to dwi space. For this to succeed, must first
        have called both t1w2dwi_align.
        """
        import sys
        import time
        import os.path as op
        import pkg_resources
        from pynets.core.utils import load_runconfig
        from nilearn.image import resample_to_img
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.core.nodemaker import gen_img_list
        from nilearn.image import math_img

        hardcoded_params = load_runconfig()
        tiss_class = hardcoded_params['tracking']["tissue_classifier"][0]

        fa_template_path = pkg_resources.resource_filename(
            "pynets", f"templates/standard/FA_{self.vox_size}.nii.gz"
        )

        if sys.platform.startswith('win') is False:
            try:
                fa_template_img = nib.load(fa_template_path)
            except indexed_gzip.ZranError as e:
                print(e,
                      f"\nCannot load FA template. Do you have git-lfs "
                      f"installed?")
        else:
            try:
                fa_template_img = nib.load(fa_template_path)
            except ImportError as e:
                print(e, f"\nCannot load FA template. Do you have git-lfs ")

        mni_template_img = nib.load(self.input_mni_brain)

        if not np.allclose(fa_template_img.affine, mni_template_img.affine) \
            or not \
            np.allclose(fa_template_img.shape, mni_template_img.shape):
            fa_template_img_res = resample_to_img(fa_template_img,
                                                  mni_template_img)
            nib.save(
                fa_template_img_res,
                self.fa_template_res)
        else:
            self.fa_template_res = fname_presuffix(
                fa_template_path, suffix="_tmp",
                newpath=op.dirname(self.reg_path_img)
            )
            copyfile(
                fa_template_path,
                self.fa_template_res,
                copy=True,
                use_hardlink=False)

        # Register Lateral Ventricles and Corpus Callosum rois to t1w
        resample_to_img(nib.load(self.mni_atlas),
                        nib.load(
                            self.input_mni_brain),
                        interpolation='nearest').to_filename(self.mni_roi_ref)

        roi_parcels = [i for j, i in enumerate(gen_img_list(self.mni_roi_ref))]

        ventricle_roi = math_img("img1 + img2", img1=roi_parcels[2],
                                 img2=roi_parcels[13])

        self.mni_vent_loc = fname_presuffix(
            self.mni_vent_loc, suffix="_tmp",
            newpath=op.dirname(self.reg_path_img)
        )
        ventricle_roi.to_filename(self.mni_vent_loc)
        del roi_parcels, ventricle_roi

        # Create transform from the HarvardOxford atlas in MNI to T1w.
        # This will be used to transform the ventricles to dwi space.
        regutils.align(
            self.mni_roi_ref,
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

        # Create transform to align roi to mni and T1w using flirt
        regutils.applyxfm(
            self.input_mni_brain,
            self.mni_vent_loc,
            self.xfm_roi2mni_init,
            self.vent_mask_mni,
        )
        time.sleep(0.5)
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
            time.sleep(0.5)

            if sys.platform.startswith('win') is False:
                try:
                    nib.load(self.corpuscallosum)
                except indexed_gzip.ZranError as e:
                    print(e,
                          f"\nCannot load Corpus Callosum ROI. "
                          f"Do you have git-lfs installed?")
            else:
                try:
                    nib.load(self.corpuscallosum)
                except ImportError as e:
                    print(e, f"\nCannot load Corpus Callosum ROI. "
                          f"Do you have git-lfs installed?")

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
            time.sleep(0.5)
            regutils.applyxfm(
                self.corpuscallosum,
                self.t1w_brain,
                self.mni2t1_xfm,
                self.corpuscallosum_mask_t1w,
            )
            time.sleep(0.5)

        # Applyxfm to map FA template image to T1w space
        regutils.applyxfm(
            self.t1w_brain,
            self.fa_template_res,
            self.mni2t1_xfm,
            self.fa_template_t1w)
        time.sleep(0.5)

        # Applyxfm tissue maps to dwi space
        if self.t1w_brain_mask is not None:
            regutils.applyxfm(
                self.ap_path,
                self.t1w_brain_mask,
                self.t1wtissue2dwi_xfm,
                self.t1w_brain_mask_in_dwi,
            )
            time.sleep(0.5)
        regutils.applyxfm(
            self.ap_path,
            self.vent_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.vent_mask_dwi)
        time.sleep(0.5)
        regutils.applyxfm(
            self.ap_path,
            self.csf_mask,
            self.t1wtissue2dwi_xfm,
            self.csf_mask_dwi)
        time.sleep(0.5)
        regutils.applyxfm(
            self.ap_path, self.gm_mask, self.t1wtissue2dwi_xfm, self.gm_in_dwi
        )
        time.sleep(0.5)
        regutils.applyxfm(
            self.ap_path, self.wm_mask, self.t1wtissue2dwi_xfm, self.wm_in_dwi
        )
        time.sleep(0.5)

        regutils.applyxfm(
            self.ap_path,
            self.corpuscallosum_mask_t1w,
            self.t1wtissue2dwi_xfm,
            self.corpuscallosum_dwi,
        )
        time.sleep(0.5)

        if tiss_class == 'wb' or tiss_class == 'cmc':
            csf_thr = 0.50
            wm_thr = 0.15
            gm_thr = 0.10
        else:
            csf_thr = 0.99
            wm_thr = 0.10
            gm_thr = 0.075

        # Threshold WM to binary in dwi space
        nib.save(math_img(f"img > {wm_thr}",
                          img=nib.load(self.wm_in_dwi)),
                 self.wm_in_dwi_bin)

        # Threshold GM to binary in dwi space
        nib.save(math_img(f"img > {gm_thr}",
                          img=nib.load(self.gm_in_dwi)),
                 self.gm_in_dwi_bin)

        # Threshold CSF to binary in dwi space
        nib.save(math_img(f"img > {csf_thr}",
                          img=nib.load(self.csf_mask_dwi)),
                 self.csf_mask_dwi_bin)

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
        math_img("(img1 + img2) > 0.0001",
                 img1=nib.load(self.csf_mask_dwi),
                 img2=nib.load(self.vent_mask_dwi)).to_filename(
            self.vent_csf_in_dwi)

        print("Creating Corpus Callosum mask...")
        math_img("(img1*img2 - img3) > 0.0001",
                 img1=nib.load(self.corpuscallosum_dwi),
                 img2=nib.load(self.wm_in_dwi_bin),
                 img3=nib.load(self.vent_csf_in_dwi)).to_filename(
            self.corpuscallosum_dwi)

        # Create GM-WM interface image
        math_img("((img1*img2 + img3)*img4) > 0.0001",
                 img1=nib.load(self.gm_in_dwi_bin),
                 img2=nib.load(self.wm_in_dwi_bin),
                 img3=nib.load(self.corpuscallosum_dwi),
                 img4=nib.load(self.B0_mask)).to_filename(
            self.wm_gm_int_in_dwi)

        return


class FmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in
    native epi space.

    References
    ----------
    .. [1] Brett M, Leff AP, Rorden C, Ashburner J (2001) Spatial Normalization
      of Brain Images with Focal Lesions Using Cost Function Masking.
      NeuroImage 14(2) doi:10.006/nimg.2001.0845.
    .. [2] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a
      hidden Markov random field model and the expectation-maximization
      algorithm. IEEE Trans Med Imaging. 2001 Jan;20(1):45–57.
      doi:10.1109/42.906424.
    """
    __slots__ = ('t1w', 'vox_size', 'template_name', 't1w_name', 'simple',
                 'basedir_path', 'basedir_path', 'reg_path', 'reg_path_mat',
                 'reg_path_warp', 'reg_path_img', 't1w2epi_xfm',
                 't12mni_xfm_init', 't12mni_xfm', 'mni2t1_xfm',
                 'mni2t1w_warp', 'warp_t1w2mni', 't1_aligned_mni',
                 't1w_brain', 't1w_head', 't1w_brain_mask', 'map_name',
                 'gm_mask', 'gm_mask_thr', 'wm_mask', 'wm_mask_thr',
                 'wm_edge', 'input_mni', 'input_mni_brain', 'input_mni_mask')

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
        self.t1w_brain = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                         f"{'_brain.nii.gz'}"
        self.t1w_head = f"{self.reg_path_img}{'/'}{self.t1w_name}" \
                        f"{'_head.nii.gz'}"
        self.t1w_brain_mask = (
            f"{self.reg_path_img}{'/'}{self.t1w_name}"
            f"{'_brain_mask.nii.gz'}"
        )
        self.map_name = f"{self.t1w_name}{'_seg'}"
        self.gm_mask = f"{self.reg_path_img}{'/'}" \
                       f"{self.t1w_name}{'_gm.nii.gz'}"
        self.gm_mask_thr = f"{self.reg_path_img}{'/'}" \
                           f"{self.t1w_name}{'_gm_thr.nii.gz'}"
        self.wm_mask = f"{self.reg_path_img}{'/'}" \
                       f"{self.t1w_name}{'_wm.nii.gz'}"
        self.wm_mask_thr = f"{self.reg_path_img}{'/'}" \
                           f"{self.t1w_name}{'_wm_thr.nii.gz'}"
        self.wm_edge = f"{self.reg_path_img}{'/'}" \
                       f"{self.t1w_name}{'_wm_edge.nii.gz'}"
        self.input_mni = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
                      f"{vox_size}.nii.gz"
        )
        self.input_mni_brain = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
                      f"brain_{vox_size}.nii.gz"
        )
        self.input_mni_mask = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.template_name}_"
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
        import time

        # Segment the t1w brain into probability maps
        if (
            wm_mask_existing is not None
            and gm_mask_existing is not None
            and overwrite is False
        ):
            print("Existing segmentations detected...")
            gm_mask = regutils.orient_reslice(
                gm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            wm_mask = regutils.orient_reslice(
                wm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
        else:
            try:
                maps = regutils.segment_t1w(self.t1w_brain, self.map_name)
                gm_mask = maps["gm_prob"]
                wm_mask = maps["wm_prob"]
            except RuntimeError as e:
                import sys
                print(e,
                      "Segmentation failed. Does the input anatomical image "
                      "still contained skull?"
                      )

        # Threshold GM to binary in func space
        math_img("img > 0.01", img=nib.load(gm_mask)).to_filename(
            self.gm_mask_thr)
        self.gm_mask = regutils.apply_mask_to_image(gm_mask,
                                                    self.gm_mask_thr,
                                                    self.gm_mask)

        # Threshold WM to binary in dwi space
        math_img("img > 0.50", img=nib.load(wm_mask)).to_filename(
            self.wm_mask_thr)

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
        import time

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
        time.sleep(0.5)
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
                time.sleep(0.5)
                # Get warp from T1w --> MNI
                regutils.inverse_warp(
                    self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni
                )
                time.sleep(0.5)
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
                time.sleep(0.5)
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
            time.sleep(0.5)
            # Get mat from MNI -> T1w
            self.t12mni_xfm = regutils.invert_xfm(self.mni2t1_xfm,
                                                  self.t12mni_xfm)
        return
