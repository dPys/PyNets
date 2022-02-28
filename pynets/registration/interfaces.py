#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import matplotlib
import warnings
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    Directory,
)

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


class _RegisterDWIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterDWI"""

    fa_path = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    in_dir = traits.Any()
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    force_create_mask = traits.Bool(True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)
    overwrite = traits.Bool(False, usedefault=True)


class _RegisterDWIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterDWI"""

    wm_in_dwi = File(exists=True, mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    csf_mask_dwi = File(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    basedir_path = Directory(exists=True, mandatory=True)
    t1w2dwi = File(exists=True, mandatory=True)
    t1w_brain_mask_in_dwi = traits.Any(mandatory=False)
    t1_aligned_mni = traits.Any(mandatory=False)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    t1wtissue2dwi_xfm = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    t1w_brain_mask = File(exists=True, mandatory=True)
    t1w2dwi_bbr_xfm = File(exists=True, mandatory=True)
    t1w2dwi_xfm = File(exists=True, mandatory=True)
    wm_gm_int_in_dwi = File(exists=True, mandatory=True)


class RegisterDWI(SimpleInterface):
    """Interface wrapper for RegisterDWI to create T1w->MNI->DWI mappings."""

    input_spec = _RegisterDWIInputSpec
    output_spec = _RegisterDWIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import time
        import glob
        import os
        import os.path as op
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.registration.utils import orient_reslice

        fa_tmp_path = fname_presuffix(
            self.inputs.fa_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.fa_path,
            fa_tmp_path,
            copy=True,
            use_hardlink=False)

        ap_tmp_path = fname_presuffix(
            self.inputs.ap_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.ap_path,
            ap_tmp_path,
            copy=True,
            use_hardlink=False)

        B0_mask_tmp_path = fname_presuffix(
            self.inputs.B0_mask, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.B0_mask,
            B0_mask_tmp_path,
            copy=True,
            use_hardlink=False)

        anat_mask_existing = [
            i
            for i in glob.glob(f"{self.inputs.in_dir}"
                               f"/*_desc-brain_mask.nii.gz")
            if "MNI" not in i
        ] + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_brain_mask.nii.gz")

        # Copy T1w mask, if provided, else use existing, if detected, else
        # compute a fresh one
        if self.inputs.mask:
            mask_tmp_path = fname_presuffix(
                self.inputs.mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mask,
                mask_tmp_path,
                copy=True,
                use_hardlink=False)
        else:
            if len(anat_mask_existing) > 0 and \
                    self.inputs.mask is None and \
                    op.isfile(anat_mask_existing[0]) and \
                    self.inputs.force_create_mask is False:
                mask_tmp_path = fname_presuffix(
                    anat_mask_existing[0], suffix="_tmp", newpath=runtime.cwd
                )
                copyfile(
                    anat_mask_existing[0],
                    mask_tmp_path,
                    copy=True,
                    use_hardlink=False)
                mask_tmp_path = orient_reslice(
                    mask_tmp_path, runtime.cwd, self.inputs.vox_size
                )
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(
            f"{self.inputs.in_dir}/*_label-GM_probseg.nii.gz") + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_seg_pve_1.nii.gz")
        if len(gm_mask_existing) > 0:
            gm_mask = fname_presuffix(gm_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                gm_mask_existing[0],
                gm_mask,
                copy=True,
                use_hardlink=False)
            gm_mask = orient_reslice(
                gm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            gm_mask = None

        wm_mask_existing = glob.glob(
            f"{self.inputs.in_dir}/*_label-WM_probseg.nii.gz") + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_seg_pve_0.nii.gz")
        if len(wm_mask_existing) > 0:
            wm_mask = fname_presuffix(wm_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                wm_mask_existing[0],
                wm_mask,
                copy=True,
                use_hardlink=False)
            wm_mask = orient_reslice(
                wm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            wm_mask = None

        csf_mask_existing = glob.glob(
            f"{self.inputs.in_dir}/*_label-CSF_probseg.nii.gz"
        ) + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_seg_pve_2.nii.gz")
        if len(csf_mask_existing) > 0:
            csf_mask = fname_presuffix(
                csf_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                csf_mask_existing[0],
                csf_mask,
                copy=True,
                use_hardlink=False)
            csf_mask = orient_reslice(
                csf_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            csf_mask = None

        anat_file_tmp_path = fname_presuffix(
            self.inputs.anat_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.anat_file,
            anat_file_tmp_path,
            copy=True,
            use_hardlink=False)

        reg = register.DmriReg(
            basedir_path=runtime.cwd,
            fa_path=fa_tmp_path,
            ap_path=ap_tmp_path,
            B0_mask=B0_mask_tmp_path,
            anat_file=anat_file_tmp_path,
            vox_size=self.inputs.vox_size,
            template_name=self.inputs.template_name,
            simple=self.inputs.simple,
        )

        # Generate T1w brain mask
        reg.gen_mask(mask_tmp_path)

        # Perform anatomical segmentation
        reg.gen_tissue(wm_mask, gm_mask, csf_mask, self.inputs.overwrite)

        # Align t1w to mni template
        # from joblib import Memory
        # import os
        # location = f"{outdir}/joblib_" \
        #            f"{self.inputs.anat_file.split('.nii')[0]}"
        # os.makedirs(location, exist_ok=True)
        # memory = Memory(location)
        # t1w2mni_align = memory.cache(reg.t1w2mni_align)
        # t1w2mni_align()
        reg.t1w2mni_align()
        time.sleep(0.5)

        if (self.inputs.overwrite is True) or (
                op.isfile(reg.t1w2dwi) is False):
            # Align t1w to dwi
            reg.t1w2dwi_align()
            time.sleep(0.5)

        if (self.inputs.overwrite is True) or (
            op.isfile(reg.wm_gm_int_in_dwi) is False
        ):
            # Align tissue
            reg.tissue2dwi_align()
            time.sleep(0.5)

        self._results["wm_in_dwi"] = reg.wm_in_dwi
        self._results["gm_in_dwi"] = reg.gm_in_dwi
        self._results["vent_csf_in_dwi"] = reg.vent_csf_in_dwi
        self._results["csf_mask_dwi"] = reg.csf_mask_dwi
        self._results["anat_file"] = anat_file_tmp_path
        self._results["t1w2dwi"] = reg.t1w2dwi
        self._results["B0_mask"] = B0_mask_tmp_path
        self._results["ap_path"] = ap_tmp_path
        self._results["gtab_file"] = self.inputs.gtab_file
        self._results["dwi_file"] = self.inputs.dwi_file
        self._results["basedir_path"] = runtime.cwd
        self._results["t1w_brain_mask_in_dwi"] = reg.t1w_brain_mask_in_dwi
        self._results["t1_aligned_mni"] = reg.t1_aligned_mni
        self._results["t1w_brain"] = reg.t1w_brain
        self._results["mni2t1w_warp"] = reg.mni2t1w_warp
        self._results["t1wtissue2dwi_xfm"] = reg.t1wtissue2dwi_xfm
        self._results["mni2t1_xfm"] = reg.mni2t1_xfm
        self._results["t1w_brain_mask"] = reg.t1w_brain_mask
        self._results["t1w2dwi_bbr_xfm"] = reg.t1w2dwi_bbr_xfm
        self._results["t1w2dwi_xfm"] = reg.t1w2dwi_xfm
        self._results["wm_gm_int_in_dwi"] = reg.wm_gm_int_in_dwi

        reg_tmp = [
            fa_tmp_path,
            mask_tmp_path,
            reg.warp_t1w2mni,
            reg.t1w_head,
            reg.wm_edge,
            reg.vent_mask_dwi,
            reg.vent_mask_t1w,
            reg.corpuscallosum_mask_t1w,
            reg.corpuscallosum_dwi
        ]
        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        gc.collect()

        return runtime


class _RegisterAtlasDWIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterAtlasDWI"""

    atlas = traits.Any(mandatory=False)
    subnet = traits.Any(mandatory=True)
    parcellation4d = traits.Any(mandatory=True)
    parcellation = traits.Any(mandatory=False)
    basedir_path = Directory(exists=True, mandatory=True)
    node_radius = traits.Any(mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    wm_in_dwi = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    mask = traits.Any(mandatory=False)
    waymask = traits.Any(mandatory=False)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    t1wtissue2dwi_xfm = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    t1w_brain_mask = File(exists=True, mandatory=True)
    t1_aligned_mni = File(exists=True, mandatory=True)
    t1w2dwi_bbr_xfm = File(exists=True, mandatory=True)
    t1w2dwi_xfm = File(exists=True, mandatory=True)
    wm_gm_int_in_dwi = File(exists=True, mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterAtlasDWIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterAtlasDWI"""

    dwi_aligned_atlas_wmgm_int = File(exists=True, mandatory=True)
    dwi_aligned_atlas = File(exists=True, mandatory=True)
    aligned_atlas_t1w = File(exists=True, mandatory=True)
    node_radius = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=False)
    parcellation4d = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    wm_in_dwi = File(exists=True, mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    waymask_in_dwi = traits.Any(mandatory=False)


class RegisterAtlasDWI(SimpleInterface):
    """Interface wrapper for RegisterAtlasDWI."""

    input_spec = _RegisterAtlasDWIInputSpec
    output_spec = _RegisterAtlasDWIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import time
        import os
        import os.path as op
        from pynets.registration import utils as regutils
        from pynets.core.nodemaker import \
            drop_coords_labels_from_restricted_parcellation
        from nipype.utils.filemanip import fname_presuffix, copyfile
        import pkg_resources
        from pynets.core import utils

        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.inputs.template_name}_brain_"
                      f"{self.inputs.vox_size}.nii.gz"
        )

        template_tmp_path = fname_presuffix(
            template, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            template,
            template_tmp_path,
            copy=True,
            use_hardlink=False)

        if self.inputs.parcellation is None:
            parcellation_tmp_path = None
        else:
            parcellation_tmp_path = fname_presuffix(
                self.inputs.parcellation, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.parcellation,
                parcellation_tmp_path,
                copy=True,
                use_hardlink=False)

        if self.inputs.parcellation4d is None:
            parcellation4d_tmp_path = None
        else:
            parcellation4d_tmp_path = fname_presuffix(
                self.inputs.parcellation4d, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.parcellation4d,
                parcellation4d_tmp_path,
                copy=True,
                use_hardlink=False,
            )

        fa_tmp_path = fname_presuffix(
            self.inputs.fa_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.fa_path,
            fa_tmp_path,
            copy=True,
            use_hardlink=False)

        ap_tmp_path = fname_presuffix(
            self.inputs.ap_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.ap_path,
            ap_tmp_path,
            copy=True,
            use_hardlink=False)

        B0_mask_tmp_path = fname_presuffix(
            self.inputs.B0_mask, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.B0_mask,
            B0_mask_tmp_path,
            copy=True,
            use_hardlink=False)

        t1w_brain_tmp_path = fname_presuffix(
            self.inputs.t1w_brain, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w_brain,
            t1w_brain_tmp_path,
            copy=True,
            use_hardlink=False)

        mni2t1w_warp_tmp_path = fname_presuffix(
            self.inputs.mni2t1w_warp, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1w_warp,
            mni2t1w_warp_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        t1wtissue2dwi_xfm_tmp_path = fname_presuffix(
            self.inputs.t1wtissue2dwi_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1wtissue2dwi_xfm,
            t1wtissue2dwi_xfm_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        mni2t1_xfm_tmp_path = fname_presuffix(
            self.inputs.mni2t1_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1_xfm,
            mni2t1_xfm_tmp_path,
            copy=True,
            use_hardlink=False)

        t1w_brain_mask_tmp_path = fname_presuffix(
            self.inputs.t1w_brain_mask, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w_brain_mask,
            t1w_brain_mask_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        t1_aligned_mni_tmp_path = fname_presuffix(
            self.inputs.t1_aligned_mni, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1_aligned_mni,
            t1_aligned_mni_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        t1w2dwi_bbr_xfm_tmp_path = fname_presuffix(
            self.inputs.t1w2dwi_bbr_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w2dwi_bbr_xfm,
            t1w2dwi_bbr_xfm_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        t1w2dwi_xfm_tmp_path = fname_presuffix(
            self.inputs.t1w2dwi_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w2dwi_xfm,
            t1w2dwi_xfm_tmp_path,
            copy=True,
            use_hardlink=False)

        wm_gm_int_in_dwi_tmp_path = fname_presuffix(
            self.inputs.wm_gm_int_in_dwi, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.wm_gm_int_in_dwi,
            wm_gm_int_in_dwi_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        if self.inputs.subnet or self.inputs.waymask:
            if self.inputs.waymask:
                wm_suf = os.path.basename(self.inputs.waymask).split('.nii')[0]
                atlas_name = f"{self.inputs.atlas}_" \
                             f"{wm_suf}"
            else:
                atlas_name = f"{self.inputs.atlas}_{self.inputs.subnet}"
        else:
            atlas_name = f"{self.inputs.atlas}"

        base_dir_tmp = f"{runtime.cwd}/atlas_{atlas_name}"
        os.makedirs(base_dir_tmp, exist_ok=True)

        mni2dwi_xfm = f"{base_dir_tmp}{'/'}{atlas_name}" \
            f"{'_mni2dwi_xfm.mat'}"

        aligned_atlas_t1mni = f"{base_dir_tmp}{'/'}{atlas_name}" \
                              f"{'_t1w_mni.nii.gz'}"
        aligned_atlas_skull = f"{base_dir_tmp}{'/'}{atlas_name}" \
                              f"{'_t1w_skull.nii.gz'}"
        dwi_aligned_atlas = f"{base_dir_tmp}{'/'}{atlas_name}" \
                            f"{'_dwi_track.nii.gz'}"
        dwi_aligned_atlas_wmgm_int = (
            f"{base_dir_tmp}{'/'}{atlas_name}{'_dwi_track_wmgm_int.nii.gz'}"
        )

        if self.inputs.node_radius is not None:
            atlas_name = f"{atlas_name}{'_'}{self.inputs.node_radius}"

        # Apply warps/coregister atlas to dwi
        [
            dwi_aligned_atlas_wmgm_int,
            dwi_aligned_atlas,
            aligned_atlas_t1w,
        ] = regutils.atlas2t1w2dwi_align(
            parcellation_tmp_path,
            parcellation4d_tmp_path,
            atlas_name,
            t1w_brain_tmp_path,
            t1w_brain_mask_tmp_path,
            mni2t1w_warp_tmp_path,
            t1_aligned_mni_tmp_path,
            ap_tmp_path,
            mni2t1_xfm_tmp_path,
            t1wtissue2dwi_xfm_tmp_path,
            wm_gm_int_in_dwi_tmp_path,
            aligned_atlas_t1mni,
            aligned_atlas_skull,
            dwi_aligned_atlas,
            dwi_aligned_atlas_wmgm_int,
            B0_mask_tmp_path,
            mni2dwi_xfm,
            self.inputs.simple,
        )

        # Correct coords and labels
        [dwi_aligned_atlas, coords, labels] = \
            drop_coords_labels_from_restricted_parcellation(
            dwi_aligned_atlas, self.inputs.coords, self.inputs.labels)

        if self.inputs.waymask:
            waymask_tmp_path = fname_presuffix(
                self.inputs.waymask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.waymask,
                waymask_tmp_path,
                copy=True,
                use_hardlink=False)

            # Align waymask
            waymask_in_t1w = (
                f"{runtime.cwd}/waymask-"
                f"{os.path.basename(self.inputs.waymask).split('.nii')[0]}_"
                f"in_t1w.nii.gz")
            waymask_in_dwi = (
                f"{runtime.cwd}/waymask-"
                f"{os.path.basename(self.inputs.waymask).split('.nii')[0]}_"
                f"in_dwi.nii.gz")

            waymask_in_dwi = regutils.waymask2dwi_align(
                waymask_tmp_path,
                t1w_brain_tmp_path,
                ap_tmp_path,
                mni2t1w_warp_tmp_path,
                mni2t1_xfm_tmp_path,
                t1wtissue2dwi_xfm_tmp_path,
                waymask_in_t1w,
                waymask_in_dwi,
                B0_mask_tmp_path,
                template_tmp_path,
                self.inputs.simple,
            )
            time.sleep(0.5)
            os.system(f"rm -f {waymask_tmp_path} &")
        else:
            waymask_in_dwi = None

        if self.inputs.parcellation is None:
            parcellation_out = self.inputs.parcellation4d
            copyfile(
                dwi_aligned_atlas,
                f"{os.path.dirname(parcellation_out)}/"
                f"{os.path.basename(dwi_aligned_atlas)}",
                copy=True,
                use_hardlink=False,
            )
        else:
            parcellation_out = self.inputs.parcellation
            copyfile(
                dwi_aligned_atlas,
                f"{os.path.dirname(parcellation_out)}/parcellations/"
                f"{os.path.basename(dwi_aligned_atlas)}",
                copy=True,
                use_hardlink=False,
            )

        # wm_img = nib.load(self.inputs.wm_in_dwi)
        # wm_data = wm_img.get_fdata().astype('bool')
        # atlas_img = nib.load(dwi_aligned_atlas)
        # atlas_data = atlas_img.get_fdata().astype('bool')
        # B0_mask_img = nib.load(B0_mask_tmp_path)
        # B0_mask_data = B0_mask_img.get_fdata().astype('bool')
        # wm_data_dil = binary_opening(wm_data, structure=atlas_data,
        #                              mask=B0_mask_data)
        # wm_in_dwi_tmp_path = fname_presuffix(
        #     self.inputs.wm_in_dwi, suffix="_tmp", newpath=runtime.cwd
        # )
        # nib.save(nib.Nifti1Image(wm_data_dil, affine=wm_img.affine,
        #                          header=wm_img.header), wm_in_dwi_tmp_path)
        #
        # wm_img.uncache()
        # atlas_img.uncache()
        # B0_mask_img.uncache()
        # del atlas_data, wm_data, B0_mask_data, wm_data_dil
        # self._results["wm_in_dwi"] = wm_in_dwi_tmp_path

        self._results["wm_in_dwi"] = self.inputs.wm_in_dwi
        self._results["dwi_aligned_atlas_wmgm_int"] = \
            dwi_aligned_atlas_wmgm_int
        self._results["dwi_aligned_atlas"] = dwi_aligned_atlas
        self._results["aligned_atlas_t1w"] = aligned_atlas_t1w
        self._results["node_radius"] = self.inputs.node_radius
        self._results["atlas"] = self.inputs.atlas
        self._results["parcellation4d"] = parcellation4d_tmp_path
        self._results["parcellation"] = parcellation_out
        self._results["coords"] = coords
        self._results["labels"] = labels
        self._results["gm_in_dwi"] = self.inputs.gm_in_dwi
        self._results["vent_csf_in_dwi"] = self.inputs.vent_csf_in_dwi
        self._results["B0_mask"] = B0_mask_tmp_path
        self._results["ap_path"] = ap_tmp_path
        self._results["gtab_file"] = self.inputs.gtab_file
        self._results["dwi_file"] = self.inputs.dwi_file
        self._results["waymask_in_dwi"] = waymask_in_dwi

        dir_path = utils.do_dir_path(
            self.inputs.atlas, os.path.dirname(self.inputs.dwi_file)
        )

        namer_dir = "{}/tractography".format(dir_path)
        if not op.isdir(namer_dir):
            os.mkdir(namer_dir)

        if not os.path.isfile(f"{namer_dir}/"
                              f"{op.basename(self.inputs.fa_path)}"):
            copyfile(
                self.inputs.fa_path,
                f"{namer_dir}/{op.basename(self.inputs.fa_path)}",
                copy=True,
                use_hardlink=False,
            )
        if not os.path.isfile(f"{namer_dir}/"
                              f"{op.basename(self.inputs.ap_path)}"):
            copyfile(
                self.inputs.ap_path,
                f"{namer_dir}/"
                f"{op.basename(self.inputs.ap_path).replace('_tmp', '')}",
                copy=True,
                use_hardlink=False,
            )
        if not os.path.isfile(f"{namer_dir}/"
                              f"{op.basename(self.inputs.B0_mask)}"):
            copyfile(
                self.inputs.B0_mask,
                f"{namer_dir}/"
                f"{op.basename(self.inputs.B0_mask).replace('_tmp', '')}",
                copy=True,
                use_hardlink=False,
            )
        if not os.path.isfile(f"{namer_dir}/"
                              f"{op.basename(self.inputs.gtab_file)}"):
            copyfile(
                self.inputs.gtab_file,
                f"{namer_dir}/{op.basename(self.inputs.gtab_file)}",
                copy=True,
                use_hardlink=False,
            )
        if not os.path.isfile(f"{namer_dir}/{op.basename(aligned_atlas_t1w)}"):
            copyfile(
                aligned_atlas_t1w,
                f"{namer_dir}/{op.basename(aligned_atlas_t1w)}",
                copy=True,
                use_hardlink=False,
            )

        reg_tmp = [
            parcellation_tmp_path,
            mni2t1w_warp_tmp_path,
            mni2t1_xfm_tmp_path,
            t1w_brain_mask_tmp_path,
            t1_aligned_mni_tmp_path,
            t1w2dwi_bbr_xfm_tmp_path,
            t1w2dwi_xfm_tmp_path,
            t1w_brain_tmp_path
        ]
        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        gc.collect()

        return runtime


class _RegisterROIDWIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterROIDWI"""

    dwi_file = File(exists=True, mandatory=True)
    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    roi = traits.Any(mandatory=False)
    fa_path = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    wm_in_dwi = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    t1wtissue2dwi_xfm = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    simple = traits.Bool(False, usedefault=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)


class _RegisterROIDWIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterROIDWI"""

    roi = traits.Any(mandatory=False)


class RegisterROIDWI(SimpleInterface):
    """Interface wrapper for RegisterROIDWI."""

    input_spec = _RegisterROIDWIInputSpec
    output_spec = _RegisterROIDWIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import time
        from pynets.registration import utils as regutils
        from nipype.utils.filemanip import fname_presuffix, copyfile
        import pkg_resources

        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.inputs.template_name}_brain_"
                      f"{self.inputs.vox_size}.nii.gz"
        )

        template_tmp_path = fname_presuffix(
            template, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            template,
            template_tmp_path,
            copy=True,
            use_hardlink=False)

        ap_tmp_path = fname_presuffix(
            self.inputs.ap_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.ap_path,
            ap_tmp_path,
            copy=True,
            use_hardlink=False)

        roi_file_tmp_path = fname_presuffix(
            self.inputs.roi, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.roi,
            roi_file_tmp_path,
            copy=True,
            use_hardlink=False)

        t1w_brain_tmp_path = fname_presuffix(
            self.inputs.t1w_brain, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w_brain,
            t1w_brain_tmp_path,
            copy=True,
            use_hardlink=False)

        mni2t1w_warp_tmp_path = fname_presuffix(
            self.inputs.mni2t1w_warp, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1w_warp,
            mni2t1w_warp_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        t1wtissue2dwi_xfm_tmp_path = fname_presuffix(
            self.inputs.t1wtissue2dwi_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1wtissue2dwi_xfm,
            t1wtissue2dwi_xfm_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        mni2t1_xfm_tmp_path = fname_presuffix(
            self.inputs.mni2t1_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1_xfm,
            mni2t1_xfm_tmp_path,
            copy=True,
            use_hardlink=False)

        roi_in_t1w = f"{runtime.cwd}/waymask-" \
                     f"{os.path.basename(self.inputs.roi).split('.nii')[0]}" \
                     f"_in_t1w.nii.gz"
        roi_in_dwi = f"{runtime.cwd}/waymask-" \
                     f"{os.path.basename(self.inputs.roi).split('.nii')[0]}" \
                     f"_in_dwi.nii.gz"

        if self.inputs.roi:
            t1w_brain_tmp_path2 = fname_presuffix(
                self.inputs.t1w_brain, suffix="2", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.t1w_brain,
                t1w_brain_tmp_path2,
                copy=True,
                use_hardlink=False)

            mni2t1w_warp_tmp_path2 = fname_presuffix(
                self.inputs.mni2t1w_warp, suffix="2", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mni2t1w_warp,
                mni2t1w_warp_tmp_path2,
                copy=True,
                use_hardlink=False)

            # Align roi
            roi_in_dwi = regutils.roi2dwi_align(
                roi_file_tmp_path,
                t1w_brain_tmp_path2,
                roi_in_t1w,
                roi_in_dwi,
                mni2t1w_warp_tmp_path2,
                t1wtissue2dwi_xfm_tmp_path,
                mni2t1_xfm_tmp_path,
                template_tmp_path,
                self.inputs.simple,
            )
            time.sleep(0.5)
        else:
            roi_in_dwi = None

        self._results["roi"] = roi_in_dwi

        reg_tmp = [
            t1w_brain_tmp_path,
            mni2t1w_warp_tmp_path,
            t1wtissue2dwi_xfm_tmp_path,
            mni2t1_xfm_tmp_path,
            template_tmp_path,
            roi_in_t1w,
            roi_file_tmp_path
        ]
        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        gc.collect()

        return runtime


class _RegisterFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterFunc"""

    anat_file = File(exists=True, mandatory=True)
    mask = traits.Any(mandatory=False)
    in_dir = traits.Any(mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    force_create_mask = traits.Bool(True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)
    overwrite = traits.Bool(False, usedefault=True)


class _RegisterFuncOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterFunc"""

    reg_fmri_complete = traits.Bool()
    basedir_path = Directory(exists=True, mandatory=True)
    t1w_brain_mask = traits.Any(mandatory=False)
    epi_brain_path = traits.Any()
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    t1_aligned_mni = File(exists=True, mandatory=True)
    gm_mask = File(exists=True, mandatory=True)
    t1w2mni_xfm = traits.Any()
    t1w2mni_warp = traits.Any()


class RegisterFunc(SimpleInterface):
    """Interface wrapper for RegisterFunc to create Func->T1w->MNI mappings."""

    input_spec = _RegisterFuncInputSpec
    output_spec = _RegisterFuncOutputSpec

    def _run_interface(self, runtime):
        import gc
        import glob
        import time
        import os.path as op
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.registration.utils import orient_reslice

        anat_mask_existing = [
            i
            for i in glob.glob(self.inputs.in_dir +
                               "/*_desc-brain_mask.nii.gz")
            if "MNI" not in i
        ] + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/"
            f"t1w_brain_mask.nii.gz")

        # Copy T1w mask, if provided, else use existing, if detected, else
        # compute a fresh one
        if self.inputs.mask:
            mask_tmp_path = fname_presuffix(
                self.inputs.mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mask,
                mask_tmp_path,
                copy=True,
                use_hardlink=False)
        else:
            if len(anat_mask_existing) > 0 and \
                self.inputs.mask is None and \
                op.isfile(anat_mask_existing[0]) \
                    and self.inputs.force_create_mask is False:
                mask_tmp_path = fname_presuffix(
                    anat_mask_existing[0], suffix="_tmp",
                    newpath=runtime.cwd
                )
                copyfile(
                    anat_mask_existing[0],
                    mask_tmp_path,
                    copy=True,
                    use_hardlink=False)
                mask_tmp_path = orient_reslice(
                    mask_tmp_path, runtime.cwd, self.inputs.vox_size
                )
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-GM_probseg.nii.gz") + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_seg_pve_1.nii.gz")
        if len(gm_mask_existing) > 0:
            gm_mask = fname_presuffix(gm_mask_existing[0],
                                      newpath=runtime.cwd)
            copyfile(
                gm_mask_existing[0],
                gm_mask,
                copy=True,
                use_hardlink=False)
            gm_mask = orient_reslice(
                gm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            gm_mask = None

        wm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-WM_probseg.nii.gz") + glob.glob(
            f"{runtime.cwd}/../../*/register_node/*/imgs/t1w_seg_pve_0.nii.gz")
        if len(wm_mask_existing) > 0:
            wm_mask = fname_presuffix(wm_mask_existing[0],
                                      newpath=runtime.cwd)
            copyfile(
                wm_mask_existing[0],
                wm_mask,
                copy=True,
                use_hardlink=False)
            wm_mask = orient_reslice(
                wm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            wm_mask = None

        anat_file_tmp_path = fname_presuffix(
            self.inputs.anat_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.anat_file,
            anat_file_tmp_path,
            copy=True,
            use_hardlink=False)

        reg = register.FmriReg(
            basedir_path=runtime.cwd,
            anat_file=anat_file_tmp_path,
            vox_size=self.inputs.vox_size,
            template_name=self.inputs.template_name,
            simple=self.inputs.simple,
        )

        # Generate T1w brain mask
        reg.gen_mask(mask_tmp_path)

        # Perform anatomical segmentation
        reg.gen_tissue(wm_mask, gm_mask, self.inputs.overwrite)

        # Align t1w to mni template
        # from joblib import Memory
        # import os
        # location = f"{outdir}/joblib_" \
        #            f"{self.inputs.anat_file.split('.nii')[0]}"
        # os.makedirs(location, exist_ok=True)
        # memory = Memory(location)
        # t1w2mni_align = memory.cache(reg.t1w2mni_align)
        # t1w2mni_align()
        reg.t1w2mni_align()
        time.sleep(0.5)

        self._results["reg_fmri_complete"] = True
        self._results["basedir_path"] = runtime.cwd
        self._results["t1w_brain_mask"] = reg.t1w_brain_mask
        self._results["t1w_brain"] = reg.t1w_brain
        self._results["mni2t1w_warp"] = reg.mni2t1w_warp
        self._results["mni2t1_xfm"] = reg.mni2t1_xfm
        self._results["t1_aligned_mni"] = reg.t1_aligned_mni
        self._results["gm_mask"] = reg.gm_mask
        self._results["t1w2mni_warp"] = reg.warp_t1w2mni
        self._results["t1w2mni_xfm"] = reg.t12mni_xfm

        gc.collect()

        return runtime


class _RegisterParcellation2MNIFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterParcellation2MNIFunc"""

    parcellation = traits.Any(mandatory=False)
    t1w2mni_xfm = File(exists=True, mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    t1w_brain = File(exists=True, mandatory=True)
    t1w2mni_warp = traits.Any(mandatory=True)
    dir_path = traits.Any(mandatory=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterParcellation2MNIFuncOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterParcellation2MNIFunc"""

    aligned_atlas_mni = File(exists=True, mandatory=True)


class RegisterParcellation2MNIFunc(SimpleInterface):
    """Interface wrapper for RegisterParcellation2MNIFunc."""

    input_spec = _RegisterParcellation2MNIFuncInputSpec
    output_spec = _RegisterParcellation2MNIFuncOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import pkg_resources
        import time
        from pynets.core.utils import prune_suffices
        from pynets.registration import utils as regutils
        from nipype.utils.filemanip import fname_presuffix, copyfile

        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.inputs.template_name}_brain_"
                      f"{self.inputs.vox_size}.nii.gz"
        )
        template_mask = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.inputs.template_name}_"
                      f"brain_mask_{self.inputs.vox_size}.nii.gz"
        )

        template_tmp_path = fname_presuffix(
            template, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            template,
            template_tmp_path,
            copy=True,
            use_hardlink=False)

        template_mask_tmp_path = fname_presuffix(
            template_mask, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            template_mask,
            template_mask_tmp_path,
            copy=True,
            use_hardlink=False)

        t1w_brain_tmp_path = fname_presuffix(
            self.inputs.t1w_brain, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w_brain,
            t1w_brain_tmp_path,
            copy=True,
            use_hardlink=False)

        parcellation_tmp_path = fname_presuffix(
            self.inputs.parcellation, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.parcellation,
            parcellation_tmp_path,
            copy=True,
            use_hardlink=False)

        atlas_name = prune_suffices(os.path.basename(self.inputs.parcellation
                                                     ).split('.nii')[0])

        base_dir_tmp = f"{runtime.cwd}/atlas_{atlas_name}"

        os.makedirs(base_dir_tmp, exist_ok=True)

        aligned_atlas_t1w = f"{base_dir_tmp}{'/'}{atlas_name}_" \
                            f"t1w_res.nii.gz"

        aligned_atlas_mni = f"{base_dir_tmp}{'/'}{atlas_name}_" \
                            f"t1w_mni.nii.gz"

        t1w2mni_xfm_tmp_path = fname_presuffix(
            self.inputs.t1w2mni_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w2mni_xfm,
            t1w2mni_xfm_tmp_path,
            copy=True,
            use_hardlink=False)

        t1w2mni_warp_tmp_path = fname_presuffix(
            self.inputs.t1w2mni_warp, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w2mni_warp,
            t1w2mni_warp_tmp_path,
            copy=True,
            use_hardlink=False)

        aligned_atlas_mni = regutils.RegisterParcellation2MNIFunc_align(
            parcellation_tmp_path,
            template_tmp_path,
            template_mask_tmp_path,
            t1w_brain_tmp_path,
            t1w2mni_xfm_tmp_path,
            aligned_atlas_t1w,
            aligned_atlas_mni,
            t1w2mni_warp_tmp_path,
            self.inputs.simple
        )
        time.sleep(0.5)

        out_dir = f"{self.inputs.dir_path}/t1w_clustered_parcellations/"
        os.makedirs(out_dir, exist_ok=True)
        copyfile(
            aligned_atlas_t1w,
            f"{out_dir}{atlas_name}_t1w_skull.nii.gz",
            copy=True,
            use_hardlink=False)

        self._results["aligned_atlas_mni"] = aligned_atlas_mni

        reg_tmp = [
            parcellation_tmp_path,
            template_tmp_path,
            t1w2mni_xfm_tmp_path,
            t1w2mni_warp_tmp_path,
            template_mask_tmp_path
        ]

        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        gc.collect()

        return runtime


class _RegisterAtlasFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterAtlasFunc"""

    atlas = traits.Any(mandatory=False)
    subnet = traits.Any(mandatory=True)
    parcellation4d = traits.Any(mandatory=True)
    parcellation = traits.Any(mandatory=False)
    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    t1w_brain_mask = File(exists=True, mandatory=True)
    t1_aligned_mni = File(exists=True, mandatory=True)
    gm_mask = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    node_radius = traits.Any(mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    reg_fmri_complete = traits.Bool()
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    already_run = traits.Bool(False, usedefault=True)
    dir_path = traits.Any(mandatory=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterAtlasFuncOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterAtlasFunc"""

    aligned_atlas_gm = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    node_radius = traits.Any()
    atlas = traits.Any()


class RegisterAtlasFunc(SimpleInterface):
    """Interface wrapper for RegisterAtlasFunc."""

    input_spec = _RegisterAtlasFuncInputSpec
    output_spec = _RegisterAtlasFuncOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import time
        import glob
        from pynets.registration import utils as regutils
        from pynets.core.nodemaker import \
            drop_coords_labels_from_restricted_parcellation
        from nipype.utils.filemanip import fname_presuffix, copyfile

        if self.inputs.subnet:
            atlas_name = f"{self.inputs.atlas}_{self.inputs.subnet}"
        else:
            atlas_name = f"{self.inputs.atlas}"

        if self.inputs.already_run is True:
            try:
                aligned_atlas_gm = [i for i in
                                    glob.glob(f"{self.inputs.dir_path}/"
                                              f"t1w_clustered_parcellations/"
                                              f"*.nii.gz") if
                                    atlas_name.strip('_mni') in i][0]

                # Correct coords and labels
                [aligned_atlas_gm, coords, labels] = \
                    drop_coords_labels_from_restricted_parcellation(
                        aligned_atlas_gm, self.inputs.coords,
                        self.inputs.labels)

            except FileNotFoundError as e:
                print(e, 'T1w-space parcellation not found. Did you delete '
                      'outputs?')
        else:
            if self.inputs.parcellation is None:
                parcellation_tmp_path = None
            else:
                parcellation_tmp_path = fname_presuffix(
                    self.inputs.parcellation, suffix="_tmp",
                    newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.parcellation,
                    parcellation_tmp_path,
                    copy=True,
                    use_hardlink=False)

            if self.inputs.parcellation4d is None:
                parcellation4d_tmp_path = None
            else:
                parcellation4d_tmp_path = fname_presuffix(
                    self.inputs.parcellation4d, suffix="_tmp",
                    newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.parcellation4d,
                    parcellation4d_tmp_path,
                    copy=True,
                    use_hardlink=False,
                )

            t1w_brain_tmp_path = fname_presuffix(
                self.inputs.t1w_brain, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.t1w_brain,
                t1w_brain_tmp_path,
                copy=True,
                use_hardlink=False)

            t1w_brain_mask_tmp_path = fname_presuffix(
                self.inputs.t1w_brain_mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.t1w_brain_mask,
                t1w_brain_mask_tmp_path,
                copy=True,
                use_hardlink=False,
            )

            t1_aligned_mni_tmp_path = fname_presuffix(
                self.inputs.t1_aligned_mni, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.t1_aligned_mni,
                t1_aligned_mni_tmp_path,
                copy=True,
                use_hardlink=False,
            )

            gm_mask_tmp_path = fname_presuffix(
                self.inputs.gm_mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.gm_mask,
                gm_mask_tmp_path,
                copy=True,
                use_hardlink=False)

            base_dir_tmp = f"{runtime.cwd}/atlas_{atlas_name}"
            os.makedirs(base_dir_tmp, exist_ok=True)

            aligned_atlas_t1mni = f"{base_dir_tmp}{'/'}{atlas_name}_" \
                                  f"t1w_mni.nii.gz"
            aligned_atlas_skull = f"{base_dir_tmp}{'/'}{atlas_name}_" \
                                  f"t1w_skull.nii.gz"
            aligned_atlas_gm = f"{base_dir_tmp}{'/'}{atlas_name}{'_gm.nii.gz'}"

            if self.inputs.node_radius is not None:
                atlas_name = f"{atlas_name}{'_'}{self.inputs.node_radius}"

            mni2t1_xfm_tmp_path = fname_presuffix(
                self.inputs.mni2t1_xfm, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mni2t1_xfm,
                mni2t1_xfm_tmp_path,
                copy=True,
                use_hardlink=False)

            mni2t1w_warp_tmp_path = fname_presuffix(
                self.inputs.mni2t1w_warp, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mni2t1w_warp,
                mni2t1w_warp_tmp_path,
                copy=True,
                use_hardlink=False,
            )

            aligned_atlas_gm, aligned_atlas_skull = regutils.atlas2t1w_align(
                parcellation_tmp_path,
                parcellation4d_tmp_path,
                atlas_name,
                t1w_brain_tmp_path,
                t1w_brain_mask_tmp_path,
                t1_aligned_mni_tmp_path,
                mni2t1w_warp_tmp_path,
                mni2t1_xfm_tmp_path,
                gm_mask_tmp_path,
                aligned_atlas_t1mni,
                aligned_atlas_skull,
                aligned_atlas_gm,
                self.inputs.simple,
            )
            time.sleep(0.5)

            # Correct coords and labels
            [aligned_atlas_gm, coords, labels] = \
                drop_coords_labels_from_restricted_parcellation(
                aligned_atlas_gm, self.inputs.coords, self.inputs.labels)

            reg_tmp = [
                parcellation4d_tmp_path,
                parcellation_tmp_path,
                gm_mask_tmp_path,
                t1_aligned_mni_tmp_path,
                t1w_brain_mask_tmp_path,
                mni2t1w_warp_tmp_path,
                mni2t1_xfm_tmp_path,
            ]

            if self.inputs.parcellation is None:
                parcellation_out = self.inputs.parcellation4d
                copyfile(
                    aligned_atlas_gm,
                    f"{os.path.dirname(parcellation_out)}/"
                    f"{os.path.basename(aligned_atlas_gm)}",
                    copy=True,
                    use_hardlink=False,
                )
            else:
                parcellation_out = self.inputs.parcellation
                copyfile(
                    aligned_atlas_gm,
                    f"{os.path.dirname(parcellation_out)}/parcellations/"
                    f"{os.path.basename(aligned_atlas_gm)}",
                    copy=True,
                    use_hardlink=False,
                )

            for j in reg_tmp:
                if j is not None:
                    if os.path.isfile(j):
                        os.system(f"rm -f {j} &")

        # Use for debugging check
        parcellation_img = nib.load(aligned_atlas_gm)

        intensities = [i for i in list(np.unique(
            np.asarray(
                parcellation_img.dataobj).astype("int"))
        ) if i != 0]
        try:
            assert len(coords) == len(labels) == len(intensities)
        except ValueError as e:
            print(e, 'Failed!')
            print(f"# Coords: {len(coords)}")
            print(f"# Labels: {len(labels)}")
            print(f"# Intensities: {len(intensities)}")

        self._results["aligned_atlas_gm"] = aligned_atlas_gm
        self._results["coords"] = coords
        self._results["labels"] = labels
        self._results["atlas"] = self.inputs.atlas
        self._results["node_radius"] = self.inputs.node_radius

        gc.collect()

        return runtime


class _RegisterROIEPIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterROIEPI"""

    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    roi = traits.Any(mandatory=False)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterROIEPIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterROIEPI"""

    roi = traits.Any(mandatory=False)


class RegisterROIEPI(SimpleInterface):
    """Interface wrapper for RegisterROIEPI."""

    input_spec = _RegisterROIEPIInputSpec
    output_spec = _RegisterROIEPIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import time
        from pynets.registration import utils as regutils
        from nipype.utils.filemanip import fname_presuffix, copyfile
        import pkg_resources

        template = pkg_resources.resource_filename(
            "pynets", f"templates/standard/{self.inputs.template_name}_brain_"
                      f"{self.inputs.vox_size}.nii.gz"
        )
        template_tmp_path = fname_presuffix(
            template, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            template,
            template_tmp_path,
            copy=True,
            use_hardlink=False)

        roi_file_tmp_path = fname_presuffix(
            self.inputs.roi, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.roi,
            roi_file_tmp_path,
            copy=True,
            use_hardlink=False)

        roi_in_t1w = f"{runtime.cwd}/roi-" \
                     f"{os.path.basename(self.inputs.roi).split('.nii')[0]}" \
                     f"_in_t1w.nii.gz"

        t1w_brain_tmp_path = fname_presuffix(
            self.inputs.t1w_brain, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.t1w_brain,
            t1w_brain_tmp_path,
            copy=True,
            use_hardlink=False)

        mni2t1w_warp_tmp_path = fname_presuffix(
            self.inputs.mni2t1w_warp, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1w_warp,
            mni2t1w_warp_tmp_path,
            copy=True,
            use_hardlink=False,
        )

        mni2t1_xfm_tmp_path = fname_presuffix(
            self.inputs.mni2t1_xfm, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.mni2t1_xfm,
            mni2t1_xfm_tmp_path,
            copy=True,
            use_hardlink=False)

        if self.inputs.roi:
            # Align roi
            roi_in_t1w = regutils.roi2t1w_align(
                roi_file_tmp_path,
                t1w_brain_tmp_path,
                mni2t1_xfm_tmp_path,
                mni2t1w_warp_tmp_path,
                roi_in_t1w,
                template_tmp_path,
                self.inputs.simple,
            )
            time.sleep(0.5)
        else:
            roi_in_t1w = None

        self._results["roi"] = roi_in_t1w

        reg_tmp = [
            t1w_brain_tmp_path,
            mni2t1w_warp_tmp_path,
            mni2t1_xfm_tmp_path,
            roi_file_tmp_path,
            template_tmp_path
        ]

        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")
        gc.collect()

        return runtime

