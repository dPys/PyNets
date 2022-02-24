#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import os
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


class _TrackingInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for Tracking"""

    B0_mask = File(exists=True, mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    wm_in_dwi = File(exists=True, mandatory=True)
    tiss_class = traits.Str(mandatory=True)
    labels_im_file_wm_gm_int = File(exists=True, mandatory=True)
    labels_im_file = File(exists=True, mandatory=True)
    curv_thr_list = traits.List(mandatory=True)
    step_list = traits.List(mandatory=True)
    track_type = traits.Str(mandatory=True)
    min_length = traits.Any(mandatory=True)
    maxcrossing = traits.Any(mandatory=True)
    traversal = traits.Str(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    subnet = traits.Any(mandatory=False)
    node_radius = traits.Any(mandatory=True)
    dens_thresh = traits.Bool(False, mandatory=False)
    ID = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=False)
    min_span_tree = traits.Bool(False, mandatory=False)
    disp_filt = traits.Bool(False, mandatory=False)
    parc = traits.Bool(mandatory=True)
    prune = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    norm = traits.Any(mandatory=True)
    binary = traits.Bool(False, usedefault=True)
    atlas_t1w = File(exists=True, mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    waymask = traits.Any(mandatory=False)
    t1w2dwi = File(exists=True, mandatory=True)


class _TrackingOutputSpec(TraitedSpec):
    """Output interface wrapper for Tracking"""

    streams = traits.Any()
    track_type = traits.Str(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    subnet = traits.Any(mandatory=False)
    node_radius = traits.Any()
    dens_thresh = traits.Bool(False, mandatory=False)
    ID = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=False)
    min_span_tree = traits.Bool(False, mandatory=False)
    disp_filt = traits.Bool(False, mandatory=False)
    parc = traits.Bool()
    prune = traits.Any()
    atlas = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    norm = traits.Any()
    binary = traits.Bool(False, usedefault=True)
    atlas_t1w = File(exists=True, mandatory=True)
    curv_thr_list = traits.List(mandatory=True)
    step_list = traits.List(mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    dm_path = traits.Any()
    traversal = traits.Str(mandatory=True)
    labels_im_file = File(exists=True, mandatory=True)
    min_length = traits.Any()


class Tracking(SimpleInterface):
    """Interface wrapper for Tracking"""

    input_spec = _TrackingInputSpec
    output_spec = _TrackingOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import time
        import os.path as op
        from dipy.io import load_pickle
        from colorama import Fore, Style
        from dipy.data import get_sphere
        from pynets.core import utils
        from pynets.core.utils import load_runconfig
        from pynets.dmri.estimation import reconstruction
        from pynets.dmri.track import (
            create_density_map,
            track_ensemble,
        )
        from dipy.io.stateful_tractogram import Space, StatefulTractogram, \
            Origin
        from dipy.io.streamline import save_tractogram
        from nipype.utils.filemanip import copyfile, fname_presuffix

        hardcoded_params = load_runconfig()
        use_life = hardcoded_params['tracking']["use_life"][0]
        roi_neighborhood_tol = hardcoded_params['tracking'][
            "roi_neighborhood_tol"][0]
        sphere = hardcoded_params['tracking']["sphere"][0]
        target_samples = hardcoded_params['tracking']["tracking_samples"][0]

        dir_path = utils.do_dir_path(
            self.inputs.atlas, os.path.dirname(self.inputs.dwi_file)
        )

        namer_dir = "{}/tractography".format(dir_path)
        if not os.path.isdir(namer_dir):
            os.makedirs(namer_dir, exist_ok=True)

        # Load diffusion data
        dwi_file_tmp_path = fname_presuffix(
            self.inputs.dwi_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.dwi_file,
            dwi_file_tmp_path,
            copy=True,
            use_hardlink=False)

        dwi_img = nib.load(dwi_file_tmp_path, mmap=True)
        dwi_data = dwi_img.get_fdata(dtype=np.float32)

        # Load FA data
        fa_file_tmp_path = fname_presuffix(
            self.inputs.fa_path, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.fa_path,
            fa_file_tmp_path,
            copy=True,
            use_hardlink=False)

        fa_img = nib.load(fa_file_tmp_path, mmap=True)

        labels_im_file_tmp_path = fname_presuffix(
            self.inputs.labels_im_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.labels_im_file,
            labels_im_file_tmp_path,
            copy=True,
            use_hardlink=False)

        # Load B0 mask
        B0_mask_tmp_path = fname_presuffix(
            self.inputs.B0_mask, suffix="_tmp",
            newpath=runtime.cwd
        )
        copyfile(
            self.inputs.B0_mask,
            B0_mask_tmp_path,
            copy=True,
            use_hardlink=False)

        streams = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
            runtime.cwd,
            "/streamlines_",
            "%s"
            % (self.inputs.subnet + "_" if self.inputs.subnet is not None
               else ""),
            "%s"
            % (
                op.basename(self.inputs.roi).split(".")[0] + "_"
                if self.inputs.roi is not None
                else ""
            ),
            self.inputs.conn_model,
            "_",
            target_samples,
            "_",
            "%s"
            % (
                "%s%s" % (self.inputs.node_radius, "mm_")
                if (
                    (self.inputs.node_radius != "parc")
                    and (self.inputs.node_radius is not None)
                )
                else "parc_"
            ),
            "curv-",
            str(self.inputs.curv_thr_list).replace(", ", "_"),
            "_step-",
            str(self.inputs.step_list).replace(", ", "_"),
            "_traversal-",
            self.inputs.traversal,
            "_minlength-",
            self.inputs.min_length,
            ".trk",
        )

        if os.path.isfile(f"{namer_dir}/{op.basename(streams)}"):
            from dipy.io.streamline import load_tractogram
            copyfile(
                f"{namer_dir}/{op.basename(streams)}",
                streams,
                copy=True,
                use_hardlink=False,
            )
            tractogram = load_tractogram(
                streams,
                fa_img,
                bbox_valid_check=False,
            )

            streamlines = tractogram.streamlines

            # Create streamline density map
            try:
                [dir_path, dm_path] = create_density_map(
                    fa_img,
                    dir_path,
                    streamlines,
                    self.inputs.conn_model,
                    self.inputs.node_radius,
                    self.inputs.curv_thr_list,
                    self.inputs.step_list,
                    self.inputs.subnet,
                    self.inputs.roi,
                    self.inputs.traversal,
                    self.inputs.min_length,
                    namer_dir,
                )
            except BaseException:
                print('Density map failed. Check tractography output.')
                dm_path = None

            del streamlines, tractogram
            fa_img.uncache()
            dwi_img.uncache()
            gc.collect()
            self._results["dm_path"] = dm_path
            self._results["streams"] = streams
            recon_path = None
        else:
            # Fit diffusion model
            # Save reconstruction to .npy
            recon_path = "%s%s%s%s%s%s%s%s" % (
                runtime.cwd,
                "/reconstruction_",
                "%s"
                % (self.inputs.subnet + "_" if self.inputs.subnet is not None
                   else ""),
                "%s"
                % (
                    op.basename(self.inputs.roi).split(".")[0] + "_"
                    if self.inputs.roi is not None
                    else ""
                ),
                self.inputs.conn_model,
                "_",
                "%s"
                % (
                    "%s%s" % (self.inputs.node_radius, "mm")
                    if (
                        (self.inputs.node_radius != "parc")
                        and (self.inputs.node_radius is not None)
                    )
                    else "parc"
                ),
                ".hdf5",
            )

            gtab_file_tmp_path = fname_presuffix(
                self.inputs.gtab_file, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.gtab_file,
                gtab_file_tmp_path,
                copy=True,
                use_hardlink=False)

            gtab = load_pickle(gtab_file_tmp_path)

            # Only re-run the reconstruction if we have to
            if not os.path.isfile(f"{namer_dir}/{op.basename(recon_path)}"):
                import h5py
                model = reconstruction(
                    self.inputs.conn_model,
                    gtab,
                    dwi_data,
                    B0_mask_tmp_path,
                )[0]
                with h5py.File(recon_path, 'w') as hf:
                    hf.create_dataset("reconstruction",
                                      data=model.astype('float32'), dtype='f4')
                hf.close()

                copyfile(
                    recon_path,
                    f"{namer_dir}/{op.basename(recon_path)}",
                    copy=True,
                    use_hardlink=False,
                )
                time.sleep(2)
                del model
            elif os.path.getsize(f"{namer_dir}/{op.basename(recon_path)}") > 0:
                print(
                    f"Found existing reconstruction with "
                    f"{self.inputs.conn_model}. Loading...")
                copyfile(
                    f"{namer_dir}/{op.basename(recon_path)}",
                    recon_path,
                    copy=True,
                    use_hardlink=False,
                )
                time.sleep(5)
            else:
                import h5py
                model = reconstruction(
                    self.inputs.conn_model,
                    gtab,
                    dwi_data,
                    B0_mask_tmp_path,
                )[0]
                with h5py.File(recon_path, 'w') as hf:
                    hf.create_dataset("reconstruction",
                                      data=model.astype('float32'), dtype='f4')
                hf.close()

                copyfile(
                    recon_path,
                    f"{namer_dir}/{op.basename(recon_path)}",
                    copy=True,
                    use_hardlink=False,
                )
                time.sleep(5)
                del model
            dwi_img.uncache()
            del dwi_data

            # Load atlas wm-gm interface reduced version for seeding
            labels_im_file_tmp_path_wm_gm_int = fname_presuffix(
                self.inputs.labels_im_file_wm_gm_int, suffix="_tmp",
                newpath=runtime.cwd
            )
            copyfile(
                self.inputs.labels_im_file_wm_gm_int,
                labels_im_file_tmp_path_wm_gm_int,
                copy=True,
                use_hardlink=False)

            t1w2dwi_tmp_path = fname_presuffix(
                self.inputs.t1w2dwi, suffix="_tmp",
                newpath=runtime.cwd
            )
            copyfile(
                self.inputs.t1w2dwi,
                t1w2dwi_tmp_path,
                copy=True,
                use_hardlink=False)

            gm_in_dwi_tmp_path = fname_presuffix(
                self.inputs.gm_in_dwi, suffix="_tmp",
                newpath=runtime.cwd
            )
            copyfile(
                self.inputs.gm_in_dwi,
                gm_in_dwi_tmp_path,
                copy=True,
                use_hardlink=False)

            vent_csf_in_dwi_tmp_path = fname_presuffix(
                self.inputs.vent_csf_in_dwi, suffix="_tmp",
                newpath=runtime.cwd
            )
            copyfile(
                self.inputs.vent_csf_in_dwi,
                vent_csf_in_dwi_tmp_path,
                copy=True,
                use_hardlink=False)

            wm_in_dwi_tmp_path = fname_presuffix(
                self.inputs.wm_in_dwi, suffix="_tmp",
                newpath=runtime.cwd
            )
            copyfile(
                self.inputs.wm_in_dwi,
                wm_in_dwi_tmp_path,
                copy=True,
                use_hardlink=False)

            if self.inputs.waymask:
                waymask_tmp_path = fname_presuffix(
                    self.inputs.waymask, suffix="_tmp",
                    newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.waymask,
                    waymask_tmp_path,
                    copy=True,
                    use_hardlink=False)
            else:
                waymask_tmp_path = None

            # Iteratively build a list of streamlines for each ROI while
            # tracking
            print(
                f"{Fore.GREEN}Target streamlines per iteration: "
                f"{Fore.BLUE} "
                f"{target_samples}"
            )
            print(Style.RESET_ALL)
            print(
                f"{Fore.GREEN}Curvature threshold(s): {Fore.BLUE} "
                f"{self.inputs.curv_thr_list}"
            )
            print(Style.RESET_ALL)
            print(f"{Fore.GREEN}Step size(s): {Fore.BLUE} "
                  f"{self.inputs.step_list}")
            print(Style.RESET_ALL)
            print(f"{Fore.GREEN}Tracking type: {Fore.BLUE} "
                  f"{self.inputs.track_type}")
            print(Style.RESET_ALL)
            if self.inputs.traversal == "prob":
                print(f"{Fore.GREEN}Direction-getting type: {Fore.BLUE}"
                      f"Probabilistic")
            elif self.inputs.traversal == "clos":
                print(f"{Fore.GREEN}Direction-getting type: "
                      f"{Fore.BLUE}Closest Peak")
            elif self.inputs.traversal == "det":
                print(
                    f"{Fore.GREEN}Direction-getting type: "
                    f"{Fore.BLUE}Deterministic Maximum"
                )
            else:
                raise ValueError("Direction-getting type not recognized!")

            print(Style.RESET_ALL)

            # Commence Ensemble Tractography
            try:
                streamlines = track_ensemble(
                    target_samples,
                    labels_im_file_tmp_path_wm_gm_int,
                    labels_im_file_tmp_path,
                    recon_path,
                    get_sphere(sphere),
                    self.inputs.traversal,
                    self.inputs.curv_thr_list,
                    self.inputs.step_list,
                    self.inputs.track_type,
                    self.inputs.maxcrossing,
                    int(roi_neighborhood_tol),
                    self.inputs.min_length,
                    waymask_tmp_path,
                    B0_mask_tmp_path,
                    t1w2dwi_tmp_path, gm_in_dwi_tmp_path,
                    vent_csf_in_dwi_tmp_path, wm_in_dwi_tmp_path,
                    self.inputs.tiss_class
                )
                gc.collect()
            except BaseException as w:
                print(f"\n{Fore.RED}Tractography failed: {w}")
                print(Style.RESET_ALL)
                streamlines = None

            if streamlines is not None:
                # import multiprocessing
                # from pynets.core.utils import kill_process_family
                # return kill_process_family(int(
                # multiprocessing.current_process().pid))

                # Linear Fascicle Evaluation (LiFE)
                if use_life is True:
                    print('Using LiFE to evaluate streamline plausibility...')
                    from pynets.dmri.utils import \
                        evaluate_streamline_plausibility
                    dwi_img = nib.load(dwi_file_tmp_path)
                    dwi_data = dwi_img.get_fdata(dtype=np.float32)
                    orig_count = len(streamlines)

                    if self.inputs.waymask:
                        mask_data = nib.load(
                            waymask_tmp_path).get_fdata().astype(
                            'bool').astype('int')
                    else:
                        mask_data = nib.load(
                            wm_in_dwi_tmp_path).get_fdata().astype(
                            'bool').astype('int')
                    try:
                        streamlines = evaluate_streamline_plausibility(
                            dwi_data, gtab, mask_data, streamlines,
                            sphere=sphere)
                    except BaseException:
                        print(f"Linear Fascicle Evaluation failed. "
                              f"Visually checking streamlines output "
                              f"{namer_dir}/{op.basename(streams)} is "
                              f"recommended.")
                    if len(streamlines) < 0.5*orig_count:
                        raise ValueError('LiFE revealed no plausible '
                                         'streamlines in the tractogram!')
                    del dwi_data, mask_data

                # Save streamlines to trk
                stf = StatefulTractogram(
                    streamlines,
                    fa_img,
                    origin=Origin.NIFTI,
                    space=Space.VOXMM)
                stf.remove_invalid_streamlines()

                save_tractogram(
                    stf,
                    streams,
                )

                del stf

                copyfile(
                    streams,
                    f"{namer_dir}/{op.basename(streams)}",
                    copy=True,
                    use_hardlink=False,
                )

                # Create streamline density map
                try:
                    [dir_path, dm_path] = create_density_map(
                        dwi_img,
                        dir_path,
                        streamlines,
                        self.inputs.conn_model,
                        self.inputs.node_radius,
                        self.inputs.curv_thr_list,
                        self.inputs.step_list,
                        self.inputs.subnet,
                        self.inputs.roi,
                        self.inputs.traversal,
                        self.inputs.min_length,
                        namer_dir,
                    )
                except BaseException:
                    print('Density map failed. Check tractography output.')
                    dm_path = None

                del streamlines
                dwi_img.uncache()
                gc.collect()
                self._results["dm_path"] = dm_path
                self._results["streams"] = streams
            else:
                self._results["streams"] = None
                self._results["dm_path"] = None
            tmp_files = [gtab_file_tmp_path,
                         wm_in_dwi_tmp_path, gm_in_dwi_tmp_path,
                         vent_csf_in_dwi_tmp_path, t1w2dwi_tmp_path]

            for j in tmp_files:
                if j is not None:
                    if os.path.isfile(j):
                        os.system(f"rm -f {j} &")

        self._results["track_type"] = self.inputs.track_type
        self._results["conn_model"] = self.inputs.conn_model
        self._results["dir_path"] = dir_path
        self._results["subnet"] = self.inputs.subnet
        self._results["node_radius"] = self.inputs.node_radius
        self._results["dens_thresh"] = self.inputs.dens_thresh
        self._results["ID"] = self.inputs.ID
        self._results["roi"] = self.inputs.roi
        self._results["min_span_tree"] = self.inputs.min_span_tree
        self._results["disp_filt"] = self.inputs.disp_filt
        self._results["parc"] = self.inputs.parc
        self._results["prune"] = self.inputs.prune
        self._results["atlas"] = self.inputs.atlas
        self._results["parcellation"] = self.inputs.parcellation
        self._results["labels"] = self.inputs.labels
        self._results["coords"] = self.inputs.coords
        self._results["norm"] = self.inputs.norm
        self._results["binary"] = self.inputs.binary
        self._results["atlas_t1w"] = self.inputs.atlas_t1w
        self._results["curv_thr_list"] = self.inputs.curv_thr_list
        self._results["step_list"] = self.inputs.step_list
        self._results["fa_path"] = fa_file_tmp_path
        self._results["traversal"] = self.inputs.traversal
        self._results["labels_im_file"] = labels_im_file_tmp_path
        self._results["min_length"] = self.inputs.min_length

        tmp_files = [B0_mask_tmp_path, dwi_file_tmp_path]

        for j in tmp_files:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        # Exercise caution when deleting copied recon_path
        # if recon_path is not None:
        #     if os.path.isfile(recon_path):
        #         os.remove(recon_path)

        return runtime


class _MakeGtabBmaskInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for MakeGtabBmask"""

    dwi_file = File(exists=True, mandatory=True)
    fbval = File(exists=True, mandatory=True)
    fbvec = File(exists=True, mandatory=True)
    b0_thr = traits.Int(default_value=50, usedefault=True)


class _MakeGtabBmaskOutputSpec(TraitedSpec):
    """Output interface wrapper for MakeGtabBmask"""

    gtab_file = File(exists=True, mandatory=True)
    B0_bet = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)


class MakeGtabBmask(SimpleInterface):
    """Interface wrapper for MakeGtabBmask"""

    input_spec = _MakeGtabBmaskInputSpec
    output_spec = _MakeGtabBmaskOutputSpec

    def _run_interface(self, runtime):
        import os
        import time
        from dipy.io import save_pickle
        from dipy.io import read_bvals_bvecs
        from dipy.core.gradients import gradient_table
        from nipype.utils.filemanip import copyfile, fname_presuffix
        # from dipy.segment.mask import median_otsu
        from pynets.registration.utils import median
        from pynets.dmri.utils import normalize_gradients, extract_b0

        B0_bet = f"{runtime.cwd}/mean_B0_bet.nii.gz"
        B0_mask = f"{runtime.cwd}/mean_B0_bet_mask.nii.gz"
        fbvec_norm = f"{runtime.cwd}/bvec_normed.bvec"
        fbval_norm = f"{runtime.cwd}/bval_normed.bvec"
        gtab_file = f"{runtime.cwd}/gtab.pkl"
        all_b0s_file = f"{runtime.cwd}/all_b0s.nii.gz"

        fbval_tmp_path = fname_presuffix(
            self.inputs.fbval, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.fbval,
            fbval_tmp_path,
            copy=True,
            use_hardlink=False)

        fbvec_tmp_path = fname_presuffix(
            self.inputs.fbvec, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.fbvec,
            fbvec_tmp_path,
            copy=True,
            use_hardlink=False)

        # loading bvecs/bvals
        bvals, bvecs = read_bvals_bvecs(fbval_tmp_path, fbvec_tmp_path)
        bvecs_norm, bvals_norm = normalize_gradients(
            bvecs, bvals, b0_threshold=self.inputs.b0_thr
        )

        # Save corrected
        np.savetxt(fbval_norm, bvals_norm)
        np.savetxt(fbvec_norm, bvecs_norm)

        # Creating the gradient table
        gtab = gradient_table(bvals_norm, bvecs_norm)

        # Correct b0 threshold
        gtab.b0_threshold = self.inputs.b0_thr

        # Correct bvals to set 0's for B0 based on thresh
        gtab_bvals = gtab.bvals.copy()
        b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
        gtab_bvals[b0_thr_ixs] = 0
        gtab.b0s_mask = gtab_bvals == 0

        # Show info
        print(gtab.info)

        # Save gradient table to pickle
        save_pickle(gtab_file, gtab)

        dwi_file_tmp_path = fname_presuffix(
            self.inputs.dwi_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.dwi_file,
            dwi_file_tmp_path,
            copy=True,
            use_hardlink=False)

        # Extract and Combine all b0s collected, make mean b0
        print("Extracting b0's...")
        all_b0s_file = extract_b0(
            dwi_file_tmp_path,
            b0_thr_ixs,
            all_b0s_file)
        med_b0_file = median(all_b0s_file)

        # TODO replace with bet and median_otsu with deep-learning classifier.

        # med_b0_img = nib.load(med_b0_file)
        # med_b0_data = np.asarray(med_b0_img.dataobj)
        # # Create mean b0 brain mask
        # b0_mask_data, mask_data = median_otsu(med_b0_data, median_radius=2,
        # numpass=1)
        #
        # hdr = med_b0_img.header.copy()
        # hdr.set_xyzt_units("mm")
        # hdr.set_data_dtype(np.float32)
        # nib.Nifti1Image(b0_mask_data, med_b0_img.affine,
        #                 hdr).to_filename(B0_bet)
        # nib.Nifti1Image(mask_data, med_b0_img.affine,
        #                 hdr).to_filename(B0_mask)

        # Get mean B0 brain mask
        cmd = f"bet {med_b0_file} {B0_bet} -m -f 0.2"
        os.system(cmd)
        time.sleep(2)

        self._results["gtab_file"] = gtab_file
        self._results["B0_bet"] = B0_bet
        self._results["B0_mask"] = B0_mask
        self._results["dwi_file"] = self.inputs.dwi_file

        tmp_files = [fbval_tmp_path, fbvec_tmp_path, dwi_file_tmp_path]
        for j in tmp_files:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        return runtime
