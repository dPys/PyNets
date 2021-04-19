#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
import nibabel as nib
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    Directory,
)

warnings.filterwarnings("ignore")


class _FetchNodesLabelsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for FetchNodesLabels"""

    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    ref_txt = traits.Any()
    in_file = traits.Any(mandatory=True)
    parc = traits.Bool(mandatory=True)
    use_parcel_naming = traits.Bool(False, usedefault=True)
    outdir = traits.Str(mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    clustering = traits.Bool(False, usedefault=True)


class _FetchNodesLabelsOutputSpec(TraitedSpec):
    """Output interface wrapper for FetchNodesLabels"""

    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    atlas = traits.Any()
    networks_list = traits.Any()
    parcel_list = traits.Any()
    par_max = traits.Any()
    uatlas = traits.Any()
    dir_path = traits.Any()


class FetchNodesLabels(SimpleInterface):
    """Interface wrapper for FetchNodesLabels."""

    input_spec = _FetchNodesLabelsInputSpec
    output_spec = _FetchNodesLabelsOutputSpec

    def _run_interface(self, runtime):
        import sys
        import pkg_resources
        from pynets.core import utils, nodemaker
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from nilearn.image import concat_imgs
        import pandas as pd
        import time
        import textwrap
        from pathlib import Path
        import os.path as op
        import glob

        base_path = utils.get_file()
        # Test if atlas is a nilearn atlas. If so, fetch coords, labels, and/or
        # networks.
        nilearn_parc_atlases = [
            "atlas_harvard_oxford",
            "atlas_aal",
            "atlas_destrieux_2009",
            "atlas_talairach_gyrus",
            "atlas_talairach_ba",
            "atlas_talairach_lobe",
        ]
        nilearn_coords_atlases = ["coords_power_2011", "coords_dosenbach_2010"]
        nilearn_prob_atlases = ["atlas_msdl", "atlas_pauli_2017"]
        local_atlases = [
            op.basename(i).split(".nii")[0]
            for i in glob.glob(f"{str(Path(base_path).parent.parent)}"
                               f"/templates/atlases/*.nii.gz")
            if "_4d" not in i
        ]

        if self.inputs.uatlas is None and self.inputs.atlas in \
                nilearn_parc_atlases:
            [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(
                self.inputs.atlas, self.inputs.parc
            )
            if uatlas:
                if not isinstance(uatlas, str):
                    nib.save(
                        uatlas, f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}")
                    uatlas = f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}"
                if self.inputs.clustering is False:
                    [uatlas,
                     labels] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                        uatlas, label_names=labels)
                [coords, atlas, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
            else:
                raise FileNotFoundError(
                    f"\nAtlas file for {self.inputs.atlas} not found!"
                )

            atlas = self.inputs.atlas
        elif (
            self.inputs.uatlas is None
            and self.inputs.parc is False
            and self.inputs.atlas in nilearn_coords_atlases
        ):
            print(
                "Fetching coords and labels from nilearn coordinate-based"
                " atlas library..."
            )
            # Fetch nilearn atlas coords
            [coords, _, networks_list,
             labels] = nodemaker.fetch_nilearn_atlas_coords(
                self.inputs.atlas)
            parcel_list = None
            par_max = None
            atlas = self.inputs.atlas
            uatlas = None
            label_intensities = None
        elif (
            self.inputs.uatlas is None
            and self.inputs.parc is False
            and self.inputs.atlas in nilearn_prob_atlases
        ):
            import matplotlib
            matplotlib.use("agg")
            from nilearn.plotting import find_probabilistic_atlas_cut_coords

            print(
                "Fetching coords and labels from nilearn probabilistic atlas"
                " library..."
            )
            # Fetch nilearn atlas coords
            [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(
                self.inputs.atlas, self.inputs.parc
            )
            coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas)
            if uatlas:
                if not isinstance(uatlas, str):
                    nib.save(
                        uatlas, f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}")
                    uatlas = f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}"
                if self.inputs.clustering is False:
                    [uatlas,
                     labels] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                        uatlas, label_names=labels)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
            else:
                raise FileNotFoundError(
                    f"\nAtlas file for {self.inputs.atlas} not found!")

            par_max = None
            atlas = self.inputs.atlas
            label_intensities = None
        elif self.inputs.uatlas is None and self.inputs.atlas in local_atlases:
            uatlas_pre = (
                f"{str(Path(base_path).parent.parent)}/templates/atlases/"
                f"{self.inputs.atlas}.nii.gz"
            )
            uatlas = fname_presuffix(
                uatlas_pre, newpath=runtime.cwd)
            copyfile(uatlas_pre, uatlas, copy=True, use_hardlink=False)
            try:
                par_img = nib.load(uatlas)
            except indexed_gzip.ZranError as e:
                print(e,
                      "\nCannot load RSN reference image. Do you have git-lfs "
                      "installed?")
            try:
                if self.inputs.clustering is False:
                    [uatlas, _] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                            uatlas)

                # Fetch user-specified atlas coords
                [coords, _, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
                # Describe user atlas coords
                print(f"\n{self.inputs.atlas} comes with {par_max} parcels\n")
            except ValueError as e:
                print(e,
                      "Either you have specified the name of an atlas that "
                      "does not exist in the nilearn or local repository or "
                      "you have not supplied a 3d atlas parcellation image!")
            labels = None
            networks_list = None
            atlas = self.inputs.atlas
        elif self.inputs.uatlas:
            if self.inputs.clustering is True:
                while True:
                    if op.isfile(self.inputs.uatlas):
                        break
                    else:
                        print("Waiting for atlas file...")
                        time.sleep(5)

            try:
                uatlas_tmp_path = fname_presuffix(
                    self.inputs.uatlas, newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.uatlas,
                    uatlas_tmp_path,
                    copy=True,
                    use_hardlink=False)
                # Fetch user-specified atlas coords
                if self.inputs.clustering is False:
                    [uatlas,
                     _] = nodemaker.enforce_hem_distinct_consecutive_labels(
                        uatlas_tmp_path)
                else:
                    uatlas = uatlas_tmp_path
                [coords, atlas, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None

                atlas = utils.prune_suffices(atlas)

                # Describe user atlas coords
                print(f"\n{atlas} comes with {par_max} parcels\n")
            except ValueError as e:
                print(e,
                      "Either you have specified the name of an atlas that "
                      "does not exist in the nilearn or local repository or "
                      "you have not supplied a 3d atlas parcellation image!")
            labels = None
            networks_list = None
        else:
            raise ValueError(
                "Either you have specified the name of an atlas that does"
                " not exist in the nilearn or local repository or you have"
                " not supplied a 3d atlas parcellation image!")

        # Labels prep
        if atlas and not labels:
            if (self.inputs.ref_txt is not None) and (
                    op.exists(self.inputs.ref_txt)):
                labels = pd.read_csv(
                    self.inputs.ref_txt, sep=" ", header=None, names=[
                        "Index", "Region"])["Region"].tolist()
            else:
                if atlas in local_atlases:
                    ref_txt = (
                        f"{str(Path(base_path).parent.parent)}/templates/"
                        f"labelcharts/"
                        f"{atlas}.txt"
                    )
                else:
                    ref_txt = self.inputs.ref_txt
                if ref_txt is not None:
                    try:
                        labels = pd.read_csv(
                            ref_txt, sep=" ", header=None, names=[
                                "Index", "Region"])["Region"].tolist()
                    except BaseException:
                        if self.inputs.use_parcel_naming is True:
                            try:
                                labels = nodemaker.parcel_naming(
                                    coords, self.inputs.vox_size)
                            except BaseException:
                                print("AAL reference labeling failed!")
                                labels = np.arange(len(coords) + 1)[
                                    np.arange(len(coords) + 1) != 0
                                ].tolist()
                        else:
                            print("Using generic index labels...")
                            labels = np.arange(len(coords) + 1)[
                                np.arange(len(coords) + 1) != 0
                            ].tolist()
                else:
                    if self.inputs.use_parcel_naming is True:
                        try:
                            labels = nodemaker.parcel_naming(
                                coords, self.inputs.vox_size)
                        except BaseException:
                            print("AAL reference labeling failed!")
                            labels = np.arange(len(coords) + 1)[
                                np.arange(len(coords) + 1) != 0
                            ].tolist()
                    else:
                        print("Using generic index labels...")
                        labels = np.arange(len(coords) + 1)[
                            np.arange(len(coords) + 1) != 0
                        ].tolist()

        dir_path = utils.do_dir_path(atlas, self.inputs.outdir)

        if len(coords) != len(labels):
            labels = [i for i in labels if (i != 'Unknown' and
                                            i != 'Background')]
            if len(coords) != len(labels):
                print("Length of coordinates is not equal to length of "
                      "label names...")
                if self.inputs.use_parcel_naming is True:
                    try:
                        print("Attempting consensus parcel naming instead...")
                        labels = nodemaker.parcel_naming(
                            coords, self.inputs.vox_size)
                    except BaseException:
                        print("Reverting to integer labels instead...")
                        labels = np.arange(len(coords) + 1)[
                            np.arange(len(coords) + 1) != 0
                        ].tolist()
                else:
                    print("Reverting to integer labels instead...")
                    labels = np.arange(len(coords) + 1)[
                        np.arange(len(coords) + 1) != 0
                    ].tolist()

        print(f"Coordinates:\n{coords}")
        print(f"Labels:\n"
              f"{textwrap.shorten(str(labels), width=1000, placeholder='...')}")

        assert len(coords) == len(labels)

        if label_intensities is not None:
            self._results["labels"] = list(zip(labels, label_intensities))
        else:
            self._results["labels"] = labels
        self._results["coords"] = coords
        self._results["atlas"] = atlas
        self._results["networks_list"] = networks_list
        # TODO: Optimize this with 4d array concatenation and .npyz

        parcel_list_4d = concat_imgs([i for i in parcel_list])
        out_path = f"{runtime.cwd}/parcel_list.nii.gz"
        nib.save(parcel_list_4d, out_path)
        self._results["parcel_list"] = out_path
        self._results["par_max"] = par_max
        self._results["uatlas"] = uatlas
        self._results["dir_path"] = dir_path

        return runtime


class NetworkAnalysisInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for NetworkAnalysis"""

    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path = File(exists=True, mandatory=True)
    roi = traits.Any(mandatory=False)
    prune = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=False)
    binary = traits.Bool(False, usedefault=True)


class NetworkAnalysisOutputSpec(TraitedSpec):
    """Output interface wrapper for NetworkAnalysis"""

    out_path_neat = File(exists=True, mandatory=True)


class NetworkAnalysis(BaseInterface):
    """Interface wrapper for NetworkAnalysis"""

    input_spec = NetworkAnalysisInputSpec
    output_spec = NetworkAnalysisOutputSpec

    def _run_interface(self, runtime):
        from pynets.stats.netstats import extractnetstats

        out = extractnetstats(
            self.inputs.ID,
            self.inputs.network,
            self.inputs.thr,
            self.inputs.conn_model,
            self.inputs.est_path,
            self.inputs.roi,
            self.inputs.prune,
            self.inputs.norm,
            self.inputs.binary,
        )
        setattr(self, "_outpath", out)
        return runtime

    def _list_outputs(self):
        import os.path as op

        return {"out_path_neat": op.abspath(getattr(self, "_outpath"))}


class CombineOutputsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for CombineOutputs"""

    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    net_mets_csv_list = traits.List(mandatory=True)
    plot_switch = traits.Bool(False, usedefault=True)
    multi_nets = traits.Any(mandatory=False)
    multimodal = traits.Bool(False, usedefault=True)
    embed = traits.Bool(False, usedefault=True)


class CombineOutputsOutputSpec(TraitedSpec):
    """Output interface wrapper for CombineOutputs"""

    combination_complete = traits.Bool()


class CombineOutputs(SimpleInterface):
    """Interface wrapper for CombineOutputs"""

    input_spec = CombineOutputsInputSpec
    output_spec = CombineOutputsOutputSpec

    def _run_interface(self, runtime):
        from pynets.core.utils import collect_pandas_df

        combination_complete = collect_pandas_df(
            self.inputs.network,
            self.inputs.ID,
            self.inputs.net_mets_csv_list,
            self.inputs.plot_switch,
            self.inputs.multi_nets,
            self.inputs.multimodal,
            self.inputs.embed,
        )
        setattr(self, "_combination_complete", combination_complete)
        return runtime

    def _list_outputs(self):
        return {"combination_complete": getattr(self, "_combination_complete")}


class _IndividualClusteringInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for IndividualClustering"""

    func_file = File(exists=True, mandatory=True)
    conf = traits.Any(mandatory=False)
    clust_mask = File(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    local_corr = traits.Str("allcorr", mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    outdir = traits.Str(mandatory=True)
    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    t1w_brain = File(exists=True, mandatory=True)
    mni2t1w_warp = File(exists=True, mandatory=True)
    mni2t1_xfm = File(exists=True, mandatory=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _IndividualClusteringOutputSpec(TraitedSpec):
    """Output interface wrapper for IndividualClustering"""

    uatlas = File(exists=True)
    atlas = traits.Str(mandatory=True)
    clustering = traits.Bool(True, usedefault=True)
    clust_mask = File(exists=True, mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)
    func_file = File(exists=True, mandatory=True)


class IndividualClustering(SimpleInterface):
    """Interface wrapper for IndividualClustering"""

    input_spec = _IndividualClusteringInputSpec
    output_spec = _IndividualClusteringOutputSpec

    def _run_interface(self, runtime):
        import os
        import gc
        import time
        import nibabel as nib
        from pynets.core.utils import load_runconfig
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.fmri import clustools
        from pynets.registration.utils import check_orient_and_dims
        from joblib import Parallel, delayed
        from joblib.externals.loky.backend import resource_tracker
        from pynets.registration import utils as regutils
        from pynets.core.utils import decompress_nifti
        import pkg_resources
        import shutil
        import tempfile
        resource_tracker.warnings = None

        template = pkg_resources.resource_filename(
            "pynets", f"templates/{self.inputs.template_name}_brain_"
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

        hardcoded_params = load_runconfig()
        c_boot = hardcoded_params["c_boot"][0]
        nthreads = hardcoded_params["nthreads"][0]

        clust_list = ["kmeans", "ward", "complete", "average", "ncut", "rena"]

        clust_mask_temp_path = check_orient_and_dims(
            self.inputs.clust_mask, runtime.cwd, self.inputs.vox_size
        )
        cm_suf = os.path.basename(self.inputs.clust_mask).split('.nii')[0]
        clust_mask_in_t1w_path = f"{runtime.cwd}/clust_mask-" \
                                 f"{cm_suf}_in_t1w.nii.gz"

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

        clust_mask_in_t1w = regutils.roi2t1w_align(
            clust_mask_temp_path,
            t1w_brain_tmp_path,
            mni2t1_xfm_tmp_path,
            mni2t1w_warp_tmp_path,
            clust_mask_in_t1w_path,
            template_tmp_path,
            self.inputs.simple,
        )
        time.sleep(0.5)

        if self.inputs.mask:
            out_name_mask = fname_presuffix(
                self.inputs.mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mask,
                out_name_mask,
                copy=True,
                use_hardlink=False)
        else:
            out_name_mask = None

        out_name_func_file = fname_presuffix(
            self.inputs.func_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.func_file,
            out_name_func_file,
            copy=True,
            use_hardlink=False)
        out_name_func_file = decompress_nifti(out_name_func_file)

        if self.inputs.conf:
            out_name_conf = fname_presuffix(
                self.inputs.conf, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.conf,
                out_name_conf,
                copy=True,
                use_hardlink=False)
        else:
            out_name_conf = None

        nip = clustools.NiParcellate(
            func_file=out_name_func_file,
            clust_mask=clust_mask_in_t1w,
            k=int(self.inputs.k),
            clust_type=self.inputs.clust_type,
            local_corr=self.inputs.local_corr,
            outdir=self.inputs.outdir,
            conf=out_name_conf,
            mask=out_name_mask,
        )

        atlas = nip.create_clean_mask()
        nip.create_local_clustering(overwrite=True, r_thresh=0.4)

        if self.inputs.clust_type in clust_list:
            if float(c_boot) > 1:
                import random
                from joblib import Memory
                from joblib.externals.loky import get_reusable_executor
                print(
                    f"Performing circular block bootstrapping with {c_boot}"
                    f" iterations..."
                )
                ts_data, block_size = nip.prep_boot()

                cache_dir = tempfile.mkdtemp()
                memory = Memory(cache_dir, verbose=0)
                ts_data = memory.cache(ts_data)

                def create_bs_imgs(ts_data, block_size, clust_mask_corr_img):
                    import nibabel as nib
                    from nilearn.masking import unmask
                    from pynets.fmri.estimation import timeseries_bootstrap
                    boot_series = timeseries_bootstrap(
                        ts_data.func, block_size)[0].astype('float32')
                    return unmask(boot_series, clust_mask_corr_img)

                def run_bs_iteration(i, ts_data, work_dir, local_corr,
                                     clust_type, _local_conn_mat_path,
                                     num_conn_comps, _clust_mask_corr_img,
                                     _standardize, _detrending, k,
                                     _local_conn, conf, _dir_path,
                                     _conn_comps):
                    import os
                    import time
                    import gc
                    from pynets.fmri.clustools import parcellate
                    print(f"\nBootstrapped iteration: {i}")
                    out_path = f"{work_dir}/boot_parc_tmp_{str(i)}.nii.gz"

                    boot_img = create_bs_imgs(ts_data, block_size,
                                              _clust_mask_corr_img)
                    try:
                        parcellation = parcellate(boot_img, local_corr,
                                                  clust_type,
                                                  _local_conn_mat_path,
                                                  num_conn_comps,
                                                  _clust_mask_corr_img,
                                                  _standardize,
                                                  _detrending, k, _local_conn,
                                                  conf, _dir_path,
                                                  _conn_comps)
                        parcellation.to_filename(out_path)
                        parcellation.uncache()
                        boot_img.uncache()
                        gc.collect()
                    except BaseException:
                        boot_img.uncache()
                        gc.collect()
                        return None
                    _clust_mask_corr_img.uncache()
                    return out_path

                time.sleep(random.randint(1, 5))
                counter = 0
                boot_parcellations = []
                while float(counter) < float(c_boot):
                    with Parallel(n_jobs=nthreads, max_nbytes='8000M',
                                  backend='loky', mmap_mode='r+',
                                  temp_folder=cache_dir,
                                  verbose=10) as parallel:
                        iter_bootedparcels = parallel(
                            delayed(run_bs_iteration)(
                                i, ts_data, runtime.cwd, nip.local_corr,
                                nip.clust_type, nip._local_conn_mat_path,
                                nip.num_conn_comps, nip._clust_mask_corr_img,
                                nip._standardize, nip._detrending, nip.k,
                                nip._local_conn, nip.conf, nip._dir_path,
                                nip._conn_comps) for i in
                            range(c_boot))

                        boot_parcellations.extend([i for i in
                                                   iter_bootedparcels if
                                                   i is not None])
                        counter = len(boot_parcellations)
                        del iter_bootedparcels
                        gc.collect()

                print('Bootstrapped samples complete:')
                print(boot_parcellations)
                print("Creating spatially-constrained consensus "
                      "parcellation...")
                consensus_parcellation = clustools.ensemble_parcellate(
                    boot_parcellations,
                    int(self.inputs.k)
                )
                nib.save(consensus_parcellation, nip.uatlas)
                memory.clear(warn=False)
                shutil.rmtree(cache_dir, ignore_errors=True)
                del parallel, memory, cache_dir
                get_reusable_executor().shutdown(wait=True)
                gc.collect()

                for i in boot_parcellations:
                    if i is not None:
                        if os.path.isfile(i):
                            os.system(f"rm -f {i} &")
            else:
                print(
                    "Creating spatially-constrained parcellation...")
                out_path = f"{runtime.cwd}/{atlas}_{str(self.inputs.k)}.nii.gz"
                func_img = nib.load(out_name_func_file)
                parcellation = clustools.parcellate(func_img,
                                                    self.inputs.local_corr,
                                                    self.inputs.clust_type,
                                                    nip._local_conn_mat_path,
                                                    nip.num_conn_comps,
                                                    nip._clust_mask_corr_img,
                                                    nip._standardize,
                                                    nip._detrending, nip.k,
                                                    nip._local_conn,
                                                    nip.conf, nip._dir_path,
                                                    nip._conn_comps)
                parcellation.to_filename(out_path)

        else:
            raise ValueError(
                "Clustering method not recognized. See: "
                "https://nilearn.github.io/modules/generated/"
                "nilearn.regions.Parcellations."
                "html#nilearn.regions.Parcellations")

        # Give it a minute
        ix = 0
        while not os.path.isfile(nip.uatlas) and ix < 60:
            print('Waiting for clustered parcellation...')
            time.sleep(1)
            ix += 1

        if not os.path.isfile(nip.uatlas):
            raise FileNotFoundError(f"Parcellation clustering failed for"
                                    f" {nip.uatlas}")

        self._results["atlas"] = atlas
        self._results["uatlas"] = nip.uatlas
        self._results["clust_mask"] = clust_mask_in_t1w_path
        self._results["k"] = self.inputs.k
        self._results["clust_type"] = self.inputs.clust_type
        self._results["clustering"] = True
        self._results["func_file"] = self.inputs.func_file

        reg_tmp = [
            t1w_brain_tmp_path,
            mni2t1w_warp_tmp_path,
            mni2t1_xfm_tmp_path,
            template_tmp_path,
            out_name_func_file
        ]
        for j in reg_tmp:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        gc.collect()

        return runtime


class _ExtractTimeseriesInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for ExtractTimeseries"""

    conf = traits.Any(mandatory=False)
    func_file = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=False)
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    smooth = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    hpass = traits.Any(mandatory=True)
    mask = traits.Any(mandatory=False)
    parc = traits.Bool()
    node_size = traits.Any(mandatory=False)
    net_parcels_nii_path = traits.Any(mandatory=False)
    extract_strategy = traits.Str("mean", mandatory=False, usedefault=True)


class _ExtractTimeseriesOutputSpec(TraitedSpec):
    """Output interface wrapper for ExtractTimeseries"""

    ts_within_nodes = traits.Any(mandatory=True)
    node_size = traits.Any(mandatory=True)
    smooth = traits.Any(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    hpass = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=True)
    extract_strategy = traits.Any(mandatory=False)


class ExtractTimeseries(SimpleInterface):
    """Interface wrapper for ExtractTimeseries"""

    input_spec = _ExtractTimeseriesInputSpec
    output_spec = _ExtractTimeseriesOutputSpec

    def _run_interface(self, runtime):
        import gc
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.fmri import estimation
        from pynets.core.utils import decompress_nifti

        # Decompressing each image to facilitate more seamless memory mapping
        if self.inputs.net_parcels_nii_path:
            out_name_net_parcels_nii_path = fname_presuffix(
                self.inputs.net_parcels_nii_path, suffix="_tmp",
                newpath=runtime.cwd)
            copyfile(
                self.inputs.net_parcels_nii_path,
                out_name_net_parcels_nii_path,
                copy=True,
                use_hardlink=False,
            )
            out_name_net_parcels_nii_path = decompress_nifti(
                out_name_net_parcels_nii_path)
        else:
            out_name_net_parcels_nii_path = None

        if self.inputs.mask:
            out_name_mask = fname_presuffix(
                self.inputs.mask, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.mask,
                out_name_mask,
                copy=True,
                use_hardlink=False)
            out_name_mask = decompress_nifti(out_name_mask)
        else:
            out_name_mask = None

        out_name_func_file = fname_presuffix(
            self.inputs.func_file, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.func_file,
            out_name_func_file,
            copy=True,
            use_hardlink=False)
        out_name_func_file = decompress_nifti(out_name_func_file)

        if self.inputs.conf:
            out_name_conf = fname_presuffix(
                self.inputs.conf, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.conf,
                out_name_conf,
                copy=True,
                use_hardlink=False)
        else:
            out_name_conf = None

        te = estimation.TimeseriesExtraction(
            net_parcels_nii_path=out_name_net_parcels_nii_path,
            node_size=self.inputs.node_size,
            conf=out_name_conf,
            func_file=out_name_func_file,
            roi=self.inputs.roi,
            dir_path=self.inputs.dir_path,
            ID=self.inputs.ID,
            network=self.inputs.network,
            smooth=self.inputs.smooth,
            hpass=self.inputs.hpass,
            mask=out_name_mask,
            extract_strategy=self.inputs.extract_strategy,
        )

        te.prepare_inputs()

        te.extract_ts_parc()

        te.save_and_cleanup()

        try:
            assert (
                len(self.inputs.coords)
                == len(self.inputs.labels)
                == te.ts_within_nodes.shape[1]
            )
        except AssertionError as e:
            e.args += ('Coords: ', len(self.inputs.coords),
                       self.inputs.coords, 'Labels:',
                       len(self.inputs.labels),
                       self.inputs.labels, te.ts_within_nodes.shape)

        self._results["ts_within_nodes"] = te.ts_within_nodes
        self._results["node_size"] = te.node_size
        self._results["smooth"] = te.smooth
        self._results["extract_strategy"] = te.extract_strategy
        self._results["dir_path"] = te.dir_path
        self._results["atlas"] = self.inputs.atlas
        self._results["uatlas"] = self.inputs.uatlas
        self._results["labels"] = self.inputs.labels
        self._results["coords"] = self.inputs.coords
        self._results["hpass"] = te.hpass
        self._results["roi"] = self.inputs.roi

        del te
        gc.collect()

        return runtime


class _PlotStructInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotStruct"""

    conn_matrix = traits.Any()
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    labels = traits.Array(mandatory=True)
    roi = traits.Any(mandatory=True)
    coords = traits.Array(mandatory=True)
    thr = traits.Any(mandatory=True)
    node_size = traits.Any(mandatory=True)
    edge_threshold = traits.Any(mandatory=True)
    prune = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    target_samples = traits.Any(mandatory=True)
    norm = traits.Any(mandatory=True)
    binary = traits.Bool(mandatory=True)
    track_type = traits.Any(mandatory=True)
    directget = traits.Any(mandatory=True)
    min_length = traits.Any(mandatory=True)
    error_margin = traits.Any(mandatory=True)


class _PlotStructOutputSpec(BaseInterfaceInputSpec):
    """Output interface wrapper for PlotStruct"""

    out = traits.Str


class PlotStruct(SimpleInterface):
    """Interface wrapper for PlotStruct"""

    input_spec = _PlotStructInputSpec
    output_spec = _PlotStructOutputSpec

    def _run_interface(self, runtime):
        from pynets.plotting import plot_gen

        if isinstance(self.inputs.conn_matrix, str):
            self.inputs.conn_matrix = np.load(self.inputs.conn_matrix)

        assert (
            len(self.inputs.coords)
            == len(self.inputs.labels)
            == self.inputs.conn_matrix.shape[0]
        )
        if self.inputs.coords.ndim == 1:
            print("Only 1 node detected. Plotting is not applicable...")
        else:
            plot_gen.plot_all_struct(
                self.inputs.conn_matrix,
                self.inputs.conn_model,
                self.inputs.atlas,
                self.inputs.dir_path,
                self.inputs.ID,
                self.inputs.network,
                self.inputs.labels.tolist(),
                self.inputs.roi,
                [tuple(coord) for coord in self.inputs.coords.tolist()],
                self.inputs.thr,
                self.inputs.node_size,
                self.inputs.edge_threshold,
                self.inputs.prune,
                self.inputs.uatlas,
                self.inputs.target_samples,
                self.inputs.norm,
                self.inputs.binary,
                self.inputs.track_type,
                self.inputs.directget,
                self.inputs.min_length,
                self.inputs.error_margin
            )

        self._results["out"] = "None"

        return runtime


class _PlotFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotFunc"""

    conn_matrix = traits.Any()
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    labels = traits.Array(mandatory=True)
    roi = traits.Any(mandatory=True)
    coords = traits.Array(mandatory=True)
    thr = traits.Any(mandatory=True)
    node_size = traits.Any(mandatory=True)
    edge_threshold = traits.Any(mandatory=True)
    smooth = traits.Any(mandatory=True)
    prune = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    norm = traits.Any(mandatory=True)
    binary = traits.Bool(mandatory=True)
    hpass = traits.Any(mandatory=True)
    extract_strategy = traits.Any(mandatory=True)
    edge_color_override = traits.Bool(mandatory=True)


class _PlotFuncOutputSpec(BaseInterfaceInputSpec):
    """Output interface wrapper for PlotFunc"""

    out = traits.Str


class PlotFunc(SimpleInterface):
    """Interface wrapper for PlotFunc"""

    input_spec = _PlotFuncInputSpec
    output_spec = _PlotFuncOutputSpec

    def _run_interface(self, runtime):
        from pynets.plotting import plot_gen

        if isinstance(self.inputs.conn_matrix, str):
            self.inputs.conn_matrix = np.load(self.inputs.conn_matrix)

        assert (
            len(self.inputs.coords)
            == len(self.inputs.labels)
            == self.inputs.conn_matrix.shape[0]
        )

        if self.inputs.coords.ndim == 1:
            print("Only 1 node detected. Plotting not applicable...")
        else:
            plot_gen.plot_all_func(
                self.inputs.conn_matrix,
                self.inputs.conn_model,
                self.inputs.atlas,
                self.inputs.dir_path,
                self.inputs.ID,
                self.inputs.network,
                self.inputs.labels.tolist(),
                self.inputs.roi,
                [tuple(coord) for coord in self.inputs.coords.tolist()],
                self.inputs.thr,
                self.inputs.node_size,
                self.inputs.edge_threshold,
                self.inputs.smooth,
                self.inputs.prune,
                self.inputs.uatlas,
                self.inputs.norm,
                self.inputs.binary,
                self.inputs.hpass,
                self.inputs.extract_strategy,
                self.inputs.edge_color_override,
            )

        self._results["out"] = "None"

        return runtime


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
        from pynets.registration.utils import check_orient_and_dims

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
        ]

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
                mask_tmp_path = check_orient_and_dims(
                    mask_tmp_path, runtime.cwd, self.inputs.vox_size
                )
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-GM_probseg.nii.gz")
        if len(gm_mask_existing) > 0:
            gm_mask = fname_presuffix(gm_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                gm_mask_existing[0],
                gm_mask,
                copy=True,
                use_hardlink=False)
            gm_mask = check_orient_and_dims(
                gm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            gm_mask = None

        wm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-WM_probseg.nii.gz")
        if len(wm_mask_existing) > 0:
            wm_mask = fname_presuffix(wm_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                wm_mask_existing[0],
                wm_mask,
                copy=True,
                use_hardlink=False)
            wm_mask = check_orient_and_dims(
                wm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            wm_mask = None

        csf_mask_existing = glob.glob(
            self.inputs.in_dir + "/*_label-CSF_probseg.nii.gz"
        )
        if len(csf_mask_existing) > 0:
            csf_mask = fname_presuffix(
                csf_mask_existing[0], newpath=runtime.cwd)
            copyfile(
                csf_mask_existing[0],
                csf_mask,
                copy=True,
                use_hardlink=False)
            csf_mask = check_orient_and_dims(
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
        time.sleep(0.5)

        # Perform anatomical segmentation
        reg.gen_tissue(wm_mask, gm_mask, csf_mask, self.inputs.overwrite)
        time.sleep(0.5)

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

    atlas = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    uatlas_parcels = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    basedir_path = Directory(exists=True, mandatory=True)
    node_size = traits.Any(mandatory=True)
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
    node_size = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas_parcels = traits.Any(mandatory=False)
    uatlas = traits.Any(mandatory=False)
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
            "pynets", f"templates/{self.inputs.template_name}_brain_"
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

        if self.inputs.uatlas is None:
            uatlas_tmp_path = None
        else:
            uatlas_tmp_path = fname_presuffix(
                self.inputs.uatlas, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.uatlas,
                uatlas_tmp_path,
                copy=True,
                use_hardlink=False)

        if self.inputs.uatlas_parcels is None:
            uatlas_parcels_tmp_path = None
        else:
            uatlas_parcels_tmp_path = fname_presuffix(
                self.inputs.uatlas_parcels, suffix="_tmp", newpath=runtime.cwd
            )
            copyfile(
                self.inputs.uatlas_parcels,
                uatlas_parcels_tmp_path,
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

        if self.inputs.network or self.inputs.waymask:
            if self.inputs.waymask:
                wm_suf = os.path.basename(self.inputs.waymask).split('.nii')[0]
                atlas_name = f"{self.inputs.atlas}_" \
                             f"{wm_suf}"
            else:
                atlas_name = f"{self.inputs.atlas}_{self.inputs.network}"
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

        if self.inputs.node_size is not None:
            atlas_name = f"{atlas_name}{'_'}{self.inputs.node_size}"

        # Apply warps/coregister atlas to dwi
        [
            dwi_aligned_atlas_wmgm_int,
            dwi_aligned_atlas,
            aligned_atlas_t1w,
        ] = regutils.atlas2t1w2dwi_align(
            uatlas_tmp_path,
            uatlas_parcels_tmp_path,
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

        if self.inputs.uatlas is None:
            uatlas_out = self.inputs.uatlas_parcels
            copyfile(
                dwi_aligned_atlas,
                f"{os.path.dirname(uatlas_out)}/"
                f"{os.path.basename(dwi_aligned_atlas)}",
                copy=True,
                use_hardlink=False,
            )
        else:
            uatlas_out = self.inputs.uatlas
            copyfile(
                dwi_aligned_atlas,
                f"{os.path.dirname(uatlas_out)}/parcellations/"
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
        self._results["node_size"] = self.inputs.node_size
        self._results["atlas"] = self.inputs.atlas
        self._results["uatlas_parcels"] = uatlas_parcels_tmp_path
        self._results["uatlas"] = uatlas_out
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
            uatlas_tmp_path,
            mni2t1w_warp_tmp_path,
            mni2t1_xfm_tmp_path,
            t1w_brain_mask_tmp_path,
            t1_aligned_mni_tmp_path,
            t1w2dwi_bbr_xfm_tmp_path,
            t1w2dwi_xfm_tmp_path,
            wm_gm_int_in_dwi_tmp_path,
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
            "pynets", f"templates/{self.inputs.template_name}_brain_"
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
        from pynets.registration.utils import check_orient_and_dims

        anat_mask_existing = [
            i
            for i in glob.glob(self.inputs.in_dir +
                               "/*_desc-brain_mask.nii.gz")
            if "MNI" not in i
        ]

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
                mask_tmp_path = check_orient_and_dims(
                    mask_tmp_path, runtime.cwd, self.inputs.vox_size
                )
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-GM_probseg.nii.gz")
        if len(gm_mask_existing) > 0:
            gm_mask = fname_presuffix(gm_mask_existing[0],
                                      newpath=runtime.cwd)
            copyfile(
                gm_mask_existing[0],
                gm_mask,
                copy=True,
                use_hardlink=False)
            gm_mask = check_orient_and_dims(
                gm_mask, runtime.cwd, self.inputs.vox_size
            )
        else:
            gm_mask = None

        wm_mask_existing = glob.glob(
            self.inputs.in_dir +
            "/*_label-WM_probseg.nii.gz")
        if len(wm_mask_existing) > 0:
            wm_mask = fname_presuffix(wm_mask_existing[0],
                                      newpath=runtime.cwd)
            copyfile(
                wm_mask_existing[0],
                wm_mask,
                copy=True,
                use_hardlink=False)
            wm_mask = check_orient_and_dims(
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
        time.sleep(0.5)

        # Perform anatomical segmentation
        reg.gen_tissue(wm_mask, gm_mask, self.inputs.overwrite)
        time.sleep(0.5)

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

    uatlas = traits.Any(mandatory=True)
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
            "pynets", f"templates/{self.inputs.template_name}_brain_"
                      f"{self.inputs.vox_size}.nii.gz"
        )
        template_mask = pkg_resources.resource_filename(
            "pynets", f"templates/{self.inputs.template_name}_"
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

        uatlas_tmp_path = fname_presuffix(
            self.inputs.uatlas, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.uatlas,
            uatlas_tmp_path,
            copy=True,
            use_hardlink=False)

        atlas_name = prune_suffices(os.path.basename(self.inputs.uatlas
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
            uatlas_tmp_path,
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
            uatlas_tmp_path,
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

    atlas = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    uatlas_parcels = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
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
    node_size = traits.Any(mandatory=True)
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
    node_size = traits.Any()
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
        from nilearn.image import new_img_like

        if self.inputs.network:
            atlas_name = f"{self.inputs.atlas}_{self.inputs.network}"
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
            if self.inputs.uatlas is None:
                uatlas_tmp_path = None
            else:
                uatlas_tmp_path = fname_presuffix(
                    self.inputs.uatlas, suffix="_tmp", newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.uatlas,
                    uatlas_tmp_path,
                    copy=True,
                    use_hardlink=False)

            if self.inputs.uatlas_parcels is None:
                uatlas_parcels_tmp_path = None
            else:
                uatlas_parcels_tmp_path = fname_presuffix(
                    self.inputs.uatlas_parcels, suffix="_tmp",
                    newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.uatlas_parcels,
                    uatlas_parcels_tmp_path,
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

            if self.inputs.node_size is not None:
                atlas_name = f"{atlas_name}{'_'}{self.inputs.node_size}"

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
                uatlas_tmp_path,
                uatlas_parcels_tmp_path,
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
                uatlas_parcels_tmp_path,
                uatlas_tmp_path,
                gm_mask_tmp_path,
                t1_aligned_mni_tmp_path,
                t1w_brain_mask_tmp_path,
                mni2t1w_warp_tmp_path,
                mni2t1_xfm_tmp_path,
            ]

            if self.inputs.uatlas is None:
                uatlas_out = self.inputs.uatlas_parcels
                copyfile(
                    aligned_atlas_gm,
                    f"{os.path.dirname(uatlas_out)}/"
                    f"{os.path.basename(aligned_atlas_gm)}",
                    copy=True,
                    use_hardlink=False,
                )
            else:
                uatlas_out = self.inputs.uatlas
                copyfile(
                    aligned_atlas_gm,
                    f"{os.path.dirname(uatlas_out)}/parcellations/"
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
        self._results["node_size"] = self.inputs.node_size

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
            "pynets", f"templates/{self.inputs.template_name}_brain_"
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


class _TrackingInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for Tracking"""

    B0_mask = File(exists=True, mandatory=True)
    gm_in_dwi = File(exists=True, mandatory=True)
    vent_csf_in_dwi = File(exists=True, mandatory=True)
    wm_in_dwi = File(exists=True, mandatory=True)
    tiss_class = traits.Str(mandatory=True)
    labels_im_file_wm_gm_int = File(exists=True, mandatory=True)
    labels_im_file = File(exists=True, mandatory=True)
    target_samples = traits.Any(mandatory=True)
    curv_thr_list = traits.List(mandatory=True)
    step_list = traits.List(mandatory=True)
    track_type = traits.Str(mandatory=True)
    min_length = traits.Any(mandatory=True)
    maxcrossing = traits.Any(mandatory=True)
    directget = traits.Str(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    network = traits.Any(mandatory=False)
    node_size = traits.Any(mandatory=True)
    dens_thresh = traits.Bool(mandatory=True)
    ID = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=False)
    min_span_tree = traits.Bool(mandatory=True)
    disp_filt = traits.Bool(mandatory=True)
    parc = traits.Bool(mandatory=True)
    prune = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=False)
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
    target_samples = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    network = traits.Any(mandatory=False)
    node_size = traits.Any()
    dens_thresh = traits.Bool()
    ID = traits.Any(mandatory=True)
    roi = traits.Any(mandatory=False)
    min_span_tree = traits.Bool()
    disp_filt = traits.Bool()
    parc = traits.Bool()
    prune = traits.Any()
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    norm = traits.Any()
    binary = traits.Bool(False, usedefault=True)
    atlas_t1w = File(exists=True, mandatory=True)
    curv_thr_list = traits.List(mandatory=True)
    step_list = traits.List(mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    dm_path = traits.Any()
    directget = traits.Str(mandatory=True)
    labels_im_file = File(exists=True, mandatory=True)
    min_length = traits.Any()


class Tracking(SimpleInterface):
    """Interface wrapper for Tracking"""

    input_spec = _TrackingInputSpec
    output_spec = _TrackingOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import sys
        import time
        import os.path as op
        from dipy.io import load_pickle
        from colorama import Fore, Style
        from dipy.data import get_sphere
        from pynets.core import utils
        from pynets.core.utils import load_runconfig
        from pynets.dmri.track import (
            reconstruction,
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
            % (self.inputs.network + "_" if self.inputs.network is not None
               else ""),
            "%s"
            % (
                op.basename(self.inputs.roi).split(".")[0] + "_"
                if self.inputs.roi is not None
                else ""
            ),
            self.inputs.conn_model,
            "_",
            self.inputs.target_samples,
            "_",
            "%s"
            % (
                "%s%s" % (self.inputs.node_size, "mm_")
                if (
                    (self.inputs.node_size != "parc")
                    and (self.inputs.node_size is not None)
                )
                else "parc_"
            ),
            "curv-",
            str(self.inputs.curv_thr_list).replace(", ", "_"),
            "_step-",
            str(self.inputs.step_list).replace(", ", "_"),
            "_directget-",
            self.inputs.directget,
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
                    self.inputs.target_samples,
                    self.inputs.node_size,
                    self.inputs.curv_thr_list,
                    self.inputs.step_list,
                    self.inputs.network,
                    self.inputs.roi,
                    self.inputs.directget,
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
                % (self.inputs.network + "_" if self.inputs.network is not None
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
                    "%s%s" % (self.inputs.node_size, "mm")
                    if (
                        (self.inputs.node_size != "parc")
                        and (self.inputs.node_size is not None)
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
                model, _ = reconstruction(
                    self.inputs.conn_model,
                    gtab,
                    dwi_data,
                    B0_mask_tmp_path,
                )
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
                model, _ = reconstruction(
                    self.inputs.conn_model,
                    gtab,
                    dwi_data,
                    B0_mask_tmp_path,
                )
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
                f"{Fore.GREEN}Target number of cumulative streamlines: "
                f"{Fore.BLUE} "
                f"{self.inputs.target_samples}"
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
            if self.inputs.directget == "prob":
                print(f"{Fore.GREEN}Direction-getting type: {Fore.BLUE}"
                      f"Probabilistic")
            elif self.inputs.directget == "clos":
                print(f"{Fore.GREEN}Direction-getting type: "
                      f"{Fore.BLUE}Closest Peak")
            elif self.inputs.directget == "det":
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
                    self.inputs.target_samples,
                    labels_im_file_tmp_path_wm_gm_int,
                    labels_im_file_tmp_path,
                    recon_path,
                    get_sphere(sphere),
                    self.inputs.directget,
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
                    self.inputs.tiss_class,
                    runtime.cwd
                )
                gc.collect()
            except BaseException:
                print(UserWarning("Tractography failed..."))
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
                        self.inputs.target_samples,
                        self.inputs.node_size,
                        self.inputs.curv_thr_list,
                        self.inputs.step_list,
                        self.inputs.network,
                        self.inputs.roi,
                        self.inputs.directget,
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
                         labels_im_file_tmp_path_wm_gm_int,
                         wm_in_dwi_tmp_path, gm_in_dwi_tmp_path,
                         vent_csf_in_dwi_tmp_path, t1w2dwi_tmp_path]

            for j in tmp_files:
                if j is not None:
                    if os.path.isfile(j):
                        os.system(f"rm -f {j} &")

        self._results["track_type"] = self.inputs.track_type
        self._results["target_samples"] = self.inputs.target_samples
        self._results["conn_model"] = self.inputs.conn_model
        self._results["dir_path"] = dir_path
        self._results["network"] = self.inputs.network
        self._results["node_size"] = self.inputs.node_size
        self._results["dens_thresh"] = self.inputs.dens_thresh
        self._results["ID"] = self.inputs.ID
        self._results["roi"] = self.inputs.roi
        self._results["min_span_tree"] = self.inputs.min_span_tree
        self._results["disp_filt"] = self.inputs.disp_filt
        self._results["parc"] = self.inputs.parc
        self._results["prune"] = self.inputs.prune
        self._results["atlas"] = self.inputs.atlas
        self._results["uatlas"] = self.inputs.uatlas
        self._results["labels"] = self.inputs.labels
        self._results["coords"] = self.inputs.coords
        self._results["norm"] = self.inputs.norm
        self._results["binary"] = self.inputs.binary
        self._results["atlas_t1w"] = self.inputs.atlas_t1w
        self._results["curv_thr_list"] = self.inputs.curv_thr_list
        self._results["step_list"] = self.inputs.step_list
        self._results["fa_path"] = fa_file_tmp_path
        self._results["directget"] = self.inputs.directget
        self._results["labels_im_file"] = labels_im_file_tmp_path
        self._results["min_length"] = self.inputs.min_length

        tmp_files = [B0_mask_tmp_path, dwi_file_tmp_path]

        for j in tmp_files:
            if j is not None:
                if os.path.isfile(j):
                    os.system(f"rm -f {j} &")

        # Exercise caution when deleting copied recon_path
        if recon_path is not None:
            if os.path.isfile(recon_path):
                os.remove(recon_path)

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
