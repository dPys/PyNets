#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import warnings
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    Directory,
)

warnings.filterwarnings("ignore")


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

    parcellation = File(exists=True)
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
        from pynets.fmri import clustering
        from pynets.registration.utils import orient_reslice
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
        nthreads = hardcoded_params["omp_threads"][0]

        clust_list = ["kmeans", "ward", "complete", "average", "ncut", "rena"]

        clust_mask_temp_path = orient_reslice(
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

        nip = clustering.NiParcellate(
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
                    from pynets.fmri.clustering import parcellate
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
                consensus_parcellation = clustering.ensemble_parcellate(
                    boot_parcellations,
                    int(self.inputs.k)
                )
                nib.save(consensus_parcellation, nip.parcellation)
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
                parcellation = clustering.parcellate(func_img,
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
        while not os.path.isfile(nip.parcellation) and ix < 60:
            print('Waiting for clustered parcellation...')
            time.sleep(1)
            ix += 1

        if not os.path.isfile(nip.parcellation):
            raise FileNotFoundError(f"Parcellation clustering failed for"
                                    f" {nip.parcellation}")

        self._results["atlas"] = atlas
        self._results["parcellation"] = nip.parcellation
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
    subnet = traits.Any(mandatory=False)
    smooth = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    hpass = traits.Any(mandatory=True)
    mask = traits.Any(mandatory=False)
    parc = traits.Bool()
    node_radius = traits.Any(mandatory=False)
    net_parcels_nii_path = traits.Any(mandatory=False)
    extract_strategy = traits.Str("mean", mandatory=False, usedefault=True)


class _ExtractTimeseriesOutputSpec(TraitedSpec):
    """Output interface wrapper for ExtractTimeseries"""

    ts_within_nodes = traits.Any(mandatory=True)
    node_radius = traits.Any(mandatory=True)
    smooth = traits.Any(mandatory=True)
    dir_path = Directory(exists=True, mandatory=True)
    atlas = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
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
            node_radius=self.inputs.node_radius,
            conf=out_name_conf,
            func_file=out_name_func_file,
            roi=self.inputs.roi,
            dir_path=self.inputs.dir_path,
            ID=self.inputs.ID,
            subnet=self.inputs.subnet,
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
        self._results["node_radius"] = te.node_radius
        self._results["smooth"] = te.smooth
        self._results["extract_strategy"] = te.extract_strategy
        self._results["dir_path"] = te.dir_path
        self._results["atlas"] = self.inputs.atlas
        self._results["parcellation"] = self.inputs.parcellation
        self._results["labels"] = self.inputs.labels
        self._results["coords"] = self.inputs.coords
        self._results["hpass"] = te.hpass
        self._results["roi"] = self.inputs.roi

        del te
        gc.collect()

        return runtime
