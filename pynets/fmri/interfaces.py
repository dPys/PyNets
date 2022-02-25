#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import matplotlib
import numpy as np
import warnings
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


class TimeseriesExtraction(object):
    """
    Class for implementing various time-series extracting routines.
    """

    __slots__ = ('net_parcels_nii_path', 'node_radius', 'conf', 'func_file',
                 'roi', 'dir_path', 'ID', 'subnet', 'smooth', 'mask',
                 'hpass', 'signal', 'ts_within_nodes',
                 '_mask_img', '_mask_path', '_func_img', '_t_r',
                 '_detrending', '_net_parcels_nii_temp_path',
                 '_net_parcels_map_nifti', '_parcel_masker', 'low_pass')

    def __init__(
        self,
        net_parcels_nii_path,
        node_radius,
        conf,
        func_file,
        roi,
        dir_path,
        ID,
        subnet,
        smooth,
        hpass,
        mask,
        signal,
    ):
        self.net_parcels_nii_path = net_parcels_nii_path
        self.node_radius = node_radius
        self.conf = conf
        self.func_file = func_file
        self.roi = roi
        self.dir_path = dir_path
        self.ID = ID
        self.subnet = subnet
        self.smooth = smooth
        self.mask = mask
        self.hpass = hpass
        self.signal = signal
        self.ts_within_nodes = None
        self._mask_img = None
        self._mask_path = None
        self._func_img = None
        self._t_r = None
        self._detrending = True
        self._net_parcels_nii_temp_path = None
        self._net_parcels_map_nifti = None
        self._parcel_masker = None

        from pynets.core.utils import load_runconfig
        hardcoded_params = load_runconfig()
        try:
            self.low_pass = hardcoded_params["low_pass"][0]
        except KeyError as e:
            print(e,
                  "ERROR: Plotting configuration not successfully "
                  "extracted from advanced.yaml"
                  )

    def prepare_inputs(self, num_std_dev=1.5):
        """Helper function to creating temporary nii's and prepare inputs from
         time-series extraction"""
        import os.path as op
        import nibabel as nib
        from nilearn.image import math_img, index_img, resample_to_img
        from nilearn.masking import intersect_masks

        if not op.isfile(self.func_file):
            raise FileNotFoundError(
                "\nFunctional data input not found! Check that the"
                " file(s) specified with the -i "
                "flag exist(s)")

        if self.conf:
            if not op.isfile(self.conf):
                raise FileNotFoundError(
                    "\nConfound regressor file not found! Check "
                    "that the file(s) specified with the -conf flag "
                    "exist(s)")

        self._func_img = nib.load(self.func_file)
        self._func_img.set_data_dtype(np.float32)

        func_vol_img = index_img(self._func_img, 1)
        func_vol_img.set_data_dtype(np.uint16)
        func_data = np.asarray(func_vol_img.dataobj, dtype=np.float32)
        func_int_thr = np.round(
            np.mean(func_data[func_data > 0])
            - np.std(func_data[func_data > 0]) * num_std_dev,
            3,
        )
        hdr = self._func_img.header

        self._net_parcels_map_nifti = nib.load(self.net_parcels_nii_path,
                                               mmap=True)
        self._net_parcels_map_nifti.set_data_dtype(np.int16)

        if self.hpass:
            if len(hdr.get_zooms()) == 4:
                self._t_r = float(hdr.get_zooms()[-1])
            else:
                self._t_r = None
        else:
            self._t_r = None

        if self.hpass is not None:
            if float(self.hpass) > 0:
                self.hpass = float(self.hpass)
                self._detrending = False
            else:
                self.hpass = None
                self._detrending = True
        else:
            self.hpass = None
            self._detrending = True

        if self.mask is not None:
            # Ensure mask is binary and contains only voxels that also
            # overlap with the parcellation and first functional volume
            self._mask_img = intersect_masks(
                [
                    math_img(f"img > {func_int_thr}", img=func_vol_img),
                    math_img("img > 0.0001",
                             img=resample_to_img(nib.load(self.mask),
                                                 func_vol_img))
                ],
                threshold=1,
                connected=False,
            )
            self._mask_img.set_data_dtype(np.uint16)
        else:
            print("Warning: Proceeding to extract time-series without a "
                  "brain mask...")
            self._mask_img = None

        if self.smooth:
            if float(self.smooth) > 0:
                print(f"Smoothing FWHM: {self.smooth} mm\n")

        if self.hpass:
            print(f"Applying high-pass filter: {self.hpass} Hz\n")

        return

    def extract_ts_parc(self):
        """
        API for employing Nilearn's NiftiLabelsMasker to extract fMRI
        time-series data from spherical ROI's based on a given 3D atlas image
        of integer-based voxel intensities. The resulting time-series can then
        optionally be resampled using circular-block bootrapping. The final 2D
        m x n array is ultimately saved to file in .npy format.
        """
        import pandas as pd
        from nilearn import input_data
        from pynets.fmri.estimation import fill_confound_nans

        self._parcel_masker = input_data.NiftiLabelsMasker(
            labels_img=self._net_parcels_map_nifti,
            background_label=0,
            standardize=True,
            smoothing_fwhm=float(self.smooth),
            low_pass=self.low_pass,
            high_pass=self.hpass,
            detrend=self._detrending,
            t_r=self._t_r,
            verbose=2,
            resampling_target="labels",
            dtype="auto",
            mask_img=self._mask_img,
            strategy=self.signal
        )

        if self.conf is not None:
            import os

            confounds = pd.read_csv(self.conf, sep="\t")

            cols = [i for i in confounds.columns if 'motion_outlier' in i
                    or i == 'framewise_displacement'
                    or i == 'white_matter' or i == 'csf'
                    or i == 'std_dvars' or i == 'rot_z'
                    or i == 'rot_y' or i == 'rot_x' or i == 'trans_z'
                    or i == 'trans_y' or i == 'trans_x'
                    or 'non_steady_state_outlier' in i]

            if len(confounds.index) == self._func_img.shape[-1]:
                if confounds.isnull().values.any():
                    conf_corr = fill_confound_nans(confounds, self.dir_path)
                    conf_corr_df = pd.read_csv(conf_corr, sep="\t")
                    cols = [i for i in cols if i in conf_corr_df.columns]
                    self.ts_within_nodes = self._parcel_masker.fit_transform(
                        self._func_img.slicer[:,:,:,5:],
                        confounds=conf_corr_df.loc[5:][cols].values
                    )
                    os.remove(conf_corr)
                else:
                    self.ts_within_nodes = self._parcel_masker.fit_transform(
                        self._func_img.slicer[:,:,:,5:],
                        confounds=pd.read_csv(self.conf,
                                              sep="\t").loc[5:][cols].values
                    )
            else:
                from nilearn.image import high_variance_confounds
                print(f"Shape of confounds ({len(confounds.index)}) does not"
                      f" equal the number of volumes "
                      f"({self._func_img.shape[-1]}) in the time-series")
                self.ts_within_nodes = self._parcel_masker.fit_transform(
                    self._func_img.slicer[:,:,:,5:],
                    confounds=pd.DataFrame(
                        high_variance_confounds(
                            self._func_img,
                            percentile=1)).loc[5:].values)
        else:
            from nilearn.image import high_variance_confounds
            self.ts_within_nodes = self._parcel_masker.fit_transform(
                self._func_img.slicer[:,:,:,5:],
                confounds=pd.DataFrame(
                    high_variance_confounds(self._func_img,
                                            percentile=1)).loc[5:].values)

        self._func_img.uncache()

        if self.ts_within_nodes is None:
            raise RuntimeError("\nTime-series extraction failed!")

        else:
            self.node_radius = "parc"

        return

    def save_and_cleanup(self):
        """Save the extracted time-series and clean cache"""
        import gc
        from pynets.core import utils

        # Save time series as file
        utils.save_ts_to_file(
            self.roi,
            self.subnet,
            self.ID,
            self.dir_path,
            self.ts_within_nodes,
            self.smooth,
            self.hpass,
            self.node_radius,
            self.signal,
        )

        if self._mask_path is not None:
            self._mask_img.uncache()

        if self._parcel_masker is not None:
            del self._parcel_masker
            self._net_parcels_map_nifti.uncache()
        gc.collect()
        return


class NiParcellate(object):
    """
    Class for implementing various clustering routines.
    """
    __slots__ = ('func_file', 'clust_mask', 'k', 'clust_type', 'conf',
                 'local_corr', 'parcellation', 'atlas', '_detrending',
                 '_standardize', '_func_img', 'mask', '_mask_img',
                 '_local_conn_mat_path', '_dir_path', '_local_conn',
                 '_clust_mask_corr_img', '_func_img_data',
                 '_masked_fmri_vol', '_conn_comps', 'num_conn_comps', 'outdir')

    def __init__(
        self,
        func_file,
        clust_mask,
        k,
        clust_type,
        local_corr,
        outdir,
        conf=None,
        mask=None,
    ):
        """
        Parameters
        ----------
        func_file : str
            File path to a 4D Nifti1Image containing fMRI data.
        clust_mask : str
            File path to a 3D NIFTI file containing a mask, which restricts the
            voxels used in the clustering.
        k : int
            Numbers of clusters that will be generated.
        clust_type : str
            Type of clustering to be performed (e.g. 'ward', 'kmeans',
            'complete', 'average').
        local_corr : str
            Type of local connectivity to use as the basis for clustering
            methods. Options are tcorr or scorr. Default is tcorr.
        outdir : str
            Path to base derivatives directory.
        conf : str
            File path to a confound regressor file for reduce noise in the
            time-series when extracting from ROI's.
        mask : str
            File path to a 3D NIFTI file containing a mask, which restricts the
            voxels used in the analysis.

        References
        ----------
        .. [1] Thirion, B., Varoquaux, G., Dohmatob, E., & Poline, J. B.
          (2014). Which fMRI clustering gives good brain parcellations?
          Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2014.00167
        .. [2] Bellec, P., Rosa-Neto, P., Lyttelton, O. C., Benali, H., &
          Evans, A. C. (2010). Multi-level bootstrap analysis of stable
          clusters in resting-state fMRI. NeuroImage.
          https://doi.org/10.1016/j.neuroimage.2010.02.082
        .. [3] Garcia-Garcia, M., Nikolaidis, A., Bellec, P.,
          Craddock, R. C., Cheung, B., Castellanos, F. X., & Milham, M. P.
          (2018). Detecting stable individual differences in the functional
          organization of the human basal ganglia. NeuroImage.
          https://doi.org/10.1016/j.neuroimage.2017.07.029

        """
        self.func_file = func_file
        self.clust_mask = clust_mask
        self.k = int(k)
        self.clust_type = clust_type
        self.conf = conf
        self.local_corr = local_corr
        self.parcellation = None
        self.atlas = None
        self._detrending = True
        self._standardize = True
        self._func_img = nib.load(self.func_file)
        self.mask = mask
        self._mask_img = None
        self._local_conn_mat_path = None
        self._dir_path = None
        _clust_est = None
        self._local_conn = None
        self._clust_mask_corr_img = None
        self._func_img_data = None
        self._masked_fmri_vol = None
        self._conn_comps = None
        self.num_conn_comps = None
        self.outdir = outdir

    def create_clean_mask(self, num_std_dev=1.5):
        """
        Create a subject-refined version of the clustering mask.
        """
        import os
        from pynets.core import utils
        from nilearn.masking import intersect_masks
        from nilearn.image import index_img, math_img, resample_img

        mask_name = os.path.basename(self.clust_mask).split(".nii")[0]
        self.atlas = f"{mask_name}{'_'}{self.clust_type}{'_k'}{str(self.k)}"
        print(
            f"\nCreating atlas using {self.clust_type} at cluster level"
            f" {str(self.k)} for {str(self.atlas)}...\n"
        )
        self._dir_path = utils.do_dir_path(self.atlas, self.outdir)
        self.parcellation = f"{self._dir_path}/{mask_name}_" \
                            f"clust-{self.clust_type}" \
                            f"_k{str(self.k)}.nii.gz"

        # Load clustering mask
        self._func_img.set_data_dtype(np.float32)
        func_vol_img = index_img(self._func_img, 1)
        func_vol_img.set_data_dtype(np.uint16)
        clust_mask_res_img = resample_img(
            nib.load(self.clust_mask),
            target_affine=func_vol_img.affine,
            target_shape=func_vol_img.shape,
            interpolation="nearest",
        )
        clust_mask_res_img.set_data_dtype(np.uint16)
        func_data = np.asarray(func_vol_img.dataobj, dtype=np.float32)
        func_int_thr = np.round(
            np.mean(func_data[func_data > 0])
            - np.std(func_data[func_data > 0]) * num_std_dev,
            3,
        )
        if self.mask is not None:
            self._mask_img = nib.load(self.mask)
            self._mask_img.set_data_dtype(np.uint16)
            mask_res_img = resample_img(
                self._mask_img,
                target_affine=func_vol_img.affine,
                target_shape=func_vol_img.shape,
                interpolation="nearest",
            )
            mask_res_img.set_data_dtype(np.uint16)
            self._clust_mask_corr_img = intersect_masks(
                [
                    math_img(f"img > {func_int_thr}", img=func_vol_img),
                    math_img("img > 0.01", img=clust_mask_res_img),
                    math_img("img > 0.01", img=mask_res_img),
                ],
                threshold=1,
                connected=False,
            )
            self._clust_mask_corr_img.set_data_dtype(np.uint16)
            self._mask_img.uncache()
            mask_res_img.uncache()
        else:
            self._clust_mask_corr_img = intersect_masks(
                [
                    math_img("img > " + str(func_int_thr), img=func_vol_img),
                    math_img("img > 0.01", img=clust_mask_res_img),
                ],
                threshold=1,
                connected=False,
            )
            self._clust_mask_corr_img.set_data_dtype(np.uint16)
        nib.save(self._clust_mask_corr_img,
                 f"{self._dir_path}{'/'}{mask_name}{'.nii.gz'}")

        del func_data
        func_vol_img.uncache()
        clust_mask_res_img.uncache()

        return self.atlas

    def create_local_clustering(self, overwrite, r_thresh, min_region_size=80):
        """
        API for performing any of a variety of clustering routines available
         through NiLearn.
        """
        import os.path as op
        from scipy.sparse import save_npz, load_npz
        from nilearn.regions import connected_regions

        try:
            conn_comps = connected_regions(
                self._clust_mask_corr_img,
                extract_type="connected_components",
                min_region_size=min_region_size,
            )
            self._conn_comps = conn_comps[0]
            self.num_conn_comps = len(conn_comps[1])
        except BaseException:
            raise ValueError("Clustering mask is empty!")

        if not self._conn_comps:
            if np.sum(np.asarray(self._clust_mask_corr_img.dataobj)) == 0:
                raise ValueError("Clustering mask is empty!")
            else:
                self._conn_comps = self._clust_mask_corr_img
                self.num_conn_comps = 1
        print(
            f"Detected {self.num_conn_comps} connected components in "
            f"clustering mask with a mininimum region "
            f"size of {min_region_size}")
        if (
            self.clust_type == "complete"
            or self.clust_type == "average"
            or self.clust_type == "single"
        ):
            if self.num_conn_comps > 1:
                raise ValueError(
                    "Clustering method unstable with spatial constrainsts "
                    "applied to multiple connected components.")

        if (
            self.clust_type == "ward" and self.num_conn_comps > 1
        ) or self.clust_type == "ncut":
            if self.k < self.num_conn_comps:
                raise ValueError(
                    "k must minimally be greater than the total number of "
                    "connected components in "
                    "the mask in the case of agglomerative clustering.")

            if self.local_corr == "tcorr" or self.local_corr == "scorr":
                self._local_conn_mat_path = (
                    f"{self.parcellation.split('.nii')[0]}_"
                    f"{self.local_corr}_conn.npz"
                )

                if (not op.isfile(self._local_conn_mat_path)) or (
                        overwrite is True):
                    from pynets.fmri.clustering import (
                        make_local_connectivity_tcorr,
                        make_local_connectivity_scorr,
                    )

                    if self.local_corr == "tcorr":
                        self._local_conn = make_local_connectivity_tcorr(
                            self._func_img, self._clust_mask_corr_img,
                            thresh=r_thresh)
                    elif self.local_corr == "scorr":
                        self._local_conn = make_local_connectivity_scorr(
                            self._func_img, self._clust_mask_corr_img,
                            thresh=r_thresh)
                    else:
                        raise ValueError(
                            "Local connectivity type not available")
                    print(
                        f"Saving spatially constrained connectivity structure"
                        f" to: {self._local_conn_mat_path}"
                    )
                    save_npz(self._local_conn_mat_path, self._local_conn)
                elif op.isfile(self._local_conn_mat_path):
                    self._local_conn = load_npz(self._local_conn_mat_path)
            elif self.local_corr == "allcorr":
                if self.clust_type == "ncut":
                    raise ValueError(
                        "Must select either `tcorr` or `scorr` local "
                        "connectivity option if you are using "
                        "`ncut` clustering method")

                self._local_conn = "auto"
            else:
                raise ValueError(
                    "Local connectivity method not recognized. Only tcorr,"
                    " scorr, and auto are currently "
                    "supported")
        else:
            self._local_conn = "auto"
        return

    def prep_boot(self, blocklength=1):
        from nilearn.masking import apply_mask

        ts_data = apply_mask(self._func_img, self._clust_mask_corr_img)
        return ts_data, int(int(np.sqrt(ts_data.shape[0])) * blocklength)


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
    signal = traits.Str("mean", mandatory=False, usedefault=True)


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
    signal = traits.Any(mandatory=False)


class ExtractTimeseries(SimpleInterface):
    """Interface wrapper for ExtractTimeseries"""

    input_spec = _ExtractTimeseriesInputSpec
    output_spec = _ExtractTimeseriesOutputSpec

    def _run_interface(self, runtime):
        import gc
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.fmri.interfaces import TimeseriesExtraction
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

        te = TimeseriesExtraction(
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
            signal=self.inputs.signal,
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
        self._results["signal"] = te.signal
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
