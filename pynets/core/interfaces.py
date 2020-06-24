#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
import indexed_gzip
import nibabel as nib
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface,
                                    Directory)
warnings.filterwarnings("ignore")


class _FetchNodesLabelsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for FetchNodesLabels"""
    atlas = traits.Any()
    uatlas = traits.Any()
    ref_txt = traits.Any()
    parc = traits.Bool()
    in_file = File(exists=True, mandatory=True)
    use_AAL_naming = traits.Bool(False, usedefault=True)
    outdir = traits.Str(mandatory=True)
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
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
        from pynets.core import utils, nodemaker
        import pandas as pd
        import time
        from pathlib import Path
        import os.path as op
        import glob

        base_path = utils.get_file()
        # Test if atlas is a nilearn atlas. If so, fetch coords, labels, and/or networks.
        nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009', 'atlas_talairach_gyrus',
                                'atlas_talairach_ba', 'atlas_talairach_lobe']
        nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
        nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
        local_atlases = [op.basename(i).split('.nii')[0] for i in
                         glob.glob(f"{str(Path(base_path).parent)}{'/atlases/*.nii.gz'}") if '_4d' not in i]

        if self.inputs.uatlas is None and self.inputs.atlas in nilearn_parc_atlases:
            [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(self.inputs.atlas, self.inputs.parc)
            if uatlas:
                if not isinstance(uatlas, str):
                    nib.save(uatlas, f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}")
                    uatlas = f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}"
                uatlas = nodemaker.enforce_consecutive_labels(uatlas)
                [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
            else:
                raise ValueError(f"\nERROR: Atlas file for {self.inputs.atlas} not found!")
            atlas = self.inputs.atlas
        elif self.inputs.uatlas is None and self.inputs.parc is False and self.inputs.atlas in nilearn_coords_atlases:
            print('Fetching coords and labels from nilearn coordinate-based atlas library...')
            # Fetch nilearn atlas coords
            [coords, _, networks_list, labels] = nodemaker.fetch_nilearn_atlas_coords(self.inputs.atlas)
            parcel_list = None
            par_max = None
            atlas = self.inputs.atlas
            uatlas = None
        elif self.inputs.uatlas is None and self.inputs.parc is False and self.inputs.atlas in nilearn_prob_atlases:
            from nilearn.plotting import find_probabilistic_atlas_cut_coords
            print('Fetching coords and labels from nilearn probabilistic atlas library...')
            # Fetch nilearn atlas coords
            [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(self.inputs.atlas, self.inputs.parc)
            coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas)
            if uatlas:
                if not isinstance(uatlas, str):
                    nib.save(uatlas, f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}")
                    uatlas = f"{runtime.cwd}{self.inputs.atlas}{'.nii.gz'}"
                uatlas = nodemaker.enforce_consecutive_labels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
            else:
                raise ValueError(f"\nAtlas file for {self.inputs.atlas} not found!")
            par_max = None
            atlas = self.inputs.atlas
        elif self.inputs.uatlas is None and self.inputs.atlas in local_atlases:
            from nipype.utils.filemanip import fname_presuffix, copyfile

            uatlas_pre = f"{str(Path(base_path).parent)}/atlases/{self.inputs.atlas}.nii.gz"
            uatlas = fname_presuffix(uatlas_pre, suffix='_tmp', newpath=runtime.cwd)
            copyfile(uatlas_pre, uatlas, copy=True, use_hardlink=False)
            try:
                uatlas = nodemaker.enforce_consecutive_labels(uatlas)
                # Fetch user-specified atlas coords
                [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None
                # Describe user atlas coords
                print(f"\n{self.inputs.atlas} comes with {par_max} parcels\n")
            except ValueError:
                print('Either you have specified the name of an atlas that does not exist in the nilearn or local '
                      'repository or you have not supplied a 3d atlas parcellation image!')
                parcel_list = None
                par_max = None
                coords = None
            labels = None
            networks_list = None
            atlas = self.inputs.atlas
        elif self.inputs.uatlas:
            if self.inputs.clustering is True:
                while True:
                    if op.isfile(self.inputs.uatlas):
                        break
                    else:
                        print('Waiting for atlas file...')
                        time.sleep(15)

            try:
                # Fetch user-specified atlas coords
                uatlas = nodemaker.enforce_consecutive_labels(self.inputs.uatlas)
                [coords, atlas, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(uatlas)
                else:
                    parcel_list = None

                atlas = utils.prune_suffices(atlas)
                uatlas = self.inputs.uatlas

                # Describe user atlas coords
                print(f"\n{atlas} comes with {par_max} parcels\n")
            except ValueError:
                print('Either you have specified the name of an atlas that does not exist in the nilearn or local '
                      'repository or you have not supplied a 3d atlas parcellation image!')
                parcel_list = None
                par_max = None
                coords = None
                atlas = None
                uatlas = None
            labels = None
            networks_list = None
        else:
            raise ValueError(
                'Either you have specified the name of an atlas that does not exist in the nilearn or local '
                'repository or you have not supplied a 3d atlas parcellation image!')

        # Labels prep
        if atlas and not labels:
            if (self.inputs.ref_txt is not None) and (op.exists(self.inputs.ref_txt)):
                labels = pd.read_csv(self.inputs.ref_txt, sep=" ",
                                     header=None, names=["Index", "Region"])['Region'].tolist()
            else:
                if atlas in local_atlases:
                    ref_txt = f"{str(Path(base_path).parent)}{'/labelcharts/'}{atlas}{'.txt'}"
                else:
                    ref_txt = self.inputs.ref_txt
                if op.exists(ref_txt):
                    try:
                        labels = pd.read_csv(ref_txt,
                                             sep=" ", header=None, names=["Index", "Region"])['Region'].tolist()
                    except:
                        if self.inputs.use_AAL_naming is True:
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
                                labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic index labels...')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                else:
                    if self.inputs.use_AAL_naming is True:
                        try:
                            labels = nodemaker.AAL_naming(coords)
                        except:
                            print('AAL reference labeling failed!')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic index labels...')
                        labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

        print(f"Labels:\n{labels}")
        dir_path = utils.do_dir_path(atlas, self.inputs.outdir)

        if len(coords) != len(labels):
            print('Length of coordinates is not equal to length of label names! Replacing with nan\'s instead...')
            labels = len(coords) * [np.nan]

        assert len(coords) == len(labels)

        self._results['labels'] = labels
        self._results['coords'] = coords
        self._results['atlas'] = atlas
        self._results['networks_list'] = networks_list
        self._results['parcel_list'] = parcel_list
        self._results['par_max'] = par_max
        self._results['uatlas'] = uatlas
        self._results['dir_path'] = dir_path

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
            self.inputs.binary)
        setattr(self, '_outpath', out)
        return runtime

    def _list_outputs(self):
        import os.path as op
        return {'out_path_neat': op.abspath(getattr(self, '_outpath'))}


class CombineOutputsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for CombineOutputs"""
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    net_mets_csv_list = traits.List(mandatory=True)
    plot_switch = traits.Bool(False, usedefault=True)
    multi_nets = traits.Any(mandatory=False)
    multimodal = traits.Bool(False, usedefault=True)


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
            self.inputs.multimodal)
        setattr(self, '_combination_complete', combination_complete)
        return runtime

    def _list_outputs(self):
        return {'combination_complete': getattr(self, '_combination_complete')}


class _IndividualClusteringInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for IndividualClustering"""
    func_file = File(exists=True, mandatory=True)
    conf = traits.Any(mandatory=False)
    clust_mask = File(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    local_corr = traits.Str('allcorr', mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    outdir = traits.Str(mandatory=True)


class _IndividualClusteringOutputSpec(TraitedSpec):
    """Output interface wrapper for IndividualClustering"""
    uatlas = File(exists=True)
    atlas = traits.Str(mandatory=True)
    clustering = traits.Bool(True, usedefault=True)
    clust_mask = File(exists=True, mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)


class IndividualClustering(SimpleInterface):
    """Interface wrapper for IndividualClustering"""
    input_spec = _IndividualClusteringInputSpec
    output_spec = _IndividualClusteringOutputSpec

    def _run_interface(self, runtime, c_boot=10):
        import gc
        from nilearn.masking import unmask
        from pynets.fmri.estimation import timeseries_bootstrap
        from nipype.utils.filemanip import fname_presuffix, copyfile
        from pynets.fmri import clustools
        from pynets.registration.reg_utils import check_orient_and_dims

        clust_list = ['kmeans', 'ward', 'complete', 'average', 'ncut', 'rena']

        clust_mask_temp_path = check_orient_and_dims(self.inputs.clust_mask, runtime.cwd, self.inputs.vox_size)

        if self.inputs.mask:
            out_name_mask = fname_presuffix(self.inputs.mask, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.mask, out_name_mask, copy=True, use_hardlink=False)
        else:
            out_name_mask = None

        out_name_func_file = fname_presuffix(self.inputs.func_file, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.func_file, out_name_func_file, copy=True, use_hardlink=False)

        if self.inputs.conf:
            out_name_conf = fname_presuffix(self.inputs.conf, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.conf, out_name_conf, copy=True, use_hardlink=False)
        else:
            out_name_conf = None

        nip = clustools.NiParcellate(func_file=out_name_func_file,
                                     clust_mask=clust_mask_temp_path,
                                     k=int(self.inputs.k),
                                     clust_type=self.inputs.clust_type,
                                     local_corr=self.inputs.local_corr,
                                     outdir=self.inputs.outdir,
                                     conf=out_name_conf,
                                     mask=out_name_mask)

        atlas = nip.create_clean_mask()
        nip.create_local_clustering(overwrite=True, r_thresh=0.4)

        if self.inputs.clust_type in clust_list:
            print(f"Performing circular block bootstrapping with {c_boot} iterations...")
            ts_data, block_size = nip.prep_boot()
            boot_parcellations = []
            for i in range(c_boot):
                print(f"\nBootstrapped iteration: {i}")
                boot_series = timeseries_bootstrap(ts_data, block_size)[0]
                func_boot_img = unmask(boot_series, nip._clust_mask_corr_img)
                out_path = f"{runtime.cwd}/boot_parc_tmp_{str(i)}.nii.gz"
                nib.save(nip.parcellate(func_boot_img), out_path)
                boot_parcellations.append(out_path)
                gc.collect()

            print('Creating spatially-constrained consensus parcellation...')
            consensus_parcellation = clustools.ensemble_parcellate(boot_parcellations, int(self.inputs.k))
            nib.save(consensus_parcellation, nip.uatlas)
        else:
            raise ValueError('Clustering method not recognized. '
                             'See: https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.'
                             'html#nilearn.regions.Parcellations')

        self._results['atlas'] = atlas
        self._results['uatlas'] = nip.uatlas
        self._results['clust_mask'] = self.inputs.clust_mask
        self._results['k'] = self.inputs.k
        self._results['clust_type'] = self.inputs.clust_type
        self._results['clustering'] = True
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
    extract_strategy = traits.Str('mean', mandatory=False, usedefault=True)


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

        if self.inputs.net_parcels_nii_path:
            out_name_net_parcels_nii_path = fname_presuffix(self.inputs.net_parcels_nii_path, suffix='_tmp',
                                                            newpath=runtime.cwd)
            copyfile(self.inputs.net_parcels_nii_path, out_name_net_parcels_nii_path, copy=True, use_hardlink=False)
        else:
            out_name_net_parcels_nii_path = None
        if self.inputs.mask:
            out_name_mask = fname_presuffix(self.inputs.mask, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.mask, out_name_mask, copy=True, use_hardlink=False)
        else:
            out_name_mask = None
        out_name_func_file = fname_presuffix(self.inputs.func_file, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.func_file, out_name_func_file, copy=True, use_hardlink=False)

        if self.inputs.conf:
            out_name_conf = fname_presuffix(self.inputs.conf, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.conf, out_name_conf, copy=True, use_hardlink=False)
        else:
            out_name_conf = None

        te = estimation.TimeseriesExtraction(net_parcels_nii_path=out_name_net_parcels_nii_path,
                                             node_size=self.inputs.node_size,
                                             conf=out_name_conf,
                                             func_file=out_name_func_file,
                                             coords=self.inputs.coords,
                                             roi=self.inputs.roi,
                                             dir_path=self.inputs.dir_path,
                                             ID=self.inputs.ID,
                                             network=self.inputs.network,
                                             smooth=self.inputs.smooth,
                                             atlas=self.inputs.atlas,
                                             uatlas=self.inputs.uatlas,
                                             labels=self.inputs.labels,
                                             hpass=self.inputs.hpass,
                                             mask=out_name_mask,
                                             extract_strategy=self.inputs.extract_strategy)

        te.prepare_inputs()

        te.extract_ts_parc()

        te.save_and_cleanup()

        self._results['ts_within_nodes'] = te.ts_within_nodes
        self._results['node_size'] = te.node_size
        self._results['smooth'] = te.smooth
        self._results['extract_strategy'] = te.extract_strategy
        self._results['dir_path'] = te.dir_path
        self._results['atlas'] = te.atlas
        self._results['uatlas'] = te.uatlas
        self._results['labels'] = te.labels
        self._results['coords'] = te.coords
        self._results['hpass'] = te.hpass
        self._results['roi'] = self.inputs.roi

        del te
        gc.collect()

        return runtime


class _PlotStructInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotStruct"""
    conn_matrix = traits.Array(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any()
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any()
    labels = traits.Array(mandatory=True)
    roi = traits.Any()
    coords = traits.Array(mandatory=True)
    thr = traits.Any()
    node_size = traits.Any()
    edge_threshold = traits.Any()
    prune = traits.Any()
    uatlas = traits.Any()
    target_samples = traits.Any()
    norm = traits.Any()
    binary = traits.Bool()
    track_type = traits.Any()
    directget = traits.Any()
    min_length = traits.Any()


class _PlotStructOutputSpec(BaseInterfaceInputSpec):
    """Output interface wrapper for PlotStruct"""
    out = traits.Str


class PlotStruct(SimpleInterface):
    """Interface wrapper for PlotStruct"""
    input_spec = _PlotStructInputSpec
    output_spec = _PlotStructOutputSpec

    def _run_interface(self, runtime):
        from pynets.plotting import plot_gen
        if self.inputs.coords.ndim == 1:
            print('Only 1 node detected. Plotting is not applicable...')
        else:
            plot_gen.plot_all_struct(self.inputs.conn_matrix,
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
                                     self.inputs.min_length)

        self._results['out'] = 'None'

        return runtime


class _PlotFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotFunc"""
    conn_matrix = traits.Array(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any()
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any()
    labels = traits.Array(mandatory=True)
    roi = traits.Any()
    coords = traits.Array(mandatory=True)
    thr = traits.Any()
    node_size = traits.Any()
    edge_threshold = traits.Any()
    smooth = traits.Any()
    prune = traits.Any()
    uatlas = traits.Any()
    norm = traits.Any()
    binary = traits.Bool()
    hpass = traits.Any()
    extract_strategy = traits.Any()
    edge_color_override = traits.Bool()


class _PlotFuncOutputSpec(BaseInterfaceInputSpec):
    """Output interface wrapper for PlotFunc"""
    out = traits.Str


class PlotFunc(SimpleInterface):
    """Interface wrapper for PlotFunc"""
    input_spec = _PlotFuncInputSpec
    output_spec = _PlotFuncOutputSpec

    def _run_interface(self, runtime):
        from pynets.plotting import plot_gen

        if self.inputs.coords.ndim == 1:
            print('Only 1 node detected. Plotting not applicable...')
        else:
            plot_gen.plot_all_func(self.inputs.conn_matrix,
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
                                   self.inputs.edge_color_override)

        self._results['out'] = 'None'

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
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    simple = traits.Bool(False, usedefault=True)
    overwrite = traits.Bool(True, usedefault=True)


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


class RegisterDWI(SimpleInterface):
    """Interface wrapper for RegisterDWI to create T1w->MNI->DWI mappings."""
    input_spec = _RegisterDWIInputSpec
    output_spec = _RegisterDWIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import glob
        import os.path as op
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile

        fa_tmp_path = fname_presuffix(self.inputs.fa_path, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.fa_path, fa_tmp_path, copy=True, use_hardlink=False)

        ap_tmp_path = fname_presuffix(self.inputs.ap_path, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.ap_path, ap_tmp_path, copy=True, use_hardlink=False)

        B0_mask_tmp_path = fname_presuffix(self.inputs.B0_mask, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.B0_mask, B0_mask_tmp_path, copy=True, use_hardlink=False)

        anat_mask_existing = [i for i in glob.glob(self.inputs.in_dir + '/*_desc-brain_mask.nii.gz') if
                              'MNI' not in i]
        if len(anat_mask_existing) > 0 and self.inputs.mask is None:
            mask_tmp_path = fname_presuffix(anat_mask_existing[0], suffix='_tmp', newpath=runtime.cwd)
            copyfile(anat_mask_existing[0], mask_tmp_path, copy=True, use_hardlink=False)
        else:
            # Apply T1w mask, if provided
            if self.inputs.mask:
                mask_tmp_path = fname_presuffix(self.inputs.mask, suffix='_tmp', newpath=runtime.cwd)
                copyfile(self.inputs.mask, mask_tmp_path, copy=True, use_hardlink=False)
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-GM_probseg.nii.gz')
        if len(gm_mask_existing) > 0:
            copyfile(gm_mask_existing[0], fname_presuffix(gm_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        wm_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-WM_probseg.nii.gz')
        if len(wm_mask_existing) > 0:
            copyfile(wm_mask_existing[0], fname_presuffix(wm_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        csf_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-CSF_probseg.nii.gz')
        if len(csf_mask_existing) > 0:
            copyfile(csf_mask_existing[0], fname_presuffix(csf_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        reg = register.DmriReg(basedir_path=runtime.cwd,
                               fa_path=fa_tmp_path,
                               ap_path=ap_tmp_path,
                               B0_mask=B0_mask_tmp_path,
                               anat_file=self.inputs.anat_file,
                               mask=mask_tmp_path,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)

        if (self.inputs.overwrite is True) or ((op.isfile(reg.wm_mask_thr) is False) and
                                               (op.isfile(reg.wm_edge) is False)):
            # Perform anatomical segmentation
            reg.gen_tissue()

        if (self.inputs.overwrite is True) or (op.isfile(reg.t1_aligned_mni) is False):
            # Align t1w to mni
            reg.t1w2mni_align()

        if (self.inputs.overwrite is True) or (op.isfile(reg.t1w2dwi) is False):
            # Align t1w to dwi
            reg.t1w2dwi_align()

        if (self.inputs.overwrite is True) or (op.isfile(reg.wm_gm_int_in_dwi) is False):
            # Align tissue
            reg.tissue2dwi_align()

        self._results['wm_in_dwi'] = reg.wm_in_dwi
        self._results['gm_in_dwi'] = reg.gm_in_dwi
        self._results['vent_csf_in_dwi'] = reg.vent_csf_in_dwi
        self._results['csf_mask_dwi'] = reg.csf_mask_dwi
        self._results['anat_file'] = self.inputs.anat_file
        self._results['t1w2dwi'] = reg.t1w2dwi
        self._results['B0_mask'] = self.inputs.B0_mask
        self._results['ap_path'] = self.inputs.ap_path
        self._results['gtab_file'] = self.inputs.gtab_file
        self._results['dwi_file'] = self.inputs.dwi_file
        self._results['basedir_path'] = runtime.cwd
        self._results['t1w_brain_mask_in_dwi'] = reg.t1w_brain_mask_in_dwi

        gc.collect()

        return runtime


class _RegisterAtlasDWIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterAtlasDWI"""
    atlas = traits.Any()
    network = traits.Any()
    uatlas_parcels = traits.Any()
    uatlas = traits.Any()
    basedir_path = Directory(exists=True, mandatory=True)
    node_size = traits.Any()
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
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterAtlasDWIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterAtlasDWI"""
    dwi_aligned_atlas_wmgm_int = File(exists=True, mandatory=True)
    dwi_aligned_atlas = File(exists=True, mandatory=True)
    aligned_atlas_t1mni = File(exists=True, mandatory=True)
    node_size = traits.Any()
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
        import shutil
        import gc
        import os
        from pynets.registration import register
        from pynets.core.utils import missing_elements
        from nipype.utils.filemanip import fname_presuffix, copyfile

        if self.inputs.uatlas is None:
            uatlas_tmp_path = None
        else:
            uatlas_tmp_path = fname_presuffix(self.inputs.uatlas, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.uatlas, uatlas_tmp_path, copy=True, use_hardlink=False)

        if self.inputs.uatlas_parcels is None:
            uatlas_parcels_tmp_path = None
        else:
            uatlas_parcels_tmp_path = fname_presuffix(self.inputs.uatlas_parcels, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.uatlas_parcels, uatlas_parcels_tmp_path, copy=True, use_hardlink=False)

        fa_tmp_path = fname_presuffix(self.inputs.fa_path, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.fa_path, fa_tmp_path, copy=True, use_hardlink=False)

        ap_tmp_path = fname_presuffix(self.inputs.ap_path, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.ap_path, ap_tmp_path, copy=True, use_hardlink=False)

        B0_mask_tmp_path = fname_presuffix(self.inputs.B0_mask, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.B0_mask, B0_mask_tmp_path, copy=True, use_hardlink=False)

        if self.inputs.mask:
            mask_tmp_path = fname_presuffix(self.inputs.mask, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.mask, mask_tmp_path, copy=True, use_hardlink=False)
        else:
            mask_tmp_path = None

        if self.inputs.network or self.inputs.waymask:
            if self.inputs.waymask:
                atlas_name = f"{self.inputs.atlas}_{self.inputs.waymask}"
            else:
                atlas_name = f"{self.inputs.atlas}_{self.inputs.network}"
        else:
            atlas_name = f"{self.inputs.atlas}"

        base_dir_tmp = f"{runtime.cwd}/atlas_{atlas_name}"
        shutil.copytree(self.inputs.basedir_path, base_dir_tmp)

        # base_dir_tmp = f"{self.inputs.basedir_path}/atlas_{atlas_name}"
        # os.makedirs(base_dir_tmp, exist_ok=True)

        reg = register.DmriReg(basedir_path=base_dir_tmp,
                               fa_path=fa_tmp_path,
                               ap_path=ap_tmp_path,
                               B0_mask=B0_mask_tmp_path,
                               anat_file=self.inputs.anat_file,
                               mask=mask_tmp_path,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)

        if self.inputs.node_size is not None:
            atlas_name = f"{atlas_name}{'_'}{self.inputs.node_size}"

        # Apply warps/coregister atlas to dwi
        [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas,
         aligned_atlas_t1mni] = reg.atlas2t1w2dwi_align(uatlas_tmp_path, uatlas_parcels_tmp_path, atlas_name)

        # Correct coords and labels
        bad_idxs = missing_elements(list(np.unique(np.asarray(nib.load(dwi_aligned_atlas).dataobj).astype('int'))))
        bad_idxs = [i-1 for i in bad_idxs]
        if len(bad_idxs) > 0:
            bad_idxs = sorted(list(set(bad_idxs)), reverse=True)
            for j in bad_idxs:
                del self.inputs.labels[j], self.inputs.coords[j]

        assert len(self.inputs.coords) == len(self.inputs.labels) == len(np.unique(np.asarray(nib.load(
            dwi_aligned_atlas).dataobj))[1:])

        if self.inputs.waymask:
            waymask_tmp_path = fname_presuffix(self.inputs.waymask, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.waymask, waymask_tmp_path, copy=True, use_hardlink=False)

            # Align waymask
            waymask_in_dwi = reg.waymask2dwi_align(waymask_tmp_path)
        else:
            waymask_in_dwi = None

        if self.inputs.uatlas is None:
            uatlas_out = self.inputs.uatlas_parcels
        else:
            uatlas_out = self.inputs.uatlas

        reg_dir = f"{os.path.dirname(self.inputs.anat_file)}/reg"
        if not os.path.isdir(reg_dir):
            os.mkdir(reg_dir)

        reg_persist = [dwi_aligned_atlas_wmgm_int, dwi_aligned_atlas, aligned_atlas_t1mni, reg.wm_in_dwi,
                       reg.gm_in_dwi, reg.vent_csf_in_dwi, reg.wm_gm_int_in_dwi, reg.t1w2dwi]
        for i in reg_persist:
            if os.path.isfile(i):
                copyfile(i, f"{reg_dir}/{os.path.basename(i)}_{self.inputs.atlas}", copy=True, use_hardlink=False)

        reg_tmp = [B0_mask_tmp_path, ap_tmp_path, fa_tmp_path, uatlas_parcels_tmp_path, uatlas_tmp_path]
        for j in reg_tmp:
            if j is not None:
                os.remove(j)

        self._results['dwi_aligned_atlas_wmgm_int'] = dwi_aligned_atlas_wmgm_int
        self._results['dwi_aligned_atlas'] = dwi_aligned_atlas
        self._results['aligned_atlas_t1mni'] = aligned_atlas_t1mni
        self._results['node_size'] = self.inputs.node_size
        self._results['atlas'] = self.inputs.atlas
        self._results['uatlas_parcels'] = uatlas_parcels_tmp_path
        self._results['uatlas'] = uatlas_out
        self._results['coords'] = self.inputs.coords
        self._results['labels'] = self.inputs.labels
        self._results['wm_in_dwi'] = reg.wm_in_dwi
        self._results['gm_in_dwi'] = reg.gm_in_dwi
        self._results['vent_csf_in_dwi'] = reg.vent_csf_in_dwi
        self._results['B0_mask'] = self.inputs.B0_mask
        self._results['ap_path'] = self.inputs.ap_path
        self._results['gtab_file'] = self.inputs.gtab_file
        self._results['dwi_file'] = self.inputs.dwi_file
        self._results['waymask_in_dwi'] = waymask_in_dwi
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
    simple = traits.Bool(False, usedefault=True)
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)


class _RegisterROIDWIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterROIDWI"""
    roi = traits.Any(mandatory=False)


class RegisterROIDWI(SimpleInterface):
    """Interface wrapper for RegisterROIDWI."""
    input_spec = _RegisterROIDWIInputSpec
    output_spec = _RegisterROIDWIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import shutil
        import os
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile

        ap_tmp_path = fname_presuffix(self.inputs.ap_path, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.ap_path, ap_tmp_path, copy=True, use_hardlink=False)

        roi_file_tmp_path = fname_presuffix(self.inputs.roi, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.roi, roi_file_tmp_path, copy=True, use_hardlink=False)

        base_dir_tmp = f"{runtime.cwd}/{self.inputs.roi}"
        shutil.copytree(self.inputs.basedir_path, base_dir_tmp)

        # base_dir_tmp = f"{self.inputs.basedir_path}/atlas_{self.inputs.roi}"
        # os.makedirs(base_dir_tmp, exist_ok=True)

        reg = register.DmriReg(basedir_path=base_dir_tmp,
                               fa_path=self.inputs.fa_path,
                               ap_path=ap_tmp_path,
                               B0_mask=self.inputs.B0_mask,
                               anat_file=self.inputs.anat_file,
                               mask=None,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)
        if self.inputs.roi:
            # Align roi
            roi_in_dwi = reg.roi2dwi_align(roi_file_tmp_path)
        else:
            roi_in_dwi = None

        reg_dir = f"{os.path.dirname(self.inputs.anat_file)}/reg"
        if not os.path.isdir(reg_dir):
            os.mkdir(reg_dir)

        self._results['roi'] = roi_in_dwi

        gc.collect()

        return runtime


class _RegisterFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterFunc"""
    anat_file = File(exists=True, mandatory=True)
    mask = traits.Any(mandatory=False)
    in_dir = traits.Any()
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)
    overwrite = traits.Bool(True, usedefault=True)


class _RegisterFuncOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterFunc"""
    reg_fmri_complete = traits.Bool()
    basedir_path = Directory(exists=True, mandatory=True)
    t1w_brain_mask = traits.Any(mandatory=False)
    epi_brain_path = traits.Any()


class RegisterFunc(SimpleInterface):
    """Interface wrapper for RegisterFunc to create Func->T1w->MNI mappings."""
    input_spec = _RegisterFuncInputSpec
    output_spec = _RegisterFuncOutputSpec

    def _run_interface(self, runtime):
        import gc
        import glob
        import os.path as op
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile

        anat_mask_existing = [i for i in glob.glob(self.inputs.in_dir + '/*_desc-brain_mask.nii.gz') if
                              'MNI' not in i]
        if len(anat_mask_existing) > 0 and self.inputs.mask is None:
            mask_tmp_path = fname_presuffix(anat_mask_existing[0], suffix='_tmp', newpath=runtime.cwd)
            copyfile(anat_mask_existing[0], mask_tmp_path, copy=True, use_hardlink=False)
        else:
            # Apply T1w mask, if provided
            if self.inputs.mask:
                mask_tmp_path = fname_presuffix(self.inputs.mask, suffix='_tmp', newpath=runtime.cwd)
                copyfile(self.inputs.mask, mask_tmp_path, copy=True, use_hardlink=False)
            else:
                mask_tmp_path = None

        gm_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-GM_probseg.nii.gz')
        if len(gm_mask_existing) > 0:
            copyfile(gm_mask_existing[0], fname_presuffix(gm_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        wm_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-WM_probseg.nii.gz')
        if len(wm_mask_existing) > 0:
            copyfile(wm_mask_existing[0], fname_presuffix(wm_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        csf_mask_existing = glob.glob(self.inputs.in_dir + '/*_label-CSF_probseg.nii.gz')
        if len(csf_mask_existing) > 0:
            copyfile(csf_mask_existing[0], fname_presuffix(csf_mask_existing[0], newpath=runtime.cwd), copy=True,
                     use_hardlink=False)

        reg = register.FmriReg(basedir_path=runtime.cwd,
                               anat_file=self.inputs.anat_file,
                               mask=mask_tmp_path,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)

        if (self.inputs.overwrite is True) or (op.isfile(reg.gm_mask_thr) is False):
            # Perform anatomical segmentation
            reg.gen_tissue()

        if (self.inputs.overwrite is True) or (op.isfile(reg.t1_aligned_mni) is False):
            # Align t1w to mni
            reg.t1w2mni_align()

        self._results['reg_fmri_complete'] = True
        self._results['basedir_path'] = runtime.cwd
        self._results['t1w_brain_mask'] = reg.t1w_brain_mask

        gc.collect()

        return runtime


class _RegisterAtlasFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterAtlasFunc"""
    atlas = traits.Any()
    network = traits.Any()
    uatlas_parcels = traits.Any()
    uatlas = traits.Any()
    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    node_size = traits.Any()
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    roi = traits.Any(mandatory=False)
    reg_fmri_complete = traits.Bool()
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterAtlasFuncOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterAtlasFunc"""
    aligned_atlas_gm = File(exists=True, mandatory=True)
    roi_in_epi = traits.Any(mandatory=False)
    coords = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    node_size = traits.Any()


class RegisterAtlasFunc(SimpleInterface):
    """Interface wrapper for RegisterAtlasFunc."""
    input_spec = _RegisterAtlasFuncInputSpec
    output_spec = _RegisterAtlasFuncOutputSpec

    def _run_interface(self, runtime):
        import gc
        import shutil
        import os
        from pynets.registration import register
        from pynets.core.utils import missing_elements
        from nipype.utils.filemanip import fname_presuffix, copyfile

        if self.inputs.uatlas is None:
            uatlas_tmp_path = None
        else:
            uatlas_tmp_path = fname_presuffix(self.inputs.uatlas, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.uatlas, uatlas_tmp_path, copy=True, use_hardlink=False)

        if self.inputs.uatlas_parcels is None:
            uatlas_parcels_tmp_path = None
        else:
            uatlas_parcels_tmp_path = fname_presuffix(self.inputs.uatlas_parcels, suffix='_tmp', newpath=runtime.cwd)
            copyfile(self.inputs.uatlas_parcels, uatlas_parcels_tmp_path, copy=True, use_hardlink=False)

        if self.inputs.network or self.inputs.roi:
            atlas_name = f"{self.inputs.atlas}_{self.inputs.network}"
        else:
            atlas_name = f"{self.inputs.atlas}"
        base_dir_tmp = f"{runtime.cwd}/atlas_{atlas_name}"
        shutil.copytree(self.inputs.basedir_path, base_dir_tmp)

        # base_dir_tmp = f"{self.inputs.basedir_path}/atlas_{atlas_name}"
        # os.makedirs(base_dir_tmp, exist_ok=True)

        reg = register.FmriReg(basedir_path=base_dir_tmp,
                               anat_file=self.inputs.anat_file,
                               mask=self.inputs.mask,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)

        if self.inputs.node_size is not None:
            atlas_name = f"{atlas_name}{'_'}{self.inputs.node_size}"

        aligned_atlas_gm, aligned_atlas_skull = reg.atlas2t1w_align(uatlas_tmp_path, uatlas_parcels_tmp_path,
                                                                    atlas_name)

        # Correct coords and labels
        bad_idxs = missing_elements(list(np.unique(np.asarray(nib.load(aligned_atlas_skull).dataobj).astype('int'))))
        bad_idxs = [i-1 for i in bad_idxs]
        if len(bad_idxs) > 0:
            bad_idxs = sorted(list(set(bad_idxs)), reverse=True)
            for j in bad_idxs:
                del self.inputs.labels[j], self.inputs.coords[j]

        assert len(self.inputs.coords) == len(self.inputs.labels) == len(np.unique(np.asarray(nib.load(
            aligned_atlas_skull).dataobj))[1:])

        reg_dir = f"{os.path.dirname(self.inputs.anat_file)}/reg"
        if not os.path.isdir(reg_dir):
            os.mkdir(reg_dir)

        reg_persist = [aligned_atlas_gm, reg.t1_aligned_mni, reg.gm_mask_thr]
        for i in reg_persist:
            if os.path.isfile(i):
                copyfile(i, f"{reg_dir}/{os.path.basename(i)}_{self.inputs.atlas}", copy=True, use_hardlink=False)

        reg_tmp = [uatlas_parcels_tmp_path, uatlas_tmp_path]
        for j in reg_tmp:
            if j is not None:
                os.remove(j)

        self._results['aligned_atlas_gm'] = aligned_atlas_gm
        self._results['coords'] = self.inputs.coords
        self._results['labels'] = self.inputs.labels
        self._results['node_size'] = self.inputs.node_size

        gc.collect()

        return runtime


class _RegisterROIFEPIInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for RegisterROIEPI"""
    basedir_path = Directory(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    roi = traits.Any(mandatory=False)
    template_name = traits.Str('MNI152_T1', mandatory=True, usedefault=True)
    simple = traits.Bool(False, usedefault=True)


class _RegisterROIEPIOutputSpec(TraitedSpec):
    """Output interface wrapper for RegisterROIEPI"""
    roi = traits.Any(mandatory=False)


class RegisterROIEPI(SimpleInterface):
    """Interface wrapper for RegisterROIEPI."""
    input_spec = _RegisterROIFEPIInputSpec
    output_spec = _RegisterROIEPIOutputSpec

    def _run_interface(self, runtime):
        import gc
        import shutil
        import os
        from pynets.registration import register
        from nipype.utils.filemanip import fname_presuffix, copyfile

        roi_file_tmp_path = fname_presuffix(self.inputs.roi, suffix='_tmp', newpath=runtime.cwd)
        copyfile(self.inputs.roi, roi_file_tmp_path, copy=True, use_hardlink=False)

        base_dir_tmp = f"{runtime.cwd}/{self.inputs.roi}"
        shutil.copytree(self.inputs.basedir_path, base_dir_tmp)

        # base_dir_tmp = f"{self.inputs.basedir_path}/atlas_{self.inputs.roi}"
        # os.makedirs(base_dir_tmp, exist_ok=True)

        reg = register.FmriReg(basedir_path=base_dir_tmp,
                               anat_file=self.inputs.anat_file,
                               mask=None,
                               vox_size=self.inputs.vox_size,
                               template_name=self.inputs.template_name,
                               simple=self.inputs.simple)

        if self.inputs.roi:
            # Align roi
            roi_in_t1w = reg.roi2t1w_align(roi_file_tmp_path)
        else:
            roi_in_t1w = None

        reg_dir = f"{os.path.dirname(self.inputs.anat_file)}/reg"
        if not os.path.isdir(reg_dir):
            os.mkdir(reg_dir)

        self._results['roi'] = roi_in_t1w

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
    atlas_mni = File(exists=True, mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    waymask = traits.Any(mandatory=False)
    t1w2dwi = File(exists=True, mandatory=True)
    roi_neighborhood_tol = traits.Any(6, mandatory=True, usedefault=True)
    sphere = traits.Str('repulsion724', mandatory=True, usedefault=True)


class _TrackingOutputSpec(TraitedSpec):
    """Output interface wrapper for Tracking"""
    streams = File(exists=True, mandatory=True)
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
    atlas_mni = File(exists=True, mandatory=True)
    curv_thr_list = traits.List(mandatory=True)
    step_list = traits.List(mandatory=True)
    fa_path = File(exists=True, mandatory=True)
    dm_path = File(exists=True, mandatory=True)
    directget = traits.Str(mandatory=True)
    labels_im_file = File(exists=True, mandatory=True)
    roi_neighborhood_tol = traits.Any()
    min_length = traits.Any()


class Tracking(SimpleInterface):
    """Interface wrapper for Tracking"""
    input_spec = _TrackingInputSpec
    output_spec = _TrackingOutputSpec

    def _run_interface(self, runtime):
        import gc
        import os
        import os.path as op
        from dipy.io import load_pickle
        from colorama import Fore, Style
        from dipy.data import get_sphere
        from pynets.core import utils
        from pynets.dmri.track import prep_tissues, reconstruction, create_density_map, track_ensemble
        from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
        from dipy.io.streamline import save_tractogram
        from nipype.utils.filemanip import copyfile

        # Load diffusion data
        dwi_img = nib.load(self.inputs.dwi_file)

        # Fit diffusion model
        model, mod = reconstruction(self.inputs.conn_model, load_pickle(self.inputs.gtab_file),
                                    np.asarray(dwi_img.dataobj), self.inputs.B0_mask)

        # Load atlas parcellation (and its wm-gm interface reduced version for seeding)
        atlas_data = np.array(nib.load(self.inputs.labels_im_file).dataobj).astype('uint16')
        atlas_data_wm_gm_int = np.asarray(nib.load(self.inputs.labels_im_file_wm_gm_int).dataobj).astype('uint16')

        # Build mask vector from atlas for later roi filtering
        parcels = []
        i = 0
        for roi_val in np.unique(atlas_data)[1:]:
            parcels.append(atlas_data == roi_val)
            i = i + 1

        if np.sum(atlas_data) == 0:
            raise ValueError(
                'ERROR: No non-zero voxels found in atlas. Check any roi masks and/or wm-gm interface images '
                'to verify overlap with dwi-registered atlas.')

        # Iteratively build a list of streamlines for each ROI while tracking
        print(f"{Fore.GREEN}Target number of samples: {Fore.BLUE} {self.inputs.target_samples}")
        print(Style.RESET_ALL)
        print(f"{Fore.GREEN}Using curvature threshold(s): {Fore.BLUE} {self.inputs.curv_thr_list}")
        print(Style.RESET_ALL)
        print(f"{Fore.GREEN}Using step size(s): {Fore.BLUE} {self.inputs.step_list}")
        print(Style.RESET_ALL)
        print(f"{Fore.GREEN}Tracking type: {Fore.BLUE} {self.inputs.track_type}")
        print(Style.RESET_ALL)
        if self.inputs.directget == 'prob':
            print(f"{Fore.GREEN}Direction-getting type: {Fore.BLUE}Probabilistic")
        elif self.inputs.directget == 'clos':
            print(f"{Fore.GREEN}Direction-getting type: {Fore.BLUE}Closest Peak")
        elif self.inputs.directget == 'det':
            print(f"{Fore.GREEN}Direction-getting type: {Fore.BLUE}Deterministic Maximum")
        else:
            raise ValueError('Direction-getting type not recognized!')
        print(Style.RESET_ALL)

        dir_path = utils.do_dir_path(self.inputs.atlas, os.path.dirname(self.inputs.dwi_file))

        # Commence Ensemble Tractography
        streamlines = track_ensemble(self.inputs.target_samples, atlas_data_wm_gm_int,
                                     parcels, model,
                                     prep_tissues(self.inputs.t1w2dwi, self.inputs.gm_in_dwi,
                                                  self.inputs.vent_csf_in_dwi, self.inputs.wm_in_dwi,
                                                  self.inputs.tiss_class),
                                     get_sphere(self.inputs.sphere), self.inputs.directget, self.inputs.curv_thr_list,
                                     self.inputs.step_list, self.inputs.track_type, self.inputs.maxcrossing,
                                     int(self.inputs.roi_neighborhood_tol), self.inputs.min_length,
                                     self.inputs.waymask, self.inputs.B0_mask)

        namer_dir = '{}/tractography'.format(dir_path)
        if not os.path.isdir(namer_dir):
            os.mkdir(namer_dir)

        # Save streamlines to trk
        streams = "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s" % (runtime.cwd, '/streamlines_',
                                                            '%s' % (self.inputs.network + '_' if
                                                                    self.inputs.network is not None else ''),
                                                            '%s' % (op.basename(self.inputs.roi).split('.')[0] + '_'
                                                                    if self.inputs.roi is not None else ''),
                                                            self.inputs.conn_model, '_', self.inputs.target_samples,
                                                            '_', '%s' % ("%s%s" % (self.inputs.node_size, 'mm_') if
                                                                         ((self.inputs.node_size != 'parc') and
                                                                          (self.inputs.node_size is not None))
                                                                         else 'parc_'),
                                                            'curv-', str(self.inputs.curv_thr_list).replace(', ', '_'),
                                                            '_step-', str(self.inputs.step_list).replace(', ', '_'),
                                                            '_dg-', self.inputs.directget,
                                                            '_ml-', self.inputs.min_length, '.trk')

        save_tractogram(StatefulTractogram(streamlines, reference=dwi_img, space=Space.RASMM, origin=Origin.TRACKVIS),
                        streams, bbox_valid_check=False)

        copyfile(streams, f"{namer_dir}/{op.basename(streams)}", copy=True, use_hardlink=False)

        # Create streamline density map
        [dir_path, dm_path] = create_density_map(dwi_img, dir_path, streamlines, self.inputs.conn_model,
                                                 self.inputs.target_samples,
                                                 self.inputs.node_size, self.inputs.curv_thr_list,
                                                 self.inputs.step_list, self.inputs.network, self.inputs.roi,
                                                 self.inputs.directget, self.inputs.min_length, namer_dir)

        self._results['streams'] = streams
        self._results['track_type'] = self.inputs.track_type
        self._results['target_samples'] = self.inputs.target_samples
        self._results['conn_model'] = self.inputs.conn_model
        self._results['dir_path'] = dir_path
        self._results['network'] = self.inputs.network
        self._results['node_size'] = self.inputs.node_size
        self._results['dens_thresh'] = self.inputs.dens_thresh
        self._results['ID'] = self.inputs.ID
        self._results['roi'] = self.inputs.roi
        self._results['min_span_tree'] = self.inputs.min_span_tree
        self._results['disp_filt'] = self.inputs.disp_filt
        self._results['parc'] = self.inputs.parc
        self._results['prune'] = self.inputs.prune
        self._results['atlas'] = self.inputs.atlas
        self._results['uatlas'] = self.inputs.uatlas
        self._results['labels'] = self.inputs.labels
        self._results['coords'] = self.inputs.coords
        self._results['norm'] = self.inputs.norm
        self._results['binary'] = self.inputs.binary
        self._results['atlas_mni'] = self.inputs.atlas_mni
        self._results['curv_thr_list'] = self.inputs.curv_thr_list
        self._results['step_list'] = self.inputs.step_list
        self._results['fa_path'] = self.inputs.fa_path
        self._results['dm_path'] = dm_path
        self._results['directget'] = self.inputs.directget
        self._results['labels_im_file'] = self.inputs.labels_im_file
        self._results['roi_neighborhood_tol'] = self.inputs.roi_neighborhood_tol
        self._results['min_length'] = self.inputs.min_length

        del streamlines, atlas_data_wm_gm_int, atlas_data, model, parcels
        dwi_img.uncache()
        gc.collect()

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
        from dipy.io import save_pickle
        from dipy.io import read_bvals_bvecs
        from dipy.core.gradients import gradient_table
        from dipy.segment.mask import median_otsu
        from pynets.registration.reg_utils import median
        from pynets.dmri.dmri_utils import normalize_gradients, extract_b0

        B0_bet = f"{runtime.cwd}/mean_B0_bet.nii.gz"
        B0_mask = f"{runtime.cwd}/mean_B0_bet_mask.nii.gz"
        fbvec_norm = f"{runtime.cwd}/bvec_normed.bvec"
        fbval_norm = f"{runtime.cwd}/bval_normed.bvec"
        gtab_file = f"{runtime.cwd}/gtab.pkl"
        all_b0s_file = f"{runtime.cwd}/all_b0s.nii.gz"

        # loading bvecs/bvals
        bvals, bvecs = read_bvals_bvecs(self.inputs.fbval, self.inputs.fbvec)
        bvecs_norm, bvals_norm = normalize_gradients(bvecs, bvals, b0_threshold=self.inputs.b0_thr)

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

        # Extract and Combine all b0s collected, make mean b0
        print("Extracting b0's...")
        all_b0s_file = extract_b0(self.inputs.dwi_file, b0_thr_ixs, all_b0s_file)
        med_b0_file = median(all_b0s_file)
        med_b0_img = nib.load(med_b0_file)
        med_b0_data = np.asarray(med_b0_img.dataobj)

        # Create mean b0 brain mask
        b0_mask_data, mask_data = median_otsu(med_b0_data, median_radius=2, numpass=1)

        hdr = med_b0_img.header.copy()
        hdr.set_xyzt_units("mm")
        hdr.set_data_dtype(np.float32)
        nib.Nifti1Image(b0_mask_data, med_b0_img.affine, hdr).to_filename(B0_bet)
        nib.Nifti1Image(mask_data, med_b0_img.affine, hdr).to_filename(B0_mask)

        self._results['gtab_file'] = gtab_file
        self._results['B0_bet'] = B0_bet
        self._results['B0_mask'] = B0_mask
        self._results['dwi_file'] = self.inputs.dwi_file

        return runtime
