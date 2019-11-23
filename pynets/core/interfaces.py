#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
from pynets.stats.netstats import extractnetstats
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, SimpleInterface


class ExtractNetStatsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for ExtractNetStats"""
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    thr = traits.Any(mandatory=True)
    conn_model = traits.Str(mandatory=True)
    est_path = File(exists=True, mandatory=True)
    roi = traits.Any(mandatory=False)
    prune = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=False)
    binary = traits.Bool(False, usedefault=True)


class ExtractNetStatsOutputSpec(TraitedSpec):
    """Output interface wrapper for ExtractNetStats"""
    out_path_neat = File(exists=True, mandatory=True)


class ExtractNetStats(BaseInterface):
    """Interface wrapper for ExtractNetStats"""
    input_spec = ExtractNetStatsInputSpec
    output_spec = ExtractNetStatsOutputSpec

    def _run_interface(self, runtime):
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


class CombinePandasDfsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for CombinePandasDfs"""
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    net_mets_csv_list = traits.List(mandatory=True)
    plot_switch = traits.Bool(False, usedefault=True)
    multi_nets = traits.Any(mandatory=False)
    multimodal = traits.Bool(False, usedefault=True)


class CombinePandasDfsOutputSpec(TraitedSpec):
    """Output interface wrapper for CombinePandasDfs"""
    combination_complete = traits.Bool()


class CombinePandasDfs(SimpleInterface):
    """Interface wrapper for CombinePandasDfs"""
    input_spec = CombinePandasDfsInputSpec
    output_spec = CombinePandasDfsOutputSpec

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

    def _run_interface(self, runtime):
        from pynets.fmri import clustools
        from pynets.core import utils
        import time
        import gc
        import os
        import os.path as op
        from pynets.registration.reg_utils import check_orient_and_dims
        from pathlib import Path

        nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']

        while utils.has_handle(self.inputs.func_file) is True:
            time.sleep(1)

        time.sleep(60)

        cwd = Path(runtime.cwd).absolute()

        func_temp_path = utils.create_temporary_copy(self.inputs.func_file,
                                                     op.basename(self.inputs.func_file).split('.nii')[0],
                                                     '.nii', cwd)

        clust_mask_temp_path = utils.create_temporary_copy(check_orient_and_dims(self.inputs.clust_mask,
                                                                                 self.inputs.vox_size),
                                                           op.basename(self.inputs.clust_mask).split('.nii')[0],
                                                           '.nii', cwd)

        nip = clustools.NilParcellate(func_file=func_temp_path,
                                      clust_mask=clust_mask_temp_path,
                                      k=self.inputs.k,
                                      clust_type=self.inputs.clust_type,
                                      local_corr=self.inputs.local_corr,
                                      conf=self.inputs.conf)

        atlas = nip.create_clean_mask()
        nip.create_local_clustering(overwrite=True, r_thresh=0.4)

        if self.inputs.clust_type in nilearn_clust_list:
            uatlas = nip.parcellate()
        else:
            raise ValueError('Clustering method not recognized. '
                             'See: https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.html#nilearn.'
                             'regions.Parcellations')
        del nip
        os.remove(func_temp_path)
        os.remove(clust_mask_temp_path)
        gc.collect()
        time.sleep(60)

        self._results['atlas'] = atlas
        self._results['uatlas'] = uatlas
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
    dir_path = traits.Str(mandatory=True)
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=False)
    smooth = traits.Any(mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=False)
    labels = traits.Any(mandatory=True)
    c_boot = traits.Any(mandatory=False)
    block_size = traits.Any(mandatory=False)
    hpass = traits.Any(mandatory=True)
    mask = traits.Any(mandatory=False)
    parc = traits.Bool()
    node_size = traits.Any(mandatory=False)
    net_parcels_nii_path = traits.Any(mandatory=False)


class _ExtractTimeseriesOutputSpec(TraitedSpec):
    """Output interface wrapper for ExtractTimeseries"""
    ts_within_nodes = traits.Any(mandatory=True)
    node_size = traits.Any(mandatory=True)
    smooth = traits.Any(mandatory=True)
    dir_path = traits.Str(mandatory=True)
    atlas = traits.Any(mandatory=True)
    uatlas = traits.Any(mandatory=True)
    labels = traits.Any(mandatory=True)
    coords = traits.Any(mandatory=True)
    c_boot = traits.Any(mandatory=True)
    hpass = traits.Any(mandatory=True)


class ExtractTimeseries(SimpleInterface):
    """Interface wrapper for ExtractTimeseries"""
    input_spec = _ExtractTimeseriesInputSpec
    output_spec = _ExtractTimeseriesOutputSpec

    def _run_interface(self, runtime):
        from pynets.fmri import estimation
        from pynets.core import utils
        import time
        import gc
        import os
        import os.path as op
        from pathlib import Path

        while utils.has_handle(self.inputs.func_file) is True:
            time.sleep(1)

        time.sleep(60)

        cwd = Path(runtime.cwd).absolute()

        func_temp_path = utils.create_temporary_copy(
            self.inputs.func_file, op.basename(self.inputs.func_file).split('.nii')[0], '.nii', cwd)

        mask_path = utils.create_temporary_copy(self.inputs.mask, op.basename(self.inputs.mask).split('.nii')[0],
                                                '.nii')

        if self.inputs.parc is True:
            net_parcels_nii_temp_path = utils.create_temporary_copy(
                self.inputs.net_parcels_nii_path, op.basename(self.inputs.net_parcels_nii_path).split('.nii')[0],
                '.nii')
        else:
            net_parcels_nii_temp_path = None

        te = estimation.TimeseriesExtraction(net_parcels_nii_path=net_parcels_nii_temp_path,
                                             node_size=self.inputs.node_size,
                                             conf=self.inputs.conf,
                                             func_file=func_temp_path,
                                             coords=self.inputs.coords,
                                             roi=self.inputs.roi,
                                             dir_path=self.inputs.dir_path,
                                             ID=self.inputs.ID,
                                             network=self.inputs.network,
                                             smooth=self.inputs.smooth,
                                             atlas=self.inputs.atlas,
                                             uatlas=self.inputs.uatlas,
                                             labels=self.inputs.labels,
                                             c_boot=self.inputs.c_boot,
                                             block_size=self.inputs.block_size,
                                             hpass=self.inputs.hpass,
                                             mask=mask_path)

        te.prepare_inputs()
        if self.inputs.parc is False:
            if len(self.inputs.coords) > 0:
                te.extract_ts_coords()
            else:
                raise RuntimeError(
                    '\nERROR: Cannot extract time-series from an empty list of coordinates. \nThis usually means '
                    'that no nodes were generated based on the specified conditions at runtime (e.g. atlas was '
                    'overly restricted by an RSN or some user-defined mask.')
        else:
            te.extract_ts_parc()

        if float(self.inputs.c_boot) > 0:
            te.bootstrap_timeseries()

        te.save_and_cleanup()

        os.remove(func_temp_path)
        if mask_path is not None:
            os.remove(mask_path)

        if self.inputs.net_parcels_nii_path is not None and os.path.isfile(net_parcels_nii_temp_path):
            os.remove(net_parcels_nii_temp_path)

        self._results['ts_within_nodes'] = te.ts_within_nodes
        self._results['node_size'] = te.node_size
        self._results['smooth'] = te.smooth
        self._results['dir_path'] = te.dir_path
        self._results['atlas'] = te.atlas
        self._results['uatlas'] = te.uatlas
        self._results['labels'] = te.labels
        self._results['coords'] = te.coords
        self._results['c_boot'] = te.c_boot
        self._results['hpass'] = te.hpass

        del te
        gc.collect()
        time.sleep(60)

        return runtime

