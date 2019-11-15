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
    """
    Input interface wrapper for ExtractNetStats
    """
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
    """
    Output interface wrapper for ExtractNetStats
    """
    out_path_neat = File(exists=True, mandatory=True)


class ExtractNetStats(BaseInterface):
    """
    Interface wrapper for ExtractNetStats
    """
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
    """
    Input interface wrapper for CombinePandasDfs
    """
    ID = traits.Any(mandatory=True)
    network = traits.Any(mandatory=True)
    net_mets_csv_list = traits.List(mandatory=True)
    plot_switch = traits.Bool(False, usedefault=True)
    multi_nets = traits.Any(mandatory=True)
    multimodal = traits.Bool(False, usedefault=True)


class CombinePandasDfsOutputSpec(TraitedSpec):
    """
    Output interface wrapper for CombinePandasDfs
    """
    combination_complete = traits.Bool()


class CombinePandasDfs(SimpleInterface):
    """
    Interface wrapper for CombinePandasDfs
    """
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
    func_file = File(exists=True, mandatory=True)
    conf = File(exists=False, mandatory=False)
    clust_mask = File(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)
    vox_size = traits.Str('2mm', mandatory=True, usedefault=True)
    local_corr = traits.Str('tcorr', mandatory=True, usedefault=True)


class _IndividualClusteringOutputSpec(TraitedSpec):
    uatlas = File(exists=True)
    atlas = traits.Str(mandatory=True)
    clustering = traits.Bool(True, usedefault=True)
    clust_mask = File(exists=True, mandatory=True)
    k = traits.Any(mandatory=True)
    clust_type = traits.Str(mandatory=True)


class IndividualClustering(SimpleInterface):
    input_spec = _IndividualClusteringInputSpec
    output_spec = _IndividualClusteringOutputSpec

    def _run_interface(self, runtime):
        from pynets.fmri import clustools
        nilearn_clust_list = ['kmeans', 'ward', 'complete', 'average']

        nip = clustools.NilParcellate(self.inputs.func_file,
                                      self.inputs.clust_mask,
                                      self.inputs.k,
                                      self.inputs.clust_type,
                                      self.inputs.conf,
                                      self.inputs.vox_size,
                                      self.inputs.local_corr)

        uatlas, atlas = nip.create_clean_mask()
        nip.create_local_clustering()

        if self.inputs.clust_type in nilearn_clust_list:
            nip.parcellate()
        else:
            raise ValueError('Clustering method not recognized. '
                             'See: https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.html#nilearn.'
                             'regions.Parcellations')

        self._results['atlas'] = atlas
        self._results['uatlas'] = uatlas
        self._results['clust_mask'] = self.inputs.clust_mask
        self._results['k'] = self.inputs.k
        self._results['clust_type'] = self.inputs.clust_type
        self._results['clustering'] = True
        return runtime
