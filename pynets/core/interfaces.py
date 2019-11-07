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
    binary = traits.Any(mandatory=False)


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
    plot_switch = traits.Any(mandatory=True)
    multi_nets = traits.Any(mandatory=True)
    multimodal = traits.Any(mandatory=True)


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
        from pynets.utils import collect_pandas_df
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
