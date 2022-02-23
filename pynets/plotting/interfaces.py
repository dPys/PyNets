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
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    traits,
    SimpleInterface,
    Directory,
)

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


class _PlotStructInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotStruct"""

    conn_matrix = traits.Any()
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any(mandatory=False)
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    subnet = traits.Any(mandatory=True)
    labels = traits.Array(mandatory=True)
    roi = traits.Any(mandatory=True)
    coords = traits.Array(mandatory=True)
    thr = traits.Any(mandatory=True)
    node_radius = traits.Any(mandatory=True)
    edge_threshold = traits.Any(mandatory=True)
    prune = traits.Any(mandatory=True)
    parcellation = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=True)
    binary = traits.Bool(mandatory=True)
    track_type = traits.Any(mandatory=True)
    traversal = traits.Any(mandatory=True)
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
        from pynets.plotting import brain

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
            brain.plot_all_struct(
                self.inputs.conn_matrix,
                self.inputs.conn_model,
                self.inputs.atlas,
                self.inputs.dir_path,
                self.inputs.ID,
                self.inputs.subnet,
                self.inputs.labels.tolist(),
                self.inputs.roi,
                [tuple(coord) for coord in self.inputs.coords.tolist()],
                self.inputs.thr,
                self.inputs.node_radius,
                self.inputs.edge_threshold,
                self.inputs.prune,
                self.inputs.parcellation,
                self.inputs.norm,
                self.inputs.binary,
                self.inputs.track_type,
                self.inputs.traversal,
                self.inputs.min_length,
                self.inputs.error_margin
            )

        self._results["out"] = "None"

        return runtime


class _PlotFuncInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for PlotFunc"""

    conn_matrix = traits.Any()
    conn_model = traits.Str(mandatory=True)
    atlas = traits.Any(mandatory=False)
    dir_path = Directory(exists=True, mandatory=True)
    ID = traits.Any(mandatory=True)
    subnet = traits.Any(mandatory=True)
    labels = traits.Array(mandatory=True)
    roi = traits.Any(mandatory=True)
    coords = traits.Array(mandatory=True)
    thr = traits.Any(mandatory=True)
    node_radius = traits.Any(mandatory=True)
    edge_threshold = traits.Any(mandatory=True)
    smooth = traits.Any(mandatory=True)
    prune = traits.Any(mandatory=True)
    parcellation = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=True)
    binary = traits.Bool(mandatory=True)
    hpass = traits.Any(mandatory=True)
    signal = traits.Any(mandatory=True)
    edge_color_override = traits.Bool(mandatory=True)


class _PlotFuncOutputSpec(BaseInterfaceInputSpec):
    """Output interface wrapper for PlotFunc"""

    out = traits.Str


class PlotFunc(SimpleInterface):
    """Interface wrapper for PlotFunc"""

    input_spec = _PlotFuncInputSpec
    output_spec = _PlotFuncOutputSpec

    def _run_interface(self, runtime):
        from pynets.plotting import brain

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
            brain.plot_all_func(
                self.inputs.conn_matrix,
                self.inputs.conn_model,
                self.inputs.atlas,
                self.inputs.dir_path,
                self.inputs.ID,
                self.inputs.subnet,
                self.inputs.labels.tolist(),
                self.inputs.roi,
                [tuple(coord) for coord in self.inputs.coords.tolist()],
                self.inputs.thr,
                self.inputs.node_radius,
                self.inputs.edge_threshold,
                self.inputs.smooth,
                self.inputs.prune,
                self.inputs.parcellation,
                self.inputs.norm,
                self.inputs.binary,
                self.inputs.hpass,
                self.inputs.signal,
                self.inputs.edge_color_override,
            )

        self._results["out"] = "None"

        return runtime
