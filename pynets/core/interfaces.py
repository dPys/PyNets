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
    traits,
    SimpleInterface,
)

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


class _FetchNodesLabelsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for FetchNodesLabels"""

    atlas = traits.Any(mandatory=False)
    parcellation = traits.Any(mandatory=False)
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
    parcellation = traits.Any()
    dir_path = traits.Any()


class FetchNodesLabels(SimpleInterface):
    """Interface wrapper for FetchNodesLabels."""

    input_spec = _FetchNodesLabelsInputSpec
    output_spec = _FetchNodesLabelsOutputSpec

    def _run_interface(self, runtime):
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

        if self.inputs.parcellation is None and self.inputs.atlas in \
                nilearn_parc_atlases:
            [labels, networks_list,
             parcellation] = nodemaker.nilearn_atlas_helper(
                self.inputs.atlas, self.inputs.parc
            )
            if parcellation:
                if not isinstance(parcellation, str):
                    nib.save(
                        parcellation, f"{runtime.cwd}"
                                      f"{self.inputs.atlas}{'.nii.gz'}")
                    parcellation = f"{runtime.cwd}" \
                                   f"{self.inputs.atlas}{'.nii.gz'}"
                if self.inputs.clustering is False:
                    [parcellation,
                     labels] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                        parcellation, label_names=labels)
                [coords, atlas, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(parcellation)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(parcellation)
                else:
                    parcel_list = None
            else:
                raise FileNotFoundError(
                    f"\nAtlas file for {self.inputs.atlas} not found!"
                )

            atlas = self.inputs.atlas
        elif (
            self.inputs.parcellation is None
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
            parcellation = None
            label_intensities = None
        elif (
            self.inputs.parcellation is None
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
            [labels, networks_list,
             parcellation] = nodemaker.nilearn_atlas_helper(
                self.inputs.atlas, self.inputs.parc
            )
            coords = find_probabilistic_atlas_cut_coords(maps_img=parcellation)
            if parcellation:
                if not isinstance(parcellation, str):
                    nib.save(
                        parcellation, f"{runtime.cwd}"
                                      f"{self.inputs.atlas}{'.nii.gz'}")
                    parcellation = f"{runtime.cwd}" \
                                   f"{self.inputs.atlas}{'.nii.gz'}"
                if self.inputs.clustering is False:
                    [parcellation,
                     labels] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                        parcellation, label_names=labels)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(parcellation)
                else:
                    parcel_list = None
            else:
                raise FileNotFoundError(
                    f"\nAtlas file for {self.inputs.atlas} not found!")

            par_max = None
            atlas = self.inputs.atlas
            label_intensities = None
        elif self.inputs.parcellation is None and self.inputs.atlas in \
            local_atlases:
            parcellation_pre = (
                f"{str(Path(base_path).parent.parent)}/templates/atlases/"
                f"{self.inputs.atlas}.nii.gz"
            )
            parcellation = fname_presuffix(
                parcellation_pre, newpath=runtime.cwd)
            copyfile(parcellation_pre, parcellation, copy=True,
                     use_hardlink=False)
            try:
                par_img = nib.load(parcellation)
            except indexed_gzip.ZranError as e:
                print(e,
                      "\nCannot load subnetwork reference image. "
                      "Do you have git-lfs installed?")
            try:
                if self.inputs.clustering is False:
                    [parcellation, _] = \
                        nodemaker.enforce_hem_distinct_consecutive_labels(
                            parcellation)

                # Fetch user-specified atlas coords
                [coords, _, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(parcellation)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(parcellation)
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
        elif self.inputs.parcellation:
            if self.inputs.clustering is True:
                while True:
                    if op.isfile(self.inputs.parcellation):
                        break
                    else:
                        print("Waiting for atlas file...")
                        time.sleep(5)

            try:
                parcellation_tmp_path = fname_presuffix(
                    self.inputs.parcellation, newpath=runtime.cwd
                )
                copyfile(
                    self.inputs.parcellation,
                    parcellation_tmp_path,
                    copy=True,
                    use_hardlink=False)
                # Fetch user-specified atlas coords
                if self.inputs.clustering is False:
                    [parcellation,
                     _] = nodemaker.enforce_hem_distinct_consecutive_labels(
                        parcellation_tmp_path)
                else:
                    parcellation = parcellation_tmp_path
                [coords, atlas, par_max, label_intensities] = \
                    nodemaker.get_names_and_coords_of_parcels(parcellation)
                if self.inputs.parc is True:
                    parcel_list = nodemaker.gen_img_list(parcellation)
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
                        f"labels/"
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
              f"{textwrap.shorten(str(labels), width=1000, placeholder='...')}"
              f"")

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
        del parcel_list
        out_path = f"{runtime.cwd}/parcel_list.nii.gz"
        nib.save(parcel_list_4d, out_path)
        self._results["parcel_list"] = out_path
        self._results["par_max"] = par_max
        self._results["parcellation"] = parcellation
        self._results["dir_path"] = dir_path

        return runtime


class CombineOutputsInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for CombineOutputs"""

    ID = traits.Any(mandatory=True)
    subnet = traits.Any(mandatory=False)
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
            self.inputs.subnet,
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
