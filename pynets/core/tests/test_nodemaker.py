#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
"""
import pytest
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams['figure.dpi'] = 100
import numpy as np
import time
import nibabel as nib
import os

try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import logging
import pkg_resources
import tempfile
import shutil
from pathlib import Path
from pynets.core import nodemaker
from nilearn._utils import data_gen


logger = logging.getLogger(__name__)
logger.setLevel(50)


@pytest.mark.parametrize(
    "atlas",
    [
        "atlas_aal",
        "atlas_talairach_gyrus",
        "atlas_talairach_ba",
        "atlas_talairach_lobe",
        "atlas_harvard_oxford",
        "atlas_destrieux_2009",
    ],
)
def test_nilearn_atlas_helper(atlas):
    parc = False
    [labels, networks_list, parlistfile] = \
        nodemaker.nilearn_atlas_helper(atlas, parc)
    print(labels)
    print(networks_list)
    print(parlistfile)
    assert labels is not None
    if (
        (parlistfile is not None)
        and isinstance(labels[0], str)
        and isinstance(parlistfile, str)
        and (atlas != "atlas_aal")
        and os.path.isfile(parlistfile)
    ):
        parcel_data = nib.load(parlistfile).get_fdata()
        assert (
            len(labels) == len(np.unique(parcel_data))
            or len(labels) == len(np.unique(parcel_data)) - 1
            or len(labels) - 1 == len(np.unique(parcel_data))
            or float(2 * len(labels)) == float(len(np.unique(parcel_data)) - 1)
            or float(2 * (len(labels)) - 1) == len(np.unique(parcel_data))
        )


def test_nodemaker_tools_parlistfile_RSN():
    """
    Test nodemaker_tools_parlistfile_RSN functionality
    """
    # Set example inputs
    template = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"
    subnet = "Default"
    parc = True

    start_time = time.time()
    coords = nodemaker.get_names_and_coords_of_parcels(parlistfile)[0]
    print(
        "%s%s%s"
        % (
            "get_names_and_coords_of_parcels --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0
                                        ].tolist()

    start_time = time.time()

    parcel_list = nodemaker.gen_img_list(parlistfile)

    [net_coords, net_parcel_list, net_labels, subnet] = \
        nodemaker.get_node_membership(subnet, template, coords, labels,
                                      parc, parcel_list
    )
    print(
        "%s%s%s"
        % (
            "get_node_membership --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(
        net_parcel_list
    )
    print(
        "%s%s%s"
        % (
            "create_parcel_atlas --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(parlistfile, subnet, net_labels,
                                             dir_path)
    print(
        "%s%s%s"
        % (
            "gen_network_parcels --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert coords is not None
    assert net_coords is not None
    assert net_labels is not None
    assert net_parcel_list is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert subnet is not None


@pytest.mark.parametrize("atlas", ["coords_dosenbach_2010",
                                   "coords_power_2011"])
def test_nodemaker_tools_nilearn_coords_RSN(atlas):
    """
    Test nodemaker_tools_nilearn_coords_RSN functionality
    """
    # Set example inputs
    template = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz"
    )
    subnet = "Default"
    parc = False
    parcel_list = None
    start_time = time.time()
    [coords, _, _, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print(
        "%s%s%s"
        % (
            "fetch_nilearn_atlas_coords --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [net_coords, _, net_labels, subnet] = nodemaker.get_node_membership(
        subnet, template, coords, labels, parc, parcel_list
    )
    print(
        "%s%s%s"
        % (
            "get_node_membership --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert coords is not None
    assert labels is not None
    assert net_coords is not None
    assert net_labels is not None
    assert subnet is not None


def test_nodemaker_tools_masking_parlistfile_RSN():
    """
    Test nodemaker_tools_masking_parlistfile_RSN functionality
    """
    # Set example inputs

    template = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz"
    )

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"

    roi = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_mni_space_img()[1].to_filename(roi)

    subnet = "Default"
    ID = "002"
    perc_overlap = 0.10
    parc = True

    start_time = time.time()
    [coords, _, _, _] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print(
        "%s%s%s"
        % (
            "get_names_and_coords_of_parcels --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0
                                        ].tolist()

    start_time = time.time()
    parcel_list = nodemaker.gen_img_list(parlistfile)
    [net_coords, net_parcel_list, net_labels, subnet] = \
        nodemaker.get_node_membership(
        subnet, template, coords, labels, parc, parcel_list
    )
    print(
        "%s%s%s"
        % (
            "get_node_membership --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [
        net_coords_masked,
        net_labels_masked,
        net_parcel_list_masked,
    ] = nodemaker.parcel_masker(
        roi,
        net_coords,
        net_parcel_list,
        net_labels,
        dir_path,
        ID,
        perc_overlap,
        vox_size="2mm",
    )
    print(
        "%s%s%s"
        % (
            "parcel_masker --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(
        net_parcel_list_masked
    )
    print(
        "%s%s%s"
        % (
            "create_parcel_atlas --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(
        parlistfile, subnet, net_labels_masked, dir_path
    )
    print(
        "%s%s%s"
        % (
            "gen_network_parcels --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert coords is not None
    assert net_coords is not None
    assert net_labels is not None
    assert net_parcel_list is not None
    assert net_coords_masked is not None
    assert net_labels_masked is not None
    assert net_parcel_list_masked is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert subnet is not None
    tmp.cleanup()


@pytest.mark.parametrize("atlas", ["coords_dosenbach_2010",
                                   "coords_power_2011"])
def test_nodemaker_tools_masking_coords_RSN(atlas):
    """
    Test nodemaker_tools_masking_coords_RSN functionality
    """
    # Set example inputs
    template = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    roi = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_mni_space_img()[1].to_filename(roi)

    subnet = "Default"
    parc = False
    parcel_list = None
    error = 2
    start_time = time.time()
    [coords, _, _, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print(
        "%s%s%s"
        % (
            "fetch_nilearn_atlas_coords (Masking subnet version) --> "
            "finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [net_coords, _, net_labels, subnet] = nodemaker.get_node_membership(
        subnet, template, coords, labels, parc, parcel_list
    )
    print(
        "%s%s%s"
        % (
            "get_node_membership (Masking subnet version) --> " "finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [net_coords_masked, net_labels_masked] = nodemaker.coords_masker(
        roi, net_coords, net_labels, error
    )
    print(
        "%s%s%s"
        % (
            "coords_masker (Masking subnet version) --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert coords is not None
    assert net_coords is not None
    assert net_coords_masked is not None
    assert net_labels is not None
    assert net_labels_masked is not None
    assert subnet is not None


def test_nodemaker_tools_parlistfile_WB():
    """
    Test nodemaker_tools_parlistfile_WB functionality
    """
    # Set example inputs
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"

    start_time = time.time()
    [WB_coords, _, _, _] = nodemaker.get_names_and_coords_of_parcels(
        parlistfile)
    print(
        "%s%s%s"
        % (
            "get_names_and_coords_of_parcels (User-atlas "
            "whole-brain version) --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    WB_labels = np.arange(len(WB_coords) + 1)[
        np.arange(len(WB_coords) + 1) != 0
    ].tolist()

    start_time = time.time()

    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(
        WB_parcel_list
    )
    print(
        "%s%s%s"
        % (
            "create_parcel_atlas (User-atlas whole-brain version) "
            "--> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert WB_coords is not None
    assert WB_labels is not None
    assert WB_parcel_list is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None


@pytest.mark.parametrize("atlas", ["coords_dosenbach_2010",
                                   "coords_power_2011"])
def test_nodemaker_tools_nilearn_coords_WB(atlas):
    """
    Test nodemaker_tools_nilearn_coords_WB functionality
    """
    start_time = time.time()
    [WB_coords, _, _, WB_labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print(
        "%s%s%s"
        % (
            "fetch_nilearn_atlas_coords (Whole-brain version) --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert WB_coords is not None
    assert WB_labels is not None


def test_nodemaker_tools_masking_parlistfile_WB():
    """
    Test nodemaker_tools_masking_parlistfile_WB functionality
    """
    # Set example inputs
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"

    atlas = "whole_brain_cluster_labels_PCA200"
    roi = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_mni_space_img()[1].to_filename(roi)
    ID = "002"
    parc = True
    perc_overlap = 0.10

    start_time = time.time()
    [WB_coords, _, _, _] = nodemaker.get_names_and_coords_of_parcels(
        parlistfile)
    print(
        "%s%s%s"
        % (
            "get_names_and_coords_of_parcels (Masking whole-brain "
            "version) --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    WB_labels = np.arange(len(WB_coords) + 1)[
        np.arange(len(WB_coords) + 1) != 0
    ].tolist()

    start_time = time.time()

    WB_parcel_list = nodemaker.gen_img_list(parlistfile)

    start_time = time.time()
    [
        WB_net_parcels_map_nifti_unmasked,
        WB_coords_unmasked,
        _,
        _,
        _,
        dir_path,
    ] = nodemaker.node_gen(
        WB_coords, WB_parcel_list, WB_labels, dir_path, ID, parc, atlas,
        parlistfile
    )
    print(
        "%s%s%s"
        % (
            "node_gen (Masking whole-brain version) --> finished: ",
            np.round(time.time() - start_time, 1),
            "s",
        )
    )

    start_time = time.time()
    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(
        WB_parcel_list
    )
    print(
        "%s%s%s"
        % (
            "create_parcel_atlas (Masking whole-brain version) --> finished: ",
            np.round(time.time() - start_time, 1),
            "s",
        )
    )

    start_time = time.time()
    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [
        WB_net_parcels_map_nifti_masked,
        WB_coords_masked,
        WB_labels_masked,
        _,
        _,
        _,
    ] = nodemaker.node_gen_masking(
        roi,
        WB_coords,
        WB_parcel_list,
        WB_labels,
        dir_path,
        ID,
        parc,
        atlas,
        parlistfile,
        vox_size="2mm",
    )

    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [_, _, WB_parcel_list_masked] = nodemaker.parcel_masker(
        roi,
        WB_coords,
        WB_parcel_list,
        WB_labels,
        dir_path,
        ID,
        perc_overlap,
        vox_size="2mm",
    )
    print(
        "%s%s%s"
        % (
            "parcel_masker (Masking whole-brain version) --> " "finished: ",
            np.round(time.time() - start_time, 1),
            "s",
        )
    )

    print(
        "%s%s%s"
        % (
            "node_gen_masking (Masking whole-brain version) --> " "finished: ",
            np.round(time.time() - start_time, 1),
            "s",
        )
    )

    assert WB_coords is not None
    assert WB_labels is not None
    assert WB_parcel_list is not None
    assert WB_coords_masked is not None
    assert WB_labels_masked is not None
    assert WB_parcel_list_masked is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert WB_net_parcels_map_nifti_unmasked is not None
    assert WB_coords_unmasked is not None
    assert WB_net_parcels_map_nifti_masked is not None
    assert WB_coords_masked is not None


@pytest.mark.parametrize("atlas", ["coords_dosenbach_2010",
                                   "coords_power_2011"])
def test_nodemaker_tools_masking_coords_WB(atlas):
    """
    Test nodemaker_tools_masking_coords_WB functionality
    """
    # Set example inputs
    roi = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_mni_space_img()[1].to_filename(roi)
    error = 2

    start_time = time.time()
    [WB_coords, _, _, WB_labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print(
        "%s%s%s"
        % (
            "fetch_nilearn_atlas_coords (Masking whole-brain "
            "coords version) --> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    start_time = time.time()
    [WB_coords_masked, WB_labels_masked] = nodemaker.coords_masker(
        roi, WB_coords, WB_labels, error
    )
    print(
        "%s%s%s"
        % (
            "coords_masker (Masking whole-brain coords version) "
            "--> finished: ",
            str(np.round(time.time() - start_time, 1)),
            "s",
        )
    )

    assert WB_coords is not None
    assert WB_coords is not None
    assert WB_coords_masked is not None
    assert WB_labels is not None
    assert WB_labels_masked is not None


def test_create_spherical_roi_volumes():
    """
    Test create_spherical_roi_volumes functionality
    """
    node_size = 2
    template_mask = pkg_resources.resource_filename(
        "pynets", "templates/standard/MNI152_T1_brain_mask_2mm.nii.gz"
    )
    [parcel_list, _, _, _] = nodemaker.create_spherical_roi_volumes(
        node_size, [(0, 0, 0), (5, 5, 5)], template_mask
    )
    assert len([i for i in parcel_list]) > 0


def test_get_sphere():
    """
    Test get_sphere functionality
    """
    img_file = pkg_resources.resource_filename(
        "pynets", f"templates/standard/MNI152_T1_brain_2mm.nii.gz"
    )
    img = nib.load(img_file)
    r = 4
    vox_dims = (2.0, 2.0, 2.0)
    coords = [[0, 0, 0], [-5, -5, -5], [5, 5, 5], [-10, -10, -10],
              [10, 10, 10]]
    neighbors = []
    for coord in coords:
        neighbors.append(nodemaker.get_sphere(coord, r, vox_dims,
                                              img.shape[0:3]))
    neighbors = [i for i in neighbors if len(i) > 0]
    assert len(neighbors) == 3


def test_parcel_naming():
    """
    Test parcel_namiing functionality
    """
    coords = [(0, 0, 0), (-5, -5, -5), (5, 5, 5), (-10, -10, -10),
              (10, 10, 10)]
    labels = nodemaker.parcel_naming(coords, vox_size="2mm")
    assert len(coords) == len(labels)


def test_enforce_hem_distinct_consecutive_labels():
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"

    parcellation = nodemaker.enforce_hem_distinct_consecutive_labels(
        parlistfile)[0]
    parcellation_img = nib.load(parcellation)
    parcels_parcellation = len(np.unique(parcellation_img.get_fdata())) - 1
    assert parcels_parcellation == 354


def test_drop_coords_labels_from_restricted_parcellation():
    from nipype.utils.filemanip import copyfile

    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )

    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"

    [coords, _, _, label_intensities] = \
        nodemaker.get_names_and_coords_of_parcels(parlistfile
    )

    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0
                                        ].tolist()
    labs = list(zip(labels, label_intensities))
    [
        parcellation_okay,
        cleaned_coords,
        cleaned_labels,
    ] = nodemaker.drop_coords_labels_from_restricted_parcellation(
        parlistfile, coords, labs
    )

    parcellation_okay_img = nib.load(parcellation_okay)
    intensities_ok = list(
        np.unique(np.asarray(parcellation_okay_img.dataobj).astype("int"))[1:]
    )

    assert len(cleaned_coords) == len(cleaned_labels) == len(intensities_ok)

    parlist_img = nib.load(parlistfile)
    parlist_img_data = parlist_img.get_fdata()
    parlist_img_data[np.where(parlist_img_data == 10)] = 0
    par_tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz")
    nib.save(nib.Nifti1Image(parlist_img_data, affine=parlist_img.affine),
             par_tmp.name)
    [
        parcellation_okay,
        cleaned_coords,
        cleaned_labels,
    ] = nodemaker.drop_coords_labels_from_restricted_parcellation(
        parlistfile, coords, labs
    )
    parcellation_okay_img = nib.load(parcellation_okay)
    intensities_ok = list(
        np.unique(np.asarray(parcellation_okay_img.dataobj).astype("int"))[1:]
    )

    assert len(cleaned_coords) == len(cleaned_labels) == len(intensities_ok)

    bad_coords = np.delete(coords, 30, axis=0)
    del labs[-30]
    par_tmp2 = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    copyfile(parlistfile, par_tmp2, copy=True, use_hardlink=False)
    [
        parcellation_mod,
        cleaned_coords,
        cleaned_labels,
    ] = nodemaker.drop_coords_labels_from_restricted_parcellation(
        par_tmp2, bad_coords, labs
    )
    parcellation_mod_img = nib.load(parcellation_mod)
    intensities_ok = list(
        np.unique(np.asarray(parcellation_mod_img.dataobj).astype("int"))[1:]
    )

    assert len(cleaned_coords) == len(cleaned_labels) == len(intensities_ok)


def test_mask_roi():
    """
    Test mask_roi functionality
    """
    mask = pkg_resources.resource_filename(
        "pynets", "templates/standard/MNI152_T1_brain_mask_2mm.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    func_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_fake_fmri()[1].to_filename(func_file)
    roi = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
    data_gen.generate_mni_space_img()[1].to_filename(roi)
    roi_masked = nodemaker.mask_roi(dir_path, roi, mask, func_file)
    assert roi_masked is not None


def test_drop_bad_ixs():
    parlistfile = pkg_resources.resource_filename(
        "pynets", "templates/atlases/whole_brain_cluster_labels_PCA200.nii.gz"
    )
    dir_path = str(tempfile.TemporaryDirectory().name)
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy2(parlistfile, f"{dir_path}/{os.path.basename(parlistfile)}")
    parlistfile = f"{dir_path}/{os.path.basename(parlistfile)}"
    unique_parcels = np.unique(nib.load(parlistfile).get_fdata())
    total_parcels = len(unique_parcels)
    pruned_parc = nodemaker.drop_badixs_from_parcellation(parlistfile, [3, 4],
                                                          False)
    unique_pruned_parcels = np.unique(nib.load(pruned_parc).get_fdata())
    total_pruned_parcels = len(unique_pruned_parcels)

    assert total_pruned_parcels == total_parcels - 2
    assert unique_parcels != unique_pruned_parcels
