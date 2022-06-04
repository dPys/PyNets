"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
"""
import sys
import warnings

import matplotlib
import numpy as np

if sys.platform.startswith("win") is False:
    import indexed_gzip

from pathlib import Path

import nibabel as nib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def get_sphere(coords, r, vox_dims, dims):
    """
    Return all points within r mm of coords. Generates a cube and then
    discards all points outside sphere.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    r : int
        Radius for sphere.
    vox_dims : array/tuple
        1D vector (x, y, z) of mm voxel resolution for sphere.
    dims : array/tuple
        1D vector (x, y, z) of image dimensions for sphere.

    Returns
    -------
    neighbors : list
        A list of indices, within the dimensions of the image, that fall
        within a spherical neighborhood defined by the specified error radius
        of the list of the input coordinates.

    References
    ----------
    .. [1] Tor D., W. (2011). NeuroSynth: a new platform for large-scale
     automated synthesis of human functional neuroimaging data.
     Frontiers in Neuroinformatics.
    """
    r = float(r)
    cube = np.vstack(
        [
            row.ravel()
            for row in np.mgrid[
                [
                    slice(-r / vox_dims[i], r / vox_dims[i] + 0.01, 1)
                    for i in range(len(coords))
                ]
            ]
        ]
    )
    sphere = np.round(
        cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= r].T
        + coords
    )
    return sphere[
        (np.min(sphere, 1) >= 0)
        & (np.max(np.subtract(sphere, dims), 1) <= -1),
        :,
    ].astype(int)


def create_parcel_atlas(parcels_4d_img, label_intensities=None):
    """
    Create a 3D Nifti1Image atlas parcellation of consecutive integer
    intensities from an input list of ROI's.

    Parameters
    ----------
    parcels_4d_img : Nifti1Image
        4D image stack of boolean numpy arrays or binarized Nifti1Images
        corresponding to ROI masks.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel
        intensities corresponding to ROI membership.
    parcel_list_exp : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding
        to ROI masks, prepended with a background image of zeros.
    """
    import gc

    from nilearn.image import index_img, new_img_like

    for ix in range(parcels_4d_img.shape[-1]):
        if ix == 0:
            template_image = index_img(parcels_4d_img, ix)
            template_affine = template_image.affine
            template_shape = template_image.shape
            template_image.uncache()
            concatted_parcels = np.asarray(
                new_img_like(
                    template_image, np.zeros(template_shape, dtype=bool)
                ).dataobj
            )[:, :, :, np.newaxis]
        concatted_parcels = np.append(
            concatted_parcels,
            np.asarray(index_img(parcels_4d_img, ix).dataobj)[
                :, :, :, np.newaxis
            ],
            axis=3,
        )
        gc.collect()

    if label_intensities is not None:
        parcel_values = np.array([0] + label_intensities).astype("float16")
    else:
        parcel_values = np.array(range(concatted_parcels.shape[-1])).astype(
            "float16"
        )

    parcel_sum = np.sum(
        parcel_values * concatted_parcels, axis=3, dtype=np.uint16
    )

    del concatted_parcels
    gc.collect()

    # Set overlapping cases to zero.
    outs = np.unique(parcel_sum[parcel_sum > np.max(parcel_values)])

    for out in outs:
        parcel_sum[parcel_sum == out] = 0

    del outs
    gc.collect()

    template_affine[3, 3] = 1
    return nib.Nifti1Image(parcel_sum, affine=template_affine), parcel_values


def fetch_nilearn_atlas_coords(atlas):
    """
    Meta-API for nilearn's coordinate atlas fetching API to retrieve any
    publically-available coordinate atlas by string name.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted coordinate atlas supported for fetching. See
        Nilearn's datasets.atlas module for more detailed reference.

    Returns
    -------
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas.
    atlas_name : str
        Name of atlas parcellation (can differ slightly from fetch API string).
    networks_list : list
        List of subnet's and their associated cooordinates, if predefined
        uniquely for a given atlas.
    labels : list
        List of string labels corresponding to atlas nodes.
    """
    from nilearn import datasets

    atlas = getattr(datasets, f"fetch_{atlas}")()
    atlas_name = atlas["description"].splitlines()[0]
    if atlas_name is None:
        atlas_name = atlas
    if "b'" in str(atlas_name):
        atlas_name = atlas_name.decode("utf-8")
    print(f"\n{atlas_name} comes with {atlas.keys()}\n")
    coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
    print(f"\nStacked atlas coords in array of shape {coords.shape}\n")
    try:
        networks_list = atlas.networks.astype("U").tolist()
    except BaseException:
        networks_list = None
    try:
        labels = np.array(
            [s.strip("b'") for s in atlas.labels.astype("U")]
        ).tolist()
    except BaseException:
        labels = np.arange(len(coords) + 1)[
            np.arange(len(coords) + 1) != 0
        ].tolist()

    if len(coords) <= 1:
        raise ValueError(
            "\nNo coords returned for specified atlas! Ensure an active"
            " internet connection."
        )

    assert len(coords) == len(labels)

    return coords, atlas_name, networks_list, labels


def nilearn_atlas_helper(atlas, parc):
    """
    Meta-API for nilearn's parcellation-based atlas fetching API to retrieve
    any publically-available parcellation-based atlas by string name.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted parcellation/label-based atlas supported for
        fetching. See Nilearn's datasets.atlas module for more detailed
        references.
    parc : bool
        Indicates whether to use the raw parcels as ROI nodes instead of
        coordinates at their center-of-mass.

    Returns
    -------
    labels : list
        List of string labels corresponding to atlas nodes.
    networks_list : list
        List of subnet's and their associated cooordinates, if predefined
        uniquely for a given atlas.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    from nilearn import datasets

    if atlas == "atlas_harvard_oxford":
        atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "atlas_name")(
            "cort-maxprob-thr0-1mm"
        )
    elif atlas == "atlas_pauli_2017":
        if parc is False:
            atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "version")(
                "prob"
            )
        else:
            atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "version")(
                "det"
            )
    elif "atlas_talairach" in atlas:
        if atlas == "atlas_talairach_lobe":
            atlas = "atlas_talairach"
            print("Fetching level: lobe...")
            atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "level")(
                "lobe"
            )
        elif atlas == "atlas_talairach_gyrus":
            atlas = "atlas_talairach"
            print("Fetching level: gyrus...")
            atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "level")(
                "gyrus"
            )
        elif atlas == "atlas_talairach_ba":
            atlas = "atlas_talairach"
            print("Fetching level: ba...")
            atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}", "level")(
                "ba"
            )
    else:
        atlas_fetch_obj = getattr(datasets, f"fetch_{atlas}")()
    if len(list(atlas_fetch_obj.keys())) > 0:
        if "maps" in list(atlas_fetch_obj.keys()):
            parcellation = atlas_fetch_obj.maps
        else:
            parcellation = None
        if "labels" in list(atlas_fetch_obj.keys()):
            try:
                labels = [i.decode("utf-8") for i in atlas_fetch_obj.labels]
            except BaseException:
                labels = [i for i in atlas_fetch_obj.labels]
        else:
            raise ValueError("No labels found.")
        if "networks" in list(atlas_fetch_obj.keys()):
            try:
                networks_list = [
                    i.decode("utf-8") for i in atlas_fetch_obj.networks
                ]
            except BaseException:
                networks_list = [i for i in atlas_fetch_obj.networks]
        else:
            networks_list = None
    else:
        raise RuntimeWarning("Extraction from nilearn datasets failed!")

    return labels, networks_list, parcellation


def mmToVox(img_affine, mmcoords):
    """
    Function to convert a list of mm coordinates to voxel coordinates.

    Parameters
    ----------
    img_affine : array
        4 x 4 2D Numpy array that is the affine of the image space that the
        coordinates inhabit.
    mmcoords : list
        List of [x, y, z] or (x, y, z) coordinates in mm-space.
    """
    return nib.affines.apply_affine(np.linalg.inv(img_affine), mmcoords)


def VoxTomm(img_affine, voxcoords):
    """
    Function to convert a list of voxel coordinates to mm coordinates.

    Parameters
    ----------
    img_affine : array
        4 x 4 2D Numpy array that is the affine of the image space that the
        coordinates inhabit.
    voxcoords : list
        List of [x, y, z] or (x, y, z) coordinates in voxel-space.
    """
    return nib.affines.apply_affine(img_affine, voxcoords)


def get_node_membership(
    subnet,
    infile,
    coords,
    labels,
    parc,
    parcels_4d,
    perc_overlap=0.75,
    error=4,
):
    """
    Evaluate the affinity of any arbitrary list of coordinate or parcel nodes
    for a user-specified subnet based on Yeo-7 or Yeo-17 definitions.

    Parameters
    ----------
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
         used to filter nodes in the study of brain subgraphs.
    infile : str
        File path to Nifti1Image object whose affine will provide resampling
        reference for evaluation spatial proximity. Typically, this is an
        MNI-space template image.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each
        parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    parcels_4d : list
        List of 3D boolean numpy arrays or binarized Nifti1Images
        corresponding to ROI masks.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a
        spatial error cushion in the case of evaluating subnet membership from
        a given list of parcel masks. Default is 0.75.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a
        spatial error cushion in the case of evaluating subnet membership from
        a given list of coordinates. Default is 4.

    Returns
    -------
    coords_mm : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity
        for the specified subnet.
    RSN_parcels : list
        Filtered list of 3D boolean numpy arrays or binarized Nifti1Images
        corresponding to ROI masks with a spatial affinity for the
        specified subnet.
    net_labels : list
        Filtered list of string labels corresponding to ROI nodes with a
        spatial affinity for the specified subnet.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.

    References
    ----------
    .. [1] Thomas Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R.,
      Lashkari, D., Hollinshead, M., … Buckner, R. L. (2011). The organization
      of the human cerebral cortex estimated by intrinsic functional
      connectivity. Journal of Neurophysiology.
      https://doi.org/10.1152/jn.00338.2011
    .. [2] Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ,
      Eickhoff SB, Yeo BTT. Local-Global parcellation of the human cerebral
      cortex from intrinsic functional connectivity MRI, Cerebral Cortex,
      29:3095-3114, 2018.
    """
    import gc
    import sys
    import tempfile

    import pandas as pd
    import pkg_resources
    from nilearn.image import index_img, resample_to_img

    from pynets.core.nodemaker import (
        VoxTomm,
        create_parcel_atlas,
        get_sphere,
        mmToVox,
        three_to_four_parcellation,
    )

    if sys.platform.startswith("win") is False:
        try:
            template_img = nib.load(infile)
        except indexed_gzip.ZranError as e:
            print(
                e,
                f"\nCannot load MNI reference. Do you have git-lfs "
                f"installed?",
            )
    else:
        try:
            template_img = nib.load(infile)
        except ImportError as e:
            print(
                e,
                f"\nCannot load MNI reference. Do you have git-lfs "
                f"installed?",
            )

    bna_aff = template_img.affine

    x_vox = np.diagonal(bna_aff[:3, 0:3])[0]
    y_vox = np.diagonal(bna_aff[:3, 0:3])[1]
    z_vox = np.diagonal(bna_aff[:3, 0:3])[2]

    if parc is True:
        if isinstance(parcels_4d, str):
            parcels_4d_img = nib.load(parcels_4d)
            parcel_atlas = create_parcel_atlas(parcels_4d_img)[0]
        else:
            parcels_4d_img = parcels_4d
            parcel_atlas = create_parcel_atlas(parcels_4d_img)[0]

        parcel_atlas_img_res = resample_to_img(
            parcel_atlas, template_img, interpolation="nearest"
        )
        par_tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".nii.gz").name
        nib.save(parcel_atlas_img_res, par_tmp)
        parcel_list_res = three_to_four_parcellation(par_tmp)
    else:
        parcel_list_res = None

    # Determine whether input is from 17-networks or 7-networks
    seven_nets = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]
    seventeen_nets = [
        "VisCent",
        "VisPeri",
        "SomMotA",
        "SomMotB",
        "DorsAttnA",
        "DorsAttnB",
        "SalVentAttnA",
        "SalVentAttnB",
        "LimbicOFC",
        "LimbicTempPole",
        "ContA",
        "ContB",
        "ContC",
        "DefaultA",
        "DefaultB",
        "DefaultC",
        "TempPar",
    ]

    # 17 triple: SalVentAttnA 7 (6), ContB 12 (11), DefaultA 14 (13)
    if subnet in seventeen_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <= 1:
            par_file = pkg_resources.resource_filename(
                "pynets", "templates/rsns/BIGREF1mm.nii.gz"
            )
        else:
            par_file = pkg_resources.resource_filename(
                "pynets", "templates/rsns/BIGREF2mm.nii.gz"
            )

        # Grab subnet reference file
        nets_ref_txt = pkg_resources.resource_filename(
            "pynets", "templates/rsns/Schaefer2018_1000_17nets_ref.txt"
        )
    elif subnet in seven_nets:
        if x_vox <= 1 and y_vox <= 1 and z_vox <= 1:
            par_file = pkg_resources.resource_filename(
                "pynets", "templates/rsns/SMALLREF1mm.nii.gz"
            )
        else:
            par_file = pkg_resources.resource_filename(
                "pynets", "templates/rsns/SMALLREF2mm.nii.gz"
            )

        # Grab subnet reference file
        nets_ref_txt = pkg_resources.resource_filename(
            "pynets", "templates/rsns/Schaefer2018_1000_7nets_ref.txt"
        )
    else:
        nets_ref_txt = None

    if not nets_ref_txt:
        raise ValueError(
            f"subnet: {str(subnet)} not found!\nSee valid subnet names "
            f"using the `--help` flag with "
            f"`pynets`"
        )

    # Create membership dictionary
    dict_df = pd.read_csv(
        nets_ref_txt,
        sep="\t",
        header=None,
        names=["Index", "Region", "X", "Y", "Z"],
    )
    dict_df.Region.unique().tolist()
    ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}

    if sys.platform.startswith("win") is False:
        try:
            rsn_img = nib.load(par_file)
        except indexed_gzip.ZranError as e:
            print(
                e,
                f"\nCannot load subnet reference image. Do you have git-lfs "
                f"installed?",
            )
    else:
        try:
            rsn_img = nib.load(par_file)
        except ImportError as e:
            print(
                e,
                f"\nCannot load subnet reference image. "
                f"Do you have git-lfs installed?",
            )

    rsn_img_res = resample_to_img(
        rsn_img, template_img, interpolation="nearest"
    )

    RSNmask = np.asarray(rsn_img_res.dataobj)[
        :, :, :, list(ref_dict.keys())[list(ref_dict.values()).index(subnet)]
    ]

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(bna_aff, i))
    coords_vox = list(
        tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x))
        for x in coords_vox
    )

    # coords_vox = list(set(list(tuple(x) for x in coords_vox)))
    if parc is False:
        i = -1
        RSN_parcels = None
        RSN_coords_vox = []
        net_labels = []
        for coords in coords_vox:
            sphere_vol = np.zeros(RSNmask.shape, dtype=bool)
            sphere_vol[tuple(coords)] = 1
            i = i + 1
            if (RSNmask.astype("bool") & sphere_vol).any():
                print(f"{coords}{' coords falls within '}{subnet}{'...'}")
                RSN_coords_vox.append(coords)
                net_labels.append(labels[i])
                continue
            else:
                inds = get_sphere(
                    coords, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape
                )
                sphere_vol[tuple(inds.T)] = 1
                if (RSNmask.astype("bool") & sphere_vol).any():
                    print(
                        f"{coords} coords is within a + or - "
                        f"{float(error):.2f} mm neighborhood of {subnet}..."
                    )
                    RSN_coords_vox.append(coords)
                    net_labels.append(labels[i])

        coords_mm = []
        for i in RSN_coords_vox:
            coords_mm.append(VoxTomm(bna_aff, i))
        coords_mm = list(set(list(tuple(x) for x in coords_mm)))
    else:
        i = 0
        coords_with_parc = []
        net_labels = []
        for p_ix in range(parcel_list_res.shape[-1]):
            parcel_vol = np.asarray(
                index_img(parcel_list_res, p_ix).dataobj
            ).astype("bool")

            # Count number of unique voxels where overlap of parcel and mask
            # occurs
            overlap_count = len(
                np.unique(
                    np.where(
                        (RSNmask.astype("uint16") == 1)
                        & (parcel_vol.astype("uint16") == 1)
                    )
                )
            )

            # Count number of total unique voxels within the parcel
            total_count = len(
                np.unique(np.where((parcel_vol.astype("uint16") == 1)))
            )

            # Calculate % overlap
            if overlap_count > 0:
                overlap = float(overlap_count / total_count)
            else:
                print(f"No overlap of parcel {i} with subnet mask...")
                i += 1
                continue

            if overlap >= perc_overlap:
                print(
                    f"{100 * overlap:.2f}% of parcel {labels[i]} falls within"
                    f" {str(subnet)} mask..."
                )
                if p_ix == 0:
                    RSN_parcels = parcel_vol[:, :, :, np.newaxis]
                RSN_parcels = np.append(
                    RSN_parcels, parcel_vol[:, :, :, np.newaxis], axis=3
                )
                coords_with_parc.append(coords[i])
                net_labels.append(labels[i])
            i += 1
        coords_mm = list(set(list(tuple(x) for x in coords_with_parc)))
        RSN_parcels = np.delete(RSN_parcels, 0, 3)
        par_aff = parcel_list_res.affine

    rsn_img.uncache()
    template_img.uncache()

    if len(coords_mm) <= 1:
        raise ValueError(
            f"\nNo coords from the specified atlas found within"
            f" {subnet} subnet."
        )

    if type(RSN_parcels) is np.ndarray:
        assert len(coords_mm) == len(net_labels) == RSN_parcels.shape[-1]
        RSN_parcellation = nib.Nifti1Image(
            RSN_parcels.astype("int16"), affine=par_aff
        )
    else:
        assert len(coords_mm) == len(net_labels)
        RSN_parcellation = None

    return coords_mm, RSN_parcellation, net_labels, subnet


def drop_badixs_from_parcellation(parcellation, bad_idxs, enf_hemi=True):
    import os

    import nibabel as nib
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    parcellation_img = nib.load(parcellation)

    bad_idxs = sorted(list(set(bad_idxs)), reverse=True)

    parlist_img_data = parcellation_img.get_fdata()
    for val in bad_idxs:
        print(f"Removing: {str(val)}...")
        parlist_img_data[np.where(parlist_img_data == int(val))] = 0

    parcellation = fname_presuffix(
        parcellation, suffix="_pruned", newpath=os.path.dirname(parcellation)
    )
    nib.save(
        nib.Nifti1Image(parlist_img_data, affine=parcellation_img.affine),
        parcellation,
    )

    print(f"{len(np.unique(parlist_img_data))} parcels remaining")
    if enf_hemi is True:
        parcellation = enforce_hem_distinct_consecutive_labels(parcellation)[0]
    return parcellation


def parcel_masker(
    roi, coords, parcels_4d, labels, dir_path, ID, perc_overlap, vox_size
):
    """
    Evaluate the affinity of any arbitrary list of parcel nodes for a
    user-specified ROI mask.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each
        parcellation node.
    parcels_4d : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding
         to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a
        spatial error cushion in the case of evaluating ROI-mask membership
        from a given list of parcel masks.

    Returns
    -------
    coords_adj : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity
        for the specified ROI mask.
    labels_adj : list
        Filtered list of string labels corresponding to ROI nodes with a
        spatial affinity for the specified ROI mask.
    parcel_list_adj : list
        Filtered list of 3D boolean numpy arrays or binarized Nifti1Images
        corresponding to ROI masks with a spatial affinity to the specified
        ROI mask.
    """
    import sys

    import pkg_resources
    from nilearn.image import index_img, iter_img, math_img, resample_to_img

    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e, "No template specified in advanced.yaml")

    template_brain = pkg_resources.resource_filename(
        "pynets", f"templates/standard/{template_name}_brain_{vox_size}.nii.gz"
    )

    if sys.platform.startswith("win") is False:
        try:
            template_img = nib.load(template_brain)
        except indexed_gzip.ZranError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )
    else:
        try:
            template_img = nib.load(template_brain)
        except ImportError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )

    mask_data = (
        resample_to_img(
            math_img("img > 0", img=nib.load(roi)),
            template_img,
            interpolation="nearest",
        )
        .get_fdata()
        .astype("bool")
    )

    if isinstance(parcels_4d, nib.Nifti1Image):
        parcels_4d_img = resample_to_img(
            parcels_4d, template_img, interpolation="nearest"
        )
    elif isinstance(parcels_4d, str):
        parcels_4d_img = resample_to_img(
            nib.load(parcels_4d), template_img, interpolation="nearest"
        )

    i = 0
    indices = []
    for ix in range(parcels_4d_img.shape[-1]):
        # Count number of unique voxels where overlap of parcel and mask occurs
        overlap_count = len(
            np.unique(
                np.where(
                    (mask_data.astype("uint16") == 1)
                    & (
                        np.asarray(index_img(parcels_4d_img, ix).dataobj)
                        .astype("bool")
                        .astype("uint16")
                        == 1
                    )
                )
            )
        )

        # Count number of total unique voxels within the parcel
        total_count = len(
            np.unique(
                np.where(
                    (
                        np.asarray(index_img(parcels_4d_img, ix).dataobj)
                        .astype("bool")
                        .astype("uint16")
                        == 1
                    )
                )
            )
        )
        # Calculate % overlap
        if overlap_count > 0:
            overlap = float(overlap_count / total_count)
        else:
            print(f"No overlap of parcel {labels[i]} with roi" f" mask...")
            indices.append(i)
            i += 1
            continue

        if overlap >= perc_overlap:
            print(
                f"{(100 * overlap):.2f}{'% of parcel '}{labels[i]} "
                f"falls within mask..."
            )
        else:
            indices.append(i)
        i += 1

    labels_adj = list(labels)
    coords_adj = list(tuple(x) for x in coords)

    try:
        for ix in sorted(indices, reverse=True):
            print(f"{'Removing: '}{labels_adj[ix]}{' at '}{coords_adj[ix]}")
            del labels_adj[ix], coords_adj[ix]

        parcel_list_adj = np.asarray(parcels_4d_img.dataobj)
        if len(indices) > 0:
            parcel_list_adj = np.delete(
                parcel_list_adj, np.array(indices).astype("int"), 3
            )

    except RuntimeError as e:
        print(
            e,
            "Restrictive masking. No parcels remain after masking with"
            " brain mask/roi...",
        )

    parcels_4d_img.uncache()

    if not coords_adj:
        raise ValueError(
            "\nROI mask was likely too restrictive and yielded < 2"
            " remaining parcels"
        )

    assert len(coords_adj) == len(labels_adj) == parcel_list_adj.shape[-1]

    return (
        coords_adj,
        labels_adj,
        nib.Nifti1Image(parcel_list_adj, affine=parcels_4d_img.affine),
    )


def coords_masker(roi, coords, labels, error, vox_size="2mm"):
    """
    Evaluate the affinity of any arbitrary list of coordinate nodes for a
    user-specified ROI mask.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each
        parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a
        spatial error cushion in the case of evaluating the spatial affinity
        of a given list of coordinates to the given ROI mask.

    Returns
    -------
    coords : list
        Filtered list of (x, y, z) tuples in mm-space with a spatial affinity
        for the specified ROI mask.
    labels : list
        Filtered list of string labels corresponding to ROI nodes with a
        spatial affinity for the specified ROI mask.
    """
    import sys

    import nibabel as nib
    import pkg_resources
    from nilearn.image import math_img, resample_to_img

    from pynets.core.nodemaker import mmToVox
    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e, "No template specified in advanced.yaml")

    template_brain = pkg_resources.resource_filename(
        "pynets", f"templates/standard/{template_name}_brain_{vox_size}.nii.gz"
    )

    if sys.platform.startswith("win") is False:
        try:
            template_img = nib.load(template_brain)
        except indexed_gzip.ZranError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )
    else:
        try:
            template_img = nib.load(template_brain)
        except ImportError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )

    mask_img_res = resample_to_img(
        math_img("img > 0", img=nib.load(roi)),
        template_img,
        interpolation="nearest",
    )

    mask_data = mask_img_res.get_fdata().astype("bool")
    mask_aff = mask_img_res.affine
    mask_img_res.uncache()

    x_vox = np.diagonal(mask_aff[:3, 0:3])[0]
    y_vox = np.diagonal(mask_aff[:3, 0:3])[1]
    z_vox = np.diagonal(mask_aff[:3, 0:3])[2]

    #    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_aff, i))
    coords_vox = list(
        tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x))
        for x in coords_vox
    )
    # coords_vox = list(set(list(tuple(x) for x in coords_vox)))
    bad_coords = []
    for coord_vox in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coord_vox)] = 1
        if (mask_data & sphere_vol).any():
            print(f"{coord_vox}{' falls within mask...'}")
            continue
        inds = get_sphere(
            coord_vox, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape
        )
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print(
                f"{coord_vox}{' is within a + or - '}{float(error):.2f} mm"
                f" neighborhood..."
            )
            continue
        bad_coords.append(coord_vox)

    bad_coords = [x for x in bad_coords if x is not None]
    indices = []
    for bad_coords in bad_coords:
        indices.append(coords_vox.index(bad_coords))

    labels = list(labels)
    coords = list(tuple(x) for x in coords)
    try:
        for ix in sorted(indices, reverse=True):
            print(f"{'Removing: '}{labels[ix]}{' at '}{coords[ix]}")
            del labels[ix], coords[ix]
    except RuntimeError as e:
        print(
            e,
            "Restrictive masking. No coords remain after masking with"
            " brain mask/roi...",
        )

    if len(coords) <= 1:
        raise ValueError(
            "\nROI mask was likely too restrictive and yielded < 2"
            " remaining coords"
        )

    assert len(coords) == len(labels)

    return coords, labels


def get_names_and_coords_of_parcels(parcellation, background_label=0):
    """
    Return list of coordinates and max label intensity for a 3D atlas
    parcellation image.

    Parameters
    ----------
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    coords : list
        List of (x, y, z) tuples corresponding to the center-of-mass of each
        parcellation node.
    atlas : str
        An arbitrary identified for the atlas based on the filename.
    par_max : int
        The maximum label intensity in the parcellation image.
    label_intensities : list
        A list of integer label intensity values from the parcellation.
    """
    import matplotlib

    matplotlib.use("agg")
    import os.path as op

    from nilearn.plotting import find_parcellation_cut_coords

    if not op.isfile(parcellation):
        raise ValueError(
            "\nUser-specified atlas input not found! Check that "
            "the file(s) specified with the -a flag exist(s)"
        )

    [coords, label_intensities] = find_parcellation_cut_coords(
        parcellation, background_label, return_label_names=True
    )
    print(f"Parcel intensities:\n{label_intensities}")

    return (
        coords,
        parcellation.split("/")[-1].split(".")[0],
        len(coords),
        label_intensities,
    )


def three_to_four_parcellation(parcellation):
    """
    Return 4d Nifti1Image of boolean nifti masks where each masks corresponds
    to a unique atlas label for the provided atlas parcellation. Path string to
    Nifti1Image is input.

    Parameters
    ----------
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    img_list : Iterator of NiftiImages
        List of binarized Nifti1Images corresponding to ROI masks for each
        unique atlas label.
    """
    import gc
    import os.path as op

    if isinstance(parcellation, nib.Nifti1Image):
        bna_img = parcellation
    else:
        bna_img = nib.load(parcellation, mmap=True)
    parc_aff = bna_img.affine
    bna_data = np.around(
        bna_img.get_fdata(caching="fill", dtype=np.float16).astype("uint16")
    )

    # Get array of unique parcel indices
    uniq_indices = np.unique(bna_data)
    par_max = len(uniq_indices) - 1

    img_stack = np.moveaxis(
        np.array(
            [bna_data == uniq_indices[idx] for idx in range(1, par_max + 1)]
        ),
        0,
        -1,
    )
    del bna_data, uniq_indices
    gc.collect()

    return nib.Nifti1Image(img_stack.astype("int16"), affine=parc_aff)


def enforce_hem_distinct_consecutive_labels(
    parcellation, label_names=None, background_label=0
):
    """
    Check for hemispherically distinct and consecutive labels and rebuild
    parcellation.

    Parameters
    ----------
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    label_names : list
        List of string label names corresponding to ROI nodes.

    Returns
    -------
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    label_names : list
        List of string label names corresponding to ROI nodes.
    """
    import gc

    from nilearn.image import (
        concat_imgs,
        index_img,
        iter_img,
        new_img_like,
        reorder_img,
    )
    from nilearn.image.resampling import coord_transform

    labels_img = reorder_img(nib.load(parcellation))
    labels_data = labels_img.get_fdata()
    labels_affine = labels_img.affine

    parcels_4d_img = three_to_four_parcellation(labels_img)

    # Grab number of unique values in 3d image
    unique_labels = set(np.unique(labels_data)) - set([background_label])
    x, y, z = coord_transform(0, 0, 0, np.linalg.inv(labels_affine))

    if label_names is not None:
        new_lab_names = []

    for ix, lab in enumerate(unique_labels):
        cur_dat = labels_data == lab

        if ix == 0:
            template_image = index_img(parcels_4d_img, ix)
            template_affine = template_image.affine
            template_shape = template_image.shape
            template_image.uncache()
            new_labs = np.asarray(
                new_img_like(
                    template_image, np.zeros(template_shape, dtype=bool)
                ).dataobj
            )[:, :, :, np.newaxis]

        # Grab hemispheres separately
        left_hemi = labels_data.copy() == lab
        right_hemi = labels_data.copy() == lab
        left_hemi[int(x) :] = 0
        right_hemi[: int(x)] = 0

        # Two connected components in both hemispheres
        if not np.all(left_hemi == False) or np.all(right_hemi == False):
            left_lab = np.copy(cur_dat)
            right_lab = np.copy(cur_dat)
            left_lab[int(x) :] = 0
            right_lab[: int(x)] = 0
            new_labs = np.append(
                new_labs, np.asarray(left_lab)[:, :, :, np.newaxis], axis=3
            )
            new_labs = np.append(
                new_labs, np.asarray(right_lab)[:, :, :, np.newaxis], axis=3
            )
            if label_names is not None:
                new_lab_names.append(f"{label_names[ix]}_Left")
                new_lab_names.append(f"{label_names[ix]}_Right")
            del left_lab, right_lab
        else:
            new_labs = np.append(
                new_labs, np.asarray(cur_dat)[:, :, :, np.newaxis], axis=3
            )
            if label_names is not None:
                new_lab_names.append(label_names[ix])
        del left_hemi, right_hemi, cur_dat
        gc.collect()

    labels_img.uncache()
    del labels_data, labels_img

    # Enforce consecutive labelings
    atlas_img_corr = create_parcel_atlas(
        new_img_like(parcels_4d_img, new_labs)
    )[0]
    nib.save(atlas_img_corr, parcellation)

    atlas_img_corr.uncache()
    del new_labs, atlas_img_corr
    gc.collect()

    return parcellation, label_names


def drop_coords_labels_from_restricted_parcellation(
    parcellation, coords, labels
):
    # from pynets.core.utils import missing_elements
    import os

    from nipype.utils.filemanip import fname_presuffix

    print("Checking parcellation for consistency...")

    parcellation_img = nib.load(parcellation)
    intensities = [
        i
        for i in list(
            np.unique(np.asarray(parcellation_img.dataobj).astype("int"))
        )
        if i != 0
    ]

    # Correct coords and labels
    # bad_idxs = missing_elements(intensities)

    if isinstance(labels[0], tuple):
        label_intensities = [i[1] for i in labels]
        bad_idxs = []
        if len(label_intensities) != len(intensities):
            print(
                "Inconsistent number of intensities and labels. "
                "Correcting parcellation..."
            )
            diff = [
                i
                for i in list(set(label_intensities) - set(intensities))
                if str(i) != "0"
            ]
            for val in diff:
                bad_idxs.append(label_intensities.index(val))
            if len(bad_idxs) > 0:
                bad_idxs = sorted(list(set(bad_idxs)), reverse=True)
                if type(coords) is np.ndarray:
                    coords = list(tuple(x) for x in coords)
                print(f"Missing parcels at indices: {bad_idxs}")
                for j in bad_idxs:
                    print(f"Removing: {(labels[j], coords[j])}...")
                    del labels[j], coords[j]

            diff = [
                i
                for i in list(set(intensities) - set(label_intensities))
                if str(i) != "0"
            ]
            parlist_img_data = parcellation_img.get_fdata()
            for val in diff:
                print(f"Removing: {str(val)}...")
                parlist_img_data[np.where(parlist_img_data == float(val))] = 0

            parcellation = fname_presuffix(
                parcellation,
                suffix="_mod",
                newpath=os.path.dirname(parcellation),
            )
            nib.save(
                nib.Nifti1Image(
                    parlist_img_data, affine=parcellation_img.affine
                ),
                parcellation,
            )

            intensity_count = len(
                [
                    i
                    for i in np.unique(parlist_img_data.astype("int"))
                    if str(i) != "0"
                ]
            )
        else:
            intensity_count = len(intensities)
    else:
        print("Warning: Labels do not include intensity values!")
        intensity_count = len(intensities)

    try:
        assert len(coords) == len(labels) == intensity_count
    except ValueError as e:
        print(e, "Failed!")
        print(f"# Coords: {len(coords)}")
        print(f"# Labels: {len(labels)}")
        print(f"# Intensities: {intensity_count}")

    return parcellation, coords, labels


def gen_network_parcels(parcellation, subnet, labels, dir_path):
    """
    Return a modified verion of an atlas parcellation label, where labels have
    been filtered based on their spatial affinity for a specified subnet
    definition.

    Parameters
    ----------
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    subnet : str
        Resting-state subnet based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
         used to filter nodes in the study of brain subgraphs.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.

    Returns
    -------
    out_path : str
        File path to a new, subnetwork-filtered atlas parcellation Nifti1Image.
    """
    import os.path as op

    from pynets.core.nodemaker import three_to_four_parcellation

    if not op.isfile(parcellation):
        raise ValueError(
            "\nUser-specified atlas input not found! Check that "
            "the file(s) specified with the -a flag exist(s)"
        )

    print(
        f"\nExtracting parcels associated with {subnet} "
        f"subnet locations...\n"
    )
    net_parcels_sum = create_parcel_atlas(
        three_to_four_parcellation(parcellation)
    )
    parcellation_name = op.basename(parcellation).split(
        op.splitext(parcellation)[1]
    )[0]
    out_path = (
        f"{dir_path}" f"/{parcellation_name}_" f"{subnet}_parcels.nii.gz"
    )
    nib.save(net_parcels_sum[0], out_path)

    return out_path


def parcel_naming(coords, vox_size):
    """
    Perform Automated-Anatomical Labeling of each coordinate from a list of a
    voxel coordinates. This was adapted from a function of the same
    name created by Cameron Craddock and included in PyClusterROI (See:
    <https://github.com/ccraddock/cluster_roi/blob/master/parcel_naming.py>).

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in voxel-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each parcellation
        node.
    vox_size : str
        Voxel resolution (`1mm` or `2mm` stored as strings with units).

    Returns
    -------
    labels : list
        List of string labels corresponding to each coordinate's closest
        anatomical label.

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333
    .. [2] N. Tzourio-Mazoyer; B. Landeau; D. Papathanassiou; F. Crivello; O.
      Etard; N. Delcroix; Bernard Mazoyer & M. Joliot (January 2002).
      "Automated Anatomical Labeling of activations in SPM using a Macroscopic
      Anatomical Parcellation of the MNI MRI single-subject brain".
      NeuroImage. 15 (1): 273–289. doi:10.1006/nimg.2001.0978.

    """
    import sys
    from collections import defaultdict

    import nibabel as nib
    import pandas as pd
    import pkg_resources
    from nilearn.image import resample_to_img

    from pynets.core.utils import load_runconfig

    hardcoded_params = load_runconfig()
    try:
        labeling_atlases = hardcoded_params["labeling_atlases"]
    except KeyError as e:
        print(e, "No labeling atlases listed in advanced.yaml")
    try:
        template_name = hardcoded_params["template"][0]
    except KeyError as e:
        print(e, "No template specified in advanced.yaml")

    template_brain = pkg_resources.resource_filename(
        "pynets", f"templates/standard/{template_name}_brain_{vox_size}.nii.gz"
    )

    if sys.platform.startswith("win") is False:
        try:
            template_img = nib.load(template_brain)
        except indexed_gzip.ZranError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )
    else:
        try:
            template_img = nib.load(template_brain)
        except ImportError as e:
            print(
                e,
                f"\nCannot load MNI template. Do you have git-lfs "
                f"installed?",
            )

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(template_img.affine, i))
    coords_vox = list(
        tuple(map(lambda y: isinstance(y, float) and int(round(y, 0)), x))
        for x in coords_vox
    )
    coords_vox_dups = list(
        set([ele for ele in coords_vox if coords_vox.count(ele) > 1])
    )
    if len(coords_vox_dups) > 1:
        raise ValueError(
            "There should be no duplicate nodes in a " "parcellation!"
        )

    label_img_dict = defaultdict()
    for label_atlas in labeling_atlases:
        label_img_dict[label_atlas] = {}
        label_path = pkg_resources.resource_filename(
            "pynets", f"/templates/labels/" f"{label_atlas}.txt"
        )
        label_img_path = pkg_resources.resource_filename(
            "pynets", f"/templates/atlases/" f"{label_atlas}" f".nii.gz"
        )
        label_img_res = resample_to_img(
            nib.load(label_img_path),
            template_img,
            interpolation="nearest",
            copy=False,
        )
        label_img_dict[label_atlas]["affine"] = label_img_res.affine
        label_img_dict[label_atlas]["data"] = np.asarray(
            label_img_res.dataobj, dtype="uint8"
        )
        df = pd.read_csv(label_path, sep=" ", names=["region_index", "label"])
        if df["label"].isna().all():
            df = pd.read_csv(label_path, names=["label"])
            df = df[(df.label != "Background")]
            df["region_index"] = np.arange(1, len(df) + 1)
        df = df[~((df.label == "Background") & (df.region_index == 0))]
        df = df[~((df.label == "Unknown") & (df.region_index == 0))]
        label_img_dict[label_atlas]["reference"] = df

    # Create a consensus labeling dictionary
    label_dict = defaultdict()
    for coord in coords_vox:
        label_dict[coord] = {}
        for label_atlas in label_img_dict.keys():
            label_dict[coord][label_atlas] = {}
            label_dict[coord][label_atlas]["intensity"] = label_img_dict[
                label_atlas
            ]["data"][coord]
            df_ref = label_img_dict[label_atlas]["reference"]
            try:
                label_dict[coord][label_atlas]["label"] = df_ref.loc[
                    df_ref["region_index"]
                    == int(label_dict[coord][label_atlas]["intensity"])
                ]["label"].values[0]
            except BaseException:
                label_dict[coord][label_atlas]["label"] = "Unlabeled"

    new_labels = []
    for coord in label_dict.keys():
        coord_dict = {}
        for atlas, i in list(
            zip(label_dict[coord].keys(), label_dict[coord].values())
        ):
            try:
                coord_dict[atlas] = i["label"]
            except BaseException:
                continue
        new_labels.append(coord_dict)

    assert len(new_labels) == len(coords)

    return new_labels


def get_node_attributes(
    node_files, emb_shape, atlas="BrainnetomeAtlasFan2016"
):
    import ast
    import re

    from pynets.statistics.utils import parse_closest_ixs

    ixs, node_dict = parse_closest_ixs(node_files, emb_shape)

    coords = [(i["coord"]) for i in node_dict.values()]
    if isinstance(node_dict[0]["label"], str):
        labels = [
            ast.literal_eval(re.search("({.+})", i["label"]).group(0))[atlas]
            for i in node_dict.values()
        ]
    else:
        labels = [i["label"][atlas] for i in node_dict.values()]

    return coords, labels, ixs


def node_gen_masking(
    roi,
    coords,
    parcels_4d,
    labels,
    dir_path,
    ID,
    parc,
    atlas,
    parcellation,
    vox_size,
    perc_overlap=0.10,
    error=2,
):
    """
    In the case that masking was applied, this function generate nodes based
    on atlas definitions established by fetch_nodes_and_labels.

    Parameters
    ----------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each
        parcellation node.
    parcels_4d : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding
         to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas
        supported for fetching. See Nilearn's datasets.atlas module for more
        detailed reference.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    perc_overlap : float
        Value 0-1 indicating a threshold of spatial overlap to use as a spatial
         error cushion in the case of evaluating mask/subnet membership from a
         given list of parcel masks. Default is 0.75.
    error : int
        Rounded euclidean distance, in units of voxel number, to use as a
        spatial error cushion in the case of evaluating mask/subnet membership
        from a given list of coordinates. Default is 4.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer
        voxel intensities corresponding to ROI membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each parcellation
        node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas
        supported for fetching. See Nilearn's datasets.atlas module for more
        detailed reference.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    import gc

    from pynets.core.nodemaker import create_parcel_atlas, parcel_masker

    if isinstance(parcels_4d, str):
        parcels_4d_img = nib.load(parcels_4d)
    else:
        parcels_4d_img = parcels_4d

    # For parcel masking, specify overlap thresh and error cushion in mm voxels
    [coords, labels, parcels_4d_masked] = parcel_masker(
        roi,
        coords,
        parcels_4d_img,
        labels,
        dir_path,
        ID,
        perc_overlap,
        vox_size,
    )
    parcels_4d_img.uncache()

    if any(isinstance(sub, tuple) for sub in labels):
        label_intensities = [i[1] for i in labels]
    elif any(isinstance(sub, dict) for sub in labels):
        label_intensities = None
    else:
        label_intensities = labels

    net_parcels_map_nifti = create_parcel_atlas(
        parcels_4d_masked, label_intensities
    )[0]

    del parcels_4d_masked
    gc.collect()

    assert (
        len(coords)
        == len(labels)
        == len(
            [
                i
                for i in np.unique(np.asarray(net_parcels_map_nifti.dataobj))
                if i != 0
            ]
        )
    )

    return net_parcels_map_nifti, coords, labels, atlas, parcellation, dir_path


def node_gen(
    coords, parcels_4d, labels, dir_path, ID, parc, atlas, parcellation
):
    """
    In the case that masking was not applied, this function generate nodes
    based on atlas definitions established by fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each
        parcellation node.
    parcels_4d : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding
         to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas
        supported for fetching. See Nilearn's datasets.atlas module for more
        detailed reference.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel
         intensities corresponding to ROI membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each parcellation
        node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas
        supported for fetching. See Nilearn's datasets.atlas module for more
        detailed reference.
    parcellation : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    import gc

    from pynets.core.nodemaker import create_parcel_atlas

    if isinstance(parcels_4d, str):
        parcels_4d_img = nib.load(parcels_4d)
    else:
        parcels_4d_img = parcels_4d

    if any(isinstance(sub, tuple) for sub in labels):
        label_intensities = [i[1] for i in labels]
    elif any(isinstance(sub, dict) for sub in labels):
        label_intensities = None
    else:
        label_intensities = labels

    net_parcels_map_nifti = create_parcel_atlas(
        parcels_4d_img, label_intensities
    )[0]
    parcels_4d_img.uncache()
    gc.collect()

    coords = list(tuple(x) for x in coords)

    assert (
        len(coords)
        == len(labels)
        == len(
            [
                i
                for i in np.unique(np.asarray(net_parcels_map_nifti.dataobj))
                if i != 0
            ]
        )
    )

    return net_parcels_map_nifti, coords, labels, atlas, parcellation, dir_path


def mask_roi(dir_path, roi, mask, img_file):
    """
    Create derivative ROI based on intersection of roi and brain mask.

    Parameters
    ----------
    dir_path : str
        Path to directory containing subject derivative data for given run.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    mask : str
        Path to binarized/boolean brain mask Nifti1Image file.
    img_file : str
        File path to Nifti1Image to use to generate an epi-mask.

    Returns
    -------
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file,
        reduced to the spatial intersection with the input brain mask.
    """
    import os.path as op

    from nilearn import masking
    from nilearn.image import math_img, resample_img
    from nilearn.masking import intersect_masks

    img_mask_path = (
        f"{dir_path}/{op.basename(img_file).split('.')[0]}" f"_mask.nii.gz"
    )
    nib.save(masking.compute_epi_mask(img_file), img_mask_path)

    if roi and mask:
        print("Refining ROI...")
        _mask_img = nib.load(img_mask_path)
        _roi_img = nib.load(roi)
        roi_res_img = resample_img(
            _roi_img,
            target_affine=_mask_img.affine,
            target_shape=_mask_img.shape,
            interpolation="nearest",
        )
        masked_roi_img = intersect_masks(
            [
                math_img("img > 0.0", img=_mask_img),
                math_img("img > 0.0", img=roi_res_img),
            ],
            threshold=1,
            connected=False,
        )

        roi_red_path = (
            f"{dir_path}/{op.basename(roi).split('.')[0]}" f"_mask.nii.gz"
        )
        nib.save(masked_roi_img, roi_red_path)
        roi = roi_red_path

    return roi


def create_spherical_roi_volumes(node_radius, coords, template_mask):
    """
    Create volume ROI mask of spheres from a given set of coordinates and
    radius.

    Parameters
    ----------
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate
        atlas used or which represent the center-of-mass of each parcellation
        node.
    template_mask : str
        Path to binarized version of standard (MNI)-space template
        Nifti1Image file.

    Returns
    -------
    parcels_4d : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding
         to ROI masks.
    par_max : int
        The maximum label intensity in the parcellation image.
    node_radius : int
        Spherical centroid node size in the case that coordinate-based
        centroids are used as ROI's for tracking.
    parc : bool
        Indicates whether to use the raw parcels as ROI nodes instead of
        coordinates at their center-of-mass.
    """
    import gc

    from nilearn.image import concat_imgs, iter_img
    from nilearn.masking import intersect_masks

    from pynets.core.nodemaker import get_sphere, mmToVox

    mask_img = nib.load(template_mask)
    mask_aff = mask_img.affine
    mask_shape = mask_img.shape
    mask_img.uncache()

    print(f"Creating spherical ROI atlas with radius: {node_radius}")

    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(mask_aff, i))
    coords_vox = list(set(list(tuple(x) for x in coords_vox)))

    x_vox = np.diagonal(mask_aff[:3, 0:3])[0]
    y_vox = np.diagonal(mask_aff[:3, 0:3])[1]
    z_vox = np.diagonal(mask_aff[:3, 0:3])[2]

    parcel_list_all = []
    i = 1
    for coord in coords_vox:
        sphere_vol = np.zeros(mask_shape, dtype=bool)
        sphere_vol[
            tuple(
                get_sphere(
                    coord,
                    node_radius,
                    (np.abs(x_vox), y_vox, z_vox),
                    mask_shape,
                ).T
            )
        ] = (
            i * 1
        )
        parcel_list_all.append(
            nib.Nifti1Image(
                sphere_vol.astype("bool").astype("uint16"), affine=mask_aff
            )
        )
        i += 1

    # remove the intersection
    parcel_intersect = np.invert(
        np.asarray(
            intersect_masks(parcel_list_all, threshold=1).dataobj
        ).astype("bool")
    )

    parcels_4d = []
    for mask in iter_img(parcel_list_all):
        non_ovlp = np.asarray(mask.dataobj) * parcel_intersect
        parcels_4d.append(
            nib.Nifti1Image(
                non_ovlp.astype("bool").astype("uint16"), affine=mask_aff
            )
        )
    del parcel_list_all
    gc.collect()

    par_max = len(coords)
    if par_max > 0:
        parc = True
    else:
        raise ValueError("Number of nodes is zero.")

    return concat_imgs(iter_img(parcels_4d)), par_max, node_radius, parc
