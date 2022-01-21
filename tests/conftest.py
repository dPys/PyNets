"""Configuration file for pytest for pynets."""

from pathlib import Path
import os
import sys
import gc
import pytest
import tempfile
import pandas as pd
import numpy as np
import nibabel as nib
if sys.platform.startswith('win') is False:
    import indexed_gzip
from dipy.core.gradients import gradient_table
from nipype.utils.filemanip import fname_presuffix
from dipy.segment.mask import median_otsu
import dipy.data as dpd
import pickle
from nilearn._utils import as_ndarray, data_gen
from dipy.io import save_pickle
from nilearn.plotting import find_parcellation_cut_coords


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nib.Nifti1Image(data, affine), nib.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


def get_t1w():
    dpd.fetch_tissue_data()
    t1w = f"{dpd.fetch_tissue_data()[1]}/{dpd.fetch_tissue_data()[0][1]}"
    return t1w


@pytest.fixture(scope="session")
def random_mni_roi_data():
    roi_img = data_gen.generate_mni_space_img(res=2)[1]
    roi_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    roi_file = str(roi_file.name)
    roi_img.to_filename(roi_file)

    yield {'roi_file': roi_file}


@pytest.fixture(scope="session")
def dmri_estimation_data():
    """Fixture for dmri estimation tests."""

    dpd.fetcher.dipy_home = f"{os.environ['HOME']}/.dipy"
    os.makedirs(dpd.fetcher.dipy_home, exist_ok=True)
    files, folder = dpd.fetcher.fetch_stanford_hardi()
    dwi_file = f"{folder}/HARDI150.nii.gz"
    fbvals = f"{folder}/HARDI150.bval"
    fbvecs = f"{folder}/HARDI150.bvec"
    files, folder = dpd.fetcher.fetch_stanford_t1()
    t1w_file = f"{folder}/t1.nii.gz"

    files, folder = dpd.fetcher.fetch_stanford_pve_maps()
    f_pve_csf = f"{folder}/pve_csf.nii.gz"
    f_pve_gm = f"{folder}/pve_gm.nii.gz"
    f_pve_wm = f"{folder}/pve_wm.nii.gz"

    bvals = np.loadtxt(fbvals)

    b0_ixs = np.where(np.logical_and(bvals <= 50, bvals >= -50))[0]

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    b0_mask, mask = median_otsu(np.asarray(dwi_data,
                                           dtype=np.float32)[..., b0_ixs[0]],
                                median_radius=4, numpass=2)
    mask_img = nib.Nifti1Image(mask.astype(np.float32), dwi_img.affine)
    B0_mask = fname_presuffix(dwi_file, suffix="_brain_mask",
                              use_ext=True)
    nib.save(mask_img, B0_mask)

    gtab = gradient_table(bvals, np.loadtxt(fbvecs))
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0

    gtab_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.pkl')
    gtab_file = str(gtab_file.name)
    save_pickle(gtab_file, gtab)

    dwi_data_small = dwi_data.copy()
    dwi_data_small = dwi_data_small[20:50, 55:85, 38:39]
    dwi_img_small = nib.Nifti1Image(dwi_data_small, header=dwi_img.header,
                               affine=dwi_img.affine)
    dwi_file_small = fname_presuffix(dwi_file, suffix="_small", use_ext=True)
    dwi_img_small.to_filename(dwi_file_small)

    del dwi_data, dwi_data_small
    dwi_img.uncache()
    dwi_img_small.uncache()
    mask_img.uncache()
    gc.collect()

    yield {'dwi_file': dwi_file, 'fbvals': fbvals, 'fbvecs': fbvecs,
           'gtab': gtab, 'gtab_file': gtab_file,
           'dwi_file_small': dwi_file_small,
           'B0_mask': B0_mask, 't1w_file': t1w_file,
           'f_pve_csf': f_pve_csf, 'f_pve_wm': f_pve_wm, 'f_pve_gm': f_pve_gm}


@pytest.fixture(scope="session")
def fmri_estimation_data():
    """Fixture for fmri estimation tests."""
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)
    length = 30

    func_img, mask_img = generate_random_img(shape1, affine=affine1,
                                             length=length)

    func_file_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    func_file = str(func_file_tmp.name)
    func_img.to_filename(func_file)

    conf_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv')
    conf_mat = np.random.rand(length)
    conf_df = pd.DataFrame({'Conf1': conf_mat,
                            "Conf2": [np.nan]*len(conf_mat)})
    conf_file = str(conf_file.name)
    conf_df.to_csv(conf_file, sep='\t', index=False)

    # Create empty mask file
    mask_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask_file = str(mask_tmp.name)
    mask_img.to_filename(mask_file)

    t1w_file = get_t1w()

    mask_img.uncache()
    func_img.uncache()
    gc.collect()

    yield {'func_file': func_file,
           'conf_file': conf_file,
           'mask_file': mask_file,
           't1w_file': t1w_file}


@pytest.fixture(scope="session")
def parcellation_data():
    """Fixture for parcellations."""

    parcels_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    labels = pd.DataFrame(
        {"label": list(map("label_{}".format, range(16)))})['label'
    ].values.tolist()
    parcellation = data_gen.generate_labeled_regions((7, 6, 5), 16)
    parcels = parcellation.get_fdata()
    net_parcels_map_nifti_file = str(parcels_tmp.name)
    parcellation.to_filename(net_parcels_map_nifti_file)

    [coords, indices] = find_parcellation_cut_coords(
        parcellation, 0, return_label_names=True
    )
    coords = list(tuple(x) for x in coords)

    yield {'net_parcels_map_nifti_file': net_parcels_map_nifti_file,
           'parcels': parcels, 'labels': labels, 'coords': coords,
           'indices': indices}


@pytest.fixture(scope="session")
def plotting_data():
    """Fixture for plotting tests."""

    base_dir = str(Path(__file__).parent/"examples")
    conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_"
                                f"nodetype-parc_model-sps_thrtype-PROP_"
                                f"thr-0.94.txt")

    labels_file_path = f"{base_dir}/miscellaneous/Default_func_" \
                       f"labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)

    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    yield {'conn_matrix': conn_matrix, 'labels': labels, 'coords': coords}


@pytest.fixture(scope="session")
def gen_mat_data(n, m, p=None, mat_type='er', binary=False, asfile=False,
                 n_graphs=1):
    import tempfile
    from graspologic.simulations.simulations import er_nm, sbm
    from graspologic.utils import symmetrize

    if binary is True:
        wt = 1
    else:
        wt = np.random.uniform

    mat_list = []
    for nm in range(n_graphs):
        if mat_type == 'er':
            mat = symmetrize(
                er_nm(n, m, wt=np.random.uniform, wtargs=dict(low=0, high=1)))
        elif mat_type == 'sb':
            if p is None:
                raise ValueError(f"for mat_type {mat_type}, p cannot be None")
            mat = symmetrize(sbm(np.array([n]), np.array([[p]]), wt=wt,
                                 wtargs=dict(low=0, high=1)))
        else:
            raise ValueError(f"mat_type {mat_type} not recognized!")

        if asfile is True:
            mat_path = str(tempfile.NamedTemporaryFile(mode='w+',
                                                   suffix='.npy').name)
            np.save(mat_path, mat)
            mat_list.append(mat_path)
        else:
            mat_list.append(mat)

    yield mat_list

