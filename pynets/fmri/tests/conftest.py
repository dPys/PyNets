"""Configuration file for pytest for pynets."""

from pathlib import Path
import sys
import gc
import pytest
import tempfile
import pandas as pd
import numpy as np
import nibabel as nib
import dipy.data as dpd
if sys.platform.startswith('win') is False:
    import indexed_gzip
from nilearn._utils import as_ndarray, data_gen
from nilearn.plotting import find_parcellation_cut_coords


@pytest.fixture(scope='package')
def random_mni_roi_data():
    roi_img = data_gen.generate_mni_space_img(res=2)[1]
    roi_file_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    roi_file = str(roi_file_tmp.name)
    roi_img.to_filename(roi_file)

    yield {'roi_file': roi_file}
    roi_file_tmp.close()


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nib.Nifti1Image(data, affine), nib.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


@pytest.fixture(scope='package')
def fmri_estimation_data():
    """Fixture for fmri estimation tests."""

    shape1 = (45, 54, 45)
    affine1 = np.array([[1.,    0.,    0.,  -22.75],
       [0.,    1.,    0., -31.5],
       [0.,    0.,    1.,  -18.],
       [0.,    0.,    0.,    0.25]])
    length = 40

    func_img, mask_img = generate_random_img(shape1, affine=affine1,
                                             length=length)

    func_file_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    func_file = str(func_file_tmp.name)
    func_img.to_filename(func_file)

    conf_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv')
    conf_mat = np.random.rand(length)
    conf_df = pd.DataFrame({'Conf1': conf_mat,
                            "Conf2": [np.nan]*len(conf_mat)})
    conf_file = str(conf_tmp.name)
    conf_df.to_csv(conf_file, sep='\t', index=False)

    # Create empty mask file
    mask_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask_file = str(mask_tmp.name)
    mask_img.to_filename(mask_file)

    t1w_file = f"{dpd.fetch_tissue_data()[1]}/{dpd.fetch_tissue_data()[0][1]}"

    mask_img.uncache()
    func_img.uncache()
    gc.collect()

    shape2 = (20, 40, 20)
    affine2 = np.eye(4)
    length = 50

    func_img2, mask_img2 = generate_random_img(shape2, affine=affine2,
                                               length=length)

    func_file_tmp2 = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    func_file2 = str(func_file_tmp2.name)
    func_img2.to_filename(func_file2)

    mask_tmp2 = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask_file2 = str(mask_tmp2.name)
    mask_img2.to_filename(mask_file2)

    func_img2.uncache()
    mask_img2.uncache()
    gc.collect()

    yield {'func_file': func_file,
           'conf_file': conf_file,
           'mask_file': mask_file,
           't1w_file': t1w_file,
           'func_file2': func_file2,
           'mask_file2': mask_file2}
    mask_tmp.close()
    conf_tmp.close()
    func_file_tmp.close()
    func_file_tmp2.close()
    mask_tmp2.close()


@pytest.fixture(scope='package')
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
    parcels_tmp.close()
