"""Configuration file for pytest for pynets."""
import os
import sys
import pytest
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams['figure.dpi'] = 100
import tempfile
import pkg_resources
import numpy as np
import nibabel as nib
if sys.platform.startswith('win') is False:
    import indexed_gzip
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from pynets.core.nodemaker import get_names_and_coords_of_parcels
from graspologic.simulations.simulations import er_nm, sbm
from graspologic.utils import symmetrize, remove_loops, \
    largest_connected_component
from nilearn._utils import data_gen


@pytest.fixture(scope='package')
def random_mni_roi_data():
    roi_img = data_gen.generate_mni_space_img(res=2)[1]
    roi_file_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    roi_file = str(roi_file_tmp.name)
    roi_img.to_filename(roi_file)

    yield {'roi_file': roi_file}
    roi_file_tmp.close()


@pytest.fixture(scope='package')
def gen_mat_data():
    def _gen_mat_data(n: int=20, m: int=20, p: int=0.50,
                      mat_type: str='sb', binary: bool=False,
                      asfile: bool=True, n_graphs: int=1):
        if binary is True:
            wt = 1
        else:
            wt = np.random.uniform

        mat_list = []
        mat_file_list = []
        for nm in range(n_graphs):
            if mat_type == 'er':
                mat = largest_connected_component(symmetrize(
                    remove_loops(er_nm(n, m, wt=np.random.uniform,
                                       wtargs=dict(low=0, high=1)))))
            elif mat_type == 'sb':
                if p is None:
                    raise ValueError(
                        f"for mat_type {mat_type}, p cannot be None")
                mat = largest_connected_component(symmetrize(
                    remove_loops(sbm(np.array([n]), np.array([[p]]),
                                     wt=wt, wtargs=dict(low=0,
                                                        high=1)))))
            else:
                raise ValueError(f"mat_type {mat_type} not recognized!")

            mat_list.append(mat)

            if asfile is True:
                mat_path_tmp = tempfile.NamedTemporaryFile(mode='w+',
                                                           suffix='.npy',
                                                           delete=False)
                mat_path = str(mat_path_tmp.name)
                np.save(mat_path, mat)
                mat_file_list.append(mat_path)
                mat_path_tmp.close()

        return {'mat_list': mat_list, 'mat_file_list': mat_file_list}

    return _gen_mat_data


@pytest.fixture(scope='package')
def connectivity_data():
    """Fixture for connectivity tests."""

    base_dir = os.path.abspath(pkg_resources.resource_filename(
        "pynets", "../data/examples"))
    func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/" \
                f"sub-25659_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    mask_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/" \
                f"sub-25659_ses-1_task-rest_space-T1w_desc-preproc_" \
                f"bold_mask.nii.gz"
    parcellation = pkg_resources.resource_filename(
        "pynets", "templates/atlases/DesikanKlein2012.nii.gz"
    )

    masker = NiftiLabelsMasker(
        labels_img=nib.load(parcellation), background_label=0,
        resampling_target="labels", dtype="auto",
        mask_img=nib.load(mask_file), standardize=True)

    time_series = masker.fit_transform(func_file)
    conn_measure = ConnectivityMeasure(
        kind="correlation")
    conn_matrix = conn_measure.fit_transform([time_series])[0]
    [coords, _, _, label_intensities] = \
        get_names_and_coords_of_parcels(parcellation)

    labels = ['ROI_' + str(idx) for idx, val in enumerate(label_intensities)]

    yield {'time_series': time_series, 'conn_matrix': conn_matrix,
           'labels': labels, 'coords': coords, 'indices': label_intensities}

