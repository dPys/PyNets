"""Configuration file for pytest for pynets."""

from pathlib import Path
import os
import sys
import gc
import pytest
import tempfile
import pandas as pd
import numpy as np
import pkg_resources
import nibabel as nib
if sys.platform.startswith('win') is False:
    import indexed_gzip
from dipy.core.gradients import gradient_table
from nipype.utils.filemanip import fname_presuffix
from dipy.segment.mask import median_otsu
import dipy.data as dpd
from nilearn._utils import as_ndarray, data_gen
from dipy.io import save_pickle
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from pynets.fmri.estimation import get_optimal_cov_estimator
from pynets.core.nodemaker import get_names_and_coords_of_parcels
from graspologic.simulations.simulations import er_nm, sbm
from graspologic.utils import symmetrize, remove_loops
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.streamline import save_trk
from nilearn.masking import intersect_masks


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nib.Nifti1Image(data, affine), nib.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


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

    mask_img_small = nib.Nifti1Image(mask[20:50, 55:85, 38:39
                                     ].astype('float32'),
                                     affine=mask_img.affine)
    B0_mask_small = fname_presuffix(dwi_file, suffix="_brain_mask_small",
                                    use_ext=True)
    nib.save(mask_img_small, B0_mask_small)

    gtab = gradient_table(bvals, np.loadtxt(fbvecs))
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0

    gtab_file_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.pkl',
                                                delete=False)
    gtab_file = str(gtab_file_tmp.name)
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
           'dwi_file_small': dwi_file_small, 'B0_mask_small': B0_mask_small,
           'B0_mask': B0_mask, 't1w_file': t1w_file,
           'f_pve_csf': f_pve_csf, 'f_pve_wm': f_pve_wm, 'f_pve_gm': f_pve_gm}
    gtab_file_tmp.close()

@pytest.fixture(scope="session")
def tractography_estimation_data(dmri_estimation_data):
    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    gtab = dmri_estimation_data['gtab']
    wm_img = nib.load(dmri_estimation_data['f_pve_wm'])
    dwi_img = nib.load(dmri_estimation_data['dwi_file'])
    dwi_data = dwi_img.get_fdata()
    B0_mask_img = nib.load(dmri_estimation_data['B0_mask'])
    mask_img = intersect_masks(
        [
            nib.Nifti1Image(np.asarray(wm_img.dataobj).astype('bool'
                                                              ).astype('int'),
                            affine=wm_img.affine),
            nib.Nifti1Image(np.asarray(B0_mask_img.dataobj).astype(
                'bool').astype('int'),
                            affine=B0_mask_img.affine)
        ],
        threshold=1,
        connected=False,
    )

    mask_data = mask_img.get_fdata()
    mask_file = fname_presuffix(dmri_estimation_data['B0_mask'],
                                suffix="tracking_mask", use_ext=True)
    mask_img.to_filename(mask_file)
    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, dwi_data, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=mask_data)

    stopping_criterion = BinaryStoppingCriterion(mask_data)

    seed_mask = (mask_data == 1)
    seeds = utils.seeds_from_mask(seed_mask, dwi_img.affine, density=[1, 1, 1])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=dwi_img.affine, step_size=.5)
    streamlines = Streamlines(streamlines_generator)
    sft = StatefulTractogram(streamlines, dwi_img, origin=Origin.NIFTI,
                    space=Space.RASMM)
    trk = f"{dir_path}/tractogram.trk"
    save_trk(sft, trk, streamlines)
    del streamlines, sft, streamlines_generator, seeds, seed_mask, csa_peaks, \
        csa_model, dwi_data, mask_data
    dwi_img.uncache()
    gc.collect()

    yield {'trk': trk, 'mask': mask_file}

    tmp.cleanup()


@pytest.fixture(scope="session")
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

    yield {'func_file': func_file,
           'conf_file': conf_file,
           'mask_file': mask_file,
           't1w_file': t1w_file,
           'func_file2': func_file2,
           'mask_file2': mask_file2}
    mask_tmp.close()
    conf_tmp.close()
    func_file_tmp.close()


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
                mat = symmetrize(
                    remove_loops(er_nm(n, m, wt=np.random.uniform,
                                       wtargs=dict(low=0, high=1))))
            elif mat_type == 'sb':
                if p is None:
                    raise ValueError(
                        f"for mat_type {mat_type}, p cannot be None")
                mat = symmetrize(
                    remove_loops(sbm(np.array([n]), np.array([[p]]),
                                     wt=wt, wtargs=dict(low=0,
                                                        high=1))))
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

        return {'mat_list': mat_list, 'mat_file_list': mat_file_list}

    return _gen_mat_data


@pytest.fixture(scope="session")
def connectivity_data(fmri_estimation_data):
    """Fixture for connectivity tests."""

    mask_file = fmri_estimation_data['mask_file']
    func_file = fmri_estimation_data['func_file']
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
