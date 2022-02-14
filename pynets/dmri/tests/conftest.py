"""Configuration file for pytest for pynets."""

from pathlib import Path
import os
import sys
import gc
import pytest
import tempfile
import numpy as np
import nibabel as nib
import dipy.data as dpd
if sys.platform.startswith('win') is False:
    import indexed_gzip
from dipy.core.gradients import gradient_table
from nipype.utils.filemanip import fname_presuffix
from dipy.segment.mask import median_otsu
from dipy.io import save_pickle
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.utils import seeds_from_mask
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram
from dipy.io.streamline import save_tractogram
from nilearn.masking import intersect_masks
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

    del mask
    mask_img.uncache()
    mask_img_small.uncache()
    gc.collect()

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
    gc.collect()

    yield {'dwi_file': dwi_file, 'fbvals': fbvals, 'fbvecs': fbvecs,
           'gtab': gtab, 'gtab_file': gtab_file,
           'dwi_file_small': dwi_file_small, 'B0_mask_small': B0_mask_small,
           'B0_mask': B0_mask, 't1w_file': t1w_file,
           'f_pve_csf': f_pve_csf, 'f_pve_wm': f_pve_wm, 'f_pve_gm': f_pve_gm}
    gtab_file_tmp.close()


@pytest.fixture(scope='package')
def tractography_estimation_data(dmri_estimation_data):
    path_tmp = tempfile.NamedTemporaryFile(mode='w+',
                                           suffix='.trk',
                                           delete=False)
    trk_path_tmp = str(path_tmp.name)
    dir_path = os.path.dirname(trk_path_tmp)

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
    seeds = seeds_from_mask(seed_mask, dwi_img.affine, density=[1, 1, 1])

    streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                          affine=dwi_img.affine, step_size=.5)
    streamlines = Streamlines(streamlines_generator)
    sft = StatefulTractogram(streamlines, B0_mask_img, origin=Origin.NIFTI,
                             space=Space.VOXMM)
    sft.remove_invalid_streamlines()
    trk = f"{dir_path}/tractogram.trk"
    os.rename(trk_path_tmp, trk)
    save_tractogram(sft, trk, bbox_valid_check=False)
    del streamlines, sft, streamlines_generator, seeds, seed_mask, csa_peaks, \
        csa_model, dwi_data, mask_data
    dwi_img.uncache()
    mask_img.uncache()
    gc.collect()

    yield {'trk': trk, 'mask': mask_file}
