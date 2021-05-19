"""Configuration file for pytest for pynets."""

from pathlib import Path
import pytest
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
import pickle
import os
from random import randint
from glob import glob


@pytest.fixture(scope='module')
def dmri_estimation_data():
    """Fixture for dmri estimation tests."""

    base_dir = str(Path(__file__).parent/"examples")
    B0_mask = f"{base_dir}/003/anat/mean_B0_bet_mask_tmp.nii.gz"

    dir_path = f"{base_dir}/003/dmri"
    dwi_file = f"{base_dir}/003/test_out/003/dwi/sub-003_dwi_reor-RAS_res-2mm.nii.gz"

    bvals = f"{dir_path}/sub-003_dwi.bval"
    bvecs = f"{base_dir}/003/test_out/003/dwi/bvecs_reor.bvec"

    gtab = gradient_table(np.loadtxt(bvals)[:11], np.loadtxt(bvecs)[:, :11])
    gtab.b0_threshold = 50
    gtab_bvals = gtab.bvals.copy()
    b0_thr_ixs = np.where(gtab_bvals < gtab.b0_threshold)[0]
    gtab_bvals[b0_thr_ixs] = 0
    gtab.b0s_mask = gtab_bvals == 0

    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()[48:64, 48:64, 28:37, :11]
    data_img = nib.Nifti1Image(dwi_data, header=dwi_img.header, affine=dwi_img.affine)

    mask_img = nib.load(B0_mask)
    mask_data = mask_img.get_fdata()[48:64, 48:64, 28:37]
    mask_img = nib.Nifti1Image(mask_data, header=mask_img.header, affine=mask_img.affine)

    yield {'gtab': gtab, 'dwi_img': data_img, 'B0_mask_img': mask_img}


@pytest.fixture(scope='function')
def plotting_data():
    """Fixture for plotting tests."""

    base_dir = str(Path(__file__).parent/"examples")
    conn_matrix = np.genfromtxt(f"{base_dir}/miscellaneous/002_rsn-Default_nodetype-parc_model-sps_thrtype-PROP_thr-0.94.txt")

    labels_file_path = f"{base_dir}/miscellaneous/Default_func_labelnames_wb.pkl"
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)

    coord_file_path = f"{base_dir}/miscellaneous/Default_func_coords_wb.pkl"
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)

    yield {'conn_matrix': conn_matrix, 'labels': labels, 'coords': coords}



@pytest.fixture(scope='module')
def gen_mat_data(n, m, p=None, binary = False, mat_type='sb', asfile=False, n_graphs=1):
    import tempfile
    from graspologic.simulations.simulations import er_nm, sbm, rdpg
    from graspologic.utils import symmetrize    
    if binary is True:
        wt = 1
    else:
        wt = np.random.uniform    
        mat_list = []
    for nm in range(n_graphs):
        if mat_type == 'er':
            mat = symmetrize(er_nm(n, m, wt=np.random.uniform, wtargs=dict(low=0, high=1)))
        elif mat_type == 'sb':
            if p is None:
                raise ValueError(f"for mat_type {mat_type}, p cannot be None")
            mat = symmetrize(sbm(np.array([n]), np.array([[p]]), wt=wt, wtargs=dict(low=0, high=1)))
        else:
            raise ValueError(f"mat_type {mat_type} not recognized!")        
        if asfile is True:
            mat_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.npy').name
            np.save(mat_path, mat)
            mat_list.append(mat_path)
        else:
            mat_list.append(mat)    
        yield mat_list

