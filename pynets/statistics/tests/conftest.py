"""Configuration file for pytest for pynets."""

from pathlib import Path
import sys
import pytest
import tempfile
import numpy as np
if sys.platform.startswith('win') is False:
    import indexed_gzip
from graspologic.simulations.simulations import er_nm, sbm
from graspologic.utils import symmetrize, remove_loops, \
    largest_connected_component
from pynets.core.thresholding import autofix


@pytest.fixture(scope='package')
def gen_mat_data():
    def _gen_mat_data(n: int=20, m: int=20, p: int=0.50,
                      mat_type: str='sb', binary: bool=False,
                      asfile: bool=True, n_graphs: int=1,
                      lcc: bool=False):
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

            if lcc is True:
                mat = largest_connected_component(mat)

            mat_list.append(autofix(mat))

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
