"""py.test configuration."""
import os
import tempfile
import pytest
import time
import shutil
from pathlib import Path
import numpy as np
import networkx as nx
import nibabel as nib
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle


@pytest.fixture(autouse=True)
def doctest_autoimport(doctest_namespace):
    """Make available some fundamental modules to doctest modules."""
    doctest_namespace['np'] = np
    doctest_namespace['nib'] = nib
    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path
    doctest_namespace['nib'] = nib
    doctest_namespace['nx'] = nx
    doctest_namespace['time'] = time
    doctest_namespace['shutil'] = shutil
    doctest_namespace['data_dir'] = Path(__file__).parent.parent / 'tests' / 'examples'
    tmpdir = tempfile.TemporaryDirectory()
    doctest_namespace['tmpdir'] = tmpdir.name
    yield
    tmpdir.cleanup()
