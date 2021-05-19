#!/usr/bin/env python
import pytest
import numpy as np
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip
from pathlib import Path
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import os
import logging
from pynets.core import database

def test_create_db():
    session = database.connection('sqlite:////Users/Keval/Documents/College Senior Classes/Research/Pynets/pynets.db')
    print(type(session))
    assert session is not None