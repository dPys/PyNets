#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
@authors: Derek Pisner & Ryan Hammonds
"""
import pytest
import os
from pathlib import Path
from pynets.statistics import utils
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_save_netmets():
    """ Test save netmets functionality using dummy metrics
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    dir_path = str(tmp.name)
    os.makedirs(dir_path, exist_ok=True)

    est_path = tempfile.NamedTemporaryFile(mode='w+', suffix='.npy',
                                           delete=False)

    metric_list_names = ['metric_a', 'metric_b', 'metric_c']
    net_met_val_list_final = [1, 2, 3]

    utils.save_netmets(dir_path, str(est_path.name), metric_list_names,
                       net_met_val_list_final)
    tmp.cleanup()
    est_path.close()
