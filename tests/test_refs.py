#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import nibabel as nib
from pathlib import Path


def test_bigref():
    base_dir = "%s%s" % (str(Path(__file__).parent), '/../pynets/rsnrefs')
    bigref1mm = nib.load("%s%s" % (base_dir, '/BIGREF1mm.nii.gz'))
    bigref2mm = nib.load("%s%s" % (base_dir, '/BIGREF2mm.nii.gz'))
    smallref1mm = nib.load("%s%s" % (base_dir, '/SMALLREF1mm.nii.gz'))
    smallref2mm = nib.load("%s%s" % (base_dir, '/SMALLREF2mm.nii.gz'))
    assert bigref1mm is not None
    assert bigref2mm is not None
    assert smallref1mm is not None
    assert smallref2mm is not None
