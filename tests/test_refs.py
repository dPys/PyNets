#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import indexed_gzip
import nibabel as nib
from pathlib import Path


def test_bigsmallref():
    """
    Test big and small ref files for existence and non-corruption
    """
    base_dir = "%s%s" % (str(Path(__file__).parent), '/../pynets/rsnrefs')
    bigref1mm = nib.load("%s%s" % (base_dir, '/BIGREF1mm.nii.gz'))
    bigref2mm = nib.load("%s%s" % (base_dir, '/BIGREF2mm.nii.gz'))
    smallref1mm = nib.load("%s%s" % (base_dir, '/SMALLREF1mm.nii.gz'))
    smallref2mm = nib.load("%s%s" % (base_dir, '/SMALLREF2mm.nii.gz'))
    assert bigref1mm is not None
    assert bigref2mm is not None
    assert smallref1mm is not None
    assert smallref2mm is not None


def test_atlases():
    """
    Test atlas files for existence and non-corruption
    """
    base_dir = "%s%s" % (str(Path(__file__).parent), '/../pynets/core/atlases')
    whole_brain_cluster_labels_PCA200 = nib.load("%s%s" % (base_dir, '/whole_brain_cluster_labels_PCA200.nii.gz'))
    AAL2zourioMazoyer2002 = nib.load("%s%s" % (base_dir, '/AAL2zourioMazoyer2002.nii.gz'))
    AALTzourioMazoyer2002 = nib.load("%s%s" % (base_dir, '/AALTzourioMazoyer2002.nii.gz'))
    AICHAJoliot2015 = nib.load("%s%s" % (base_dir, '/AICHAJoliot2015.nii.gz'))
    AICHAreorderedJoliot2015 = nib.load("%s%s" % (base_dir, '/AICHAreorderedJoliot2015.nii.gz'))
    BrainnetomeAtlasFan2016 = nib.load("%s%s" % (base_dir, '/BrainnetomeAtlasFan2016.nii.gz'))
    CorticalAreaParcellationfromRestingStateCorrelationsGordon2014 = nib.load(
        "%s%s" % (base_dir, '/CorticalAreaParcellationfromRestingStateCorrelationsGordon2014.nii.gz'))
    DesikanKlein2012 = nib.load("%s%s" % (base_dir, '/DesikanKlein2012.nii.gz'))
    destrieux2009_rois = nib.load("%s%s" % (base_dir, '/destrieux2009_rois.nii.gz'))
    Hammers_mithAtlasn30r83Hammers2003Gousias2008 = nib.load(
        "%s%s" % (base_dir, '/Hammers_mithAtlasn30r83Hammers2003Gousias2008.nii.gz'))
    HarvardOxfordThr252mmWholeBrainMakris2006 = nib.load(
        "%s%s" % (base_dir, '/HarvardOxfordThr252mmWholeBrainMakris2006.nii.gz'))
    Juelichgmthr252mmEickhoff2005 = nib.load("%s%s" % (base_dir, '/Juelichgmthr252mmEickhoff2005.nii.gz'))
    MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics = nib.load(
        "%s%s" % (base_dir, '/MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics.nii.gz'))
    RandomParcellationsc05meanalll43Craddock2011 = nib.load(
        "%s%s" % (base_dir, '/RandomParcellationsc05meanalll43Craddock2011.nii.gz'))
    VoxelwiseParcellationt058kLeadDBS = nib.load("%s%s" % (base_dir, '/VoxelwiseParcellationt058kLeadDBS.nii.gz'))
    VoxelwiseParcellationt0435kLeadDBS = nib.load("%s%s" % (base_dir, '/VoxelwiseParcellationt0435kLeadDBS.nii.gz'))
    VoxelwiseParcellationt0515kLeadDBS = nib.load("%s%s" % (base_dir, '/VoxelwiseParcellationt0515kLeadDBS.nii.gz'))
    whole_brain_cluster_labels_PCA100 = nib.load("%s%s" % (base_dir, '/whole_brain_cluster_labels_PCA100.nii.gz'))
    assert whole_brain_cluster_labels_PCA200 is not None
    assert AAL2zourioMazoyer2002 is not None
    assert AALTzourioMazoyer2002 is not None
    assert AICHAJoliot2015 is not None
    assert AICHAreorderedJoliot2015 is not None
    assert BrainnetomeAtlasFan2016 is not None
    assert CorticalAreaParcellationfromRestingStateCorrelationsGordon2014 is not None
    assert DesikanKlein2012 is not None
    assert destrieux2009_rois is not None
    assert Hammers_mithAtlasn30r83Hammers2003Gousias2008 is not None
    assert HarvardOxfordThr252mmWholeBrainMakris2006 is not None
    assert Juelichgmthr252mmEickhoff2005 is not None
    assert MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics is not None
    assert RandomParcellationsc05meanalll43Craddock2011 is not None
    assert VoxelwiseParcellationt058kLeadDBS is not None
    assert VoxelwiseParcellationt0435kLeadDBS is not None
    assert VoxelwiseParcellationt0515kLeadDBS is not None
    assert whole_brain_cluster_labels_PCA100 is not None


def test_templates():
    """
    Test template files for existence and non-corruption
    """
    base_dir = "%s%s" % (str(Path(__file__).parent), '/../pynets/templates')
    ch2better = nib.load("%s%s" % (base_dir, '/ch2better.nii.gz'))

    FA_2mm = nib.load("%s%s" % (base_dir, '/FA_2mm.nii.gz'))
    MNI152_T1_2mm_brain = nib.load("%s%s" % (base_dir, '/MNI152_T1_2mm_brain.nii.gz'))
    CorpusCallosum_2mm = nib.load("%s%s" % (base_dir, '/CorpusCallosum_2mm.nii.gz'))
    LateralVentricles_2mm = nib.load("%s%s" % (base_dir, '/LateralVentricles_2mm.nii.gz'))
    MNI152_T1_2mm_brain_mask = nib.load("%s%s" % (base_dir, '/MNI152_T1_2mm_brain_mask.nii.gz'))

    FA_1mm = nib.load("%s%s" % (base_dir, '/FA_1mm.nii.gz'))
    MNI152_T1_1mm_brain = nib.load("%s%s" % (base_dir, '/MNI152_T1_1mm_brain.nii.gz'))
    CorpusCallosum_1mm = nib.load("%s%s" % (base_dir, '/CorpusCallosum_1mm.nii.gz'))
    LateralVentricles_1mm = nib.load("%s%s" % (base_dir, '/LateralVentricles_1mm.nii.gz'))
    MNI152_T1_1mm_brain_mask = nib.load("%s%s" % (base_dir, '/MNI152_T1_1mm_brain_mask.nii.gz'))


    assert ch2better is not None
    assert FA_2mm is not None
    assert MNI152_T1_2mm_brain is not None
    assert CorpusCallosum_2mm is not None
    assert LateralVentricles_2mm is not None
    assert MNI152_T1_2mm_brain_mask is not None

    assert FA_1mm is not None
    assert MNI152_T1_1mm_brain is not None
    assert CorpusCallosum_1mm is not None
    assert LateralVentricles_1mm is not None
    assert MNI152_T1_1mm_brain_mask is not None
