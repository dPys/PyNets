import time
from pynets.core.database import *


def test_ConnectomeEnsemble():
    """
    Test the ConnectomeEnsemble class.
    """
    from pynets.core.database import ConnectomeEnsemble

    # Test the constructor
    ce = ConnectomeEnsemble()
    ce.subject_id = "test"
    ce.created_at = time.time()
    ce.updated_at = time.time()
    ce.modality = "func"
    ce.embed_meta = "ASE"
    ce.parcellation_meta = "atlas_AAL"
    ce.subnet_meta = "Default"
    ce.template = "MNI152_T1_2mm_brain"
    ce.thr_type = "MST"
    ce.thr = 0.95
    ce.node_type = "parcels"
    ce.signal_meta = "mean"
    ce.traversal_meta = "prob"
    ce.minlength_meta = 30
    ce.hpass_meta = 0.01
    ce.model_meta = "covariance"
    ce.granularity_meta = 200
    ce.smooth_meta = 3
    ce.error_margin_meta = 2

    # Test the getters and setters
    assert ce.subject_id == "test"
    assert ce.created_at == ce.created_at
    assert ce.updated_at == ce.updated_at
    assert ce.modality == "func"
    assert ce.embed_meta == "ASE"
    assert ce.parcellation_meta == "atlas_AAL"
    assert ce.subnet_meta == "Default"
    assert ce.template == "MNI152_T1_2mm_brain"
    assert ce.thr_type == "MST"
    assert ce.thr == 0.95
    assert ce.node_type == "parcels"
    assert ce.signal_meta == "mean"
    assert ce.traversal_meta == "prob"
    assert ce.minlength_meta == 30
    assert ce.hpass_meta == 0.01
    assert ce.model_meta == "covariance"
    assert ce.granularity_meta == 200
    assert ce.smooth_meta == 3
    assert ce.error_margin_meta == 2
