import numpy as np
import pytest
from mtare_topo.evaluation.gse_window_surface_scoring import score_complete_window_openings as score


def test_float32_surface_roundoff_is_not_discarded():
    direction=np.array([[1.,2.,3.]],dtype=np.float32)
    xyz=10*direction/np.linalg.norm(direction,axis=1)[:,None]
    result=score(xyz,xyz,np.ones(1),complete_window=True)
    assert result['tp']==1 and result['f1']==1 and result['discarded_by_radius']==0


def test_duplicates_count_and_empty_predictions_fail():
    xyz=np.array([[10.,0.,0.]],dtype=np.float32)
    assert score(xyz,np.repeat(xyz,2,axis=0),np.ones(2),complete_window=True)['fp']==1
    assert score(xyz,np.empty((0,3),np.float32),np.empty(0),complete_window=True)['fn']==1


def test_unknown_or_true_off_surface_rejects_not_discards():
    xyz=np.array([[10.,0.,0.]],dtype=np.float32)
    with pytest.raises(ValueError):score(xyz,xyz,np.ones(1),complete_window=False)
    with pytest.raises(ValueError):score(xyz,xyz*1.01,np.zeros(1),complete_window=True)
