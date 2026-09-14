import numpy as np
import pytest
from mtare_topo.evaluation.gse_surface_detection_metrics_v1 import detection_counts


def score(p,t,**kwargs):
    values=dict(predicted_xyz_m=np.asarray(p).reshape(-1,3),confidence=np.ones(len(p)),
        target_xyz_m=np.asarray(t).reshape(-1,3),threshold=.5,maximum_error_m=1.,
        score_center_m=[0,0,0],score_radius_m=10.,region_complete=True)
    values.update(kwargs)
    return detection_counts(**values)


def test_complete_region_extra_prediction_counts():
    r=score([[0,0,0],[3,0,0]],[[0,0,0]])
    assert (r['true_positive'],r['known_region_false_positive'],r['full_f1'])==(1,1,2/3)


def test_unknown_not_negative_nor_perfect_full_f1():
    r=score([[0,0,0],[3,0,0]],[[0,0,0]],region_complete=False)
    assert r['known_region_false_positive']==0 and r['unknown_unmatched_predictions']==1
    assert r['full_f1'] is None and not r['full_scoring_available']


def test_duplicates_are_false_positives():
    r=score([[0,0,0],[0,0,0]],[[0,0,0]])
    assert r['true_positive']==1 and r['known_region_false_positive']==1


def test_global_cardinality_not_greedy():
    r=score([[0,0,0],[1,0,0]],[[.1,0,0],[-.9,0,0]])
    assert r['true_positive']==2


def test_64_queries_polynomial_matching_and_permutation():
    p=np.column_stack([np.linspace(-4,4,64),np.zeros((64,2))])
    r=score(p,p,maximum_error_m=.01)
    s=score(p[::-1],p[::-1],maximum_error_m=.01)
    assert r['true_positive']==s['true_positive']==64
    assert r['full_f1']==s['full_f1']==1


def test_3d_outside_and_empty_not_success():
    r=score([[0,0,11]],[[0,0,0]])
    assert r['false_negative']==1 and r['unknown_unmatched_predictions']==1 and r['full_f1'] is None
    assert score([],[])['full_f1'] is None


def test_capacity_not_truncated():
    with pytest.raises(ValueError):score(np.zeros((65,3)),[])
