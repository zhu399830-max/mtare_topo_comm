import numpy as np
from mtare_topo.representation.grouping_pair_v1 import spatial_assignment
from mtare_topo.evaluation.grouping_center_scoring_v1 import center_score


def test_spatial_determinism_permutation_and_count():
    x=np.random.default_rng(3).normal(size=(100,3)).astype(np.float32)
    a=spatial_assignment(x,7);order=np.arange(100)[::-1]
    assert len(np.unique(a))==7
    assert np.array_equal(spatial_assignment(x[order],7),a[order])
    assert np.array_equal(spatial_assignment(x,7),a)


def test_known_exact_duplicate_negative_unknown():
    p=np.array([[0.,0,0],[0,0,0],[3,0,0],[5,0,0]])
    r=center_score(p,np.zeros((1,3)),np.array([True,True,True,False]),np.ones(4,bool))
    assert (r['tp'],r['fp'],r['fn'],r['ignored'])==(1,2,0,1)


def test_maximum_cardinality_before_distance():
    p=np.array([[.6,0,0],[1.5,0,0]])
    t=np.array([[0.,0,0],[.9,0,0]])
    r=center_score(p,t,np.ones(2,bool),np.ones(2,bool))
    assert r['tp']==2


def test_height_counts_no_xy_projection():
    r=center_score(np.array([[0.,0,2]]),np.zeros((1,3)),np.ones(1,bool),np.ones(1,bool))
    assert r['tp']==0 and r['fn']==1


def test_empty_and_exact_reference():
    t=np.array([[0.,0,0],[3,0,0]])
    r=center_score(t,t,np.ones(2,bool),np.ones(2,bool))
    assert r['precision']==r['recall']==1
    r=center_score(np.empty((0,3)),t,np.empty(0,bool),np.empty(0,bool))
    assert r['fn']==2
