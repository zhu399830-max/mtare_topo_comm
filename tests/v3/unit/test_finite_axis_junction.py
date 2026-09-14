import numpy as np
import pytest
from mtare_topo.semantics.finite_axis_junction import finite_axis_intersection as intersect

OPTIONS=dict(max_residual_m=.01,min_crossing_sine=.1,endpoint_tolerance_m=1e-8)


def test_t_junction_and_endpoint_reversal():
    s=np.array([[[-2.,0,0],[2,0,0]],[[0,0,0],[0,2,0]]])
    a=intersect(s,**OPTIONS);b=intersect(s[::-1,::-1],**OPTIONS)
    assert a.reason=='geometric_candidate_only' and np.allclose(a.position_m,[0,0,0])
    assert np.allclose(a.position_m,b.position_m) and not a.connectivity_verified


def test_stacked_crossing_rejected():
    r=intersect([[[-2,0,0],[2,0,0]],[[0,-2,2],[0,2,2]]],**OPTIONS)
    assert r.position_m is None and r.reason=='separated_axes'


def test_infinite_extensions_do_not_create_junction():
    r=intersect([[[1,0,0],[2,0,0]],[[0,1,0],[0,2,0]]],**OPTIONS)
    assert r.reason=='intersection_outside_observed_extent'


def test_parallel_duplicate_and_degenerate():
    s=np.array([[[-2.,0,0],[2,0,0]],[[-2,1,0],[2,1,0]]])
    assert intersect(s,**OPTIONS).reason=='parallel_or_nearly_parallel'
    assert intersect(np.repeat(s[:1],3,axis=0),**OPTIONS).position_m is None
    s[0,1]=s[0,0]
    assert intersect(s,**OPTIONS).reason=='degenerate_axis'


def test_rigid_transform_and_permutation():
    s=np.array([[[-2.,0,0],[2,0,0]],[[0,0,0],[0,2,0]],[[0,0,-2],[0,0,2]]])
    q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)));shift=np.array([3.,4,2])
    t=s[[2,0,1]]@q.T+shift
    r=intersect(t,**OPTIONS)
    assert np.allclose(r.position_m,shift)


def test_far_crossings_not_one_group():
    s=[[[-5,0,0],[5,0,0]],[[-3,-2,0],[-3,2,0]],[[3,-2,0],[3,2,0]]]
    assert intersect(s,**OPTIONS).reason=='separated_axes'


def test_explicit_invalid_inputs_and_empty():
    assert intersect(np.empty((0,2,3)),**OPTIONS).reason=='insufficient_axes'
    with pytest.raises(ValueError):intersect(np.zeros((33,2,3)),**OPTIONS)
    with pytest.raises(ValueError):intersect(np.full((2,2,3),np.nan),**OPTIONS)
    with pytest.raises(ValueError):intersect(np.zeros((2,2,3)),**(OPTIONS|{'min_crossing_sine':0}))
