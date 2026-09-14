import numpy as np
import pytest
from mtare_topo.semantics.axis_intersection_candidates import axis_intersection_candidates as propose

P=dict(max_residual_m=.01,min_crossing_sine=.1,endpoint_tolerance_m=1e-8,maximum_candidates=32)


def test_two_junctions_without_teacher_grouping():
    s=np.array([[[-5.,0,0],[5,0,0]],[[-3,0,0],[-3,2,0]],[[3,-2,0],[3,0,0]]])
    out=propose(s,**P)
    assert len(out)==2
    assert np.allclose([v.position_m for v in out],[[-3,0,0],[3,0,0]])
    assert {v.segment_indices for v in out}=={(0,1),(0,2)}
    assert not any(v.connectivity_verified for v in out)


def test_multiple_pairs_same_crossing_one_candidate():
    s=[[[-2,0,0],[2,0,0]],[[0,-2,0],[0,2,0]],[[0,0,-2],[0,0,2]]]
    out=propose(s,**P)
    assert len(out)==1 and out[0].segment_indices==(0,1,2)


def test_stacked_and_extensions_do_not_propose():
    assert not propose([[[-2,0,0],[2,0,0]],[[0,-2,2],[0,2,2]]],**P)
    assert not propose([[[1,0,0],[2,0,0]],[[0,1,0],[0,2,0]]],**P)


def test_order_and_endpoint_reversal():
    s=np.array([[[-5.,0,0],[5,0,0]],[[-3,0,0],[-3,2,0]],[[3,-2,0],[3,0,0]]])
    a=propose(s,**P);b=propose(s[[2,0,1],::-1],**P)
    assert np.allclose([v.position_m for v in a],[v.position_m for v in b])


def test_capacity_and_degenerate_fail():
    s=np.array([[[-5.,0,0],[5,0,0]],[[-3,0,0],[-3,2,0]],[[3,-2,0],[3,0,0]]])
    with pytest.raises(ValueError,match='capacity'):propose(s,**(P|{'maximum_candidates':1}))
    s[0,1]=s[0,0]
    with pytest.raises(ValueError,match='degenerate'):propose(s,**P)
