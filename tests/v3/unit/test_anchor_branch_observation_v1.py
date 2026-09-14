import pytest
from mtare_topo.topology.anchor_branch_observation_v1 import (
    BranchDirectionCandidateV1 as B, AnchorBranchCandidateV1 as A, AnchorBranchObservationV1 as O)


def test_layout_does_not_collapse_direction_queries_across_anchors():
    branches=tuple(B(i,(1.,0.,0.),.5) for i in range(64))
    o=O('stream',0,(10,11,12,13,14),'a'*64,(A(0,(0.,0.,0.),.5,branches),A(1,(1.,0.,0.),.5,branches)))
    assert sum(len(a.branches) for a in o.anchors)==128
    assert o.coordinate_frame=='current_sensor'
    assert 'opening_position' in o.unavailable_fields
    assert not hasattr(o.anchors[0].branches[0],'position_current_sensor_m')


def test_unknown_not_synthesized_on_empty_observation():
    o=O('stream',0,(0,1,2,3,4),'b'*64,())
    assert o.anchors==() and 'verified_connection' in o.unavailable_fields


def test_invalid_direction_and_duplicate_queries_rejected():
    with pytest.raises(ValueError):B(0,(0.,0.,0.),.5)
    b=B(0,(1.,0.,0.),.5)
    with pytest.raises(ValueError):A(0,(0.,0.,0.),.5,(b,b))
    with pytest.raises(ValueError):O('stream',0,(0,2,1,3,4),'a'*64,())
