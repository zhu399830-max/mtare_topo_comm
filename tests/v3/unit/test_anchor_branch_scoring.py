import numpy as np
import pytest
from mtare_topo.evaluation.anchor_branch_scoring import score_anchor_branches


def score(p,pd,t,td,complete=True,allowed=None,scoreable=None):
    return score_anchor_branches(np.array(p,float).reshape(-1,3),pd,np.array(t,float).reshape(-1,3),td,
        anchor_scoreable=np.array([True]*len(p) if scoreable is None else scoreable,bool),
        association_allowed=np.array([True]*len(p) if allowed is None else allowed,bool),
        anchor_reference_complete=complete,branch_reference_complete=[complete]*len(t),
        matching_radius_m=1.,matching_angle_deg=10.)


def test_missing_node_misses_its_branches():
    r=score([],[],[[0,0,0]],[[[1,0,0],[-1,0,0]]])
    assert r['anchors']['fn']==1 and r['branches']['fn']==2 and r['branches']['f1']==0


def test_wrong_node_cannot_get_direction_credit():
    r=score([[5,0,0]],[[[1,0,0]]],[[0,0,0]],[[[1,0,0]]])
    assert r['branches']['tp']==0 and r['branches']['known_fp']==1 and r['branches']['fn']==1


def test_unknown_node_predictions_and_partial_reference_do_not_invent_f1():
    r=score([[5,0,0]],[[[1,0,0]]],[[0,0,0]],[[[1,0,0]]],False,scoreable=[False])
    assert r['branches']['unresolved_predictions']==1 and r['branches']['known_fp']==0
    assert r['branches']['fn']==1 and r['branches']['f1'] is None


def test_unknown_reference_conflict_vetoes_association():
    r=score([[0,0,0]],[[[1,0,0]]],[[0,0,0]],[[[1,0,0]]],False,allowed=[False],scoreable=[False])
    assert r['anchors']['tp']==0 and r['branches']['fn']==1


def test_matched_pair_and_duplicate_node():
    r=score([[0,0,0],[.5,0,0]],[[[1,0,0]],[[1,0,0]]],[[0,0,0]],[[[1,0,0]]])
    assert r['branches']['tp']==1 and r['branches']['known_fp']==1


def test_tied_anchor_positions_do_not_use_semantics_to_choose():
    with pytest.raises(ValueError,match='ambiguous'):
        score([[0,0,0],[0,0,0]],[[[1,0,0]],[[-1,0,0]]],[[0,0,0]],[[[1,0,0]]])
