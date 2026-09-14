from dataclasses import replace
import pytest
from mtare_topo.teacher.gse_observed_chain_candidate import SourceSpan, ObservedJoin, RayChain, qualify_candidate


def split_chain():
    return RayChain(0,12,(0.,0.,0.),(1.,0.,0.),10.,0.,8.,'a','J',('b','side','far'),'target:12',
        (SourceSpan('a',0.,4.,'S'),SourceSpan('b',4.,8.,'J')),
        (ObservedJoin('S',('a','b'),4.,'a','b','join:12'),))


def test_direct_and_observed_subdivision_have_same_target():
    split=split_chain()
    direct=replace(split,start_source='b',spans=(SourceSpan('b',0.,8.,'J'),),joins=())
    a,b=qualify_candidate(direct),qualify_candidate(split)
    assert a['status']==b['status']=='CONDITIONAL_RAY_CHAIN_CANDIDATE'
    assert a['target_node_teacher_only']==b['target_node_teacher_only']=='J'
    assert b['training_qualified'] is False


@pytest.mark.parametrize('change,reason',[
    ({'joins':()},'MISSING_CHAIN_SPANS_OR_TRANSITIONS'),
    ({'observed_end_m':11.},'NOT_WITHIN_OBSERVED_PRE_RETURN_INTERVAL'),
    ({'conflicting_nodes':('other',)},'CONFLICTING_STRUCTURE_EVIDENCE'),
    ({'target_evidence_id':''},'NO_OBSERVED_TARGET_WITNESS'),
    ({'target_node':'another_layer'},'TARGET_INCIDENCE_MISMATCH'),
    ({'start_source':'unrelated'},'START_SOURCE_MISMATCH'),
])
def test_unobserved_or_conflicting_does_not_become_positive(change,reason):
    assert qualify_candidate(replace(split_chain(),**change))['reason']==reason


@pytest.mark.parametrize('join_change,reason',[
    ({'incident_sources':('a','b','hidden_branch')},'INTERMEDIATE_NOT_DEGREE_TWO'),
    ({'incident_sources':('a','upper_layer')},'INCIDENCE_MISMATCH'),
    ({'evidence_id':''},'NO_OBSERVED_TRANSITION'),
    ({'from_source':'b','to_source':'a'},'DIRECTED_TRANSITION_MISMATCH'),
    ({'t':3.},'GAP_OR_UNBOUND_TRANSITION'),
])
def test_transition_rejections(join_change,reason):
    chain=split_chain()
    assert qualify_candidate(replace(chain,joins=(replace(chain.joins[0],**join_change),)))['reason']==reason


def test_gap_not_bridged_by_construction_incidence():
    chain=split_chain()
    spans=(chain.spans[0],replace(chain.spans[1],t0=4.001))
    assert qualify_candidate(replace(chain,spans=spans))['status']=='UNKNOWN'


def test_common_rotation_about_current_sensor_does_not_change_relation():
    chain=split_chain()
    assert qualify_candidate(chain)==qualify_candidate(replace(chain,direction_xyz=(0.,0.,1.)))


def test_same_xy_but_outside_local_height_is_rejected():
    chain=replace(split_chain(),origin_xyz=(0.,0.,9.))
    assert qualify_candidate(chain)['reason']=='OUTSIDE_ORIGINAL_10M_DOMAIN'


def test_future_frame_rejected():
    with pytest.raises(ValueError): qualify_candidate(replace(split_chain(),frame_slot=5))


def test_three_segment_chain_requires_every_transition():
    chain=split_chain()
    spans=(SourceSpan('a',0.,2.,'S0'),SourceSpan('mid',2.,4.,'S'),SourceSpan('b',4.,8.,'J'))
    joins=(ObservedJoin('S0',('a','mid'),2.,'a','mid','p0'),ObservedJoin('S',('mid','b'),4.,'mid','b','p1'))
    assert qualify_candidate(replace(chain,spans=spans,joins=joins))['status']=='CONDITIONAL_RAY_CHAIN_CANDIDATE'
    assert qualify_candidate(replace(chain,spans=spans,joins=joins[:1]))['status']=='UNKNOWN'
