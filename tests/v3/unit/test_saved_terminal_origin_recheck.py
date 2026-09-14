from copy import deepcopy
import numpy as np
import pytest
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_terminal_continuation_contract import recheck_saved_positive_origin


def target(value=True):
    record=dict(membership=[[value]])
    return dict(record=record,target_record_sha256=canonical_sha(record),teacher_provenance=dict(
        terminal_anchor_start=0,terminals=[dict(endpoint_key_teacher_only=['a',0])],
        openings=[dict(primitive_id_teacher_only='a')],
        relations=[dict(anchor_index=0,opening_index=0,status='PARTIAL_TERMINAL_SOURCE_COMPONENT_CORRESPONDENCE',evidence_indices=[0])],
        terminal_relations=[dict(anchor_index=0,opening_index=0,status='TWO_SIDED_SOURCE_WITNESS_ONLY',blocked_ray_indices=[],
            component=dict(status='REFERENCE_COMPONENT_ONLY'),common_frame_slots=[0,1])]))


def check(t,d,spacing=.025):
    return recheck_saved_positive_origin(t,opening_index=0,terminal_index=0,source_ids=['a','b'],origin_distances=d,field_spacing_m=spacing)


def test_old_evidence_is_immutable_and_no_new_label_is_written():
    t=target();before=deepcopy(t);r=check(t,np.tile([-1.,1.],(5,1)))
    assert t==before and r['retained_witness_frames']==[0,1] and r['membership'] is None
    assert not r['complete_reference_qualified']


@pytest.mark.parametrize('value',[None,False])
def test_never_promote_unknown_or_negative(value):
    assert check(target(value),np.tile([-1.,1.],(5,1)))['status']=='NOT_AN_EXISTING_POSITIVE'


def test_new_field_cannot_invent_new_witness_frames():
    d=np.tile([-1.,1.],(5,1));d[:2]=[1.,-1.]
    assert check(target(),d)['status']=='ORIGIN_PREREQUISITE_NOT_RETAINED'


@pytest.mark.parametrize('row',[[-1.,-1.],[0.,1.],[1.,1.]])
def test_overlap_boundary_and_outside_do_not_retain(row):
    assert check(target(),np.tile(row,(5,1)))['retained_witness_frames']==[]


def test_old_spacing_is_not_silently_mixed():
    with pytest.raises(ValueError):check(target(),np.ones((5,2)),spacing=.01)


def test_bound_record_drift_is_rejected():
    t=target();t['record']['membership'][0][0]=None
    with pytest.raises(ValueError):check(t,np.ones((5,2)))


def test_blocked_original_reference_is_rejected():
    t=target();t['teacher_provenance']['terminal_relations'][0]['blocked_ray_indices']=[0]
    with pytest.raises(ValueError):check(t,np.ones((5,2)))
