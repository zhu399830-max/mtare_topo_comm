from copy import deepcopy
import pytest

from mtare_topo.integration.native_epoch_visit_binding import NativeEpochVisitBinding, classify_local_matches


def visit(event=1,cid=0,stamp=1000):
    return dict(topic='/n/feedback',record_ref=f'trace:{event}',payload=dict(
        session_id='s',event_index=event,event='NATIVE_REGION_VISIT',candidate_id=cid,
        evidence_id=f's:e:{event}',pose_binding_exact=True,pose_frame_id='map',
        pose_evidence_kind='native_state_estimation_cell_membership',
        source_frame_keys=[f'scan:{stamp}'],robot_position_xyz_m=[0,0,0],pose_stamp_ns=stamp))


def snapshot(epoch=1,stamp=1000,depot=0):
    return dict(topic='/n/candidates',record_ref=f'snapshot:{epoch}',payload=dict(
        schema_version='native_route_snapshot_v1',epoch=f's:{epoch}',stamp_ns=stamp,max_age_ns=100,
        max_extra_cost_units=0,source_frame_keys=[f'scan:{stamp}'],candidate_ids=[3],
        original_route=[depot,3,depot],original_cost_units=20,
        native_edges=[[depot,3,10],[3,depot,10]],robot_position_xyz_m=[0,0,0]))


def test_exact_visit_and_repeat_visit_not_global_identity():
    b=NativeEpochVisitBinding();b.consume(visit())
    first=b.consume(snapshot());assert first['binding_status']=='EXACT_NATIVE_METRIC_VISIT'
    b.consume(visit(2));second=b.consume(snapshot(2))
    assert first['metric_visit_id']==second['metric_visit_id']
    b.consume(visit(3,cid=1));b.consume(snapshot(3,depot=1))
    b.consume(visit(4));fourth=b.consume(snapshot(4))
    assert first['metric_visit_id']!=fourth['metric_visit_id']
    assert not fourth['direction_identity_known'] and not fourth['task_completed']


@pytest.mark.parametrize('key,value', [('pose_stamp_ns',999),('source_frame_keys',['wrong']),
    ('robot_position_xyz_m',[0,0,3]),('candidate_id',2)])
def test_wrong_time_sources_level_or_region_not_nearest_bound(key,value):
    b=NativeEpochVisitBinding();v=visit();v['payload'][key]=value;b.consume(v)
    assert b.consume(snapshot())['binding_status']=='UNKNOWN'


def test_unknown_and_prefix_no_retroactive_repair():
    b=NativeEpochVisitBinding();first=b.consume(snapshot());frozen=deepcopy(first)
    b.consume(visit());second=b.consume(snapshot(2))
    assert first==frozen and first['binding_status']=='UNKNOWN'
    assert second['binding_status']=='EXACT_NATIVE_METRIC_VISIT'
    invalid=visit(2);invalid['payload']['pose_binding_exact']=False;b.consume(invalid)
    assert b.consume(snapshot(3))['binding_status']=='UNKNOWN'


def test_cross_session_and_duplicate_sequence_rejected():
    b=NativeEpochVisitBinding();b.consume(visit())
    with pytest.raises(ValueError):b.consume(visit())
    b.consume(snapshot())
    with pytest.raises(ValueError):b.consume(snapshot())


def test_multiple_frames_one_visit_are_not_multiple_confirmed_places():
    current=dict(epoch='s:3',binding_status='EXACT_NATIVE_METRIC_VISIT',metric_visit_id='visitB')
    history={f's:{i}':dict(binding_status='EXACT_NATIVE_METRIC_VISIT',metric_visit_id='visitA') for i in [1,2]}
    candidates=[dict(current_epoch='s:3',historical_epoch=f's:{i}',accepted=True,record_ref=f'ICP:{i}') for i in [1,2]]
    result=classify_local_matches(current,candidates,history)
    assert result['accepted_scan_count']==2 and result['accepted_visit_group_count']==1
    assert result['unresolved_identity'] and not result['task_completed'] and not result['live_registration_available']
    history['s:2']['metric_visit_id']='visitB'
    assert classify_local_matches(current,candidates,history)['accepted_visit_group_count']==2


def test_unknown_binding_and_rejected_fit_are_retained():
    current=dict(epoch='s:3',binding_status='UNKNOWN',metric_visit_id=None)
    history={'s:1':dict(binding_status='EXACT_NATIVE_METRIC_VISIT',metric_visit_id='v')}
    candidates=[dict(current_epoch='s:3',historical_epoch='s:1',accepted=True,record_ref='ICP:1')]
    result=classify_local_matches(current,candidates,history)
    assert result['scan_candidates'][0]['kind']=='LOCAL_FIT_WITHOUT_EXACT_VISIT_BINDING'
    candidates[0]['accepted']=False
    assert classify_local_matches(current,candidates,history)['scan_candidates'][0]['kind']=='REGISTRATION_REJECTED'
    with pytest.raises(ValueError):classify_local_matches(current,candidates,{})


def test_missing_feedback_breaks_contiguous_visit_group():
    b=NativeEpochVisitBinding();b.consume(visit(1));first=b.consume(snapshot(1))
    b.consume(visit(4));second=b.consume(snapshot(2))
    assert first['metric_visit_id'] != second['metric_visit_id']
