"""Actual native message shape with synthetic coordinates; no maps/model runs."""
from copy import deepcopy
import json

import pytest

from test_native_region_task_registry import registry, event, feed
from test_native_route_intent_registry import decision
from test_native_planning_paths import paths


def pose(index, sequence, xyz, previous_stamp, **changes):
    return event('NATIVE_POSE', index, pose_sequence=sequence,
                 robot_position_xyz_m=xyz, previous_pose_stamp_ns=previous_stamp, **changes)


def send(r, message):
    return feed(r, message, order=message['event_index'] + 1)


def publication(*, xyz=(4., 0., 2.), start=(0., 0., 2.), mutation=None, intent=True):
    r = registry()
    if intent:
        assert r.consume('decision', decision(), order=1, record_ref='native:decision')['accepted']
    p = pose(1, 1, list(start), 0)
    send(r, p)
    send(r, paths())
    send(r, event('WAYPOINT', 3, xyz_m=list(xyz)))
    d = event('NATIVE_REGION_DISPATCH', 4, candidate_id=99, waypoint_xyz_m=list(xyz),
        robot_position_xyz_m=p['robot_position_xyz_m'],
        robot_orientation_xyzw=p['robot_orientation_xyzw'], pose_frame_id='map',
        pose_sequence=p['pose_sequence'], pose_source_key=p['pose_source_key'],
        pose_stamp_ns=p['pose_stamp_ns'], pose_binding_exact=True)
    d.update(mutation or {})
    return r, p, send(r, d)


def state(r):
    return r.snapshot()['local_execution']


def test_local_motion_binds_even_when_waypoint_is_not_remote_vrp_candidate():
    r, p, verdict = publication()
    assert not verdict['accepted']  # Original regional scope remains unchanged.
    assert verdict['reason'] == 'WAYPOINT_REGION_NOT_A_NATIVE_CANDIDATE'
    assert verdict['local_execution']['bound']
    send(r, pose(5, 2, [1., 0., 2.], p['pose_stamp_ns']))
    s = state(r)
    command = s['commands']['synthetic:event:4']
    assert command['native_waypoint_region_id'] == 99
    assert command['route_intent_record_ref'] == 'native:decision'
    assert command['planning_path_record_ref'] == 'synthetic:feedback:2'
    assert command['waypoint_evidence_id'] == 'synthetic:event:3'
    assert command['local_command_direction_xyz'] == (1., 0., 0.)
    motion = s['events'][-1]
    assert motion['kind'] == 'odometry_observed'
    assert motion['observed_step_chord_m'] == 1.
    assert motion['displacement_along_command_m'] == 1.
    assert motion['distance_to_published_waypoint_m'] == 3.
    assert r.snapshot()['attempts'] == {}
    assert r.snapshot()['direction_task_count'] == 0
    assert not r.snapshot()['confirmed_traversals']
    assert not command['controller_acknowledged'] and not command['task_completed']
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize('mutation', [
    dict(waypoint_xyz_m=[5.,0.,2.]), dict(epoch='synthetic:2'),
    dict(source_frame_keys=['different']), dict(pose_binding_exact=False),
    dict(pose_source_key='unreceived:1010'), dict(pose_sequence=99),
    dict(pose_stamp_ns=999), dict(pose_frame_id='sensor'),
    dict(robot_position_xyz_m=[0.,0.,9.]), dict(robot_orientation_xyzw=[0.,0.,1.,0.]),
    dict(dispatch_scope='route_first_cell'),
])
def test_dispatch_cannot_invent_pose_or_reinterpret_local_command(mutation):
    r, _, _ = publication(mutation=mutation)
    assert state(r)['commands'] == {}


@pytest.mark.parametrize('change', [dict(pose_sequence=3), dict(previous_pose_stamp_ns=999),
    dict(event_index=6, evidence_id='synthetic:event:6'),
    dict(pose_stamp_ns=1020, pose_source_key='state_estimation:1020')])
def test_missing_source_or_precommand_pose_ends_interval_not_bridge_motion(change):
    r, p, _ = publication()
    next_pose = pose(5, 2, [1.,0.,2.], p['pose_stamp_ns'])
    next_pose.update(change)
    send(r, next_pose)
    assert state(r)['active_command_id'] is None
    assert not any(e['kind'] == 'odometry_observed' for e in state(r)['events'])
    assert state(r)['events'][-1]['kind'] == 'interval_closed'


def test_new_waypoint_closes_interval_even_if_new_dispatch_is_unbound():
    r, p, _ = publication()
    prefix = deepcopy(state(r))
    send(r, event('WAYPOINT', 5, xyz_m=[-4.,0.,2.]))
    send(r, pose(6, 2, [-1.,0.,2.], p['pose_stamp_ns']))
    assert state(r)['active_command_id'] is None
    assert state(r)['commands'] == prefix['commands']
    assert state(r)['events'][:len(prefix['events'])] == prefix['events']
    assert not any(e['kind'] == 'odometry_observed' for e in state(r)['events'])


def test_opposite_and_vertical_commands_remain_distinct_not_shared_completion():
    r, p, _ = publication(xyz=(-4.,0.,2.))
    send(r, pose(5, 2, [-1.,0.,2.], p['pose_stamp_ns']))
    assert state(r)['commands']['synthetic:event:4']['local_command_direction_xyz'] == (-1.,0.,0.)
    assert state(r)['events'][-1]['displacement_along_command_m'] == 1.
    vertical, p, _ = publication(xyz=(0.,0.,6.))
    send(vertical, pose(5, 2, [0.,0.,3.], p['pose_stamp_ns']))
    assert state(vertical)['commands']['synthetic:event:4']['local_command_direction_xyz'] == (0.,0.,1.)
    assert state(vertical)['events'][-1]['displacement_along_command_m'] == 1.
    assert state(vertical)['structural_task_completions'] == 0


def test_exact_arrival_and_native_finish_never_complete_direction():
    r, p, _ = publication()
    send(r, pose(5, 2, [4.,0.,2.], p['pose_stamp_ns']))
    send(r, event('NATIVE_EXECUTION_STATE', 6, native_finish=True, at_home=True))
    movement = state(r)['events'][-1]
    assert movement['distance_to_published_waypoint_m'] == 0
    assert movement['position_reached'] is None
    assert state(r)['structural_task_completions'] == 0
    assert not r.snapshot()['full_exploration_proven']


def test_zero_length_command_keeps_unknown_direction_and_never_nan():
    r, p, _ = publication(xyz=(0.,0.,2.))
    send(r, pose(5, 2, [0.,0.,2.], p['pose_stamp_ns']))
    assert state(r)['commands']['synthetic:event:4']['local_command_direction_xyz'] is None
    assert state(r)['events'][-1]['displacement_along_command_m'] is None
    json.dumps(state(r), allow_nan=False)


def test_missing_decision_does_not_prevent_motion_record_or_backfill_route():
    r, p, _ = publication(intent=False)
    send(r, pose(5, 2, [1.,0.,2.], p['pose_stamp_ns']))
    prefix = deepcopy(state(r))
    r.consume('decision', decision(), order=8, record_ref='late:decision')
    assert state(r) == prefix
    assert state(r)['commands']['synthetic:event:4']['route_intent_record_ref'] is None


def test_feedback_gap_between_publication_and_dispatch_cannot_start_command():
    r = registry()
    send(r, pose(1,1,[0.,0.,0.],0))
    send(r, event('WAYPOINT',2,xyz_m=[4.,0.,0.]))
    send(r, event('NATIVE_REGION_DISPATCH',4,waypoint_xyz_m=[4.,0.,0.]))
    assert state(r)['commands'] == {}


def test_rejected_pose_breaks_chain_on_next_event_not_silently_skipped():
    r, p, _ = publication()
    assert not send(r, pose(5,2,[float('nan'),0.,2.],p['pose_stamp_ns']))['accepted']
    send(r, pose(6,3,[1.,0.,2.],1050))
    assert state(r)['active_command_id'] is None
    assert state(r)['events'][-1]['reason'] == 'FEEDBACK_GAP'


def test_no_cross_session_binding():
    r, p, _ = publication()
    before = deepcopy(state(r))
    result = send(r, pose(5,2,[1.,0.,2.],p['pose_stamp_ns'],
                         session_id='other',evidence_id='other:event:5'))
    assert not result['accepted']
    assert state(r) == before


def test_repeated_pose_source_is_not_new_movement_or_rewritten_evidence():
    r, p, _ = publication()
    send(r, pose(5,1,p['robot_position_xyz_m'],0,pose_stamp_ns=p['pose_stamp_ns'],
                 pose_source_key=p['pose_source_key']))
    assert len(state(r)['events']) == 1
    send(r, pose(6,2,[1.,0.,2.],p['pose_stamp_ns']))
    assert state(r)['events'][-1]['previous_pose_ref'] == 'synthetic:feedback:1'


def test_reused_sequence_with_different_pose_stamp_breaks_motion_chain():
    r, p, _ = publication()
    send(r,pose(5,1,[1.,0.,2.],p['pose_stamp_ns']))
    assert state(r)['active_command_id'] is None
    assert state(r)['events'][-1]['reason'] == 'ODOMETRY_SOURCE_NOT_ADVANCING'


def test_dispatch_before_waypoint_time_is_rejected_by_envelope():
    r, _, verdict = publication(mutation=dict(stamp_ns=1020))
    assert not verdict['accepted'] and verdict['reason'] == 'NONCAUSAL_OR_DUPLICATE_FEEDBACK'
    assert state(r)['commands'] == {}


def test_stale_epoch_command_can_record_local_motion_but_not_current_route_binding():
    from test_native_region_task_registry import native_snapshot
    r, p, _ = publication()
    assert r.consume('snapshot',native_snapshot('synthetic:2',stamp=1045),
                     order=6,record_ref='new:snapshot')['accepted']
    feed(r,event('WAYPOINT',5,xyz_m=[-4.,0.,2.]),order=7)
    d = event('NATIVE_REGION_DISPATCH',6,candidate_id=99,waypoint_xyz_m=[-4.,0.,2.],
        robot_position_xyz_m=p['robot_position_xyz_m'],robot_orientation_xyzw=p['robot_orientation_xyzw'],
        pose_stamp_ns=p['pose_stamp_ns'],pose_sequence=1,pose_source_key=p['pose_source_key'],
        pose_frame_id='map',pose_binding_exact=True)
    verdict = feed(r,d,order=8)
    assert not verdict['accepted'] and verdict['reason'] == 'DISPATCH_NOT_BOUND_TO_CURRENT_SNAPSHOT'
    assert verdict['local_execution']['bound']  # Distinct, explicitly limited scope.
    command = state(r)['commands']['synthetic:event:6']
    assert command['route_intent_record_ref'] is None and command['native_route_changed'] is None
    assert command['remote_task_id'] is None and not command['task_completed']


def test_numeric_overflow_never_becomes_nonfinite_saved_motion():
    r, _, verdict = publication(xyz=(1.e308,0.,2.),start=(-1.e308,0.,2.))
    assert not verdict['local_execution']['bound']
    assert verdict['local_execution']['reason'] == 'LOCAL_COMMAND_GEOMETRY_OUT_OF_NUMERIC_RANGE'
    json.dumps(state(r),allow_nan=False)


def test_real_sidecar_callbacks_preserve_command_motion_chain(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from test_native_region_task_registry import native_snapshot
    from test_native_live_cache_sidecar import _ros_modules
    import native_structure_advice_node as node

    callbacks, published, shutdown, topics = {}, [], [], []
    _, start, _ = publication()
    dispatch = event('NATIVE_REGION_DISPATCH',4,candidate_id=99,waypoint_xyz_m=[4.,0.,2.],
        robot_position_xyz_m=start['robot_position_xyz_m'],
        robot_orientation_xyzw=start['robot_orientation_xyzw'],pose_frame_id='map',
        pose_sequence=1,pose_source_key=start['pose_source_key'],pose_stamp_ns=start['pose_stamp_ns'],
        pose_binding_exact=True)
    def spin():
        prefix = '/sensor_coverage_planner/native_structure/'
        callbacks[prefix+'candidates'](NS(data=json.dumps(native_snapshot())))
        callbacks[prefix+'decision'](NS(data=json.dumps(decision())))
        for payload in [start, paths(), event('WAYPOINT',3,xyz_m=[4.,0.,2.]),
                        dispatch,pose(5,2,[1.,0.,2.],start['pose_stamp_ns'])]:
            callbacks[prefix+'feedback'](NS(data=json.dumps(payload)))
    for name, module in _ros_modules(callbacks,published,shutdown,spin,topics).items():
        monkeypatch.setitem(sys.modules,name,module)
    output = tmp_path/'sidecar'
    monkeypatch.setattr(sys,'argv',['sidecar','--output',str(output)])
    assert node.main() == 0
    saved = json.loads((output/'native_region_tasks.json').read_text())
    assert len(saved['local_execution']['commands']) == 1
    assert saved['local_execution']['events'][-1]['displacement_along_command_m'] == 1.
    assert saved['attempts'] == {} and saved['direction_task_count'] == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['local_execution_command_count'] == 1
    assert summary['local_execution_odometry_sample_count'] == 1
    assert summary['local_execution_observed_chord_m'] == 1.
    assert not summary['local_execution_is_structural_completion']
    assert not summary['waypoint_published'] and not shutdown
    assert topics == ['/sensor_coverage_planner/native_structure/advice']
    assert len(published) == 1
    assert json.loads(published[0])['route'] == native_snapshot()['original_route']
