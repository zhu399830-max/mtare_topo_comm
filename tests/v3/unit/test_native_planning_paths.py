from copy import deepcopy
import pytest
from test_native_region_task_registry import registry, event, feed
from test_native_route_intent_registry import decision


def paths(index=2, **changes):
    value = event('NATIVE_PLANNING_PATHS', index,
        path_schema='native_planning_paths_v1', coordinate_frame='map',
        evidence_scope='native_planned_paths_not_executed_traversal', paths_complete=True,
        global_node_count=2, exploration_node_count=1,
        global_path=[dict(xyz_m=[0,0,0],native_node_type=0,native_subspace_index=-1),
                     dict(xyz_m=[10,0,5],native_node_type=1,native_subspace_index=1)],
        exploration_path=[dict(xyz_m=[1,0,0],native_node_type=2,native_subspace_index=-1)],
        lookahead_xyz_m=[1,0,0], lookahead_updated=False, follow_local_path_from_start=False)
    value.update(changes)
    return value


def test_actual_paths_link_extended_waypoint_without_claiming_task_execution():
    r = registry()
    r.consume('decision', decision(), order=1, record_ref='intent:1')
    assert feed(r, paths())['accepted']
    assert feed(r, event('WAYPOINT',3,xyz_m=[8,0,0]))['accepted']
    s = r.snapshot()
    assert s['waypoint_route_links'][-1]['planning_path_bound']
    assert s['feedback_records']['synthetic:event:2']['payload']['global_path'][1]['xyz_m'][2] == 5
    assert not s['attempts'] and not s['confirmed_traversals']
    assert not s['waypoint_route_links'][-1]['task_completed']


@pytest.mark.parametrize('mutation', [dict(coordinate_frame='sensor'),dict(global_node_count=3),
    dict(paths_complete=1),dict(lookahead_updated=None),dict(lookahead_xyz_m=[float('nan'),0,0]),
    dict(evidence_scope='executed'),dict(global_path=None),dict(paths_complete=False),
    dict(global_path=[dict(xyz_m=[0,0,0],native_node_type=9,native_subspace_index=-1)] ,global_node_count=1)])
def test_invalid_paths_cannot_bind_or_mutate_tasks(mutation):
    r = registry(); before = r.snapshot()['tasks']
    assert not feed(r, paths(**mutation))['accepted']
    feed(r, event('WAYPOINT',3,xyz_m=[8,0,0]))
    assert not r.snapshot()['waypoint_route_links'][-1]['planning_path_bound']
    assert r.snapshot()['tasks'] == before


@pytest.mark.parametrize('interruption',['gap','rejected','sources','epoch'])
def test_no_stale_path_reuse(interruption):
    r = registry(); feed(r, paths())
    idx = 3
    kwargs = {}
    if interruption == 'gap': idx=4
    if interruption == 'rejected':
        feed(r,event('NATIVE_PLANNING_PATHS_REJECTED',3,reason='bad')); idx=4
    if interruption == 'sources': kwargs['source_frame_keys']=['different']
    if interruption == 'epoch': kwargs['epoch']='synthetic:2'
    feed(r,event('WAYPOINT',idx,xyz_m=[8,0,0],**kwargs))
    assert not r.snapshot()['waypoint_route_links'][-1]['planning_path_bound']


def test_capacity_reports_missing_evidence_not_truncated_path():
    r=registry()
    value=paths(global_node_count=16385,global_path=None,exploration_path=None,paths_complete=False)
    assert feed(r,value)['accepted']
    feed(r,event('WAYPOINT',3,xyz_m=[8,0,0]))
    assert not r.snapshot()['waypoint_route_links'][-1]['planning_path_bound']


def test_later_path_does_not_rewrite_prefix_or_imply_connection():
    r=registry(); feed(r,event('WAYPOINT',1,xyz_m=[8,0,0]))
    prefix=deepcopy(r.snapshot()['waypoint_route_links'])
    feed(r,paths())
    assert r.snapshot()['waypoint_route_links']==prefix
    assert not r.snapshot()['confirmed_traversals']
