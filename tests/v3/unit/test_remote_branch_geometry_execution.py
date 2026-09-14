from types import SimpleNamespace
import numpy as np
from mtare_topo.integration.branch_geometry_execution import BranchGeometryExecution

def proposal(target):return dict(axis_start_xyz_m=[0,0,0],axis_target_xyz_m=target,source_refs=['current/ray:1'],current_direction_supported=True)
def observation(t,x,proposals):
    pose=np.eye(4);pose[0,3]=x
    return dict(geometry=dict(timestamp=t,coordinate_frame='map',sensor_to_local_odometry=pose.tolist()),
                decision=dict(node=1,proposals=proposals))
def sample(t,x):return dict(order=t,xyz_m=[x,0,0])

def test_remote_return_then_current_geometry_revalidation_preserves_task_identity():
    r=BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
        place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3),remote_returns=True)
    p=[proposal([-4,0,0]),proposal([0,4,0]),proposal([4,0,0])]
    for t in range(3):r.places.observe(observation(t,0,p)['geometry'],p,navigation_node_id=0)
    # Fixture: straight/back tasks were previously completed; side task remains.
    for i in (0,2):
        r.places.record_attempt(0,i,outcome='requested',stamp_sec=2)
        r.places.record_attempt(0,i,outcome='xy_arrived',stamp_sec=3)
    graph=SimpleNamespace(nodes=[dict(id=0,xyz_m=[0,0,0]),dict(id=1,xyz_m=[4,0,0])],
        edges=[dict(source=0,target=1,length_m=4,trace=[sample(0,0),sample(2,0),sample(2.5,2),sample(3,4)])],
        pending=[sample(3,4),sample(4,5)],current=1)
    forward=[proposal([9,0,0])]
    q=r.update_geometry(observation(4,5,forward),body_xyz_m=[5,0,0],body_forward_world=[1,0,0],stamp_sec=4,navigation_graph=graph)
    assert q['status']=='FOLLOW_RECORDED_RETURN' and q['xyz_m']==[4,0,0]
    assert len(r.deferred)==1 and r.deferred[0]['state']=='unattempted'
    for t,x,next_x in ((5,4,2),(6,2,0)):
        q=r.update_geometry(observation(t,x,forward),body_xyz_m=[x,0,0],body_forward_world=[-1,0,0],stamp_sec=t,navigation_graph=graph)
        assert q['xyz_m']==[next_x,0,0]
    history_only=proposal([0,4,0]);history_only['current_direction_supported']=False
    q=r.update_geometry(observation(7,0,[history_only]),body_xyz_m=[0,0,0],body_forward_world=[0,1,0],stamp_sec=7,navigation_graph=graph)
    assert q['status']=='HOLD_RETURN_REOBSERVATION'
    assert r.places.places[0]['frontiers'][1]['state']=='unattempted'
    q=r.update_geometry(observation(8,0,[proposal([0,4,0])]),body_xyz_m=[0,0,0],body_forward_world=[0,1,0],stamp_sec=8,navigation_graph=graph)
    assert q['status']=='NEW_LOCAL_PLANNING_REQUEST'
    assert q['active']['branch_task']==dict(place_id=0,frontier_id=1)
    assert r.places.places[0]['frontiers'][1]['state']=='requested'
    assert [e['status'] for e in r.return_events][-2:]==['RETURN_PLACE_REACHED','RETURN_DIRECTION_REOBSERVED']


def test_missing_graph_is_not_replaced_with_direct_remote_goal():
    import pytest
    r=BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
        place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3),remote_returns=True)
    with pytest.raises(ValueError,match='recorded navigation graph'):
        r.update_geometry(observation(0,0,[]),body_xyz_m=[0,0,0],body_forward_world=[1,0,0],stamp_sec=0)


def test_updated_frontier_returns_to_its_own_observation_not_place_creation():
    r=BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
        place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3),remote_returns=True)
    p=[proposal([-4,0,0]),proposal([0,4,0]),proposal([4,0,0])]
    for t,x in ((0,0),(1,1),(2,2)):
        r.places.observe(observation(t,x,p)['geometry'],p,navigation_node_id=t)
    tasks=r._remote_candidates(np.array([10,0,0]))
    assert len(tasks)==3
    assert all(t['order']==2 and t['navigation_node_id']==2 and t['observer_xyz_m']==[2,0,0] for t in tasks)
    assert r.places.places[0]['observer_xyz_m']==[0,0,0]
    # If geometry cannot be updated, preserve its matching older binding too.
    r.places.observe(observation(3,3,[])['geometry'],[],navigation_node_id=3)
    assert all(t['order']==2 for t in r._remote_candidates(np.array([10,0,0])))


def test_stale_or_missing_frontier_binding_is_not_guessed():
    import pytest
    r=BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
        place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3),remote_returns=True)
    p=[proposal([-4,0,0]),proposal([0,4,0]),proposal([4,0,0])]
    for t in range(3):r.places.observe(observation(t,0,p)['geometry'],p,navigation_node_id=0)
    task=r.places.places[0]['frontiers'][0]
    task['return_observation']['order']=3
    with pytest.raises(ValueError,match='cannot follow'):r._remote_candidates(np.array([10,0,0]))
    del task['return_observation']
    assert len(r._remote_candidates(np.array([10,0,0])))==2
    assert task['state']=='unattempted'


def test_history_only_update_keeps_directly_supported_return_snapshot():
    from copy import deepcopy
    r=BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
        place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3),remote_returns=True)
    p=[proposal([-4,0,0]),proposal([0,4,0]),proposal([4,0,0])]
    for t in range(3):r.places.observe(observation(t,0,p)['geometry'],p,navigation_node_id=0)
    old=deepcopy(r.places.places[0]['frontiers'][1]['return_observation'])
    history=deepcopy(p)
    for v in history:v['current_direction_supported']=False
    history[1]['axis_target_xyz_m']=[.2,4,0]
    r.places.observe(observation(3,2,history)['geometry'],history,navigation_node_id=2)
    task=r.places.places[0]['frontiers'][1]
    assert task['last_order']==3 and task['target_xyz_m']==[.2,4,0]
    assert task['return_observation']==old
    remote=next(t for t in r._remote_candidates(np.array([10,0,0])) if t['key']=='branch:0:1')
    assert remote['order']==2 and remote['target_xyz_m']==[0,4,0]
    assert remote['observer_xyz_m']==[0,0,0] and remote['navigation_node_id']==0
