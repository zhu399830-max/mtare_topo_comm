import numpy as np
from mtare_topo.integration.branch_geometry_execution import BranchGeometryExecution

def runtime():return BranchGeometryExecution(execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1),
    place_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3))
def result(i):
    p=[dict(axis_start_xyz_m=[0,0,0],axis_target_xyz_m=t,source_refs=['ray:'+str(j)],current_direction_supported=True) for j,t in enumerate([[4,0,0],[-4,0,0],[0,4,0]])]
    return dict(geometry=dict(timestamp=i,coordinate_frame='map',sensor_to_local_odometry=np.eye(4).tolist()),decision=dict(proposals=p,node=0))

def test_geometry_place_changes_next_goal_and_records_exact_task_arrival():
    r=runtime()
    for i in (0,1):r.update_geometry(result(i),body_xyz_m=[0,0,0],body_forward_world=[1,0,0],stamp_sec=i)
    q=r.update_geometry(result(2),body_xyz_m=[4,0,0],body_forward_world=[1,0,0],stamp_sec=2)
    assert q['xyz_m']==[0.,4.,0.]
    task=q['active']['branch_task']
    assert r.places.places[task['place_id']]['frontiers'][task['frontier_id']]['state']=='requested'
    r.update_geometry(result(3),body_xyz_m=[0,4,0],body_forward_world=[0,1,0],stamp_sec=3)
    assert r.places.places[task['place_id']]['frontiers'][task['frontier_id']]['state']=='xy_arrived'
    assert r.places.snapshot()['verified_edges']==[]

def test_remote_task_not_sent_as_unverified_straight_line_goal():
    r=runtime()
    for i in range(3):r.update_geometry(result(i),body_xyz_m=[0,0,0],body_forward_world=[1,0,0],stamp_sec=i)
    other=result(3);other['decision']['proposals']=[]
    q=r.update_geometry(other,body_xyz_m=[4,0,0],body_forward_world=[1,0,0],stamp_sec=3)
    assert q['status']=='HOLD_NO_UNVISITED_GEOMETRY_TARGET'
    assert r.snapshot()['remote_return_implemented'] is False

def test_learned_corridor_fallback_requires_current_support_and_keeps_raw():
    r=runtime();data=result(0);data['geometry']['frontend']='frozen_learned_geometry'
    data['decision']['proposals'][0]['current_direction_supported']=False
    data['decision']['proposals'][1]['current_direction_supported']=None
    q=r.update_geometry(data,body_xyz_m=[0,0,0],body_forward_world=[1,0,0],stamp_sec=0)
    assert q['xyz_m']==[0.,4.,0.]
    assert len(data['decision']['proposals'])==3
    assert data['execution_support_audit']==dict(total_proposals=3,supported_proposals=1,
        unsupported_proposals=1,unknown_proposals=1,raw_graph_proposals_preserved=True)
    assert not r.places.places  # unsupported proposals cannot create a place

def test_learned_all_unknown_holds_without_issuing_target():
    r=runtime();data=result(0);data['geometry']['frontend']='frozen_learned_geometry'
    for p in data['decision']['proposals']:p['current_direction_supported']=None
    q=r.update_geometry(data,body_xyz_m=[0,0,0],body_forward_world=[1,0,0],stamp_sec=0)
    assert q['status']=='HOLD_NO_UNVISITED_GEOMETRY_TARGET'
    assert not r.events and not r.places.places
