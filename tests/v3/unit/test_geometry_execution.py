from mtare_topo.integration.geometry_execution import GeometryExecution
import pytest

def runtime():return GeometryExecution(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1)
def proposals():return [dict(axis_target_xyz_m=[x,0,1],source_refs=['scan:1/ray:4']) for x in (-4,4)]
def update(r,t,xyz=(0,0,.75),heading=(1,0,0)):
    return r.update(proposals(),body_xyz_m=xyz,body_forward_world=heading,stamp_sec=t)

def test_body_heading_selects_goal_not_fixed_lidar_heading():
    assert update(runtime(),1,heading=(-1,0,0))['xyz_m'][0]==-4

def test_request_does_not_claim_ground_or_execution():
    r=runtime();a=update(r,1);b=update(r,2)
    assert not a['ground_connection_verified'] and not a['executed']
    assert b['status']=='PENDING_LOCAL_EXECUTION'
    assert len(r.events)==1

def test_arrival_from_pose_and_previously_arrived_goal_not_reselected():
    r=runtime();update(r,1);a=update(r,2,xyz=(4,0,.75))
    assert r.events[1]['kind']=='XY_ARRIVAL_FROM_ODOMETRY'
    assert not r.events[1]['edge_verified']
    assert a['xyz_m'][0]==-4

def test_timeout_holds_and_does_not_restart_failed_goal():
    r=runtime();update(r,1)
    assert update(r,32)['status']=='HOLD_AFTER_TIMEOUT'
    assert update(r,33)['xyz_m']==[0.,0.,.75]

def test_future_reordering_rejected():
    r=runtime();update(r,1)
    with pytest.raises(ValueError):update(r,1)
