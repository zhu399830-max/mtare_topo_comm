import numpy as np
import pytest
from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline


def pipeline():
    return LiveGeometryPipeline(composition_policy=dict(max_residual_m=.01,
        min_crossing_sine=.1,endpoint_tolerance_m=1e-8,maximum_candidates=32),
        anchor_spacing_m=4.,lookahead_m=4.)


def push(p,i,pose=None):
    return p.push(np.full((16,720),50.,np.float32),np.zeros((16,720),np.uint8),
        sensor_to_map=np.eye(4) if pose is None else pose,stamp_sec=float(i),
        source_key='scan:'+str(i),coordinate_frame='map')


def test_empty_live_sequence_does_not_invent_geometry_or_control():
    p=pipeline()
    for i in range(4):assert push(p,i) is None
    result=push(p,4)
    assert result['geometry']['source_frame_keys']==['scan:'+str(i) for i in range(5)]
    assert result['geometry']['primitives']==[]
    assert result['decision']['selected'] is None
    assert result['control_published'] is False
    assert push(p,5)['geometry']['timestamp_kind']=='scan_seconds'


def test_relative_tilt_rejected_without_mutating_history():
    p=pipeline();push(p,0)
    pose=np.eye(4);a=.1
    pose[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
    with pytest.raises(ValueError,match='roll/pitch'):push(p,1,pose)
    assert len(p.frames)==1


def test_noncausal_scan_rejected():
    p=pipeline();push(p,1)
    with pytest.raises(ValueError,match='causal'):push(p,1)


def test_current_pose_not_replaced_with_replay_origin():
    p=pipeline();pose=np.eye(4);pose[:3,3]=[30,40,2]
    for i in range(5):result=push(p,i,pose)
    assert result['geometry']['sensor_to_local_odometry']==pose.tolist()
    assert p.graph.snapshot()['nodes'][0]['xyz_m']==[30,40,2]
