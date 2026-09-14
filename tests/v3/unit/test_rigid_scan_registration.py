import numpy as np
import pytest
from mtare_topo.semantics.rigid_scan_registration import register_returns,rotate_sector_mask
from mtare_topo.semantics.primitive_relation_nonlearning import _register_points,_sector_mask
from mtare_topo.integration.live_geometry_pipeline import LiveGeometryPipeline


def values():
    v=np.zeros((5,2,16,720),np.float32)
    v[:,0]=1;v[:,0,8,0]=.1;v[:,1,8,0]=1
    return v


def test_yaw_registration_agrees_with_existing_implementation():
    y=np.array([20,15,10,5,0]);a=np.radians(y)
    r=np.array([[[np.cos(x),-np.sin(x),0],[np.sin(x),np.cos(x),0],[0,0,1]] for x in a])
    t=np.zeros((5,3));t[:,0]=[-4,-3,-2,-1,0]
    actual,ids=register_returns(values(),t,r)
    expected,_=_register_points(values(),t,y)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_array_equal(ids,np.arange(5)*11520+8*720)


def test_pitch_and_roll_change_xyz_not_just_xy():
    r=np.repeat(np.eye(3)[None],5,axis=0)
    r[0]=[[0,0,1],[0,1,0],[-1,0,0]]
    t=np.zeros((5,3));t[0]=[1,2,3]
    registered,_=register_returns(values(),t,r)
    baseline,_=register_returns(values(),np.zeros((5,3)),np.repeat(np.eye(3)[None],5,axis=0))
    p=baseline[0]
    np.testing.assert_allclose(registered[0],[p[2]+1,p[1]+2,-p[0]+3])


def test_rotated_sector_changes_candidate_heading():
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_array_equal(rotate_sector_mask(_sector_mask(0,10),r),_sector_mask(90,10))


def test_invalid_rotation_rejected():
    r=np.repeat(np.eye(3)[None],5,axis=0);r[0,0,0]=2
    with pytest.raises(ValueError):register_returns(values(),np.zeros((5,3)),r)


def test_live_rigid_mode_accepts_nonzero_tilt_without_flattening():
    p=LiveGeometryPipeline(composition_policy=dict(max_residual_m=.01,min_crossing_sine=.1,
        endpoint_tolerance_m=1e-8,maximum_candidates=32),anchor_spacing_m=4,lookahead_m=4,rigid_motion=True)
    for i in range(5):
        a=i*.01;pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
        result=p.push(np.full((16,720),50,np.float32),np.zeros((16,720)),sensor_to_map=pose,
                      stamp_sec=i,source_key=str(i),coordinate_frame='map')
    assert result['geometry']['registration']=='full_rigid'
    np.testing.assert_array_equal(result['geometry']['sensor_to_local_odometry'],pose)


def test_general_attitude_self_transform_is_exact_identity():
    from scipy.spatial.transform import Rotation
    p=LiveGeometryPipeline(composition_policy=dict(max_residual_m=.01,min_crossing_sine=.1,
        endpoint_tolerance_m=1e-8,maximum_candidates=32),anchor_spacing_m=4,lookahead_m=4,rigid_motion=True)
    for i in range(5):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.123,.078,.41]).as_matrix()
        result=p.push(np.full((16,720),50,np.float32),np.zeros((16,720)),sensor_to_map=pose,
                      stamp_sec=i,source_key=str(i),coordinate_frame='map')
    assert result['geometry']['registration']=='full_rigid'
