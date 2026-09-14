import numpy as np
import pytest
from mtare_topo.representation.gse_local_ray_segments import clip_observed_rays


def test_outside_return_preserves_interior_ray_not_fake_surface():
    r=clip_observed_rays([[0,0,0]],[[20,0,0]],np.array([True]))
    assert np.array_equal(r.end_xyz_m,[[10,0,0]])
    assert not r.end_is_observed_return[0]
    assert not r.crop_endpoint_is_opening and not r.physical_traversability_qualified
    assert np.array_equal(r.original_return_xyz_m,[[20,0,0]])


def test_inside_return_keeps_measured_surface_and_invalid_unknown():
    r=clip_observed_rays([[0,0,0],[0,0,0]],[[3,0,0],[50,0,0]],np.array([True,False]))
    assert r.ray_indices.tolist()==[0] and r.end_is_observed_return.tolist()==[True]
    assert np.array_equal(r.end_xyz_m,[[3,0,0]])


def test_translated_history_origin_crosses_window():
    r=clip_observed_rays([[-20,0,0]],[ [20,0,0]],np.array([True]))
    assert np.array_equal(r.start_xyz_m,[[-10,0,0]])
    assert np.array_equal(r.end_xyz_m,[[10,0,0]])
    assert r.start_distance_from_origin_m.tolist()==[10]
    assert r.end_distance_from_origin_m.tolist()==[30]


def test_tangent_or_obstacle_before_window_adds_no_segment():
    r=clip_observed_rays([[-20,10,0],[-20,0,0]],[[20,10,0],[-15,0,0]],np.ones(2,dtype=bool))
    assert len(r.ray_indices)==0


def test_rotation_and_order_preserve_geometry():
    o=np.array([[1.,2,3],[0,0,0]]);p=np.array([[15.,2,3],[0,5,0]])
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    a=clip_observed_rays(o,p,np.ones(2,dtype=bool))
    b=clip_observed_rays(o[::-1]@rot.T,p[::-1]@rot.T,np.ones(2,dtype=bool))
    assert np.allclose(a.end_xyz_m[::-1]@rot.T,b.end_xyz_m)
    assert np.array_equal(a.end_is_observed_return[::-1],b.end_is_observed_return)


def test_valid_zero_length_rejected():
    with pytest.raises(ValueError,match='coincide'):
        clip_observed_rays([[0,0,0]],[[0,0,0]],np.array([True]))
