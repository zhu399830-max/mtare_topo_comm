import numpy as np
import pytest
from mtare_topo.evaluation.gse_ray_triangle_audit import ray_triangle_audit


def check(*,x=.25,t=1.):
    vertices=np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32)
    rays=np.array([[x,.25,0,0,0,1]],np.float32)
    return ray_triangle_audit(vertices,np.array([[0,1,2]]),rays,[0],[0],[t])


def test_exact_triangle_and_normal_direction():
    r=check()
    assert r['exact_float32_t_matches']==1 and r['minimum_barycentric']==.25
    assert r['leaving_hits']==1 and r['entering_hits']==0
    assert not r['label_qualification']


def test_wrong_parameter_is_measured_not_silently_accepted():
    r=check(t=2.)
    assert r['max_abs_t_error_m']==1. and r['exact_float32_t_matches']==0


def test_plane_intersection_outside_triangle_is_detected():
    r=check(x=2.)
    assert r['negative_barycentric_hits']==1 and not r['all_hits_in_triangle_at_float64']


def test_original_quantization_is_required():
    with pytest.raises(ValueError,match='float32'):
        ray_triangle_audit(np.zeros((3,3)),np.array([[0,1,2]]),np.zeros((1,6)),[0],[0],[1.])


def test_world_coordinate_quantization_can_create_forward_entrance():
    vertices=np.array([[70.000003,0,0],[70.123003,1,0],[70.000003,0,1]],np.float64)
    origin=np.array([70.000003+.123*.2,.2,.2])
    origin[0]=np.nextafter(origin[0],-np.inf)
    direction=np.array([-1.,0,0])
    normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0])
    original_t=(vertices[0]-origin)@normal/(direction@normal)
    packed=np.concatenate((origin,direction))[None].astype(np.float32)
    quantized=ray_triangle_audit(vertices.astype(np.float32),np.array([[0,1,2]]),packed,[0],[0],[1.])
    assert original_t<=0 and quantized['minimum_t']>0
    assert quantized['entering_hits']==1 and quantized['all_hits_in_triangle_at_float64']
    assert not quantized['label_qualification']
