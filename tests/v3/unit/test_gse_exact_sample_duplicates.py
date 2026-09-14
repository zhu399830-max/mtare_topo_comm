import numpy as np
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive, _sample_operand
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse


def test_rotated_translated_endpoint_no_nan_or_duplicate_ring():
    angle=np.deg2rad(17.);c,s=np.cos(angle),np.sin(angle)
    rot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    for axis in ([[-1.,0.,0.],[15.,0.,0.]],[[1.,0.,0.],[-15.,0.,0.]],[[0.,1.,0.],[0.,-15.,0.]]):
        xyz=np.array(axis)@rot.T+np.array([70.000003,70.000003,0.])
        p=SweptSuperellipsePrimitive('p',xyz,((2.,2.),(2.,2.)),(2.,2.))
        sampled=_sample_operand(p,.05)
        assert np.isfinite(sampled.tangents).all()
        assert np.all(np.any(np.diff(sampled.points,axis=0)!=0,axis=1))
        assert np.array_equal(sampled.points[-1],xyz[-1])
        mesh=mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64)
        assert np.isfinite(mesh.vertices_xyz_m).all()


def test_regular_sampling_retains_all_original_arcs_and_points():
    xyz=np.array([[0.,0.,0.],[2.,0.,0.]])
    p=SweptSuperellipsePrimitive('p',xyz,((2.,2.),(2.,2.)),(2.,2.))
    sampled=_sample_operand(p,.05)
    expected=np.unique(np.append(np.arange(0.,2.,.05),2.))
    assert np.array_equal(sampled.arc_m,expected)
    assert np.array_equal(sampled.points,np.stack((expected,np.zeros_like(expected),np.zeros_like(expected)),axis=1))
