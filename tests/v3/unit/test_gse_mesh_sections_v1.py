import numpy as np
import pytest

from mtare_topo.teacher.gse_mesh_sections_v1 import mesh_section


def prism(offset=0):
    ring = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    vertices = np.array([[x,y,z+offset] for x in (-2,2) for y,z in ring],dtype=float)
    faces = []
    for a in range(4):
        b = (a+1)%4
        faces.extend([[a,b,b+4],[a,b+4,a+4]])
    return vertices,np.array(faces)


def section(v,f):
    return mesh_section(v,f,center_m=[0,0,0],normal=[1,0,0])


def test_actual_mesh_cross_section_without_hull():
    v,f = prism(); result = section(v,f)
    assert len(result.loops_m)==1 and len(result.loops_m[0])==8
    assert len(result.intersected_triangles)==8
    assert np.all(result.loops_m[0][:,0]==0)
    np.testing.assert_equal(np.max(np.abs(result.loops_m[0][:,1:]),axis=0),[1,1])


def test_stacked_sections_never_join():
    a,fa = prism(); b,fb = prism(5)
    result = section(np.concatenate([a,b]),np.concatenate([fa,fb+8]))
    assert len(result.loops_m)==2
    assert all(np.ptp(loop[:,2])==2 for loop in result.loops_m)


def test_open_gap_does_not_get_closed():
    v,f = prism()
    with pytest.raises(ValueError,match="open or nonmanifold"): section(v,f[:-1])


def test_vertex_contact_is_not_perturbed():
    v,f = prism()
    with pytest.raises(ValueError,match="source vertex"):
        mesh_section(v,f,center_m=[2,0,0],normal=[1,0,0])


def test_face_permutation_and_normal_flip_preserve_geometry():
    v,f = prism(); a = section(v,f)
    b = mesh_section(v,f[::-1],center_m=[0,0,0],normal=[-1,0,0])
    np.testing.assert_array_equal(a.loops_m[0],b.loops_m[0])
    with pytest.raises(ValueError): a.loops_m[0][0,0]=3


def test_duplicate_faces_rejected():
    v,f = prism()
    with pytest.raises(ValueError,match="duplicate"): section(v,np.concatenate([f,f[:1]]))


def test_plane_outside_is_empty_not_closed_portal():
    v,f = prism()
    assert not mesh_section(v,f,center_m=[3,0,0],normal=[1,0,0]).loops_m


@pytest.mark.parametrize("length,valid,expected",[(4,True,True),(0.5,True,False),(1,True,False),(4,False,False)])
def test_finite_first_return_evidence(length,valid,expected):
    from mtare_topo.teacher.gse_mesh_sections_v1 import section_ray_witnesses
    from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
    v,f = prism()
    rays = CausalRaySegments(np.array([[-1.,0,0]]),np.array([[1.,0,0]]),
        np.array([length],dtype=np.float32),np.array([valid]),np.array([0]),0,0.)
    _, witnesses = section_ray_witnesses(v,f,center_m=[0,0,0],normal=[1,0,0],rays=rays)
    assert bool(witnesses[0]) == expected
