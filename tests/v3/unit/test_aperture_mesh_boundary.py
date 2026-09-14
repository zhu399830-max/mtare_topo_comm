import numpy as np
import pytest
pytest.importorskip('open3d')
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse,ClosedPrimitiveMesh
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
from mtare_topo.teacher.aperture_mesh_boundary import OriginalMeshBoundary
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere
from mtare_topo.teacher.aperture_component_contract import partition_boundary


def mesh(points,axes=((.8,.8),(.8,.8))):
    p=SweptSuperellipsePrimitive('p',np.array(points,float),axes,(2.,2.))
    return mesh_swept_superellipse(p,axial_spacing_m=.5,angular_segments=16)


@pytest.mark.parametrize('points,axes',[
    ([[-4,0,0],[4,0,0]],((.8,.8),(.8,.8))),
    ([[-4,0,0],[0,0,0],[0,4,0]],((.8,.8),(.8,.8))),
    ([[-4,0,0],[4,0,0]],((.8,.6),(1.2,.9))),
])
def test_original_curved_and_tapered_mesh_without_replacement(points,axes):
    m=mesh(points,axes);before=m.vertices_xyz_m.copy();v,f,a=octahedral_sphere(3)
    result=OriginalMeshBoundary([m]).classify(v,f,radius_m=2.)
    np.testing.assert_array_equal(before,m.vertices_xyz_m)
    p=partition_boundary(a,result.state,np.full(len(f),-1))
    assert len(p.geometry_components)==2 and not p.unresolved_component_pairs
    assert np.any(result.state==-1) and not result.training_labels_qualified


def test_closed_mesh_required_and_work_cap_does_not_truncate():
    m=mesh([[-4,0,0],[4,0,0]])
    bad=ClosedPrimitiveMesh('bad',m.vertices_xyz_m,m.triangle_vertex_indices[:-1],m.triangle_normals[:-1])
    with pytest.raises(ValueError):OriginalMeshBoundary([bad])
    v,f,a=octahedral_sphere(2)
    with pytest.raises(RuntimeError):OriginalMeshBoundary([m]).classify(v,f,radius_m=2.,max_triangle_point_evaluations=0)


def test_winding_failure_remains_unknown_not_blocked(monkeypatch):
    import mtare_topo.teacher.aperture_mesh_boundary as module
    m=mesh([[-4,0,0],[4,0,0]]);v,f,a=octahedral_sphere(3)
    engine=OriginalMeshBoundary([m]);baseline=engine.classify(v,f,radius_m=2.)
    def fail(*args):raise ValueError('unqualified winding')
    monkeypatch.setattr(module,'oriented_winding',fail)
    unknown=engine.classify(v,f,radius_m=2.)
    assert np.all(unknown.state[baseline.state==1]==-1)


def test_nested_shell_multiplicity_and_separate_layer():
    outer=mesh([[-4,0,0],[4,0,0]]);inner=mesh([[-4,0,0],[4,0,0]],((.6,.6),(.6,.6)))
    combined=ClosedPrimitiveMesh('overlap',np.concatenate([outer.vertices_xyz_m,inner.vertices_xyz_m]),
        np.concatenate([outer.triangle_vertex_indices,inner.triangle_vertex_indices+len(outer.vertices_xyz_m)]),
        np.concatenate([outer.triangle_normals,inner.triangle_normals]))
    v,f,a=octahedral_sphere(3);result=OriginalMeshBoundary([combined]).classify(v,f,radius_m=2.)
    assert len(partition_boundary(a,result.state,np.full(len(f),-1)).geometry_components)==2
    lower=mesh([[-8,0,0],[8,0,0]]);upper=mesh([[-8,0,3],[8,0,3]])
    v,f,a=octahedral_sphere(4);result=OriginalMeshBoundary([lower,upper]).classify(v,f,radius_m=5.)
    assert len(partition_boundary(a,result.state,np.full(len(f),-1)).geometry_components)==4
