import numpy as np
import pytest
from mtare_topo.teacher.csg_mesh_provenance import ClosedPrimitiveMesh,CSGMeshProvenanceRaycaster
from mtare_topo.teacher.mesh_interval_exit import interval_winding_exit


def box(lo,hi):
    v=np.array([[x,y,z] for x in (lo,hi) for y in (-1.,1.) for z in (-1.,1.)])
    faces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
    indices=np.array([t for a,b,c,d in faces for t in ((a,b,c),(a,c,d))])
    normals=np.cross(v[indices[:,1]]-v[indices[:,0]],v[indices[:,2]]-v[indices[:,0]])
    for i in range(len(indices)):
        if normals[i]@(v[indices[i]].mean(axis=0)-v.mean(axis=0))<0:indices[i]=indices[i,::-1]
    return v,indices


def mesh(boxes):
    vs=[];ts=[];offset=0
    for lo,hi in boxes:
        v,t=box(lo,hi);vs.append(v);ts.append(t+offset);offset+=len(v)
    v=np.concatenate(vs);t=np.concatenate(ts)
    n=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
    n/=np.linalg.norm(n,axis=1)[:,None]
    return ClosedPrimitiveMesh('box',v,t,n)


def query(meshes,distances,sources,initial=None):
    if initial is None:initial=[True]*len(meshes)
    return interval_winding_exit(meshes,[0.,.13,.17],[1.,0.,0.],initial,distances,sources)


def test_internal_overlap_exits_last_not_nearest_face():
    m=mesh([(-1.,.012),(.004,.006)])
    assert query([m],[.004,.006,.012],[0,0,0])==(.012,(0,))


def test_duplicate_same_surface_and_tangent_event_do_not_create_exit():
    m=mesh([(-1.,1.)])
    assert query([m],[.5,1.,1.],[0,0,0])==(1.,(0,))
    # An entire duplicate closed shell changes multiplicity, not boundary.
    duplicate=mesh([(-1.,1.),(-1.,1.)])
    assert query([duplicate],[1.,1.],[0,0])==(1.,(0,))


def test_separate_tunnel_does_not_steal_first_exit():
    assert query([mesh([(-1.,1.)]),mesh([(2.,3.)])],[1.,2.,3.],[0,1,1],[True,False])==(1.,(0,))


def test_coincident_operands_keep_all_sources():
    m=mesh([(-1.,1.)])
    assert query([m,m],[1.,1.],[0,1])==(1.,(0,1))


def test_incomplete_or_inconsistent_evidence_rejects():
    m=mesh([(-1.,1.)])
    assert query([m],[1.],[0],[False]) is None
    assert query([m],np.arange(1.,131.),[0]*130) is None
    damaged=ClosedPrimitiveMesh('damaged',m.vertices_xyz_m,m.triangle_vertex_indices[:-1],m.triangle_normals[:-1])
    assert query([damaged],[1.],[0]) is None


def test_explicit_caster_rescue_preserves_legacy_and_valid_results():
    pytest.importorskip('open3d')
    m=mesh([(-1.,.012),(.004,.006)])
    args=(np.array([[0.,.13,.17]]),np.array([[1.,0.,0.]]),np.array([[True]]))
    old=CSGMeshProvenanceRaycaster([m]);new=CSGMeshProvenanceRaycaster([m],rescue_missing_with_interval_winding=True)
    assert old.ray_exit_hits(*args)[0] is None
    assert new.ray_exit_hits(*args)[0].distance_m==pytest.approx(.012,abs=1e-8)
    ordinary=mesh([(-1.,1.)])
    old=CSGMeshProvenanceRaycaster([ordinary]);new=CSGMeshProvenanceRaycaster([ordinary],rescue_missing_with_interval_winding=True)
    assert old.ray_exit_hits(*args)==new.ray_exit_hits(*args)
    with pytest.raises(ValueError):CSGMeshProvenanceRaycaster([m],rescue_missing_with_interval_winding=True,require_unique_qualified_candidate=True)
