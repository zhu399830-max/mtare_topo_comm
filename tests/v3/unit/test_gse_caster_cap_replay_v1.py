from types import SimpleNamespace
import numpy as np
import pytest
from mtare_topo.teacher.gse_caster_cap_replay_v1 import pack_caster_inputs,replay_cap


def test_pack_matches_original_operation_order_without_second_normalization():
    v=np.array([[1000.00003,0.,0.]])
    o=np.array([[999.,.2,.2]]);d=np.array([[1.,2.,3.]])
    a,b=pack_caster_inputs(v,o,d)
    np.testing.assert_array_equal(a,v.astype(np.float32))
    np.testing.assert_array_equal(b,np.concatenate((o,d/np.linalg.norm(d,axis=1)[:,None]),axis=1).astype(np.float32))
    assert a[0,0]==1000.


def test_actual_backend_matches_quantized_cap_not_float64_plane():
    o3d=pytest.importorskip('open3d')
    mesh=SimpleNamespace(vertices_xyz_m=np.array([[1000.00003,0.,0.],[1000.00003,1.,0.],[1000.00003,0.,1.]]),
                         triangle_vertex_indices=np.array([[0,1,2]]))
    kw=dict(cap_face_indices=np.array([0]),origins=np.array([[999.,.2,.2]]),
            directions=np.array([[1.,0.,0.]]),valid=np.array([True]))
    # Baseline is the actual source backend parameter, not analytic 1 metre.
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(mesh.vertices_xyz_m.astype(np.float32)),
        o3d.core.Tensor(mesh.triangle_vertex_indices.astype(np.uint32))))
    raw=scene.list_intersections(o3d.core.Tensor(np.array([[999.,.2,.2,1.,0.,0.]],np.float32)))
    stored=raw['t_hit'].numpy().copy()
    r=replay_cap(mesh,first_return=stored,**kw)
    assert r['exact_cap_witnesses']==[dict(ray_index=0,triangle_index=0,stored_t=float(stored[0]))]
    r=replay_cap(mesh,first_return=np.array([2.],dtype=np.float32),**kw)
    assert r['exact_cap_witnesses']==[]
