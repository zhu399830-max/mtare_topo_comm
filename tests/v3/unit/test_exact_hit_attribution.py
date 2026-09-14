from types import SimpleNamespace
import numpy as np
from mtare_topo.teacher.exact_hit_attribution import attribute_hit


def fixture():
    return SimpleNamespace(vertices_xyz_m=np.array([[2,0,0],[2,1,0],[2,0,1],[2,1,1]],float),
                           triangle_vertex_indices=np.array([[0,1,2],[1,3,2]]))


def test_exact_same_plane_neighbor_replacement_preserves_distance():
    m=fixture();r=attribute_hit(m,0,[0,.75,.75],[1,0,0])
    assert r.replaced and r.original_triangle==0 and r.valid_triangles==(1,) and r.exact_t==2
    assert not attribute_hit(m,0,[0,.25,.25],[1,0,0]).replaced


def test_no_shared_edge_or_nonplanar_neighbor_is_not_a_replacement():
    m=fixture();m.vertices_xyz_m[3,0]=3.
    assert attribute_hit(m,0,[0,.75,.75],[1,0,0]) is None
    m=fixture();m.triangle_vertex_indices=m.triangle_vertex_indices[:1]
    assert attribute_hit(m,0,[0,.75,.75],[1,0,0]) is None


def test_shared_boundary_valid_original_is_retained():
    assert not attribute_hit(fixture(),0,[0,.5,.5],[1,0,0]).replaced
