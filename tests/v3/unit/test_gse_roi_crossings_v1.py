import numpy as np
import pytest

from mtare_topo.teacher.gse_roi_crossings_v1 import roi_crossings


def run(points):
    return roi_crossings(points,center_m=[0,0,0])


def test_through_tunnel_has_two_boundary_candidates_not_nodes():
    r = run([[-20,0,0],[20,0,0]])
    np.testing.assert_allclose(r.positions_m,[[-10,0,0],[10,0,0]])
    np.testing.assert_allclose(r.outward_tangents,[[-1,0,0],[1,0,0]])
    np.testing.assert_allclose(r.source_arc_m,[10,30])
    assert not r.observed_openings and not r.persistent_nodes
    assert not r.ambiguous_segment_indices


def test_reverse_preserves_geometry_and_outward_direction():
    p = [[-20,0,3],[0,0,3],[20,0,3]]
    a,b = run(p),run(p[::-1])
    np.testing.assert_allclose(a.positions_m,b.positions_m[::-1])
    np.testing.assert_allclose(a.outward_tangents,b.outward_tangents[::-1])


def test_stacked_layers_and_rigid_transform_are_3d():
    p = np.array([[-20.,0,3],[20,0,3]])
    a = run(p)
    b = run(p*np.array([1,1,-1]))
    assert not np.array_equal(a.positions_m,b.positions_m)
    rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    shift = np.array([8.,-2,4])
    c = roi_crossings(p@rotation.T+shift,center_m=shift)
    np.testing.assert_allclose(c.positions_m,a.positions_m@rotation.T+shift)
    np.testing.assert_allclose(c.outward_tangents,a.outward_tangents@rotation.T)


def test_inside_crop_end_is_not_an_opening():
    assert len(run([[-2,0,0],[2,0,0]]).positions_m) == 0


@pytest.mark.parametrize("points",[[[-20,10,0],[20,10,0]],[[-20,0,0],[-10,0,0],[0,0,0]]])
def test_tangent_or_vertex_is_reported_not_shifted(points):
    r = run(points)
    assert r.ambiguous_segment_indices and len(r.positions_m) == 0


def test_all_reentries_retained_no_tunnel_collapse():
    r = run([[-20,0,0],[20,0,0],[-20,0,0]])
    assert len(r.positions_m) == 4
    assert r.segment_indices.tolist() == [0,0,1,1]
    with pytest.raises(ValueError): r.positions_m[0,0] = 9


@pytest.mark.parametrize("points",[[[0,0,0],[0,0,0]],[[0,0,0],[float('nan'),0,0]],[[1,2],[3,4]]])
def test_invalid_sources_fail(points):
    with pytest.raises(ValueError): run(points)


def test_deterministic_and_outside_no_false_crossing():
    assert not len(run([[-20,11,0],[20,11,0]]).positions_m)
    p = [[-14,2,0],[12,4,1]]
    a,b = run(p),run(p)
    assert a.positions_m.tobytes() == b.positions_m.tobytes()
    np.testing.assert_allclose(np.linalg.norm(a.positions_m,axis=1),10,atol=1e-12)
