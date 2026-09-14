import numpy as np
import pytest
from mtare_topo.teacher.gse_cap_return_evidence_v1 import cap_return_evidence
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments


def check(length=1., valid=True, yz=(.2, .2), frame=4):
    vertices = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]])
    rays = CausalRaySegments(np.array([[0., *yz]]), np.array([[1., 0., 0.]]),
        np.array([length], dtype=np.float32), np.array([valid]), np.array([frame]), 4, 0.)
    return cap_return_evidence(vertices, np.array([[0, 1, 2]]),
                               cap_face_indices=np.array([0]), rays=rays)


def test_true_cap_return_preserves_ray_and_frame_without_fake_terminal():
    result = check()
    assert result['cap_return_witnesses'] == [dict(ray_index=0, triangle_index=0, source_frame_index=4)]
    assert result['terminal_label'] is None and not result['training_eligible']


@pytest.mark.parametrize('length', [.5, 2.])
def test_foreground_occlusion_and_return_beyond_plane_are_not_cap_hits(length):
    assert not check(length)['cap_surface_observed']


def test_invalid_return_and_outside_triangle():
    assert not check(float('nan'), False)['cap_surface_observed']
    assert not check(yz=(.8, .8))['cap_surface_observed']


def test_triangle_boundary_is_explicitly_ambiguous():
    result = check(yz=(.5, .5))
    assert result['ambiguous_ray_indices'] == [0]
    assert not result['cap_surface_observed']


def test_future_return_rejected():
    with pytest.raises(ValueError, match='future'):
        check(frame=5)


def test_source_cap_binding_uses_actual_mesh_builder_and_both_endpoints():
    from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipsePrimitive
    from mtare_topo.teacher.gse_cap_return_evidence_v1 import source_endpoint_cap_faces
    primitive = SweptSuperellipsePrimitive('synthetic', np.array([[-2., 0., 0.], [2., 0., 0.]]),
                                           ((1., 1.), (1., 1.)), (2., 2.))
    mesh = mesh_swept_superellipse(primitive, axial_spacing_m=.05, angular_segments=64)
    start = source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=0)
    end = source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=1)
    assert len(start) == len(end) == 64 and not set(start) & set(end)
    assert np.all(mesh.triangle_vertex_indices[start, 0] == len(mesh.vertices_xyz_m)-2)
    # A changed/ROI triangulation cannot masquerade as this original cap.
    mesh.triangle_vertex_indices[start[0]] = mesh.triangle_vertex_indices[0]
    with pytest.raises(ValueError, match='ownership'):
        source_endpoint_cap_faces(mesh, angular_segments=64, endpoint_index=0)
