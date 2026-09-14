import numpy as np
import pytest

from mtare_topo.teacher.gse_local_section_evidence_v1 import local_section_evidence
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments


def fixture():
    # A plain uninterrupted tunnel, not a junction. All geometry deliberately
    # avoids grid planes so this checks semantics rather than round-off.
    ring = [(.1, .1), (.6, .1), (.6, .6), (.1, .6)]
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y, z in ring])
    faces = np.array([[a, (a+1) % 4, (a+1) % 4+4] for a in range(4)]
                     + [[a, (a+1) % 4+4, a+4] for a in range(4)])
    surface = np.array([[.1, y, z] for y in (.1, .35, .6)
                        for z in (.1, .35, .6) if y != .35 or z != .35])
    # Return on each of eight boundary cells, plus a finite crossing ray.
    origins = np.concatenate([surface - [1., 0., 0.], [[-.9, .35, .35]]])
    directions = np.tile([1., 0., 0.], (9, 1))
    rays = CausalRaySegments(origins, directions, np.array([1.] * 8 + [2.]),
                            np.ones(9, dtype=bool), np.arange(9) % 5, 4, 0.)
    return vertices, faces, rays


def run(rays):
    v, f, _ = fixture()
    return local_section_evidence(v, f, center_m=[.1, .35, .35],
                                  normal=[1., 0., 0.], rays=rays)['sections'][0]


def test_straight_tunnel_satisfies_both_without_creating_node_or_label():
    _, _, rays = fixture()
    result = run(rays)
    assert result['joint_local_evidence']
    assert result['exclusive_crossing_ray_indices'] == [8]
    assert result['anchor_label'] is None
    assert result['semantic_opening_label'] is None
    assert result['training_eligible'] is False


def test_interior_ray_without_observed_boundary_stays_incomplete():
    _, _, rays = fixture()
    rays.valid[:8] = False
    result = run(rays)
    assert not result['joint_local_evidence']
    assert result['missing_evidence'] == ['contour_surface_support']


def test_boundary_without_crossing_stays_incomplete():
    _, _, rays = fixture()
    rays.first_return_m[8] = .5
    result = run(rays)
    assert not result['joint_local_evidence']
    assert result['missing_evidence'] == ['finite_interior_crossing']


def test_future_evidence_rejected():
    _, _, rays = fixture()
    rays.source_frame_index[0] = 5
    with pytest.raises(ValueError, match='future'):
        run(rays)
