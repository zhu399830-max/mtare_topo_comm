import numpy as np
import pytest

from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipsePrimitive,
    SweptSuperellipseProvenanceField,
    superellipse_primitives_from_circular_construction,
)
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph


def _primitive(identity, points, axes=((1.0, 1.0), (1.0, 1.0)), exponent=(2.0, 2.0)):
    return SweptSuperellipsePrimitive(identity, np.asarray(points, dtype=float), axes, exponent)


def _hit(field, origin, direction):
    value = field.ray_exit_hits(np.asarray([origin], dtype=float), np.asarray([direction], dtype=float), maximum_m=20.0)[0]
    assert value is not None
    return value


def test_ellipse_recovers_distinct_lateral_and_vertical_half_axes():
    field = SweptSuperellipseProvenanceField([_primitive("ellipse", [[-2, 0, 0], [2, 0, 0]], ((1.2, .8), (1.2, .8)))], spacing_m=.005)
    assert _hit(field, [0, 0, 0], [0, 1, 0]).distance_m == pytest.approx(1.2, abs=.012)
    assert _hit(field, [0, 0, 0], [0, 0, 1]).distance_m == pytest.approx(.8, abs=.012)


def test_rounded_rectangle_exponent_changes_diagonal_without_changing_axes():
    ellipse = SweptSuperellipseProvenanceField([_primitive("e", [[-2, 0, 0], [2, 0, 0]])], spacing_m=.005)
    rounded = SweptSuperellipseProvenanceField([_primitive("r", [[-2, 0, 0], [2, 0, 0]], exponent=(8, 8))], spacing_m=.005)
    direction = np.array([0., 1., 1.]) / np.sqrt(2.)
    assert _hit(ellipse, [0, 0, 0], direction).distance_m == pytest.approx(1., abs=.012)
    assert _hit(rounded, [0, 0, 0], direction).distance_m == pytest.approx(2 ** (0.5 - 1 / 8), abs=.012)


def test_taper_and_shape_transition_use_c1_endpoint_interpolation():
    primitive = _primitive("mixed", [[0, 0, 0], [4, 0, 0]], ((.5, .6), (1.5, 1.2)), (2, 8))
    axes, exponent = primitive.parameters_at_fraction(np.array([0., .5, 1.]))
    assert axes[1].tolist() == pytest.approx([1., .9])
    assert exponent[1] == pytest.approx(5.)
    epsilon = 1e-5
    near_start = primitive.parameters_at_fraction(epsilon)[0]
    near_end = primitive.parameters_at_fraction(1-epsilon)[0]
    assert np.max(np.abs((near_start - axes[0]) / epsilon)) < 1e-3
    assert np.max(np.abs((axes[-1] - near_end) / epsilon)) < 1e-3
    field = SweptSuperellipseProvenanceField([primitive], spacing_m=.005)
    assert _hit(field, [2, 0, 0], [0, 1, 0]).distance_m == pytest.approx(1., abs=.015)


def test_curved_and_sloped_sweep_has_finite_local_cross_section():
    theta = np.linspace(0, np.pi / 2, 81)
    points = np.column_stack((3*np.cos(theta), 3*np.sin(theta), .4*theta))
    field = SweptSuperellipseProvenanceField([_primitive("curve", points, ((.7, .5), (.7, .5)))], spacing_m=.01)
    middle = points[len(points)//2]
    tangent = points[len(points)//2+1] - points[len(points)//2-1]
    lateral = np.cross([0, 0, 1], tangent); lateral = lateral / np.linalg.norm(lateral)
    hit = _hit(field, middle, lateral)
    assert hit.distance_m == pytest.approx(.7, abs=.025)
    assert hit.source_primitive_ids == ("curve",)


@pytest.mark.parametrize("branches", [
    [(0, 3, 0)],
    [(-2.2, 2.2, 0), (2.2, 2.2, 0)],
    [(0, 3, 0), (0, -3, 0)],
])
def test_t_y_x_unions_exit_through_outer_branch_not_internal_wall(branches):
    primitives = [_primitive("trunk", [[-3, 0, 0], [3, 0, 0]], ((.5, .5), (.5, .5)))]
    for index, endpoint in enumerate(branches):
        primitives.append(_primitive(f"branch{index}", [[0, 0, 0], endpoint], ((.5, .5), (.5, .5))))
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.005)
    direction = np.asarray(branches[0], dtype=float); direction /= np.linalg.norm(direction)
    hit = _hit(field, [0, 0, 0], direction)
    assert hit.distance_m > np.linalg.norm(branches[0])
    assert f"branch0" in hit.source_primitive_ids


def test_stacked_nonincident_superellipses_preserve_first_surface_identity():
    lower = _primitive("lower", [[-2, 0, 0], [2, 0, 0]], ((1, .8), (1, .8)), (6, 6))
    upper = _primitive("upper", [[-2, 0, 3], [2, 0, 3]], ((1, .8), (1, .8)), (6, 6))
    field = SweptSuperellipseProvenanceField([lower, upper], spacing_m=.005)
    hit = _hit(field, [0, 0, 0], [0, 0, 1])
    assert hit.distance_m == pytest.approx(.8, abs=.012)
    assert hit.source_primitive_ids == ("lower",)


def test_identical_shape_operands_remain_explicitly_ambiguous():
    one = _primitive("one", [[-2, 0, 0], [2, 0, 0]], exponent=(6, 6))
    two = _primitive("two", [[-2, 0, 0], [2, 0, 0]], exponent=(6, 6))
    hit = _hit(SweptSuperellipseProvenanceField([one, two], spacing_m=.005), [0, 0, 0], [0, 1, 0])
    assert hit.source_primitive_ids == ("one", "two")
    assert not hit.provenance_unique


def test_archived_circle_lifts_without_identity_or_radius_change():
    graph = {
        "nodes": [{"id": "a", "xyz": [0, 0, 0], "degree": 1}, {"id": "b", "xyz": [2, 0, 0], "degree": 1}],
        "edges": [{"id": "edge", "node_ids": ["a", "b"], "tunnel_ids": [7]}],
    }
    construction = build_primitive_construction_graph(
        graph,
        {"tunnels": [{"tunnel_id": 7, "points": [[0, 0, 0], [2, 0, 0]]}]},
        {"tunnels": [{"tunnel_id": 7, "radius_m": .75}]},
    )
    lifted = superellipse_primitives_from_circular_construction(construction)
    assert len(lifted) == 1 and lifted[0].primitive_id == "primitive:edge"
    assert lifted[0].endpoint_half_axes_m == ((.75, .75), (.75, .75))
    assert lifted[0].endpoint_shape_exponent == (2., 2.)
    hit = _hit(SweptSuperellipseProvenanceField(lifted, spacing_m=.005), [1, 0, 0], [0, 1, 0])
    assert hit.distance_m == pytest.approx(.75, abs=.012)


def test_sparse_operand_query_matches_full_values_for_all_possible_sources():
    primitives = [
        _primitive("one", [[-2, 0, 0], [2, 0, 0]], ((1.2, .8), (1.2, .8)), (4, 4)),
        _primitive("two", [[0, -2, 0], [0, 2, 0]], ((.7, 1.1), (.7, 1.1)), (8, 8)),
    ]
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.01)
    points = np.asarray([[0,0,0], [0,1.0,0], [1.1,0,0], [10,10,10]], dtype=float)
    full = field.operand_signed_distances(points)
    sparse = field.operand_signed_distances_sparse(points)
    possible = np.isfinite(sparse)
    assert sparse[possible] == pytest.approx(full[possible])
    assert np.array_equal(np.min(sparse, axis=1) <= 0, np.min(full, axis=1) <= 0)
