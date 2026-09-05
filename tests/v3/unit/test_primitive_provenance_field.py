import numpy as np
import pytest

from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.primitive_provenance_field import PrimitiveProvenanceField


def _field(graph, splines, radii, spacing=0.005):
    geometry = {"tunnels": [{"tunnel_id": key, "radius_m": value} for key, value in radii.items()]}
    construction = build_primitive_construction_graph(graph, {"tunnels": splines}, geometry)
    return PrimitiveProvenanceField(construction, spacing_m=spacing)


def test_union_raycast_ignores_internal_t_junction_surfaces():
    graph = {"coordinate_frame": "world", "nodes": [
        {"id": "l", "xyz": [-2, 0, 0], "degree": 1},
        {"id": "j", "xyz": [0, 0, 0], "degree": 3},
        {"id": "r", "xyz": [2, 0, 0], "degree": 1},
        {"id": "t", "xyz": [0, 2, 0], "degree": 1}],
        "edges": [
            {"id": "left", "node_ids": ["l", "j"], "tunnel_ids": [1]},
            {"id": "right", "node_ids": ["j", "r"], "tunnel_ids": [1]},
            {"id": "top", "node_ids": ["j", "t"], "tunnel_ids": [2]}]}
    splines = [
        {"tunnel_id": 1, "points": [[-2, 0, 0], [0, 0, 0], [2, 0, 0]]},
        {"tunnel_id": 2, "points": [[0, 0, 0], [0, 2, 0]]}]
    field = _field(graph, splines, {1: 0.5, 2: 0.5})
    hit = field.ray_exit_hits(np.array([[0., 0., 0.]]), np.array([[0., 1., 0.]]))[0]
    assert hit is not None
    assert hit.distance_m == pytest.approx(2.5, abs=0.012)
    assert hit.source_primitive_ids == ("primitive:top",)


def test_stacked_nonincident_tunnel_cannot_steal_first_exit():
    graph = {"coordinate_frame": "world", "nodes": [
        {"id": "a", "xyz": [-2, 0, 0], "degree": 1}, {"id": "b", "xyz": [2, 0, 0], "degree": 1},
        {"id": "c", "xyz": [-2, 0, 3], "degree": 1}, {"id": "d", "xyz": [2, 0, 3], "degree": 1}],
        "edges": [
            {"id": "lower", "node_ids": ["a", "b"], "tunnel_ids": [1]},
            {"id": "upper", "node_ids": ["c", "d"], "tunnel_ids": [2]}]}
    splines = [
        {"tunnel_id": 1, "points": [[-2, 0, 0], [2, 0, 0]]},
        {"tunnel_id": 2, "points": [[-2, 0, 3], [2, 0, 3]]}]
    field = _field(graph, splines, {1: 1., 2: 1.})
    hit = field.ray_exit_hits(np.array([[0., 0., 0.]]), np.array([[0., 0., 1.]]))[0]
    assert hit is not None
    assert hit.distance_m == pytest.approx(1., abs=0.012)
    assert hit.source_primitive_ids == ("primitive:lower",)


def test_equal_operands_are_explicitly_ambiguous_not_arbitrarily_assigned():
    graph = {"coordinate_frame": "world", "nodes": [
        {"id": "a", "xyz": [-2, 0, 0], "degree": 2}, {"id": "b", "xyz": [2, 0, 0], "degree": 2}],
        "edges": [
            {"id": "one", "node_ids": ["a", "b"], "tunnel_ids": [1]},
            {"id": "two", "node_ids": ["a", "b"], "tunnel_ids": [2]}]}
    splines = [
        {"tunnel_id": 1, "points": [[-2, 0, 0], [2, 0, 0]]},
        {"tunnel_id": 2, "points": [[-2, 0, 0], [2, 0, 0]]}]
    field = _field(graph, splines, {1: 1., 2: 1.})
    hit = field.ray_exit_hits(np.array([[0., 0., 0.]]), np.array([[0., 1., 0.]]))[0]
    assert hit is not None and not hit.provenance_unique
    assert hit.source_primitive_ids == ("primitive:one", "primitive:two")


def test_ray_origin_outside_union_fails_closed():
    graph = {"coordinate_frame": "world", "nodes": [
        {"id": "a", "xyz": [-1, 0, 0], "degree": 1}, {"id": "b", "xyz": [1, 0, 0], "degree": 1}],
        "edges": [{"id": "e", "node_ids": ["a", "b"], "tunnel_ids": [1]}]}
    field = _field(graph, [{"tunnel_id": 1, "points": [[-1, 0, 0], [1, 0, 0]]}], {1: .5})
    with pytest.raises(ValueError, match="start inside"):
        field.ray_exit_hits(np.array([[0., 2., 0.]]), np.array([[1., 0., 0.]]))
