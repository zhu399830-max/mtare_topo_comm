from pathlib import Path

import numpy as np
import pytest

from mtare_topo.teacher.primitive_construction_supervisor import (
    audit_archived_supervision,
    build_primitive_construction_graph,
    obj_has_face_provenance,
)


def _t_scene():
    graph = {
        "coordinate_frame": "world",
        "nodes": [
            {"id": "left", "xyz": [-2, 0, 0], "degree": 1},
            {"id": "joint", "xyz": [0, 0, 0], "degree": 3},
            {"id": "right", "xyz": [2, 0, 0], "degree": 1},
            {"id": "top", "xyz": [0, 2, 0], "degree": 1},
        ],
        "edges": [
            {"id": "e0", "node_ids": ["left", "joint"], "tunnel_ids": [10]},
            {"id": "e1", "node_ids": ["joint", "right"], "tunnel_ids": [10]},
            {"id": "e2", "node_ids": ["joint", "top"], "tunnel_ids": [20]},
        ],
    }
    splines = {
        "tunnels": [
            {"tunnel_id": 10, "points": [[-2, 0, 0], [0, 0, 0], [2, 0, 0]]},
            {"tunnel_id": 20, "points": [[0, 0, 0], [0, 2, 0]]},
        ]
    }
    geometry = {
        "tunnels": [
            {"tunnel_id": 10, "radius_m": 2.0},
            {"tunnel_id": 20, "radius_m": 1.5},
        ]
    }
    return graph, splines, geometry


def test_edge_segmentation_represents_interior_branch_without_tunnel_collapse():
    graph, splines, geometry = _t_scene()
    result = build_primitive_construction_graph(graph, splines, geometry)
    assert len(result.primitives) == 3
    assert [value.source_tunnel_id for value in result.primitives] == ["10", "10", "20"]
    assert [value.length_m for value in result.primitives] == pytest.approx([2.0, 2.0, 2.0])
    joint = next(value for value in result.compositions if value.node_id == "joint")
    assert joint.degree == 3
    assert {value.primitive_id for value in joint.member_endpoints} == {
        "primitive:e0",
        "primitive:e1",
        "primitive:e2",
    }


def test_stacked_disconnected_primitives_remain_disconnected():
    graph = {
        "coordinate_frame": "world",
        "nodes": [
            {"id": "a0", "xyz": [-1, 0, 0], "degree": 1},
            {"id": "a1", "xyz": [1, 0, 0], "degree": 1},
            {"id": "b0", "xyz": [0, -1, 3], "degree": 1},
            {"id": "b1", "xyz": [0, 1, 3], "degree": 1},
        ],
        "edges": [
            {"id": "lower", "node_ids": ["a0", "a1"], "tunnel_ids": [1]},
            {"id": "upper", "node_ids": ["b0", "b1"], "tunnel_ids": [2]},
        ],
    }
    splines = {"tunnels": [
        {"tunnel_id": 1, "points": [[-1, 0, 0], [1, 0, 0]]},
        {"tunnel_id": 2, "points": [[0, -1, 3], [0, 1, 3]]},
    ]}
    geometry = {"tunnels": [
        {"tunnel_id": 1, "radius_m": 1.0},
        {"tunnel_id": 2, "radius_m": 1.0},
    ]}
    result = build_primitive_construction_graph(graph, splines, geometry)
    assert len(result.primitives) == 2
    assert all(value.degree == 1 for value in result.compositions)


def test_projection_contract_fails_closed():
    graph, splines, geometry = _t_scene()
    graph["nodes"][0]["xyz"] = [-2, 1, 0]
    with pytest.raises(ValueError, match="projection exceeds contract"):
        build_primitive_construction_graph(graph, splines, geometry)


def test_free_space_overlap_attachment_preserves_axis_endpoint_and_graph_anchor():
    graph, splines, geometry = _t_scene()
    graph["nodes"][0]["xyz"] = [-2, .3, 0]
    result = build_primitive_construction_graph(
        graph, splines, geometry, endpoint_attachment_mode="free_space_overlap"
    )
    endpoint = result.primitives[0].endpoints[0]
    assert endpoint.xyz_m == pytest.approx((-2., 0., 0.))
    assert endpoint.composition_anchor_xyz_m == pytest.approx((-2., .3, 0.))
    composition = next(value for value in result.compositions if value.node_id == "left")
    assert composition.anchor_xyz_m == pytest.approx((-2., .3, 0.))
    assert result.endpoint_attachment_mode == "free_space_overlap"


def test_free_space_overlap_attachment_fails_when_anchor_is_outside_tunnel():
    graph, splines, geometry = _t_scene()
    graph["nodes"][0]["xyz"] = [-2, 2.1, 0]
    with pytest.raises(ValueError, match="outside declared tunnel free space"):
        build_primitive_construction_graph(
            graph, splines, geometry, endpoint_attachment_mode="free_space_overlap"
        )


def test_edge_incidence_degree_mode_records_stale_declared_degree_without_fabricating_edge():
    graph, splines, geometry = _t_scene()
    graph["nodes"][0]["degree"] = 2
    with pytest.raises(ValueError, match="differs from graph degree"):
        build_primitive_construction_graph(graph, splines, geometry)
    result = build_primitive_construction_graph(
        graph, splines, geometry, node_degree_source="edge_incidence"
    )
    assert len(result.primitives) == 3
    assert result.node_degree_source == "edge_incidence"
    assert [value.as_dict() for value in result.node_degree_mismatches] == [{
        "node_id": "left", "declared_degree": 2, "edge_incidence_degree": 1
    }]


def test_obj_face_provenance_detection(tmp_path: Path):
    plain = tmp_path / "plain.obj"
    plain.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    grouped = tmp_path / "grouped.obj"
    grouped.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\ng primitive_0\nf 1 2 3\n",
        encoding="utf-8",
    )
    assert not obj_has_face_provenance(plain)
    assert obj_has_face_provenance(grouped)


def test_archive_audit_requires_mesh_and_lidar_hit_identity(tmp_path: Path):
    graph, splines, geometry = _t_scene()
    construction = build_primitive_construction_graph(graph, splines, geometry)
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    audit = audit_archived_supervision(
        construction,
        graph,
        mesh_obj_path=mesh,
        lidar_array_names=("range_m", "valid_mask"),
    )
    assert audit.edge_level_primitives_unique
    assert audit.endpoint_compositions_complete
    assert not audit.mesh_triangle_provenance_present
    assert not audit.lidar_hit_provenance_present
    assert not audit.construction_supervision_complete
    assert len(audit.failure_reasons) == 2


def test_archive_audit_passes_when_both_provenance_channels_exist(tmp_path: Path):
    graph, splines, geometry = _t_scene()
    construction = build_primitive_construction_graph(graph, splines, geometry)
    mesh = tmp_path / "mesh.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\ng primitive:e0\nf 1 2 3\n",
        encoding="utf-8",
    )
    audit = audit_archived_supervision(
        construction,
        graph,
        mesh_obj_path=mesh,
        lidar_array_names=("range_m", "valid_mask", "primitive_hit_id"),
    )
    assert audit.construction_supervision_complete
    assert not audit.failure_reasons
