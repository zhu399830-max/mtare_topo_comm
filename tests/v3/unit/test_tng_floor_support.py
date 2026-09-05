import numpy as np
import pytest

from mtare_topo.data.tng_floor_support import build_tng_floor_support_geometry


def _geometry():
    return {
        "fta_distance_m": -1.0,
        "tunnels": [
            {"tunnel_id": 1, "radius_m": 5.0},
            {"tunnel_id": 2, "radius_m": 5.0},
        ],
    }


def test_floor_geometry_is_byte_deterministic_and_trajectory_free():
    graph = {
        "nodes": [
            {"id": "n0", "xyz": [0, 0, 0], "incident_tunnel_ids": [1]},
            {"id": "n1", "xyz": [10, 0, 0], "incident_tunnel_ids": [1]},
            {"id": "n2", "xyz": [0, 0, 20], "incident_tunnel_ids": [2]},
            {"id": "n3", "xyz": [10, 0, 20], "incident_tunnel_ids": [2]},
        ]
    }
    splines = {
        "tunnels": [
            {"tunnel_id": 1, "points": [[0, 0, 0], [5, 0, 0], [10, 0, 0]]},
            {"tunnel_id": 2, "points": [[0, 0, 20], [5, 0, 20], [10, 0, 20]]},
        ]
    }
    first = build_tng_floor_support_geometry(graph, splines, _geometry())
    second = build_tng_floor_support_geometry(graph, splines, _geometry())
    assert first.obj_bytes() == second.obj_bytes()
    assert first.sha256() == second.sha256()
    assert set(np.unique(first.vertices[:, 2])) == {-1.0, 19.0}
    # No triangle is allowed to bridge the vertically stacked non-incident tunnels.
    assert np.max(np.ptp(first.vertices[first.triangles, 2], axis=1)) == 0.0


def test_only_explicit_incident_tunnel_gets_a_node_connector():
    graph = {
        "nodes": [
            {"id": "branch", "xyz": [0, 0.2, 0], "incident_tunnel_ids": [1]},
            {"id": "end1", "xyz": [10, 0, 0], "incident_tunnel_ids": [1]},
            {"id": "end2a", "xyz": [0, 0, 10], "incident_tunnel_ids": [2]},
            {"id": "end2b", "xyz": [10, 0, 10], "incident_tunnel_ids": [2]},
        ]
    }
    splines = {
        "tunnels": [
            {"tunnel_id": 1, "points": [[0, 0, 0], [10, 0, 0]]},
            {"tunnel_id": 2, "points": [[0, 0, 10], [10, 0, 10]]},
        ]
    }
    result = build_tng_floor_support_geometry(graph, splines, _geometry())
    assert result.connector_count == 1
    assert any(group == "connector_branch_tunnel_1" for group in result.face_groups)
    assert not any("connector_branch_tunnel_2" == group for group in result.face_groups)


def test_rejects_connector_beyond_frozen_limit():
    graph = {
        "nodes": [
            {"id": "bad", "xyz": [0, 1, 0], "incident_tunnel_ids": [1]},
            {"id": "a", "xyz": [10, 0, 0], "incident_tunnel_ids": [1]},
            {"id": "b", "xyz": [0, 0, 10], "incident_tunnel_ids": [2]},
            {"id": "c", "xyz": [10, 0, 10], "incident_tunnel_ids": [2]},
        ]
    }
    splines = {
        "tunnels": [
            {"tunnel_id": 1, "points": [[0, 0, 0], [10, 0, 0]]},
            {"tunnel_id": 2, "points": [[0, 0, 10], [10, 0, 10]]},
        ]
    }
    with pytest.raises(ValueError, match="connector is"):
        build_tng_floor_support_geometry(graph, splines, _geometry())


def test_rejects_floor_cross_section_narrower_than_contract():
    graph = {"nodes": []}
    splines = {"tunnels": [{"tunnel_id": 1, "points": [[0, 0, 0], [1, 0, 0]]}]}
    geometry = {"fta_distance_m": -1.0, "tunnels": [{"tunnel_id": 1, "radius_m": 1.01}]}
    with pytest.raises(ValueError, match="below connector width"):
        build_tng_floor_support_geometry(graph, splines, geometry)
