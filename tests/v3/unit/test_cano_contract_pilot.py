from __future__ import annotations

import numpy as np

from mtare_topo.data.cano_contract_pilot import (
    PARENT_REGISTRY,
    anchor_selection_audit,
    graph_family_metrics,
    match_headings,
    select_canonical_anchors,
)


def _single_turn_fixture() -> tuple[dict, dict]:
    points = np.asarray(
        [(-60.0, 0.0, 0.0), (-20.0, 0.0, 0.0), (20.0, 20.0, 0.0), (60.0, 60.0, 0.0)]
    )
    graph = {
        "nodes": [
            {"id": "a", "xyz": points[0].tolist(), "degree": 1, "incident_tunnel_ids": [1]},
            {"id": "b", "xyz": points[-1].tolist(), "degree": 1, "incident_tunnel_ids": [1]},
        ],
        "statistics": {"connected_components": 1, "cycle_rank": 0},
    }
    splines = {
        "tunnels": [
            {
                "tunnel_id": 1,
                "points": points.tolist(),
            }
        ]
    }
    return graph, splines


def test_registry_has_five_distinct_seed_separated_parents() -> None:
    assert len(PARENT_REGISTRY) == 5
    assert len({item["parent_id"] for item in PARENT_REGISTRY}) == 5
    assert len({item["topology_seed"] for item in PARENT_REGISTRY}) == 5
    assert len({item["geometry_seed"] for item in PARENT_REGISTRY}) == 5


def test_straight_turn_family_contract_accepts_turn_fixture() -> None:
    graph, splines = _single_turn_fixture()
    metrics = graph_family_metrics("P01_straight_turn", graph, splines)
    assert metrics["passed"]
    assert metrics["terminal_count"] == 2
    assert metrics["cumulative_horizontal_heading_change_rad"] > 0.2


def test_anchor_selection_is_deterministic_and_spaced() -> None:
    points = np.column_stack((np.linspace(-60.0, 60.0, 241), np.zeros(241), np.zeros(241)))
    graph = {
        "nodes": [
            {"id": "a", "xyz": points[0].tolist(), "degree": 1, "incident_tunnel_ids": [1]},
            {"id": "b", "xyz": points[-1].tolist(), "degree": 1, "incident_tunnel_ids": [1]},
        ],
        "statistics": {"connected_components": 1, "cycle_rank": 0},
    }
    splines = {"tunnels": [{"tunnel_id": 1, "points": points.tolist()}]}
    first = select_canonical_anchors(graph, splines, -1.5, count=10)
    second = select_canonical_anchors(graph, splines, -1.5, count=10)
    assert first == second
    points = np.asarray([item["axis_xyz_m"] for item in first])
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    distances += np.eye(len(points)) * 1e6
    assert float(np.min(distances)) >= 5.0 - 1e-6
    assert all(item["objective_structure"]["topological_role"] in {"terminal", "corridor"} for item in first)


def test_anchor_selection_reaches_target_and_balances_four_sparse_arms() -> None:
    center = np.zeros(3)
    endpoints = {
        1: np.asarray((82.0, 0.0, 0.0)),
        2: np.asarray((-82.0, 0.0, 0.0)),
        3: np.asarray((0.0, 82.0, 0.0)),
        4: np.asarray((0.0, -82.0, 0.0)),
    }
    graph = {
        "nodes": [
            {
                "id": "center",
                "xyz": center.tolist(),
                "degree": 4,
                "incident_tunnel_ids": [1, 2, 3, 4],
            },
            *[
                {
                    "id": f"terminal_{tunnel_id}",
                    "xyz": endpoint.tolist(),
                    "degree": 1,
                    "incident_tunnel_ids": [tunnel_id],
                }
                for tunnel_id, endpoint in endpoints.items()
            ],
        ],
        "statistics": {"connected_components": 1, "cycle_rank": 0},
    }
    splines = {
        "tunnels": [
            {
                "tunnel_id": tunnel_id,
                "points": [center.tolist(), (endpoint / 2.0).tolist(), endpoint.tolist()],
            }
            for tunnel_id, endpoint in endpoints.items()
        ]
    }

    anchors = select_canonical_anchors(graph, splines, None, count=50)
    audit = anchor_selection_audit(graph, splines, anchors, requested_count=50)

    assert audit["passed"]
    assert audit["selected_anchor_count"] == 50
    assert audit["minimum_pairwise_distance_m"] >= 5.0 - 1e-6
    assert audit["maximum_structural_event_to_anchor_distance_m"] <= 5.0 + 1e-6
    assert audit["all_tunnels_covered"]
    assert audit["fill_quotas_match"]
    assert audit["maximum_same_tunnel_or_shared_event_coverage_radius_m"] <= 7.5
    assert all("sensor_xyz_m" not in anchor for anchor in anchors)


def test_anchor_selection_allocates_more_fill_to_longer_tunnel() -> None:
    center = np.zeros(3)
    short_end = np.asarray((60.0, 0.0, 0.0))
    long_end = np.asarray((0.0, 180.0, 0.0))
    graph = {
        "nodes": [
            {
                "id": "center",
                "xyz": center.tolist(),
                "degree": 2,
                "incident_tunnel_ids": [1, 2],
            },
            {"id": "short", "xyz": short_end.tolist(), "degree": 1, "incident_tunnel_ids": [1]},
            {"id": "long", "xyz": long_end.tolist(), "degree": 1, "incident_tunnel_ids": [2]},
        ],
        "statistics": {"connected_components": 1, "cycle_rank": 0},
    }
    splines = {
        "tunnels": [
            {"tunnel_id": 1, "points": [center.tolist(), short_end.tolist()]},
            {"tunnel_id": 2, "points": [center.tolist(), long_end.tolist()]},
        ]
    }

    anchors = select_canonical_anchors(graph, splines, None, count=32)
    audit = anchor_selection_audit(graph, splines, anchors, requested_count=32)

    assert audit["passed"]
    assert audit["observed_fill_quota_per_tunnel"]["2"] > audit["observed_fill_quota_per_tunnel"]["1"]
    assert audit["maximum_same_tunnel_or_shared_event_coverage_radius_m"] <= 7.5


def test_heading_match_is_one_to_one_and_circular() -> None:
    result = match_headings([359.0, 91.0, 210.0], [1.0, 90.0])
    assert result["matched"] == 2
    assert result["predicted"] == 3
    assert result["truth"] == 2
    assert sorted(result["angular_errors_deg"]) == [1.0, 2.0]
