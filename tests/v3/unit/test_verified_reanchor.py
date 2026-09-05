from __future__ import annotations

from copy import deepcopy

import pytest

from mtare_topo.topology.verified_reanchor import evaluate_verified_reanchor


def graph():
    return {
        "config": {"loop_merge_radius_m": 4.0},
        "current_node": 1,
        "nodes": [
            {"id": 0, "xyz_m": [0.0, 0.0, 0.0]},
            {"id": 1, "xyz_m": [8.0, 0.0, 0.0]},
        ],
        "edges": [{
            "from": 0,
            "to": 1,
            "kind": "verified_traversed",
            "traversals": [{"length_m": 8.2, "trace_frame_count": 24}],
        }],
    }


def target():
    return {
        "mode": "graph_backtrack",
        "next_hop_node_id": 0,
        "graph_path_node_ids": [1, 0],
    }


def evaluate(snapshot=None, active=None, **kwargs):
    return evaluate_verified_reanchor(
        graph() if snapshot is None else snapshot,
        target() if active is None else active,
        sensor_xyz_m=kwargs.pop("sensor_xyz_m", [1.0, 0.0, 0.0]),
        route_arc_m=kwargs.pop("route_arc_m", 20.0),
        current_node_route_arc_m=kwargs.pop("current_node_route_arc_m", 8.0),
        trace_frame_count=kwargs.pop("trace_frame_count", 30),
        **kwargs,
    )


def test_verified_backtrack_arrival_is_eligible_and_deterministic() -> None:
    first = evaluate()
    second = evaluate()
    assert first == second
    assert first.eligible
    assert first.reason == "verified_backtrack_arrival"
    assert first.from_node_id == 1
    assert first.next_hop_node_id == 0
    assert first.distance_to_current_m == pytest.approx(7.0)
    assert first.distance_to_next_hop_m == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["frontier_exit", "hold", "CANDIDATE_COMPLETE"])
def test_non_backtrack_modes_fail_closed(mode) -> None:
    result = evaluate(active={"mode": mode, "next_hop_node_id": 0, "graph_path_node_ids": [1, 0]})
    assert not result.eligible
    assert result.reason == "active_target_is_not_graph_backtrack"


def test_topology_and_trace_guards_fail_closed() -> None:
    no_edge = graph(); no_edge["edges"] = []
    assert evaluate(snapshot=no_edge).reason == "next_hop_has_no_verified_edge"
    assert evaluate(trace_frame_count=1).reason == "physical_trace_is_not_positive"
    assert evaluate(route_arc_m=8.0).reason == "physical_trace_is_not_positive"
    assert evaluate(active={"mode": "graph_backtrack", "next_hop_node_id": 1, "graph_path_node_ids": [1, 1]}).reason == "next_hop_is_already_current"


@pytest.mark.parametrize(
    "active",
    [
        {"mode": "graph_backtrack", "next_hop_node_id": 0},
        {"mode": "graph_backtrack", "next_hop_node_id": 0, "graph_path_node_ids": [0, 1]},
        {"mode": "graph_backtrack", "next_hop_node_id": 0, "graph_path_node_ids": [1, 7]},
    ],
)
def test_stale_active_target_path_fails_closed(active) -> None:
    result = evaluate(active=active)
    assert not result.eligible
    assert result.reason == "active_target_path_is_stale"


def test_geometry_guards_require_next_hop_radius_and_strictly_closer_pose() -> None:
    outside = evaluate(sensor_xyz_m=[4.5, 0.0, 0.0])
    assert outside.reason == "pose_is_outside_next_hop_radius"
    equal = evaluate(sensor_xyz_m=[4.0, 0.0, 0.0])
    assert equal.reason == "pose_is_not_closer_to_next_hop"
    overlapping = graph()
    overlapping["nodes"][1]["xyz_m"] = [6.0, 0.0, 0.0]
    current_side = evaluate(snapshot=overlapping, sensor_xyz_m=[3.5, 0.0, 0.0])
    assert current_side.reason == "pose_is_not_closer_to_next_hop"


def test_invalid_verified_evidence_and_nonfinite_geometry_raise() -> None:
    invalid = graph(); invalid["edges"][0]["traversals"] = []
    with pytest.raises(ValueError, match="no traversal evidence"):
        evaluate(snapshot=invalid)
    invalid = graph(); invalid["edges"][0]["traversals"][0]["length_m"] = 0.0
    with pytest.raises(ValueError, match="evidence is invalid"):
        evaluate(snapshot=invalid)
    with pytest.raises(ValueError, match="sensor_xyz_m must be finite"):
        evaluate(sensor_xyz_m=[float("nan"), 0.0, 0.0])


def test_evaluator_metadata_cannot_change_decision() -> None:
    first_graph = graph()
    second_graph = deepcopy(first_graph)
    first_graph["edges"][0]["traversals"][0]["evaluator_gt_edge_ids"] = ["secret_a"]
    second_graph["edges"][0]["traversals"][0]["evaluator_gt_edge_ids"] = ["secret_b", "secret_c"]
    assert evaluate(snapshot=first_graph) == evaluate(snapshot=second_graph)
