from __future__ import annotations

import copy

import pytest

from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r2 import (
    audit_frontier_attempt_evidence_v1r2,
)


def _target(mode, node, stub, path):
    return {
        "status": "TARGET",
        "reason": "best_deterministic_rule_based_frontier",
        "mode": mode,
        "frontier": {"node_id": node, "stub_index": stub},
        "frontier_node_id": node,
        "next_hop_node_id": path[1] if len(path) > 1 else node,
        "graph_path_node_ids": path,
    }


def _row(frame, arc, node, reason, target):
    return {
        "frame_index": frame,
        "route_arc_m": arc,
        "graph_update": {
            "frame_index": frame,
            "route_arc_m": arc,
            "node_id": node,
            "reason": reason,
        },
        "target": target,
    }


def _fixture():
    graph = {
        "schema_version": "causal_topometric_graph_v2",
        "current_node": 0,
        "nodes": [
            {"id": 0, "xyz_m": [0.0, 0.0, 0.0], "exit_stubs": [
                {"heading_world_deg": 0.0, "state": "traversed"},
                {"heading_world_deg": 90.0, "state": "observed"},
            ]},
            {"id": 1, "xyz_m": [10.0, 0.0, 0.0], "exit_stubs": [
                {"heading_world_deg": 180.0, "state": "traversed"},
            ]},
        ],
        "edges": [{
            "id": 0, "from": 0, "to": 1, "kind": "verified_traversed",
            "traversals": [
                {"from": 0, "to": 1, "start_frame": 0, "end_frame": 1,
                 "start_route_arc_m": 0.0, "end_route_arc_m": 8.0,
                 "trace_frame_count": 2},
                {"from": 1, "to": 0, "start_frame": 1, "end_frame": 2,
                 "start_route_arc_m": 8.0, "end_route_arc_m": 16.0,
                 "trace_frame_count": 2},
            ],
        }],
    }
    rows = [
        _row(0, 0.0, 0, "route_start", _target("frontier_exit", 0, 0, [0])),
        _row(1, 8.0, 1, "distance_anchor", _target("graph_backtrack", 0, 1, [1, 0])),
        _row(2, 16.0, 0, "distance_anchor", _target("frontier_exit", 0, 1, [0])),
        _row(3, 24.0, 0, "loop_merge", _target("frontier_exit", 0, 1, [0])),
    ]
    return rows, graph


def test_v1r2_validates_complete_traversal_and_source_planner_contract():
    rows, graph = _fixture()
    value = audit_frontier_attempt_evidence_v1r2(rows, graph)
    assert value["verified_traversal_integrity_count"] == 2
    assert value["frontier_target_contract_frame_count"] == 4
    assert value["historical_target_stub_state_validated_by_source_planner_contract"] is True
    assert value["historical_stub_creation_frame_directly_reconstructable"] is False


@pytest.mark.parametrize("mutation", ["start_frame", "start_arc", "trace_count"])
def test_v1r2_rejects_incomplete_traversal_evidence(mutation):
    rows, graph = _fixture()
    traversal = graph["edges"][0]["traversals"][0]
    if mutation == "start_frame":
        traversal["start_frame"] = traversal["end_frame"]
    elif mutation == "start_arc":
        traversal["start_route_arc_m"] = traversal["end_route_arc_m"]
    else:
        traversal["trace_frame_count"] = 1
    with pytest.raises(ValueError, match="historical trace evidence"):
        audit_frontier_attempt_evidence_v1r2(rows, graph)


def test_v1r2_rejects_final_current_path_loop_and_planner_contract():
    rows, graph = _fixture()
    graph["current_node"] = 1
    with pytest.raises(ValueError, match="final graph current node"):
        audit_frontier_attempt_evidence_v1r2(rows, graph)
    rows, graph = _fixture()
    rows[1]["target"]["graph_path_node_ids"] = [1, 0, 1, 0]
    rows[1]["target"]["next_hop_node_id"] = 0
    with pytest.raises(ValueError, match="repeat a node"):
        audit_frontier_attempt_evidence_v1r2(rows, graph)
    rows, graph = _fixture()
    rows[0]["target"]["reason"] = "forged"
    with pytest.raises(ValueError, match="planner output contract"):
        audit_frontier_attempt_evidence_v1r2(rows, graph)
