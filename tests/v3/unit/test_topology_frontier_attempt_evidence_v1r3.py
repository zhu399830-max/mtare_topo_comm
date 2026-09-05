from __future__ import annotations

import copy

import pytest

from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r3 import (
    audit_frontier_attempt_evidence_v1r3,
)
from test_topology_frontier_attempt_evidence_v1r2 import _fixture


def _fixture_v1r3():
    rows, graph = _fixture()
    for row in rows:
        row["target"].update({
            "waypoint_xyz_m": [0.0, 0.0, 0.0],
            "exploration_potential": 1.0,
            "graph_cost_m": 0.0,
            "utility": 1.0,
        })
    return rows, graph


def test_v1r3_requires_exact_target_schema_and_traversal_event_bijection():
    rows, graph = _fixture_v1r3()
    value = audit_frontier_attempt_evidence_v1r3(rows, graph)
    assert value["exact_planner_target_schema_frame_count"] == 4
    assert value["verified_traversal_event_bijection_count"] == 2
    assert value["verified_traversal_trace_endpoint_node_binding"] is True


@pytest.mark.parametrize("mutation", ["mode", "missing_frontier", "extra_key"])
def test_v1r3_rejects_malformed_target_schema(mutation):
    rows, graph = _fixture_v1r3()
    if mutation == "mode":
        rows[0]["target"]["mode"] = "forged"
    elif mutation == "missing_frontier":
        rows[0]["target"].pop("frontier")
    else:
        rows[0]["target"]["unexpected"] = True
    with pytest.raises((ValueError, KeyError), match="target|frontier"):
        audit_frontier_attempt_evidence_v1r3(rows, graph)


def test_v1r3_rejects_traversal_start_node_drift_and_unused_traversal():
    rows, graph = _fixture_v1r3()
    graph["edges"][0]["traversals"][1]["start_frame"] = 0
    graph["edges"][0]["traversals"][1]["start_route_arc_m"] = 0.0
    graph["edges"][0]["traversals"][1]["trace_frame_count"] = 3
    with pytest.raises(ValueError, match="trace endpoints"):
        audit_frontier_attempt_evidence_v1r3(rows, graph)

    rows, graph = _fixture_v1r3()
    extra = copy.deepcopy(graph["edges"][0]["traversals"][0])
    extra.update({"start_frame": 2, "end_frame": 3,
                  "start_route_arc_m": 16.0, "end_route_arc_m": 24.0})
    graph["edges"][0]["traversals"].append(extra)
    with pytest.raises(ValueError, match="trace endpoints|bijective"):
        audit_frontier_attempt_evidence_v1r3(rows, graph)
