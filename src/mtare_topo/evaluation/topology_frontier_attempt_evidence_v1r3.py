"""V1R3 exhaustive target and traversal-to-event validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r2 import (
    audit_frontier_attempt_evidence_v1r2,
)


TARGET_KEYS = {
    "status", "reason", "mode", "frontier", "frontier_node_id",
    "next_hop_node_id", "graph_path_node_ids", "waypoint_xyz_m",
    "exploration_potential", "graph_cost_m", "utility",
}
TARGET_MODES = {"frontier_exit", "graph_backtrack"}


def _validate_exact_target(target: Mapping[str, Any]) -> None:
    if set(target) != TARGET_KEYS or target.get("mode") not in TARGET_MODES:
        raise ValueError("decision target does not match the exact planner target schema")
    frontier = target.get("frontier")
    path = target.get("graph_path_node_ids")
    waypoint = target.get("waypoint_xyz_m")
    if (
        not isinstance(frontier, Mapping)
        or set(frontier) != {"node_id", "stub_index"}
        or not isinstance(path, list)
        or not path
        or not isinstance(waypoint, list)
        or len(waypoint) != 3
        or not all(math.isfinite(float(value)) for value in waypoint)
        or not all(
            math.isfinite(float(target[key]))
            for key in ("exploration_potential", "graph_cost_m", "utility")
        )
    ):
        raise ValueError("decision target does not match the exact planner target schema")


def audit_frontier_attempt_evidence_v1r3(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact planner targets and a bijection between traversals and events."""

    base = audit_frontier_attempt_evidence_v1r2(decision_rows, graph_snapshot)
    for row in decision_rows:
        _validate_exact_target(row["target"])

    transitions = {
        (frame, int(decision_rows[frame - 1]["graph_update"]["node_id"]),
         int(decision_rows[frame]["graph_update"]["node_id"]))
        for frame in range(1, len(decision_rows))
        if int(decision_rows[frame - 1]["graph_update"]["node_id"])
        != int(decision_rows[frame]["graph_update"]["node_id"])
    }
    traversal_events: set[tuple[int, int, int]] = set()
    for edge in graph_snapshot["edges"]:
        for traversal in edge["traversals"]:
            first, second = int(traversal["from"]), int(traversal["to"])
            start_frame = int(traversal["start_frame"])
            end_frame = int(traversal["end_frame"])
            event = (end_frame, first, second)
            if (
                int(decision_rows[start_frame]["graph_update"]["node_id"]) != first
                or int(decision_rows[end_frame]["graph_update"]["node_id"]) != second
                or event in traversal_events
            ):
                raise ValueError("verified traversal is not uniquely bound to its trace endpoints")
            traversal_events.add(event)
    if traversal_events != transitions:
        raise ValueError("verified traversals and decision transition events are not bijective")

    result = dict(base)
    result.update({
        "schema_version": "topology_frontier_attempt_evidence_v1r3",
        "exact_planner_target_schema_frame_count": len(decision_rows),
        "verified_traversal_event_bijection_count": len(traversal_events),
        "verified_traversal_trace_endpoint_node_binding": True,
    })
    return result


__all__ = ["audit_frontier_attempt_evidence_v1r3"]
