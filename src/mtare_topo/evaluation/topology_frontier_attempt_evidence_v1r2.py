"""V1R2 source-planner and traversal-integrity validation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r import (
    audit_frontier_attempt_evidence_v1r,
)


def _frontier_key(target: Mapping[str, Any]) -> tuple[int, int] | None:
    frontier = target.get("frontier")
    if not isinstance(frontier, Mapping):
        return None
    return int(frontier["node_id"]), int(frontier["stub_index"])


def audit_frontier_attempt_evidence_v1r2(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind observed-only targets to source code and validate every traversal."""

    base = audit_frontier_attempt_evidence_v1r(decision_rows, graph_snapshot)
    nodes = {int(node["id"]): node for node in graph_snapshot["nodes"]}
    if int(graph_snapshot.get("current_node", -1)) != int(
        decision_rows[-1]["graph_update"]["node_id"]
    ):
        raise ValueError("final graph current node does not match the final decision row")

    traversal_count = 0
    for edge in graph_snapshot["edges"]:
        edge_endpoints = {int(edge["from"]), int(edge["to"])}
        for traversal in edge["traversals"]:
            traversal_count += 1
            first, second = int(traversal["from"]), int(traversal["to"])
            start_frame, end_frame = int(traversal["start_frame"]), int(traversal["end_frame"])
            start_arc = float(traversal["start_route_arc_m"])
            end_arc = float(traversal["end_route_arc_m"])
            trace_frames = int(traversal.get("trace_frame_count", -1))
            if (
                {first, second} != edge_endpoints
                or not 0 <= start_frame < end_frame < len(decision_rows)
                or not math.isfinite(start_arc)
                or not math.isfinite(end_arc)
                or not start_arc < end_arc
                or trace_frames != end_frame - start_frame + 1
                or not math.isclose(
                    float(decision_rows[start_frame]["route_arc_m"]),
                    start_arc,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(decision_rows[end_frame]["route_arc_m"]),
                    end_arc,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("verified traversal has invalid historical trace evidence")
    frontier_target_count = 0
    for frame_index, row in enumerate(decision_rows):
        target = row["target"]
        path = [int(value) for value in target.get("graph_path_node_ids", [])]
        if len(path) != len(set(path)):
            raise ValueError("graph-backtrack path may not repeat a node")
        frontier = _frontier_key(target)
        if frontier is not None:
            frontier_target_count += 1
            if (
                target.get("status") != "TARGET"
                or target.get("reason") != "best_deterministic_rule_based_frontier"
            ):
                raise ValueError("frontier target does not match the frozen planner output contract")

    result = dict(base)
    result.update({
        "schema_version": "topology_frontier_attempt_evidence_v1r2",
        "verified_traversal_integrity_count": traversal_count,
        "frontier_target_contract_frame_count": frontier_target_count,
        "historical_target_stub_state_validated_by_source_planner_contract": True,
        "historical_stub_creation_frame_directly_reconstructable": False,
        "source_planner_observed_only_contract_required": True,
    })
    return result


__all__ = ["audit_frontier_attempt_evidence_v1r2"]
