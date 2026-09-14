"""Read-only evidence for repeated frontier selection and exit-stub growth."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence


def _frontier_key(target: Mapping[str, Any]) -> tuple[int, int] | None:
    frontier = target.get("frontier")
    if not isinstance(frontier, Mapping):
        return None
    return int(frontier["node_id"]), int(frontier["stub_index"])


def audit_frontier_lifecycle_evidence(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure lifecycle persistence without declaring a tuned failure threshold."""

    if not decision_rows:
        raise ValueError("frontier lifecycle evidence requires decision rows")
    modes: Counter[str] = Counter()
    graph_reasons: Counter[str] = Counter()
    frontier_frames: Counter[tuple[int, int]] = Counter()
    longest_by_frontier: Counter[tuple[int, int]] = Counter()
    current_key: tuple[int, int] | None = None
    current_run = 0
    for expected_index, row in enumerate(decision_rows):
        if int(row.get("frame_index", -1)) != expected_index:
            raise ValueError("decision frames must be contiguous and zero-based")
        arc = float(row.get("route_arc_m", math.nan))
        if not math.isfinite(arc):
            raise ValueError("route arc must be finite")
        target = row.get("target")
        update = row.get("graph_update")
        if not isinstance(target, Mapping) or not isinstance(update, Mapping):
            raise ValueError("decision target and graph update must be mappings")
        mode = str(target.get("mode"))
        reason = str(update.get("reason"))
        modes[mode] += 1
        graph_reasons[reason] += 1
        key = _frontier_key(target)
        if key is None:
            current_key = None
            current_run = 0
            continue
        frontier_frames[key] += 1
        current_run = current_run + 1 if key == current_key else 1
        current_key = key
        longest_by_frontier[key] = max(longest_by_frontier[key], current_run)

    nodes = graph_snapshot.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("frontier lifecycle graph requires nonempty nodes")
    node_rows = []
    for node in nodes:
        node_id = int(node["id"])
        branch_count = int(node["branch_count"])
        stubs = node.get("exit_stubs")
        if branch_count < 0 or not isinstance(stubs, list):
            raise ValueError("graph node branch/stub evidence is invalid")
        states = Counter(str(stub.get("state")) for stub in stubs)
        if set(states) - {"observed", "traversed"}:
            raise ValueError("graph node has an invalid exit-stub state")
        node_rows.append({
            "node_id": node_id,
            "branch_count": branch_count,
            "exit_stub_count": len(stubs),
            "observed_exit_stub_count": states["observed"],
            "traversed_exit_stub_count": states["traversed"],
            "exit_stub_minus_branch_count": len(stubs) - branch_count,
        })
    frame_count = len(decision_rows)
    most_selected = frontier_frames.most_common(1)
    longest = max(longest_by_frontier.values(), default=0)
    return {
        "schema_version": "topology_frontier_lifecycle_evidence_v1",
        "frame_count": frame_count,
        "target_mode_counts": dict(sorted(modes.items())),
        "graph_update_reason_counts": dict(sorted(graph_reasons.items())),
        "frontier_target_frame_count": sum(frontier_frames.values()),
        "unique_frontier_target_count": len(frontier_frames),
        "most_selected_frontier": (
            None
            if not most_selected
            else {
                "node_id": most_selected[0][0][0],
                "stub_index": most_selected[0][0][1],
                "frame_count": most_selected[0][1],
                "frame_fraction": most_selected[0][1] / frame_count,
            }
        ),
        "longest_constant_frontier_run_frames": longest,
        "frontier_frame_counts": [
            {"node_id": key[0], "stub_index": key[1], "frame_count": count}
            for key, count in sorted(frontier_frames.items())
        ],
        "frontier_longest_run_frames": [
            {"node_id": key[0], "stub_index": key[1], "frame_count": count}
            for key, count in sorted(longest_by_frontier.items())
        ],
        "final_node_count": len(node_rows),
        "final_exit_stub_count": sum(row["exit_stub_count"] for row in node_rows),
        "final_observed_exit_stub_count": sum(
            row["observed_exit_stub_count"] for row in node_rows
        ),
        "maximum_node_exit_stub_count": max(row["exit_stub_count"] for row in node_rows),
        "maximum_node_exit_stub_minus_branch_count": max(
            row["exit_stub_minus_branch_count"] for row in node_rows
        ),
        "nodes": node_rows,
        "uses_evaluator_gt": False,
    }


__all__ = ["audit_frontier_lifecycle_evidence"]
