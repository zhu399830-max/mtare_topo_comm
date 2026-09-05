"""Read-only causal evidence for frontier-exit execution outcomes."""

from __future__ import annotations

from collections import Counter
import math
import sys
from typing import Any, Mapping, Sequence

from mtare_topo.topology.causal_graph_v2 import circular_distance_deg, wrap_deg


def _frontier_key(target: Mapping[str, Any]) -> tuple[int, int] | None:
    frontier = target.get("frontier")
    if not isinstance(frontier, Mapping):
        return None
    return int(frontier["node_id"]), int(frontier["stub_index"])


def _departure_stub_index(
    source: Mapping[str, Any], destination: Mapping[str, Any]
) -> int:
    origin = tuple(float(value) for value in source["xyz_m"])
    target = tuple(float(value) for value in destination["xyz_m"])
    if len(origin) != 3 or len(target) != 3:
        raise ValueError("frontier attempt nodes require xyz coordinates")
    if not all(math.isfinite(value) for value in (*origin, *target)):
        raise ValueError("frontier attempt node coordinates must be finite")
    delta_x, delta_y = target[0] - origin[0], target[1] - origin[1]
    # Keep the runtime and offline audit bit-identical on frozen Python 3.8.
    if math.hypot(delta_x, delta_y) <= sys.float_info.epsilon:
        raise ValueError("frontier attempt transition nodes must be spatially distinct")
    stubs = source.get("exit_stubs")
    if not isinstance(stubs, list) or not stubs:
        raise ValueError("frontier attempt source node requires exit stubs")
    departure_heading = wrap_deg(math.degrees(math.atan2(delta_y, delta_x)))
    headings = [float(stub["heading_world_deg"]) for stub in stubs]
    if not all(math.isfinite(value) for value in headings):
        raise ValueError("frontier attempt stub headings must be finite")
    return min(
        range(len(headings)),
        key=lambda index: (circular_distance_deg(departure_heading, headings[index]), index),
    )


def audit_frontier_attempt_evidence(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify graph events caused by the preceding current-node frontier target.

    A transition to another node is compared with the exact exit stub that the
    graph would consume for that verified traversal.  A loop merge back to the
    same current node is kept separate.  The audit declares no tuned failure
    threshold and reads no evaluator or map ground truth.
    """

    if not decision_rows:
        raise ValueError("frontier attempt evidence requires decision rows")
    raw_nodes = graph_snapshot.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("frontier attempt graph requires nonempty nodes")
    nodes = {int(node["id"]): node for node in raw_nodes}
    if len(nodes) != len(raw_nodes):
        raise ValueError("frontier attempt graph node IDs must be unique")

    outcomes: Counter[str] = Counter()
    by_frontier: Counter[tuple[int, int, str]] = Counter()
    event_rows: list[dict[str, Any]] = []
    target_run_starts: list[int] = []
    previous_target_key: tuple[str, tuple[int, int] | None, int] | None = None
    previous_arc: float | None = None
    for expected_index, row in enumerate(decision_rows):
        if int(row.get("frame_index", -1)) != expected_index:
            raise ValueError("decision frames must be contiguous and zero-based")
        arc = float(row.get("route_arc_m", math.nan))
        if not math.isfinite(arc) or (previous_arc is not None and arc < previous_arc):
            raise ValueError("route arc must be finite and nondecreasing")
        previous_arc = arc
        target = row.get("target")
        update = row.get("graph_update")
        if not isinstance(target, Mapping) or not isinstance(update, Mapping):
            raise ValueError("decision target and graph update must be mappings")
        target_key = (
            str(target.get("mode")),
            _frontier_key(target),
            int(update["node_id"]),
        )
        target_run_starts.append(
            target_run_starts[-1]
            if target_key == previous_target_key and target_run_starts
            else expected_index
        )
        previous_target_key = target_key

    for frame_index in range(1, len(decision_rows)):
        previous = decision_rows[frame_index - 1]
        current = decision_rows[frame_index]
        previous_target = previous["target"]
        if previous_target.get("mode") != "frontier_exit":
            continue
        frontier = _frontier_key(previous_target)
        if frontier is None:
            raise ValueError("frontier_exit target requires a frontier identity")
        previous_node = int(previous["graph_update"]["node_id"])
        if frontier[0] != previous_node:
            raise ValueError("frontier_exit target must belong to the current node")
        update = current["graph_update"]
        current_node = int(update["node_id"])
        reason = str(update.get("reason"))
        if previous_node not in nodes or current_node not in nodes:
            raise ValueError("decision graph update references an unknown final graph node")

        if current_node == previous_node:
            if reason != "loop_merge":
                continue
            outcome = "same_node_loop_merge"
            actual_stub = None
        else:
            actual_stub = _departure_stub_index(nodes[previous_node], nodes[current_node])
            outcome = (
                "matched_verified_departure"
                if actual_stub == frontier[1]
                else "divergent_verified_departure"
            )
        outcomes[outcome] += 1
        by_frontier[(frontier[0], frontier[1], outcome)] += 1
        run_start_frame = target_run_starts[frame_index - 1]
        run_route_arc_m = (
            float(current["route_arc_m"])
            - float(decision_rows[run_start_frame]["route_arc_m"])
        )
        event_rows.append({
            "frame_index": frame_index,
            "route_arc_m": float(current["route_arc_m"]),
            "source_node_id": previous_node,
            "destination_node_id": current_node,
            "target_stub_index": frontier[1],
            "actual_departure_stub_index": actual_stub,
            "target_run_start_frame": run_start_frame,
            "target_run_frames_before_event": frame_index - run_start_frame,
            "target_run_route_arc_m_before_event": run_route_arc_m,
            "outcome": outcome,
            "graph_update_reason": reason,
        })

    classified = sum(outcomes.values())
    nonmatching = (
        outcomes["divergent_verified_departure"]
        + outcomes["same_node_loop_merge"]
    )
    return {
        "schema_version": "topology_frontier_attempt_evidence_v1",
        "frame_count": len(decision_rows),
        "classified_attempt_event_count": classified,
        "outcome_counts": dict(sorted(outcomes.items())),
        "nonmatching_attempt_event_count": nonmatching,
        "nonmatching_attempt_event_fraction": (
            0.0 if classified == 0 else nonmatching / classified
        ),
        "maximum_target_run_frames_before_event": max(
            (row["target_run_frames_before_event"] for row in event_rows),
            default=0,
        ),
        "maximum_target_run_route_arc_m_before_event": max(
            (row["target_run_route_arc_m_before_event"] for row in event_rows),
            default=0.0,
        ),
        "frontier_outcome_counts": [
            {
                "node_id": node_id,
                "stub_index": stub_index,
                "outcome": outcome,
                "event_count": count,
            }
            for (node_id, stub_index, outcome), count in sorted(by_frontier.items())
        ],
        "events": event_rows,
        "uses_evaluator_gt": False,
    }


__all__ = ["audit_frontier_attempt_evidence"]
