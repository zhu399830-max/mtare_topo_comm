"""Read-only evidence extraction for stale graph-backtrack node anchoring."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _frontier_key(target: Mapping[str, Any]) -> tuple[int, int] | None:
    frontier = target.get("frontier")
    if not isinstance(frontier, Mapping):
        return None
    return int(frontier["node_id"]), int(frontier["stub_index"])


def audit_reanchor_evidence(
    decision_rows: Sequence[Mapping[str, Any]],
    graph_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Quantify frames that prove arrival without graph-current re-anchoring.

    An exact next-hop waypoint is emitted by the frozen planner only after the
    robot lies within its existing lookahead distance of that node.  Combining
    this with a verified graph path and a different current node yields a
    conservative, replayable proxy for the missing re-anchor transition.
    """

    if not decision_rows:
        raise ValueError("re-anchor evidence requires decision rows")
    nodes = {int(node["id"]): tuple(float(v) for v in node["xyz_m"]) for node in graph_snapshot.get("nodes", [])}
    if len(nodes) != len(graph_snapshot.get("nodes", [])):
        raise ValueError("graph node IDs must be unique")
    if any(len(xyz) != 3 or not all(math.isfinite(value) for value in xyz) for xyz in nodes.values()):
        raise ValueError("graph node coordinates must be finite xyz")

    previous_arc: float | None = None
    last_arc_growth_frame = 0
    graph_backtrack_frames = 0
    exact_arrival_proxy_frames = 0
    longest_run = 0
    current_run = 0
    previous_key: tuple[int, int, int, int] | None = None
    first_proxy_frame: int | None = None
    last_proxy_frame: int | None = None
    first_proxy_arc: float | None = None
    last_proxy_arc: float | None = None
    proxy_arc_growth_frames = 0

    for expected_index, row in enumerate(decision_rows):
        frame = int(row.get("frame_index", -1))
        if frame != expected_index:
            raise ValueError("decision frames must be contiguous and zero-based")
        arc = float(row.get("route_arc_m", math.nan))
        if not math.isfinite(arc) or (previous_arc is not None and arc < previous_arc):
            raise ValueError("route arc must be finite and nondecreasing")
        arc_grew = previous_arc is None or arc > previous_arc
        if arc_grew:
            last_arc_growth_frame = frame
        previous_arc = arc

        target = row.get("target", {})
        if target.get("mode") != "graph_backtrack":
            current_run = 0
            previous_key = None
            continue
        graph_backtrack_frames += 1
        current_node = int(row.get("graph_update", {}).get("node_id", -1))
        next_raw = target.get("next_hop_node_id")
        path = tuple(int(value) for value in target.get("graph_path_node_ids", []))
        waypoint = target.get("waypoint_xyz_m")
        frontier = _frontier_key(target)
        if next_raw is None or waypoint is None or frontier is None:
            current_run = 0
            previous_key = None
            continue
        next_hop = int(next_raw)
        waypoint_xyz = tuple(float(value) for value in waypoint)
        path_proves_next_hop = len(path) >= 2 and path[0] == current_node and path[1] == next_hop
        exact_node_waypoint = next_hop in nodes and waypoint_xyz == nodes[next_hop]
        is_proxy = current_node != next_hop and path_proves_next_hop and exact_node_waypoint
        if not is_proxy:
            current_run = 0
            previous_key = None
            continue
        exact_arrival_proxy_frames += 1
        first_proxy_frame = frame if first_proxy_frame is None else first_proxy_frame
        last_proxy_frame = frame
        first_proxy_arc = arc if first_proxy_arc is None else first_proxy_arc
        last_proxy_arc = arc
        proxy_arc_growth_frames += int(arc_grew)
        key = (current_node, next_hop, frontier[0], frontier[1])
        current_run = current_run + 1 if key == previous_key else 1
        previous_key = key
        longest_run = max(longest_run, current_run)

    final_frame = len(decision_rows) - 1
    return {
        "schema_version": "topology_reanchor_evidence_v1",
        "frame_count": len(decision_rows),
        "graph_backtrack_frame_count": graph_backtrack_frames,
        "exact_arrival_proxy_frame_count": exact_arrival_proxy_frames,
        "exact_arrival_proxy_fraction": exact_arrival_proxy_frames / len(decision_rows),
        "longest_constant_proxy_run_frames": longest_run,
        "first_proxy_frame": first_proxy_frame,
        "last_proxy_frame": last_proxy_frame,
        "proxy_frames_with_route_arc_growth": proxy_arc_growth_frames,
        "proxy_interval_route_arc_growth_m": (
            None if first_proxy_arc is None or last_proxy_arc is None
            else last_proxy_arc - first_proxy_arc
        ),
        "last_route_arc_growth_frame": last_arc_growth_frame,
        "tail_without_route_arc_growth_frames": final_frame - last_arc_growth_frame,
        "uses_evaluator_gt": False,
    }


__all__ = ["audit_reanchor_evidence"]
