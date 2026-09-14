"""Causal eligibility guard for re-anchoring on a verified graph backtrack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class VerifiedReanchorDecision:
    eligible: bool
    reason: str
    from_node_id: int | None
    next_hop_node_id: int | None
    distance_to_current_m: float | None
    distance_to_next_hop_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _xyz(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _decision(
    eligible: bool,
    reason: str,
    current: int | None,
    next_hop: int | None,
    current_distance: float | None = None,
    next_distance: float | None = None,
) -> VerifiedReanchorDecision:
    return VerifiedReanchorDecision(
        eligible,
        reason,
        current,
        next_hop,
        current_distance,
        next_distance,
    )


def evaluate_verified_reanchor(
    snapshot: Mapping[str, Any],
    active_target: Mapping[str, Any] | None,
    *,
    sensor_xyz_m: Sequence[float],
    route_arc_m: float,
    current_node_route_arc_m: float,
    trace_frame_count: int,
) -> VerifiedReanchorDecision:
    """Return whether the current physical trace proves arrival at a next hop.

    The guard deliberately has no evaluator or GT input.  It may authorize a
    future graph state transition only for a next hop selected on an already
    trace-verified edge.
    """

    if active_target is None or active_target.get("mode") != "graph_backtrack":
        return _decision(False, "active_target_is_not_graph_backtrack", None, None)

    current_raw = snapshot.get("current_node")
    next_raw = active_target.get("next_hop_node_id")
    if current_raw is None or next_raw is None:
        return _decision(False, "current_or_next_hop_is_missing", None, None)
    current, next_hop = int(current_raw), int(next_raw)
    if current == next_hop:
        return _decision(False, "next_hop_is_already_current", current, next_hop)
    path_raw = active_target.get("graph_path_node_ids")
    if not isinstance(path_raw, (list, tuple)) or len(path_raw) < 2:
        return _decision(False, "active_target_path_is_stale", current, next_hop)
    path = tuple(int(value) for value in path_raw)
    if path[0] != current or path[1] != next_hop:
        return _decision(False, "active_target_path_is_stale", current, next_hop)

    nodes = {int(node["id"]): node for node in snapshot.get("nodes", [])}
    if len(nodes) != len(snapshot.get("nodes", [])):
        raise ValueError("graph node IDs must be unique")
    if current not in nodes or next_hop not in nodes:
        return _decision(False, "current_or_next_hop_is_stale", current, next_hop)

    verified = []
    for edge in snapshot.get("edges", []):
        if {int(edge.get("from", -1)), int(edge.get("to", -1))} != {current, next_hop}:
            continue
        if edge.get("kind") != "verified_traversed":
            continue
        traversals = edge.get("traversals", [])
        if not traversals:
            raise ValueError("verified next-hop edge has no traversal evidence")
        for traversal in traversals:
            length = float(traversal.get("length_m", math.nan))
            frames = int(traversal.get("trace_frame_count", 0))
            if not math.isfinite(length) or length <= 0 or frames < 2:
                raise ValueError("verified next-hop traversal evidence is invalid")
        verified.append(edge)
    if not verified:
        return _decision(False, "next_hop_has_no_verified_edge", current, next_hop)
    if len(verified) != 1:
        raise ValueError("verified next-hop edge must be unique")

    if isinstance(trace_frame_count, bool) or int(trace_frame_count) != trace_frame_count:
        raise ValueError("trace_frame_count must be an integer")
    route_arc = float(route_arc_m)
    anchor_arc = float(current_node_route_arc_m)
    if not math.isfinite(route_arc) or not math.isfinite(anchor_arc):
        raise ValueError("route arcs must be finite")
    if int(trace_frame_count) < 2 or route_arc <= anchor_arc:
        return _decision(False, "physical_trace_is_not_positive", current, next_hop)

    pose = _xyz(sensor_xyz_m, name="sensor_xyz_m")
    current_xyz = _xyz(nodes[current]["xyz_m"], name="current_node.xyz_m")
    next_xyz = _xyz(nodes[next_hop]["xyz_m"], name="next_hop.xyz_m")
    distance_to_current = math.dist(pose, current_xyz)
    distance_to_next = math.dist(pose, next_xyz)
    radius = float(snapshot.get("config", {}).get("loop_merge_radius_m", math.nan))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("loop_merge_radius_m must be finite and positive")
    if distance_to_next > radius:
        return _decision(
            False, "pose_is_outside_next_hop_radius", current, next_hop,
            distance_to_current, distance_to_next,
        )
    if distance_to_next >= distance_to_current:
        return _decision(
            False, "pose_is_not_closer_to_next_hop", current, next_hop,
            distance_to_current, distance_to_next,
        )
    return _decision(
        True, "verified_backtrack_arrival", current, next_hop,
        distance_to_current, distance_to_next,
    )


__all__ = ["VerifiedReanchorDecision", "evaluate_verified_reanchor"]
