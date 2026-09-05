"""Frozen frame contract for the Gate-4 C08 causal topology replay."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def validate_m1d_checkpoint_identity(checkpoint: dict, expected_seed: int) -> None:
    """Validate the exact metadata contract written by the frozen M1D trainer."""
    actual_seed = int(checkpoint.get("seed", -1))
    actual_mode = checkpoint.get("mode")
    if actual_seed != int(expected_seed) or actual_mode != "M1D":
        raise ValueError(
            f"checkpoint identity mismatch: expected seed={expected_seed}, mode='M1D'; "
            f"observed seed={actual_seed}, mode={actual_mode!r}"
        )

from mtare_topo.data.cano_phase2_dataset import ROLE_RADIUS_M


ROLE_NAMES = ("interior", "junction", "terminal")


def merge_clearance_risk_segments(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive failed frame records into auditable route segments."""

    risky = [dict(record) for record in records if not record["clearance_passed"]]
    if not risky:
        return []
    groups: list[list[dict[str, Any]]] = [[risky[0]]]
    for record in risky[1:]:
        if int(record["frame_index"]) == int(groups[-1][-1]["frame_index"]) + 1:
            groups[-1].append(record)
        else:
            groups.append([record])
    result = []
    for index, group in enumerate(groups):
        minimum = min(group, key=lambda item: item["minimum_horizontal_clearance_m"])
        result.append({
            "segment_index": index,
            "start_frame": group[0]["frame_index"], "end_frame": group[-1]["frame_index"],
            "frame_count": len(group),
            "start_route_arc_m": group[0]["route_arc_m"], "end_route_arc_m": group[-1]["route_arc_m"],
            "sampled_span_m": float(group[-1]["route_arc_m"] - group[0]["route_arc_m"]),
            "minimum_clearance_m": minimum["minimum_horizontal_clearance_m"],
            "minimum_frame": minimum["frame_index"],
            "edge_ids": sorted({str(item["edge_id"]) for item in group}),
            "tunnel_ids": sorted({int(item["tunnel_id"]) for item in group}),
            "minimum_axis_xyz_m": minimum["axis_xyz_m"],
        })
    return result


def traversal_index_for_arcs(
    route_arc_m: Sequence[float], traversals: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Map each sampled route arc to its physical directed traversal.

    Route boundaries belong to the traversal that starts at that boundary.  The
    final sampled point is strictly before the route endpoint by contract.
    """

    arcs = np.asarray(route_arc_m, dtype=np.float64)
    starts = np.asarray([item["route_start_m"] for item in traversals], dtype=np.float64)
    ends = np.asarray([item["route_end_m"] for item in traversals], dtype=np.float64)
    if arcs.ndim != 1 or not len(starts):
        raise ValueError("route arcs and traversals must be non-empty")
    if np.any(np.diff(arcs) <= 0) or np.any(np.diff(starts) < 0) or np.any(ends <= starts):
        raise ValueError("route/traversal arcs are not strictly ordered")
    indices = np.searchsorted(starts, arcs, side="right") - 1
    if np.any(indices < 0) or np.any(arcs >= ends[indices] + 1e-7):
        raise ValueError("sampled route arc falls outside traversal contract")
    return indices.astype(np.int32)


def objective_role(
    axis_xyz_m: Sequence[float], tunnel_id: int, graph: Mapping[str, Any]
) -> tuple[str, list[str], list[str]]:
    """Apply the frozen terminal-first Gate-1 structural-role teacher."""

    point = np.asarray(axis_xyz_m, dtype=np.float64)
    junction_ids: list[str] = []
    terminal_ids: list[str] = []
    for node in graph["nodes"]:
        degree = int(node["degree"])
        distance = float(np.linalg.norm(point - np.asarray(node["xyz"], dtype=np.float64)))
        if distance > ROLE_RADIUS_M + 1e-9:
            continue
        if degree >= 3:
            junction_ids.append(str(node["id"]))
        elif degree == 1 and int(tunnel_id) in {
            int(value) for value in node["incident_tunnel_ids"]
        }:
            terminal_ids.append(str(node["id"]))
    role = "terminal" if terminal_ids else "junction" if junction_ids else "interior"
    return role, sorted(junction_ids), sorted(terminal_ids)


def frame_contract(
    *,
    xyz_m: np.ndarray | None = None,
    teacher_axis_xyz_m: np.ndarray | None = None,
    graph_axis_xyz_m: np.ndarray | None = None,
    sensor_xyz_m: np.ndarray | None = None,
    tangent_world: np.ndarray,
    route_arc_m: np.ndarray,
    traversals: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    fta_distance_m: float,
) -> list[dict[str, Any]]:
    """Create deterministic frame metadata without raycasting or future data."""
    explicit = (teacher_axis_xyz_m, graph_axis_xyz_m, sensor_xyz_m)
    if any(value is not None for value in explicit):
        if xyz_m is not None or not all(value is not None for value in explicit):
            raise ValueError("use either legacy xyz_m or all three explicit teacher/graph/sensor coordinates")
        teacher_xyz = np.asarray(teacher_axis_xyz_m, dtype=np.float64)
        graph_xyz = np.asarray(graph_axis_xyz_m, dtype=np.float64)
        sensor_xyz = np.asarray(sensor_xyz_m, dtype=np.float64)
        if teacher_xyz.shape != graph_xyz.shape or graph_xyz.shape != sensor_xyz.shape:
            raise ValueError("teacher, graph, and sensor coordinate arrays must share shape")
    else:
        if xyz_m is None:
            raise ValueError("trajectory coordinates are required")
        teacher_xyz = np.asarray(xyz_m, dtype=np.float64)
        graph_xyz = teacher_xyz.copy()
        sensor_xyz = graph_xyz.copy()
        sensor_xyz[:, 2] += float(fta_distance_m) + 1.0
    tangents = np.asarray(tangent_world, dtype=np.float64)
    arcs = np.asarray(route_arc_m, dtype=np.float64)
    if graph_xyz.shape != tangents.shape or graph_xyz.ndim != 2 or graph_xyz.shape[1] != 3:
        raise ValueError("xyz and tangent arrays must share shape [frames,3]")
    if arcs.shape != (len(graph_xyz),):
        raise ValueError("route arc array does not match trajectory frames")
    mapping = traversal_index_for_arcs(arcs, traversals)
    result: list[dict[str, Any]] = []
    for frame_index, (teacher_axis, graph_axis, sensor, tangent, arc, traversal_index) in enumerate(
        zip(teacher_xyz, graph_xyz, sensor_xyz, tangents, arcs, mapping)
    ):
        traversal = traversals[int(traversal_index)]
        tunnel_id = int(traversal["tunnel_id"])
        yaw_deg = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 360.0)
        role, junction_ids, terminal_ids = objective_role(teacher_axis, tunnel_id, graph)
        result.append(
            {
                "frame_index": frame_index,
                "route_arc_m": float(arc),
                "axis_xyz_m": graph_axis,
                "teacher_axis_xyz_m": teacher_axis,
                "graph_axis_xyz_m": graph_axis,
                "sensor_xyz_m": sensor,
                "yaw_deg": yaw_deg,
                "traversal_index": int(traversal_index),
                "edge_id": str(traversal["edge_id"]),
                "tunnel_id": tunnel_id,
                "objective_role": role,
                "objective_role_index": ROLE_NAMES.index(role),
                "junction_event_ids": junction_ids,
                "terminal_event_ids": terminal_ids,
            }
        )
    return result
