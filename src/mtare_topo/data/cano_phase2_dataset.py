"""Frozen Phase-2 Cano range-dataset selection and teacher contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    ELEVATION_DEG,
    LIDAR_HEIGHT_ABOVE_FLOOR_M,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    interpolate_polyline,
    lidar_local_directions,
    structural_label,
    world_directions,
)

ROLE_ORDER = ("interior", "junction", "terminal")
ROLE_RADIUS_M = 10.0
CLUSTER_LENGTH_M = 5.0
FRAME_STEP_M = 1.0
FIRST_FRAME_ARC_M = 0.5
LOS_MARGIN_M = 0.25
MINIMUM_CLEARANCE_M = 0.8


def spline_arrays(document: Mapping[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in document["tunnels"]
    }


def candidate_clusters(
    parent_id: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the frozen non-overlapping 5 m lattice with independent role flags."""

    splines = spline_arrays(spline_document)
    junctions = [node for node in graph["nodes"] if int(node["degree"]) >= 3]
    terminals = [node for node in graph["nodes"] if int(node["degree"]) == 1]
    result: list[dict[str, Any]] = []
    for tunnel_id, points in sorted(splines.items()):
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for cluster_index in range(int(math.floor(length / CLUSTER_LENGTH_M))):
            frame_arcs = FIRST_FRAME_ARC_M + CLUSTER_LENGTH_M * cluster_index + np.arange(5)
            center_arc = float(frame_arcs[2])
            center, _ = interpolate_polyline(points, center_arc)
            junction_distances = {
                str(node["id"]): float(np.linalg.norm(center - np.asarray(node["xyz"], dtype=np.float64)))
                for node in junctions
                if np.linalg.norm(center - np.asarray(node["xyz"], dtype=np.float64))
                <= ROLE_RADIUS_M + 1e-9
            }
            terminal_distances = {
                str(node["id"]): float(np.linalg.norm(center - np.asarray(node["xyz"], dtype=np.float64)))
                for node in terminals
                if tunnel_id in {int(value) for value in node["incident_tunnel_ids"]}
                and np.linalg.norm(center - np.asarray(node["xyz"], dtype=np.float64))
                <= ROLE_RADIUS_M + 1e-9
            }
            junction_ids = sorted(junction_distances)
            terminal_ids = sorted(terminal_distances)
            primary_role = (
                "terminal" if terminal_ids else "junction" if junction_ids else "interior"
            )
            result.append(
                {
                    "cluster_id": f"{parent_id}_t{tunnel_id:04d}_k{cluster_index:05d}",
                    "parent_id": parent_id,
                    "tunnel_id": tunnel_id,
                    "cluster_index": cluster_index,
                    "frame_arcs_m": frame_arcs.astype(float).tolist(),
                    "center_arc_m": center_arc,
                    "near_junction": bool(junction_ids),
                    "near_terminal": bool(terminal_ids),
                    "junction_event_ids": junction_ids,
                    "terminal_event_ids": terminal_ids,
                    "junction_event_distance_m": junction_distances,
                    "terminal_event_distance_m": terminal_distances,
                    "primary_role": primary_role,
                }
            )
    return result


def select_valid_clusters(
    candidates: Sequence[Mapping[str, Any]],
    quota: Mapping[str, int],
    validator: Callable[[Mapping[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select event-first clusters, then deterministic same-role replacements."""

    by_role = {
        role: [dict(item) for item in candidates if item["primary_role"] == role]
        for role in ROLE_ORDER
    }
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    validity: dict[str, bool] = {}

    def accept(item: dict[str, Any]) -> bool:
        identity = str(item["cluster_id"])
        if identity in selected_ids:
            return True
        if identity not in validity:
            validity[identity] = bool(validator(item))
            if not validity[identity]:
                rejected.append(item)
        if validity[identity]:
            selected.append(item)
            selected_ids.add(identity)
            return True
        return False

    for role, event_key in (("junction", "junction_event_ids"), ("terminal", "terminal_event_ids")):
        event_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in by_role[role]:
            for event_id in item[event_key]:
                event_candidates[str(event_id)].append(item)
        for event_id in sorted(event_candidates):
            distance_key = f"{role}_event_distance_m"
            event_candidates[event_id].sort(
                key=lambda item: (
                    float(item.get(distance_key, {}).get(event_id, float("inf"))),
                    str(item["cluster_id"]),
                )
            )
            if not any(
                chosen["primary_role"] == role and event_id in chosen[event_key]
                for chosen in selected
            ):
                if not any(accept(item) for item in event_candidates[event_id]):
                    raise RuntimeError(f"no valid {role} candidate for event {event_id}")

    for role in ROLE_ORDER:
        target = int(quota[role])
        current = sum(item["primary_role"] == role for item in selected)
        if current > target:
            raise RuntimeError(f"mandatory {role} clusters exceed quota: {current}>{target}")
        for item in by_role[role]:
            if current >= target:
                break
            if item["cluster_id"] not in selected_ids and accept(item):
                current += 1
        if current != target:
            raise RuntimeError(f"valid {role} quota shortage: {current}!={target}")
    return selected, rejected


def frame_contract(
    cluster: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    fta_distance_m: float,
) -> list[dict[str, Any]]:
    points = spline_arrays(spline_document)[int(cluster["tunnel_id"])]
    frames = []
    for index, arc in enumerate(cluster["frame_arcs_m"]):
        axis, tangent = interpolate_polyline(points, float(arc))
        yaw = float(math.degrees(math.atan2(tangent[1], tangent[0])) % 360.0)
        sensor = axis.copy()
        sensor[2] += float(fta_distance_m) + LIDAR_HEIGHT_ABOVE_FLOOR_M
        frames.append(
            {
                "frame_index": index,
                "arc_m": float(arc),
                "axis_xyz_m": axis,
                "sensor_xyz_m": sensor,
                "yaw_deg": yaw,
            }
        )
    return frames


def cast_ranges(scene: Any, origin: np.ndarray, directions: np.ndarray, o3d: Any) -> tuple[np.ndarray, np.ndarray]:
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, np.asarray(directions, dtype=np.float32)), axis=-1)
    hit = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy()
    hit = hit.reshape(directions.shape[:-1])
    valid = np.isfinite(hit) & (hit >= NEAR_RANGE_M) & (hit <= MAX_RANGE_M)
    return np.where(valid, hit, MAX_RANGE_M).astype(np.float32), valid.astype(np.uint8)


def evaluate_frame(
    scene: Any,
    frame: Mapping[str, Any],
    splines: Mapping[int, np.ndarray],
    o3d: Any,
) -> dict[str, Any]:
    """Cast one scan and enforce the frozen objective teacher/LOS contract."""

    origin = np.asarray(frame["sensor_xyz_m"], dtype=np.float64)
    label = structural_label(
        frame.get("teacher_axis_xyz_m", frame["axis_xyz_m"]),
        float(frame["yaw_deg"]),
        splines,
    )
    horizontal_angle = np.radians(np.arange(720, dtype=np.float32) * 0.5)
    horizontal = np.stack((np.cos(horizontal_angle), np.sin(horizontal_angle), np.zeros(720)), axis=-1)
    clearance_range, clearance_valid = cast_ranges(scene, origin, horizontal, o3d)
    hits = clearance_range[clearance_valid.astype(bool)]
    clearance = float(np.min(hits)) if hits.size else MAX_RANGE_M
    directions = world_directions(lidar_local_directions(), float(frame["yaw_deg"]))
    range_m, valid_mask = cast_ranges(scene, origin, directions, o3d)
    representative_points = []
    los = []
    for heading in label["headings_world_deg"]:
        source = min(
            label["raw_intersections"],
            key=lambda item: abs((float(item["heading_world_deg"]) - float(heading) + 180.0) % 360.0 - 180.0),
        )
        target = np.asarray(source["xyz"], dtype=np.float64)
        vector = target - origin
        distance = float(np.linalg.norm(vector))
        branch_range, branch_valid = cast_ranges(scene, origin, (vector / distance).reshape(1, 3), o3d)
        clear = bool((not branch_valid[0]) or float(branch_range[0]) >= distance - LOS_MARGIN_M)
        representative_points.append(target.astype(float).tolist())
        los.append(clear)
    teacher_eligible = bool(1 <= int(label["branch_count"]) <= 8 and los and all(los))
    native_clearance_passed = bool(clearance >= MINIMUM_CLEARANCE_M)
    eligible = bool(teacher_eligible and native_clearance_passed)
    return {
        "eligible": eligible,
        "teacher_eligible": teacher_eligible,
        "native_clearance_passed": native_clearance_passed,
        "range_m": range_m,
        "valid_mask": valid_mask,
        "exit_target": np.asarray(label["label_720"], dtype=np.float32),
        "headings_robot_deg": label["headings_robot_deg"],
        "branch_count": int(label["branch_count"]),
        "minimum_horizontal_clearance_m": clearance,
        "representative_branch_points_world": representative_points,
        "branch_los": los,
    }
