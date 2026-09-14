"""Typed relative spatial-event Teacher with native-mesh LOS qualification."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np


EventType = Literal["junction", "terminal", "geometry_transition"]
BatchRaycastFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
MAX_EVENT_RANGE_M = 50.0
LOS_MARGIN_M = 0.25


def event_type_for_degree(degree: int) -> EventType | None:
    """Map objective graph degree to decision-event semantics."""

    value = int(degree)
    if value == 1:
        return "terminal"
    if value >= 3:
        return "junction"
    return None


def robot_relative_xyz(
    sensor_xyz_m: Sequence[float],
    robot_yaw_deg: float,
    target_xyz_m: Sequence[float],
) -> tuple[float, float, float]:
    """Express one world target as forward/left/up robot coordinates."""

    sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    yaw = float(robot_yaw_deg)
    if sensor.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite(sensor)) or not np.all(np.isfinite(target)) or not math.isfinite(yaw):
        raise ValueError("spatial event relative-coordinate input is invalid")
    delta = target - sensor
    angle = math.radians(yaw)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        float(cosine * delta[0] + sine * delta[1]),
        float(-sine * delta[0] + cosine * delta[1]),
        float(delta[2]),
    )


def has_opposite_headings(yaw_degrees: Sequence[float]) -> bool:
    """Return whether one observed heading pair differs by 120--240 degrees."""

    angles = np.sort(np.mod(np.asarray(yaw_degrees, dtype=np.float64), 360.0))
    if angles.ndim != 1 or len(angles) < 2 or not np.all(np.isfinite(angles)):
        return False
    doubled = np.concatenate((angles, angles + 360.0))
    for angle in angles:
        index = int(np.searchsorted(doubled, angle + 120.0 - 1e-9, side="left"))
        if index < len(doubled) and doubled[index] <= angle + 240.0 + 1e-9:
            return True
    return False


@dataclass(frozen=True)
class SpatialStructureEventTarget:
    """One objective structural event expressed in the current robot frame."""

    identity: str
    node_id: str
    event_type: EventType
    degree: int
    objective_axis_xyz_m: tuple[float, float, float]
    target_sensor_xyz_m: tuple[float, float, float]
    relative_xyz_m: tuple[float, float, float]
    distance_m: float
    visible: bool
    first_hit_distance_m: float | None
    source: str = "tng_node_native_mesh_los"

    def __post_init__(self) -> None:
        vectors = (self.objective_axis_xyz_m, self.target_sensor_xyz_m, self.relative_xyz_m)
        if (
            not self.identity or not self.node_id
            or self.event_type not in ("junction", "terminal", "geometry_transition")
            or self.degree < 1
            or any(len(value) != 3 or not all(math.isfinite(float(item)) for item in value) for value in vectors)
            or not math.isfinite(float(self.distance_m)) or self.distance_m < 0.0
            or self.first_hit_distance_m is not None and (
                not math.isfinite(float(self.first_hit_distance_m)) or self.first_hit_distance_m < 0.0
            )
        ):
            raise ValueError("spatial structure event target is invalid")


def spatial_structure_event_targets(
    *,
    parent_id: str,
    graph: Mapping[str, Any],
    current_sensor_xyz_m: Sequence[float],
    robot_yaw_deg: float,
    fta_distance_m: float,
    cast_distances: BatchRaycastFunction,
    maximum_range_m: float = MAX_EVENT_RANGE_M,
    los_margin_m: float = LOS_MARGIN_M,
) -> tuple[SpatialStructureEventTarget, ...]:
    """Return all range-qualified decision nodes with independent LOS flags.

    Objective node identity and world coordinates are Teacher outputs only.
    They are never part of the student input.  A target ray ends at the same
    sensor-height convention used by the frozen causal LiDAR poses.
    """

    sensor = np.asarray(current_sensor_xyz_m, dtype=np.float64)
    yaw = float(robot_yaw_deg)
    fta = float(fta_distance_m)
    maximum = float(maximum_range_m)
    margin = float(los_margin_m)
    if (
        not parent_id or sensor.shape != (3,) or not np.all(np.isfinite(sensor))
        or not math.isfinite(yaw) or not math.isfinite(fta)
        or not math.isfinite(maximum) or maximum <= 0.0
        or not math.isfinite(margin) or margin < 0.0
    ):
        raise ValueError("spatial multi-event Teacher inputs are invalid")
    raw_nodes = list(graph.get("nodes", ()))
    ids = [str(node.get("id", "")) for node in raw_nodes]
    if not raw_nodes or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("spatial multi-event graph node identity drift")

    candidates = []
    for node in raw_nodes:
        event_type = event_type_for_degree(int(node["degree"]))
        if event_type is None:
            continue
        axis = np.asarray(node["xyz"], dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("spatial multi-event graph coordinate drift")
        target = axis.copy()
        target[2] += fta + 1.0
        vector = target - sensor
        distance = float(np.linalg.norm(vector))
        if distance <= maximum + 1e-9:
            candidates.append((node, event_type, axis, target, vector, distance))
    if not candidates:
        return ()

    origins = np.repeat(sensor[None, :], len(candidates), axis=0)
    directions = np.zeros((len(candidates), 1, 3), dtype=np.float64)
    for index, value in enumerate(candidates):
        distance = value[-1]
        directions[index, 0] = value[-2] / distance if distance > 1e-8 else np.asarray((1.0, 0.0, 0.0))
    raw_hits = np.asarray(cast_distances(origins, directions), dtype=np.float64)
    if raw_hits.shape != (len(candidates), 1) or np.any(np.isnan(raw_hits)) or np.any(raw_hits < 0.0):
        raise ValueError("spatial multi-event raycast result drift")

    result = []
    for index, (node, event_type, axis, target, _, distance) in enumerate(candidates):
        raw_hit = float(raw_hits[index, 0])
        hit = raw_hit if math.isfinite(raw_hit) else None
        visible = bool(distance <= 1e-8 or hit is None or hit >= distance - margin)
        result.append(SpatialStructureEventTarget(
            identity=f"{parent_id}:node:{node['id']}",
            node_id=str(node["id"]),
            event_type=event_type,
            degree=int(node["degree"]),
            objective_axis_xyz_m=tuple(float(value) for value in axis),
            target_sensor_xyz_m=tuple(float(value) for value in target),
            relative_xyz_m=robot_relative_xyz(sensor, yaw, target),
            distance_m=distance,
            visible=visible,
            first_hit_distance_m=hit,
        ))
    return tuple(sorted(result, key=lambda value: (value.distance_m, value.identity)))


__all__ = [
    "BatchRaycastFunction", "EventType", "LOS_MARGIN_M", "MAX_EVENT_RANGE_M",
    "SpatialStructureEventTarget", "event_type_for_degree", "robot_relative_xyz",
    "has_opposite_headings", "spatial_structure_event_targets",
]
