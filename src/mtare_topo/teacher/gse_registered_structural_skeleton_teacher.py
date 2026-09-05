"""Objective visible-skeleton Teacher primitives for ERCSS.

The TNG, splines, geometry and native mesh are Teacher-only.  Student input is
the registered causal LiDAR point set; no world, traversal or identity enters
the learned representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Any

import numpy as np

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG, MAX_RANGE_M
from mtare_topo.data.local_implicit_union import edge_arc_incidence
from mtare_topo.representation.gse_registered_structural_skeleton import (
    EgoConnectedStructuralSkeleton,
)
from mtare_topo.teacher.gse_geometry_teacher import local_geometry_target


LOS_MARGIN_M = 0.25
SKELETON_SAMPLE_SPACING_M = 1.0
BatchRaycastFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SampledPhysicalSkeleton:
    node_xyz_world_m: np.ndarray
    segment_node_indices: np.ndarray
    segment_edge_index: np.ndarray
    segment_geometry: np.ndarray
    node_identity: np.ndarray
    edge_identity: tuple[str, ...]

    def __post_init__(self) -> None:
        nodes = np.asarray(self.node_xyz_world_m, dtype=np.float64)
        segments = np.asarray(self.segment_node_indices, dtype=np.int32)
        segment_edge = np.asarray(self.segment_edge_index, dtype=np.int32)
        identities = np.asarray(self.node_identity, dtype=np.int64)
        geometry = np.asarray(self.segment_geometry, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.all(np.isfinite(nodes)):
            raise ValueError("sampled physical skeleton nodes must be finite [N,3]")
        if segments.ndim != 2 or segments.shape[1] != 2 or segment_edge.shape != (len(segments),):
            raise ValueError("sampled physical skeleton segment contract drift")
        if identities.shape != (len(nodes),) or geometry.shape != (len(segments), 4):
            raise ValueError("sampled physical skeleton identity/geometry drift")
        if np.any(segments < 0) or np.any(segments >= len(nodes)):
            raise ValueError("sampled physical skeleton segment endpoint drift")
        if np.any(segment_edge < 0) or np.any(segment_edge >= len(self.edge_identity)):
            raise ValueError("sampled physical skeleton edge lookup drift")
        object.__setattr__(self, "node_xyz_world_m", nodes)
        object.__setattr__(self, "segment_node_indices", segments)
        object.__setattr__(self, "segment_edge_index", segment_edge)
        object.__setattr__(self, "segment_geometry", geometry)
        object.__setattr__(self, "node_identity", identities)


def _interpolate_polyline_interval(
    points: np.ndarray,
    first_arc_m: float,
    second_arc_m: float,
    spacing_m: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if np.any(lengths <= 1e-12) or spacing_m <= 0.0:
        raise ValueError("physical spline/spacing is degenerate")
    direction = 1.0 if second_arc_m > first_arc_m else -1.0
    distance = abs(float(second_arc_m) - float(first_arc_m))
    count = max(1, int(math.ceil(distance / float(spacing_m))))
    arcs = np.linspace(float(first_arc_m), float(second_arc_m), count + 1)
    indices = np.minimum(np.searchsorted(cumulative, arcs, side="right") - 1, len(lengths) - 1)
    indices = np.maximum(indices, 0)
    ratios = (arcs - cumulative[indices]) / lengths[indices]
    result = points[indices] + ratios[:, None] * (points[indices + 1] - points[indices])
    if direction < 0.0 and not np.all(np.diff(arcs) < 0.0):
        raise RuntimeError("reverse physical edge interpolation drift")
    return result


def sample_physical_skeleton(
    graph: Mapping[str, Any],
    splines: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    spacing_m: float = SKELETON_SAMPLE_SPACING_M,
) -> SampledPhysicalSkeleton:
    """Sample every TNG physical edge while sharing exact topology endpoints."""

    graph_nodes = {str(row["id"]): np.asarray(row["xyz"], dtype=np.float64) for row in graph["nodes"]}
    curves = {int(row["tunnel_id"]): np.asarray(row["points"], dtype=np.float64) for row in splines["tunnels"]}
    radii = {int(row["tunnel_id"]): float(row["radius_m"]) for row in geometry["tunnels"]}
    incidence_by_edge: dict[str, list[Any]] = {}
    for record in edge_arc_incidence(graph, splines):
        incidence_by_edge.setdefault(record.edge_id, []).append(record)

    node_rows: list[np.ndarray] = []
    node_ids: list[int] = []
    graph_node_index: dict[str, int] = {}
    for identity, node_id in enumerate(sorted(graph_nodes)):
        graph_node_index[node_id] = len(node_rows)
        node_rows.append(graph_nodes[node_id])
        node_ids.append(identity)

    segments: list[tuple[int, int]] = []
    segment_edge: list[int] = []
    edge_ids: list[str] = []
    segment_geometry: list[tuple[float, float, float, float]] = []
    next_node_identity = len(node_ids)
    edges_by_id = {str(row["id"]): row for row in graph["edges"]}
    for edge_index, edge_id in enumerate(sorted(edges_by_id)):
        records = sorted(incidence_by_edge.get(edge_id, ()), key=lambda row: row.endpoint)
        if len(records) != 2 or {row.endpoint for row in records} != {"first", "second"}:
            raise ValueError(f"physical edge {edge_id} lacks two endpoint incidences")
        first = next(row for row in records if row.endpoint == "first")
        second = next(row for row in records if row.endpoint == "second")
        if first.tunnel_id != second.tunnel_id or first.tunnel_id not in radii:
            raise ValueError(f"physical edge {edge_id} tunnel identity drift")
        sampled = _interpolate_polyline_interval(
            curves[first.tunnel_id], first.arc_m, second.arc_m, float(spacing_m),
        )
        # Topological endpoints are shared exactly.  Interior points remain on
        # the physical spline and are never collapsed by tunnel identity.
        path = [graph_node_index[first.node_id]]
        for point in sampled[1:-1]:
            path.append(len(node_rows)); node_rows.append(point); node_ids.append(next_node_identity)
            next_node_identity += 1
        path.append(graph_node_index[second.node_id])
        for left, right in zip(path[:-1], path[1:]):
            segments.append((left, right)); segment_edge.append(edge_index)
        radius = radii[first.tunnel_id]
        target = local_geometry_target(
            curves[first.tunnel_id], 0.5 * (first.arc_m + second.arc_m),
            tunnel_radius_m=radius,
        )
        edge_ids.append(edge_id)
        segment_geometry.extend(
            [(target.width_m, target.height_m, target.slope_deg, target.curvature_per_m)]
            * (len(path) - 1)
        )
    return SampledPhysicalSkeleton(
        np.asarray(node_rows), np.asarray(segments, dtype=np.int32),
        np.asarray(segment_edge, dtype=np.int32), np.asarray(segment_geometry, dtype=np.float64),
        np.asarray(node_ids, dtype=np.int64), tuple(edge_ids),
    )


def causal_mesh_visibility(
    points_world_m: np.ndarray,
    sensor_xyz_world_m: np.ndarray,
    yaw_world_deg: np.ndarray,
    cast_distances: BatchRaycastFunction,
    *,
    maximum_range_m: float = MAX_RANGE_M,
    los_margin_m: float = LOS_MARGIN_M,
) -> np.ndarray:
    """Mark Teacher points visible from any past/current LiDAR pose.

    Native-mesh LOS is intersected with the frozen LiDAR vertical field of
    view.  This avoids declaring skeleton behind a wall or outside all 16
    beams observable.
    """

    points = np.asarray(points_world_m, dtype=np.float64)
    sensors = np.asarray(sensor_xyz_world_m, dtype=np.float64)
    yaws = np.asarray(yaw_world_deg, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or sensors.ndim != 2 or sensors.shape[1] != 3:
        raise ValueError("visibility points/sensors must be [N,3]/[T,3]")
    if yaws.shape != (len(sensors),) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(sensors)) or not np.all(np.isfinite(yaws)):
        raise ValueError("visibility inputs must be finite and aligned")
    if maximum_range_m <= 0.0 or los_margin_m < 0.0:
        raise ValueError("visibility distance contract is invalid")
    minimum_elevation = float(np.min(ELEVATION_DEG)) - 1e-9
    maximum_elevation = float(np.max(ELEVATION_DEG)) + 1e-9
    # Form all causal pose/point pairs and cast them in one batch.  This keeps
    # full five-frame semantics without thousands of tiny raycaster calls.
    delta = points[None, :, :] - sensors[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    horizontal = np.linalg.norm(delta[..., :2], axis=2)
    elevation = np.degrees(np.arctan2(delta[..., 2], horizontal))
    eligible = (distance <= maximum_range_m + 1e-9) & (elevation >= minimum_elevation) & (elevation <= maximum_elevation)
    sensor_index, point_index = np.nonzero(eligible)
    visible = np.zeros(len(points), dtype=bool)
    if len(point_index) == 0:
        return visible
    directions = delta[sensor_index, point_index].copy()
    pair_distance = distance[sensor_index, point_index]
    nonzero = pair_distance > 1e-8
    directions[nonzero] /= pair_distance[nonzero, None]
    directions[~nonzero] = np.asarray((1.0, 0.0, 0.0))
    origins = sensors[sensor_index]
    hits = np.asarray(cast_distances(origins, directions), dtype=np.float64)
    if hits.shape != (len(point_index),) or np.any(np.isnan(hits)) or np.any(hits < 0.0):
        raise ValueError("native-mesh visibility raycast drift")
    pair_visible = (~np.isfinite(hits)) | (hits >= pair_distance - los_margin_m)
    visible[np.unique(point_index[pair_visible])] = True
    return visible


def ego_connected_component(
    sampled: SampledPhysicalSkeleton,
    visible_node_mask: np.ndarray,
    ego_xyz_world_m: np.ndarray,
) -> np.ndarray:
    """Return the unique visible discrete component nearest the current axis."""

    visible = np.asarray(visible_node_mask, dtype=bool)
    ego = np.asarray(ego_xyz_world_m, dtype=np.float64)
    if visible.shape != (len(sampled.node_xyz_world_m),) or ego.shape != (3,) or not np.all(np.isfinite(ego)):
        raise ValueError("visible component input drift")
    candidates = np.flatnonzero(visible)
    if len(candidates) == 0:
        return np.zeros_like(visible)
    distances = np.linalg.norm(sampled.node_xyz_world_m[candidates] - ego, axis=1)
    start = int(candidates[int(np.argmin(distances))])
    adjacency: list[list[int]] = [[] for _ in range(len(visible))]
    for first, second in sampled.segment_node_indices:
        if visible[first] and visible[second]:
            adjacency[int(first)].append(int(second)); adjacency[int(second)].append(int(first))
    selected = np.zeros_like(visible)
    selected[start] = True
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if not selected[neighbor]:
                selected[neighbor] = True; stack.append(neighbor)
    return selected


def materialize_visible_skeleton(
    sampled: SampledPhysicalSkeleton,
    component_mask: np.ndarray,
    ego_xyz_world_m: np.ndarray,
    current_sensor_xyz_world_m: np.ndarray,
    current_yaw_deg: float,
) -> EgoConnectedStructuralSkeleton:
    """Materialize the visible component without inventing unobserved links."""

    selected = np.asarray(component_mask, dtype=bool)
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        raise ValueError("cannot materialize an empty visible skeleton")
    old_to_new = np.full(len(selected), -1, dtype=np.int32)
    old_to_new[indices] = np.arange(len(indices), dtype=np.int32)
    edge_mask = selected[sampled.segment_node_indices[:, 0]] & selected[sampled.segment_node_indices[:, 1]]
    segments = old_to_new[sampled.segment_node_indices[edge_mask]]
    segment_edge = sampled.segment_edge_index[edge_mask]
    geometry = sampled.segment_geometry[edge_mask]
    valid = np.isfinite(geometry)
    sensor = np.asarray(current_sensor_xyz_world_m, dtype=np.float64)
    yaw = math.radians(float(current_yaw_deg))
    inverse = np.asarray(((math.cos(yaw), math.sin(yaw), 0.0), (-math.sin(yaw), math.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
    local_nodes = (sampled.node_xyz_world_m[indices] - sensor) @ inverse.T
    ego_old = int(indices[np.argmin(np.linalg.norm(sampled.node_xyz_world_m[indices] - np.asarray(ego_xyz_world_m), axis=1))])
    return EgoConnectedStructuralSkeleton(
        local_nodes, segments, geometry, valid, int(old_to_new[ego_old]),
        sampled.node_identity[indices], segment_edge.astype(np.int64),
    )


__all__ = [
    "LOS_MARGIN_M", "SKELETON_SAMPLE_SPACING_M", "SampledPhysicalSkeleton",
    "causal_mesh_visibility", "ego_connected_component",
    "materialize_visible_skeleton", "sample_physical_skeleton",
]
