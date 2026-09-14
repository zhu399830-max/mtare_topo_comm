"""Unique-frame pose and batched range export contracts for GSE-Graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    LIDAR_HEIGHT_ABOVE_FLOOR_M,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
)
from mtare_topo.data.gse_sequence_inventory import deduplicated_frame_arcs_and_sequence_indices
from mtare_topo.data.gse_teacher_inventory import oriented_edge_polyline
from mtare_topo.teacher.gse_geometry_teacher import PolylineGeometrySampler
from mtare_topo.teacher.gse_mesh_geometry_teacher import sensor_origin_from_axis


FullScanBatchCast = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WorldUniqueFramePoses:
    global_frame_indices: np.ndarray
    traversal_ids: tuple[str, ...]
    local_frame_indices: np.ndarray
    arc_m: np.ndarray
    axis_xyz_m: np.ndarray
    sensor_xyz_m: np.ndarray
    tangent_world_xyz: np.ndarray
    yaw_deg: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.global_frame_indices)
        if len(self.traversal_ids) != count:
            raise ValueError("traversal identity count mismatch")
        for value in (
            self.local_frame_indices,
            self.arc_m,
            self.axis_xyz_m,
            self.sensor_xyz_m,
            self.tangent_world_xyz,
            self.yaw_deg,
        ):
            if len(value) != count:
                raise ValueError("frame pose array count mismatch")
        if (
            self.axis_xyz_m.shape != (count, 3)
            or self.sensor_xyz_m.shape != (count, 3)
            or self.tangent_world_xyz.shape != (count, 3)
        ):
            raise ValueError("axis, sensor and tangent arrays must be [frames,3]")
        if count and (
            not np.all(np.diff(self.global_frame_indices) == 1)
            or not np.all(np.isfinite(self.arc_m))
            or not np.all(np.isfinite(self.axis_xyz_m))
            or not np.all(np.isfinite(self.sensor_xyz_m))
            or not np.all(np.isfinite(self.tangent_world_xyz))
            or not np.all(np.isfinite(self.yaw_deg))
            or not np.allclose(
                np.linalg.norm(self.tangent_world_xyz, axis=1),
                1.0,
                rtol=0.0,
                atol=1e-8,
            )
        ):
            raise ValueError("frame poses must be contiguous and finite")


@dataclass(frozen=True)
class FramePoseCorrection:
    global_frame_index: int
    traversal_id: str
    local_frame_index: int
    original_arc_m: float
    corrected_arc_m: float

    @property
    def inward_shift_m(self) -> float:
        return abs(self.corrected_arc_m - self.original_arc_m)


def world_unique_frame_poses(
    *,
    parent_id: str,
    traversal_manifest: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> WorldUniqueFramePoses:
    """Reconstruct each manifest-referenced directed frame exactly once."""

    records = [record for record in traversal_manifest if str(record["parent_id"]) == parent_id]
    records.sort(key=lambda record: int(record["global_frame_offset"]))
    nodes = {str(node["id"]): np.asarray(node["xyz"], dtype=np.float64) for node in graph["nodes"]}
    splines = {
        str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    edges = {str(edge["id"]): edge for edge in graph["edges"]}
    fta_distance_m = float(geometry_parameters["fta_distance_m"])
    global_indices: list[int] = []
    traversal_ids: list[str] = []
    local_indices: list[int] = []
    arcs: list[float] = []
    axes: list[np.ndarray] = []
    sensors: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    yaws: list[float] = []
    for record in records:
        traversal_id = str(record["traversal_id"])
        direction_index = int(traversal_id.rsplit(":d", 1)[1])
        if direction_index not in (0, 1):
            raise ValueError("directed traversal suffix must be d0 or d1")
        points, _, from_node_id, to_node_id = oriented_edge_polyline(
            edge=edges[str(record["edge_id"])],
            nodes=nodes,
            splines=splines,
            reverse=bool(direction_index),
        )
        if from_node_id != str(record["from_node_id"]) or to_node_id != str(record["to_node_id"]):
            raise RuntimeError("traversal endpoint identity drift")
        sampler = PolylineGeometrySampler(points)
        frame_arcs, anchors, _ = deduplicated_frame_arcs_and_sequence_indices(
            sampler.length_m,
            spacing_m=spacing_m,
            history_frames=history_frames,
        )
        if len(frame_arcs) != int(record["unique_frame_count"]) or len(anchors) != int(record["sequence_count"]):
            raise RuntimeError("traversal frame count drift")
        offset = int(record["global_frame_offset"])
        for local_index, arc in enumerate(frame_arcs):
            axis = sampler.interpolate(float(arc))
            tangent = np.asarray(
                sampler.target(float(arc), tunnel_radius_m=1.0).axis,
                dtype=np.float64,
            )
            yaw = math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0
            global_indices.append(offset + local_index)
            traversal_ids.append(traversal_id)
            local_indices.append(local_index)
            arcs.append(float(arc))
            axes.append(axis)
            sensors.append(
                sensor_origin_from_axis(
                    axis,
                    fta_distance_m=fta_distance_m,
                    sensor_height_above_floor_m=LIDAR_HEIGHT_ABOVE_FLOOR_M,
                )
            )
            tangents.append(tangent)
            yaws.append(yaw)
    count = len(global_indices)
    return WorldUniqueFramePoses(
        global_frame_indices=np.asarray(global_indices, dtype=np.int64),
        traversal_ids=tuple(traversal_ids),
        local_frame_indices=np.asarray(local_indices, dtype=np.int32),
        arc_m=np.asarray(arcs, dtype=np.float64),
        axis_xyz_m=np.asarray(axes, dtype=np.float64).reshape(count, 3),
        sensor_xyz_m=np.asarray(sensors, dtype=np.float64).reshape(count, 3),
        tangent_world_xyz=np.asarray(tangents, dtype=np.float64).reshape(count, 3),
        yaw_deg=np.asarray(yaws, dtype=np.float64),
    )


def world_finite_union_qualified_frame_poses(
    *,
    parent_id: str,
    traversal_manifest: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    spacing_m: float = 1.0,
    history_frames: int = 5,
    signed_distance_fields: Sequence[Callable[[np.ndarray], np.ndarray]],
    inside_tolerance_m: float = 1e-9,
    search_step_m: float = 0.025,
    inward_guard_m: float = 0.025,
    maximum_shift_m: float = 0.75,
    bisection_steps: int = 40,
) -> tuple[WorldUniqueFramePoses, tuple[FramePoseCorrection, ...]]:
    """Move only union-unsafe endpoint poses minimally inward along traversal.

    Every supplied frozen geometry realization must contain the resulting
    sensor origin.  Geometry, frame identity, count and order are untouched.
    """

    if not signed_distance_fields:
        raise ValueError("at least one signed-distance field is required")
    for name, value in (
        ("inside_tolerance_m", inside_tolerance_m),
        ("search_step_m", search_step_m),
        ("inward_guard_m", inward_guard_m),
        ("maximum_shift_m", maximum_shift_m),
    ):
        if not math.isfinite(float(value)) or (name != "inside_tolerance_m" and value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if inside_tolerance_m < 0.0 or bisection_steps < 1:
        raise ValueError("inside tolerance must be nonnegative and bisection_steps positive")

    original = world_unique_frame_poses(
        parent_id=parent_id, traversal_manifest=traversal_manifest, graph=graph,
        spline_document=spline_document, geometry_parameters=geometry_parameters,
        spacing_m=spacing_m, history_frames=history_frames,
    )
    original_values = np.stack(
        [np.asarray(query(original.sensor_xyz_m), dtype=np.float64) for query in signed_distance_fields],
        axis=1,
    )
    if original_values.shape != (len(original.global_frame_indices), len(signed_distance_fields)):
        raise ValueError("signed-distance field returned the wrong shape")
    unsafe = np.flatnonzero(np.any(original_values > inside_tolerance_m, axis=1))

    nodes = {str(node["id"]): np.asarray(node["xyz"], dtype=np.float64) for node in graph["nodes"]}
    splines = {str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64) for item in spline_document["tunnels"]}
    edges = {str(edge["id"]): edge for edge in graph["edges"]}
    records_by_traversal = {
        str(record["traversal_id"]): record
        for record in traversal_manifest
        if str(record["parent_id"]) == parent_id
    }
    samplers: dict[str, PolylineGeometrySampler] = {}
    for traversal_id, record in records_by_traversal.items():
        direction_index = int(traversal_id.rsplit(":d", 1)[1])
        points, _, _, _ = oriented_edge_polyline(
            edge=edges[str(record["edge_id"])], nodes=nodes, splines=splines,
            reverse=bool(direction_index),
        )
        samplers[traversal_id] = PolylineGeometrySampler(points)

    corrected_arc = original.arc_m.copy()
    corrected_axis = original.axis_xyz_m.copy()
    corrected_sensor = original.sensor_xyz_m.copy()
    corrected_tangent = original.tangent_world_xyz.copy()
    corrected_yaw = original.yaw_deg.copy()
    fta_distance_m = float(geometry_parameters["fta_distance_m"])
    corrections: list[FramePoseCorrection] = []

    def sensor_at(sampler: PolylineGeometrySampler, arc_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        axis = sampler.interpolate(float(arc_m))
        tangent = np.asarray(sampler.target(float(arc_m), tunnel_radius_m=1.0).axis, dtype=np.float64)
        sensor = sensor_origin_from_axis(
            axis, fta_distance_m=fta_distance_m,
            sensor_height_above_floor_m=LIDAR_HEIGHT_ABOVE_FLOOR_M,
        )
        yaw = math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0
        return axis, sensor, tangent, yaw

    def is_inside(sensor: np.ndarray) -> bool:
        query = sensor.reshape(1, 3)
        return all(float(np.asarray(field(query)).reshape(-1)[0]) <= inside_tolerance_m for field in signed_distance_fields)

    for index in unsafe:
        traversal_id = original.traversal_ids[int(index)]
        traversal_indices = np.flatnonzero(np.asarray(original.traversal_ids, dtype=object) == traversal_id)
        position = int(np.flatnonzero(traversal_indices == index)[0])
        if position == 0:
            direction = 1.0
            neighbour_arc = float(original.arc_m[traversal_indices[1]]) if len(traversal_indices) > 1 else math.inf
        elif position == len(traversal_indices) - 1:
            direction = -1.0
            neighbour_arc = float(original.arc_m[traversal_indices[-2]]) if len(traversal_indices) > 1 else -math.inf
        else:
            raise RuntimeError("union-unsafe pose is not a traversal endpoint frame")
        sampler = samplers[traversal_id]
        source_arc = float(original.arc_m[index])
        safe_shift = None
        shift = float(search_step_m)
        while shift <= maximum_shift_m + 1e-12:
            candidate_arc = float(np.clip(source_arc + direction * shift, 0.0, sampler.length_m))
            if (direction > 0 and candidate_arc >= neighbour_arc - 1e-9) or (direction < 0 and candidate_arc <= neighbour_arc + 1e-9):
                break
            if is_inside(sensor_at(sampler, candidate_arc)[1]):
                safe_shift = shift
                break
            shift += float(search_step_m)
        if safe_shift is None:
            raise RuntimeError(f"no finite-union-safe endpoint pose within {maximum_shift_m} m: {traversal_id}")
        low, high = 0.0, safe_shift
        for _ in range(bisection_steps):
            middle = 0.5 * (low + high)
            if is_inside(sensor_at(sampler, source_arc + direction * middle)[1]):
                high = middle
            else:
                low = middle
        final_shift = high + float(inward_guard_m)
        final_arc = source_arc + direction * final_shift
        if final_shift > maximum_shift_m + 1e-12 or final_arc < 0.0 or final_arc > sampler.length_m:
            raise RuntimeError("guarded finite-union correction exceeds traversal bound")
        if (direction > 0 and final_arc >= neighbour_arc - 1e-9) or (direction < 0 and final_arc <= neighbour_arc + 1e-9):
            raise RuntimeError("finite-union correction collapses frame order")
        axis, sensor, tangent, yaw = sensor_at(sampler, final_arc)
        if not is_inside(sensor):
            raise RuntimeError("guarded finite-union correction remains outside")
        corrected_arc[index] = final_arc; corrected_axis[index] = axis; corrected_sensor[index] = sensor
        corrected_tangent[index] = tangent; corrected_yaw[index] = yaw
        corrections.append(FramePoseCorrection(
            global_frame_index=int(original.global_frame_indices[index]), traversal_id=traversal_id,
            local_frame_index=int(original.local_frame_indices[index]), original_arc_m=source_arc,
            corrected_arc_m=final_arc,
        ))

    corrected = WorldUniqueFramePoses(
        global_frame_indices=original.global_frame_indices.copy(), traversal_ids=original.traversal_ids,
        local_frame_indices=original.local_frame_indices.copy(), arc_m=corrected_arc,
        axis_xyz_m=corrected_axis, sensor_xyz_m=corrected_sensor,
        tangent_world_xyz=corrected_tangent, yaw_deg=corrected_yaw,
    )
    return corrected, tuple(corrections)


def world_frame_poses_with_corrections(
    *,
    parent_id: str,
    traversal_manifest: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    corrections: Sequence[FramePoseCorrection],
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> WorldUniqueFramePoses:
    """Reproduce a sealed set of arc-only frame corrections exactly."""

    original = world_unique_frame_poses(
        parent_id=parent_id, traversal_manifest=traversal_manifest, graph=graph,
        spline_document=spline_document, geometry_parameters=geometry_parameters,
        spacing_m=spacing_m, history_frames=history_frames,
    )
    relevant = tuple(value for value in corrections if value.traversal_id.startswith(f"{parent_id}:"))
    if len({value.global_frame_index for value in relevant}) != len(relevant):
        raise ValueError("frame corrections contain duplicate global identities")
    if any(value.inward_shift_m <= 0.0 for value in relevant):
        raise ValueError("frame corrections must change arc length")
    index_by_global = {
        int(value): index for index, value in enumerate(original.global_frame_indices)
    }
    nodes = {str(node["id"]): np.asarray(node["xyz"], dtype=np.float64) for node in graph["nodes"]}
    splines = {
        str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    edges = {str(edge["id"]): edge for edge in graph["edges"]}
    records_by_traversal = {
        str(record["traversal_id"]): record
        for record in traversal_manifest
        if str(record["parent_id"]) == parent_id
    }
    samplers: dict[str, PolylineGeometrySampler] = {}
    arc_m = original.arc_m.copy(); axis_xyz_m = original.axis_xyz_m.copy()
    sensor_xyz_m = original.sensor_xyz_m.copy(); tangent_xyz = original.tangent_world_xyz.copy()
    yaw_deg = original.yaw_deg.copy()
    fta_distance_m = float(geometry_parameters["fta_distance_m"])
    for correction in relevant:
        if correction.global_frame_index not in index_by_global:
            raise ValueError("frame correction is outside the parent shard")
        index = index_by_global[correction.global_frame_index]
        if (
            original.traversal_ids[index] != correction.traversal_id
            or int(original.local_frame_indices[index]) != correction.local_frame_index
            or abs(float(original.arc_m[index]) - correction.original_arc_m) > 1e-12
        ):
            raise RuntimeError("sealed frame correction source identity drift")
        sampler = samplers.get(correction.traversal_id)
        if sampler is None:
            record = records_by_traversal[correction.traversal_id]
            direction_index = int(correction.traversal_id.rsplit(":d", 1)[1])
            points, _, _, _ = oriented_edge_polyline(
                edge=edges[str(record["edge_id"])], nodes=nodes, splines=splines,
                reverse=bool(direction_index),
            )
            sampler = PolylineGeometrySampler(points)
            samplers[correction.traversal_id] = sampler
        corrected_arc = float(correction.corrected_arc_m)
        if corrected_arc < 0.0 or corrected_arc > sampler.length_m:
            raise ValueError("corrected arc lies outside traversal")
        axis = sampler.interpolate(corrected_arc)
        tangent = np.asarray(
            sampler.target(corrected_arc, tunnel_radius_m=1.0).axis, dtype=np.float64
        )
        sensor = sensor_origin_from_axis(
            axis, fta_distance_m=fta_distance_m,
            sensor_height_above_floor_m=LIDAR_HEIGHT_ABOVE_FLOOR_M,
        )
        arc_m[index] = corrected_arc; axis_xyz_m[index] = axis; sensor_xyz_m[index] = sensor
        tangent_xyz[index] = tangent
        yaw_deg[index] = math.degrees(math.atan2(float(tangent[1]), float(tangent[0]))) % 360.0
    for traversal_id in sorted(set(original.traversal_ids)):
        indices = np.flatnonzero(np.asarray(original.traversal_ids, dtype=object) == traversal_id)
        if len(indices) > 1 and np.any(np.diff(arc_m[indices]) <= 1e-9):
            raise RuntimeError("sealed frame correction collapses traversal order")
    return WorldUniqueFramePoses(
        global_frame_indices=original.global_frame_indices.copy(), traversal_ids=original.traversal_ids,
        local_frame_indices=original.local_frame_indices.copy(), arc_m=arc_m,
        axis_xyz_m=axis_xyz_m, sensor_xyz_m=sensor_xyz_m,
        tangent_world_xyz=tangent_xyz, yaw_deg=yaw_deg,
    )


def cast_unique_frame_ranges(
    poses: WorldUniqueFramePoses,
    *,
    cast_hit_distances: FullScanBatchCast,
    batch_frames: int = 32,
    near_range_m: float = NEAR_RANGE_M,
    maximum_range_m: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast the frozen 16x720 scan contract in bounded frame batches."""

    if batch_frames < 1:
        raise ValueError("batch_frames must be positive")
    local = lidar_local_directions().astype(np.float64)
    count = len(poses.global_frame_indices)
    ranges = np.empty((count, *local.shape[:2]), dtype=np.float32)
    valid = np.empty((count, *local.shape[:2]), dtype=np.uint8)
    for start in range(0, count, batch_frames):
        stop = min(start + batch_frames, count)
        yaw = np.radians(poses.yaw_deg[start:stop])
        cosine = np.cos(yaw)[:, None, None]
        sine = np.sin(yaw)[:, None, None]
        directions = np.empty((stop - start, *local.shape), dtype=np.float64)
        directions[..., 0] = cosine * local[None, ..., 0] - sine * local[None, ..., 1]
        directions[..., 1] = sine * local[None, ..., 0] + cosine * local[None, ..., 1]
        directions[..., 2] = local[None, ..., 2]
        hits = np.asarray(
            cast_hit_distances(poses.sensor_xyz_m[start:stop], directions),
            dtype=np.float64,
        )
        if hits.shape != directions.shape[:-1]:
            raise ValueError("full-scan cast returned the wrong shape")
        mask = np.isfinite(hits) & (hits >= near_range_m) & (hits <= maximum_range_m)
        ranges[start:stop] = np.where(mask, hits, maximum_range_m).astype(np.float32)
        valid[start:stop] = mask.astype(np.uint8)
    if not np.all(np.isfinite(ranges)):
        raise RuntimeError("range export contains non-finite values")
    return ranges, valid


__all__ = [
    "FullScanBatchCast",
    "FramePoseCorrection",
    "WorldUniqueFramePoses",
    "cast_unique_frame_ranges",
    "world_unique_frame_poses",
    "world_finite_union_qualified_frame_poses",
    "world_frame_poses_with_corrections",
]
