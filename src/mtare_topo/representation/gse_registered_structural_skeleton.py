"""Causal ego-motion registration and typed local skeletons for ERCSS.

Only relative translation and yaw are exposed by this module.  Absolute poses
are accepted solely to form relative transforms and are never retained in the
returned student representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions


SKELETON_GEOMETRY_NAMES = (
    "width_m",
    "height_m",
    "slope_deg",
    "curvature_per_m",
)


def _rotation_z(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _wrap_radians(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class RelativePoseWindow:
    """Past-to-current poses expressed in the current sensor frame."""

    translation_current_m: np.ndarray
    yaw_current_rad: np.ndarray
    current_index: int

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_current_m, dtype=np.float64)
        yaw = np.asarray(self.yaw_current_rad, dtype=np.float64)
        if translation.ndim != 2 or translation.shape[1] != 3:
            raise ValueError("relative translations must have shape [T,3]")
        if yaw.shape != (len(translation),) or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(yaw)):
            raise ValueError("relative pose window must be finite and aligned")
        if self.current_index != len(translation) - 1:
            raise ValueError("ERCSS input must end at the current frame; future frames are forbidden")
        if not np.allclose(translation[self.current_index], 0.0, atol=1e-9, rtol=0.0):
            raise ValueError("current relative translation must be zero")
        if not math.isclose(float(yaw[self.current_index]), 0.0, abs_tol=1e-9):
            raise ValueError("current relative yaw must be zero")
        object.__setattr__(self, "translation_current_m", translation)
        object.__setattr__(self, "yaw_current_rad", yaw)


@dataclass(frozen=True)
class RegisteredCausalPointSet:
    """Finite LiDAR returns and their ray origins in the current frame."""

    xyz_current_m: np.ndarray
    source_frame_index: np.ndarray
    ray_origin_current_m: np.ndarray
    range_m: np.ndarray
    relative_poses: RelativePoseWindow

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_current_m, dtype=np.float64)
        source = np.asarray(self.source_frame_index, dtype=np.int16)
        origin = np.asarray(self.ray_origin_current_m, dtype=np.float64)
        ranges = np.asarray(self.range_m, dtype=np.float64)
        rows = len(xyz)
        if xyz.shape != (rows, 3) or origin.shape != (rows, 3):
            raise ValueError("registered point coordinates must have shape [N,3]")
        if source.shape != (rows,) or ranges.shape != (rows,):
            raise ValueError("registered point metadata must align with points")
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(ranges)):
            raise ValueError("registered point set must be finite")
        if np.any(ranges <= 0.0) or np.any(source < 0) or np.any(source > self.relative_poses.current_index):
            raise ValueError("registered point metadata violates causal/range contract")
        expected_origins = self.relative_poses.translation_current_m[source]
        if not np.allclose(origin, expected_origins, atol=1e-8, rtol=0.0):
            raise ValueError("registered ray origins drift from relative odometry")
        object.__setattr__(self, "xyz_current_m", xyz)
        object.__setattr__(self, "source_frame_index", source)
        object.__setattr__(self, "ray_origin_current_m", origin)
        object.__setattr__(self, "range_m", ranges)


@dataclass(frozen=True)
class EgoConnectedStructuralSkeleton:
    """Capacity-bounded visible local tunnel skeleton used as an ERCSS target."""

    node_xyz_current_m: np.ndarray
    edge_node_indices: np.ndarray
    edge_geometry: np.ndarray
    edge_geometry_valid: np.ndarray
    ego_node_index: int
    node_identity: np.ndarray
    edge_identity: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.node_xyz_current_m, dtype=np.float64)
        edges = np.asarray(self.edge_node_indices, dtype=np.int32)
        geometry = np.asarray(self.edge_geometry, dtype=np.float64)
        valid = np.asarray(self.edge_geometry_valid, dtype=bool)
        node_identity = np.asarray(self.node_identity, dtype=np.int64)
        edge_identity = np.asarray(self.edge_identity, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0 or not np.all(np.isfinite(nodes)):
            raise ValueError("skeleton nodes must be a nonempty finite [N,3] array")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("skeleton edges must have shape [E,2]")
        if geometry.shape != (len(edges), len(SKELETON_GEOMETRY_NAMES)) or valid.shape != geometry.shape:
            raise ValueError("skeleton edge geometry shape drift")
        if node_identity.shape != (len(nodes),) or edge_identity.shape != (len(edges),):
            raise ValueError("skeleton identity arrays must align")
        if not 0 <= self.ego_node_index < len(nodes):
            raise ValueError("ego node index is outside the skeleton")
        if np.any(edges < 0) or np.any(edges >= len(nodes)) or np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("skeleton contains invalid edge endpoints")
        canonical_edges = np.sort(edges, axis=1)
        if len(canonical_edges) and len(np.unique(canonical_edges, axis=0)) != len(canonical_edges):
            raise ValueError("skeleton contains duplicate undirected edges")
        if np.any(valid & ~np.isfinite(geometry)):
            raise ValueError("valid skeleton geometry must be finite")
        visited = {int(self.ego_node_index)}
        while True:
            before = len(visited)
            for first, second in edges:
                if int(first) in visited:
                    visited.add(int(second))
                if int(second) in visited:
                    visited.add(int(first))
            if len(visited) == before:
                break
        if len(visited) != len(nodes):
            raise ValueError("teacher skeleton must be ego-connected")
        object.__setattr__(self, "node_xyz_current_m", nodes)
        object.__setattr__(self, "edge_node_indices", edges)
        object.__setattr__(self, "edge_geometry", geometry)
        object.__setattr__(self, "edge_geometry_valid", valid)
        object.__setattr__(self, "node_identity", node_identity)
        object.__setattr__(self, "edge_identity", edge_identity)


def relative_pose_window(
    sensor_xyz_world_m: np.ndarray,
    yaw_world_deg: Sequence[float],
) -> RelativePoseWindow:
    """Convert a causal absolute-pose window to relative current-frame poses."""

    xyz = np.asarray(sensor_xyz_world_m, dtype=np.float64)
    yaw_deg = np.asarray(yaw_world_deg, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("sensor poses must have shape [T,3]")
    if yaw_deg.shape != (len(xyz),) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(yaw_deg)):
        raise ValueError("sensor pose/yaw window must be finite and aligned")
    current = len(xyz) - 1
    current_rotation_inverse = _rotation_z(-math.radians(float(yaw_deg[current])))
    translation = (xyz - xyz[current]) @ current_rotation_inverse.T
    relative_yaw = _wrap_radians(np.radians(yaw_deg - yaw_deg[current]))
    translation[current] = 0.0
    relative_yaw[current] = 0.0
    return RelativePoseWindow(translation, relative_yaw, current)


def register_causal_range_window(
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    sensor_xyz_world_m: np.ndarray,
    yaw_world_deg: Sequence[float],
) -> RegisteredCausalPointSet:
    """Register past and current finite returns into the current sensor frame."""

    ranges = np.asarray(range_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if ranges.ndim != 3 or ranges.shape != valid.shape:
        raise ValueError("causal range/valid windows must be aligned [T,H,W]")
    directions = np.asarray(lidar_local_directions(), dtype=np.float64)
    if ranges.shape[1:] != directions.shape[:2]:
        raise ValueError("LiDAR raster shape differs from the frozen 16x720 sensor")
    poses = relative_pose_window(sensor_xyz_world_m, yaw_world_deg)
    if len(ranges) != len(poses.translation_current_m):
        raise ValueError("scan and pose window lengths differ")

    points: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    finite_ranges: list[np.ndarray] = []
    for frame in range(len(ranges)):
        mask = valid[frame] & np.isfinite(ranges[frame]) & (ranges[frame] > 0.0)
        selected_ranges = ranges[frame][mask]
        if len(selected_ranges) == 0:
            continue
        local_points = directions[mask] * selected_ranges[:, None]
        rotated = local_points @ _rotation_z(float(poses.yaw_current_rad[frame])).T
        frame_points = rotated + poses.translation_current_m[frame]
        points.append(frame_points)
        sources.append(np.full(len(frame_points), frame, dtype=np.int16))
        origins.append(np.repeat(poses.translation_current_m[frame][None, :], len(frame_points), axis=0))
        finite_ranges.append(selected_ranges)
    if points:
        xyz = np.concatenate(points, axis=0)
        source = np.concatenate(sources, axis=0)
        origin = np.concatenate(origins, axis=0)
        selected = np.concatenate(finite_ranges, axis=0)
    else:
        xyz = np.empty((0, 3), dtype=np.float64)
        source = np.empty((0,), dtype=np.int16)
        origin = np.empty((0, 3), dtype=np.float64)
        selected = np.empty((0,), dtype=np.float64)
    return RegisteredCausalPointSet(xyz, source, origin, selected, poses)


def deterministic_voxel_subsample(
    points: RegisteredCausalPointSet,
    *,
    voxel_size_m: float,
) -> RegisteredCausalPointSet:
    """Retain one deterministic return per current-frame voxel."""

    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("voxel size must be finite and positive")
    if len(points.xyz_current_m) == 0:
        return points
    voxel = np.floor(points.xyz_current_m / float(voxel_size_m)).astype(np.int64)
    # Prefer the current/latest observation, then the shortest range, then the
    # original stable raster order.  This makes overlaps deterministic.
    original = np.arange(len(voxel), dtype=np.int64)
    order = np.lexsort((original, points.range_m, -points.source_frame_index, voxel[:, 2], voxel[:, 1], voxel[:, 0]))
    ordered_voxel = voxel[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered_voxel[1:] != ordered_voxel[:-1], axis=1)
    keep = order[first]
    keep.sort()
    return RegisteredCausalPointSet(
        points.xyz_current_m[keep],
        points.source_frame_index[keep],
        points.ray_origin_current_m[keep],
        points.range_m[keep],
        points.relative_poses,
    )


__all__ = [
    "EgoConnectedStructuralSkeleton",
    "RegisteredCausalPointSet",
    "RelativePoseWindow",
    "SKELETON_GEOMETRY_NAMES",
    "deterministic_voxel_subsample",
    "register_causal_range_window",
    "relative_pose_window",
]
