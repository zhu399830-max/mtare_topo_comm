"""Native-mesh local geometry targets for GSE-Graph.

The student never receives these rays. They are an objective teacher derived
from the same sealed perception mesh used to render the causal LiDAR input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np

from mtare_topo.teacher.gse_geometry_teacher import LocalGeometryTarget


RaycastFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
BatchRaycastFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MeshGeometryTeacherConfig:
    """Frozen physical sampling contract for one local cross-section."""

    sensor_height_above_floor_m: float = 1.0
    maximum_ray_distance_m: float = 30.0
    fan_offsets_deg: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0)
    minimum_valid_rays_per_side: int = 3
    transition_comparison_span_m: float = 5.0
    transition_width_change_m: float = 1.0
    transition_height_change_m: float = 1.0

    def __post_init__(self) -> None:
        if self.sensor_height_above_floor_m <= 0.0:
            raise ValueError("sensor height must be positive")
        if self.maximum_ray_distance_m <= 0.0:
            raise ValueError("maximum ray distance must be positive")
        if len(self.fan_offsets_deg) < 3 or 0.0 not in self.fan_offsets_deg:
            raise ValueError("ray fan must contain zero and at least three offsets")
        if tuple(sorted(self.fan_offsets_deg)) != self.fan_offsets_deg:
            raise ValueError("ray fan offsets must be sorted")
        if any(abs(value) >= 30.0 for value in self.fan_offsets_deg):
            raise ValueError("ray fan offsets must remain within 30 degrees")
        if not 1 <= self.minimum_valid_rays_per_side <= len(self.fan_offsets_deg):
            raise ValueError("invalid minimum valid ray count")
        if self.transition_comparison_span_m <= 0.0:
            raise ValueError("transition comparison span must be positive")
        if self.transition_width_change_m <= 0.0 or self.transition_height_change_m <= 0.0:
            raise ValueError("transition changes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeshGeometryMeasurement:
    left_m: float
    right_m: float
    up_m: float
    down_m: float
    width_m: float
    height_m: float
    valid_rays: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        values = (self.left_m, self.right_m, self.up_m, self.down_m, self.width_m, self.height_m)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("mesh geometry measurement must be positive and finite")
        if not math.isclose(self.width_m, self.left_m + self.right_m, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("width must equal left plus right")
        if not math.isclose(self.height_m, self.up_m + self.down_m, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("height must equal up plus down")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sensor_origin_from_axis(
    axis_xyz_m: Sequence[float],
    *,
    fta_distance_m: float,
    sensor_height_above_floor_m: float = 1.0,
) -> np.ndarray:
    """Use the sealed Cano floor-to-axis convention used by LiDAR export."""

    axis = np.asarray(axis_xyz_m, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis_xyz_m must contain three finite values")
    if not math.isfinite(float(fta_distance_m)):
        raise ValueError("fta_distance_m must be finite")
    if not math.isfinite(float(sensor_height_above_floor_m)) or sensor_height_above_floor_m <= 0.0:
        raise ValueError("sensor height must be positive and finite")
    sensor = axis.copy()
    sensor[2] += float(fta_distance_m) + float(sensor_height_above_floor_m)
    return sensor


def local_cross_section_rays(
    tangent_xyz: Sequence[float],
    *,
    fan_offsets_deg: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice, slice]]:
    """Return world-frame rays and their primary-axis projection factors."""

    tangent = np.asarray(tangent_xyz, dtype=np.float64)
    if tangent.shape != (3,) or not np.all(np.isfinite(tangent)):
        raise ValueError("tangent_xyz must contain three finite values")
    horizontal = tangent.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-8:
        raise ValueError("vertical traversal has no horizontal cross-section frame")
    forward = horizontal / norm
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    left = np.cross(up, forward)
    offsets = np.radians(np.asarray(tuple(fan_offsets_deg), dtype=np.float64))
    if offsets.ndim != 1 or len(offsets) == 0 or not np.all(np.isfinite(offsets)):
        raise ValueError("fan offsets must be a nonempty finite sequence")

    groups: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    primary_axes = (left, -left, up, -up)
    for primary in primary_axes:
        directions = np.cos(offsets)[:, None] * primary + np.sin(offsets)[:, None] * forward
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        groups.append(directions)
        projections.append(directions @ primary)
    rays = np.vstack(groups)
    factors = np.concatenate(projections)
    count = len(offsets)
    slices = tuple(slice(index * count, (index + 1) * count) for index in range(4))
    return rays, factors, slices  # type: ignore[return-value]


def measure_mesh_geometry(
    *,
    origin_xyz_m: Sequence[float],
    tangent_xyz: Sequence[float],
    cast_distances: RaycastFunction,
    config: MeshGeometryTeacherConfig | None = None,
) -> MeshGeometryMeasurement:
    """Measure robust local width and height from a sealed triangle mesh."""

    cfg = config or MeshGeometryTeacherConfig()
    origin = np.asarray(origin_xyz_m, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_xyz_m must contain three finite values")
    rays, projection_factors, groups = local_cross_section_rays(
        tangent_xyz,
        fan_offsets_deg=cfg.fan_offsets_deg,
    )
    distances = np.asarray(cast_distances(origin, rays), dtype=np.float64)
    if distances.shape != (len(rays),):
        raise ValueError("raycaster returned the wrong number of distances")
    valid = np.isfinite(distances) & (distances > 0.0) & (distances <= cfg.maximum_ray_distance_m)
    projected = distances * projection_factors
    side_values: list[float] = []
    side_counts: list[int] = []
    for group in groups:
        group_valid = valid[group]
        count = int(np.sum(group_valid))
        if count < cfg.minimum_valid_rays_per_side:
            raise ValueError("mesh cross-section has insufficient finite ray support")
        side_counts.append(count)
        side_values.append(float(np.median(projected[group][group_valid])))
    left, right, up, down = side_values
    return MeshGeometryMeasurement(
        left_m=left,
        right_m=right,
        up_m=up,
        down_m=down,
        width_m=left + right,
        height_m=up + down,
        valid_rays=tuple(side_counts),  # type: ignore[arg-type]
    )


def measure_mesh_geometry_batch(
    *,
    origins_xyz_m: np.ndarray,
    tangents_xyz: np.ndarray,
    cast_distances: BatchRaycastFunction,
    config: MeshGeometryTeacherConfig | None = None,
) -> tuple[list[MeshGeometryMeasurement | None], np.ndarray]:
    """Vectorized cross-section measurement with fail-closed per-frame output."""

    cfg = config or MeshGeometryTeacherConfig()
    origins = np.asarray(origins_xyz_m, dtype=np.float64)
    tangents = np.asarray(tangents_xyz, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3 or tangents.shape != origins.shape:
        raise ValueError("origins and tangents must have aligned shape [N,3]")
    if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(tangents)):
        raise ValueError("origins and tangents must be finite")
    if len(origins) == 0:
        return [], np.zeros(0, dtype=bool)
    ray_rows: list[np.ndarray] = []
    factor_rows: list[np.ndarray] = []
    groups: tuple[slice, slice, slice, slice] | None = None
    for tangent in tangents:
        rays, factors, current_groups = local_cross_section_rays(
            tangent,
            fan_offsets_deg=cfg.fan_offsets_deg,
        )
        ray_rows.append(rays)
        factor_rows.append(factors)
        groups = current_groups
    directions = np.stack(ray_rows)
    factors = np.stack(factor_rows)
    distances = np.asarray(cast_distances(origins, directions), dtype=np.float64)
    if distances.shape != directions.shape[:2]:
        raise ValueError("batch raycaster returned the wrong distance shape")
    valid = np.isfinite(distances) & (distances > 0.0) & (distances <= cfg.maximum_ray_distance_m)
    projected = distances * factors
    assert groups is not None
    results: list[MeshGeometryMeasurement | None] = []
    complete = np.zeros(len(origins), dtype=bool)
    for row_index in range(len(origins)):
        values: list[float] = []
        counts: list[int] = []
        for group in groups:
            group_valid = valid[row_index, group]
            count = int(np.sum(group_valid))
            counts.append(count)
            if count < cfg.minimum_valid_rays_per_side:
                break
            values.append(float(np.median(projected[row_index, group][group_valid])))
        if len(values) != 4:
            results.append(None)
            continue
        left, right, up, down = values
        results.append(
            MeshGeometryMeasurement(
                left_m=left,
                right_m=right,
                up_m=up,
                down_m=down,
                width_m=left + right,
                height_m=up + down,
                valid_rays=tuple(counts),  # type: ignore[arg-type]
            )
        )
        complete[row_index] = True
    return results, complete


def geometry_transition_mask(
    width_m: Sequence[float],
    height_m: Sequence[float],
    *,
    spacing_m: float,
    valid_mask: Sequence[bool] | None = None,
    config: MeshGeometryTeacherConfig | None = None,
) -> np.ndarray:
    """Label sustained local cross-section changes using adjacent medians."""

    cfg = config or MeshGeometryTeacherConfig()
    width = np.asarray(width_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    if width.ndim != 1 or height.shape != width.shape or len(width) == 0:
        raise ValueError("width and height profiles must be nonempty aligned vectors")
    if not math.isfinite(float(spacing_m)) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    valid = np.ones(len(width), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != width.shape:
        raise ValueError("valid mask must align with geometry profiles")
    valid &= np.isfinite(width) & np.isfinite(height) & (width > 0.0) & (height > 0.0)
    span = max(1, int(round(cfg.transition_comparison_span_m / float(spacing_m))))
    labels = np.zeros(len(width), dtype=bool)
    for index in range(span, len(width) - span):
        before = slice(index - span, index)
        after = slice(index + 1, index + 1 + span)
        if not np.all(valid[before]) or not np.all(valid[after]):
            continue
        width_change = abs(float(np.median(width[after])) - float(np.median(width[before])))
        height_change = abs(float(np.median(height[after])) - float(np.median(height[before])))
        labels[index] = bool(
            width_change >= cfg.transition_width_change_m
            or height_change >= cfg.transition_height_change_m
        )
    return labels


def apply_mesh_cross_section(
    spline_target: LocalGeometryTarget,
    measurement: MeshGeometryMeasurement,
) -> LocalGeometryTarget:
    """Keep objective spline kinematics and replace the radius proxy geometry."""

    return LocalGeometryTarget(
        axis=spline_target.axis,
        width_m=measurement.width_m,
        height_m=measurement.height_m,
        slope_deg=spline_target.slope_deg,
        curvature_per_m=spline_target.curvature_per_m,
        heading_change_deg=spline_target.heading_change_deg,
    )


__all__ = [
    "BatchRaycastFunction",
    "MeshGeometryMeasurement",
    "MeshGeometryTeacherConfig",
    "RaycastFunction",
    "apply_mesh_cross_section",
    "geometry_transition_mask",
    "local_cross_section_rays",
    "measure_mesh_geometry",
    "measure_mesh_geometry_batch",
    "sensor_origin_from_axis",
]
