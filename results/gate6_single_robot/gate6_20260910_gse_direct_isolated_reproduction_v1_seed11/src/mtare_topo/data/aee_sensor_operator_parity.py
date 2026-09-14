"""Same-pose AEE observation-operator parity primitives.

The module is deliberately independent of model inference and of the AEE
teacher.  It constructs the frozen 16x720 sensor rays, casts them against an
already resolved DAE triangle mesh, and reports real-versus-ideal observation
differences without inventing returns or repairing either source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    AZIMUTH_COLUMNS,
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
)

AEE_HORIZONTAL_SAMPLES = 350
AEE_GAZEBO_MIN_AZIMUTH_RAD = -np.pi
AEE_GAZEBO_MAX_AZIMUTH_RAD = np.pi
AEE_GAZEBO_RAW_MAX_RANGE_M = 130.0


@dataclass(frozen=True)
class AeeOrganizedSourceAudit:
    raw_records: int
    raw_finite_returns: int
    raw_no_returns: int
    model_valid_returns: int
    model_near_rejections: int
    model_far_rejections: int
    rings: int
    azimuth_samples: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def organized_aee_gazebo_points_to_ranges(
    points_xyz_sensor_m: np.ndarray,
    ring_index: np.ndarray,
    *,
    near_range_m: float = NEAR_RANGE_M,
    max_range_m: float = MAX_RANGE_M,
    raw_invalid_range_m: float = AEE_GAZEBO_RAW_MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, AeeOrganizedSourceAudit]:
    """Decode the frozen organized PointCloud2 layout into [16,350].

    The plugin writes records azimuth-major and ring-minor, then publishes
    ``width=16`` and ``height=350``.  Invalid physical returns are all-NaN
    xyz records but still retain their ring field.  Raw provenance and the
    model's 0.3--50 m view are returned separately.
    """

    points = np.asarray(points_xyz_sensor_m, dtype=np.float64)
    rings = np.asarray(ring_index)
    expected_shape = (AEE_HORIZONTAL_SAMPLES, len(ELEVATION_DEG))
    if points.shape != expected_shape + (3,) or rings.shape != expected_shape:
        raise ValueError("AEE organized points must be [350,16,3] with rings [350,16]")
    if near_range_m < 0 or max_range_m <= near_range_m or raw_invalid_range_m < max_range_m:
        raise ValueError("invalid raw/model range contract")
    finite_components = np.isfinite(points)
    all_finite = np.all(finite_components, axis=2)
    all_nonfinite = np.all(~finite_components, axis=2)
    if not np.all(all_finite | all_nonfinite):
        raise ValueError("AEE invalid xyz record must contain three nonfinite components")
    expected_rings = np.broadcast_to(
        np.arange(len(ELEVATION_DEG), dtype=np.int64)[None, :], expected_shape
    )
    if not np.array_equal(rings.astype(np.int64), expected_rings):
        raise ValueError("AEE organized ring order drifted from azimuth-major/ring-minor")

    distance = np.linalg.norm(np.where(all_finite[..., None], points, 0.0), axis=2)
    near_rejected = all_finite & (distance < near_range_m)
    far_rejected = all_finite & (distance > max_range_m)
    model_valid = all_finite & ~near_rejected & ~far_rejected
    raw_range = np.where(all_finite, distance, raw_invalid_range_m).T.astype(np.float32)
    raw_valid = all_finite.T.astype(np.uint8)
    model_range = np.where(model_valid, distance, max_range_m).T.astype(np.float32)
    model_valid_out = model_valid.T.astype(np.uint8)
    audit = AeeOrganizedSourceAudit(
        raw_records=int(points.shape[0] * points.shape[1]),
        raw_finite_returns=int(np.count_nonzero(all_finite)),
        raw_no_returns=int(np.count_nonzero(~all_finite)),
        model_valid_returns=int(np.count_nonzero(model_valid)),
        model_near_rejections=int(np.count_nonzero(near_rejected)),
        model_far_rejections=int(np.count_nonzero(far_rejected)),
        rings=len(ELEVATION_DEG),
        azimuth_samples=AEE_HORIZONTAL_SAMPLES,
    )
    return raw_range, raw_valid, model_range, model_valid_out, audit


def aee_gazebo_source_azimuth_rad(
    source_azimuth_samples: int = AEE_HORIZONTAL_SAMPLES,
) -> np.ndarray:
    """Return the frozen Gazebo source angles, including both endpoints."""

    if source_azimuth_samples < 3:
        raise ValueError("AEE Gazebo source requires at least three samples")
    return np.linspace(
        AEE_GAZEBO_MIN_AZIMUTH_RAD,
        AEE_GAZEBO_MAX_AZIMUTH_RAD,
        source_azimuth_samples,
        endpoint=True,
        dtype=np.float64,
    )


def aee_gazebo_lidar_directions_sensor(
    elevation_deg: np.ndarray = ELEVATION_DEG,
) -> np.ndarray:
    """Return the exact frozen 16x350 source rays in sensor coordinates."""

    elevation = np.asarray(elevation_deg, dtype=np.float64)
    if (
        elevation.shape != (len(ELEVATION_DEG),)
        or not np.all(np.isfinite(elevation))
        or np.any(np.diff(elevation) <= 0)
    ):
        raise ValueError("AEE elevation rows must be one ordered 16-vector")
    el = np.deg2rad(elevation)[:, None]
    az = aee_gazebo_source_azimuth_rad()[None, :]
    cos_el = np.cos(el)
    return np.stack(
        (
            np.broadcast_to(cos_el * np.cos(az), (len(elevation), AEE_HORIZONTAL_SAMPLES)),
            np.broadcast_to(cos_el * np.sin(az), (len(elevation), AEE_HORIZONTAL_SAMPLES)),
            np.broadcast_to(np.sin(el), (len(elevation), AEE_HORIZONTAL_SAMPLES)),
        ),
        axis=-1,
    )


def canonicalize_aee_gazebo_azimuth(
    source_range_m: np.ndarray,
    source_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge only the duplicated +/-pi beam and sort unique directions.

    All 350 raw records remain mandatory dataset provenance.  This derived
    view has 349 unique directions.  Duplicate valid endpoint returns keep
    the nearer range, matching the existing first-return rasterizer.
    """

    ranges = np.asarray(source_range_m, dtype=np.float32)
    valid = np.asarray(source_valid_mask, dtype=bool)
    if (
        ranges.shape != valid.shape
        or ranges.ndim < 2
        or ranges.shape[-1] != AEE_HORIZONTAL_SAMPLES
    ):
        raise ValueError("AEE source range/valid must share [..., 350] shape")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("AEE source ranges must be finite after NaN conversion")

    unique_range = ranges[..., :-1].copy()
    unique_valid = valid[..., :-1].copy()
    first_range = ranges[..., 0]
    last_range = ranges[..., -1]
    first_valid = valid[..., 0]
    last_valid = valid[..., -1]
    unique_valid[..., 0] = first_valid | last_valid
    unique_range[..., 0] = np.where(
        first_valid & last_valid,
        np.minimum(first_range, last_range),
        np.where(first_valid, first_range, last_range),
    )

    physical_angle = np.mod(aee_gazebo_source_azimuth_rad()[:-1], 2.0 * np.pi)
    order = np.argsort(physical_angle, kind="stable")
    return (
        unique_range[..., order].astype(np.float32, copy=False),
        unique_valid[..., order].astype(np.uint8, copy=False),
        physical_angle[order],
    )


def resample_aee_gazebo_organized_azimuth(
    source_range_m: np.ndarray,
    source_valid_mask: np.ndarray,
    *,
    target_azimuth_columns: int = AZIMUTH_COLUMNS,
    invalid_range_m: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample exact AEE Gazebo angles without inventing missing returns."""

    if target_azimuth_columns < 1 or not np.isfinite(invalid_range_m) or invalid_range_m <= 0:
        raise ValueError("invalid target raster contract")
    ranges, valid_u8, source_angle = canonicalize_aee_gazebo_azimuth(
        source_range_m, source_valid_mask
    )
    valid = valid_u8.astype(bool)
    target_angle = (
        2.0 * np.pi * np.arange(target_azimuth_columns, dtype=np.float64)
        / target_azimuth_columns
    )
    right = np.searchsorted(source_angle, target_angle, side="left") % len(source_angle)
    left = (right - 1) % len(source_angle)
    left_angle = source_angle[left].copy()
    right_angle = source_angle[right].copy()
    left_angle = np.where(left_angle > target_angle, left_angle - 2.0 * np.pi, left_angle)
    right_angle = np.where(right_angle < target_angle, right_angle + 2.0 * np.pi, right_angle)
    span = right_angle - left_angle
    if np.any(span <= 0):
        raise RuntimeError("AEE physical-angle bracketing is not strictly positive")
    fraction = (target_angle - left_angle) / span
    tolerance = 8.0 * np.finfo(np.float64).eps
    exact_left = np.isclose(target_angle, left_angle, rtol=0.0, atol=tolerance)
    exact_right = np.isclose(target_angle, right_angle, rtol=0.0, atol=tolerance)

    left_range = ranges[..., left]
    right_range = ranges[..., right]
    left_valid = valid[..., left]
    right_valid = valid[..., right]
    target_valid = np.where(
        exact_left,
        left_valid,
        np.where(exact_right, right_valid, left_valid & right_valid),
    )
    interpolated = left_range + (right_range - left_range) * fraction.astype(np.float32)
    target_range = np.where(target_valid, interpolated, np.float32(invalid_range_m)).astype(np.float32)
    return target_range, target_valid.astype(np.uint8)


def resample_organized_azimuth(
    source_range_m: np.ndarray,
    source_valid_mask: np.ndarray,
    *,
    target_azimuth_columns: int = AZIMUTH_COLUMNS,
    invalid_range_m: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a half-open circular scan without inventing missing returns.

    The last input dimension is the physical source-beam identity (350 for the
    frozen AEE VLP-16).  A target sample is valid only when the exact source
    beam is valid or, between beams, when both bracketing source beams are
    valid.  Consequently a physical no-return beam remains a no-return wedge;
    it cannot be confused with an empty column created by 350-to-720
    rasterization.  Formal datasets must retain the source arrays alongside
    the resampled arrays as provenance.  Raw AEE Gazebo messages instead use
    a closed ``[-pi,+pi]`` grid and must call
    :func:`resample_aee_gazebo_organized_azimuth`.
    """

    ranges = np.asarray(source_range_m, dtype=np.float32)
    valid = np.asarray(source_valid_mask, dtype=bool)
    if ranges.shape != valid.shape or ranges.ndim < 2 or ranges.shape[-1] < 2:
        raise ValueError("source range/valid must share [..., azimuth_samples] shape")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("source ranges must be finite after NaN-to-invalid conversion")
    if target_azimuth_columns < 1 or not np.isfinite(invalid_range_m) or invalid_range_m <= 0:
        raise ValueError("invalid target raster contract")

    source_columns = ranges.shape[-1]
    source_coordinate = (
        np.arange(target_azimuth_columns, dtype=np.float64) * source_columns / target_azimuth_columns
    )
    left = np.floor(source_coordinate).astype(np.int64) % source_columns
    fraction = source_coordinate - np.floor(source_coordinate)
    exact = np.isclose(fraction, 0.0, rtol=0.0, atol=8.0 * np.finfo(np.float64).eps)
    right = (left + 1) % source_columns

    left_range = ranges[..., left]
    right_range = ranges[..., right]
    left_valid = valid[..., left]
    right_valid = valid[..., right]
    target_valid = left_valid & (exact | right_valid)
    interpolated = left_range + (right_range - left_range) * fraction.astype(np.float32)
    target_range = np.where(target_valid, interpolated, np.float32(invalid_range_m)).astype(np.float32)
    return target_range, target_valid.astype(np.uint8)


def fill_unmeasured_azimuth_gaps(
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    source_azimuth_samples: int = AEE_HORIZONTAL_SAMPLES,
    target_azimuth_columns: int = AZIMUTH_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded diagnostic interpolation for legacy unorganized exports.

    The maximum filled run is derived from the two frozen raster sizes:
    ``ceil(target/source)-1``.  Longer runs remain invalid because they may be
    genuine no-return sectors.  Existing measurements are never changed.

    A legacy valid mask cannot distinguish an unmeasured raster column from a
    physical no-return beam.  This helper is therefore suitable only for the
    sealed same-pose diagnostic.  Formal corrective data must use
    :func:`resample_organized_azimuth` with preserved source-beam identity.
    """

    ranges = np.asarray(range_m, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if ranges.shape != valid.shape or ranges.ndim < 2 or ranges.shape[-1] != target_azimuth_columns:
        raise ValueError("range/valid must share [..., target_azimuth_columns] shape")
    if not np.all(np.isfinite(ranges)) or source_azimuth_samples < 1 or source_azimuth_samples > target_azimuth_columns:
        raise ValueError("invalid finite range or source raster contract")
    maximum_missing = int(np.ceil(target_azimuth_columns / source_azimuth_samples)) - 1
    result_range = ranges.copy()
    result_valid = valid.copy()
    flat_range = result_range.reshape(-1, target_azimuth_columns)
    flat_valid = result_valid.reshape(-1, target_azimuth_columns)
    for row_range, row_valid in zip(flat_range, flat_valid, strict=True):
        measured = np.flatnonzero(row_valid)
        if len(measured) < 2:
            continue
        for left, right in zip(measured, np.roll(measured, -1), strict=True):
            span = (int(right) - int(left)) % target_azimuth_columns
            missing = span - 1
            if missing < 1 or missing > maximum_missing:
                continue
            delta = float(row_range[right] - row_range[left])
            for step in range(1, span):
                column = (int(left) + step) % target_azimuth_columns
                row_range[column] = row_range[left] + delta * (step / span)
                row_valid[column] = True
    return result_range, result_valid.astype(np.uint8)


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    """Select endpoints and integer-even interior samples without randomness."""

    if size < 1 or count < 1 or count > size:
        raise ValueError("sample count must be in [1, size]")
    if count == 1:
        return np.asarray([0], dtype=np.int64)
    result = np.asarray([(index * (size - 1)) // (count - 1) for index in range(count)], dtype=np.int64)
    if len(np.unique(result)) != count or result[0] != 0 or result[-1] != size - 1:
        raise RuntimeError("integer-even selection contract failed")
    return result


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("orientation must be one finite xyzw quaternion")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("orientation quaternion must have nonzero norm")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def lidar_directions_sensor(
    elevation_deg: np.ndarray = ELEVATION_DEG,
    azimuth_columns: int = AZIMUTH_COLUMNS,
) -> np.ndarray:
    """Return unit directions in the sensor frame as [rings, columns, xyz]."""

    elevation = np.asarray(elevation_deg, dtype=np.float64)
    if elevation.ndim != 1 or not len(elevation) or not np.all(np.isfinite(elevation)):
        raise ValueError("elevation rows must be a finite vector")
    if np.any(np.diff(elevation) <= 0) or azimuth_columns < 1:
        raise ValueError("invalid ordered elevation/azimuth grid")
    el = np.deg2rad(elevation)[:, None]
    az = (2.0 * np.pi * np.arange(azimuth_columns, dtype=np.float64) / azimuth_columns)[None, :]
    cos_el = np.cos(el)
    return np.stack(
        (
            np.broadcast_to(cos_el * np.cos(az), (len(elevation), azimuth_columns)),
            np.broadcast_to(cos_el * np.sin(az), (len(elevation), azimuth_columns)),
            np.broadcast_to(np.sin(el), (len(elevation), azimuth_columns)),
        ),
        axis=-1,
    )


def world_rays(sensor_xyz_m: Sequence[float], sensor_orientation_xyzw: Sequence[float]) -> np.ndarray:
    origin = np.asarray(sensor_xyz_m, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("sensor origin must contain three finite values")
    rotation = quaternion_xyzw_to_matrix(sensor_orientation_xyzw)
    sensor_directions = lidar_directions_sensor()
    directions = sensor_directions.reshape(-1, 3) @ rotation.T
    origins = np.broadcast_to(origin, directions.shape)
    return np.concatenate((origins, directions), axis=1).astype(np.float32)


def build_open3d_scene(vertices_xyz_m: np.ndarray, triangle_vertex_indices: np.ndarray, o3d: Any) -> Any:
    vertices = np.asarray(vertices_xyz_m, dtype=np.float32)
    triangles = np.asarray(triangle_vertex_indices, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be finite [N,3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangles must be nonempty [M,3]")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangle index outside vertex array")
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def cast_ideal_range(scene: Any, sensor_xyz_m: Sequence[float], sensor_orientation_xyzw: Sequence[float], o3d: Any) -> tuple[np.ndarray, np.ndarray]:
    rays = world_rays(sensor_xyz_m, sensor_orientation_xyzw)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(len(ELEVATION_DEG), AZIMUTH_COLUMNS)
    valid = np.isfinite(hit) & (hit >= NEAR_RANGE_M) & (hit <= MAX_RANGE_M)
    ranges = np.where(valid, hit, MAX_RANGE_M).astype(np.float32)
    return ranges, valid.astype(np.uint8)


@dataclass(frozen=True)
class ObservationParityMetrics:
    cells: int
    real_valid: int
    ideal_valid: int
    both_valid: int
    real_only: int
    ideal_only: int
    valid_agreement: float
    valid_iou: float
    common_valid_mae_m: float | None
    common_valid_rmse_m: float | None
    common_valid_bias_real_minus_ideal_m: float | None
    common_valid_p95_m: float | None
    common_valid_p99_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_parity_metrics(
    real_range_m: np.ndarray,
    real_valid_mask: np.ndarray,
    ideal_range_m: np.ndarray,
    ideal_valid_mask: np.ndarray,
) -> ObservationParityMetrics:
    real_range = np.asarray(real_range_m, dtype=np.float64)
    ideal_range = np.asarray(ideal_range_m, dtype=np.float64)
    real_valid = np.asarray(real_valid_mask, dtype=bool)
    ideal_valid = np.asarray(ideal_valid_mask, dtype=bool)
    if real_range.shape != ideal_range.shape or real_valid.shape != real_range.shape or ideal_valid.shape != real_range.shape:
        raise ValueError("real and ideal range/valid arrays must share one shape")
    if not np.all(np.isfinite(real_range)) or not np.all(np.isfinite(ideal_range)):
        raise ValueError("range arrays must be finite")
    both = real_valid & ideal_valid
    union = real_valid | ideal_valid
    difference = real_range[both] - ideal_range[both]
    absolute = np.abs(difference)
    return ObservationParityMetrics(
        cells=int(real_range.size),
        real_valid=int(real_valid.sum()),
        ideal_valid=int(ideal_valid.sum()),
        both_valid=int(both.sum()),
        real_only=int((real_valid & ~ideal_valid).sum()),
        ideal_only=int((ideal_valid & ~real_valid).sum()),
        valid_agreement=float(np.mean(real_valid == ideal_valid)),
        valid_iou=float(both.sum() / union.sum()) if np.any(union) else 1.0,
        common_valid_mae_m=float(absolute.mean()) if len(absolute) else None,
        common_valid_rmse_m=float(np.sqrt(np.mean(difference * difference))) if len(difference) else None,
        common_valid_bias_real_minus_ideal_m=float(difference.mean()) if len(difference) else None,
        common_valid_p95_m=float(np.quantile(absolute, 0.95)) if len(absolute) else None,
        common_valid_p99_m=float(np.quantile(absolute, 0.99)) if len(absolute) else None,
    )
