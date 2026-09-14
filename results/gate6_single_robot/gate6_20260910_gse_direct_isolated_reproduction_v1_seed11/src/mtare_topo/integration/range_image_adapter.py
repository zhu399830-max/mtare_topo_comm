"""Convert registered M-TARE PointCloud2 geometry to the trained 16x720 grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    AZIMUTH_COLUMNS,
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
)


@dataclass(frozen=True)
class RangeImageConversionAudit:
    input_points: int
    finite_points: int
    accepted_points: int
    unique_valid_cells: int
    duplicate_cell_returns: int
    out_of_range_points: int
    out_of_ring_points: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def registered_points_to_range_image(
    points_world: np.ndarray,
    *,
    sensor_origin_world: Sequence[float],
    sensor_orientation_xyzw: Sequence[float],
    elevation_rows_deg: np.ndarray = ELEVATION_DEG,
    azimuth_columns: int = AZIMUTH_COLUMNS,
    near_range_m: float = NEAR_RANGE_M,
    max_range_m: float = MAX_RANGE_M,
    maximum_elevation_error_deg: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, RangeImageConversionAudit]:
    """Rasterize unordered map-frame returns without using future/map state.

    Azimuth zero is robot-forward and columns increase counter-clockwise toward
    robot-left, exactly matching the synthetic training rays.  Duplicate
    returns keep the nearest range (first-return semantics).
    """

    points = np.asarray(points_world, dtype=np.float64)
    origin = np.asarray(sensor_origin_world, dtype=np.float64)
    rows = np.asarray(elevation_rows_deg, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N, 3]")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("sensor origin must contain three finite values")
    orientation = np.asarray(sensor_orientation_xyzw, dtype=np.float64)
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError("sensor orientation must contain four finite xyzw values")
    orientation_norm = float(np.linalg.norm(orientation))
    if orientation_norm <= np.finfo(np.float64).tiny:
        raise ValueError("sensor orientation quaternion must have nonzero norm")
    orientation = orientation / orientation_norm
    if len(rows) == 0 or not np.all(np.isfinite(rows)) or np.any(np.diff(rows) <= 0):
        raise ValueError("elevation rows must be finite and strictly increasing")
    if azimuth_columns <= 0 or near_range_m < 0 or max_range_m <= near_range_m:
        raise ValueError("invalid sensor range/grid contract")

    input_points = len(points)
    finite_mask = np.all(np.isfinite(points), axis=1)
    finite = points[finite_mask]
    delta_world = finite - origin
    distance = np.linalg.norm(delta_world, axis=1)
    range_mask = (distance >= near_range_m) & (distance <= max_range_m)

    x, y, z, w = orientation
    world_from_sensor = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    delta_sensor = delta_world @ world_from_sensor
    forward = delta_sensor[:, 0]
    left = delta_sensor[:, 1]
    azimuth_deg = np.degrees(np.arctan2(left, forward)) % 360.0
    safe_distance = np.maximum(distance, np.finfo(np.float64).tiny)
    elevation_deg = np.degrees(np.arcsin(np.clip(delta_sensor[:, 2] / safe_distance, -1.0, 1.0)))

    insertion = np.searchsorted(rows, elevation_deg)
    upper = np.clip(insertion, 0, len(rows) - 1)
    lower = np.clip(insertion - 1, 0, len(rows) - 1)
    choose_upper = np.abs(rows[upper] - elevation_deg) < np.abs(rows[lower] - elevation_deg)
    row_index = np.where(choose_upper, upper, lower)
    elevation_error = np.abs(rows[row_index] - elevation_deg)
    ring_mask = elevation_error <= maximum_elevation_error_deg
    accepted_mask = range_mask & ring_mask
    column_index = np.rint(azimuth_deg / 360.0 * azimuth_columns).astype(np.int64) % azimuth_columns

    image = np.full((len(rows), azimuth_columns), max_range_m, dtype=np.float32)
    valid = np.zeros((len(rows), azimuth_columns), dtype=np.uint8)
    accepted_rows = row_index[accepted_mask]
    accepted_columns = column_index[accepted_mask]
    accepted_ranges = distance[accepted_mask].astype(np.float32)
    if len(accepted_ranges):
        flat_indices = accepted_rows * azimuth_columns + accepted_columns
        np.minimum.at(image.reshape(-1), flat_indices, accepted_ranges)
        valid.reshape(-1)[np.unique(flat_indices)] = 1
    unique_cells = int(np.count_nonzero(valid))
    accepted_points = int(np.count_nonzero(accepted_mask))
    audit = RangeImageConversionAudit(
        input_points=input_points,
        finite_points=len(finite),
        accepted_points=accepted_points,
        unique_valid_cells=unique_cells,
        duplicate_cell_returns=accepted_points - unique_cells,
        out_of_range_points=int(np.count_nonzero(~range_mask)),
        out_of_ring_points=int(np.count_nonzero(range_mask & ~ring_mask)),
    )
    return image, valid, audit
