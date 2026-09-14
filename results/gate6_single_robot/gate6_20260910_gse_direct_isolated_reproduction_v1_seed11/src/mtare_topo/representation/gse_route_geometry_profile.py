"""Typed causal route-geometry profile primitives for GSE-Graph.

The objective profile may be assembled from sealed mesh/spline targets during
teacher generation.  Student-side observability is measured only from the
current 16x720 range image in the robot frame.  Keeping these two operations
separate prevents objective route geometry from entering the model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG


PROFILE_OFFSETS_M = (0.0, 5.0, 10.0, 15.0, 20.0)
PROFILE_DIMENSIONS = ("width_m", "height_m", "slope_deg", "curvature_per_m")


@dataclass(frozen=True)
class RouteGeometryProfile:
    """Objective or observed geometry indexed by forward route offset."""

    offsets_m: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_m, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if offsets.ndim != 1 or len(offsets) == 0 or np.any(np.diff(offsets) <= 0.0):
            raise ValueError("route profile offsets must be nonempty and strictly increasing")
        if values.shape != (len(offsets), len(PROFILE_DIMENSIONS)):
            raise ValueError("route profile values must have shape [offset,4]")
        if valid.shape != values.shape:
            raise ValueError("route profile validity must align with values")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("valid route profile entries must be finite")
        object.__setattr__(self, "offsets_m", offsets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class RangeProfileSupport:
    """Direct surface support and deterministic proxy geometry from one scan."""

    offsets_m: np.ndarray
    side_counts: np.ndarray
    proxy_width_height_m: np.ndarray
    center_left_up_m: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_m, dtype=np.float64)
        counts = np.asarray(self.side_counts, dtype=np.int64)
        geometry = np.asarray(self.proxy_width_height_m, dtype=np.float64)
        center = np.asarray(self.center_left_up_m, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        rows = len(offsets)
        if counts.shape != (rows, 4) or geometry.shape != (rows, 2):
            raise ValueError("range profile support shape drift")
        if center.shape != (rows, 2) or valid.shape != (rows,):
            raise ValueError("range profile center/validity shape drift")
        if np.any(counts < 0) or np.any(valid & ~np.all(np.isfinite(geometry), axis=1)):
            raise ValueError("range profile support contains invalid values")
        object.__setattr__(self, "offsets_m", offsets)
        object.__setattr__(self, "side_counts", counts)
        object.__setattr__(self, "proxy_width_height_m", geometry)
        object.__setattr__(self, "center_left_up_m", center)
        object.__setattr__(self, "valid_mask", valid)


def objective_profile_from_sequence(
    geometry: np.ndarray,
    geometry_valid_mask: np.ndarray,
    *,
    current_sequence_index: int,
    row_by_sequence_index: Mapping[int, int],
    offsets_m: Sequence[float] = PROFILE_OFFSETS_M,
    spacing_m: float = 1.0,
) -> RouteGeometryProfile:
    """Gather a teacher-only forward profile without crossing a traversal."""

    values = np.asarray(geometry, dtype=np.float64)
    validity = np.asarray(geometry_valid_mask, dtype=bool)
    offsets = np.asarray(tuple(offsets_m), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(PROFILE_DIMENSIONS):
        raise ValueError("geometry must have shape [rows,4]")
    if validity.shape != values.shape or spacing_m <= 0.0:
        raise ValueError("geometry validity or spacing contract drift")
    steps = np.rint(offsets / float(spacing_m)).astype(np.int64)
    if not np.allclose(steps * float(spacing_m), offsets, atol=1e-9, rtol=0.0):
        raise ValueError("route profile offsets must align with spatial spacing")
    output = np.full((len(offsets), len(PROFILE_DIMENSIONS)), np.nan, dtype=np.float64)
    output_valid = np.zeros_like(output, dtype=bool)
    for profile_row, step in enumerate(steps):
        source_row = row_by_sequence_index.get(int(current_sequence_index + step))
        if source_row is None:
            continue
        output[profile_row] = values[source_row]
        output_valid[profile_row] = validity[source_row] & np.isfinite(values[source_row])
    return RouteGeometryProfile(offsets, output, output_valid)


def range_image_profile_support(
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    offsets_m: Sequence[float] = PROFILE_OFFSETS_M,
    longitudinal_half_window_m: float = 2.5,
    elevation_deg: np.ndarray = ELEVATION_DEG,
) -> RangeProfileSupport:
    """Project one scan into fixed robot-forward slabs.

    A slab is supported only when first returns occur on both lateral and both
    vertical sides.  This is deliberately a visibility proof, not a learned
    estimator or a replacement for the objective mesh teacher.
    """

    ranges = np.asarray(range_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    elevations = np.asarray(elevation_deg, dtype=np.float64)
    offsets = np.asarray(tuple(offsets_m), dtype=np.float64)
    if ranges.ndim != 2 or ranges.shape != valid.shape:
        raise ValueError("range image and validity must be aligned rank-two arrays")
    if ranges.shape[0] != len(elevations) or ranges.shape[1] <= 0:
        raise ValueError("range image sensor shape drift")
    if longitudinal_half_window_m <= 0.0:
        raise ValueError("longitudinal half window must be positive")
    azimuth = np.arange(ranges.shape[1], dtype=np.float64) * (2.0 * np.pi / ranges.shape[1])
    elevation = np.radians(elevations)
    cos_elevation = np.cos(elevation)[:, None]
    forward = ranges * cos_elevation * np.cos(azimuth)[None, :]
    left = ranges * cos_elevation * np.sin(azimuth)[None, :]
    up = ranges * np.sin(elevation)[:, None]
    finite = valid & np.isfinite(ranges) & (ranges > 0.0)

    counts = np.zeros((len(offsets), 4), dtype=np.int64)
    proxy = np.full((len(offsets), 2), np.nan, dtype=np.float64)
    center = np.full((len(offsets), 2), np.nan, dtype=np.float64)
    supported = np.zeros(len(offsets), dtype=bool)
    for index, offset in enumerate(offsets):
        # The current cross-section uses the forward half of the first slab so
        # that returns behind the robot cannot enter a forward-route profile.
        lower = max(0.0, float(offset) - longitudinal_half_window_m)
        upper = float(offset) + longitudinal_half_window_m
        slab = finite & (forward >= lower) & (forward < upper)
        groups = (slab & (left > 0.0), slab & (left < 0.0), slab & (up > 0.0), slab & (up < 0.0))
        counts[index] = [int(np.count_nonzero(group)) for group in groups]
        if np.any(counts[index] == 0):
            continue
        left_extent = float(np.max(left[groups[0]]))
        right_extent = float(-np.min(left[groups[1]]))
        up_extent = float(np.max(up[groups[2]]))
        down_extent = float(-np.min(up[groups[3]]))
        proxy[index] = (left_extent + right_extent, up_extent + down_extent)
        center[index] = ((left_extent - right_extent) / 2.0, (up_extent - down_extent) / 2.0)
        supported[index] = np.all(np.isfinite(proxy[index])) & np.all(proxy[index] > 0.0)
    return RangeProfileSupport(offsets, counts, proxy, center, supported)


def reverse_consistency_error(forward: RouteGeometryProfile, reverse: RouteGeometryProfile) -> dict[str, float]:
    """Compare profiles at the same canonical locations in opposite traversal directions."""

    if not np.array_equal(forward.offsets_m, reverse.offsets_m):
        raise ValueError("reverse profile offsets differ")
    common = forward.valid_mask & reverse.valid_mask
    # Width, height and curvature are direction invariant; slope changes sign.
    transformed = reverse.values.copy()
    transformed[:, 2] *= -1.0
    names = PROFILE_DIMENSIONS
    result: dict[str, float] = {}
    for column, name in enumerate(names):
        mask = common[:, column]
        result[name] = float(np.max(np.abs(forward.values[mask, column] - transformed[mask, column]))) if np.any(mask) else float("nan")
    return result


__all__ = [
    "PROFILE_DIMENSIONS",
    "PROFILE_OFFSETS_M",
    "RangeProfileSupport",
    "RouteGeometryProfile",
    "objective_profile_from_sequence",
    "range_image_profile_support",
    "reverse_consistency_error",
]
