"""Vectorized per-frame provenance summaries for P1b target materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.primitive_relation_dataset import VisiblePrimitiveWindowTargets
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


@dataclass(frozen=True)
class PrimitiveFrameSupport:
    support_ray_count: np.ndarray
    minimum_sample_index: np.ndarray
    maximum_sample_index: np.ndarray
    azimuth_support_packed: np.ndarray

    def __post_init__(self) -> None:
        primitives = len(self.support_ray_count)
        if self.minimum_sample_index.shape != (primitives,) or self.maximum_sample_index.shape != (primitives,):
            raise ValueError("primitive support extrema shape mismatch")
        if self.azimuth_support_packed.shape != (primitives, 90):
            raise ValueError("packed 720-column azimuth support must have 90 bytes per primitive")
        visible = self.support_ray_count > 0
        if np.any(self.minimum_sample_index[visible] < 0) or np.any(self.maximum_sample_index[visible] < self.minimum_sample_index[visible]):
            raise ValueError("visible primitive sample extrema are invalid")
        if np.any(self.minimum_sample_index[~visible] != -1) or np.any(self.maximum_sample_index[~visible] != -1):
            raise ValueError("invisible primitives must use -1 extrema")

    @property
    def azimuth_support(self) -> np.ndarray:
        return np.unpackbits(self.azimuth_support_packed, axis=1, count=720, bitorder="little").astype(bool)


def summarize_primitive_frame_support(
    *,
    range_m: np.ndarray,
    primitive_membership_code: np.ndarray,
    source_sets: Sequence[Sequence[int]],
    field: SweptSuperellipseProvenanceField,
    sensor_xyz_m: np.ndarray,
    yaw_deg: float,
) -> PrimitiveFrameSupport:
    """Reduce one complete frame without dropping any source membership."""

    ranges = np.asarray(range_m, dtype=np.float64)
    codes = np.asarray(primitive_membership_code)
    if ranges.shape != (16, 720) or codes.shape != (16, 720) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("range/code inputs must be 16x720")
    if not np.isfinite(ranges).all() or np.any(codes < 0) or np.any(codes >= len(source_sets)):
        raise ValueError("range/code frame contains invalid values")
    primitive_count = len(field.primitive_ids)
    normalized_sets = tuple(tuple(int(value) for value in source) for source in source_sets)
    if normalized_sets[0] != () or any(any(value < 0 or value >= primitive_count for value in source) for source in normalized_sets):
        raise ValueError("codebook source sets are invalid")
    counts = np.zeros(primitive_count, dtype=np.int32)
    low = np.full(primitive_count, -1, dtype=np.int32)
    high = np.full(primitive_count, -1, dtype=np.int32)
    azimuth = np.zeros((primitive_count, 720), dtype=np.uint8)
    directions = world_directions(lidar_local_directions().reshape(-1, 3).astype(np.float64), float(yaw_deg))
    flat_codes = codes.reshape(-1); flat_ranges = ranges.reshape(-1); sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    if sensor.shape != (3,) or not np.isfinite(sensor).all():
        raise ValueError("sensor origin must be finite xyz")
    for code in np.unique(flat_codes):
        code_value = int(code)
        if code_value == 0:
            continue
        indices = np.flatnonzero(flat_codes == code_value)
        points = sensor + flat_ranges[indices, None] * directions[indices]
        columns = np.unique(indices % 720)
        for primitive_index in normalized_sets[code_value]:
            _, sampled = field.operands[primitive_index].tree.query(points, k=1, workers=1)
            sampled = np.asarray(sampled, dtype=np.int64)
            counts[primitive_index] += len(indices)
            candidate_low, candidate_high = int(np.min(sampled)), int(np.max(sampled))
            low[primitive_index] = candidate_low if low[primitive_index] < 0 else min(int(low[primitive_index]), candidate_low)
            high[primitive_index] = max(int(high[primitive_index]), candidate_high)
            azimuth[primitive_index, columns] = 1
    packed = np.packbits(azimuth, axis=1, bitorder="little")
    return PrimitiveFrameSupport(counts, low, high, packed)


def _current_sensor_transform(points: np.ndarray, origin: np.ndarray, yaw_deg: float) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    yaw = np.radians(float(yaw_deg)); cosine, sine = np.cos(yaw), np.sin(yaw)
    result = delta.copy(); result[..., 0] = cosine * delta[..., 0] + sine * delta[..., 1]; result[..., 1] = -sine * delta[..., 0] + cosine * delta[..., 1]
    return result


def visible_primitive_targets_from_frame_support(
    *,
    field: SweptSuperellipseProvenanceField,
    frame_supports: Sequence[PrimitiveFrameSupport],
    current_sensor_xyz_m: np.ndarray,
    current_yaw_deg: float,
    maximum_slots: int = 32,
) -> VisiblePrimitiveWindowTargets:
    """Aggregate five cached frame summaries into the exact cropped target."""

    if len(frame_supports) != 5 or maximum_slots < 1:
        raise ValueError("five frame supports and positive capacity are required")
    primitive_count = len(field.primitive_ids)
    if any(len(value.support_ray_count) != primitive_count for value in frame_supports):
        raise ValueError("frame support primitive population drift")
    total = np.sum([value.support_ray_count for value in frame_supports], axis=0)
    active = np.flatnonzero(total > 0)
    if len(active) > maximum_slots:
        raise OverflowError(f"visible primitive cardinality {len(active)} exceeds {maximum_slots} slots")
    primitive_index = np.full(maximum_slots, -1, dtype=np.int32); mask = np.zeros(maximum_slots, dtype=np.uint8)
    axes = np.zeros((maximum_slots, 3, 3), dtype=np.float32); half_axes = np.zeros((maximum_slots, 2, 2), dtype=np.float32); exponent = np.zeros((maximum_slots, 2), dtype=np.float32)
    counts = np.zeros(maximum_slots, dtype=np.int32); temporal = np.zeros((5, maximum_slots), dtype=np.uint8)
    for slot, index in enumerate(active):
        lows = [int(value.minimum_sample_index[index]) for value in frame_supports if value.support_ray_count[index] > 0]
        highs = [int(value.maximum_sample_index[index]) for value in frame_supports if value.support_ray_count[index] > 0]
        low, high = min(lows), max(highs); operand = field.operands[int(index)]
        middle_arc = .5 * (operand.arc_m[low] + operand.arc_m[high]); middle = int(np.argmin(np.abs(operand.arc_m - middle_arc)))
        controls = operand.points[[low, middle, high]]; fractions = operand.arc_m[[low, high]] / operand.arc_m[-1]
        parameters, shapes = operand.primitive.parameters_at_fraction(fractions)
        primitive_index[slot] = int(index); mask[slot] = 1
        axes[slot] = _current_sensor_transform(controls, current_sensor_xyz_m, current_yaw_deg)
        half_axes[slot] = parameters; exponent[slot] = shapes; counts[slot] = int(total[index])
        temporal[:, slot] = np.asarray([value.support_ray_count[index] > 0 for value in frame_supports], dtype=np.uint8)
    return VisiblePrimitiveWindowTargets(primitive_index, mask, axes, half_axes, exponent, counts, temporal)


__all__ = ["PrimitiveFrameSupport", "summarize_primitive_frame_support", "visible_primitive_targets_from_frame_support"]
