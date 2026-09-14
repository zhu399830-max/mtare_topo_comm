"""Streaming LiDAR and lossless primitive-provenance export primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mtare_topo.data.cano_sensor_smoke import (
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
    world_directions,
)
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


class ProvenanceRaycaster(Protocol):
    def ray_exit_hits(
        self,
        origins_xyz_m: np.ndarray,
        directions_xyz: np.ndarray,
        initial_inside: np.ndarray,
        *,
        maximum_m: float = 50.0,
    ) -> tuple[PrimitiveRayHit | None, ...]: ...


@dataclass(frozen=True)
class PrimitiveSensorFrame:
    range_m: np.ndarray
    valid_mask: np.ndarray
    primitive_membership_code: np.ndarray
    ambiguous_ray_count: int

    def __post_init__(self) -> None:
        expected = (16, 720)
        if self.range_m.shape != expected or self.valid_mask.shape != expected or self.primitive_membership_code.shape != expected:
            raise ValueError("primitive sensor frame arrays must be 16x720")
        if self.range_m.dtype != np.float32 or self.valid_mask.dtype != np.uint8 or self.primitive_membership_code.dtype != np.uint16:
            raise ValueError("primitive sensor frame dtypes drift")
        if not np.isfinite(self.range_m).all() or np.any(self.range_m < NEAR_RANGE_M) or np.any(self.range_m > MAX_RANGE_M):
            raise ValueError("range frame violates sensor bounds")
        if np.any((self.primitive_membership_code == 0) != (self.valid_mask == 0)):
            raise ValueError("valid rays and primitive membership codes disagree")


def render_primitive_sensor_frame(
    *,
    raycaster: ProvenanceRaycaster,
    field: SweptSuperellipseProvenanceField,
    codebook: PrimitiveMembershipCodebook,
    sensor_xyz_m: np.ndarray,
    yaw_deg: float,
    inside_tolerance_m: float = 1e-9,
    near_range_m: float = NEAR_RANGE_M,
    maximum_range_m: float = MAX_RANGE_M,
) -> PrimitiveSensorFrame:
    """Render one frozen 16x720 frame and encode every complete source set."""

    sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
    if sensor.shape != (3,) or not np.isfinite(sensor).all() or not np.isfinite(yaw_deg):
        raise ValueError("sensor pose must be finite")
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = world_directions(local, float(yaw_deg))
    origins = np.broadcast_to(sensor, directions.shape)
    source_inside = field.operand_signed_distances_sparse(sensor.reshape(1, 3))[0] <= inside_tolerance_m
    if not np.any(source_inside):
        raise RuntimeError("sensor origin is outside finite primitive union")
    inside = np.broadcast_to(source_inside, (len(directions), len(source_inside)))
    hits = raycaster.ray_exit_hits(origins, directions, inside, maximum_m=maximum_range_m)
    if len(hits) != len(directions):
        raise RuntimeError("raycaster returned wrong ray count")
    valid_hits: list[PrimitiveRayHit | None] = []
    ranges = np.full(len(hits), maximum_range_m, dtype=np.float32)
    ambiguous = 0
    for index, hit in enumerate(hits):
        valid = hit is not None and near_range_m <= hit.distance_m <= maximum_range_m
        if valid:
            ranges[index] = np.float32(hit.distance_m)
            valid_hits.append(hit)
            ambiguous += int(len(hit.source_primitive_ids) > 1)
        else:
            valid_hits.append(None)
    codes = codebook.encode(valid_hits)
    valid_mask = (codes > 0).astype(np.uint8)
    shape = (16, 720)
    return PrimitiveSensorFrame(
        range_m=ranges.reshape(shape),
        valid_mask=valid_mask.reshape(shape),
        primitive_membership_code=codes.reshape(shape),
        ambiguous_ray_count=ambiguous,
    )


__all__ = [
    "PrimitiveSensorFrame",
    "ProvenanceRaycaster",
    "render_primitive_sensor_frame",
]
