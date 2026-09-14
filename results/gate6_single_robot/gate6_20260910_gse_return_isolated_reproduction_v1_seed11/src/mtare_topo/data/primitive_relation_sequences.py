"""Exact five-frame reference rows for paired primitive-relation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimitiveRelationSequenceReferences:
    source_global_sequence_index: np.ndarray
    variant_global_sequence_index: np.ndarray
    frame_row: np.ndarray
    traversal_id: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.source_global_sequence_index)
        if self.variant_global_sequence_index.shape != (count,) or self.frame_row.shape != (count, 5) or len(self.traversal_id) != count:
            raise ValueError("sequence reference shapes disagree")
        if count and (
            len(np.unique(self.source_global_sequence_index)) != count
            or len(np.unique(self.variant_global_sequence_index)) != count
            or np.any(np.diff(self.source_global_sequence_index) <= 0)
        ):
            raise ValueError("sequence identities must be unique and ordered")


@dataclass(frozen=True)
class CausalRelativeOdometry:
    translation_current_sensor_m: np.ndarray
    yaw_current_sensor_deg: np.ndarray

    def __post_init__(self) -> None:
        if self.translation_current_sensor_m.shape != (5, 3) or self.yaw_current_sensor_deg.shape != (5,):
            raise ValueError("causal relative odometry must have shapes [5,3] and [5]")
        if not np.isfinite(self.translation_current_sensor_m).all() or not np.isfinite(self.yaw_current_sensor_deg).all():
            raise ValueError("causal relative odometry must be finite")
        if not np.allclose(self.translation_current_sensor_m[-1], 0.0, atol=1e-12) or abs(float(self.yaw_current_sensor_deg[-1])) > 1e-12:
            raise ValueError("current frame must be the exact relative-odometry origin")


def causal_relative_odometry(sensor_xyz_m: np.ndarray, yaw_deg: np.ndarray) -> CausalRelativeOdometry:
    """Express five past/current sensor poses in the current sensor frame."""

    xyz = np.asarray(sensor_xyz_m, dtype=np.float64); yaw = np.asarray(yaw_deg, dtype=np.float64)
    if xyz.shape != (5, 3) or yaw.shape != (5,) or not np.isfinite(xyz).all() or not np.isfinite(yaw).all():
        raise ValueError("relative odometry source must contain five finite poses")
    delta = xyz - xyz[-1]
    angle = np.radians(float(yaw[-1])); cosine, sine = np.cos(angle), np.sin(angle)
    translation = delta.copy()
    translation[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
    translation[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
    relative_yaw = (yaw - yaw[-1] + 180.0) % 360.0 - 180.0
    translation[-1] = 0.0; relative_yaw[-1] = 0.0
    return CausalRelativeOdometry(translation, relative_yaw)


def world_primitive_relation_sequence_references(
    *,
    parent_id: str,
    traversal_manifest: Sequence[Mapping],
    shard_global_frame_indices: np.ndarray,
    realization_index: int,
    source_sequence_population: int = 188126,
) -> PrimitiveRelationSequenceReferences:
    """Map the sealed source sequence IDs into one paired realization."""

    if realization_index not in (0, 1, 2) or source_sequence_population < 1:
        raise ValueError("realization index/population is invalid")
    frames = np.asarray(shard_global_frame_indices, dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) != 1):
        raise ValueError("shard global frame identities must be contiguous")
    records = sorted(
        (row for row in traversal_manifest if str(row["parent_id"]) == parent_id),
        key=lambda row: int(row["global_sequence_offset"]),
    )
    sequence_ids: list[int] = []; frame_rows: list[np.ndarray] = []; traversal_ids: list[str] = []
    first_frame = int(frames[0])
    for record in records:
        sequence_count = int(record["sequence_count"]); frame_offset = int(record["global_frame_offset"]); sequence_offset = int(record["global_sequence_offset"])
        unique_count = int(record["unique_frame_count"])
        if sequence_count and unique_count != sequence_count + 4:
            raise RuntimeError("five-frame traversal count drift")
        for local_sequence in range(sequence_count):
            global_references = frame_offset + local_sequence + np.arange(5, dtype=np.int64)
            local_references = global_references - first_frame
            if np.any(local_references < 0) or np.any(local_references >= len(frames)) or not np.array_equal(frames[local_references], global_references):
                raise RuntimeError("sequence references do not resolve inside shard")
            sequence_ids.append(sequence_offset + local_sequence); frame_rows.append(local_references); traversal_ids.append(str(record["traversal_id"]))
    source = np.asarray(sequence_ids, dtype=np.int64); variant = source + int(realization_index) * int(source_sequence_population)
    return PrimitiveRelationSequenceReferences(
        source_global_sequence_index=source, variant_global_sequence_index=variant,
        frame_row=np.asarray(frame_rows, dtype=np.int32).reshape(len(source), 5),
        traversal_id=tuple(traversal_ids),
    )


__all__ = [
    "CausalRelativeOdometry", "PrimitiveRelationSequenceReferences",
    "causal_relative_odometry", "world_primitive_relation_sequence_references",
]
