"""Leakage-safe training reader for paired P1a sensors and P1b targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.primitive_relation_storage import (
    PackedPrimitiveRelationTargets,
    unpack_disconnected_overlap,
    unpack_endpoint_attachment,
)


SENSOR_SHAPE = (16, 720)
WINDOW_FRAMES = 5
MAXIMUM_SLOTS = 32
MAXIMUM_RANGE_M = 50.0


@dataclass(frozen=True)
class PrimitiveRelationStudentInput:
    """The complete and only model-visible input for one causal window."""

    range_valid: np.ndarray
    relative_translation_current_sensor_m: np.ndarray
    relative_yaw_current_sensor_deg: np.ndarray

    def __post_init__(self) -> None:
        if self.range_valid.shape != (WINDOW_FRAMES, 2, *SENSOR_SHAPE) or self.range_valid.dtype != np.float32:
            raise ValueError("student range/valid tensor must be float32 [5,2,16,720]")
        if self.relative_translation_current_sensor_m.shape != (WINDOW_FRAMES, 3) or self.relative_translation_current_sensor_m.dtype != np.float32:
            raise ValueError("student relative translation must be float32 [5,3]")
        if self.relative_yaw_current_sensor_deg.shape != (WINDOW_FRAMES,) or self.relative_yaw_current_sensor_deg.dtype != np.float32:
            raise ValueError("student relative yaw must be float32 [5]")
        if not np.isfinite(self.range_valid).all() or not np.isfinite(self.relative_translation_current_sensor_m).all() or not np.isfinite(self.relative_yaw_current_sensor_deg).all():
            raise ValueError("student input contains non-finite values")
        if np.any((self.range_valid[:, 1] != 0.0) & (self.range_valid[:, 1] != 1.0)):
            raise ValueError("student valid channel must remain binary")
        if not np.array_equal(self.relative_translation_current_sensor_m[-1], np.zeros(3, dtype=np.float32)) or self.relative_yaw_current_sensor_deg[-1] != 0.0:
            raise ValueError("current-frame relative odometry must be exactly zero")


@dataclass(frozen=True)
class PrimitiveRelationTrainingTargets:
    """Teacher-only geometry, relation and correspondence targets."""

    primitive_index: np.ndarray
    primitive_mask: np.ndarray
    axis_control_current_sensor_m: np.ndarray
    endpoint_half_axes_m: np.ndarray
    endpoint_shape_exponent: np.ndarray
    support_ray_count: np.ndarray
    temporal_visibility: np.ndarray
    frame_primitive_index: np.ndarray
    temporal_destination: np.ndarray
    endpoint_attachment: np.ndarray
    disconnected_overlap: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "primitive_index": ((32,), np.dtype("i4")),
            "primitive_mask": ((32,), np.dtype("u1")),
            "axis_control_current_sensor_m": ((32, 3, 3), np.dtype("f4")),
            "endpoint_half_axes_m": ((32, 2, 2), np.dtype("f4")),
            "endpoint_shape_exponent": ((32, 2), np.dtype("f4")),
            "support_ray_count": ((32,), np.dtype("i4")),
            "temporal_visibility": ((5, 32), np.dtype("u1")),
            "frame_primitive_index": ((5, 32), np.dtype("i4")),
            "temporal_destination": ((5, 32), np.dtype("i1")),
            "endpoint_attachment": ((32, 2, 32, 2), np.dtype("u1")),
            "disconnected_overlap": ((32, 32), np.dtype("u1")),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"Teacher target contract drift: {name}")
        if not np.array_equal(self.primitive_mask, (self.primitive_index >= 0).astype(np.uint8)):
            raise ValueError("primitive mask/index mismatch")
        if np.any(self.temporal_destination < -1) or np.any(self.temporal_destination > 32):
            raise ValueError("temporal destination outside slots+dustbin")
        if np.any(self.endpoint_attachment > 1) or np.any(self.disconnected_overlap > 1):
            raise ValueError("relation targets must be binary")


@dataclass(frozen=True)
class PrimitiveRelationTrainingExample:
    student: PrimitiveRelationStudentInput
    targets: PrimitiveRelationTrainingTargets
    source_global_sequence_index: int
    variant_global_sequence_index: int


class PrimitiveRelationTrainingShard:
    """Read one paired P1a/P1b shard without exposing Teacher metadata to input."""

    def __init__(self, sensor_path: Path, teacher_path: Path) -> None:
        self.sensor_path = Path(sensor_path)
        self.teacher_path = Path(teacher_path)
        self.sensor = zarr.open_group(str(self.sensor_path), mode="r")
        self.teacher = zarr.open_group(str(self.teacher_path), mode="r")
        self._validate_shards()

    def _validate_shards(self) -> None:
        for key in ("parent_id", "partition", "geometry_realization"):
            if self.sensor.attrs.get(key) != self.teacher.attrs.get(key):
                raise ValueError(f"paired P1a/P1b attribute drift: {key}")
        if self.sensor.attrs.get("sensor_shape") != list(SENSOR_SHAPE) or float(self.sensor.attrs.get("maximum_range_m", -1.0)) != MAXIMUM_RANGE_M:
            raise ValueError("P1a sensor contract drift")
        if self.sensor.attrs.get("student_pose_input_forbidden") is not True:
            raise ValueError("P1a student pose prohibition is missing")
        if self.teacher.attrs.get("student_identity_input_forbidden") is not True or int(self.teacher.attrs.get("maximum_slots", -1)) != MAXIMUM_SLOTS or int(self.teacher.attrs.get("window_frames", -1)) != WINDOW_FRAMES:
            raise ValueError("P1b student identity/capacity contract drift")
        frame_count = int(self.sensor["range_m"].shape[0])
        if self.sensor["valid_mask"].shape != (frame_count, *SENSOR_SHAPE):
            raise ValueError("P1a valid-mask shape drift")
        sequence_count = int(self.teacher["frame_row"].shape[0])
        if self.teacher["frame_row"].shape != (sequence_count, WINDOW_FRAMES):
            raise ValueError("P1b frame-row shape drift")
        if self.teacher["source_global_sequence_index"].shape != (sequence_count,) or self.teacher["variant_global_sequence_index"].shape != (sequence_count,):
            raise ValueError("P1b sequence identity shape drift")
        rows = np.asarray(self.teacher["frame_row"][:], dtype=np.int64)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= frame_count):
            raise ValueError("P1b frame row outside paired P1a shard")
        required_targets = {
            "primitive_index", "primitive_mask", "axis_control_current_sensor_m",
            "endpoint_half_axes_m", "endpoint_shape_exponent", "support_ray_count",
            "temporal_visibility", "frame_primitive_index", "temporal_destination",
            "endpoint_neighbor", "disconnected_overlap_packed",
            "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg",
        }
        missing = required_targets - set(self.teacher.array_keys())
        if missing:
            raise ValueError(f"P1b target inventory missing: {sorted(missing)}")

    def __len__(self) -> int:
        return int(self.teacher["frame_row"].shape[0])

    def __getitem__(self, index: int) -> PrimitiveRelationTrainingExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rows = np.asarray(self.teacher["frame_row"][index], dtype=np.int64)
        ranges = np.asarray(self.sensor["range_m"].get_orthogonal_selection((rows, slice(None), slice(None))), dtype=np.float32)
        valid = np.asarray(self.sensor["valid_mask"].get_orthogonal_selection((rows, slice(None), slice(None))), dtype=np.uint8)
        if not np.isfinite(ranges).all() or np.any(ranges < 0.0) or np.any(ranges > MAXIMUM_RANGE_M + 1e-4):
            raise ValueError("P1a range values violate the frozen sensor contract")
        range_valid = np.stack((ranges / np.float32(MAXIMUM_RANGE_M), valid.astype(np.float32)), axis=1).astype(np.float32, copy=False)
        student = PrimitiveRelationStudentInput(
            range_valid=range_valid,
            relative_translation_current_sensor_m=np.asarray(self.teacher["relative_translation_current_sensor_m"][index], dtype=np.float32),
            relative_yaw_current_sensor_deg=np.asarray(self.teacher["relative_yaw_current_sensor_deg"][index], dtype=np.float32),
        )
        packed = PackedPrimitiveRelationTargets(
            temporal_destination=np.asarray(self.teacher["temporal_destination"][index], dtype=np.int8),
            endpoint_neighbor=np.asarray(self.teacher["endpoint_neighbor"][index], dtype=np.int8),
            disconnected_overlap_packed=np.asarray(self.teacher["disconnected_overlap_packed"][index], dtype=np.uint8),
        )
        targets = PrimitiveRelationTrainingTargets(
            primitive_index=np.asarray(self.teacher["primitive_index"][index], dtype=np.int32),
            primitive_mask=np.asarray(self.teacher["primitive_mask"][index], dtype=np.uint8),
            axis_control_current_sensor_m=np.asarray(self.teacher["axis_control_current_sensor_m"][index], dtype=np.float32),
            endpoint_half_axes_m=np.asarray(self.teacher["endpoint_half_axes_m"][index], dtype=np.float32),
            endpoint_shape_exponent=np.asarray(self.teacher["endpoint_shape_exponent"][index], dtype=np.float32),
            support_ray_count=np.asarray(self.teacher["support_ray_count"][index], dtype=np.int32),
            temporal_visibility=np.asarray(self.teacher["temporal_visibility"][index], dtype=np.uint8),
            frame_primitive_index=np.asarray(self.teacher["frame_primitive_index"][index], dtype=np.int32),
            temporal_destination=packed.temporal_destination,
            endpoint_attachment=unpack_endpoint_attachment(packed),
            disconnected_overlap=unpack_disconnected_overlap(packed),
        )
        return PrimitiveRelationTrainingExample(
            student=student, targets=targets,
            source_global_sequence_index=int(self.teacher["source_global_sequence_index"][index]),
            variant_global_sequence_index=int(self.teacher["variant_global_sequence_index"][index]),
        )


__all__ = [
    "PrimitiveRelationStudentInput", "PrimitiveRelationTrainingTargets",
    "PrimitiveRelationTrainingExample", "PrimitiveRelationTrainingShard",
]
