"""Deterministic shard-local batch reader for primitive-relation training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from mtare_topo.data.primitive_relation_training import (
    MAXIMUM_RANGE_M,
    PrimitiveRelationTrainingShard,
)


@dataclass(frozen=True)
class PrimitiveRelationNumpyBatch:
    range_valid: np.ndarray
    relative_translation_current_sensor_m: np.ndarray
    relative_yaw_current_sensor_deg: np.ndarray
    primitive_index: np.ndarray
    primitive_mask: np.ndarray
    axis_control_current_sensor_m: np.ndarray
    endpoint_half_axes_m: np.ndarray
    endpoint_shape_exponent: np.ndarray
    temporal_visibility: np.ndarray
    endpoint_attachment: np.ndarray
    disconnected_overlap: np.ndarray
    source_global_sequence_index: np.ndarray
    variant_global_sequence_index: np.ndarray

    def __post_init__(self) -> None:
        batch = len(self.range_valid)
        expected = {
            "range_valid": (batch, 5, 2, 16, 720),
            "relative_translation_current_sensor_m": (batch, 5, 3),
            "relative_yaw_current_sensor_deg": (batch, 5),
            "primitive_index": (batch, 32), "primitive_mask": (batch, 32),
            "axis_control_current_sensor_m": (batch, 32, 3, 3),
            "endpoint_half_axes_m": (batch, 32, 2, 2),
            "endpoint_shape_exponent": (batch, 32, 2),
            "temporal_visibility": (batch, 5, 32),
            "endpoint_attachment": (batch, 32, 2, 32, 2),
            "disconnected_overlap": (batch, 32, 32),
            "source_global_sequence_index": (batch,),
            "variant_global_sequence_index": (batch,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"primitive batch shape drift: {name}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"primitive batch contains non-finite values: {name}")
        if self.range_valid.dtype != np.float32 or self.relative_translation_current_sensor_m.dtype != np.float32 or self.relative_yaw_current_sensor_deg.dtype != np.float32:
            raise ValueError("student batch tensors must be float32")
        if np.any((self.range_valid[:, :, 0] < 0.0) | (self.range_valid[:, :, 0] > 1.0)):
            raise ValueError("normalized batch ranges must lie in [0,1]")
        if np.any((self.range_valid[:, :, 1] != 0.0) & (self.range_valid[:, :, 1] != 1.0)):
            raise ValueError("batch valid-return channel must remain binary")
        if not np.array_equal(
            self.relative_translation_current_sensor_m[:, -1],
            np.zeros((batch, 3), dtype=np.float32),
        ) or not np.array_equal(
            self.relative_yaw_current_sensor_deg[:, -1],
            np.zeros(batch, dtype=np.float32),
        ):
            raise ValueError("current-frame batch relative odometry must be exactly zero")
        if not np.array_equal(self.primitive_mask, (self.primitive_index >= 0).astype(np.uint8)):
            raise ValueError("primitive batch mask/index mismatch")


def _batch_relations(neighbors: np.ndarray, packed_overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = len(neighbors)
    attachment = np.zeros((batch, 32, 2, 32, 2), dtype=np.uint8)
    batch_index, slot, endpoint, neighbor_index = np.indices(neighbors.shape)
    destination = neighbors.astype(np.int16)
    valid = destination >= 0
    attachment[
        batch_index[valid], slot[valid], endpoint[valid],
        destination[valid] // 2, destination[valid] % 2,
    ] = 1
    overlap = np.unpackbits(
        packed_overlap, axis=2, count=32, bitorder="little",
    ).astype(np.uint8)
    return attachment, overlap


class PrimitiveRelationBatchLoader:
    """Read complete P1a/P1b populations with deterministic shard-local shuffle."""

    def __init__(self, sensor_root: Path, teacher_root: Path) -> None:
        self.sensor_root = Path(sensor_root)
        self.teacher_root = Path(teacher_root)
        sensor_tasks = {value.name for value in self.sensor_root.glob("*.zarr") if value.is_dir()}
        teacher_tasks = {value.name for value in self.teacher_root.glob("*.zarr") if value.is_dir()}
        if not sensor_tasks or sensor_tasks != teacher_tasks:
            raise ValueError("paired primitive batch roots are empty or misaligned")
        self.task_names = tuple(sorted(sensor_tasks))
        self._lengths: dict[str, int] = {}
        for name in self.task_names:
            shard = PrimitiveRelationTrainingShard(self.sensor_root / name, self.teacher_root / name)
            self._lengths[name] = len(shard)

    def __len__(self) -> int:
        return sum(self._lengths.values())

    def _read(self, name: str, indices: np.ndarray) -> PrimitiveRelationNumpyBatch:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("primitive batch indices must be a non-empty vector")
        if name not in self._lengths or np.any(indices < 0) or np.any(indices >= self._lengths[name]):
            raise IndexError("primitive batch index outside paired shard")
        shard = PrimitiveRelationTrainingShard(self.sensor_root / name, self.teacher_root / name)
        sensor, teacher = shard.sensor, shard.teacher
        rows = np.asarray(teacher["frame_row"].oindex[indices], dtype=np.int64)
        unique_rows, inverse = np.unique(rows.reshape(-1), return_inverse=True)
        ranges_unique = np.asarray(sensor["range_m"].oindex[unique_rows], dtype=np.float32)
        valid_unique = np.asarray(sensor["valid_mask"].oindex[unique_rows], dtype=np.uint8)
        ranges = ranges_unique[inverse].reshape(len(indices), 5, 16, 720)
        valid = valid_unique[inverse].reshape(len(indices), 5, 16, 720)
        range_valid = np.stack(
            (ranges / np.float32(MAXIMUM_RANGE_M), valid.astype(np.float32)), axis=2,
        ).astype(np.float32, copy=False)

        def read(name_: str, dtype) -> np.ndarray:
            return np.asarray(teacher[name_].oindex[indices], dtype=dtype)

        neighbors = read("endpoint_neighbor", np.int8)
        packed_overlap = read("disconnected_overlap_packed", np.uint8)
        attachment, overlap = _batch_relations(neighbors, packed_overlap)
        return PrimitiveRelationNumpyBatch(
            range_valid=range_valid,
            relative_translation_current_sensor_m=read("relative_translation_current_sensor_m", np.float32),
            relative_yaw_current_sensor_deg=read("relative_yaw_current_sensor_deg", np.float32),
            primitive_index=read("primitive_index", np.int32),
            primitive_mask=read("primitive_mask", np.uint8),
            axis_control_current_sensor_m=read("axis_control_current_sensor_m", np.float32),
            endpoint_half_axes_m=read("endpoint_half_axes_m", np.float32),
            endpoint_shape_exponent=read("endpoint_shape_exponent", np.float32),
            temporal_visibility=read("temporal_visibility", np.uint8),
            endpoint_attachment=attachment, disconnected_overlap=overlap,
            source_global_sequence_index=read("source_global_sequence_index", np.int64),
            variant_global_sequence_index=read("variant_global_sequence_index", np.int64),
        )

    def iter_epoch(
        self,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
        shuffle: bool,
        block_size: int = 256,
    ) -> Iterator[PrimitiveRelationNumpyBatch]:
        if batch_size < 1 or block_size < batch_size:
            raise ValueError("batch/block size contract is invalid")
        generator = np.random.default_rng(int(seed) * 100003 + int(epoch))
        tasks = list(self.task_names)
        if shuffle:
            generator.shuffle(tasks)
        for name in tasks:
            length = self._lengths[name]
            blocks = [np.arange(start, min(start + block_size, length), dtype=np.int64) for start in range(0, length, block_size)]
            if shuffle:
                generator.shuffle(blocks)
            for block in blocks:
                if shuffle:
                    generator.shuffle(block)
                for start in range(0, len(block), batch_size):
                    yield self._read(name, np.sort(block[start:start + batch_size]))


__all__ = ["PrimitiveRelationBatchLoader", "PrimitiveRelationNumpyBatch"]
