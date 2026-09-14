"""Training loader for aligned endpoint-observability and anchor sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.primitive_composition_anchor_sidecar import (
    validate_materialized_current_sensor_anchors,
)
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
    ObservablePrimitiveRelationNumpyBatch,
)


ANCHOR_SIDECAR_SCHEMA = "primitive_composition_anchor_target_sidecar_v1"


@dataclass(frozen=True)
class CompositionAnchorNumpyBatch:
    base: ObservablePrimitiveRelationNumpyBatch
    anchor_current_sensor_m: np.ndarray

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor_current_sensor_m)
        validate_materialized_current_sensor_anchors(
            anchor, self.base.base.primitive_mask,
        )
        if len(anchor) != len(self.base.base.range_valid):
            raise ValueError("composition-anchor/base batch length differs")


class CompositionAnchorBatchLoader(ObservablePrimitiveRelationBatchLoader):
    """Bind P1a/P1b, observability and numeric anchor targets by sequence."""

    def __init__(
        self,
        sensor_root: Path,
        teacher_root: Path,
        observability_root: Path,
        anchor_root: Path,
    ) -> None:
        super().__init__(sensor_root, teacher_root, observability_root)
        self.anchor_root = Path(anchor_root)
        anchor_tasks = {
            value.name for value in self.anchor_root.glob("*.zarr") if value.is_dir()
        }
        if anchor_tasks != set(self.task_names):
            raise ValueError("composition-anchor sidecars are empty or task-misaligned")
        for name in self.task_names:
            anchor = zarr.open_group(str(self.anchor_root / name), mode="r")
            if anchor.attrs.get("schema_version") != ANCHOR_SIDECAR_SCHEMA:
                raise ValueError("composition-anchor sidecar schema drift")
            if anchor.attrs.get("coordinate_frame") != "current_sensor_xyz_yaw_only":
                raise ValueError("composition-anchor coordinate-frame drift")
            if float(anchor.attrs.get("inactive_slot_value", np.nan)) != 0.0:
                raise ValueError("composition-anchor inactive-slot contract drift")
            if anchor.attrs.get("student_construction_identity_input_forbidden") is not True:
                raise ValueError("composition-anchor identity isolation drift")
            length = self._lengths[name]
            if (
                anchor["anchor_current_sensor_m"].shape != (length, 32, 2, 3)
                or np.dtype(anchor["anchor_current_sensor_m"].dtype) != np.dtype("<f4")
                or anchor["source_global_sequence_index"].shape != (length,)
            ):
                raise ValueError("composition-anchor sidecar array contract drift")
            teacher = zarr.open_group(str(self.teacher_root / name), mode="r")
            if not np.array_equal(
                anchor["source_global_sequence_index"][:],
                teacher["source_global_sequence_index"][:],
            ):
                raise ValueError("composition-anchor source sequence identity drift")

    def _read(self, name: str, indices: np.ndarray) -> CompositionAnchorNumpyBatch:
        base = super()._read(name, indices)
        anchor = zarr.open_group(str(self.anchor_root / name), mode="r")
        indices = np.asarray(indices, dtype=np.int64)
        value = np.asarray(
            anchor["anchor_current_sensor_m"].oindex[indices], dtype=np.float32,
        )
        sequence = np.asarray(
            anchor["source_global_sequence_index"].oindex[indices], dtype=np.int64,
        )
        if not np.array_equal(sequence, base.base.source_global_sequence_index):
            raise ValueError("composition-anchor batch source sequence drift")
        return CompositionAnchorNumpyBatch(
            base=base, anchor_current_sensor_m=value,
        )


__all__ = [
    "ANCHOR_SIDECAR_SCHEMA",
    "CompositionAnchorBatchLoader",
    "CompositionAnchorNumpyBatch",
]
