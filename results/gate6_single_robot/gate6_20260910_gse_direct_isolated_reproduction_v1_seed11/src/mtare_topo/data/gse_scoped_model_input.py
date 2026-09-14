"""Exact selected-window sensor reader; no teacher geometry or identity input.

Execution against real sources requires a separate approved export/diagnostic
Data Card. The earlier metadata-only inventory card does not authorize it.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.data.gse_scoped_inventory import EvidenceStore, validate_selection
from mtare_topo.data.primitive_relation_training import PrimitiveRelationStudentInput


SENSOR_FIELDS = frozenset(("range_m", "valid_mask"))
WINDOW_FIELDS = frozenset(("frame_row", "source_global_sequence_index",
                          "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"))


@dataclass(frozen=True)
class ScopedModelInputs:
    student: tuple[PrimitiveRelationStudentInput, ...]
    frame_rows: np.ndarray                # provenance only, never passed to model
    source_sequence_indices: np.ndarray   # provenance only, never passed to model


class ScopedCompositionModelReader:
    def __init__(self, sensor_root, teacher_root, records, *, expected_sha256=None):
        self.selection = validate_selection(records)  # before any filesystem access
        if sum(map(len, self.selection.values())) > 180:
            raise ValueError("scoped field recovery is limited to 180 existing observations")
        self.sensor_root = Path(sensor_root).resolve(strict=True)
        self.teacher_root = Path(teacher_root).resolve(strict=True)
        self.expected_sha256 = expected_sha256
        self.opened = {}

    def _open(self, root, task, fields):
        path = root / (task + ".zarr")
        if path.resolve().parent != root:
            raise PermissionError("selected shard escapes its root")
        return zarr.open_group(store=EvidenceStore(path, fields, self.opened, self.expected_sha256), mode="r")

    def read_task(self, task):
        if task not in self.selection:
            raise PermissionError("task outside selected population")
        records = self.selection[task]
        indices = np.asarray([r["row_index"] for r in records], dtype=np.int64)
        teacher = self._open(self.teacher_root, task, WINDOW_FIELDS)
        expected = {"parent_id": task.split("__")[0], "partition": "fit", "geometry_realization": "c1_mixed"}
        if any(teacher.attrs.get(k) != v for k, v in expected.items()):
            raise ValueError("teacher split/identity mismatch")
        if (teacher.attrs.get("student_identity_input_forbidden") is not True
                or teacher.attrs.get("window_frames") != 5):
            raise ValueError("teacher causal input contract missing")
        count = teacher["frame_row"].shape[0]
        if teacher["frame_row"].shape != (count, 5) or np.any(indices >= count):
            raise ValueError("selected windows out of bounds")
        source = np.asarray(teacher["source_global_sequence_index"].oindex[indices])
        if not np.array_equal(source, [r["source_global_sequence_index"] for r in records]):
            raise ValueError("selected source sequence drift")
        rows = np.asarray(teacher["frame_row"].oindex[indices])
        if rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(np.diff(rows, axis=1) <= 0):
            raise ValueError("noncausal frame-row order")
        translation = np.asarray(teacher["relative_translation_current_sensor_m"].oindex[indices])
        yaw = np.asarray(teacher["relative_yaw_current_sensor_deg"].oindex[indices])
        if (translation.shape != (len(indices), 5, 3) or yaw.shape != (len(indices), 5)
                or translation.dtype != np.float32 or yaw.dtype != np.float32):
            raise ValueError("relative odometry shape/dtype drift")
        sensor = self._open(self.sensor_root, task, SENSOR_FIELDS)
        if any(sensor.attrs.get(k) != v for k, v in expected.items()):
            raise ValueError("paired sensor split/identity mismatch")
        if (sensor.attrs.get("sensor_shape") != [16, 720] or sensor.attrs.get("maximum_range_m") != 50.
                or sensor.attrs.get("student_pose_input_forbidden") is not True):
            raise ValueError("sensor input contract mismatch")
        frames = sensor["range_m"].shape[0]
        if (sensor["range_m"].shape != (frames, 16, 720) or sensor["range_m"].dtype != np.float32
                or sensor["valid_mask"].shape != (frames, 16, 720) or sensor["valid_mask"].dtype != np.uint8
                or np.any(rows >= frames)):
            raise ValueError("sensor shape/dtype/frame bounds drift")
        unique, inverse = np.unique(rows, return_inverse=True)
        ranges = np.asarray(sensor["range_m"].oindex[unique])
        valid = np.asarray(sensor["valid_mask"].oindex[unique])
        if (not np.isfinite(ranges).all() or np.any(ranges < 0) or np.any(ranges > 50. + 1e-4)
                or np.any(valid > 1)):
            raise ValueError("raw sensor values violate frozen contract")
        inverse = inverse.reshape(len(indices), 5)
        examples = tuple(PrimitiveRelationStudentInput(
            np.stack((ranges[lookup] / np.float32(50.), valid[lookup].astype(np.float32)), axis=1),
            translation[i], yaw[i],
        ) for i, lookup in enumerate(inverse))
        return ScopedModelInputs(examples, rows, source)
