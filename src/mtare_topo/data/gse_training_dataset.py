"""Read the sealed deduplicated GSE sequence dataset without pose leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.data.gse_training_targets import circular_roll_training_example


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class GSESequenceDataset:
    """Return five unique-frame references and complete masked GSE targets."""

    def __init__(
        self,
        run_dir: str | Path,
        split: str,
        *,
        augment_azimuth: bool = False,
        augmentation_seed: int = 0,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if split == "validation" and augment_azimuth:
            raise ValueError("validation azimuth augmentation is forbidden")
        self.run_dir = Path(run_dir).resolve()
        self.split = split
        self.augment_azimuth = bool(augment_azimuth)
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self.records = [
            row
            for row in _read_jsonl(self.run_dir / "artifacts/sequence_manifest.jsonl")
            if str(row["split"]) == split
        ]
        self._groups: dict[str, Any] = {}
        self._association_labels: np.ndarray | None = None
        self._event_labels: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def association_labels(self) -> np.ndarray:
        """Return frozen place identities aligned with dataset indices."""

        if self._association_labels is None:
            labels = np.full(len(self.records), -1, dtype=np.int64)
            by_parent: dict[str, list[tuple[int, int]]] = {}
            for dataset_index, record in enumerate(self.records):
                by_parent.setdefault(str(record["parent_id"]), []).append(
                    (dataset_index, int(record["world_sequence_row"]))
                )
            for parent_id, indexed_rows in by_parent.items():
                group = self._group(parent_id)
                rows = np.asarray([row for _, row in indexed_rows], dtype=np.int64)
                identities = np.asarray(group["association_identity"].oindex[rows], dtype=np.int64)
                valid = np.asarray(group["association_valid_mask"].oindex[rows], dtype=np.uint8)
                if identities.shape != rows.shape or valid.shape != rows.shape:
                    raise RuntimeError("association label arrays do not align with sequence rows")
                if np.any((valid == 0) & (identities != -1)) or np.any((valid == 1) & (identities < 0)):
                    raise RuntimeError("association identity and validity mask disagree")
                for (dataset_index, _), identity, is_valid in zip(
                    indexed_rows, identities, valid, strict=True
                ):
                    if is_valid:
                        labels[dataset_index] = int(identity)
            labels.setflags(write=False)
            self._association_labels = labels
        return self._association_labels

    def event_labels(self) -> np.ndarray:
        """Return frozen structural event indices aligned with dataset indices."""

        if self._event_labels is None:
            labels = np.empty(len(self.records), dtype=np.int64)
            by_parent: dict[str, list[tuple[int, int]]] = {}
            for dataset_index, record in enumerate(self.records):
                by_parent.setdefault(str(record["parent_id"]), []).append(
                    (dataset_index, int(record["world_sequence_row"]))
                )
            for parent_id, indexed_rows in by_parent.items():
                group = self._group(parent_id)
                rows = np.asarray([row for _, row in indexed_rows], dtype=np.int64)
                values = np.asarray(group["event_index"].oindex[rows], dtype=np.int64)
                if values.shape != rows.shape or np.any(values < 0) or np.any(values >= 5):
                    raise RuntimeError("event labels do not align with the five-class contract")
                for (dataset_index, _), value in zip(indexed_rows, values, strict=True):
                    labels[dataset_index] = int(value)
            labels.setflags(write=False)
            self._event_labels = labels
        return self._event_labels

    def _group(self, parent_id: str):
        import zarr

        if parent_id not in self._groups:
            path = self.run_dir / "artifacts/dataset" / self.split / f"{parent_id}.zarr"
            self._groups[parent_id] = zarr.open_group(str(path), mode="r")
        return self._groups[parent_id]

    def _shift(self, global_sequence_index: int) -> int:
        if not self.augment_azimuth:
            return 0
        payload = f"gse-roll-v1:{self.augmentation_seed}:{self.epoch}:{global_sequence_index}".encode()
        # The encoder has four-column azimuth stride. Restricting the train-only
        # roll to that lattice preserves exact circular equivariance while still
        # covering all 180 distinct orientations.
        return 4 * (int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 180)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        group = self._group(str(record["parent_id"]))
        row = int(record["world_sequence_row"])
        references = np.asarray(group["local_frame_references"][row], dtype=np.int64)
        if references.shape != (5,) or np.any(references < 0):
            raise RuntimeError("sequence does not contain five valid local frame references")
        range_m = np.asarray(group["range_m"].oindex[references], dtype=np.float32)
        valid = np.asarray(group["valid_mask"].oindex[references], dtype=np.float32)
        if range_m.shape != (5, 16, 720) or valid.shape != range_m.shape:
            raise RuntimeError("deduplicated sequence array shape mismatch")
        if (
            not np.all(np.isfinite(range_m))
            or np.any(range_m < NEAR_RANGE_M)
            or np.any(range_m > MAX_RANGE_M)
            or not np.all((valid == 0.0) | (valid == 1.0))
        ):
            raise RuntimeError("sensor sequence violates the frozen finite range/mask contract")
        student = np.stack((range_m / MAX_RANGE_M, valid), axis=1).astype(np.float32, copy=False)
        geometry = np.asarray(group["geometry"][row], dtype=np.float32)
        local_axis = np.asarray(group["local_axis_robot"][row], dtype=np.float32)
        geometry_mask = np.asarray(group["geometry_valid_mask"][row], dtype=np.uint8)
        exit_mask = np.asarray(group["exit_mask"][row], dtype=np.uint8)
        exit_heading = np.asarray(group["exit_heading_unit"][row], dtype=np.float32)
        exit_width = np.asarray(group["exit_opening_width_m"][row], dtype=np.float32)
        exit_width_mask = np.asarray(group["exit_width_valid_mask"][row], dtype=np.uint8)
        exit_profile = np.asarray(group["exit_vertical_profile_m"][row], dtype=np.float32)
        exit_identity = np.asarray(group["exit_identity"][row], dtype=np.int64)
        if (
            geometry.shape != (4,)
            or local_axis.shape != (3,)
            or geometry_mask.shape != (4,)
            or exit_mask.shape != (6,)
            or exit_heading.shape != (6, 2)
            or exit_width.shape != (6,)
            or exit_width_mask.shape != (6,)
            or exit_profile.shape != (6, 4)
            or exit_identity.shape != (6,)
            or not np.all(np.isfinite(local_axis))
            or not np.all(np.isfinite(geometry))
            or not np.all(np.isfinite(exit_heading))
            or not np.all(np.isfinite(exit_width))
            or not np.all(np.isfinite(exit_profile))
            or np.any(exit_width_mask > exit_mask)
            or np.any(exit_identity[exit_mask == 0] != -1)
        ):
            raise RuntimeError("GSE target arrays violate the frozen shape/mask contract")
        targets: dict[str, Any] = {
            "event_index": np.int64(group["event_index"][row]),
            "local_axis": local_axis,
            "width_m": geometry[0],
            "height_m": geometry[1],
            "slope_deg": geometry[2],
            "curvature_per_m": geometry[3],
            "geometry_valid_mask": geometry_mask,
            "association_identity": np.int64(group["association_identity"][row]),
            "association_valid_mask": np.uint8(group["association_valid_mask"][row]),
            "exit_mask": exit_mask,
            "exit_heading_unit": exit_heading,
            "exit_opening_width_m": exit_width,
            "exit_width_valid_mask": exit_width_mask,
            "exit_vertical_profile": exit_profile,
            "exit_identity": exit_identity,
        }
        student, targets = circular_roll_training_example(
            student,
            targets,
            shift_columns=self._shift(int(record["global_sequence_index"])),
        )
        return {
            "student": student,
            "targets": targets,
            "observation_id": str(record["observation_id"]),
            "parent_id": str(record["parent_id"]),
            "global_sequence_index": int(record["global_sequence_index"]),
        }


__all__ = ["GSESequenceDataset"]
