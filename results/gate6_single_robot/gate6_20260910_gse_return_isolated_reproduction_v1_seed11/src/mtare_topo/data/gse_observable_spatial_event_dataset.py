"""Exact C01-C08 joins for observable spatial-event Teacher V2 training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


OBSERVATION_COUNT = 188126
FIT_OBSERVATION_COUNT = 142184
SELECTION_OBSERVATION_COUNT = 45942
VISIBLE_TOKEN_COUNT = 131424
FIT_TOKEN_COUNT = 98279
QUERY_COUNT = 16


@dataclass(frozen=True)
class ObservableSpatialEventTeacherArrays:
    global_sequence_index: np.ndarray
    partition_code: np.ndarray
    event_type_index: np.ndarray
    event_relative_xyz_m: np.ndarray
    event_identity_index: np.ndarray
    event_mask: np.ndarray
    set_cardinality: np.ndarray
    parent_id: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.global_sequence_index)
        mask = self.event_mask.astype(bool)
        if (
            count != OBSERVATION_COUNT
            or self.global_sequence_index.shape != (count,)
            or self.partition_code.shape != (count,)
            or self.event_type_index.shape != (count, QUERY_COUNT)
            or self.event_relative_xyz_m.shape != (count, QUERY_COUNT, 3)
            or self.event_identity_index.shape != (count, QUERY_COUNT)
            or self.event_mask.shape != (count, QUERY_COUNT)
            or self.set_cardinality.shape != (count,)
            or self.parent_id.shape != (count,)
            or np.any(np.diff(self.global_sequence_index) <= 0)
            or len(np.unique(self.global_sequence_index)) != count
            or not np.all(np.isfinite(self.event_relative_xyz_m))
            or int(mask.sum()) != VISIBLE_TOKEN_COUNT
            or not np.array_equal(mask.sum(axis=1), self.set_cardinality)
            or int(np.sum(self.partition_code == 0)) != FIT_OBSERVATION_COUNT
            or int(np.sum(self.partition_code == 1)) != SELECTION_OBSERVATION_COUNT
            or np.any((self.partition_code != 0) & (self.partition_code != 1))
            or np.any((self.event_type_index[mask] < 0) | (self.event_type_index[mask] > 1))
            or np.any(self.event_type_index[~mask] != -1)
            or np.any(self.event_identity_index[mask] < 0)
            or np.any(self.event_identity_index[~mask] != -1)
            or int(self.set_cardinality.max()) != 5
        ):
            raise RuntimeError("observable spatial event Teacher V2 contract drift")

    @property
    def fit_rows(self) -> np.ndarray:
        return np.flatnonzero(self.partition_code == 0)

    @property
    def selection_rows(self) -> np.ndarray:
        return np.flatnonzero(self.partition_code == 1)

    def targets(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        selected = np.asarray(rows, dtype=np.int64)
        return {
            "event_type_index": np.asarray(self.event_type_index[selected], dtype=np.int64),
            "event_relative_xyz_m": np.asarray(self.event_relative_xyz_m[selected], dtype=np.float32),
            "event_identity_index": np.asarray(self.event_identity_index[selected], dtype=np.int64),
            "event_mask": np.asarray(self.event_mask[selected], dtype=np.uint8),
        }


def load_observable_spatial_event_teacher(
    teacher_root: str | Path,
) -> ObservableSpatialEventTeacherArrays:
    import zarr

    root = Path(teacher_root).resolve()
    paths = sorted(root.glob("*/*.zarr"), key=lambda path: path.stem)
    if len(paths) != 80:
        raise RuntimeError("observable Teacher V2 must contain 80 shards")
    names = (
        "global_sequence_index",
        "event_type_index",
        "event_relative_xyz_m",
        "event_identity_index",
        "event_mask",
        "set_cardinality",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    partitions: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        parent = path.stem
        partition = str(group.attrs.get("partition"))
        expected = "fit" if parent.endswith(tuple(f"C0{value}" for value in range(1, 7))) else "selection"
        if (
            group.attrs.get("schema_version") != "gse_observable_spatial_event_teacher_v2"
            or group.attrs.get("parent_id") != parent
            or partition != expected
        ):
            raise RuntimeError(f"observable Teacher V2 shard identity drift: {parent}")
        length = len(group["global_sequence_index"])
        for name in names:
            chunks[name].append(np.asarray(group[name][:]))
        partitions.append(np.full(length, 0 if partition == "fit" else 1, dtype=np.uint8))
        parents.append(np.full(length, parent, dtype=f"<U{max(1, len(parent))}"))
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return ObservableSpatialEventTeacherArrays(
        global_sequence_index=values["global_sequence_index"].astype(np.int64),
        partition_code=np.concatenate(partitions),
        event_type_index=values["event_type_index"].astype(np.int8),
        event_relative_xyz_m=values["event_relative_xyz_m"].astype(np.float32),
        event_identity_index=values["event_identity_index"].astype(np.int32),
        event_mask=values["event_mask"].astype(np.uint8),
        set_cardinality=values["set_cardinality"].astype(np.uint8),
        parent_id=np.concatenate(parents),
    )


def observable_fit_only_loss_weights(
    teacher: ObservableSpatialEventTeacherArrays,
) -> dict[str, object]:
    rows = teacher.fit_rows
    mask = teacher.event_mask[rows].astype(bool)
    positive = int(mask.sum())
    slots = int(mask.size)
    counts = np.bincount(teacher.event_type_index[rows][mask].astype(np.int64), minlength=2)
    if positive != FIT_TOKEN_COUNT or counts.tolist() != [23136, 75143]:
        raise RuntimeError("observable Teacher V2 fit-only event mass drift")
    return {
        "positive_slots": positive,
        "negative_slots": slots - positive,
        "presence_positive_weight": float((slots - positive) / positive),
        "event_type_counts": counts.tolist(),
        "event_type_class_weights": [float(positive / (2 * value)) for value in counts],
    }


__all__ = [
    "FIT_OBSERVATION_COUNT",
    "FIT_TOKEN_COUNT",
    "OBSERVATION_COUNT",
    "ObservableSpatialEventTeacherArrays",
    "QUERY_COUNT",
    "SELECTION_OBSERVATION_COUNT",
    "VISIBLE_TOKEN_COUNT",
    "load_observable_spatial_event_teacher",
    "observable_fit_only_loss_weights",
]
