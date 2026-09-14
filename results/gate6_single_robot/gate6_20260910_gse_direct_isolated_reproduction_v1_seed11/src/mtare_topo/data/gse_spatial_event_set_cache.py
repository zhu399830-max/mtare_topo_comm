"""Frozen causal feature cache and joined Teacher for spatial event sets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


OBSERVATION_COUNT = 188_126
FIT_OBSERVATION_COUNT = 142_184
SELECTION_OBSERVATION_COUNT = 45_942
VISIBLE_TOKEN_COUNT = 133_055
QUERY_COUNT = 16
ENCODER_DIM = 128
DIRECTIONAL_BINS = 180


@dataclass(frozen=True)
class SpatialEventTeacherArrays:
    """One exact global-index join of the fixed-16 native-LOS Teacher."""

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
        ):
            raise RuntimeError("spatial event Teacher population/shape drift")
        mask = self.event_mask.astype(bool)
        if (
            int(mask.sum()) != VISIBLE_TOKEN_COUNT
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
            raise RuntimeError("spatial event Teacher semantic contract drift")

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
            "event_relative_xyz_m": np.asarray(
                self.event_relative_xyz_m[selected], dtype=np.float32
            ),
            "event_identity_index": np.asarray(
                self.event_identity_index[selected], dtype=np.int64
            ),
            "event_mask": np.asarray(self.event_mask[selected], dtype=np.uint8),
        }


def load_spatial_event_teacher(teacher_root: str | Path) -> SpatialEventTeacherArrays:
    """Load only the 80 C01--C08 shards and require exact global ordering."""

    import zarr

    root = Path(teacher_root).resolve()
    paths = sorted(root.glob("fit/*.zarr")) + sorted(root.glob("selection/*.zarr"))
    # Parent ids sort globally, while partition directories do not. Re-sort by
    # shard basename before concatenating to preserve global sequence order.
    paths = sorted(paths, key=lambda path: path.stem)
    if len(paths) != 80:
        raise RuntimeError("spatial event Teacher must contain exactly 80 shards")
    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "global_sequence_index",
            "event_type_index",
            "event_relative_xyz_m",
            "event_identity_index",
            "event_mask",
            "set_cardinality",
        )
    }
    partitions: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        parent_id = path.stem
        partition = str(group.attrs.get("partition"))
        expected_partition = "fit" if parent_id.endswith(tuple(f"C0{x}" for x in range(1, 7))) else "selection"
        if group.attrs.get("parent_id") != parent_id or partition != expected_partition:
            raise RuntimeError(f"spatial event Teacher shard identity drift: {parent_id}")
        length = len(group["global_sequence_index"])
        for name in chunks:
            chunks[name].append(np.asarray(group[name][:]))
        partitions.append(np.full(length, 0 if partition == "fit" else 1, dtype=np.uint8))
        parents.append(np.full(length, parent_id, dtype=f"<U{max(1, len(parent_id))}"))
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return SpatialEventTeacherArrays(
        global_sequence_index=values["global_sequence_index"].astype(np.int64),
        partition_code=np.concatenate(partitions),
        event_type_index=values["event_type_index"].astype(np.int8),
        event_relative_xyz_m=values["event_relative_xyz_m"].astype(np.float32),
        event_identity_index=values["event_identity_index"].astype(np.int32),
        event_mask=values["event_mask"].astype(np.uint8),
        set_cardinality=values["set_cardinality"].astype(np.uint8),
        parent_id=np.concatenate(parents),
    )


class SpatialEventSetFeatureCache:
    """Memory-map one seed's frozen five-frame causal encoder outputs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "gse_spatial_event_set_feature_cache_v1"
            or manifest.get("observations") != OBSERVATION_COUNT
            or manifest.get("encoder_dim") != ENCODER_DIM
            or manifest.get("directional_bins") != DIRECTIONAL_BINS
        ):
            raise RuntimeError("spatial event set feature cache schema drift")
        self.manifest = manifest
        self.global_sequence_index = np.load(
            self.cache_dir / "global_sequence_index.npy", mmap_mode="r"
        )
        self.context = np.load(self.cache_dir / "context.npy", mmap_mode="r")
        self.directional = np.load(self.cache_dir / "directional.npy", mmap_mode="r")
        self.legacy_event_logits = np.load(
            self.cache_dir / "legacy_event_logits.npy", mmap_mode="r"
        )
        if (
            self.global_sequence_index.shape != (OBSERVATION_COUNT,)
            or self.context.shape != (OBSERVATION_COUNT, ENCODER_DIM)
            or self.directional.shape != (OBSERVATION_COUNT, ENCODER_DIM, DIRECTIONAL_BINS)
            or self.legacy_event_logits.shape != (OBSERVATION_COUNT, 5)
            or self.global_sequence_index.dtype != np.int64
            or self.context.dtype != np.float16
            or self.directional.dtype != np.float16
            or self.legacy_event_logits.dtype != np.float16
            or np.any(np.diff(self.global_sequence_index) <= 0)
        ):
            raise RuntimeError("spatial event set feature cache array drift")

    def gather(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        selected = np.asarray(rows, dtype=np.int64)
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= OBSERVATION_COUNT)
        ):
            raise ValueError("invalid spatial event cache rows")
        return {
            "context": np.asarray(self.context[selected], dtype=np.float32),
            "directional": np.asarray(self.directional[selected], dtype=np.float32),
            "legacy_event_logits": np.asarray(
                self.legacy_event_logits[selected], dtype=np.float32
            ),
        }


def fit_only_loss_weights(teacher: SpatialEventTeacherArrays) -> dict[str, object]:
    """Return deterministic inverse-frequency weights from C01--C06 only."""

    rows = teacher.fit_rows
    mask = teacher.event_mask[rows].astype(bool)
    positive = int(mask.sum())
    slots = int(mask.size)
    types = teacher.event_type_index[rows][mask]
    counts = np.bincount(types.astype(np.int64), minlength=2)
    if positive != 99_492 or counts.tolist() != [23_193, 76_299]:
        raise RuntimeError("fit-only event mass drift")
    return {
        "positive_slots": positive,
        "negative_slots": slots - positive,
        "presence_positive_weight": float((slots - positive) / positive),
        "event_type_counts": counts.tolist(),
        "event_type_class_weights": [float(positive / (2 * value)) for value in counts],
    }


__all__ = [
    "DIRECTIONAL_BINS",
    "ENCODER_DIM",
    "FIT_OBSERVATION_COUNT",
    "OBSERVATION_COUNT",
    "QUERY_COUNT",
    "SELECTION_OBSERVATION_COUNT",
    "SpatialEventSetFeatureCache",
    "SpatialEventTeacherArrays",
    "fit_only_loss_weights",
    "load_spatial_event_teacher",
]
