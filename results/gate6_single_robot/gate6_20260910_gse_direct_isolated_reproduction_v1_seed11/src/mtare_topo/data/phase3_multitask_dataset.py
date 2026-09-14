"""Sealed V2R range-image dataset contract for Phase-3 multitask learning."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROLE_TO_INDEX = {"interior": 0, "junction": 1, "terminal": 2}
INDEX_TO_ROLE = tuple(ROLE_TO_INDEX)
RANGE_NORMALIZER_M = 50.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_phase3_manifest(
    frame_records: Iterable[dict[str, Any]],
    cluster_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate frame/cluster identity and all three objective supervision heads."""

    frames = list(frame_records)
    clusters = list(cluster_records)
    cluster_by_id = {str(item["cluster_id"]): item for item in clusters}
    if len(cluster_by_id) != len(clusters):
        raise ValueError("duplicate cluster_id")
    frame_ids: set[str] = set()
    frame_counts: Counter[str] = Counter()
    split_frames: Counter[str] = Counter()
    split_clusters: Counter[str] = Counter()
    role_clusters: Counter[tuple[str, str]] = Counter()
    branch_frames: Counter[tuple[str, int]] = Counter()
    for cluster in clusters:
        role = str(cluster["primary_role"])
        split = str(cluster["split"])
        if role not in ROLE_TO_INDEX or split not in {"train", "validation"}:
            raise ValueError(f"invalid cluster role/split: {role}/{split}")
        split_clusters[split] += 1
        role_clusters[(split, role)] += 1
    for frame in frames:
        frame_id = str(frame["frame_id"])
        if frame_id in frame_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        frame_ids.add(frame_id)
        cluster_id = str(frame["cluster_id"])
        if cluster_id not in cluster_by_id:
            raise ValueError(f"unknown cluster_id: {cluster_id}")
        cluster = cluster_by_id[cluster_id]
        for key in ("parent_id", "split", "primary_role", "near_junction", "near_terminal", "tunnel_id"):
            if frame[key] != cluster[key]:
                raise ValueError(f"{frame_id}: cluster metadata mismatch for {key}")
        branch_count = int(frame["branch_count"])
        if not 1 <= branch_count <= 6:
            raise ValueError(f"{frame_id}: unsupported branch_count={branch_count}")
        if len(frame["headings_robot_deg"]) != branch_count:
            raise ValueError(f"{frame_id}: heading/count mismatch")
        frame_counts[cluster_id] += 1
        split_frames[str(frame["split"])] += 1
        branch_frames[(str(frame["split"]), branch_count)] += 1
    bad_clusters = {key: value for key, value in frame_counts.items() if value != 5}
    missing_clusters = sorted(set(cluster_by_id) - set(frame_counts))
    if bad_clusters or missing_clusters:
        raise ValueError(f"five-frame cluster contract failed: bad={bad_clusters}, missing={missing_clusters}")
    return {
        "passed": True,
        "frames": len(frames),
        "clusters": len(clusters),
        "split_frames": dict(split_frames),
        "split_clusters": dict(split_clusters),
        "role_clusters": {f"{split}:{role}": count for (split, role), count in sorted(role_clusters.items())},
        "branch_frames": {f"{split}:{count}": value for (split, count), value in sorted(branch_frames.items())},
    }


class CanoV2RMultitaskDataset:
    """Read only approved student arrays and objective targets from sealed shards."""

    def __init__(self, run_dir: str | Path, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.run_dir = Path(run_dir).resolve()
        self.split = split
        all_records = _read_jsonl(self.run_dir / "artifacts/manifest.jsonl")
        self.records = [item for item in all_records if item["split"] == split]
        self._groups: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _group(self, parent_id: str):
        import zarr

        if parent_id not in self._groups:
            path = self.run_dir / "artifacts/dataset" / self.split / f"{parent_id}.zarr"
            self._groups[parent_id] = zarr.open_group(str(path), mode="r")
        return self._groups[parent_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        group = self._group(str(record["parent_id"]))
        row = int(record["zarr_row"])
        range_m = np.asarray(group["range_m"][row], dtype=np.float32)
        valid = np.asarray(group["valid_mask"][row], dtype=np.float32)
        direction = np.asarray(group["exit_target"][row], dtype=np.float32)
        if range_m.shape != (16, 720) or valid.shape != (16, 720) or direction.shape != (720,):
            raise RuntimeError(f"{record['frame_id']}: array shape contract failed")
        if not np.isfinite(range_m).all() or not np.isfinite(direction).all():
            raise RuntimeError(f"{record['frame_id']}: non-finite array")
        student = np.stack((np.clip(range_m, 0.3, RANGE_NORMALIZER_M) / RANGE_NORMALIZER_M, valid))
        return {
            "student": student.astype(np.float32, copy=False),
            "direction_target": direction,
            "count_target": np.int64(int(record["branch_count"]) - 1),
            "role_target": np.int64(ROLE_TO_INDEX[str(record["primary_role"])]),
            "frame_id": str(record["frame_id"]),
            "cluster_id": str(record["cluster_id"]),
            "parent_id": str(record["parent_id"]),
            "headings_robot_deg": tuple(float(value) for value in record["headings_robot_deg"]),
        }
