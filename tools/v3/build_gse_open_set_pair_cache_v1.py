#!/usr/bin/env python3
"""Freeze the reproducible C01--C08 open-set association pair population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_open_set_pairs import (
    cohort_number,
    compact_global_sequence_identity,
    corrective_partition,
    load_corrective_sequence_metadata,
    nearest_past_open_set_pairs,
)


EXPECTED = {
    "fit_existing": 78166,
    "selection_existing": 27261,
    "fit_open_set": 57066,
    "selection_open_set": 18111,
    "fit_total": 135232,
    "selection_total": 45372,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def build(dataset_run: Path, output_dir: Path) -> dict[str, Any]:
    dataset_run = dataset_run.resolve()
    output_dir = output_dir.resolve()
    dataset_run.relative_to(PROJECT_ROOT)
    output_dir.relative_to(PROJECT_ROOT)
    if output_dir.exists():
        raise RuntimeError("open-set pair cache output already exists")
    output_dir.mkdir(parents=True)
    metadata = load_corrective_sequence_metadata(dataset_run)
    if len(metadata) != 188126:
        raise RuntimeError("C01--C08 raw sequence metadata population drift")
    raw_global_indices = [int(row["global_sequence_index"]) for row in metadata]
    if len(set(raw_global_indices)) != len(raw_global_indices):
        raise RuntimeError("C01--C08 sequence metadata has duplicate global identity")
    by_global = {int(row["global_sequence_index"]): row for row in metadata}
    if len(by_global) != 188126:
        raise RuntimeError("C01--C08 sequence metadata population drift")
    open_rows = nearest_past_open_set_pairs(metadata, maximum_distance_m=16.0)

    existing_path = dataset_run / "artifacts/association_pairs_numeric.jsonl"
    existing_rows = []
    with existing_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cohort = cohort_number(row["parent_id"])
            if cohort > 8:
                continue
            left = int(row["anchor_global_sequence_index"])
            right = int(row["paired_global_sequence_index"])
            if left not in by_global or right not in by_global:
                raise RuntimeError("existing pair points outside C01--C08 metadata")
            distance = float(np.linalg.norm(
                np.asarray(by_global[left]["sensor_xyz_m"]) - np.asarray(by_global[right]["sensor_xyz_m"])
            ))
            existing_rows.append(
                {
                    "anchor_global_sequence_index": left,
                    "paired_global_sequence_index": right,
                    "parent_id": str(row["parent_id"]),
                    "same_identity": bool(row["same_identity"]),
                    "pair_kind": str(row["pair_kind"]),
                    "spatial_distance_m": distance,
                    "partition": corrective_partition(row["parent_id"]),
                }
            )
    all_rows = sorted(
        existing_rows + open_rows,
        key=lambda row: (
            0 if row["partition"] == "fit" else 1,
            int(row["anchor_global_sequence_index"]),
            int(row["paired_global_sequence_index"]),
            str(row["pair_kind"]),
        ),
    )
    counts = {
        "fit_existing": sum(row["partition"] == "fit" for row in existing_rows),
        "selection_existing": sum(row["partition"] == "selection" for row in existing_rows),
        "fit_open_set": sum(row["partition"] == "fit" for row in open_rows),
        "selection_open_set": sum(row["partition"] == "selection" for row in open_rows),
        "fit_total": sum(row["partition"] == "fit" for row in all_rows),
        "selection_total": sum(row["partition"] == "selection" for row in all_rows),
    }
    if counts != EXPECTED:
        raise RuntimeError(f"corrective pair counts drift: {counts}")
    global_indices, compact_by_global = compact_global_sequence_identity(by_global)
    if (
        global_indices.shape != (188126,)
        or np.any(np.diff(global_indices) <= 0)
        or global_indices[0] != 0
        or global_indices[-1] != 208227
    ):
        raise RuntimeError("C01--C08 global sequence identity order drift")
    valid = np.asarray([by_global[index]["association_valid"] for index in global_indices], dtype=np.uint8)
    parent = np.asarray([by_global[index]["parent_id"] for index in global_indices], dtype="U64")
    partition_code = np.asarray(
        [0 if corrective_partition(value) == "fit" else 1 for value in parent], dtype=np.uint8
    )
    xyz = np.stack([by_global[index]["sensor_xyz_m"] for index in global_indices]).astype(np.float64)
    arrays: dict[str, np.ndarray] = {
        "compact_to_global_sequence_index": global_indices,
        "association_valid": valid,
        "parent_id": parent,
        "partition_code": partition_code,
        "sensor_xyz_m": xyz,
    }
    for partition in ("fit", "selection"):
        selected = [row for row in all_rows if row["partition"] == partition]
        arrays[f"{partition}_left"] = np.asarray(
            [compact_by_global[int(row["anchor_global_sequence_index"])] for row in selected], dtype=np.int64
        )
        arrays[f"{partition}_right"] = np.asarray(
            [compact_by_global[int(row["paired_global_sequence_index"])] for row in selected], dtype=np.int64
        )
        arrays[f"{partition}_left_global_sequence_index"] = np.asarray(
            [row["anchor_global_sequence_index"] for row in selected], dtype=np.int64
        )
        arrays[f"{partition}_right_global_sequence_index"] = np.asarray(
            [row["paired_global_sequence_index"] for row in selected], dtype=np.int64
        )
        arrays[f"{partition}_label"] = np.asarray(
            [row["same_identity"] for row in selected], dtype=np.uint8
        )
        arrays[f"{partition}_distance_m"] = np.asarray(
            [row["spatial_distance_m"] for row in selected], dtype=np.float32
        )
        arrays[f"{partition}_family"] = np.asarray(
            [str(row["parent_id"]).split("_", 1)[0] for row in selected], dtype="U3"
        )
    np.savez_compressed(output_dir / "pairs.npz", **arrays)
    _write_jsonl(output_dir / "open_set_pairs.jsonl", open_rows)
    summary = {
        "schema_version": "gse_open_set_pair_cache_v1",
        "overall_status": "PASS_GSE_OPEN_SET_PAIR_CACHE_V1",
        "definition": "same_parent_strictly_past_3d_euclidean_distance_le_16m",
        "counts": counts,
        "worlds": {"fit": 60, "selection": 20},
        "sequences": {"fit": 142184, "selection": 45942},
        "global_sequence_identity": {
            "minimum": int(global_indices[0]),
            "maximum": int(global_indices[-1]),
            "strictly_increasing": True,
            "legal_c09_gaps": int(np.sum(np.diff(global_indices) > 1)),
            "missing_inside_span": int(global_indices[-1] - global_indices[0] + 1 - len(global_indices))
        },
        "identity_valid": {
            "fit": int(np.sum(valid[partition_code == 0])),
            "selection": int(np.sum(valid[partition_code == 1])),
        },
        "pairs_npz_sha256": _sha256(output_dir / "pairs.npz"),
        "open_set_jsonl_sha256": _sha256(output_dir / "open_set_pairs.jsonl"),
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    if summary["identity_valid"] != {"fit": 39310, "selection": 13693}:
        raise RuntimeError("identity-valid partition counts drift")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset_run, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
