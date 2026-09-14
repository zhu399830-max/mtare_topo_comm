"""Deterministic C01--C08 pair construction for open-set GSE association."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np


_COHORT = re.compile(r"_C(\d{2})$")


def cohort_number(parent_id: str) -> int:
    match = _COHORT.search(str(parent_id))
    if match is None:
        raise ValueError(f"parent_id has no frozen Cxx cohort suffix: {parent_id}")
    return int(match.group(1))


def corrective_partition(parent_id: str) -> str:
    cohort = cohort_number(parent_id)
    if 1 <= cohort <= 6:
        return "fit"
    if 7 <= cohort <= 8:
        return "selection"
    raise ValueError("open-set corrective may read only C01--C08")


def compact_global_sequence_identity(
    global_sequence_indices: Iterable[int],
) -> tuple[np.ndarray, dict[int, int]]:
    """Return a deterministic compact row mapping for sparse global IDs.

    ``global_sequence_index`` belongs to the complete C01--C10 export.  A
    C01--C08-only corrective therefore has legal holes wherever reserved C09
    records occur.  Array rows must be compact, while the original global IDs
    remain the immutable join key retained in evidence.
    """

    values = np.asarray(list(global_sequence_indices), dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("global sequence identities must be a nonempty nonnegative vector")
    ordered = np.sort(values, kind="stable")
    if np.any(np.diff(ordered) <= 0):
        raise ValueError("global sequence identities must be unique")
    return ordered, {int(global_id): row for row, global_id in enumerate(ordered)}


def nearest_past_open_set_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    maximum_distance_m: float = 16.0,
) -> list[dict[str, Any]]:
    """Pair each invalid query with its nearest strictly-past valid candidate.

    Records are grouped by parent and ordered by ``world_sequence_row``.  Ties
    are resolved by the lower global sequence index.  Queries with no valid
    past candidate inside the fixed radius remain unpaired, never relabeled.
    """

    if not np.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise ValueError("maximum_distance_m must be positive and finite")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in records:
        row = dict(source)
        parent = str(row["parent_id"])
        corrective_partition(parent)
        xyz = np.asarray(row["sensor_xyz_m"], dtype=np.float64)
        if (
            xyz.shape != (3,)
            or not np.all(np.isfinite(xyz))
            or int(row["global_sequence_index"]) < 0
            or int(row["world_sequence_row"]) < 0
        ):
            raise ValueError("open-set record has invalid position or sequence identity")
        row["sensor_xyz_m"] = xyz
        row["association_valid"] = bool(row["association_valid"])
        grouped.setdefault(parent, []).append(row)
    result: list[dict[str, Any]] = []
    for parent in sorted(grouped):
        rows = sorted(
            grouped[parent],
            key=lambda row: (int(row["world_sequence_row"]), int(row["global_sequence_index"])),
        )
        world_rows = [int(row["world_sequence_row"]) for row in rows]
        if len(set(world_rows)) != len(world_rows):
            raise ValueError(f"duplicate world_sequence_row in {parent}")
        valid_past: list[dict[str, Any]] = []
        for query in rows:
            if query["association_valid"]:
                valid_past.append(query)
                continue
            if not valid_past:
                continue
            positions = np.stack([candidate["sensor_xyz_m"] for candidate in valid_past])
            distances = np.linalg.norm(positions - query["sensor_xyz_m"], axis=1)
            eligible = np.flatnonzero(distances <= maximum_distance_m + 1e-12)
            if len(eligible) == 0:
                continue
            candidate_index = min(
                eligible.tolist(),
                key=lambda index: (
                    float(distances[index]), int(valid_past[index]["global_sequence_index"])
                ),
            )
            candidate = valid_past[candidate_index]
            if int(candidate["world_sequence_row"]) >= int(query["world_sequence_row"]):
                raise RuntimeError("open-set candidate is not strictly past")
            result.append(
                {
                    "anchor_global_sequence_index": int(query["global_sequence_index"]),
                    "paired_global_sequence_index": int(candidate["global_sequence_index"]),
                    "parent_id": parent,
                    "same_identity": False,
                    "pair_kind": "open_set_query_nearest_past_structural_negative",
                    "spatial_distance_m": float(distances[candidate_index]),
                    "partition": corrective_partition(parent),
                }
            )
    result.sort(key=lambda row: (row["anchor_global_sequence_index"], row["paired_global_sequence_index"]))
    return result


def load_corrective_sequence_metadata(dataset_run: str | Path) -> list[dict[str, Any]]:
    """Read only identity/order/pose metadata from the sealed C01--C08 shards."""

    root = Path(dataset_run).resolve()
    manifest = root / "artifacts/sequence_manifest.jsonl"
    records = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            if source["split"] != "train" or cohort_number(source["parent_id"]) > 8:
                continue
            records.append(source)
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_parent.setdefault(str(record["parent_id"]), []).append(record)
    output: list[dict[str, Any]] = []
    import zarr

    for parent in sorted(by_parent):
        group = zarr.open_group(str(root / "artifacts/dataset/train" / f"{parent}.zarr"), mode="r")
        rows = sorted(by_parent[parent], key=lambda row: int(row["world_sequence_row"]))
        sequence_rows = np.asarray([int(row["world_sequence_row"]) for row in rows], dtype=np.int64)
        anchor_rows = np.asarray([int(row["world_anchor_frame_row"]) for row in rows], dtype=np.int64)
        valid = np.asarray(group["association_valid_mask"].oindex[sequence_rows], dtype=np.uint8)
        xyz = np.asarray(group["sensor_xyz_m"].oindex[anchor_rows], dtype=np.float64)
        if valid.shape != (len(rows),) or xyz.shape != (len(rows), 3):
            raise RuntimeError(f"C01--C08 metadata arrays are misaligned for {parent}")
        for record, is_valid, position in zip(rows, valid, xyz, strict=True):
            output.append(
                {
                    "parent_id": parent,
                    "world_sequence_row": int(record["world_sequence_row"]),
                    "global_sequence_index": int(record["global_sequence_index"]),
                    "sensor_xyz_m": position,
                    "association_valid": bool(is_valid),
                }
            )
    return output


__all__ = [
    "cohort_number",
    "compact_global_sequence_identity",
    "corrective_partition",
    "load_corrective_sequence_metadata",
    "nearest_past_open_set_pairs",
]
