"""Load pose/route metadata for causal GSE validation replay without scan leakage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class GSEReplayWorld:
    parent_id: str
    traversal_records: tuple[dict[str, Any], ...]
    teacher_observations: tuple[dict[str, Any], ...]
    pose_by_sequence_index: dict[int, dict[str, Any]]


def load_gse_replay_worlds(
    dataset_run: str | Path,
    teacher_run: str | Path,
    *,
    split: str = "validation",
) -> dict[str, GSEReplayWorld]:
    if split not in {"train", "validation"}:
        raise ValueError("GSE replay loader permits development train or validation only")
    dataset_run = Path(dataset_run).resolve()
    teacher_run = Path(teacher_run).resolve()
    sequence_rows: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(dataset_run / "artifacts/sequence_manifest.jsonl"):
        if str(record["split"]) == split:
            sequence_rows.setdefault(str(record["parent_id"]), []).append(record)
    traversal_rows: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(teacher_run / "artifacts/traversal_manifest.jsonl"):
        if str(record["split"]) == split:
            traversal_rows.setdefault(str(record["parent_id"]), []).append(record)
    teacher_rows: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(teacher_run / "artifacts/teacher_observations.jsonl"):
        if str(record["split"]) == split:
            teacher_rows.setdefault(str(record["parent_id"]), []).append(record)
    parents = sorted(set(sequence_rows) | set(traversal_rows) | set(teacher_rows))
    if not parents or any(parent not in sequence_rows or parent not in traversal_rows or parent not in teacher_rows for parent in parents):
        raise RuntimeError("replay source manifests have different parent populations")

    import zarr

    worlds: dict[str, GSEReplayWorld] = {}
    global_sequences: set[int] = set()
    for parent in parents:
        group = zarr.open_group(str(dataset_run / "artifacts/dataset" / split / f"{parent}.zarr"), mode="r")
        poses: dict[int, dict[str, Any]] = {}
        observation_ids = set()
        for record in sequence_rows[parent]:
            row = int(record["world_sequence_row"])
            sequence_index = int(record["global_sequence_index"])
            if sequence_index in global_sequences:
                raise RuntimeError("global sequence appears in multiple replay worlds")
            if int(group["global_sequence_index"][row]) != sequence_index:
                raise RuntimeError("sequence manifest and Zarr global index disagree")
            local_references = np.asarray(group["local_frame_references"][row], dtype=np.int64)
            if local_references.shape != (5,) or np.any(np.diff(local_references) <= 0):
                raise RuntimeError("replay sequence does not have five ordered frame references")
            anchor = int(local_references[-1])
            if int(record["world_anchor_frame_row"]) != anchor:
                raise RuntimeError("sequence anchor row and Zarr reference disagree")
            axis = np.asarray(group["axis_xyz_m"][anchor], dtype=np.float64)
            sensor = np.asarray(group["sensor_xyz_m"][anchor], dtype=np.float64)
            tangent = np.asarray(group["tangent_world_xyz"][anchor], dtype=np.float64)
            yaw = float(group["yaw_deg"][anchor])
            if axis.shape != (3,) or sensor.shape != (3,) or tangent.shape != (3,) or not np.all(np.isfinite(np.concatenate((axis, sensor, tangent, (yaw,))))):
                raise RuntimeError("replay pose metadata is non-finite")
            poses[sequence_index] = {
                "axis_xyz_m": axis.tolist(),
                "sensor_xyz_m": sensor.tolist(),
                "tangent_world_xyz": tangent.tolist(),
                "yaw_deg": yaw,
                "route_arc_m": float(group["route_arc_m"][anchor]),
                "observation_id": str(record["observation_id"]),
                "traversal_id": str(record["traversal_id"]),
            }
            observation_ids.add(str(record["observation_id"]))
            global_sequences.add(sequence_index)
        teacher_ids = {str(record["observation_id"]) for record in teacher_rows[parent]}
        teacher_sequences = {int(record["global_sequence_index"]) for record in teacher_rows[parent]}
        if teacher_ids != observation_ids or teacher_sequences != set(poses):
            raise RuntimeError("Teacher observations and replay pose rows are not bijective")
        traversals = tuple(sorted(traversal_rows[parent], key=lambda record: str(record["traversal_id"])))
        observations = tuple(sorted(teacher_rows[parent], key=lambda record: int(record["global_sequence_index"])))
        worlds[parent] = GSEReplayWorld(
            parent_id=parent,
            traversal_records=traversals,
            teacher_observations=observations,
            pose_by_sequence_index=poses,
        )
    return worlds


__all__ = ["GSEReplayWorld", "load_gse_replay_worlds"]
