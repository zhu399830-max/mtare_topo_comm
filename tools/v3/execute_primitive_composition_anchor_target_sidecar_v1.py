#!/usr/bin/env python3
"""Materialize exact current-sensor composition-anchor Teacher targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from numcodecs import Blosc
import numpy as np
import zarr

from mtare_topo.data.primitive_composition_anchor_sidecar import (
    display_anchor_norm_histogram,
    materialize_current_sensor_anchors,
    validate_materialized_current_sensor_anchors,
)
from mtare_topo.evaluation.primitive_attachment_observability import (
    unique_attachment_indices,
)
from mtare_topo.evaluation.primitive_composition_anchor_teacher import (
    construction_endpoint_anchor_table,
)


BATCH_SIZE = 512
SCHEMA = "primitive_composition_anchor_target_sidecar_v1"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    total = 0
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
        total += value.stat().st_size
    return digest.hexdigest(), len(files), total


def _stream_array_hash(array, *, rows: int) -> str:
    digest = hashlib.sha256()
    dtype = np.dtype(array.dtype).newbyteorder("<")
    digest.update(str(dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    for start in range(0, len(array), rows):
        value = np.asarray(array[start : start + rows], dtype=dtype)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _task(
    row: dict,
    *,
    sensor_root: Path,
    construction_root: Path,
    teacher_root: Path,
    observability_root: Path,
    observability_manifest: dict[str, dict],
    output_root: Path,
) -> dict:
    started = time.monotonic()
    task_id = str(row["task_id"])
    split = str(row["partition"])
    paths = {
        "sensor": sensor_root / split / f"{task_id}.zarr",
        "construction": construction_root / split / f"{task_id}.json",
        "teacher": teacher_root / split / f"{task_id}.zarr",
        "observability": observability_root / split / f"{task_id}.zarr",
    }
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError(f"composition-anchor source missing: {task_id}")
    if task_id not in observability_manifest:
        raise RuntimeError(f"composition-anchor observability manifest missing: {task_id}")

    document = _load(paths["construction"])
    table = construction_endpoint_anchor_table(document)
    sensor = zarr.open_group(str(paths["sensor"]), mode="r")
    teacher = zarr.open_group(str(paths["teacher"]), mode="r")
    observability = zarr.open_group(str(paths["observability"]), mode="r")
    rows = int(teacher["primitive_index"].shape[0])
    if rows != int(row["sequences"]):
        raise RuntimeError(f"composition-anchor task population drift: {task_id}")
    if rows != int(observability["source_global_sequence_index"].shape[0]):
        raise RuntimeError(f"composition-anchor observability rows differ: {task_id}")
    if (
        str(document["parent_id"]) != str(row["world"])
        or str(document["geometry_realization"]) != str(row["realization"])
        or str(teacher.attrs["parent_id"]) != str(row["world"])
        or str(teacher.attrs["geometry_realization"]) != str(row["realization"])
        or str(teacher.attrs["partition"]) != split
        or str(observability.attrs["task_id"]) != task_id
    ):
        raise RuntimeError(f"composition-anchor task provenance differs: {task_id}")

    output = output_root / split / f"{task_id}.zarr"
    if output.exists():
        raise RuntimeError(f"composition-anchor output already exists: {task_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
    group = zarr.open_group(str(output), mode="w")
    target = group.create_dataset(
        "anchor_current_sensor_m",
        shape=(rows, 32, 2, 3),
        chunks=(min(BATCH_SIZE, rows), 32, 2, 3),
        dtype="<f4",
        compressor=compressor,
    )
    sequence_output = group.create_dataset(
        "source_global_sequence_index",
        shape=(rows,),
        chunks=(min(2048, rows),),
        dtype="<i8",
        compressor=compressor,
    )
    observation_row = observability_manifest[task_id]
    group.attrs.update({
        "schema_version": SCHEMA,
        "task_id": task_id,
        "parent_id": str(row["world"]),
        "partition": split,
        "geometry_realization": str(row["realization"]),
        "coordinate_frame": "current_sensor_xyz_yaw_only",
        "inactive_slot_value": 0.0,
        "student_construction_identity_input_forbidden": True,
        "source_p1b_shard_tree_sha256": str(row["shard_tree_sha256"]),
        "source_observability_sidecar_tree_sha256": str(observation_row["sidecar_tree_sha256"]),
        "source_construction_sha256": _sha(paths["construction"]),
    })

    active_primitives = attachment_pairs = active_endpoints = 0
    maximum_float32_error_m = 0.0
    maximum_attachment_anchor_distance_m = 0.0
    maximum_anchor_norm_m = 0.0
    anchor_norm_overflow_count = 0
    norm_histogram = np.zeros(160, dtype=np.int64)
    norm_edges = np.linspace(0.0, 80.0, 161)
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
        index = np.asarray(teacher["primitive_index"][start:stop], dtype=np.int32)
        mask = np.asarray(teacher["primitive_mask"][start:stop], dtype=np.uint8)
        frame_row = np.asarray(teacher["frame_row"][start:stop, -1], dtype=np.int64)
        sequence = np.asarray(
            teacher["source_global_sequence_index"][start:stop], dtype=np.int64,
        )
        observed_sequence = np.asarray(
            observability["source_global_sequence_index"][start:stop], dtype=np.int64,
        )
        if not np.array_equal(sequence, observed_sequence):
            raise RuntimeError(f"composition-anchor sequence alignment drift: {task_id}")
        sensor_xyz = np.asarray(sensor["sensor_xyz_m"].oindex[frame_row], dtype=np.float64)
        yaw = np.asarray(sensor["yaw_deg"].oindex[frame_row], dtype=np.float64)
        value = materialize_current_sensor_anchors(
            primitive_index=index,
            primitive_mask=mask,
            anchor_world_m=table.anchor_world_m,
            sensor_xyz_m=sensor_xyz,
            yaw_deg=yaw,
        )
        replay = materialize_current_sensor_anchors(
            primitive_index=index,
            primitive_mask=mask,
            anchor_world_m=table.anchor_world_m,
            sensor_xyz_m=sensor_xyz,
            yaw_deg=yaw,
        )
        if not np.array_equal(value, replay):
            raise RuntimeError(f"composition-anchor deterministic replay drift: {task_id}")
        validate_materialized_current_sensor_anchors(value, mask)

        active = mask.astype(bool)
        safe_index = np.where(active, index, 0)
        gathered_world = table.anchor_world_m[safe_index]
        delta = gathered_world - sensor_xyz[:, None, None, :]
        angle = np.radians(yaw)[:, None, None]
        reference = delta.copy()
        reference[..., 0] = np.cos(angle) * delta[..., 0] + np.sin(angle) * delta[..., 1]
        reference[..., 1] = -np.sin(angle) * delta[..., 0] + np.cos(angle) * delta[..., 1]
        maximum_float32_error_m = max(
            maximum_float32_error_m,
            float(np.max(np.abs(reference[active] - value[active]), initial=0.0)),
        )

        pairs = unique_attachment_indices(
            np.asarray(teacher["endpoint_neighbor"][start:stop], dtype=np.int8),
        )
        if len(pairs):
            local_row, first, second = pairs.T
            flat = value.reshape(len(value), 64, 3)
            separation = np.linalg.norm(
                flat[local_row, first] - flat[local_row, second], axis=1,
            )
            maximum_attachment_anchor_distance_m = max(
                maximum_attachment_anchor_distance_m,
                float(np.max(separation, initial=0.0)),
            )
            if np.any(separation != 0):
                raise RuntimeError(f"connected endpoints do not share one anchor: {task_id}")
        norms = np.linalg.norm(value, axis=3)[np.repeat(active[:, :, None], 2, axis=2)]
        counts, overflow, maximum = display_anchor_norm_histogram(norms, norm_edges)
        maximum_anchor_norm_m = max(maximum_anchor_norm_m, maximum)
        anchor_norm_overflow_count += overflow
        norm_histogram += counts
        active_primitives += int(np.count_nonzero(active))
        active_endpoints += int(len(norms))
        attachment_pairs += int(len(pairs))
        target[start:stop] = value
        sequence_output[start:stop] = sequence

    reopened = zarr.open_group(str(output), mode="r")
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
        index = np.asarray(teacher["primitive_index"][start:stop], dtype=np.int32)
        mask = np.asarray(teacher["primitive_mask"][start:stop], dtype=np.uint8)
        frame_row = np.asarray(teacher["frame_row"][start:stop, -1], dtype=np.int64)
        expected = materialize_current_sensor_anchors(
            primitive_index=index,
            primitive_mask=mask,
            anchor_world_m=table.anchor_world_m,
            sensor_xyz_m=np.asarray(sensor["sensor_xyz_m"].oindex[frame_row], dtype=np.float64),
            yaw_deg=np.asarray(sensor["yaw_deg"].oindex[frame_row], dtype=np.float64),
        )
        actual = np.asarray(reopened["anchor_current_sensor_m"][start:stop], dtype=np.float32)
        actual_sequence = np.asarray(
            reopened["source_global_sequence_index"][start:stop], dtype=np.int64,
        )
        source_sequence = np.asarray(
            teacher["source_global_sequence_index"][start:stop], dtype=np.int64,
        )
        if not np.array_equal(actual, expected) or not np.array_equal(actual_sequence, source_sequence):
            raise RuntimeError(f"composition-anchor reopen/rederive drift: {task_id}")
        validate_materialized_current_sensor_anchors(actual, mask)

    tree_sha, files, bytes_ = _tree_hash(output)
    return {
        "task_id": task_id,
        "world": str(row["world"]),
        "partition": split,
        "realization": str(row["realization"]),
        "sequences": rows,
        "primitive_count": len(table.primitive_ids),
        "active_primitives": active_primitives,
        "active_endpoints": active_endpoints,
        "attachment_anchor_pairs_checked": attachment_pairs,
        "maximum_attachment_anchor_distance_m": maximum_attachment_anchor_distance_m,
        "maximum_float32_error_m": maximum_float32_error_m,
        "maximum_anchor_norm_m": maximum_anchor_norm_m,
        "anchor_norm_overflow_count": anchor_norm_overflow_count,
        "anchor_norm_histogram_edges_m": norm_edges.tolist(),
        "anchor_norm_histogram_counts": norm_histogram.tolist(),
        "anchor_array_sha256": _stream_array_hash(reopened["anchor_current_sensor_m"], rows=BATCH_SIZE),
        "sequence_array_sha256": _stream_array_hash(reopened["source_global_sequence_index"], rows=2048),
        "source_p1b_shard_tree_sha256": str(row["shard_tree_sha256"]),
        "source_observability_sidecar_tree_sha256": str(observation_row["sidecar_tree_sha256"]),
        "source_construction_sha256": _sha(paths["construction"]),
        "sidecar_tree_sha256": tree_sha,
        "sidecar_files": files,
        "sidecar_bytes": bytes_,
        "duration_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--observability-root", required=True, type=Path)
    parser.add_argument("--teacher-manifest", required=True, type=Path)
    parser.add_argument("--observability-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher_rows = [
        row for row in _load(args.teacher_manifest.resolve())["tasks"]
        if str(row["partition"]) in {"fit", "c07"}
    ]
    teacher_rows.sort(key=lambda row: (str(row["partition"]), str(row["task_id"])))
    observation_rows = {
        str(row["task_id"]): row
        for row in _load(args.observability_manifest.resolve())["tasks"]
    }
    records = []
    progress = output / "task_progress.jsonl"
    with progress.open("w", encoding="utf-8") as stream:
        for position, row in enumerate(teacher_rows, start=1):
            record = _task(
                row,
                sensor_root=args.sensor_root.resolve(),
                construction_root=args.construction_root.resolve(),
                teacher_root=args.teacher_root.resolve(),
                observability_root=args.observability_root.resolve(),
                observability_manifest=observation_rows,
                output_root=output / "anchor_targets",
            )
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps({
                "completed": position, "of": len(teacher_rows),
                "task_id": record["task_id"], "sidecar_bytes": record["sidecar_bytes"],
            }), flush=True)
    split = {}
    for name in ("fit", "c07"):
        subset = [row for row in records if row["partition"] == name]
        histogram = np.sum(
            np.asarray([row["anchor_norm_histogram_counts"] for row in subset], dtype=np.int64),
            axis=0,
        )
        split[name] = {
            "parents": len({row["world"] for row in subset}),
            "tasks": len(subset),
            "sequences": sum(row["sequences"] for row in subset),
            "active_primitives": sum(row["active_primitives"] for row in subset),
            "active_endpoints": sum(row["active_endpoints"] for row in subset),
            "attachment_anchor_pairs_checked": sum(
                row["attachment_anchor_pairs_checked"] for row in subset
            ),
            "maximum_anchor_norm_m": max(row["maximum_anchor_norm_m"] for row in subset),
            "anchor_norm_overflow_count": sum(row["anchor_norm_overflow_count"] for row in subset),
            "sidecar_bytes": sum(row["sidecar_bytes"] for row in subset),
            "anchor_norm_histogram_edges_m": records[0]["anchor_norm_histogram_edges_m"],
            "anchor_norm_histogram_counts": histogram.tolist(),
        }
    summary = {
        "schema_version": SCHEMA,
        "tasks": len(records),
        "sequences": sum(row["sequences"] for row in records),
        "sidecar_bytes": sum(row["sidecar_bytes"] for row in records),
        "maximum_attachment_anchor_distance_m": max(
            row["maximum_attachment_anchor_distance_m"] for row in records
        ),
        "maximum_float32_error_m": max(row["maximum_float32_error_m"] for row in records),
        "maximum_anchor_norm_m": max(row["maximum_anchor_norm_m"] for row in records),
        "anchor_norm_overflow_count": sum(row["anchor_norm_overflow_count"] for row in records),
        "source_sequence_alignment_exact": True,
        "deterministic_full_rederive_exact": True,
        "inactive_slots_exact_zero": True,
        "split": split,
    }
    _write(output / "task_manifest.json", {"schema_version": SCHEMA, "tasks": records})
    _write(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
