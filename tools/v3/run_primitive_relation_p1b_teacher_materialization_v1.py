#!/usr/bin/env python3
"""Formal P1b materialization of five-frame primitive geometry/relation targets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_frame_support import (
    PrimitiveFrameSupport,
    summarize_primitive_frame_support,
    visible_primitive_targets_from_frame_support,
)
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry, world_primitive_relation_sequence_references
from mtare_topo.data.primitive_relation_storage import pack_primitive_relation_targets
from mtare_topo.data.primitive_relation_targets import primitive_relation_targets_from_frame_support
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


RUN_ID = "gate3_20260830_primitive_relation_p1b_teacher_materialization_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_P1B_TEACHER_MATERIALIZATION_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_P1B_TEACHER_MATERIALIZATION_V1"
P1A_DIRECT_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"
P1A_CORRECTED_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
ALLOWED_P1A_RUNS = {P1A_DIRECT_RUN.resolve(), P1A_CORRECTED_RUN.resolve()}
MAXIMUM_SLOTS = 32


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256(); files = sorted(value for value in path.rglob("*") if value.is_file()); total = 0
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative); digest.update(bytes.fromhex(_sha(value)))
        total += value.stat().st_size
    return digest.hexdigest(), len(files), total


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run_dir.rglob("*") if value.is_file() and value != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for value in files:
            stream.write(f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _verify_evidence_manifest(path: Path) -> int:
    checked = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            expected, relative = line.rstrip("\n").split("  ", 1)
            target = PROJECT_ROOT / relative
            if not target.is_file() or _sha(target) != expected:
                raise RuntimeError(f"sealed P1a artifact drift: {relative}")
            checked += 1
    if checked < 1:
        raise RuntimeError("P1a evidence manifest is empty")
    return checked


def _load_traversals(source_run: Path) -> list[dict]:
    values = []
    with (source_run / "artifacts/traversal_manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            values.append(json.loads(line))
    return values


def _array(group, name: str, shape: tuple[int, ...], dtype: str, compressor, chunk: int = 256):
    return group.create_dataset(name, shape=shape, chunks=(max(1, min(chunk, shape[0])), *shape[1:]), dtype=dtype, compressor=compressor)


def _task_paths(task: dict) -> tuple[Path, Path, Path]:
    partition = str(task["partition"]); task_id = str(task["task_id"])
    source_run = Path(task["source_run"])
    return (
        source_run / "artifacts/dataset" / partition / f"{task_id}.zarr",
        source_run / "artifacts/codebooks" / partition / f"{task_id}.json",
        source_run / "artifacts/constructions" / partition / f"{task_id}.json",
    )


def _export_task(task: dict) -> dict:
    started = time.monotonic(); sensor_path, codebook_path, construction_path = _task_paths(task)
    source = zarr.open_group(str(sensor_path), mode="r")
    codebook = load_json(codebook_path); construction_document = load_json(construction_path)
    construction, primitives = load_p1a_realized_construction(construction_document)
    primitive_ids = tuple(value.primitive_id for value in primitives)
    if tuple(codebook["primitive_ids"]) != primitive_ids:
        raise RuntimeError("codebook/construction primitive ordering drift")
    source_sets = tuple(tuple(int(index) for index in value) for value in codebook["source_sets"])
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
    frame_count = int(source["range_m"].shape[0]); primitive_count = len(primitives)
    sensor_xyz = source["sensor_xyz_m"][:]
    yaw_deg = source["yaw_deg"][:]
    counts = np.empty((frame_count, primitive_count), dtype=np.int32)
    low = np.empty((frame_count, primitive_count), dtype=np.int32)
    high = np.empty((frame_count, primitive_count), dtype=np.int32)
    azimuth = np.empty((frame_count, primitive_count, 90), dtype=np.uint8)
    for start in range(0, frame_count, 16):
        stop = min(start + 16, frame_count)
        ranges = source["range_m"][start:stop]; codes = source["primitive_membership_code"][start:stop]
        origins = sensor_xyz[start:stop]; yaws = yaw_deg[start:stop]
        for offset in range(stop - start):
            support = summarize_primitive_frame_support(
                range_m=ranges[offset], primitive_membership_code=codes[offset], source_sets=source_sets,
                field=field, sensor_xyz_m=origins[offset], yaw_deg=float(yaws[offset]),
            )
            counts[start + offset] = support.support_ray_count; low[start + offset] = support.minimum_sample_index
            high[start + offset] = support.maximum_sample_index; azimuth[start + offset] = support.azimuth_support_packed

    references = world_primitive_relation_sequence_references(
        parent_id=str(task["world"]), traversal_manifest=task["traversals"],
        shard_global_frame_indices=source["global_frame_index"][:], realization_index=int(task["realization_index"]),
    )
    sequence_count = len(references.source_global_sequence_index)
    arrays = {
        "primitive_index": np.full((sequence_count, 32), -1, dtype=np.int32),
        "primitive_mask": np.zeros((sequence_count, 32), dtype=np.uint8),
        "axis_control_current_sensor_m": np.zeros((sequence_count, 32, 3, 3), dtype=np.float32),
        "endpoint_half_axes_m": np.zeros((sequence_count, 32, 2, 2), dtype=np.float32),
        "endpoint_shape_exponent": np.zeros((sequence_count, 32, 2), dtype=np.float32),
        "support_ray_count": np.zeros((sequence_count, 32), dtype=np.int32),
        "temporal_visibility": np.zeros((sequence_count, 5, 32), dtype=np.uint8),
        "frame_primitive_index": np.full((sequence_count, 5, 32), -1, dtype=np.int32),
        "temporal_destination": np.full((sequence_count, 5, 32), -1, dtype=np.int8),
        "endpoint_neighbor": np.full((sequence_count, 32, 2, 3), -1, dtype=np.int8),
        "disconnected_overlap_packed": np.zeros((sequence_count, 32, 4), dtype=np.uint8),
        "relative_translation_current_sensor_m": np.zeros((sequence_count, 5, 3), dtype=np.float32),
        "relative_yaw_current_sensor_deg": np.zeros((sequence_count, 5), dtype=np.float32),
    }
    maximum_visible = attachment_count = overlap_count = dustbin_count = 0; first_digest = None
    for sequence_index, rows in enumerate(references.frame_row):
        supports = tuple(PrimitiveFrameSupport(counts[row], low[row], high[row], azimuth[row]) for row in rows)
        current = int(rows[-1])
        visible = visible_primitive_targets_from_frame_support(
            field=field, frame_supports=supports, current_sensor_xyz_m=sensor_xyz[current],
            current_yaw_deg=float(yaw_deg[current]), maximum_slots=MAXIMUM_SLOTS,
        )
        relation = primitive_relation_targets_from_frame_support(
            construction=construction, primitive_ids=primitive_ids, frame_supports=supports, maximum_slots=MAXIMUM_SLOTS,
        )
        if not np.array_equal(visible.primitive_index, relation.primitive_index):
            raise RuntimeError("geometry and relation aggregate slots disagree")
        packed = pack_primitive_relation_targets(relation)
        odometry = causal_relative_odometry(sensor_xyz[rows], yaw_deg[rows])
        arrays["primitive_index"][sequence_index] = visible.primitive_index
        arrays["primitive_mask"][sequence_index] = visible.mask
        arrays["axis_control_current_sensor_m"][sequence_index] = visible.axis_control_current_sensor_m
        arrays["endpoint_half_axes_m"][sequence_index] = visible.endpoint_half_axes_m
        arrays["endpoint_shape_exponent"][sequence_index] = visible.endpoint_shape_exponent
        arrays["support_ray_count"][sequence_index] = visible.support_ray_count
        arrays["temporal_visibility"][sequence_index] = visible.temporal_visibility
        arrays["frame_primitive_index"][sequence_index] = relation.frame_primitive_index
        arrays["temporal_destination"][sequence_index] = packed.temporal_destination
        arrays["endpoint_neighbor"][sequence_index] = packed.endpoint_neighbor
        arrays["disconnected_overlap_packed"][sequence_index] = packed.disconnected_overlap_packed
        arrays["relative_translation_current_sensor_m"][sequence_index] = odometry.translation_current_sensor_m
        arrays["relative_yaw_current_sensor_deg"][sequence_index] = odometry.yaw_current_sensor_deg
        active = int(np.sum(visible.mask)); maximum_visible = max(maximum_visible, active)
        attachment_count += int(np.sum(relation.endpoint_attachment))
        overlap_count += int(np.sum(relation.disconnected_angular_overlap) // 2)
        dustbin_count += int(np.sum(packed.temporal_destination == 32))
        if sequence_index == 0:
            digest = hashlib.sha256()
            for name in arrays: digest.update(arrays[name][0].tobytes())
            first_digest = digest.hexdigest()

    output = Path(task["output_root"]) / "teacher" / str(task["partition"]) / f"{task['task_id']}.zarr"
    if output.exists(): raise RuntimeError("P1b task output already exists")
    output.parent.mkdir(parents=True, exist_ok=True); compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group = zarr.open_group(str(output), mode="w")
    group.attrs.update({
        "schema_version": "primitive_relation_p1b_teacher_shard_v1", "parent_id": str(task["world"]),
        "partition": str(task["partition"]), "geometry_realization": str(task["realization"]),
        "maximum_slots": 32, "window_frames": 5, "student_identity_input_forbidden": True,
        "unexplored_port_semantics": "deterministic execution state, deliberately not a perception Teacher label",
    })
    references_to_write = {
        "source_global_sequence_index": references.source_global_sequence_index,
        "variant_global_sequence_index": references.variant_global_sequence_index,
        "frame_row": references.frame_row,
    }
    for name, values in {**references_to_write, **arrays}.items():
        data = np.asarray(values); target = _array(group, name, data.shape, data.dtype.str, compressor)
        target[:] = data
    group.attrs["traversal_ids"] = list(references.traversal_id)
    reopened = zarr.open_group(str(output), mode="r")
    if reopened["primitive_index"].shape != (sequence_count, 32): raise RuntimeError("P1b written shard shape drift")
    repeat = hashlib.sha256()
    for name in arrays: repeat.update(reopened[name][0].tobytes())
    if sequence_count and repeat.hexdigest() != first_digest: raise RuntimeError("P1b first target write/reopen drift")
    tree_sha, file_count, shard_bytes = _tree_hash(output)
    return {
        "task_id": str(task["task_id"]), "world": str(task["world"]), "partition": str(task["partition"]),
        "realization": str(task["realization"]), "frames": frame_count, "sequences": sequence_count,
        "primitive_count": primitive_count, "maximum_visible_primitives": maximum_visible,
        "directed_attachment_labels": attachment_count, "undirected_disconnected_overlap_labels": overlap_count,
        "temporal_dustbin_labels": dustbin_count, "first_target_digest": first_digest,
        "shard_tree_sha256": tree_sha, "shard_files": file_count, "shard_bytes": shard_bytes,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "duration_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); parser.add_argument("--workers", type=int, default=12); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve()); started = time.monotonic(); overall, error = FAIL, None
    try:
        if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("P1b executes exactly once")
        if args.workers != 12 or spec.get("gate") != 3 or spec.get("operation") != "teacher_generation" or spec.get("user_authorization", {}).get("status") != "APPROVED": raise RuntimeError("scope/worker/authorization mismatch")
        source_run = (PROJECT_ROOT / str(spec["source_p1a_run"])).resolve()
        if source_run not in ALLOWED_P1A_RUNS: raise RuntimeError("unrecognized P1a source run")
        p1a_summary = load_json(source_run / "metrics/summary.json"); p1a_state = load_json(source_run / "RUN_STATE.json")
        if not p1a_summary.get("scientific_pass") or p1a_state.get("state") != "COMPLETED": raise RuntimeError("P1a is not a sealed scientific PASS")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = os.environ.copy(); environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        test_paths = [
            "tests/v3/unit/test_primitive_frame_support.py",
            "tests/v3/unit/test_primitive_relation_targets.py",
            "tests/v3/unit/test_primitive_relation_storage.py",
            "tests/v3/unit/test_primitive_relation_sequences.py",
            "tests/v3/unit/test_primitive_relation_materialization.py",
        ]
        tests = subprocess.run(
            ["/home/zeng-workstation/anaconda3/bin/python", "-m", "pytest", "-q", *test_paths],
            cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "10 passed" not in tests.stdout: raise RuntimeError("expected exactly 10 P1b unit tests")
        pilot = subprocess.run(
            [sys.executable, "tools/v3/check_primitive_relation_p1b_real_shard_contract.py"],
            cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/real_shard_contract.log").write_text(pilot.stdout, encoding="utf-8")
        if pilot.returncode or "PASS_PRIMITIVE_RELATION_P1B_REAL_SHARD_CONTRACT" not in pilot.stdout:
            raise RuntimeError("real P1b repeated-shard contract failed")
        verified_p1a_files = _verify_evidence_manifest(source_run / "artifacts/evidence_sha256.txt")
        manifest = load_json(source_run / "artifacts/task_manifest.json")["tasks"]; traversals = _load_traversals(source_run)
        traversal_groups: dict[str, list[dict]] = {}
        for traversal in traversals: traversal_groups.setdefault(str(traversal["parent_id"]), []).append(traversal)
        tasks = [{**row, "source_run": str(source_run), "realization_index": {"ellipse": 0, "rounded_rectangle": 1, "c1_mixed": 2}[row["realization"]], "traversals": traversal_groups[row["world"]], "output_root": str(run_dir / "artifacts")} for row in manifest]
        rows = []; progress_path = run_dir / "logs/task_progress.jsonl"; context = mp.get_context("spawn")
        with progress_path.open("w", encoding="utf-8") as progress, ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(_export_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result(); rows.append(row); progress.write(json.dumps(row, sort_keys=True) + "\n"); progress.flush()
                print(json.dumps({"completed": len(rows), "of": len(tasks), "task": row["task_id"], "sequences": row["sequences"]}, sort_keys=True), flush=True)
        rows.sort(key=lambda value: value["task_id"]); total_bytes = sum(value["shard_bytes"] for value in rows)
        checks = {
            "exact_240_tasks_80_worlds": len(rows) == 240 and len({value["world"] for value in rows}) == 80,
            "exact_757290_source_frames": sum(value["frames"] for value in rows) == 757290,
            "exact_564378_variant_sequences": sum(value["sequences"] for value in rows) == 564378,
            "exact_24117_realized_primitives": sum(value["primitive_count"] for value in rows) == 24117,
            "capacity_never_exceeds_32": max(value["maximum_visible_primitives"] for value in rows) <= 32,
            "attachment_supervision_nonempty": sum(value["directed_attachment_labels"] for value in rows) > 0,
            "disconnected_overlap_supervision_nonempty": sum(value["undirected_disconnected_overlap_labels"] for value in rows) > 0,
            "temporal_dustbin_supervision_nonempty": sum(value["temporal_dustbin_labels"] for value in rows) > 0,
            "all_shards_hashed": all(value["shard_bytes"] > 0 and len(value["shard_tree_sha256"]) == 64 for value in rows),
            "result_bytes_le_12gib": total_bytes <= 12 * 1024**3,
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run_dir / "artifacts/task_manifest.json", {"schema_version": "primitive_relation_p1b_task_manifest_v1", "tasks": rows})
        shutil.copy2(source_run / "artifacts/evidence_sha256.txt", run_dir / "artifacts/p1a_evidence_sha256.txt")
        summary = {
            "schema_version": "primitive_relation_p1b_teacher_materialization_v1", "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "tasks": len(rows),
            "frames": sum(value["frames"] for value in rows), "sequences": sum(value["sequences"] for value in rows),
            "realized_primitives": sum(value["primitive_count"] for value in rows),
            "maximum_visible_primitives": max(value["maximum_visible_primitives"] for value in rows),
            "directed_attachment_labels": sum(value["directed_attachment_labels"] for value in rows),
            "undirected_disconnected_overlap_labels": sum(value["undirected_disconnected_overlap_labels"] for value in rows),
            "temporal_dustbin_labels": sum(value["temporal_dustbin_labels"] for value in rows),
            "dataset_bytes": total_bytes, "workers": args.workers, "optimizer_steps": 0, "model_inference_frames": 0,
            "verified_p1a_evidence_files": verified_p1a_files,
            "source_p1a_run": str(source_run.relative_to(PROJECT_ROOT)),
            "c09_c10_worlds_read": 0, "graph_replays": 0, "duration_seconds": time.monotonic() - started,
            "decision": "ALLOW_P2_PRIMITIVE_RELATION_MODEL_TRAINING" if overall == PASS else "STOP_P1B_TEACHER_FAILED",
            "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "config/environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "zarr": zarr.__version__, "workers": args.workers})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
