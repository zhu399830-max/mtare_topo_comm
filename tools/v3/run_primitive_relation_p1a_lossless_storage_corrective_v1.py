#!/usr/bin/env python3
"""Formal storage-only corrective for the immutable P1a sensor corpus."""

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
from mtare_topo.data.primitive_relation_lossless_repack import repack_zarr_group_losslessly
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_P1A_LOSSLESS_STORAGE_CORRECTIVE_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_P1A_LOSSLESS_STORAGE_CORRECTIVE_V1"
SOURCE_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"


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
                raise RuntimeError(f"sealed source P1a drift: {relative}")
            checked += 1
    if checked < 1:
        raise RuntimeError("source P1a evidence manifest is empty")
    return checked


def _task_paths(root: Path, task: dict) -> tuple[Path, Path, Path]:
    partition = str(task["partition"]); task_id = str(task["task_id"])
    return (
        root / "dataset" / partition / f"{task_id}.zarr",
        root / "codebooks" / partition / f"{task_id}.json",
        root / "constructions" / partition / f"{task_id}.json",
    )


def _repack_task(task: dict) -> dict:
    started = time.monotonic()
    source_dataset, source_codebook, source_construction = _task_paths(SOURCE_RUN / "artifacts", task)
    output_root = Path(task["output_root"])
    destination_dataset, destination_codebook, destination_construction = _task_paths(output_root, task)
    source_tree, source_files, source_bytes = _tree_hash(source_dataset)
    if source_tree != task["shard_tree_sha256"] or source_files != int(task["shard_files"]) or source_bytes != int(task["shard_bytes"]):
        raise RuntimeError(f"source task tree drift: {task['task_id']}")
    destination_dataset.parent.mkdir(parents=True, exist_ok=True)
    destination_codebook.parent.mkdir(parents=True, exist_ok=True)
    destination_construction.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
    arrays = repack_zarr_group_losslessly(source_dataset, destination_dataset, compressor=compressor)
    shutil.copy2(source_codebook, destination_codebook)
    shutil.copy2(source_construction, destination_construction)
    if _sha(source_codebook) != _sha(destination_codebook) or _sha(source_construction) != _sha(destination_construction):
        raise RuntimeError(f"side document byte drift: {task['task_id']}")
    destination_tree, destination_files, destination_bytes = _tree_hash(destination_dataset)
    if len(arrays) != 11 or not all(value.chunks_checked > 0 and len(value.raw_sha256) == 64 for value in arrays):
        raise RuntimeError(f"unexpected array evidence inventory: {task['task_id']}")
    return {
        **task,
        "source_shard_tree_sha256": source_tree,
        "source_shard_bytes": source_bytes,
        "corrected_shard_tree_sha256": destination_tree,
        "corrected_shard_files": destination_files,
        "corrected_shard_bytes": destination_bytes,
        "array_count": len(arrays),
        "array_raw_evidence": [value.as_dict() for value in arrays],
        "compressor": {"codec": "blosc", "cname": "zstd", "clevel": 9, "shuffle": "byte"},
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "duration_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); parser.add_argument("--workers", type=int, default=12); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve()); started = time.monotonic(); overall, error = FAIL, None
    try:
        if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("P1a lossless storage corrective executes exactly once")
        if args.workers != 12 or spec.get("gate") != 3 or spec.get("operation") != "data_export" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("scope/worker/authorization mismatch")
        source_summary = load_json(SOURCE_RUN / "metrics/summary.json"); source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
        source_checks = dict(source_summary.get("checks", {}))
        if source_state.get("state") != "COMPLETED" or source_state.get("error") is not None:
            raise RuntimeError("source P1a did not complete cleanly")
        if source_summary.get("overall_status") != "FAIL_PRIMITIVE_RELATION_P1A_SENSOR_PROVENANCE_EXPORT_V1":
            raise RuntimeError("corrective requires the frozen P1a FAIL result")
        false_checks = sorted(name for name, value in source_checks.items() if not value)
        if false_checks != ["result_bytes_le_20gib"]:
            raise RuntimeError(f"corrective permits only the storage check to fail, got {false_checks}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = os.environ.copy(); environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        tests = subprocess.run(
            [sys.executable, "tests/v3/unit/test_primitive_relation_lossless_repack.py"],
            cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "Ran 2 tests" not in tests.stdout or "OK" not in tests.stdout:
            raise RuntimeError("expected exactly two lossless repack unit tests")
        verified_source_files = _verify_evidence_manifest(SOURCE_RUN / "artifacts/evidence_sha256.txt")
        source_manifest = load_json(SOURCE_RUN / "artifacts/task_manifest.json")["tasks"]
        if len(source_manifest) != 240:
            raise RuntimeError("expected exactly 240 source tasks")
        tasks = [{**task, "output_root": str(run_dir / "artifacts")} for task in source_manifest]
        rows = []; progress_path = run_dir / "logs/task_progress.jsonl"; context = mp.get_context("spawn")
        with progress_path.open("w", encoding="utf-8") as progress, ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {pool.submit(_repack_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result(); rows.append(row); progress.write(json.dumps(row, sort_keys=True) + "\n"); progress.flush()
                print(json.dumps({"completed": len(rows), "of": len(tasks), "task": row["task_id"], "mib": row["corrected_shard_bytes"] / 1024**2}, sort_keys=True), flush=True)
        rows.sort(key=lambda value: value["task_id"])
        source_bytes = sum(int(value["source_shard_bytes"]) for value in rows)
        corrected_bytes = sum(int(value["corrected_shard_bytes"]) for value in rows)
        checks = {
            "source_failure_is_resource_only": false_checks == ["result_bytes_le_20gib"],
            "exact_240_tasks_80_worlds": len(rows) == 240 and len({value["world"] for value in rows}) == 80,
            "source_counts_preserved": sum(value["frames"] for value in rows) == 757290 and sum(value["rays"] for value in rows) == 8723980800 and sum(value["primitive_count"] for value in rows) == 24117 and sum(value["pose_corrections"] for value in rows) == 402,
            "exact_11_arrays_per_shard": all(value["array_count"] == 11 for value in rows),
            "all_array_chunks_bit_exact": all(len(array["raw_sha256"]) == 64 and array["chunks_checked"] > 0 for value in rows for array in value["array_raw_evidence"]),
            "all_source_trees_verified": all(value["source_shard_tree_sha256"] == value["shard_tree_sha256"] for value in rows),
            "all_corrected_trees_nonempty_hashed": all(value["corrected_shard_bytes"] > 0 and len(value["corrected_shard_tree_sha256"]) == 64 for value in rows),
            "corrected_bytes_lt_source_bytes": corrected_bytes < source_bytes,
            "corrected_bytes_le_20gib": corrected_bytes <= 20 * 1024**3,
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run_dir / "artifacts/task_manifest.json", {"schema_version": "primitive_relation_p1a_lossless_storage_corrective_task_manifest_v1", "tasks": rows})
        for name in ("traversal_manifest.jsonl", "frame_pose_corrections.json"):
            shutil.copy2(SOURCE_RUN / "artifacts" / name, run_dir / "artifacts" / name)
        shutil.copy2(SOURCE_RUN / "artifacts/evidence_sha256.txt", run_dir / "artifacts/source_p1a_evidence_sha256.txt")
        summary = {
            "schema_version": "primitive_relation_p1a_lossless_storage_corrective_v1", "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "tasks": len(rows), "worlds": len({value["world"] for value in rows}),
            "frames": sum(value["frames"] for value in rows), "rays": sum(value["rays"] for value in rows),
            "primitives": sum(value["primitive_count"] for value in rows), "paired_pose_corrections": sum(value["pose_corrections"] for value in rows),
            "source_dataset_bytes": source_bytes, "dataset_bytes": corrected_bytes,
            "compression_ratio": corrected_bytes / source_bytes, "verified_source_evidence_files": verified_source_files,
            "workers": args.workers, "optimizer_steps": 0, "model_inference_frames": 0, "c09_c10_worlds_read": 0, "graph_replays": 0,
            "decision": "ALLOW_P1B_RELATION_TEACHER_MATERIALIZATION_FROM_CORRECTED_P1A" if overall == PASS else "STOP_P1A_STORAGE_CORRECTIVE_FAILED",
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "config/environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "zarr": zarr.__version__, "workers": args.workers, "compressor": "Blosc(zstd, clevel=9, byte-shuffle)"})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
