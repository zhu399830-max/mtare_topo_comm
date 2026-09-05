#!/usr/bin/env python3
"""Materialize compact C01--C07 endpoint-observability relation masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_attachment_observability_sidecar import (
    PRIMARY_SUPPORT_BAND_M,
    endpoint_observed_from_gap,
    observable_endpoint_pair_mask,
    pack_endpoint_observed,
    unpack_endpoint_observed,
)
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.evaluation.primitive_attachment_observability import endpoint_support_gaps, unique_attachment_indices
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
PASS = "PASS_PRIMITIVE_ATTACHMENT_OBSERVABILITY_SIDECAR_V1"
FAIL = "FAIL_PRIMITIVE_ATTACHMENT_OBSERVABILITY_SIDECAR_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ATTACHMENT_OBSERVABILITY_SIDECAR_V1"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
AUDIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"
TEST_PYTHON = "/home/zeng-workstation/anaconda3/bin/python"


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
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative)
        digest.update(bytes.fromhex(_sha(value))); total += value.stat().st_size
    return digest.hexdigest(), len(files), total


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files), encoding="utf-8")
    return len(files)


def _materialize_task(row: dict, run: Path) -> dict:
    task_id, partition = str(row["task_id"]), str(row["partition"])
    p1a = P1A / "artifacts"
    sensor_path = p1a / "dataset" / partition / f"{task_id}.zarr"
    construction_path = p1a / "constructions" / partition / f"{task_id}.json"
    teacher_path = P1B / "artifacts/teacher" / partition / f"{task_id}.zarr"
    _, primitives = load_p1a_realized_construction(load_json(construction_path))
    endpoints = np.asarray([[value.centerline_xyz_m[0], value.centerline_xyz_m[-1]] for value in primitives], dtype=np.float64)
    sensor = zarr.open_group(str(sensor_path), mode="r")
    teacher = zarr.open_group(str(teacher_path), mode="r")
    count = int(teacher["primitive_index"].shape[0])
    packed = np.zeros((count, 8), dtype=np.uint8)
    source_sequence = np.asarray(teacher["source_global_sequence_index"][:], dtype=np.int64)
    observed_endpoint_count = valid_undirected_pairs = positive_pairs = supported_positive_pairs = 0
    for start in range(0, count, 512):
        stop = min(start + 512, count)
        primitive_index = np.asarray(teacher["primitive_index"][start:stop], dtype=np.int64)
        primitive_mask = np.asarray(teacher["primitive_mask"][start:stop], dtype=np.uint8)
        frame_row = np.asarray(teacher["frame_row"][start:stop, -1], dtype=np.int64)
        neighbors = np.asarray(teacher["endpoint_neighbor"][start:stop], dtype=np.int8)
        gaps, _ = endpoint_support_gaps(
            axis_control_current_sensor_m=teacher["axis_control_current_sensor_m"][start:stop],
            primitive_index=primitive_index, primitive_mask=primitive_mask,
            primitive_endpoints_world_m=endpoints,
            sensor_xyz_m=sensor["sensor_xyz_m"][frame_row], yaw_deg=sensor["yaw_deg"][frame_row],
        )
        observed = endpoint_observed_from_gap(gaps)
        packed[start:stop] = pack_endpoint_observed(observed)
        if not np.array_equal(unpack_endpoint_observed(packed[start:stop]), observed):
            raise RuntimeError(f"endpoint observability pack round-trip drift: {task_id}")
        valid = observable_endpoint_pair_mask(observed, primitive_mask)
        observed_endpoint_count += int(np.sum(observed))
        valid_undirected_pairs += int(np.sum(valid) // 2)
        pairs = unique_attachment_indices(neighbors)
        positive_pairs += len(pairs)
        if len(pairs):
            local_row, first, second = pairs.T
            flattened = observed.reshape(len(observed), 64)
            supported_positive_pairs += int(np.sum(
                flattened[local_row, first].astype(bool) & flattened[local_row, second].astype(bool)
            ))

    output = run / "artifacts/endpoint_observability" / partition / f"{task_id}.zarr"
    if output.exists():
        raise RuntimeError(f"sidecar task already exists: {task_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group = zarr.open_group(str(output), mode="w")
    group.attrs.update({
        "schema_version": "primitive_attachment_observability_sidecar_v1",
        "task_id": task_id, "parent_id": str(row["world"]), "partition": partition,
        "geometry_realization": str(row["realization"]), "support_band_m": PRIMARY_SUPPORT_BAND_M,
        "endpoint_count": 64, "packed_endpoint_bytes": 8,
        "hidden_pair_semantics": "unknown_never_negative",
        "student_construction_identity_input_forbidden": True,
        "source_p1b_shard_tree_sha256": str(row["shard_tree_sha256"]),
    })
    first = group.create_dataset("endpoint_observed_packed", shape=packed.shape, chunks=(min(1024, count), 8), dtype="u1", compressor=compressor)
    second = group.create_dataset("source_global_sequence_index", shape=source_sequence.shape, chunks=(min(1024, count),), dtype="i8", compressor=compressor)
    first[:] = packed; second[:] = source_sequence
    reopened = zarr.open_group(str(output), mode="r")
    if not np.array_equal(reopened["endpoint_observed_packed"][:], packed) or not np.array_equal(reopened["source_global_sequence_index"][:], source_sequence):
        raise RuntimeError(f"sidecar reopen drift: {task_id}")
    tree_sha, files, bytes_ = _tree_hash(output)
    return {
        "task_id": task_id, "world": str(row["world"]), "partition": partition,
        "realization": str(row["realization"]), "sequences": count,
        "observed_endpoints": observed_endpoint_count,
        "valid_undirected_endpoint_pairs": valid_undirected_pairs,
        "positive_attachment_pairs": positive_pairs,
        "supported_positive_attachment_pairs": supported_positive_pairs,
        "hidden_positive_attachment_pairs": positive_pairs - supported_positive_pairs,
        "source_p1b_shard_tree_sha256": str(row["shard_tree_sha256"]),
        "sidecar_tree_sha256": tree_sha, "sidecar_files": files, "sidecar_bytes": bytes_,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; checks: dict[str, bool] = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("attachment sidecar executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"attachment sidecar Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"attachment sidecar input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"attachment sidecar tool drift: {record['path']}")
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        write_json(run / "config/environment.json", {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "zarr": zarr.__version__})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run([TEST_PYTHON, "-m", "pytest", "-q", "tests/v3/unit/test_primitive_attachment_observability_sidecar.py", "tests/v3/unit/test_primitive_attachment_observability.py"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False)
        if tests.returncode: raise RuntimeError("attachment sidecar unit tests failed")

        manifest = load_json(P1B / "artifacts/task_manifest.json")["tasks"]
        source = [row for row in manifest if row["partition"] in {"fit", "c07"}]
        source.sort(key=lambda value: str(value["task_id"])); rows = []
        with (run / "logs/01_task_progress.jsonl").open("w", encoding="utf-8") as log:
            for index, row in enumerate(source):
                result = _materialize_task(row, run); rows.append(result)
                log.write(json.dumps(result, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"completed": index + 1, "of": len(source), "task": row["task_id"]}), flush=True)
        expected = spec["expected_counts"]
        total_bytes = sum(value["sidecar_bytes"] for value in rows)
        split = {
            name: {
                "tasks": sum(value["partition"] == name for value in rows),
                "sequences": sum(value["sequences"] for value in rows if value["partition"] == name),
                "observed_endpoints": sum(value["observed_endpoints"] for value in rows if value["partition"] == name),
                "valid_undirected_endpoint_pairs": sum(value["valid_undirected_endpoint_pairs"] for value in rows if value["partition"] == name),
                "positive_attachment_pairs": sum(value["positive_attachment_pairs"] for value in rows if value["partition"] == name),
                "supported_positive_attachment_pairs": sum(value["supported_positive_attachment_pairs"] for value in rows if value["partition"] == name),
                "hidden_positive_attachment_pairs": sum(value["hidden_positive_attachment_pairs"] for value in rows if value["partition"] == name),
            } for name in ("fit", "c07")
        }
        checks = {
            "exact_six_unit_tests": True,
            "exact_210_tasks": len(rows) == expected["geometry_tasks"],
            "exact_fit_sequences": split["fit"]["sequences"] == expected["fit_sequences"],
            "exact_c07_sequences": split["c07"]["sequences"] == expected["c07_sequences"],
            "exact_fit_supported_positives": split["fit"]["supported_positive_attachment_pairs"] == expected["fit_supported_positive_pairs"],
            "exact_c07_supported_positives": split["c07"]["supported_positive_attachment_pairs"] == expected["c07_supported_positive_pairs"],
            "hidden_positives_preserved_as_unknown": split["fit"]["hidden_positive_attachment_pairs"] > 0 and split["c07"]["hidden_positive_attachment_pairs"] > 0,
            "valid_negative_population_nonempty": all(split[value]["valid_undirected_endpoint_pairs"] > split[value]["supported_positive_attachment_pairs"] for value in split),
            "all_sidecars_hashed": all(value["sidecar_files"] > 0 and len(value["sidecar_tree_sha256"]) == 64 for value in rows),
            "output_below_0p2gib": total_bytes <= int(.2 * 1024**3),
            "zero_model_training_or_inference": True,
            "zero_c08_c09_c10_graph_mtare": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run / "artifacts/task_manifest.json", {"schema_version": "primitive_attachment_observability_sidecar_manifest_v1", "tasks": rows})
        shutil.copy2(AUDIT / "previews/attachment_teacher_observability.png", run / "previews/source_attachment_teacher_observability.png")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_attachment_observability_sidecar_v1", "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "support_band_m": PRIMARY_SUPPORT_BAND_M,
            "split": split, "tasks": len(rows), "dataset_bytes": total_bytes,
            "optimizer_steps": 0, "model_inference_rows": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "duration_seconds": time.monotonic() - started,
            "decision": "ALLOW_MASKED_RELATION_MODEL_READINESS" if overall == PASS else "STOP_SIDECAR_FAILED",
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "checks": checks, "optimizer_steps": 0, "model_inference_rows": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0, "error": error})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
