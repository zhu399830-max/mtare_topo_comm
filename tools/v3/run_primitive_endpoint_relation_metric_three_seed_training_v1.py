#!/usr/bin/env python3
"""Formal three-seed endpoint relation metric training and full-C07 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_primitive_relation_sparse_port_three_seed_training_v1r import _run_training_monitored


RUN_ID = "gate3_20260904_primitive_endpoint_relation_metric_three_seed_training_v1_seed0"
PASS = "PASS_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_V1"
FAIL = "FAIL_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260904_primitive_endpoint_relation_metric_readiness_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
THRESHOLDS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
EXPECTED_TESTS = 22; EXPECTED_STEPS = 10_233; EXPECTED_TOTAL_STEPS = 30_699
MAX_GPU = 16 * 1024**3; MAX_HOST = 4 * 1024**3; MAX_RESULT = 2 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _size(path: Path) -> int: return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"; files = sorted(value for value in run.rglob("*") if value.is_file() and value != target)
    target.write_text("".join(f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n" for value in files), encoding="utf-8"); return len(files)


def _stage(command, log, env, timeout):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve(); started = time.monotonic()
    overall = FAIL; error = None; scientific_pass = False; checks = {}; evaluation = {}; subprocesses = []; before = {}; optimizer_steps = 0; peaks = {"allocated": 0, "reserved": 0, "gpu_process": 0, "host": 0}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("endpoint metric training executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS: raise RuntimeError(f"Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        readiness = load_json(READINESS / "metrics/summary.json")
        if readiness.get("decision") != "ALLOW_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_DATA_CARD" or readiness.get("scientific_pass") is not True: raise RuntimeError("readiness prerequisite drift")
        versions = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "zarr": zarr.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        expected_versions = {"python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if versions != expected_versions: raise RuntimeError(f"environment drift: {versions}")
        write_json(run / "config/environment.json", {"versions": versions, "platform": platform.platform()}); write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT))); env["PYTHONHASHSEED"] = "0"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = ["tests/v3/unit/test_primitive_endpoint_relation_metric.py", "tests/v3/unit/test_train_primitive_endpoint_relation_metric_v1.py", "tests/v3/unit/test_evaluate_primitive_endpoint_relation_metric_three_seed_v1.py", "tests/v3/unit/test_primitive_local_composition_slot_training.py", "tests/v3/unit/test_primitive_relation_observable_model.py"]
        rc = _stage([PYTHON, "-m", "pytest", "-q", *tests], run / "logs/00_unit_tests.log", env, 600); subprocesses.append({"stage": "unit_tests", "returncode": rc})
        if rc or f"{EXPECTED_TESTS} passed" not in (run / "logs/00_unit_tests.log").read_text(encoding="utf-8"): raise RuntimeError(f"expected exactly {EXPECTED_TESTS} tests")
        models = run / "artifacts/models"; models.mkdir(parents=True)
        roots = {"fit_sensor": P1A / "artifacts/dataset/fit", "fit_teacher": P1B / "artifacts/teacher/fit", "fit_observability": OBS / "artifacts/endpoint_observability/fit", "c07_sensor": P1A / "artifacts/dataset/c07", "c07_teacher": P1B / "artifacts/teacher/c07", "c07_observability": OBS / "artifacts/endpoint_observability/c07"}
        for seed in range(3):
            command = [PYTHON, str(PROJECT_ROOT / "tools/v3/train_primitive_endpoint_relation_metric_v1.py"), "--fit-sensor-root", str(roots["fit_sensor"]), "--fit-teacher-root", str(roots["fit_teacher"]), "--fit-observability-root", str(roots["fit_observability"]), "--c07-sensor-root", str(roots["c07_sensor"]), "--c07-teacher-root", str(roots["c07_teacher"]), "--c07-observability-root", str(roots["c07_observability"]), "--source-checkpoint", str(SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt"), "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed), "--epochs", "3", "--batch-size", "128", "--evaluation-batch-size", "128", "--learning-rate", "0.001", "--weight-decay", "0.0001"]
            (run / f"config/seed{seed}_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            monitor = _run_training_monitored(command, log=run / f"logs/0{seed+1}_seed{seed}_training.log", environment=env, timeout_seconds=36_000, maximum_host_rss_bytes=MAX_HOST); write_json(run / f"metrics/seed{seed}_resource_monitor.json", monitor); subprocesses.append({"stage": f"seed{seed}_training", **monitor})
            if monitor["returncode"] or monitor["killed_for_host_limit"]: raise RuntimeError(f"seed{seed} training failed")
            summary = load_json(models / f"seed{seed}/summary.json"); history = summary.get("history", [])
            seed_checks = {"schema": summary.get("schema_version") == "primitive_endpoint_relation_metric_seed_training_v1", "seed": summary.get("seed") == seed, "steps": summary.get("optimizer_steps") == EXPECTED_STEPS, "epochs": len(history) == 3, "population": all(row["training"]["rows"] == 426552 and row["selection"]["rows"] == 64644 for row in history), "parameters": summary.get("trainable_parameters") == 426818 and summary.get("frozen_parameters") == 2635631, "frozen": summary.get("frozen_state_sha256_before") == summary.get("frozen_state_sha256_after"), "isolation": summary.get("c08_rows_read") == 0, "selected_exact": _sha(models / f"seed{seed}/selected.pt") == _sha(models / f"seed{seed}/epoch_{int(summary['best_epoch']):02d}.pt")}
            write_json(run / f"metrics/seed{seed}_contract_checks.json", seed_checks)
            if not all(seed_checks.values()): raise RuntimeError(f"seed{seed} contract drift: {seed_checks}")
            optimizer_steps += int(summary["optimizer_steps"]); peaks["allocated"] = max(peaks["allocated"], int(summary["peak_cuda_allocated_bytes"])); peaks["reserved"] = max(peaks["reserved"], int(summary["peak_cuda_reserved_bytes"])); peaks["gpu_process"] = max(peaks["gpu_process"], int(summary["peak_gpu_process_memory_bytes"])); peaks["host"] = max(peaks["host"], int(monitor["peak_host_rss_bytes"]))
            if max(peaks["allocated"], peaks["reserved"], peaks["gpu_process"]) > MAX_GPU or peaks["host"] > MAX_HOST:
                raise RuntimeError("resource cap exceeded")
        if optimizer_steps != EXPECTED_TOTAL_STEPS: raise RuntimeError("optimizer step drift")
        output = run / "metrics/evaluation"; command = [PYTHON, str(PROJECT_ROOT / "tools/v3/evaluate_primitive_endpoint_relation_metric_three_seed_v1.py"), "--models-root", str(models), "--sensor-root", str(P1A / "artifacts/dataset"), "--teacher-root", str(P1B / "artifacts/teacher"), "--observability-root", str(OBS / "artifacts/endpoint_observability"), "--baseline-c07-summary", str(BASELINE / "metrics/summary.json"), "--existence-threshold-source", str(THRESHOLDS / "metrics/diagnostic/summary.json"), "--output-dir", str(output)]
        (run / "config/evaluation_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        monitor = _run_training_monitored(command, log=run / "logs/04_c07_evaluation.log", environment=env, timeout_seconds=14_400, maximum_host_rss_bytes=MAX_HOST); write_json(run / "metrics/evaluation_resource_monitor.json", monitor); subprocesses.append({"stage": "c07_evaluation", **monitor}); peaks["host"] = max(peaks["host"], int(monitor["peak_host_rss_bytes"]))
        if monitor["returncode"] not in (0, 2) or monitor["killed_for_host_limit"]: raise RuntimeError("C07 evaluator system failure")
        evaluation = load_json(output / "summary.json"); scientific_pass = bool(evaluation["scientific_pass"]); overall = PASS if scientific_pass else FAIL
        for suffix in ("png", "pdf", "svg"): shutil.copy2(output / f"primitive_endpoint_relation_metric_c07_comparison.{suffix}", run / "previews" / f"primitive_endpoint_relation_metric_c07_comparison.{suffix}")
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks = {"frozen_inputs_unchanged": after == before, "optimizer_steps": optimizer_steps == EXPECTED_TOTAL_STEPS, "three_models": all((models / f"seed{seed}/selected.pt").is_file() for seed in range(3)), "resource_caps": max(peaks["allocated"], peaks["reserved"], peaks["gpu_process"]) <= MAX_GPU and peaks["host"] <= MAX_HOST, "c07_result": evaluation.get("overall_status") in {"PASS_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_C07_V1", "FAIL_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_C07_V1"}, "zero_c08_graph_mtare": evaluation.get("c08_rows_read") == 0 and evaluation.get("graph_replays") == 0 and evaluation.get("mtare_worlds_read") == 0}
        if not all(checks.values()): raise RuntimeError(f"outer evidence drift: {checks}")
        write_json(run / "config/source_integrity_after.json", after); size = _size(run)
        if size > MAX_RESULT: raise RuntimeError("result exceeds 2 GiB")
        write_json(run / "metrics/summary.json", {"schema_version": "primitive_endpoint_relation_metric_three_seed_training_outer_v1", "overall_status": overall, "scientific_pass": scientific_pass, "decision": evaluation["decision"], "checks": checks, "optimizer_steps": optimizer_steps, "peaks": peaks, "result_bytes_before_seal": size, "subprocesses": subprocesses, "evaluation": evaluation, "c08_rows_read": 0, "graph_replays": 0, "mtare_worlds_read": 0, "duration_seconds": time.monotonic() - started, "error": None})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {"schema_version": "primitive_endpoint_relation_metric_three_seed_training_outer_v1", "overall_status": FAIL, "scientific_pass": False, "decision": "STOP_ENDPOINT_RELATION_METRIC_TRAINING_SYSTEM_FAILURE", "checks": checks, "optimizer_steps": optimizer_steps, "subprocesses": subprocesses, "c08_rows_read": 0, "graph_replays": 0, "mtare_worlds_read": 0, "duration_seconds": time.monotonic() - started, "error": error})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error, "duration_seconds": time.monotonic() - started}); entries = _seal(run); print(json.dumps({"overall_status": overall, "error": error, "scientific_pass": scientific_pass, "optimizer_steps": optimizer_steps, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
