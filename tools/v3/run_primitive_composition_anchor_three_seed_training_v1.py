#!/usr/bin/env python3
"""Formal three-seed head-only composition-anchor training and C07 gate."""

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
from run_primitive_relation_sparse_port_three_seed_training_v1r import (
    _run_training_monitored,
)


RUN_ID = "gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0"
PASS = "PASS_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_TRAINING_V1"
FAIL = "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_TRAINING_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_TRAINING_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"

P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHORS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
BASELINE_C07 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"

EXPECTED_TESTS = 37
EXPECTED_STEPS_PER_SEED = 10_233
EXPECTED_TOTAL_STEPS = 30_699
EXPECTED_FIT_ROWS = 426_552
EXPECTED_C07_ROWS = 64_644
EXPECTED_HEAD_PARAMETERS = 22_278
EXPECTED_FROZEN_PARAMETERS = 2_635_631
MAXIMUM_GPU_BYTES = 16 * 1024**3
MAXIMUM_HOST_RSS_BYTES = 4 * 1024**3
MAXIMUM_RESULT_BYTES = 2 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run.rglob("*") if value.is_file() and value != target)
    target.write_text("".join(
        f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n" for value in files
    ), encoding="utf-8")
    return len(files)


def _run_stage(command: list[str], *, log: Path, environment: dict[str, str], timeout_seconds: int) -> int:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
            timeout=timeout_seconds, check=False,
        )
    return int(completed.returncode)


def _prerequisites_valid() -> dict[str, bool]:
    required_pass = {
        "p1a": P1A,
        "p1b": P1B,
        "observability": OBSERVABILITY,
        "anchors": ANCHORS,
        "readiness": READINESS,
        "model_readiness": MODEL_READINESS,
        "baseline_c07": BASELINE_C07,
    }
    checks: dict[str, bool] = {}
    for name, source in required_pass.items():
        state = load_json(source / "RUN_STATE.json")
        summary = load_json(source / "metrics/summary.json")
        checks[name] = (
            state.get("state") == "COMPLETED"
            and state.get("error") is None
            and summary.get("scientific_pass") is True
        )
    diagnostic_state = load_json(DIAGNOSTIC / "RUN_STATE.json")
    diagnostic_summary = load_json(DIAGNOSTIC / "metrics/summary.json")
    checks["diagnostic"] = (
        diagnostic_state.get("state") == "COMPLETED"
        and diagnostic_state.get("error") is None
        and diagnostic_summary.get("decision")
        == "ALLOW_OE_COMPOSITION_ANCHOR_RESIDUAL_UNCERTAINTY_MODEL_READINESS"
    )
    source_state = load_json(SOURCE_MODELS / "RUN_STATE.json")
    checks["source_models_completed_for_reuse"] = (
        source_state.get("state") == "COMPLETED" and source_state.get("error") is None
        and all((SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt").is_file() for seed in range(3))
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    scientific_pass = False
    optimizer_steps = 0
    peak_cuda = peak_gpu_process = peak_host_rss = 0
    before: dict[str, str] = {}
    subprocesses: list[dict] = []
    evaluation_summary: dict = {}
    checks: dict[str, bool] = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("composition-anchor three-seed run executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"composition-anchor training Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"composition-anchor training input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"composition-anchor training tool drift: {record['path']}")
        prerequisite_checks = _prerequisites_valid()
        if not all(prerequisite_checks.values()):
            raise RuntimeError(f"composition-anchor training prerequisite drift: {prerequisite_checks}")
        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "zarr": zarr.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected_environment:
            raise RuntimeError(f"composition-anchor training environment drift: {environment}")
        write_json(run / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
        })
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/prerequisite_checks.json", prerequisite_checks)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_composition_anchor_batches.py",
            "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
            "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
            "tests/v3/unit/test_primitive_composition_anchor_model.py",
            "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
            "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py",
        ]
        returncode = _run_stage(
            [PYTHON, "-m", "pytest", "-q", *tests],
            log=run / "logs/00_unit_tests.log", environment=env, timeout_seconds=600,
        )
        subprocesses.append({"stage": "unit_tests", "returncode": returncode})
        test_text = (run / "logs/00_unit_tests.log").read_text(encoding="utf-8")
        if returncode or f"{EXPECTED_TESTS} passed" not in test_text:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} composition-anchor training tests")

        models = run / "artifacts/models"
        models.mkdir(parents=True)
        roots = {
            "fit_sensor": P1A / "artifacts/dataset/fit",
            "fit_teacher": P1B / "artifacts/teacher/fit",
            "fit_observability": OBSERVABILITY / "artifacts/endpoint_observability/fit",
            "fit_anchor": ANCHORS / "artifacts/materialized/anchor_targets/fit",
            "c07_sensor": P1A / "artifacts/dataset/c07",
            "c07_teacher": P1B / "artifacts/teacher/c07",
            "c07_observability": OBSERVABILITY / "artifacts/endpoint_observability/c07",
            "c07_anchor": ANCHORS / "artifacts/materialized/anchor_targets/c07",
        }
        for seed in range(3):
            command = [
                PYTHON, str(PROJECT_ROOT / "tools/v3/train_primitive_composition_anchor_v1.py"),
                "--fit-sensor-root", str(roots["fit_sensor"]),
                "--fit-teacher-root", str(roots["fit_teacher"]),
                "--fit-observability-root", str(roots["fit_observability"]),
                "--fit-anchor-root", str(roots["fit_anchor"]),
                "--c07-sensor-root", str(roots["c07_sensor"]),
                "--c07-teacher-root", str(roots["c07_teacher"]),
                "--c07-observability-root", str(roots["c07_observability"]),
                "--c07-anchor-root", str(roots["c07_anchor"]),
                "--source-checkpoint", str(SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt"),
                "--output-dir", str(models / f"seed{seed}"),
                "--seed", str(seed), "--epochs", "3", "--batch-size", "128",
                "--evaluation-batch-size", "128", "--learning-rate", "0.001",
                "--weight-decay", "0.0001",
            ]
            (run / f"config/seed{seed}_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            monitor = _run_training_monitored(
                command, log=run / f"logs/0{seed + 1}_seed{seed}_training.log",
                environment=env, timeout_seconds=21_600,
                maximum_host_rss_bytes=MAXIMUM_HOST_RSS_BYTES,
            )
            write_json(run / f"metrics/seed{seed}_resource_monitor.json", monitor)
            subprocesses.append({"stage": f"seed{seed}_training", **monitor})
            if monitor["returncode"] or monitor["killed_for_host_limit"]:
                raise RuntimeError(f"composition-anchor seed{seed} training failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            history = summary.get("history", [])
            seed_checks = {
                "schema": summary.get("schema_version") == "primitive_composition_anchor_seed_training_v1",
                "seed": summary.get("seed") == seed,
                "steps": summary.get("optimizer_steps") == EXPECTED_STEPS_PER_SEED,
                "epochs": summary.get("epochs") == 3 and len(history) == 3,
                "population": all(
                    row.get("training", {}).get("rows") == EXPECTED_FIT_ROWS
                    and row.get("selection", {}).get("rows") == EXPECTED_C07_ROWS
                    for row in history
                ),
                "parameters": summary.get("trainable_parameters") == EXPECTED_HEAD_PARAMETERS
                    and summary.get("frozen_parameters") == EXPECTED_FROZEN_PARAMETERS,
                "frozen": summary.get("frozen_state_sha256_before")
                    == summary.get("frozen_state_sha256_after"),
                "isolation": summary.get("c08_rows_read") == 0
                    and summary.get("c09_c10_worlds_read") == 0,
                "selected_exact": _sha(models / f"seed{seed}/selected.pt")
                    == _sha(models / f"seed{seed}/epoch_{int(summary['best_epoch']):02d}.pt"),
            }
            write_json(run / f"metrics/seed{seed}_contract_checks.json", seed_checks)
            if not all(seed_checks.values()):
                raise RuntimeError(f"composition-anchor seed{seed} contract drift: {seed_checks}")
            optimizer_steps += int(summary["optimizer_steps"])
            peak_cuda = max(peak_cuda, int(summary["peak_cuda_allocated_bytes"]))
            peak_gpu_process = max(peak_gpu_process, int(summary["peak_gpu_process_memory_bytes"]))
            peak_host_rss = max(peak_host_rss, int(monitor["peak_host_rss_bytes"]))
            if max(peak_cuda, peak_gpu_process) > MAXIMUM_GPU_BYTES or peak_host_rss > MAXIMUM_HOST_RSS_BYTES:
                raise RuntimeError(f"composition-anchor seed{seed} resource cap exceeded")
        if optimizer_steps != EXPECTED_TOTAL_STEPS:
            raise RuntimeError("composition-anchor total optimizer-step drift")

        evaluation = run / "metrics/evaluation"
        command = [
            PYTHON, str(PROJECT_ROOT / "tools/v3/evaluate_primitive_composition_anchor_three_seed_v1.py"),
            "--models-root", str(models),
            "--sensor-root", str(P1A / "artifacts/dataset"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--observability-root", str(OBSERVABILITY / "artifacts/endpoint_observability"),
            "--anchor-root", str(ANCHORS / "artifacts/materialized/anchor_targets"),
            "--baseline-c07-summary", str(BASELINE_C07 / "metrics/summary.json"),
            "--existence-threshold-source", str(DIAGNOSTIC / "metrics/diagnostic/summary.json"),
            "--output-dir", str(evaluation),
        ]
        (run / "config/evaluation_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        monitor = _run_training_monitored(
            command, log=run / "logs/04_c07_evaluation.log", environment=env,
            timeout_seconds=10_800, maximum_host_rss_bytes=MAXIMUM_HOST_RSS_BYTES,
        )
        write_json(run / "metrics/evaluation_resource_monitor.json", monitor)
        subprocesses.append({"stage": "c07_evaluation", **monitor})
        if monitor["killed_for_host_limit"]:
            raise RuntimeError("composition-anchor C07 evaluator exceeded host RSS cap")
        peak_host_rss = max(peak_host_rss, int(monitor["peak_host_rss_bytes"]))
        if (evaluation / "summary.json").is_file():
            evaluation_summary = load_json(evaluation / "summary.json")
        recognized = {
            "PASS_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1",
            "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_C07_V1",
        }
        if monitor["returncode"] not in (0, 2) or evaluation_summary.get("overall_status") not in recognized:
            raise RuntimeError("composition-anchor C07 evaluator produced no scientific result")
        scientific_pass = bool(evaluation_summary.get("scientific_pass"))
        overall = PASS if scientific_pass else FAIL
        for suffix in ("png", "pdf", "svg"):
            source = evaluation / f"primitive_composition_anchor_c07_comparison.{suffix}"
            if source.is_file():
                shutil.copy2(source, run / "previews" / source.name)
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks = {
            "all_frozen_inputs_unchanged": after == before,
            "exact_optimizer_steps": optimizer_steps == EXPECTED_TOTAL_STEPS,
            "three_seed_models": all((models / f"seed{seed}/selected.pt").is_file() for seed in range(3)),
            "resource_caps": peak_cuda <= MAXIMUM_GPU_BYTES
                and peak_gpu_process <= MAXIMUM_GPU_BYTES
                and peak_host_rss <= MAXIMUM_HOST_RSS_BYTES,
            "c07_scientific_result_present": evaluation_summary.get("overall_status") in recognized,
            "zero_c08_graph_mtare": evaluation_summary.get("c08_rows_read") == 0
                and evaluation_summary.get("graph_replays") == 0
                and evaluation_summary.get("mtare_worlds_read") == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"composition-anchor final evidence contract failed: {checks}")
        write_json(run / "config/source_integrity_after.json", after)
        if _directory_size(run) > MAXIMUM_RESULT_BYTES:
            raise RuntimeError("composition-anchor result exceeds 2 GiB")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_composition_anchor_three_seed_training_runner_v1",
            "overall_status": overall, "scientific_pass": scientific_pass,
            "checks": checks, "evaluation": evaluation_summary,
            "subprocesses": subprocesses,
            "optimizer_steps": optimizer_steps,
            "fit_rows_per_seed_epoch": EXPECTED_FIT_ROWS,
            "c07_rows_per_seed_epoch": EXPECTED_C07_ROWS,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_gpu_process_memory_bytes": peak_gpu_process,
            "peak_host_rss_bytes": peak_host_rss,
            "result_bytes_before_seal": _directory_size(run),
            "duration_seconds": time.monotonic() - started,
            "decision": evaluation_summary.get("decision"), "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_composition_anchor_three_seed_training_runner_v1",
            "overall_status": FAIL, "scientific_pass": False,
            "checks": checks, "evaluation": evaluation_summary,
            "subprocesses": subprocesses, "optimizer_steps": optimizer_steps,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "duration_seconds": time.monotonic() - started, "error": error,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "scientific_pass": scientific_pass,
        "error": error, "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
