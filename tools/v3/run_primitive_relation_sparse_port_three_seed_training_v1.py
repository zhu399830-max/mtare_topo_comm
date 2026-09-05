#!/usr/bin/env python3
"""Execute, evaluate and seal the immutable sparse-port three-seed run."""

from __future__ import annotations

import argparse
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
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_primitive_relation_nonlearning_readiness_v1 import _seal, _sha


RUN_ID = "gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_TRAINING_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_TRAINING_V1"
CARD_STATUS = (
    "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_"
    "SPARSE_PORT_THREE_SEED_TRAINING_V1"
)
PYTHON = (
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/"
    "phase3_torch290_cu129_zarr2187_v1/bin/python"
)
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
V2_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0"
V1_ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0"
BASELINE_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
BASELINE_C07 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"
MAXIMUM_GPU_BYTES = 16 * 1024**3
MAXIMUM_HOST_BYTES = 16 * 1024**3
MAXIMUM_RESULT_BYTES = 6 * 1024**3
EXPECTED_STEPS_PER_SEED = 26_736 * 7
EXPECTED_TOTAL_STEPS = EXPECTED_STEPS_PER_SEED * 3


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _run_stage(
    command: list[str],
    *,
    log: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    before: dict[str, str] = {}
    subprocesses: list[dict] = []
    optimizer_steps = 0
    peak_cuda = peak_host = 0
    scientific_pass = False
    evaluation_summary: dict = {}
    try:
        if (
            run.name != RUN_ID
            or load_json(run / "RUN_STATE.json").get("state")
            != "CREATED_NOT_EXECUTED"
        ):
            raise RuntimeError("sparse-port training executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"sparse-port training Data Card invalid: {report.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"sparse-port frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(
                    f"sparse-port frozen tool drift: {record['path']}"
                )
        prerequisites = (
            P1A, P1B, V2_READINESS, V1_ATTRIBUTION,
            BASELINE_READINESS, BASELINE_C07,
        )
        for source in prerequisites:
            state = load_json(source / "RUN_STATE.json")
            summary = load_json(source / "metrics/summary.json")
            if (
                state.get("state") != "COMPLETED"
                or state.get("error") is not None
                or not summary.get("scientific_pass")
            ):
                raise RuntimeError(
                    f"sparse-port training prerequisite failed: {source.name}"
                )
        environment = {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "zarr": zarr.__version__,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"sparse-port training environment drift: {environment}")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/environment.json", {
            "versions": environment,
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT / "src") + os.pathsep
            + str(PROJECT_ROOT / "tools/v3")
        )
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_relation_training.py",
            "tests/v3/unit/test_primitive_relation_model.py",
            "tests/v3/unit/test_primitive_relation_nonlearning.py",
            "tests/v3/unit/test_primitive_relation_metrics.py",
            "tests/v3/unit/test_primitive_relation_training_schedule.py",
            "tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
            "tests/v3/unit/test_primitive_relation_sparse_port.py",
            "tests/v3/unit/test_primitive_relation_sparse_port_training.py",
            "tests/v3/unit/test_evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        ]
        returncode = _run_stage(
            [PYTHON, "-m", "pytest", "-q", *tests],
            log=run / "logs/00_unit_tests.log",
            environment=env,
            timeout_seconds=600,
        )
        subprocesses.append({"stage": "unit_tests", "returncode": returncode})
        if returncode:
            raise RuntimeError("sparse-port training unit tests failed")
        unit_log = (run / "logs/00_unit_tests.log").read_text(encoding="utf-8")
        if "44 passed" not in unit_log:
            raise RuntimeError("expected exactly 44 sparse-port training tests")

        models = run / "artifacts/models"
        models.mkdir(parents=True)
        fit_sensor = P1A / "artifacts/dataset/fit"
        fit_teacher = P1B / "artifacts/teacher/fit"
        c07_sensor = P1A / "artifacts/dataset/c07"
        c07_teacher = P1B / "artifacts/teacher/c07"
        for seed in range(3):
            command = [
                PYTHON,
                str(PROJECT_ROOT / "tools/v3/train_primitive_relation_sparse_port_v1.py"),
                "--fit-sensor-root", str(fit_sensor),
                "--fit-teacher-root", str(fit_teacher),
                "--c07-sensor-root", str(c07_sensor),
                "--c07-teacher-root", str(c07_teacher),
                "--output-dir", str(models / f"seed{seed}"),
                "--seed", str(seed),
                "--epochs", "7", "--batch-size", "16",
                "--evaluation-batch-size", "128",
                "--learning-rate", "0.0003", "--weight-decay", "0.0001",
            ]
            (run / f"config/seed{seed}_command.txt").write_text(
                " ".join(command) + "\n", encoding="utf-8",
            )
            returncode = _run_stage(
                command,
                log=run / f"logs/0{seed + 1}_seed{seed}_training.log",
                environment=env,
                timeout_seconds=43_200,
            )
            subprocesses.append({
                "stage": f"seed{seed}_training", "returncode": returncode,
            })
            if returncode:
                raise RuntimeError(f"sparse-port seed{seed} training failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            if (
                summary.get("c08_rows_read") != 0
                or summary.get("c09_c10_worlds_read") != 0
                or int(summary.get("optimizer_steps", -1)) != EXPECTED_STEPS_PER_SEED
            ):
                raise RuntimeError(f"sparse-port seed{seed} contract drift")
            optimizer_steps += int(summary["optimizer_steps"])
            peak_cuda = max(peak_cuda, int(summary["peak_cuda_allocated_bytes"]))
            peak_host = max(peak_host, int(summary["peak_process_memory_bytes"]))
            if peak_cuda > MAXIMUM_GPU_BYTES or peak_host > MAXIMUM_HOST_BYTES:
                raise RuntimeError(f"sparse-port seed{seed} resource cap exceeded")
        if optimizer_steps != EXPECTED_TOTAL_STEPS:
            raise RuntimeError("sparse-port total optimizer-step drift")

        evaluation = run / "metrics/evaluation"
        command = [
            PYTHON,
            str(PROJECT_ROOT / "tools/v3/evaluate_primitive_relation_sparse_port_three_seed_v1.py"),
            "--models-root", str(models),
            "--sensor-root", str(P1A / "artifacts/dataset"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--baseline-c07-summary", str(BASELINE_C07 / "metrics/summary.json"),
            "--output-dir", str(evaluation),
        ]
        (run / "config/evaluation_command.txt").write_text(
            " ".join(command) + "\n", encoding="utf-8",
        )
        returncode = _run_stage(
            command,
            log=run / "logs/04_evaluation.log",
            environment=env,
            timeout_seconds=18_000,
        )
        subprocesses.append({"stage": "evaluation", "returncode": returncode})
        if (evaluation / "summary.json").is_file():
            evaluation_summary = load_json(evaluation / "summary.json")
        recognized = {
            "PASS_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_V1",
            "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_C07",
            "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_C08",
        }
        if (
            returncode not in (0, 2)
            or evaluation_summary.get("overall_status") not in recognized
        ):
            raise RuntimeError("sparse-port evaluation produced no scientific result")
        scientific_pass = bool(evaluation_summary.get("scientific_pass"))
        overall = PASS if scientific_pass else FAIL
        for suffix in ("png", "pdf", "svg"):
            source = evaluation / f"primitive_relation_sparse_port_comparison.{suffix}"
            if source.is_file():
                shutil.copy2(source, run / "previews" / source.name)
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("sparse-port frozen sources changed")
        write_json(run / "config/source_integrity_after.json", after)
        if _directory_size(run) > MAXIMUM_RESULT_BYTES:
            raise RuntimeError("sparse-port result exceeds 6 GiB")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_sparse_port_three_seed_training_runner_v1",
            "overall_status": overall,
            "scientific_pass": scientific_pass,
            "evaluation": evaluation_summary,
            "subprocesses": subprocesses,
            "optimizer_steps": optimizer_steps,
            "fit_rows_per_seed_epoch": 426_552,
            "c07_rows_per_seed_epoch": 64_644,
            "c08_rows_read": int(evaluation_summary.get("c08_rows_read", 0)),
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_process_memory_bytes": peak_host,
            "result_bytes_before_seal": _directory_size(run),
            "duration_seconds": time.monotonic() - started,
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8",
        )
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_sparse_port_three_seed_training_runner_v1",
            "overall_status": FAIL,
            "scientific_pass": False,
            "error": error,
            "subprocesses": subprocesses,
            "optimizer_steps": optimizer_steps,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "duration_seconds": time.monotonic() - started,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1",
        "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall,
        "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall,
        "scientific_pass": scientific_pass,
        "error": error,
        "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
