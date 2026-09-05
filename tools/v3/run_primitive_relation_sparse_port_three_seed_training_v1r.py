#!/usr/bin/env python3
"""Resource-evidence corrective for the immutable sparse-port training run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
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
from run_primitive_relation_sparse_port_three_seed_training_v1 import (
    BASELINE_C07,
    BASELINE_READINESS,
    EXPECTED_STEPS_PER_SEED,
    EXPECTED_TOTAL_STEPS,
    MAXIMUM_GPU_BYTES,
    MAXIMUM_RESULT_BYTES,
    P1A,
    P1B,
    PYTHON,
    V1_ATTRIBUTION,
    V2_READINESS,
    _directory_size,
    _run_stage,
)


RUN_ID = "gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
PASS = "PASS_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_TRAINING_V1R"
FAIL = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_THREE_SEED_TRAINING_V1R"
CARD_STATUS = (
    "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_"
    "SPARSE_PORT_THREE_SEED_TRAINING_V1R"
)
FAILED_V1 = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0"
)
MAXIMUM_HOST_RSS_BYTES = 16 * 1024**3


def _parse_proc_status(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, raw = line.split(":", 1)
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"unexpected /proc memory unit: {line}")
            values[name] = int(fields[0]) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise ValueError("/proc status lacks VmRSS or VmHWM")
    return {"current_rss_bytes": values["VmRSS"], "peak_rss_bytes": values["VmHWM"]}


def _read_process_memory(pid: int) -> dict[str, int] | None:
    try:
        return _parse_proc_status(
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _run_training_monitored(
    command: list[str],
    *,
    log: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    poll_seconds: float = 1.0,
    maximum_host_rss_bytes: int = MAXIMUM_HOST_RSS_BYTES,
) -> dict[str, int | bool]:
    log.parent.mkdir(parents=True, exist_ok=True)
    peak_host = 0
    samples = 0
    killed_for_host_limit = False
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                memory = _read_process_memory(process.pid)
                if memory is not None:
                    samples += 1
                    peak_host = max(peak_host, int(memory["peak_rss_bytes"]))
                    if peak_host > maximum_host_rss_bytes:
                        killed_for_host_limit = True
                        _terminate(process)
                        break
                if time.monotonic() - started > timeout_seconds:
                    _terminate(process)
                    raise TimeoutError("sparse-port seed training timeout")
                time.sleep(poll_seconds)
        finally:
            _terminate(process)
        returncode = int(process.returncode)
    # Linux ru_maxrss is an exact high-water mark in KiB for terminated children.
    # It is cumulative/max over all children, therefore conservative after pytest.
    children_peak = int(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    )
    peak_host = max(peak_host, children_peak)
    return {
        "returncode": returncode,
        "monitor_samples": samples,
        "peak_host_rss_bytes": peak_host,
        "children_ru_maxrss_bytes": children_peak,
        "killed_for_host_limit": killed_for_host_limit,
    }


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
    peak_cuda_allocated = peak_gpu_process = peak_host_rss = 0
    scientific_pass = False
    evaluation_summary: dict = {}
    try:
        if (
            run.name != RUN_ID
            or load_json(run / "RUN_STATE.json").get("state")
            != "CREATED_NOT_EXECUTED"
        ):
            raise RuntimeError("sparse-port V1R executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        report = validate_data_card(card)
        if not report.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"sparse-port V1R Data Card invalid: {report.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"sparse-port V1R frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"sparse-port V1R tool drift: {record['path']}")
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
                raise RuntimeError(f"sparse-port V1R prerequisite failed: {source.name}")
        failed_state = load_json(FAILED_V1 / "RUN_STATE.json")
        failed_summary = load_json(FAILED_V1 / "metrics/summary.json")
        if (
            failed_state.get("state") != "FAILED"
            or failed_summary.get("failure_class")
            != "SYSTEM_RESOURCE_EVIDENCE_CONTRACT"
            or failed_summary.get("completed_epochs") != 0
            or failed_summary.get("checkpoint_files") != 0
        ):
            raise RuntimeError("sparse-port V1R corrective trigger mismatch")
        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "zarr": zarr.__version__,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"sparse-port V1R environment drift: {environment}")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "RUNNING",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
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
            "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py",
        ]
        returncode = _run_stage(
            [PYTHON, "-m", "pytest", "-q", *tests],
            log=run / "logs/00_unit_tests.log", environment=env,
            timeout_seconds=600,
        )
        subprocesses.append({"stage": "unit_tests", "returncode": returncode})
        if returncode or "48 passed" not in (run / "logs/00_unit_tests.log").read_text():
            raise RuntimeError("expected exactly 48 sparse-port V1R tests")

        models = run / "artifacts/models"
        models.mkdir(parents=True)
        fit_sensor = P1A / "artifacts/dataset/fit"
        fit_teacher = P1B / "artifacts/teacher/fit"
        c07_sensor = P1A / "artifacts/dataset/c07"
        c07_teacher = P1B / "artifacts/teacher/c07"
        for seed in range(3):
            command = [
                PYTHON, str(PROJECT_ROOT / "tools/v3/train_primitive_relation_sparse_port_v1.py"),
                "--fit-sensor-root", str(fit_sensor), "--fit-teacher-root", str(fit_teacher),
                "--c07-sensor-root", str(c07_sensor), "--c07-teacher-root", str(c07_teacher),
                "--output-dir", str(models / f"seed{seed}"), "--seed", str(seed),
                "--epochs", "7", "--batch-size", "16", "--evaluation-batch-size", "128",
                "--learning-rate", "0.0003", "--weight-decay", "0.0001",
            ]
            (run / f"config/seed{seed}_command.txt").write_text(" ".join(command) + "\n")
            monitor = _run_training_monitored(
                command, log=run / f"logs/0{seed + 1}_seed{seed}_training.log",
                environment=env, timeout_seconds=43_200,
            )
            write_json(run / f"metrics/seed{seed}_resource_monitor.json", monitor)
            subprocesses.append({"stage": f"seed{seed}_training", **monitor})
            if monitor["returncode"] or monitor["killed_for_host_limit"]:
                raise RuntimeError(f"sparse-port V1R seed{seed} failed")
            summary = load_json(models / f"seed{seed}/summary.json")
            if (
                summary.get("c08_rows_read") != 0
                or summary.get("c09_c10_worlds_read") != 0
                or int(summary.get("optimizer_steps", -1)) != EXPECTED_STEPS_PER_SEED
            ):
                raise RuntimeError(f"sparse-port V1R seed{seed} contract drift")
            optimizer_steps += int(summary["optimizer_steps"])
            peak_cuda_allocated = max(peak_cuda_allocated, int(summary["peak_cuda_allocated_bytes"]))
            # V1 trainer's historically named process metric is explicitly GPU memory.
            peak_gpu_process = max(peak_gpu_process, int(summary["peak_process_memory_bytes"]))
            peak_host_rss = max(peak_host_rss, int(monitor["peak_host_rss_bytes"]))
            if (
                peak_cuda_allocated > MAXIMUM_GPU_BYTES
                or peak_gpu_process > MAXIMUM_GPU_BYTES
                or peak_host_rss > MAXIMUM_HOST_RSS_BYTES
            ):
                raise RuntimeError(f"sparse-port V1R seed{seed} resource cap exceeded")
        if optimizer_steps != EXPECTED_TOTAL_STEPS:
            raise RuntimeError("sparse-port V1R total optimizer-step drift")

        evaluation = run / "metrics/evaluation"
        command = [
            PYTHON, str(PROJECT_ROOT / "tools/v3/evaluate_primitive_relation_sparse_port_three_seed_v1.py"),
            "--models-root", str(models), "--sensor-root", str(P1A / "artifacts/dataset"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--baseline-c07-summary", str(BASELINE_C07 / "metrics/summary.json"),
            "--output-dir", str(evaluation),
        ]
        (run / "config/evaluation_command.txt").write_text(" ".join(command) + "\n")
        returncode = _run_stage(
            command, log=run / "logs/04_evaluation.log", environment=env,
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
        if returncode not in (0, 2) or evaluation_summary.get("overall_status") not in recognized:
            raise RuntimeError("sparse-port V1R evaluation produced no scientific result")
        scientific_pass = bool(evaluation_summary.get("scientific_pass"))
        overall = PASS if scientific_pass else FAIL
        for suffix in ("png", "pdf", "svg"):
            source = evaluation / f"primitive_relation_sparse_port_comparison.{suffix}"
            if source.is_file():
                shutil.copy2(source, run / "previews" / source.name)
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("sparse-port V1R frozen sources changed")
        write_json(run / "config/source_integrity_after.json", after)
        if _directory_size(run) > MAXIMUM_RESULT_BYTES:
            raise RuntimeError("sparse-port V1R result exceeds 6 GiB")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_sparse_port_three_seed_training_runner_v1r",
            "overall_status": overall, "scientific_pass": scientific_pass,
            "evaluation": evaluation_summary, "subprocesses": subprocesses,
            "optimizer_steps": optimizer_steps,
            "fit_rows_per_seed_epoch": 426_552, "c07_rows_per_seed_epoch": 64_644,
            "c08_rows_read": int(evaluation_summary.get("c08_rows_read", 0)),
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_gpu_process_memory_bytes": peak_gpu_process,
            "peak_host_rss_bytes": peak_host_rss,
            "resource_corrective": "Independent host VmHWM plus RUSAGE_CHILDREN; old process metric explicitly GPU.",
            "result_bytes_before_seal": _directory_size(run),
            "duration_seconds": time.monotonic() - started, "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc())
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_sparse_port_three_seed_training_runner_v1r",
            "overall_status": FAIL, "scientific_pass": False, "error": error,
            "subprocesses": subprocesses, "optimizer_steps": optimizer_steps,
            "c09_c10_worlds_read": 0, "graph_replays": 0,
            "duration_seconds": time.monotonic() - started,
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
