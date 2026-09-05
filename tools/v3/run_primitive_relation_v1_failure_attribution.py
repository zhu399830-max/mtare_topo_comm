#!/usr/bin/env python3
"""Run and seal the one immutable C07-only V1 failure attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
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


RUN_ID = "gate3_20260831_primitive_relation_v1_failure_attribution_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_three_seed_training_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


def _sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; before = {}; after = {}; subprocesses = []
    peak_rss_kib = None; attribution = {}; corrective_inference_rows = 0
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("primitive relation attribution executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"primitive relation attribution Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"primitive relation attribution input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"primitive relation attribution tool drift: {record['path']}")

        training_state = load_json(TRAINING / "RUN_STATE.json")
        training_summary = load_json(TRAINING / "metrics/summary.json")
        if training_state.get("state") != "COMPLETED" or training_state.get("error") is not None:
            raise RuntimeError("failed V1 source run is not a clean completed scientific result")
        if training_summary.get("scientific_pass") is not False or training_summary.get("c08_rows_read") != 0:
            raise RuntimeError("failed V1 source decision or C08 isolation drift")
        if training_summary.get("evaluation", {}).get("decision") != "STOP_BEFORE_C08_AND_GRAPH":
            raise RuntimeError("failed V1 stop decision drift")

        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "zarr": zarr.__version__,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3",
            "scipy": "1.15.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"primitive relation attribution environment drift: {environment}")
        write_json(run / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["PYTHONHASHSEED"] = "0"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_relation_failure_attribution.py",
            "tests/v3/unit/test_primitive_relation_metrics.py",
            "tests/v3/unit/test_evaluate_primitive_relation_three_seed_v1.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
            )
        subprocesses.append({"stage": "unit_tests", "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError("primitive relation attribution unit tests failed")

        formal_evaluation_root = TRAINING / "metrics/evaluation"
        if spec.get("corrected_c07_tf32_evaluation") is True:
            corrected = run / "metrics/c07_tf32_corrective"
            corrective_command = [
                PYTHON, str(PROJECT_ROOT / "tools/v3/evaluate_primitive_relation_three_seed_c07_tf32_corrective_v1.py"),
                "--models-root", str(TRAINING / "artifacts/models"),
                "--sensor-root", str(P1A / "artifacts/dataset/c07"),
                "--teacher-root", str(P1B / "artifacts/teacher/c07"),
                "--baseline-c07-summary", str(BASELINE / "metrics/summary.json"),
                "--output-dir", str(corrected),
            ]
            (run / "config/c07_tf32_corrective_command.txt").write_text(" ".join(corrective_command) + "\n", encoding="utf-8")
            with (run / "logs/01_c07_tf32_corrective.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    corrective_command, cwd=PROJECT_ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=7_200, check=False,
                )
            subprocesses.append({"stage": "c07_tf32_corrective", "returncode": result.returncode})
            if result.returncode:
                raise RuntimeError("primitive relation C07 TF32 corrective failed")
            corrected_summary = load_json(corrected / "summary.json")
            if corrected_summary.get("overall_status") != "PASS_PRIMITIVE_RELATION_C07_TF32_CORRECTIVE_EVALUATION_COMPLETE":
                raise RuntimeError("primitive relation C07 TF32 corrective evidence invalid")
            if corrected_summary.get("c08_rows_read") != 0 or corrected_summary.get("optimizer_steps") != 0:
                raise RuntimeError("primitive relation C07 TF32 corrective isolation drift")
            corrective_inference_rows = int(corrected_summary.get("model_inference_rows", 0))
            if corrective_inference_rows != 387_864:
                raise RuntimeError("primitive relation C07 TF32 corrective inference population drift")
            formal_evaluation_root = corrected

        output = run / "metrics/attribution"
        command = [
            PYTHON, str(PROJECT_ROOT / "tools/v3/execute_primitive_relation_v1_failure_attribution.py"),
            "--models-root", str(TRAINING / "artifacts/models"),
            "--sensor-root", str(P1A / "artifacts/dataset/c07"),
            "--teacher-root", str(P1B / "artifacts/teacher/c07"),
            "--formal-evaluation-root", str(formal_evaluation_root),
            "--baseline-summary", str(BASELINE / "metrics/summary.json"),
            "--output-dir", str(output),
        ]
        (run / "config/attribution_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        attribution_log = "logs/02_attribution.log" if spec.get("corrected_c07_tf32_evaluation") is True else "logs/01_attribution.log"
        with (run / attribution_log).open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=10_800, check=False,
            )
        subprocesses.append({"stage": "attribution", "returncode": result.returncode})
        log_text = (run / attribution_log).read_text(encoding="utf-8")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log_text)
        peak_rss_kib = int(match.group(1)) if match else None
        if result.returncode:
            raise RuntimeError("primitive relation attribution subprocess failed")
        attribution = load_json(output / "summary.json")
        if attribution.get("overall_status") != PASS or attribution.get("scientific_pass") is not True:
            raise RuntimeError("primitive relation attribution did not resolve the failure")
        if attribution.get("rows_per_seed") != 64_644 or attribution.get("model_inference_rows") != 193_932:
            raise RuntimeError("primitive relation attribution population drift")
        if attribution.get("optimizer_steps") != 0 or attribution.get("c08_rows_read") != 0:
            raise RuntimeError("primitive relation attribution isolation drift")
        if peak_rss_kib is None or peak_rss_kib > 16 * 1024 * 1024:
            raise RuntimeError("primitive relation attribution host RAM evidence missing or exceeded")
        if int(attribution["peak_cuda_reserved_bytes"]) > 16 * 1024**3:
            raise RuntimeError("primitive relation attribution GPU memory exceeded")
        required = [
            output / "summary.json", output / "per_task.csv", output / "figure_source.json",
            *[output / f"primitive_relation_v1_failure_attribution.{suffix}" for suffix in ("png", "pdf", "svg")],
            *[output / f"seed{seed}.json" for seed in range(3)],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("primitive relation attribution evidence incomplete")
        for suffix in ("png", "pdf", "svg"):
            shutil.copy2(output / f"primitive_relation_v1_failure_attribution.{suffix}", run / "previews")

        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("primitive relation attribution changed frozen inputs")
        write_json(run / "config/source_integrity_after.json", after)
        if time.monotonic() - started > 10_800:
            raise RuntimeError("primitive relation attribution wall-time limit exceeded")
        if sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > 200 * 1024**2:
            raise RuntimeError("primitive relation attribution output limit exceeded")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    duration = time.monotonic() - started
    write_json(run / "metrics/summary.json", {
        "schema_version": "primitive_relation_v1_failure_attribution_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "model_v1_scientific_pass": False, "error": error,
        "subprocesses": subprocesses, "attribution": attribution,
        "peak_host_rss_kib": peak_rss_kib, "duration_seconds": duration,
        "optimizer_steps": 0,
        "model_inference_rows": corrective_inference_rows + int(attribution.get("model_inference_rows", 0)),
        "corrective_evaluation_inference_rows": corrective_inference_rows,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "source_unchanged": bool(before and before == after),
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": duration,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "error": error,
        "diagnosis": attribution.get("diagnosis"), "decision": attribution.get("decision"),
        "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
