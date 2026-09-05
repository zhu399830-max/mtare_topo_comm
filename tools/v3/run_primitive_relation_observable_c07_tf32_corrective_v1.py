#!/usr/bin/env python3
"""One immutable same-checkpoint observable C07 TF32-off corrective run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
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
    MAXIMUM_GPU_BYTES, _directory_size, _run_stage,
)
from run_primitive_relation_sparse_port_three_seed_training_v1r import (
    MAXIMUM_HOST_RSS_BYTES, _run_training_monitored,
)


RUN_ID = "gate3_20260902_primitive_relation_observable_c07_tf32_corrective_v1_seed0"
PASS = "PASS_SYSTEM_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1"
FAIL = "FAIL_SYSTEM_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
EXPECTED_ROWS = 64_644
EXPECTED_FORWARD_ROWS = 387_864
MAXIMUM_RESULT_BYTES = 256 * 1024**2


def _metric_snapshot(summary: dict) -> list[dict]:
    records = []
    for item in summary["c07"]["seeds"]:
        metrics = item["metrics"]
        comparison = item["comparison"]
        records.append({
            "seed": int(item["seed"]),
            "selected_epoch": int(item["selected_epoch"]),
            "pass": bool(comparison["pass"]),
            "surface_improvement": float(comparison["surface_improvement"]),
            "geometry_macro_improvement": float(comparison["geometry_macro_improvement"]),
            "attachment_f1_gain": float(comparison["attachment_f1_gain"]),
            "attachment_precision": float(metrics["attachment"]["precision"]),
            "attachment_recall": float(metrics["attachment"]["recall"]),
            "attachment_f1": float(metrics["attachment"]["f1"]),
            "safe_true_positive": int(metrics["attachment_safe_selection"]["true_positive"]),
            "safe_false_positive": int(metrics["attachment_safe_selection"]["false_positive"]),
            "safe_precision": float(metrics["attachment_safe_selection"]["precision"]),
            "safe_recall": float(metrics["attachment_safe_selection"]["recall"]),
        })
    return records


def _comparison(old: dict, corrected: dict) -> dict:
    old_records = _metric_snapshot(old)
    new_records = _metric_snapshot(corrected)
    if [item["seed"] for item in old_records] != [0, 1, 2] or [item["seed"] for item in new_records] != [0, 1, 2]:
        raise RuntimeError("observable corrective seed inventory drift")
    deltas = []
    for before, after in zip(old_records, new_records, strict=True):
        deltas.append({
            "seed": before["seed"],
            "selected_epoch_unchanged": before["selected_epoch"] == after["selected_epoch"],
            "surface_improvement_delta": after["surface_improvement"] - before["surface_improvement"],
            "geometry_macro_improvement_delta": after["geometry_macro_improvement"] - before["geometry_macro_improvement"],
            "attachment_f1_gain_delta": after["attachment_f1_gain"] - before["attachment_f1_gain"],
            "attachment_precision_delta": after["attachment_precision"] - before["attachment_precision"],
            "attachment_recall_delta": after["attachment_recall"] - before["attachment_recall"],
            "safe_true_positive_delta": after["safe_true_positive"] - before["safe_true_positive"],
            "pass_changed": before["pass"] != after["pass"],
        })
    return {
        "schema_version": "primitive_relation_observable_c07_tf32_old_new_comparison_v1",
        "old_contract": "independent_final_evaluator_with_default_cudnn_tf32_true",
        "corrected_contract": "deterministic_matmul_and_cudnn_tf32_false_highest",
        "old": old_records,
        "corrected": new_records,
        "deltas": deltas,
        "old_passing_seeds": int(old["c07"]["passing_seeds"]),
        "corrected_passing_seeds": int(corrected["c07"]["passing_seeds"]),
        "scientific_decision_changed": bool(old["scientific_pass"] != corrected["scientific_pass"]),
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
    scientific_pass = False
    before: dict[str, str] = {}
    monitor: dict[str, object] = {}
    corrected: dict = {}
    old: dict = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("observable C07 corrective executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"observable corrective Data Card invalid: {validation.errors}")
        for relative, digest in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            if not path.is_file() or _sha(path) != digest:
                raise RuntimeError(f"observable corrective frozen input drift: {relative}")
        before = {
            record["path"]: _sha(PROJECT_ROOT / record["path"])
            for record in spec["frozen_tools"].values()
        }
        if any(before[record["path"]] != record["sha256"] for record in spec["frozen_tools"].values()):
            raise RuntimeError("observable corrective frozen tool drift")
        source_state = load_json(SOURCE / "RUN_STATE.json")
        source_summary = load_json(SOURCE / "metrics/summary.json")
        if (
            source_state.get("state") != "COMPLETED"
            or source_summary.get("error") is not None
            or int(source_summary.get("optimizer_steps", -1)) != 240_624
            or int(source_summary.get("c08_rows_read", -1)) != 0
        ):
            raise RuntimeError("observable corrective source run drift")
        old = load_json(SOURCE / "metrics/evaluation/summary.json")
        if old.get("schema_version") != "primitive_relation_observable_three_seed_evaluation_v1":
            raise RuntimeError("observable corrective old evaluation drift")

        for directory in ("config", "logs", "metrics", "artifacts", "previews"):
            (run / directory).mkdir(exist_ok=True)
        shutil.copy2(args.spec, run / "config/run_spec.json")
        shutil.copy2(PROJECT_ROOT / spec["data_card"], run / "config/data_card.json")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/environment.json", {
            "schema_version": "primitive_relation_observable_c07_tf32_corrective_environment_v1",
            "platform": platform.platform(), "python": platform.python_version(),
            "executable": PYTHON, "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__,
            "scipy": scipy.__version__, "zarr": zarr.__version__,
            "declared_numerical_contract": {
                "deterministic_algorithms": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
            "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
        ]
        test_command = [PYTHON, "-m", "pytest", "-q", *tests]
        write_json(run / "config/test_command.json", test_command)
        test_return = _run_stage(
            test_command, log=run / "logs/00_unit_tests.log",
            environment=env, timeout_seconds=600,
        )
        if test_return or "3 passed" not in (run / "logs/00_unit_tests.log").read_text():
            raise RuntimeError("observable corrective tests failed")

        output = run / "metrics/evaluation"
        command = [
            PYTHON,
            str(PROJECT_ROOT / "tools/v3/evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py"),
            "--models-root", str(SOURCE / "artifacts/models"),
            "--sensor-root", str(P1A / "artifacts/dataset"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--sidecar-root", str(SIDECAR / "artifacts/endpoint_observability"),
            "--baseline-c07-summary", str(BASELINE / "metrics/summary.json"),
            "--output-dir", str(output),
        ]
        write_json(run / "config/evaluation_command.json", command)
        monitor = _run_training_monitored(
            command, log=run / "logs/01_c07_tf32_corrective.log",
            environment=env, timeout_seconds=10_800,
        )
        write_json(run / "metrics/resource_monitor.json", monitor)
        if int(monitor.get("returncode", -1)) not in (0, 2) or bool(monitor.get("killed_for_host_limit")):
            raise RuntimeError("observable corrective evaluator failed")
        corrected = load_json(output / "summary.json")
        expected_contract = {
            "deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        if (
            corrected.get("schema_version")
            != "primitive_relation_observable_three_seed_c07_tf32_corrective_v1"
            or corrected.get("numerical_contract") != expected_contract
            or int(corrected.get("c08_rows_read", -1)) != 0
            or int(corrected.get("c09_c10_worlds_read", -1)) != 0
            or int(corrected.get("graph_replays", -1)) != 0
            or len(corrected.get("c07", {}).get("seeds", [])) != 3
            or any(int(item["metrics"]["rows"]) != EXPECTED_ROWS for item in corrected["c07"]["seeds"])
        ):
            raise RuntimeError("observable corrective result contract drift")
        peak_cuda = int(corrected.get("peak_cuda_allocated_bytes", -1))
        peak_reserved = int(corrected.get("peak_cuda_reserved_bytes", -1))
        peak_gpu = int(monitor.get("peak_gpu_process_memory_bytes", -1))
        peak_host = int(monitor.get("peak_host_rss_bytes", -1))
        if max(peak_cuda, peak_reserved, peak_gpu) > MAXIMUM_GPU_BYTES or peak_host > MAXIMUM_HOST_RSS_BYTES:
            raise RuntimeError("observable corrective resource cap exceeded")
        comparison = _comparison(old, corrected)
        write_json(run / "metrics/old_vs_corrected.json", comparison)
        scientific_pass = bool(corrected["scientific_pass"])
        for suffix in ("png", "pdf", "svg"):
            source = output / f"primitive_relation_observable_comparison.{suffix}"
            if not source.is_file():
                raise RuntimeError("observable corrective paper figure missing")
            shutil.copy2(source, run / "previews" / f"primitive_relation_observable_tf32_corrected.{suffix}")
        write_json(run / "previews/README.json", {
            "figure": "primitive_relation_observable_tf32_corrected",
            "source": "metrics/evaluation/summary.json",
            "old_vs_corrected": "metrics/old_vs_corrected.json",
        })
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        write_json(run / "config/source_integrity_after.json", after)
        if after != before:
            raise RuntimeError("observable corrective source changed during execution")
        overall = PASS
        summary = {
            "schema_version": "primitive_relation_observable_c07_tf32_corrective_runner_v1",
            "overall_status": overall, "scientific_pass": scientific_pass,
            "error": None, "optimizer_steps": 0,
            "model_forward_rows": EXPECTED_FORWARD_ROWS,
            "c07_rows_per_seed": EXPECTED_ROWS, "frozen_seeds": 3,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "old_passing_seeds": int(old["c07"]["passing_seeds"]),
            "corrected_passing_seeds": int(corrected["c07"]["passing_seeds"]),
            "scientific_decision": corrected["decision"],
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_cuda_reserved_bytes": peak_reserved,
            "peak_gpu_process_memory_bytes": peak_gpu,
            "peak_host_rss_bytes": peak_host,
            "result_bytes_before_seal": _directory_size(run),
            "duration_seconds": time.monotonic() - started,
        }
        write_json(run / "metrics/summary.json", summary)
        if _directory_size(run) > MAXIMUM_RESULT_BYTES:
            raise RuntimeError("observable corrective result cap exceeded")
    except Exception as exc:  # pragma: no cover - formal failure evidence
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_observable_c07_tf32_corrective_runner_v1",
            "overall_status": FAIL, "scientific_pass": False,
            "error": error, "optimizer_steps": 0,
            "model_forward_rows": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0,
            "mtare_worlds_read": 0, "duration_seconds": time.monotonic() - started,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    write_json(run / "artifacts/seal_summary.json", {
        "entries": len(entries), "overall_status": overall,
        "scientific_pass": scientific_pass, "error": error,
    })
    print(json.dumps({
        "overall_status": overall, "scientific_pass": scientific_pass,
        "error": error, "evidence_entries": len(entries),
    }, indent=2, sort_keys=True))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
