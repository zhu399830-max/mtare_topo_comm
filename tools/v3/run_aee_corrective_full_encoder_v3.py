#!/home/zeng-workstation/anaconda3/bin/python
"""Execute, aggregate and seal the approved three-seed corrective training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.governance import load_json, write_json
import run_aee_head_adaptation_v1 as base


RUN_ID = "gate2_20260822_aee_corrective_full_encoder_v3_seed20260822"
TRAINER = PROJECT_ROOT / "tools/v3/train_aee_corrective_full_encoder_v3.py"
STATUS_PASS = "PASS_AEE_CORRECTIVE_FULL_ENCODER_V3"
STATUS_FAIL = "FAIL_AEE_CORRECTIVE_FULL_ENCODER_V3"
TIME_LIMIT_SECONDS = 4 * 3600
DISK_LIMIT_BYTES = 8 * 1024**3


def child_argv(spec: dict[str, Any], run_dir: Path, checkpoint: dict[str, Any], python: Path) -> list[str]:
    seed = int(checkpoint["seed"])
    return [
        str(python), str(TRAINER),
        "--dataset-run", str((PROJECT_ROOT / spec["source_dataset_run"]).resolve()),
        "--source-checkpoint", str((PROJECT_ROOT / checkpoint["path"]).resolve()),
        "--source-checkpoint-sha256", str(checkpoint["sha256"]),
        "--output-dir", str(run_dir / f"artifacts/models/m1d_seed{seed}"),
        "--seed", str(seed), "--epochs", "10", "--batch-size", "128",
        "--learning-rate", "0.0001", "--weight-decay", "0.0001", "--workers", "0",
    ]


def validate_seed_summary(summary: dict[str, Any], seed: int) -> None:
    if summary.get("status") != "COMPLETED_AEE_CORRECTIVE_FULL_ENCODER_SEED_V3":
        raise RuntimeError(f"seed {seed} did not complete")
    if summary.get("seed") != seed or summary.get("epochs_completed") != 10:
        raise RuntimeError(f"seed/epoch identity drift: {seed}")
    if summary.get("independent_train_samples_per_epoch") != {"cano": 5000, "aee": 1000}:
        raise RuntimeError(f"independent sample count drift: {seed}")
    if summary.get("training_views_per_epoch") != {"cano_dense": 5000, "cano_mask_matched": 5000, "aee": 1000}:
        raise RuntimeError(f"training view count drift: {seed}")
    if (summary.get("cano_dense_validation_frames"), summary.get("cano_sparse_validation_frames"), summary.get("aee_train_diagnostic_frames")) != (5000, 5000, 1000):
        raise RuntimeError(f"evaluation frame count drift: {seed}")
    if any(summary.get(key) != 0 for key in ("c09_frames_read", "c10_frames_read", "formal_benchmark_frames_read")):
        raise RuntimeError(f"forbidden data read: {seed}")
    expected = {
        "sparse_direction_improvement_0p05", "sparse_direction_vs_b0", "sparse_empty_rate",
        "sparse_count_1_to_4_macro_f1", "sparse_role_macro_f1", "dense_direction_retention",
        "all_model_tensors_finite", "encoder_changed", "embedding_changed", "semantic_heads_changed",
    }
    if not isinstance(summary.get("gate"), dict) or set(summary["gate"]) != expected:
        raise RuntimeError(f"scientific gate schema drift: {seed}")


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [item.get("seed") for item in summaries] != [0, 1, 2]:
        raise RuntimeError("seeds 0/1/2 required exactly once in order")
    for item in summaries:
        validate_seed_summary(item, int(item["seed"]))
    sparse_f1 = [float(item["adapted_cano_sparse_validation"]["direction"]["f1"]) for item in summaries]
    sparse_delta = [
        float(item["adapted_cano_sparse_validation"]["direction"]["f1"] - item["source_cano_sparse_validation"]["direction"]["f1"])
        for item in summaries
    ]
    dense_delta = [
        float(item["adapted_cano_dense_validation"]["direction"]["f1"] - item["source_cano_dense_validation"]["direction"]["f1"])
        for item in summaries
    ]
    return {
        "scientific_gate_passed_by_seed": [bool(item["scientific_gate_passed"]) for item in summaries],
        "all_seed_gates_passed": all(bool(item["scientific_gate_passed"]) for item in summaries),
        "sparse_direction_f1_by_seed": sparse_f1,
        "sparse_direction_f1_median": float(np.median(sparse_f1)),
        "sparse_direction_improvement_by_seed": sparse_delta,
        "dense_direction_change_by_seed": dense_delta,
        "sparse_count_1_to_4_macro_f1_by_seed": [item["adapted_cano_sparse_validation"]["count"]["macro_f1_count_1_to_4"] for item in summaries],
        "sparse_role_macro_f1_by_seed": [item["adapted_cano_sparse_validation"]["role"]["macro_f1_present"] for item in summaries],
    }


def _seal_failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {"schema_version": "aee_corrective_full_encoder_summary_v3", "overall_status": STATUS_FAIL, "execution_failure": True, "failure_reason": message, "completed_seeds": completed, "duration_seconds": time.monotonic() - started, "c09_frames_read": 0, "c10_frames_read": 0, "formal_benchmark_frames_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL})
    base.seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("corrective training run identity/state mismatch")
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        authorization = spec.get("user_authorization", {})
        if spec.get("gate") != 2 or spec.get("operation") != "training" or authorization.get("status") != "APPROVED":
            raise RuntimeError("approved Gate-2 corrective training scope required")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_AEE_CORRECTIVE_FULL_ENCODER_V3":
            raise RuntimeError("corrective training Data Card status drift")
        approval = card.get("approval", {})
        if approval.get("authorized_gates") != [2] or not {"training", "checkpoint_selection"}.issubset(approval.get("authorized_operations", [])):
            raise RuntimeError("corrective training authorization drift")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen tool drift: {name}")
        source = (PROJECT_ROOT / spec["source_dataset_run"]).resolve()
        source.relative_to(PROJECT_ROOT)
        seal_entries = base.verify_seal(source, spec["source_dataset_seal_sha256"])
        source_summary = load_json(source / "metrics/summary.json")
        if source_summary.get("overall_status") != "PASS_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R4" or (source_summary.get("cano_frames"), source_summary.get("aee_frames"), source_summary.get("teacher_labels")) != (10000, 1000, 11000):
            raise RuntimeError("V1R4 source dataset status/count drift")
        checkpoints = spec.get("source_checkpoints")
        if not isinstance(checkpoints, list) or [item.get("seed") for item in checkpoints] != [0, 1, 2]:
            raise RuntimeError("source checkpoints must be seeds 0/1/2")
        for checkpoint in checkpoints:
            if sha256(PROJECT_ROOT / checkpoint["path"]) != checkpoint["sha256"]:
                raise RuntimeError(f"source checkpoint identity drift: {checkpoint['seed']}")
        python = base.preserve_venv_executable(spec["python_executable"])
        environment = base.probe_environment(python)
        freeze = spec["sidecar_freeze"]
        if sha256(PROJECT_ROOT / freeze["path"]) != freeze["sha256"] or sha256(PROJECT_ROOT / freeze["pip_freeze_path"]) != freeze["pip_freeze_sha256"]:
            raise RuntimeError("sidecar freeze identity drift")
        live = subprocess.run([str(python), "-m", "pip", "freeze"], text=True, capture_output=True, check=False)
        if live.returncode != 0 or base.sorted_pip_freeze_sha256(live.stdout) != freeze["pip_freeze_sha256"]:
            raise RuntimeError("live pip-freeze identity drift")
        pip_check = subprocess.run([str(python), "-m", "pip", "check"], text=True, capture_output=True, check=False)
        pip_evidence = base.validate_pip_check(pip_check.returncode, pip_check.stdout, pip_check.stderr)
        write_json(run_dir / "config/input_integrity.json", {"source_dataset": {"run": str(source.relative_to(PROJECT_ROOT)), "seal_entries": seal_entries, "seal_sha256": spec["source_dataset_seal_sha256"]}, "checkpoints": checkpoints, "environment": environment, "python_executable": str(python), "live_pip_freeze_sha256": freeze["pip_freeze_sha256"], "pip_check": pip_evidence, "host": platform.platform()})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/models").mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for checkpoint in checkpoints:
            seed = int(checkpoint["seed"])
            remaining = max(1, TIME_LIMIT_SECONDS - int(time.monotonic() - started))
            completed = subprocess.run(child_argv(spec, run_dir, checkpoint, python), cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=remaining, check=False)
            (run_dir / f"logs/m1d_seed{seed}.log").write_text(completed.stdout + f"\nexit_code={completed.returncode}\n", encoding="utf-8")
            if completed.returncode != 0 or "does not have a deterministic implementation" in completed.stdout:
                raise RuntimeError(f"corrective training child execution failed: seed {seed}, exit {completed.returncode}")
            summary = load_json(run_dir / f"artifacts/models/m1d_seed{seed}/summary.json")
            validate_seed_summary(summary, seed)
            summaries.append(summary)
            write_json(run_dir / "metrics/progress.json", {"schema_version": "aee_corrective_full_encoder_progress_v3", "completed_seeds": len(summaries), "planned_seeds": 3, "last_seed": seed, "duration_seconds": time.monotonic() - started})
        aggregate_metrics = aggregate(summaries)
        overall_status = STATUS_PASS if aggregate_metrics["all_seed_gates_passed"] else STATUS_FAIL
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if result_bytes > DISK_LIMIT_BYTES or time.monotonic() - started > TIME_LIMIT_SECONDS:
            raise RuntimeError("corrective training resource limit exceeded")
        summary = {"schema_version": "aee_corrective_full_encoder_summary_v3", "overall_status": overall_status, "execution_failure": False, "completed_seeds": 3, "seeds": [0, 1, 2], "aggregate": aggregate_metrics, "per_seed": summaries, "result_bytes_before_seal": result_bytes, "disk_limit_bytes": DISK_LIMIT_BYTES, "duration_seconds": time.monotonic() - started, "time_limit_seconds": TIME_LIMIT_SECONDS, "train_samples_per_seed_epoch": {"cano": 5000, "aee": 1000}, "validation_frames_per_seed": {"cano_dense": 5000, "cano_sparse": 5000}, "aee_train_diagnostic_frames_per_seed": 1000, "c09_frames_read": 0, "c10_frames_read": 0, "formal_benchmark_frames_read": 0, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": overall_status})
        sealed = base.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0 if overall_status == STATUS_PASS else 2
    except Exception as exc:
        _seal_failure(run_dir, started, len(summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
