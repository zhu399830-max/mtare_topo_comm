#!/usr/bin/env python3
"""Recover the interrupted Gate-2 corrective by training only missing seed 2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_phase3_masking_corrective_ray_dropout_v1r3 import (
    B0,
    DATASET,
    DISK_LIMIT,
    PYTHON,
    SOURCE,
    TRAINER,
    plot_comparison,
    seal_manifest,
    sha256,
    verify_seal,
)

RUN_ID = "gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2"
INTERRUPTED = PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0"
EXPECTED_CONFIG = {
    "mode": "M1D",
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "patience": 6,
    "workers": 0,
}


def validate_completed_seed(seed: int) -> tuple[dict, dict]:
    child = INTERRUPTED / f"artifacts/models/m1d_seed{seed}"
    summary_path = child / "summary.json"
    checkpoint_path = child / "best.pt"
    log_path = INTERRUPTED / f"logs/m1d_seed{seed}.log"
    if not all(path.is_file() for path in (summary_path, checkpoint_path, log_path)):
        raise RuntimeError(f"seed {seed} recovery evidence is incomplete")
    summary = load_json(summary_path)
    if summary.get("status") != "COMPLETED_PHASE3_MASKING_CORRECTIVE_EXECUTOR":
        raise RuntimeError(f"seed {seed} summary status mismatch")
    if summary.get("seed") != seed or summary.get("mode") != "M1D":
        raise RuntimeError(f"seed {seed} identity mismatch")
    if summary.get("train_frames") != 100000 or summary.get("validation_frames") != 12500:
        raise RuntimeError(f"seed {seed} frame count mismatch")
    if summary.get("strict_test_frames_read") != 0 or summary.get("mtare_frames_read") != 0:
        raise RuntimeError(f"seed {seed} forbidden data read")
    augmentation = summary.get("augmentation", {})
    if augmentation.get("scope") != "train_only" or augmentation.get("probability") != 0.5 or augmentation.get("period_columns") != 10:
        raise RuntimeError(f"seed {seed} augmentation mismatch")
    if augmentation.get("samples") != 100000 * summary.get("epochs_completed", -1):
        raise RuntimeError(f"seed {seed} augmentation sample count mismatch")
    if "exit_code=0" not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"seed {seed} log did not finish cleanly")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise RuntimeError(f"seed {seed} checkpoint config mismatch: {key}")
    if checkpoint.get("seed") != seed or checkpoint.get("mode") != "M1D":
        raise RuntimeError(f"seed {seed} checkpoint identity mismatch")
    evidence = {
        "seed": seed,
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": sha256(summary_path),
        "best_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "log": str(log_path.relative_to(PROJECT_ROOT)),
        "log_sha256": sha256(log_path),
    }
    return summary, evidence


def run_record(summary: dict) -> dict:
    validation = summary["best_validation"]
    cpu = summary["stability"]["cpu"]
    cuda = summary["stability"]["cuda"]
    return {
        "mode": "M1D",
        "seed": summary["seed"],
        "direction_f1": validation["direction"]["f1"],
        "angular_error_deg": validation["direction"]["mean_matched_angular_error_deg"],
        "role_macro_f1": validation["role"]["macro_f1_present"],
        "role_recall": validation["role"]["recall"],
        "count_1_4_macro_f1": validation["count"]["macro_f1_count_1_to_4"],
        "same_cluster_mean": validation["representation"]["same_cluster_cosine"]["mean"],
        "same_cluster_p05": validation["representation"]["same_cluster_cosine"]["p05"],
        "masking_cosine_mean_cpu": cpu["fixed_masking_z_role_cosine_mean"],
        "masking_cosine_p05_cpu": cpu["fixed_masking_z_role_cosine_p05"],
        "masking_cosine_mean_cuda": cuda["fixed_masking_z_role_cosine_mean"],
        "rotation_direction_max_error_cpu": cpu["rotation_direction_max_absolute_logit_error"],
        "rotation_direction_max_error_cuda": cuda["rotation_direction_max_absolute_logit_error"],
        "rotation_z_min_cosine_cpu": cpu["rotation_z_role_minimum_cosine"],
        "augmentation": summary["augmentation"],
        "epochs": summary["epochs_completed"],
        "best_epoch": summary["best_epoch"],
    }


def aggregate_runs(runs: list[dict]) -> tuple[str, dict]:
    b0 = load_json(B0)
    direction_median = float(np.median([r["direction_f1"] for r in runs]))
    direction_floor = min(r["direction_f1"] for r in runs)
    role_median = float(np.median([r["role_macro_f1"] for r in runs]))
    role_recall_medians = np.median(np.asarray([r["role_recall"] for r in runs]), axis=0).tolist()
    count_median = float(np.median([r["count_1_4_macro_f1"] for r in runs]))
    same_mean = float(np.median([r["same_cluster_mean"] for r in runs]))
    same_p05 = float(np.median([r["same_cluster_p05"] for r in runs]))
    masking = float(np.median([r["masking_cosine_mean_cpu"] for r in runs]))
    rotation_error = max(r["rotation_direction_max_error_cpu"] for r in runs)
    rotation_cos = min(r["rotation_z_min_cosine_cpu"] for r in runs)
    direction_pass = direction_median >= b0["f1"] and direction_floor >= 0.75
    auxiliary_pass = (
        role_median >= 0.70
        and min(role_recall_medians) >= 0.60
        and count_median >= 0.70
        and same_mean >= 0.90
        and same_p05 >= 0.75
        and masking >= 0.85
        and rotation_error <= 2e-5
        and rotation_cos >= 0.999
    )
    gate_result = "GATE_PASS" if direction_pass and auxiliary_pass else "GATE_MIXED" if direction_pass else "GATE_FAIL"
    return gate_result, {
        "direction_median_f1": direction_median,
        "direction_seed_floor_f1": direction_floor,
        "frozen_b0_f1": b0["f1"],
        "role_median_macro_f1": role_median,
        "role_recall_medians": role_recall_medians,
        "count_1_4_median_macro_f1": count_median,
        "same_cluster_median_mean_cosine": same_mean,
        "same_cluster_median_p05_cosine": same_p05,
        "masking_median_mean_cosine_cpu": masking,
        "rotation_max_logit_error_cpu": rotation_error,
        "rotation_min_z_cosine_cpu": rotation_cos,
        "direction_pass": direction_pass,
        "auxiliary_representation_pass": auxiliary_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("recovery run state mismatch")
    authorization = spec.get("user_authorization", {})
    if spec.get("gate") != 2 or spec.get("operation") != "training" or authorization.get("status") != "APPROVED":
        raise RuntimeError("recovery approval mismatch")
    if load_json(INTERRUPTED / "RUN_STATE.json").get("state") != "RUNNING":
        raise RuntimeError("interrupted source state mismatch")
    if (INTERRUPTED / "artifacts/models/m1d_seed2").exists():
        raise RuntimeError("seed 2 already exists in interrupted run")
    if verify_seal(DATASET / "artifacts/evidence_sha256.txt", 19338)["mismatch_count"]:
        raise RuntimeError("dataset seal mismatch")
    summaries, references = [], []
    for seed in (0, 1):
        summary, evidence = validate_completed_seed(seed)
        summaries.append(summary)
        references.append(evidence)
    write_json(run / "artifacts/reused_seed_evidence.json", {"source_run": str(INTERRUPTED.relative_to(PROJECT_ROOT)), "seeds": references})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING", "note": "Strict recovery: reuse sealed seed0/1 evidence and train only missing seed2; zero C10/M-TARE/graph/planner."})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    child = run / "artifacts/models/m1d_seed2"
    argv = [str(PYTHON), str(TRAINER), "--dataset-run", str(DATASET), "--output-dir", str(child), "--mode", "M1D", "--seed", "2", "--epochs", "30", "--batch-size", "128", "--learning-rate", "0.0003", "--weight-decay", "0.0001", "--patience", "6", "--workers", "0"]
    child_started = time.monotonic()
    done = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5400, check=False)
    (run / "logs/m1d_seed2.log").write_text(done.stdout + f"\nduration_seconds={time.monotonic()-child_started:.6f}\nexit_code={done.returncode}\n", encoding="utf-8")
    print(done.stdout, end="", flush=True)
    if done.returncode != 0 or "does not have a deterministic implementation" in done.stdout:
        raise RuntimeError(f"seed 2 training failed with exit code {done.returncode}")
    seed2 = load_json(child / "summary.json")
    if seed2.get("strict_test_frames_read") != 0 or seed2.get("mtare_frames_read") != 0:
        raise RuntimeError("seed 2 forbidden data read")
    summaries.append(seed2)
    runs = [run_record(summary) for summary in summaries]
    if [record["seed"] for record in runs] != [0, 1, 2]:
        raise RuntimeError("three-seed identity mismatch")
    gate_result, aggregate = aggregate_runs(runs)
    old = load_json(SOURCE / "metrics/summary.json")["runs"]
    plot_comparison([item for item in old if item["mode"] == "M1"], runs, run / "previews/v1r3_vs_m1d_corrective.png")
    combined_model_bytes = sum(path.stat().st_size for path in (INTERRUPTED / "artifacts/models").rglob("*") if path.is_file()) + sum(path.stat().st_size for path in (run / "artifacts/models").rglob("*") if path.is_file())
    integrity = all(record["augmentation"]["samples"] == 100000 * record["epochs"] for record in runs) and combined_model_bytes <= DISK_LIMIT
    overall = "PASS_CANO_PHASE3_MASKING_CORRECTIVE_RECOVERY" if integrity else "FAIL_CANO_PHASE3_MASKING_CORRECTIVE_RECOVERY"
    result = {
        "schema_version": "cano_phase3_masking_corrective_recovery_summary_v1",
        "overall_status": overall,
        "gate_result": gate_result if integrity else "GATE_FAIL",
        "integrity_passed": integrity,
        "recovery_reason": "Original outer timeout expired after host suspension; seed0/1 completed cleanly, runner stopped before seed2 and aggregation.",
        "runs": runs,
        "aggregate": aggregate,
        "duration_seconds_recovery": time.monotonic() - started,
        "combined_model_bytes": combined_model_bytes,
        "disk_limit_bytes": DISK_LIMIT,
        "unique_train_frames": 100000,
        "validation_frames": 12500,
        "newly_trained_seeds": [2],
        "reused_completed_seeds": [0, 1],
        "strict_test_frames_read": 0,
        "mtare_frames_read": 0,
        "graph_runs": 0,
        "planner_changes": 0,
    }
    write_json(run / "metrics/summary.json", result)
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if integrity else "FAILED", "overall_status": overall, "gate_result": result["gate_result"], "note": "Strict seed2 recovery; no C10/M-TARE/graph/planner."})
    sealed = seal_manifest(run)
    print(json.dumps({"overall_status": overall, "gate_result": result["gate_result"], "sealed_files": sealed}, indent=2))
    return 0 if integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
