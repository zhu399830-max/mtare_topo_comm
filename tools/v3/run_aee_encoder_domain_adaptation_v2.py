#!/home/zeng-workstation/anaconda3/bin/python
"""Execute and seal three-seed canonical-target AEE encoder adaptation V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
import run_aee_head_adaptation_v1 as base


RUN_ID = "gate2_20260821_aee_encoder_domain_adaptation_v2_seed20260820"
TRAINER = PROJECT_ROOT / "tools/v3/train_aee_encoder_domain_adaptation_v2.py"
STATUS_PASS = "PASS_AEE_ENCODER_DOMAIN_ADAPTATION_V2"
STATUS_FAIL = "FAIL_AEE_ENCODER_DOMAIN_ADAPTATION_V2"


def child_argv(
    spec: dict[str, Any],
    run_dir: Path,
    checkpoint: dict[str, Any],
    python: Path = base.PYTHON,
) -> list[str]:
    seed = int(checkpoint["seed"])
    if seed not in (0, 1, 2):
        raise RuntimeError("AEE checkpoint seed must be 0, 1 or 2")
    return [
        str(python),
        str(TRAINER),
        "--cano-dataset-run", str((PROJECT_ROOT / spec["cano_dataset_run"]).resolve()),
        "--aee-sensor-run", str((PROJECT_ROOT / spec["source_sensor_run"]).resolve()),
        "--aee-teacher-run", str((PROJECT_ROOT / spec["source_teacher_run"]).resolve()),
        "--source-checkpoint", str((PROJECT_ROOT / checkpoint["path"]).resolve()),
        "--source-checkpoint-sha256", str(checkpoint["sha256"]),
        "--output-dir", str(run_dir / f"artifacts/models/m1d_seed{seed}"),
        "--seed", str(seed),
        "--epochs", "10",
        "--batch-size", "128",
        "--learning-rate", "0.0001",
        "--weight-decay", "0.0001",
        "--workers", "0",
    ]


def validate_seed_summary(summary: dict[str, Any], seed: int) -> None:
    if summary.get("status") != "PASS_AEE_ENCODER_DOMAIN_ADAPTATION_SEED_V2":
        raise RuntimeError(f"AEE encoder adaptation seed {seed} failed qualification")
    if summary.get("seed") != seed or summary.get("epochs_completed") != 10:
        raise RuntimeError(f"AEE encoder adaptation seed identity/epoch drift: {seed}")
    if summary.get("independent_train_samples_per_epoch") != {"cano": 3000, "aee": 3000}:
        raise RuntimeError(f"independent training sample count drift: seed {seed}")
    if summary.get("augmented_cano_views_per_epoch") != 3000:
        raise RuntimeError(f"augmented Cano view count drift: seed {seed}")
    if summary.get("aee_validation_frames") != 3000 or summary.get("cano_validation_frames") != 12500:
        raise RuntimeError(f"AEE/Cano validation count drift: seed {seed}")
    if summary.get("strict_test_frames_read") != 0 or summary.get("c10_frames_read") != 0:
        raise RuntimeError(f"strict-test data read by seed {seed}")
    if summary.get("cano_c09_validation_frames_read") != 12500:
        raise RuntimeError(f"sealed Cano retention read count drift: seed {seed}")
    if summary.get("later_sealed_world_frames_read") != 0:
        raise RuntimeError(f"later sealed-world data read by seed {seed}")
    expected_gates = {
        "aee_direction_vs_b0",
        "aee_empty_rate",
        "aee_count_1_to_4_macro_f1",
        "aee_role_macro_f1",
        "cano_direction_retention",
        "all_model_tensors_finite",
        "encoder_changed",
        "embedding_changed",
        "semantic_heads_changed",
    }
    gates = summary.get("gate")
    if not isinstance(gates, dict) or set(gates) != expected_gates or not all(gates.values()):
        raise RuntimeError(f"AEE encoder representation gate failed: seed {seed}")
    count = summary["adapted_aee_validation"]["count"]
    if count.get("gate_branch_counts") != [1, 2, 3, 4]:
        raise RuntimeError(f"count common-class Gate drift: seed {seed}")
    if set(count.get("rare_branch_counts_diagnostic_only", {})) != {"5", "6"}:
        raise RuntimeError(f"rare count diagnostic drift: seed {seed}")


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [item.get("seed") for item in summaries] != [0, 1, 2]:
        raise RuntimeError("AEE encoder adaptation must contain seeds 0/1/2 exactly once in order")
    for item in summaries:
        validate_seed_summary(item, int(item["seed"]))
    return {
        "aee_direction_f1_by_seed": [item["adapted_aee_validation"]["direction"]["f1"] for item in summaries],
        "aee_direction_f1_median": float(np.median([item["adapted_aee_validation"]["direction"]["f1"] for item in summaries])),
        "aee_empty_rate_by_seed": [item["adapted_aee_validation"]["direction"]["empty_rate"] for item in summaries],
        "aee_count_1_to_4_macro_f1_by_seed": [item["adapted_aee_validation"]["count"]["macro_f1_count_1_to_4"] for item in summaries],
        "aee_count_5_to_6_diagnostic_by_seed": [item["adapted_aee_validation"]["count"]["rare_branch_counts_diagnostic_only"] for item in summaries],
        "aee_role_macro_f1_by_seed": [item["adapted_aee_validation"]["role"]["macro_f1_present"] for item in summaries],
        "cano_direction_f1_change_by_seed": [
            item["adapted_cano_validation"]["direction"]["f1"]
            - item["source_cano_validation"]["direction"]["f1"]
            for item in summaries
        ],
        "all_seed_gates_passed": True,
    }


base.RUN_ID = RUN_ID
base.TRAINER = TRAINER
base.STATUS_PASS = STATUS_PASS
base.STATUS_FAIL = STATUS_FAIL
base.SUMMARY_SCHEMA = "aee_encoder_domain_adaptation_summary_v2"
base.PROGRESS_SCHEMA = "aee_encoder_domain_adaptation_progress_v2"
base.EXTRA_SUMMARY = {
    "direction_target_contract": "cano_gaussian_component_centers_sigma_3deg",
    "independent_train_samples_per_seed_epoch": {"cano": 3000, "aee": 3000},
    "augmented_cano_views_per_seed_epoch": 3000,
    "count_gate_branch_counts": [1, 2, 3, 4],
    "rare_count_diagnostic_only": [5, 6],
}
base.child_argv = child_argv
base.validate_seed_summary = validate_seed_summary
base.aggregate = aggregate


if __name__ == "__main__":
    raise SystemExit(base.main())
